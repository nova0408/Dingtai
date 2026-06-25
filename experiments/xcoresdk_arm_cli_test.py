#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))
    
LOCAL_IP = "192.168.1.116"
MM_PER_M = 1000.0
DEFAULT_CARTESIAN_SPEED = 50.0
DEFAULT_CARTESIAN_ZONE = 1.0
DEFAULT_POWER_ON_TIMEOUT_S = 3.0
EXPECTED_ARM_TYPES = {
    "left": "AR5-5_0.8L-W4C1C9-ZY2",
    "right": "AR5-5_0.8R-W4C1C9-ZY2",
}

from sdk.xcoresdk import xCoreSDK_python 
DEFAULT_LEFT_ANGLE=[-85.00, -100.00, 45.00, -50.00, -10.00, -15.00, -5.00]
DEFAULT_RIGHT_ANGLE=[-100.00, 100.00, 135.00, -55.00, 0.00, -15.00, 10.00]

# region 数据结构
@dataclass(frozen=True, slots=True)
class RobotConnectionConfig:
    """机器人连接配置。

    该配置只保存交互式 CLI 需要的最小连接参数，不持有 SDK 对象或后台线程。

    Attributes
    ----------
    robot_ip:
        机器人控制器 IP 地址。
    local_ip:
        本机网卡 IP 地址。部分现场网络环境需要显式提供。
    """

    robot_ip: str
    "机器人控制器 IP 地址。"

    local_ip: str | None
    "本机网卡 IP 地址，未提供时保持 `None`。"


@dataclass(slots=True)
class ConnectedArm:
    """单台已连接机械臂的运行上下文。"""

    arm_side: str
    "机械臂侧别，取值为 `left` 或 `right`。"

    config: RobotConnectionConfig
    "创建连接时使用的网络配置。"

    robot: xCoreSDK_python.xMateErProRobot
    "SDK 机器人对象。"

    robot_type: str
    "控制器上报的机器人型号。"

    robot_uid: str
    "控制器上报的机器人唯一标识。"

    ec: dict[str, object]
    "该机械臂独立复用的 SDK 错误码字典。"


# endregion


# region 基础解析
def _parse_float_list(raw_text: str, expected_len: int | None = None) -> list[float]:
    """解析用户输入的浮点数序列。

    Parameters
    ----------
    raw_text:
        用户输入文本，允许使用空格、英文逗号或中文逗号分隔。
    expected_len:
        期望长度。若不为 `None`，解析结果长度必须匹配。

    Returns
    -------
    list[float]
        解析后的浮点数列表。

    Raises
    ------
    ValueError
        当输入为空、包含非法数值或长度不匹配时抛出。
    """

    normalized = raw_text.replace("，", " ").replace(",", " ")
    values = [token for token in normalized.split() if token]
    if not values:
        raise ValueError("未输入任何数值")
    parsed = [float(value) for value in values]
    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError(f"数值个数不匹配，expected={expected_len}, actual={len(parsed)}")
    return parsed


def _input_optional_float_list(prompt: str, expected_len: int) -> list[float] | None:
    """读取浮点数列表，支持输入 q 返回上一级。"""

    raw_text = input(prompt).strip()
    if raw_text.lower() == "q":
        return None
    return _parse_float_list(raw_text, expected_len=expected_len)


def _make_cartesian_position(values: list[float]) -> xCoreSDK_python.CartesianPosition:
    """构建笛卡尔位姿对象。

    Parameters
    ----------
    values:
        长度为 6 的数值序列，依次为 xyzrpy。

    Returns
    -------
    xCoreSDK_python.CartesianPosition
        SDK 的 `CartesianPosition` 对象。
    """

    return xCoreSDK_python.CartesianPosition([float(value) for value in values])


def _make_joint_position(values: list[float]) -> xCoreSDK_python.JointPosition:
    """构建关节位姿对象。

    Parameters
    ----------
    values:
        关节角序列，单位弧度，长度由机械臂自由度决定。

    Returns
    -------
    xCoreSDK_python.JointPosition
        SDK 的 `JointPosition` 对象。
    """

    return xCoreSDK_python.JointPosition([float(value) for value in values])


# endregion


# region 状态查询
def _format_sequence(values: list[float] | tuple[float, ...], decimals: int = 2) -> str:
    return ", ".join(f"{float(value):.{decimals}f}" for value in values)


def _mm_to_m(values_mm: list[float]) -> list[float]:
    """将毫米转换为米。"""

    return [float(value) / MM_PER_M for value in values_mm]


def _m_to_mm(values_m: list[float] | tuple[float, ...]) -> list[float]:
    """将米转换为毫米。"""

    return [float(value) * MM_PER_M for value in values_m]


def _deg_to_rad(values_deg: list[float]) -> list[float]:
    """将角度转换为弧度。"""

    return [math.radians(float(value)) for value in values_deg]


def _rad_to_deg(values_rad: list[float] | tuple[float, ...]) -> list[float]:
    """将弧度转换为角度。"""

    return [math.degrees(float(value)) for value in values_rad]


def _print_sdk_result(action: str, ec: dict[str, object]) -> None:
    """打印 SDK 调用结果，便于现场排查控制器返回信息。"""

    message = str(ec.get("message", ""))
    code = ec.get("ec", 0)
    print(f"{action}: ec={code}, message={message}")


def _describe_power_state(power_state: xCoreSDK_python.PowerState) -> str:
    """把电源状态转换成更直观的中文说明。"""

    if power_state == xCoreSDK_python.PowerState.on:
        return "上电"
    if power_state == xCoreSDK_python.PowerState.off:
        return "下电"
    if power_state == xCoreSDK_python.PowerState.estop:
        return "急停被按下"
    if power_state == xCoreSDK_python.PowerState.gstop:
        return "安全门打开"
    return "未知"


def _copy_cartesian_pose_context(
    source: xCoreSDK_python.CartesianPosition,
    target: xCoreSDK_python.CartesianPosition,
) -> None:
    """复制笛卡尔位姿的上下文约束字段。"""

    target.confData = list(source.confData)
    target.hasElbow = source.hasElbow
    target.elbow = source.elbow


def _ensure_nrt_motion_ready(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> bool:
    """确保非实时运动指令满足执行前提。"""

    robot.stop(ec)
    _print_sdk_result("stop", ec)
    if ec.get("ec", 0) != 0:
        return False
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _print_sdk_result("setMotionControlMode(NrtCommandMode)", ec)
    if ec.get("ec", 0) != 0:
        return False
    robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    _print_sdk_result("setOperateMode(automatic)", ec)
    if ec.get("ec", 0) != 0:
        return False
    robot.setPowerState(True, ec)
    _print_sdk_result("setPowerState(True)", ec)
    if ec.get("ec", 0) != 0:
        return False
    if not _wait_for_power_on(robot, ec):
        print("上电状态未在超时内确认完成，请检查现场使能、急停和安全门")
        return False
    robot.setDefaultConfOpt(False, ec)
    _print_sdk_result("setDefaultConfOpt(False)", ec)
    if ec.get("ec", 0) != 0:
        return False
    robot.setDefaultSpeed(DEFAULT_CARTESIAN_SPEED, ec)
    _print_sdk_result(f"setDefaultSpeed({DEFAULT_CARTESIAN_SPEED:.2f})", ec)
    if ec.get("ec", 0) != 0:
        return False
    robot.setDefaultZone(DEFAULT_CARTESIAN_ZONE, ec)
    _print_sdk_result(f"setDefaultZone({DEFAULT_CARTESIAN_ZONE:.2f})", ec)
    if ec.get("ec", 0) != 0:
        return False
    current_power_state = robot.powerState(ec)
    print(f"当前电机状态: {current_power_state} ({_describe_power_state(current_power_state)})")
    return current_power_state == xCoreSDK_python.PowerState.on


def _wait_for_power_on(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float = DEFAULT_POWER_ON_TIMEOUT_S,
) -> bool:
    """等待机器人确认进入上电状态。"""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        power_state = robot.powerState(ec)
        print(f"当前电机状态: {power_state} ({_describe_power_state(power_state)})")
        if power_state == xCoreSDK_python.PowerState.on:
            return True
        time.sleep(0.1)
    return False


def _validate_cartesian_target(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    target_pose: xCoreSDK_python.CartesianPosition,
) -> bool:
    """在执行 MoveL 前检查当前到目标的直线路径是否可达。"""

    start_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    start_joint = list(robot.jointPos(ec))
    result_joint = robot.checkPath(start_pose, start_joint, target_pose, ec)
    _print_sdk_result("checkPath", ec)
    if ec.get("ec", 0) != 0:
        return False
    print(f"checkPath 目标关节(deg): {_format_sequence(_rad_to_deg(result_joint))}")
    return True


def _print_cartesian_ik_preview(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    target_pose: xCoreSDK_python.CartesianPosition,
) -> None:
    """输入笛卡尔目标后，立即打印当前关节与逆解结果。"""

    current_joint_deg = _rad_to_deg(robot.jointPos(ec))
    robot_model = robot.model()
    toolset = xCoreSDK_python.Toolset()
    target_joint = robot_model.calcIk(target_pose, toolset, ec)
    _print_sdk_result("calcIk", ec)
    print(f"当前关节值(deg): {_format_sequence(current_joint_deg)}")
    if ec.get("ec", 0) != 0:
        return
    print(f"目标逆解值(deg): {_format_sequence(_rad_to_deg(target_joint))}")


def _select_cartesian_motion_type() -> str | None:
    """选择笛卡尔目标的执行方式。"""

    print("笛卡尔运动方式:")
    print("  1. MoveL 直线运动")
    print("  2. MoveJ 关节插补到笛卡尔目标")
    print("  q. 返回上一级")
    choice = input("请选择运动方式: ").strip().lower()
    if choice == "1":
        return "movel"
    if choice == "2":
        return "movej"
    if choice == "q":
        return None
    raise ValueError("无效运动方式")


def _recover_estop(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """执行急停复位。"""

    robot.recoverState(1, ec)
    _print_sdk_result("recoverState(1)", ec)


def _toggle_soft_limit(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """打开或关闭软限位，使用官方示例中的硬编码值。"""
    return
    soft_limits = [
        [-2.0543261909900767, 2.0543261909900767],
        [-1.356194490192345, 1.356194490192345],
        [-1.96705972839036, 1.443460952792061],
        [-2.0543261909900767, 2.0543261909900767],
        [-2.0543261909900767, 2.0543261909900767],
        [-2.0543261909900767, 2.0543261909900767],
    ]
    print("软限位开关:")
    print("  1. 打开软限位")
    print("  2. 关闭软限位")
    print("  q. 返回主菜单")
    choice = input("请选择: ").strip().lower()
    if choice == "q":
        return
    if choice not in {"1", "2"}:
        print("无效选择")
        return

    robot.setPowerState(False, ec)
    _print_sdk_result("setPowerState(False)", ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _print_sdk_result("setOperateMode(manual)", ec)
    if not _wait_for_power_off(robot, ec):
        print("下电状态未确认完成，当前不允许切换软限位")
        return

    if choice == "1":
        robot.setSoftLimit(True, ec, soft_limits)
        _print_sdk_result("setSoftLimit(True)", ec)
    else:
        robot.setSoftLimit(False, ec)
        _print_sdk_result("setSoftLimit(False)", ec)
        print("软限位已关闭，当前机械臂将不受上述软限位范围限制，请确保现场安全")


def _print_robot_snapshot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """打印机器人当前模式、状态和电机信息。

    Parameters
    ----------
    robot:
        已连接的 SDK 机器人对象。
    ec:
        SDK 错误码字典，由调用方复用。
    """

    operate_mode = robot.operateMode(ec)
    operation_state = robot.operationState(ec)
    power_state = robot.powerState(ec)
    joint_values = robot.jointPos(ec)
    cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    robot_info = robot.robotInfo(ec)

    print("当前模式:", operate_mode)
    print("当前状态:", operation_state)
    print(f"电机状态: {power_state} ({_describe_power_state(power_state)})")
    print("机器人型号:", robot_info.type)
    print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
    print(f"当前笛卡尔位姿(mm/deg): trans={_format_sequence(_m_to_mm(cart_pose.trans))} rpy={_format_sequence(_rad_to_deg(cart_pose.rpy))}")
    print(
        "当前笛卡尔上下文: "
        f"hasElbow={cart_pose.hasElbow}, "
        f"elbow(deg)={math.degrees(cart_pose.elbow):.2f}, "
        f"confData={cart_pose.confData}"
    )


def _print_connected_arm_snapshot(connected_arm: ConnectedArm) -> None:
    """打印当前选中机械臂的连接信息与状态快照。"""

    print(
        f"当前机械臂: {connected_arm.arm_side} "
        f"(ip={connected_arm.config.robot_ip}, type={connected_arm.robot_type}, uid={connected_arm.robot_uid})"
    )
    _print_robot_snapshot(connected_arm.robot, connected_arm.ec)


def _print_cartesian_pose(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """打印当前笛卡尔空间位姿。"""

    pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    print("当前笛卡尔空间位姿:")
    print(f"  trans(mm): {_format_sequence(_m_to_mm(pose.trans))}")
    print(f"  rpy(deg): {_format_sequence(_rad_to_deg(pose.rpy))}")
    print(f"  hasElbow: {pose.hasElbow}, elbow(deg): {math.degrees(pose.elbow):.2f}, confData: {pose.confData}")


def _wait_until_idle(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    prompt: str,
    timeout_s: float = 15.0,
) -> bool:
    """轮询等待机器人运动结束。

    Parameters
    ----------
    robot:
        已连接的 SDK 机器人对象。
    ec:
        SDK 错误码字典，由调用方复用。
    prompt:
        轮询期间显示的提示文本。
    timeout_s:
        等待超时时间，超时后返回 `False`。
    """

    has_observed_active_state = False
    deadline = time.time() + timeout_s
    while True:
        if time.time() >= deadline:
            print(f"{prompt} 超时：超过 {timeout_s:.1f} 秒仍未结束")
            return False
        state = robot.operationState(ec)
        if state == xCoreSDK_python.OperationState.idle and has_observed_active_state:
            print("运动已结束")
            return True
        if state not in (xCoreSDK_python.OperationState.idle, xCoreSDK_python.OperationState.unknown):
            has_observed_active_state = True
        print(f"{prompt}: {state}", end="\r")
        time.sleep(0.2)


def _wait_for_power_off(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float = 3.0,
) -> bool:
    """等待机器人确认进入下电状态。"""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        power_state = robot.powerState(ec)
        print(f"当前电机状态: {power_state} ({_describe_power_state(power_state)})")
        if power_state == xCoreSDK_python.PowerState.off:
            return True
        time.sleep(0.2)
    return False


def _prepare_predefined_joint_motion_loop(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> bool:
    """在进入硬编码关节循环前，一次性准备好所需运动状态。"""

    if not _ensure_nrt_motion_ready(robot, ec):
        return False
    robot.moveReset(ec)
    _print_sdk_result("moveReset(pre-loop)", ec)
    if ec.get("ec", 0) != 0:
        return False
    operation_state = robot.operationState(ec)
    print(f"循环开始前操作状态: {operation_state}")
    return operation_state in (
        xCoreSDK_python.OperationState.idle,
        xCoreSDK_python.OperationState.unknown,
    )


def _ensure_drag_prerequisites(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> bool:
    """切换到拖动所需的前置状态，并确认状态真正生效。"""

    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _print_sdk_result("setMotionControlMode(NrtCommandMode)", ec)
    if ec.get("ec", 0) != 0:
        return False
    robot.setPowerState(False, ec)
    _print_sdk_result("setPowerState(False)", ec)
    if ec.get("ec", 0) != 0:
        return False
    if not _wait_for_power_off(robot, ec):
        print("下电状态未确认完成，当前不允许继续打开拖动")
        return False
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _print_sdk_result("setOperateMode(manual)", ec)
    if ec.get("ec", 0) != 0:
        return False
    robot.moveReset(ec)
    _print_sdk_result("moveReset", ec)
    if ec.get("ec", 0) != 0:
        return False
    print("已确认处于手动模式、下电状态，并执行 moveReset")
    return True


def _drag_record_loop(
    connected_arm: ConnectedArm,
) -> list[dict[str, str]]:
    """拖动开启后的记录模式。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    records: list[dict[str, str]] = []
    print("已进入记录模式。")
    print("直接回车记录当前信息，输入 q 退出并保存。")
    try:
        while True:
            raw_text = input("请输入: ").strip().lower()
            if raw_text == "q":
                print("退出记录模式")
                return records
            if raw_text != "":
                print("无效输入，请直接回车或输入 q")
                continue
            joint_values = robot.jointPos(ec)
            cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
            _print_sdk_result("jointPos", ec)
            _print_sdk_result("cartPosture(endInRef)", ec)
            joint_values_deg = _rad_to_deg(joint_values)
            trans_mm = _m_to_mm(cart_pose.trans)
            rpy_deg = _rad_to_deg(cart_pose.rpy)
            record: dict[str, str] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "side": connected_arm.arm_side,
                "j1": f"{joint_values_deg[0]:.6f}",
                "j2": f"{joint_values_deg[1]:.6f}",
                "j3": f"{joint_values_deg[2]:.6f}",
                "j4": f"{joint_values_deg[3]:.6f}",
                "j5": f"{joint_values_deg[4]:.6f}",
                "j6": f"{joint_values_deg[5]:.6f}",
                "j7": f"{joint_values_deg[6]:.6f}",
                "x": f"{trans_mm[0]:.6f}",
                "y": f"{trans_mm[1]:.6f}",
                "z": f"{trans_mm[2]:.6f}",
                "rx": f"{rpy_deg[0]:.6f}",
                "ry": f"{rpy_deg[1]:.6f}",
                "rz": f"{rpy_deg[2]:.6f}",
                "has_elbow": str(bool(cart_pose.hasElbow)),
                "elbow": f"{math.degrees(float(cart_pose.elbow)):.6f}",
                "conf": str(list(cart_pose.confData)),
            }
            records.append(record)
            print(f"已记录第 {len(records)} 条")
            print(f"  臂别: {record['side']}, 时间: {record['timestamp']}")
            print(f"  关节(deg): {_format_sequence(joint_values_deg)}")
            print(f"  位姿(mm/deg): trans={_format_sequence(trans_mm)} rpy={_format_sequence(rpy_deg)}")
            print(
                "  上下文: "
                f"hasElbow={cart_pose.hasElbow}, "
                f"elbow(deg)={math.degrees(float(cart_pose.elbow)):.2f}, "
                f"confData={list(cart_pose.confData)}"
            )
    except KeyboardInterrupt:
        print()
        print("用户中断，退出记录模式")
        return records


def _write_drag_records_csv(records: list[dict[str, str]], arm_side: str) -> Path | None:
    if not records:
        return None
    csv_path = Path.cwd() / f"xcoresdk_drag_records_{arm_side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "timestamp",
        "side",
        "j1",
        "j2",
        "j3",
        "j4",
        "j5",
        "j6",
        "j7",
        "x",
        "y",
        "z",
        "rx",
        "ry",
        "rz",
        "has_elbow",
        "elbow",
        "conf",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


# endregion


# region 机器人控制
def _connect_robot(config: RobotConnectionConfig) -> xCoreSDK_python.xMateErProRobot:
    """创建并连接机器人对象。"""

    if config.local_ip:
        robot = xCoreSDK_python.xMateErProRobot(config.robot_ip, config.local_ip)
    else:
        robot = xCoreSDK_python.xMateErProRobot(config.robot_ip)
    return robot


def _default_connection_configs() -> list[RobotConnectionConfig]:
    """返回脚本内置的候选机器人连接配置。"""

    return [
        RobotConnectionConfig(robot_ip="192.168.1.161", local_ip=LOCAL_IP),
        RobotConnectionConfig(robot_ip="192.168.1.160", local_ip=LOCAL_IP),
    ]


def _detect_arm_side(robot_type: str) -> str:
    """根据控制器上报的机型名称判断左右臂。"""

    for arm_side, expected_robot_type in EXPECTED_ARM_TYPES.items():
        if robot_type == expected_robot_type:
            return arm_side
    raise ValueError(f"未识别的机器人型号: {robot_type}")


def _connect_arms(configs: list[RobotConnectionConfig]) -> dict[str, ConnectedArm]:
    """连接多台机械臂，并按控制器上报的型号归类左右臂。"""

    connected_arms: dict[str, ConnectedArm] = {}
    try:
        for config in configs:
            ec: dict[str, object] = {}
            robot = _connect_robot(config)
            robot_info = robot.robotInfo(ec)
            _print_sdk_result(f"robotInfo({config.robot_ip})", ec)
            if ec.get("ec", 0) != 0:
                raise RuntimeError(f"读取机器人信息失败: ip={config.robot_ip}")
            arm_side = _detect_arm_side(robot_info.type)
            if arm_side in connected_arms:
                raise RuntimeError(
                    f"检测到重复的 {arm_side} 机械臂: "
                    f"existing={connected_arms[arm_side].config.robot_ip}, current={config.robot_ip}"
                )
            connected_arm = ConnectedArm(
                arm_side=arm_side,
                config=config,
                robot=robot,
                robot_type=robot_info.type,
                robot_uid=robot_info.id,
                ec=ec,
            )
            _prepare_robot(robot, ec)
            connected_arms[arm_side] = connected_arm
            print(
                f"已连接 {arm_side} arm: ip={config.robot_ip}, "
                f"type={robot_info.type}, uid={robot_info.id}"
            )
        missing_arm_sides = [arm_side for arm_side in EXPECTED_ARM_TYPES if arm_side not in connected_arms]
        if missing_arm_sides:
            raise RuntimeError(f"缺少目标机械臂连接: {', '.join(missing_arm_sides)}")
        return connected_arms
    except Exception:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)
        raise


def _prepare_robot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """完成基础上电前准备。"""

    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    robot.setPowerState(False, ec)


def _shutdown_robot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """安全停止并断开单台机械臂连接。"""

    try:
        robot.stop(ec)
    except Exception:
        pass
    try:
        robot.disableDrag(ec)
    except Exception:
        pass
    try:
        robot.setPowerState(False, ec)
    except Exception:
        pass
    try:
        robot.disconnectFromRobot(ec)
    except Exception:
        pass


def _set_motor_state(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object], on: bool) -> None:
    robot.setPowerState(on, ec)
    print(f"电机已{'打开' if on else '关闭'}")


def _switch_mode(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    print("可选模式:")
    print("  1. manual")
    print("  2. automatic")
    choice = input("请选择模式: ").strip()
    if choice == "1":
        robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    elif choice == "2":
        robot.setOperateMode(xCoreSDK_python.OperateMode.automatic, ec)
    else:
        raise ValueError("无效模式选择")
    print("模式切换完成")


def _toggle_drag(connected_arm: ConnectedArm) -> None:
    robot = connected_arm.robot
    ec = connected_arm.ec
    csv_path: Path | None = None
    try:
        if not _ensure_drag_prerequisites(robot, ec):
            return
        robot.enableDrag(
            int(xCoreSDK_python.DragParameterSpace.cartesianSpace),
            int(xCoreSDK_python.DragParameterType.freely),
            ec,
            enable_drag_button=False
        )
        _print_sdk_result("enableDrag(cartesianSpace, freely, ec)", ec)

        if ec.get("ec", 0) != 0:
            return
        records = _drag_record_loop(connected_arm)
        csv_path = _write_drag_records_csv(records, connected_arm.arm_side)
    finally:
        robot.disableDrag(ec)
        _print_sdk_result("disableDrag", ec)
    if csv_path is None:
        print("没有记录到任何数据，已关闭拖动")
    else:
        print(f"已保存到: {csv_path}")
        print("拖动已关闭")


def _cartesian_control_loop(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """笛卡尔空间控制循环。"""

    if not _ensure_nrt_motion_ready(robot, ec):
        print("笛卡尔运动前置状态未准备完成，请先检查控制器状态")
        return

    while True:
        _print_cartesian_pose(robot, ec)
        print("输入新的 xyzrpy，单位分别为 mm 和 deg")
        print("输入 q 返回主菜单")
        raw_text = input("目标 xyzrpy: ").strip()
        if raw_text.lower() == "q":
            return
        try:
            target_values = _parse_float_list(raw_text, expected_len=6)
        except ValueError as exc:
            print(f"输入格式错误: {exc}")
            continue
        current_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
        target_pose = _make_cartesian_position(_mm_to_m(target_values[:3]) + _deg_to_rad(target_values[3:]))
        _copy_cartesian_pose_context(current_pose, target_pose)
        _print_cartesian_ik_preview(robot, ec, target_pose)
        try:
            motion_type = _select_cartesian_motion_type()
        except ValueError as exc:
            print(exc)
            continue
        if motion_type is None:
            continue
        print(
            "目标笛卡尔位姿: "
            f"trans(mm)={_format_sequence(target_values[:3])}, "
            f"rpy(deg)={_format_sequence(target_values[3:])}, "
            f"motion={motion_type}, "
            f"hasElbow={target_pose.hasElbow}, "
            f"elbow(deg)={math.degrees(target_pose.elbow):.2f}, "
            f"confData={target_pose.confData}"
        )
        if motion_type == "movel" and not _validate_cartesian_target(robot, ec, target_pose):
            print("当前目标未通过路径检查，已取消本次笛卡尔运动")
            continue
        cmd_id = xCoreSDK_python.PyString()
        robot.moveReset(ec)
        _print_sdk_result("moveReset", ec)
        if ec.get("ec", 0) != 0:
            continue
        if motion_type == "movel":
            robot.moveAppend(
                [xCoreSDK_python.MoveLCommand(target_pose, DEFAULT_CARTESIAN_SPEED, DEFAULT_CARTESIAN_ZONE)],
                cmd_id,
                ec,
            )
            _print_sdk_result("moveAppend(MoveL)", ec)
        else:
            robot.moveAppend(
                [xCoreSDK_python.MoveJCommand(target_pose, DEFAULT_CARTESIAN_SPEED, DEFAULT_CARTESIAN_ZONE)],
                cmd_id,
                ec,
            )
            _print_sdk_result("moveAppend(MoveJ)", ec)
        if ec.get("ec", 0) != 0:
            continue
        robot.moveStart(ec)
        _print_sdk_result("moveStart", ec)
        if ec.get("ec", 0) != 0:
            current_power_state = robot.powerState(ec)
            current_operate_mode = robot.operateMode(ec)
            current_operation_state = robot.operationState(ec)
            print(f"moveStart 失败时电机状态: {current_power_state} ({_describe_power_state(current_power_state)})")
            print(f"moveStart 失败时模式: {current_operate_mode}")
            print(f"moveStart 失败时操作状态: {current_operation_state}")
            continue
        print(f"已下发笛卡尔运动，cmd_id={cmd_id.content()}")
        _wait_until_idle(robot, ec, "等待笛卡尔运动")
        _print_cartesian_pose(robot, ec)


def _joint_control_loop(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """关节空间控制循环。"""

    if not _prepare_predefined_joint_motion_loop(robot, ec):
        print("关节运动前置状态未准备完成，请先检查控制器状态")
        return

    while True:
        joint_values = robot.jointPos(ec)
        print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
        print("输入新的关节值，单位 deg，支持空格、英文逗号或中文逗号分隔")
        print("输入 q 返回主菜单")
        raw_text = input("目标关节值: ").strip()
        if raw_text.lower() == "q":
            return
        try:
            target_values = _parse_float_list(raw_text, expected_len=len(joint_values))
        except ValueError as exc:
            print(f"输入格式错误: {exc}")
            continue
        target_joint = _make_joint_position(_deg_to_rad(target_values))
        cmd_id = xCoreSDK_python.PyString()
        robot.moveReset(ec)
        _print_sdk_result("moveReset", ec)
        if ec.get("ec", 0) != 0:
            return
        robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, 1000, 10)], cmd_id, ec)
        _print_sdk_result("moveAppend(MoveAbsJ)", ec)
        if ec.get("ec", 0) != 0:
            return
        robot.moveStart(ec)
        _print_sdk_result("moveStart", ec)
        if ec.get("ec", 0) != 0:
            current_power_state = robot.powerState(ec)
            current_operate_mode = robot.operateMode(ec)
            current_operation_state = robot.operationState(ec)
            print(f"moveStart 失败时电机状态: {current_power_state} ({_describe_power_state(current_power_state)})")
            print(f"moveStart 失败时模式: {current_operate_mode}")
            print(f"moveStart 失败时操作状态: {current_operation_state}")
            return
        print(f"已下发关节运动，cmd_id={cmd_id.content()}")
        _wait_until_idle(robot, ec, "等待关节运动")


def _single_joint_control_loop(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """单关节控制循环。"""

    if not _prepare_predefined_joint_motion_loop(robot, ec):
        print("单关节运动前置状态未准备完成，请先检查控制器状态")
        return

    while True:
        joint_values = robot.jointPos(ec)
        print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
        print("输入 q 返回主菜单")
        axis_text = input(f"请选择轴编号 1-{len(joint_values)}: ").strip().lower()
        if axis_text == "q":
            return
        try:
            axis_index = int(axis_text)
        except ValueError:
            print("轴编号输入无效")
            continue
        if not 1 <= axis_index <= len(joint_values):
            print("轴编号超出范围")
            continue
        while True:
            print(f"当前所选轴 J{axis_index} 值(deg): {math.degrees(joint_values[axis_index - 1]):.2f}")
            print("输入当前轴的目标值，单位 deg")
            print("输入 q 返回前一级选轴")
            raw_text = input("目标轴值: ").strip()
            if raw_text.lower() == "q":
                break
            try:
                target_value = math.radians(float(raw_text))
            except ValueError:
                print("轴目标值输入无效")
                continue
            target_joint_values = list(joint_values)
            target_joint_values[axis_index - 1] = target_value
            target_joint = _make_joint_position(target_joint_values)
            cmd_id = xCoreSDK_python.PyString()
        try:
            robot.moveReset(ec)
            _print_sdk_result("moveReset", ec)
            if ec.get("ec", 0) != 0:
                continue
            robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, 1000, 10)], cmd_id, ec)
            _print_sdk_result("moveAppend(MoveAbsJ)", ec)
            if ec.get("ec", 0) != 0:
                continue
            robot.moveStart(ec)
            _print_sdk_result("moveStart", ec)
        except Exception as exc:
            print(f"单关节运动指令执行异常: {exc}")
            continue
        if ec.get("ec", 0) != 0:
            current_power_state = robot.powerState(ec)
            current_operate_mode = robot.operateMode(ec)
            current_operation_state = robot.operationState(ec)
            print(f"moveStart 失败时电机状态: {current_power_state} ({_describe_power_state(current_power_state)})")
            print(f"moveStart 失败时模式: {current_operate_mode}")
            print(f"moveStart 失败时操作状态: {current_operation_state}")
            continue
        print(f"已下发单关节运动，cmd_id={cmd_id.content()}")
        if not _wait_until_idle(robot, ec, "等待单关节运动"):
            continue
            joint_values = robot.jointPos(ec)
        continue


def _loop_predefined_joint_motion(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """按硬编码关节值循环移动，直到用户中断。"""

    waypoints = [
        [23.14, -14.05, -6.15, -18.01, -5.38, -14.87, 9.22],
        [131.17, -12.99, -6.15, -24.95, -116.70, -8.74, 5.25],
        [147.30, -24.60, -6.15, -13.24, -130.37, 15.81, -12.85],
        [177.45, -36.53, -6.15, -10.04, -120.67, 17.51, 2.21],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]

    if not _prepare_predefined_joint_motion_loop(robot, ec):
        print("循环关节运动前置状态未准备完成，请先检查控制器状态")
        return

    joint_count = len(robot.jointPos(ec))
    for index, target_values in enumerate(waypoints, start=1):
        if len(target_values) != joint_count:
            print(
                f"第 {index} 个 waypoint 关节数不匹配，"
                f"expected={joint_count}, actual={len(target_values)}"
            )
            return

    print("开始循环移动。按 Ctrl+C 中断并退出。")
    try:
        while True:
            for target_values in waypoints:
                print(f"移动到关节值(deg): {_format_sequence(target_values)}")
                target_joint = _make_joint_position(_deg_to_rad(list(target_values)))
                cmd_id = xCoreSDK_python.PyString()
                robot.moveReset(ec)
                _print_sdk_result("moveReset", ec)
                if ec.get("ec", 0) != 0:
                    return
                robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, 500, 10)], cmd_id, ec)
                _print_sdk_result("moveAppend(MoveAbsJ)", ec)
                if ec.get("ec", 0) != 0:
                    return
                robot.moveStart(ec)
                _print_sdk_result("moveStart", ec)
                if ec.get("ec", 0) != 0:
                    current_power_state = robot.powerState(ec)
                    current_operate_mode = robot.operateMode(ec)
                    current_operation_state = robot.operationState(ec)
                    print(f"moveStart 失败时电机状态: {current_power_state} ({_describe_power_state(current_power_state)})")
                    print(f"moveStart 失败时模式: {current_operate_mode}")
                    print(f"moveStart 失败时操作状态: {current_operation_state}")
                    return
                print(f"已下发循环关节运动，cmd_id={cmd_id.content()}")
                _wait_until_idle(robot, ec, "等待循环关节运动")
    except KeyboardInterrupt:
        print("用户中断，停止循环移动")


# endregion


# region 主菜单
def _select_active_arm(connected_arms: dict[str, ConnectedArm], current_arm_side: str) -> str:
    """在主菜单中切换当前控制的机械臂。"""

    print("可选机械臂:")
    print("  1. left")
    print("  2. right")
    print("  q. 返回主菜单")
    choice = input("请选择机械臂: ").strip().lower()
    if choice == "1":
        return "left"
    if choice == "2":
        return "right"
    if choice == "q":
        return current_arm_side
    print("无效选择，保持当前机械臂不变")
    return current_arm_side


def _main_menu(connected_arms: dict[str, ConnectedArm]) -> None:
    """主菜单循环。"""

    current_arm_side = "left"
    while True:
        connected_arm = connected_arms[current_arm_side]
        robot = connected_arm.robot
        ec = connected_arm.ec
        print(
            f"\n当前机械臂: {current_arm_side} "
            f"(ip={connected_arm.config.robot_ip}, type={connected_arm.robot_type})"
        )
        print("可选操作:")
        print("  0. 切换机械臂")
        print("  1. 打开电机")
        print("  2. 关闭电机")
        print("  3. 切换模式")
        print("  4. 开关拖动")
        print("  5. 笛卡尔空间控制")
        print("  6. 关节空间控制")
        print("  7. 单关节控制")
        print("  8. 硬编码关节值循环移动")
        print("  10. 急停复位")
        print("  11. 软限位开关")
        print("  13. 打印当前机械臂状态")
        print("  q. 退出")
        choice = input("请选择: ").strip().lower()
        if choice == "0":
            current_arm_side = _select_active_arm(connected_arms, current_arm_side)
        elif choice == "1":
            _set_motor_state(robot, ec, True)
        elif choice == "2":
            _set_motor_state(robot, ec, False)
        elif choice == "3":
            _switch_mode(robot, ec)
        elif choice == "4":
            _toggle_drag(connected_arm)
        elif choice == "5":
            _cartesian_control_loop(robot, ec)
        elif choice == "6":
            _joint_control_loop(robot, ec)
        elif choice == "7":
            _single_joint_control_loop(robot, ec)
        elif choice == "8":
            _loop_predefined_joint_motion(robot, ec)
        elif choice == "10":
            _recover_estop(robot, ec)
        elif choice == "11":
            _toggle_soft_limit(robot, ec)
        elif choice == "13":
            _print_connected_arm_snapshot(connected_arm)
        elif choice == "q":
            return
        else:
            print("无效选择")


def main() -> int:
    """程序入口。"""

    connected_arms = _connect_arms(_default_connection_configs())
    try:
        _print_connected_arm_snapshot(connected_arms["left"])
        _main_menu(connected_arms)
        return 0
    finally:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)


if __name__ == "__main__":
    raise SystemExit(main())
