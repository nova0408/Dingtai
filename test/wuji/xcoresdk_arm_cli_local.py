#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdk.xcoresdk import xCoreSDK_python

# 机械臂控制器位于 Orin 所连交换机的新网段。
# LEFT_ARM_CONTROLLER_IP = "192.168.100.161"
# RIGHT_ARM_CONTROLLER_IP = "192.168.100.160"
LEFT_ARM_CONTROLLER_IP = "192.168.100.218"
RIGHT_ARM_CONTROLLER_IP = "192.168.100.181"
MM_PER_M = 1000.0
DEFAULT_CARTESIAN_SPEED = 50.0
DEFAULT_CARTESIAN_ZONE = 1.0
DEFAULT_JOINT_SPEED = 1000.0
DEFAULT_JOINT_ZONE = 10.0
# 预设关节轨迹的默认末端线速度，单位 mm/s。
DEFAULT_PREDEFINED_JOINT_SPEED = 1000.0
DEFAULT_PREDEFINED_JOINT_ZONE = 10.0
DEFAULT_POWER_ON_TIMEOUT_S = 3.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
POSITION_POLL_INTERVAL_S = 0.2
# 运动状态轮询间隔，单位 s。
MOTION_STATE_POLL_INTERVAL_S = 0.1
# 预设轨迹目标与当前关节角一致时的跳过容差，单位 deg。
PREDEFINED_JOINT_SKIP_TOLERANCE_DEG = 0.01
DEFAULT_TOOL_NAME = "g_tool_0"
DEFAULT_WOBJ_NAME = "g_wobj_0"
EXPECTED_ARM_TYPES = {
    "left": "AR5-5_0.8L-W4C6C11",
    "right": "AR5-5_0.8R-W4C6C11",
    # "left": "AR5-5_0.8L-W4C1C9-ZY2",
    # "right": "AR5-5_0.8R-W4C1C9-ZY2",
}
# 左臂硬编码循环轨迹，每个 waypoint 为 J1-J7 关节角，单位 deg。
LEFT_ARM_PREDEFINED_JOINT_WAYPOINTS_DEG: tuple[tuple[float, ...], ...] = (
    (0.00, 50.00, 0.00, 50.00, 0.00, 0.00, 0.00),
    (10.00, 50.00, 20.00, 50.00, 10.00, 0.00, 0.00),
    (0.00, -20.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
)
# 右臂硬编码循环轨迹，每个 waypoint 为 J1-J7 关节角，单位 deg。
RIGHT_ARM_PREDEFINED_JOINT_WAYPOINTS_DEG: tuple[tuple[float, ...], ...] = (
    (0.00, 50.00, 0.00, 50.00, 0.00, 0.00, 0.00),
    (10.00, 50.00, 20.00, 50.00, 10.00, 0.00, 0.00),
    (0.00, -20.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
)


@dataclass(slots=True)
class ConnectedArm:
    """单台已连接机械臂的运行上下文。"""

    arm_side: str
    "机械臂侧别，取值为 `left` 或 `right`。"

    robot_ip: str
    "SDK 直连的机械臂控制器地址。"

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


@dataclass(frozen=True, slots=True)
class ParsedCartesianPoseInput:
    """笛卡尔位姿输入解析结果。"""

    xyz_mm: tuple[float, float, float]
    "平移分量，单位 mm。"

    rpy_deg: tuple[float, float, float]
    "姿态分量，单位 deg。"

    has_elbow: bool | None
    "肘部约束是否显式输入；`None` 表示沿用当前位姿上下文。"

    elbow_deg: float | None
    "肘部角度，单位 deg；`None` 表示沿用当前位姿上下文。"

    conf_data: tuple[int, ...] | None
    "构型约束；`None` 表示沿用当前位姿上下文。"


def _parse_cartesian_pose_input(raw_text: str) -> ParsedCartesianPoseInput:
    """解析笛卡尔目标，兼容纯 xyzrpy 与完整 pose 记录格式。"""

    stripped_text = raw_text.strip()
    if stripped_text == "":
        raise ValueError("未输入任何笛卡尔目标")
    if "[" not in stripped_text and "]" not in stripped_text:
        target_values = _parse_float_list(stripped_text, expected_len=6)
        return ParsedCartesianPoseInput(
            xyz_mm=(target_values[0], target_values[1], target_values[2]),
            rpy_deg=(target_values[3], target_values[4], target_values[5]),
            has_elbow=None,
            elbow_deg=None,
            conf_data=None,
        )
    parsed = ast.literal_eval(stripped_text)
    if not isinstance(parsed, list):
        raise ValueError("笛卡尔目标必须是 list 格式")
    if len(parsed) == 6:
        target_values = [float(value) for value in parsed]
        return ParsedCartesianPoseInput(
            xyz_mm=(target_values[0], target_values[1], target_values[2]),
            rpy_deg=(target_values[3], target_values[4], target_values[5]),
            has_elbow=None,
            elbow_deg=None,
            conf_data=None,
        )
    if len(parsed) != 9:
        raise ValueError(f"笛卡尔目标长度无效，expected=6 or 9, actual={len(parsed)}")
    conf_data_raw = parsed[8]
    if not isinstance(conf_data_raw, list):
        raise ValueError("笛卡尔目标 confData 必须是 list")
    return ParsedCartesianPoseInput(
        xyz_mm=(float(parsed[0]), float(parsed[1]), float(parsed[2])),
        rpy_deg=(float(parsed[3]), float(parsed[4]), float(parsed[5])),
        has_elbow=bool(parsed[6]),
        elbow_deg=float(parsed[7]),
        conf_data=tuple(int(value) for value in conf_data_raw),
    )


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


def _format_frame_values(frame: xCoreSDK_python.Frame) -> str:
    """格式化 frame 的位姿参数。"""

    return (
        f"trans(m)=[{_format_sequence(frame.trans, decimals=4)}], "
        f"rpy(deg)=[{_format_sequence(_rad_to_deg(frame.rpy), decimals=2)}]"
    )


def _apply_named_toolset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> xCoreSDK_python.Toolset | None:
    """把 CLI 当前使用的工具/工件坐标系固定到命名对象。"""

    toolset = robot.setToolset(DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME, ec)
    _print_sdk_result(f"setToolset({DEFAULT_TOOL_NAME}, {DEFAULT_WOBJ_NAME})", ec)
    if ec.get("ec", 0) != 0:
        return None
    return toolset


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
    if _apply_named_toolset(robot, ec) is None:
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
    print(f"当前电机状态：{current_power_state} ({_describe_power_state(current_power_state)})")
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
        print(f"当前电机状态：{power_state} ({_describe_power_state(power_state)})")
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
    print(f"checkPath 目标关节 (deg): {_format_sequence(_rad_to_deg(result_joint))}")
    return True


def _print_cartesian_ik_preview(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    target_pose: xCoreSDK_python.CartesianPosition,
) -> xCoreSDK_python.JointPosition | None:
    """输入笛卡尔目标后，立即打印当前关节与逆解结果。"""

    current_joint_deg = _rad_to_deg(robot.jointPos(ec))
    robot_model = robot.model()
    toolset = robot.toolset(ec)
    _print_sdk_result("toolset", ec)
    if ec.get("ec", 0) != 0:
        return None
    target_joint = robot_model.calcIk(target_pose, toolset, ec)
    _print_sdk_result("calcIk", ec)
    print(f"当前关节值 (deg): {_format_sequence(current_joint_deg)}")
    if ec.get("ec", 0) != 0:
        return None
    print(f"目标逆解值 (deg): {_format_sequence(_rad_to_deg(target_joint))}")
    fk_pose = robot_model.calcFk(target_joint, toolset, ec)
    _print_sdk_result("calcFk(calcIk(target))", ec)
    if ec.get("ec", 0) != 0:
        return target_joint
    fk_xyz_mm = _m_to_mm(fk_pose.trans)
    fk_rpy_deg = _rad_to_deg(fk_pose.rpy)
    target_xyz_mm = _m_to_mm(target_pose.trans)
    target_rpy_deg = _rad_to_deg(target_pose.rpy)
    xyz_error_mm = [fk_xyz_mm[index] - target_xyz_mm[index] for index in range(3)]
    rpy_error_deg = [fk_rpy_deg[index] - target_rpy_deg[index] for index in range(3)]
    print(f"逆解回代 trans(mm): {_format_sequence(fk_xyz_mm)}")
    print(f"逆解回代 rpy(deg): {_format_sequence(fk_rpy_deg)}")
    print(f"逆解回代误差 trans(mm): {_format_sequence(xyz_error_mm, decimals=4)}")
    print(f"逆解回代误差 rpy(deg): {_format_sequence(rpy_error_deg, decimals=4)}")
    return target_joint


def _prompt_motion_speed(current_speed: float, label: str) -> float:
    """调整当前模式下的速度参数。"""

    while True:
        print(f"当前{label}速度：{current_speed:.2f}")
        raw_text = input(f"请输入新的{label}速度，直接回车保持当前值，或输入 q 返回：").strip().lower()
        if raw_text in {"", "q"}:
            return current_speed
        try:
            new_speed = float(raw_text)
        except ValueError:
            print("速度输入无效")
            continue
        if new_speed <= 0:
            print("速度必须大于 0")
            continue
        return new_speed


def _print_motion_speed_status(label: str, speed: float, zone: float | None = None) -> None:
    """打印当前运动参数，方便在进入模式后直接确认。"""

    if zone is None:
        print(f"当前{label}速度：{speed:.2f}")
        return
    print(f"当前{label}速度：{speed:.2f}, zone: {zone:.2f}")


def _recover_estop(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """执行急停复位。"""

    robot.recoverState(1, ec)
    _print_sdk_result("recoverState(1)", ec)


def _print_current_arm_state(connected_arm: ConnectedArm) -> None:
    """打印主菜单进入时需要确认的状态。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    operate_mode = robot.operateMode(ec)
    operation_state = robot.operationState(ec)
    power_state = robot.powerState(ec)
    print(
        f"当前机械臂：{connected_arm.arm_side} (ip={connected_arm.robot_ip}, type={connected_arm.robot_type}, uid={connected_arm.robot_uid})"
    )
    print(
        f"当前模式/状态/电机：{operate_mode} / {operation_state} / {power_state} ({_describe_power_state(power_state)})"
    )
    toolset = robot.toolset(ec)
    _print_sdk_result("toolset", ec)
    if ec.get("ec", 0) == 0:
        print(f"当前 CLI 笛卡尔参考：tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
        print(f"当前工具坐标系 {DEFAULT_TOOL_NAME}: {_format_frame_values(toolset.end)}")
        print(f"当前工件坐标系 {DEFAULT_WOBJ_NAME}: {_format_frame_values(toolset.ref)}")


def _print_cartesian_pose(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """打印当前笛卡尔空间位姿。"""

    pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    _print_sdk_result("cartPosture(endInRef)", ec)
    if ec.get("ec", 0) != 0:
        return
    toolset = robot.toolset(ec)
    _print_sdk_result("toolset", ec)
    print("当前笛卡尔空间位姿：")
    print(f"  基准：endInRef @ tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}")
    print(f"  trans(mm): {_format_sequence(_m_to_mm(pose.trans))}")
    print(f"  rpy(deg): {_format_sequence(_rad_to_deg(pose.rpy))}")
    print(f"  hasElbow: {pose.hasElbow}, elbow(deg): {math.degrees(pose.elbow):.2f}, confData: {pose.confData}")
    if ec.get("ec", 0) == 0:
        print(f"  tool frame: {_format_frame_values(toolset.end)}")
        print(f"  wobj frame: {_format_frame_values(toolset.ref)}")


def _wait_until_idle(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    prompt: str,
    timeout_s: float = 5.0,
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

    deadline = time.monotonic() + timeout_s
    last_state: xCoreSDK_python.OperationState | None = None
    while True:
        time.sleep(MOTION_STATE_POLL_INTERVAL_S)
        state = robot.operationState(ec)
        if ec.get("ec", 0) != 0:
            print(f"{prompt} 查询失败：ec={ec.get('ec', 0)}, message={ec.get('message', '')}")
            return False
        if state in (
            xCoreSDK_python.OperationState.idle,
            xCoreSDK_python.OperationState.unknown,
        ):
            print(f"{prompt}已结束：{state}")
            return True
        if state != last_state:
            print(f"{prompt}：{state}")
            last_state = state
        if time.monotonic() >= deadline:
            print(f"{prompt} 超时：超过 {timeout_s:.1f} 秒仍未结束")
            return False


def _wait_for_power_off(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float = 3.0,
) -> bool:
    """等待机器人确认进入下电状态。"""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        power_state = robot.powerState(ec)
        print(f"当前电机状态：{power_state} ({_describe_power_state(power_state)})")
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
    print(f"循环开始前操作状态：{operation_state}")
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


def _drag_record_loop(connected_arm: ConnectedArm) -> list[dict[str, str]]:
    """拖动开启后的记录模式。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    records: list[dict[str, str]] = []
    print("已进入记录模式。")
    try:
        while True:
            print("直接回车记录当前臂的 joints / pose，输入 q 退出并保存。")
            raw_text = input("请输入：").strip().lower()
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
            record: dict[str, str] = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "type": "arm",
                "joints": _format_joint_values(_rad_to_deg([float(value) for value in joint_values])),
                "pose": _format_pose_values(cart_pose),
            }
            records.append(record)
            print(f"已记录第 {len(records)} 条")
            print(f"  type: {record['type']}, 时间：{record['timestamp']}")
            print(f"  joints: {record['joints']}")
            print(f"  pose: {record['pose']}")
    except KeyboardInterrupt:
        print()
        print("用户中断，退出记录模式")
        return records


def _format_joint_values(values: list[float] | tuple[float, ...] | None) -> str:
    if values is None:
        return "NaN"
    return "[" + ", ".join(f"{float(value):.2f}" for value in values) + "]"


def _format_pose_values(pose: xCoreSDK_python.CartesianPosition | None) -> str:
    if pose is None:
        return "NaN"
    trans_mm = _m_to_mm(pose.trans)
    rpy_deg = _rad_to_deg(pose.rpy)
    return (
        f"[{trans_mm[0]:.2f}, {trans_mm[1]:.2f}, {trans_mm[2]:.2f}, "
        f"{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}, "
        f"{bool(pose.hasElbow)}, {math.degrees(float(pose.elbow)):.2f}, {list(pose.confData)}]"
    )


def _write_drag_records_csv(records: list[dict[str, str]], arm_side: str) -> Path | None:
    if not records:
        return None
    csv_path = Path.cwd() / f"xcoresdk_drag_records_{arm_side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "timestamp",
        "type",
        "joints",
        "pose",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return csv_path


# endregion


# region 机器人控制
def _detect_arm_side(robot_type: str) -> str:
    """根据控制器上报的机型名称判断左右臂。"""

    for arm_side, expected_robot_type in EXPECTED_ARM_TYPES.items():
        if robot_type == expected_robot_type:
            return arm_side
    raise ValueError(f"未识别的机器人型号：{robot_type}")


def _connect_arms(configs: list[tuple[str, str]]) -> dict[str, ConnectedArm]:
    """连接多台机械臂，并按控制器上报的型号归类左右臂。"""

    connected_arms: dict[str, ConnectedArm] = {}
    try:
        for expected_side, robot_ip in configs:
            ec: dict[str, object] = {}
            robot = xCoreSDK_python.xMateErProRobot(robot_ip)
            robot_info = robot.robotInfo(ec)
            _print_sdk_result(f"robotInfo({robot_ip})", ec)
            if ec.get("ec", 0) != 0:
                raise RuntimeError(f"读取机器人信息失败：ip={robot_ip}")
            if _apply_named_toolset(robot, ec) is None:
                raise RuntimeError(
                    f"设置默认工具/工件失败：ip={robot_ip}, tool={DEFAULT_TOOL_NAME}, wobj={DEFAULT_WOBJ_NAME}"
                )
            arm_side = _detect_arm_side(robot_info.type)
            if arm_side != expected_side:
                raise RuntimeError(
                    f"机器人型号与预期侧别不匹配：ip={robot_ip}, expected={expected_side}, actual={arm_side}"
                )
            if arm_side in connected_arms:
                raise RuntimeError(
                    f"检测到重复的 {arm_side} 机械臂："
                    f"existing={connected_arms[arm_side].robot_ip}, current={robot_ip}"
                )
            connected_arm = ConnectedArm(
                arm_side=arm_side,
                robot_ip=robot_ip,
                robot=robot,
                robot_type=robot_info.type,
                robot_uid=robot_info.id,
                ec=ec,
            )
            connected_arms[arm_side] = connected_arm
            print(f"已连接 {arm_side} arm: ip={robot_ip}, " f"type={robot_info.type}, uid={robot_info.id}")
        missing_arm_sides = [arm_side for arm_side in EXPECTED_ARM_TYPES if arm_side not in connected_arms]
        if missing_arm_sides:
            raise RuntimeError(f"缺少目标机械臂连接：{', '.join(missing_arm_sides)}")
        return connected_arms
    except Exception:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)
        raise


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
    print("可选模式：")
    print("  1. manual")
    print("  2. automatic")
    choice = input("请选择模式：").strip()
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
            enable_drag_button=False,
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
        print(f"已保存到：{csv_path}")
        print("拖动已关闭")


def _cartesian_control_loop(connected_arm: ConnectedArm) -> None:
    """笛卡尔空间控制循环。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    if not _ensure_nrt_motion_ready(robot, ec):
        print("笛卡尔运动前置状态未准备完成，请先检查控制器状态")
        return

    cartesian_speed = DEFAULT_CARTESIAN_SPEED
    cartesian_zone = DEFAULT_CARTESIAN_ZONE
    while True:
        _print_cartesian_pose(robot, ec)
        _print_motion_speed_status("笛卡尔", cartesian_speed, cartesian_zone)
        print("输入新的 xyzrpy，单位分别为 mm 和 deg")
        print("也支持输入完整 pose 列格式：[x, y, z, r, p, y, hasElbow, elbowDeg, confData]")
        print("输入 s 调整速度，输入 q 返回主菜单")
        raw_text = input("目标 xyzrpy: ").strip()
        if raw_text.lower() == "s":
            cartesian_speed = _prompt_motion_speed(cartesian_speed, "笛卡尔")
            continue
        if raw_text.lower() == "q":
            return
        try:
            parsed_pose = _parse_cartesian_pose_input(raw_text)
        except ValueError as exc:
            logger.warning("笛卡尔输入格式错误：{}", exc)
            print(f"输入格式错误：{exc}")
            continue
        current_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
        target_xyz_mm = list(parsed_pose.xyz_mm)
        target_rpy_deg = list(parsed_pose.rpy_deg)
        target_pose = xCoreSDK_python.CartesianPosition(_mm_to_m(target_xyz_mm) + _deg_to_rad(target_rpy_deg))
        _copy_cartesian_pose_context(current_pose, target_pose)
        if parsed_pose.has_elbow is not None:
            target_pose.hasElbow = parsed_pose.has_elbow
        if parsed_pose.elbow_deg is not None:
            target_pose.elbow = _deg_to_rad([parsed_pose.elbow_deg])[0]
        if parsed_pose.conf_data is not None:
            target_pose.confData = list(parsed_pose.conf_data)
        target_joint = _print_cartesian_ik_preview(robot, ec, target_pose)
        if target_joint is None:
            logger.warning("笛卡尔目标未得到可用逆解，已取消执行")
            print("当前目标未得到可用逆解，已取消本次笛卡尔运动")
            continue
        print(
            "目标笛卡尔位姿："
            f"trans(mm)={_format_sequence(target_xyz_mm)}, "
            f"rpy(deg)={_format_sequence(target_rpy_deg)}, "
            f"speed={cartesian_speed:.2f}, "
            f"zone={cartesian_zone:.2f}, "
            "motion=movel->moveabsj fallback, "
            f"hasElbow={target_pose.hasElbow}, "
            f"elbow(deg)={math.degrees(target_pose.elbow):.2f}, "
            f"confData={target_pose.confData}"
        )
        should_fallback_to_move_abs_j = not _validate_cartesian_target(robot, ec, target_pose)
        if should_fallback_to_move_abs_j:
            logger.warning(
                "MoveL 路径检查失败，回退 MoveAbsJ: xyz(mm)=[{}] rpy(deg)=[{}]",
                _format_sequence(target_xyz_mm),
                _format_sequence(target_rpy_deg),
            )
            print("当前目标未通过 MoveL 路径检查，回退为 MoveAbsJ 关节空间运动")
        cmd_id = xCoreSDK_python.PyString()
        robot.moveReset(ec)
        _print_sdk_result("moveReset", ec)
        if ec.get("ec", 0) != 0:
            continue
        if should_fallback_to_move_abs_j:
            robot.moveAppend(
                [xCoreSDK_python.MoveAbsJCommand(target_joint, DEFAULT_JOINT_SPEED, DEFAULT_JOINT_ZONE)],
                cmd_id,
                ec,
            )
            _print_sdk_result("moveAppend(MoveAbsJ)", ec)
        else:
            robot.moveAppend(
                [xCoreSDK_python.MoveLCommand(target_pose, cartesian_speed, cartesian_zone)],
                cmd_id,
                ec,
            )
            _print_sdk_result("moveAppend(MoveL)", ec)
        if ec.get("ec", 0) != 0:
            continue
        robot.moveStart(ec)
        _print_sdk_result("moveStart", ec)
        if ec.get("ec", 0) != 0:
            logger.warning(
                "运动启动失败：motion={} ec={} message={}",
                "MoveAbsJ" if should_fallback_to_move_abs_j else "MoveL",
                ec.get("ec", 0),
                ec.get("message", ""),
            )
            current_power_state = robot.powerState(ec)
            current_operate_mode = robot.operateMode(ec)
            current_operation_state = robot.operationState(ec)
            print(f"moveStart 失败时电机状态：{current_power_state} ({_describe_power_state(current_power_state)})")
            print(f"moveStart 失败时模式：{current_operate_mode}")
            print(f"moveStart 失败时操作状态：{current_operation_state}")
            continue
        if should_fallback_to_move_abs_j:
            print(f"已下发 MoveAbsJ 回退运动，cmd_id={cmd_id.content()}")
        else:
            print(f"已下发笛卡尔运动，cmd_id={cmd_id.content()}")
        _wait_until_idle(robot, ec, "等待笛卡尔运动")
        _print_cartesian_pose(robot, ec)


def _joint_control_loop(connected_arm: ConnectedArm) -> None:
    """关节空间控制循环。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    if not _prepare_predefined_joint_motion_loop(robot, ec):
        print("关节运动前置状态未准备完成，请先检查控制器状态")
        return

    joint_speed = DEFAULT_JOINT_SPEED
    joint_zone = DEFAULT_JOINT_ZONE
    while True:
        joint_values = robot.jointPos(ec)
        print(f"当前关节值 (deg): {_format_sequence(_rad_to_deg(joint_values))}")
        _print_motion_speed_status("关节", joint_speed, joint_zone)
        print("输入新的关节值，单位 deg，支持空格、英文逗号或中文逗号分隔")
        print("输入 s 调整速度，输入 q 返回主菜单")
        raw_text = input("目标关节值：").strip()
        if raw_text.lower() == "s":
            joint_speed = _prompt_motion_speed(joint_speed, "关节")
            continue
        if raw_text.lower() == "q":
            return
        try:
            target_values = _parse_float_list(raw_text, expected_len=len(joint_values))
        except ValueError as exc:
            print(f"输入格式错误：{exc}")
            continue
        target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(target_values))
        cmd_id = xCoreSDK_python.PyString()
        robot.moveReset(ec)
        _print_sdk_result("moveReset", ec)
        if ec.get("ec", 0) != 0:
            return
        robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, joint_speed, joint_zone)], cmd_id, ec)
        _print_sdk_result("moveAppend(MoveAbsJ)", ec)
        if ec.get("ec", 0) != 0:
            return
        robot.moveStart(ec)
        _print_sdk_result("moveStart", ec)
        if ec.get("ec", 0) != 0:
            current_power_state = robot.powerState(ec)
            current_operate_mode = robot.operateMode(ec)
            current_operation_state = robot.operationState(ec)
            print(f"moveStart 失败时电机状态：{current_power_state} ({_describe_power_state(current_power_state)})")
            print(f"moveStart 失败时模式：{current_operate_mode}")
            print(f"moveStart 失败时操作状态：{current_operation_state}")
            return
        print(f"已下发关节运动，cmd_id={cmd_id.content()}")
        _wait_until_idle(robot, ec, "等待关节运动")


def _single_joint_control_loop(connected_arm: ConnectedArm) -> None:
    """单关节控制循环。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    if not _prepare_predefined_joint_motion_loop(robot, ec):
        print("单关节运动前置状态未准备完成，请先检查控制器状态")
        return

    single_joint_speed = DEFAULT_JOINT_SPEED
    single_joint_zone = DEFAULT_JOINT_ZONE
    while True:
        joint_values = robot.jointPos(ec)
        print(f"当前关节值 (deg): {_format_sequence(_rad_to_deg(joint_values))}")
        _print_motion_speed_status("单关节", single_joint_speed, single_joint_zone)
        print("输入 q 返回主菜单")
        print("输入 s 调整速度")
        axis_text = input(f"请选择轴编号 1-{len(joint_values)}: ").strip().lower()
        if axis_text == "s":
            single_joint_speed = _prompt_motion_speed(single_joint_speed, "单关节")
            continue
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
            print(f"当前所选轴 J{axis_index} 值 (deg): {math.degrees(joint_values[axis_index - 1]):.2f}")
            print("输入当前轴的目标值，单位 deg")
            print("输入 q 返回前一级选轴")
            raw_text = input("目标轴值：").strip()
            if raw_text.lower() == "q":
                break
            try:
                target_value = math.radians(float(raw_text))
            except ValueError:
                print("轴目标值输入无效")
                continue
            target_joint_values = list(joint_values)
            target_joint_values[axis_index - 1] = target_value
            target_joint = xCoreSDK_python.JointPosition(target_joint_values)
            cmd_id = xCoreSDK_python.PyString()
            try:
                robot.moveReset(ec)
                _print_sdk_result("moveReset", ec)
                if ec.get("ec", 0) != 0:
                    continue
                robot.moveAppend(
                    [xCoreSDK_python.MoveAbsJCommand(target_joint, single_joint_speed, single_joint_zone)], cmd_id, ec
                )
                _print_sdk_result("moveAppend(MoveAbsJ)", ec)
                if ec.get("ec", 0) != 0:
                    continue
                robot.moveStart(ec)
                _print_sdk_result("moveStart", ec)
            except Exception as exc:
                print(f"单关节运动指令执行异常：{exc}")
                continue
            if ec.get("ec", 0) != 0:
                current_power_state = robot.powerState(ec)
                current_operate_mode = robot.operateMode(ec)
                current_operation_state = robot.operationState(ec)
                print(f"moveStart 失败时电机状态：{current_power_state} ({_describe_power_state(current_power_state)})")
                print(f"moveStart 失败时模式：{current_operate_mode}")
                print(f"moveStart 失败时操作状态：{current_operation_state}")
                continue
            print(f"已下发单关节运动，cmd_id={cmd_id.content()}")
            if not _wait_until_idle(robot, ec, "等待单关节运动"):
                continue
            continue


def _loop_predefined_joint_motion(connected_arm: ConnectedArm) -> None:
    """按硬编码关节值循环移动，直到用户中断。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    waypoints = (
        LEFT_ARM_PREDEFINED_JOINT_WAYPOINTS_DEG
        if connected_arm.arm_side == "left"
        else RIGHT_ARM_PREDEFINED_JOINT_WAYPOINTS_DEG
    )

    if not _prepare_predefined_joint_motion_loop(robot, ec):
        print("循环关节运动前置状态未准备完成，请先检查控制器状态")
        return

    predefined_joint_speed = DEFAULT_PREDEFINED_JOINT_SPEED
    predefined_joint_zone = DEFAULT_PREDEFINED_JOINT_ZONE
    joint_count = len(robot.jointPos(ec))
    for index, target_values in enumerate(waypoints, start=1):
        if len(target_values) != joint_count:
            print(f"第 {index} 个 waypoint 关节数不匹配，" f"expected={joint_count}, actual={len(target_values)}")
            return

    print("开始循环移动。按 Ctrl+C 中断并退出。")
    try:
        while True:
            _print_motion_speed_status("循环关节", predefined_joint_speed, predefined_joint_zone)
            print("输入 s 调整速度，或直接按回车继续执行预设轨迹")
            raw_text = input("继续/调整：").strip().lower()
            if raw_text == "s":
                predefined_joint_speed = _prompt_motion_speed(predefined_joint_speed, "循环关节")
                continue
            for target_values in waypoints:
                print(f"移动到关节值 (deg): {_format_sequence(target_values)}")
                try:
                    _play_predefined_joint_waypoint(
                        connected_arm,
                        target_values,
                        predefined_joint_speed,
                        predefined_joint_zone,
                    )
                except RuntimeError as exc:
                    logger.warning("单臂预设轨迹停止：{}", exc)
                    return
    except KeyboardInterrupt:
        print("用户中断，停止循环移动")


def _play_predefined_joint_waypoint(
    connected_arm: ConnectedArm,
    target_values_deg: tuple[float, ...],
    speed_mm_s: float,
    zone: float,
) -> None:
    """向单臂下发一个预设关节 waypoint，并等待运动完成。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    arm_side = connected_arm.arm_side
    current_values_deg = _rad_to_deg(robot.jointPos(ec))
    _print_sdk_result(f"{arm_side}.jointPos", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{arm_side} 机械臂读取当前关节角失败")
    if all(
        abs(current_value_deg - target_value_deg) <= PREDEFINED_JOINT_SKIP_TOLERANCE_DEG
        for current_value_deg, target_value_deg in zip(current_values_deg, target_values_deg, strict=True)
    ):
        print(
            f"{arm_side} 当前关节角已与目标一致，跳过该 waypoint，"
            f"tolerance={PREDEFINED_JOINT_SKIP_TOLERANCE_DEG:.3f} deg"
        )
        return

    target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(list(target_values_deg)))
    cmd_id = xCoreSDK_python.PyString()
    robot.moveReset(ec)
    _print_sdk_result(f"{arm_side}.moveReset", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{arm_side} 机械臂 moveReset 失败")
    robot.moveAppend(
        [xCoreSDK_python.MoveAbsJCommand(target_joint, speed_mm_s, zone)],
        cmd_id,
        ec,
    )
    _print_sdk_result(f"{arm_side}.moveAppend(MoveAbsJ)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{arm_side} 机械臂 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result(f"{arm_side}.moveStart", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError(f"{arm_side} 机械臂 moveStart 失败")
    print(f"{arm_side} 已下发循环关节运动，cmd_id={cmd_id.content()}")
    if not _wait_until_idle(robot, ec, f"等待 {arm_side} 循环关节运动"):
        raise RuntimeError(f"{arm_side} 机械臂等待运动完成失败")


def _run_arm_predefined_joint_motion_thread(
    connected_arm: ConnectedArm,
    waypoints: tuple[tuple[float, ...], ...],
    speed_mm_s: float,
    zone: float,
    stop_event: Event,
) -> None:
    """在独立线程中持续播放单臂预设轨迹。"""

    try:
        robot = connected_arm.robot
        ec = connected_arm.ec
        arm_side = connected_arm.arm_side
        if not _prepare_predefined_joint_motion_loop(robot, ec):
            raise RuntimeError(f"{arm_side} 机械臂循环运动前置状态未准备完成")
        joint_count = len(robot.jointPos(ec))
        for index, target_values in enumerate(waypoints, start=1):
            if len(target_values) != joint_count:
                raise RuntimeError(
                    f"{arm_side} 第 {index} 个 waypoint 关节数不匹配："
                    f"expected={joint_count}, actual={len(target_values)}"
                )

        while not stop_event.is_set():
            for target_values in waypoints:
                if stop_event.is_set():
                    return
                print(f"{arm_side} 移动到关节值 (deg): {_format_sequence(target_values)}")
                _play_predefined_joint_waypoint(connected_arm, target_values, speed_mm_s, zone)
    except Exception:
        logger.error("{} 机械臂预设轨迹线程异常", connected_arm.arm_side)
        stop_event.set()


def _loop_both_arms_predefined_joint_motion(connected_arms: dict[str, ConnectedArm]) -> None:
    """使用两个独立线程持续播放左右臂预设轨迹，直到用户中断。"""

    speed_mm_s = DEFAULT_PREDEFINED_JOINT_SPEED
    raw_text = input("输入 s 设置双臂循环速度，其他输入直接开始：").strip().lower()
    if raw_text == "s":
        speed_mm_s = _prompt_motion_speed(speed_mm_s, "双臂循环")
    zone = DEFAULT_PREDEFINED_JOINT_ZONE
    _print_motion_speed_status("双臂循环", speed_mm_s, zone)

    stop_event = Event()
    threads = (
        Thread(
            target=_run_arm_predefined_joint_motion_thread,
            args=(
                connected_arms["left"],
                LEFT_ARM_PREDEFINED_JOINT_WAYPOINTS_DEG,
                speed_mm_s,
                zone,
                stop_event,
            ),
            name="left-arm-play",
        ),
        Thread(
            target=_run_arm_predefined_joint_motion_thread,
            args=(
                connected_arms["right"],
                RIGHT_ARM_PREDEFINED_JOINT_WAYPOINTS_DEG,
                speed_mm_s,
                zone,
                stop_event,
            ),
            name="right-arm-play",
        ),
    )
    for thread in threads:
        thread.start()

    print("双臂已分别开始持续循环播放。按 Ctrl+C 同时停止两臂并返回主菜单。")
    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.1)
    except KeyboardInterrupt:
        print("用户中断，正在同时停止两臂")
    finally:
        stop_event.set()
        for connected_arm in connected_arms.values():
            connected_arm.robot.stop(connected_arm.ec)
            _print_sdk_result(f"{connected_arm.arm_side}.stop", connected_arm.ec)
        for thread in threads:
            thread.join(timeout=DEFAULT_REQUEST_TIMEOUT_S)


# endregion


# region 主菜单
def _toggle_motor_state(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """根据当前电机状态翻转上/下电。"""

    current_power_state = robot.powerState(ec)
    target_power_state = current_power_state != xCoreSDK_python.PowerState.on
    print(f"当前电机状态：{current_power_state} ({_describe_power_state(current_power_state)})")
    print(f"切换目标状态：{target_power_state}")
    robot.setPowerState(target_power_state, ec)
    _print_sdk_result("setPowerState(toggle)", ec)


def _toggle_operate_mode(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """在手动和自动模式之间切换。"""

    current_mode = robot.operateMode(ec)
    target_mode = (
        xCoreSDK_python.OperateMode.automatic
        if current_mode == xCoreSDK_python.OperateMode.manual
        else xCoreSDK_python.OperateMode.manual
    )
    print(f"当前模式：{current_mode}")
    print(f"切换目标模式：{target_mode}")
    robot.setOperateMode(target_mode, ec)
    _print_sdk_result("setOperateMode(toggle)", ec)


def _main_menu(connected_arms: dict[str, ConnectedArm]) -> None:
    """主菜单循环。"""

    current_arm_side = "left"
    while True:
        connected_arm = connected_arms[current_arm_side]
        robot = connected_arm.robot
        ec = connected_arm.ec
        print("")
        _print_current_arm_state(connected_arm)
        print("可选操作：")
        print("  0. 切换机械臂到另一侧 1. 电机开/关切换 2. 手动/自动模式切换 3. 开关拖动")
        print("  4. 笛卡尔空间控制 5. 关节空间控制 6. 单关节控制 7. 硬编码关节值循环移动")
        print("  8. 急停复位 9. 双臂同时循环播放 q. 退出")
        choice = input("请选择：").strip().lower()
        if choice == "0":
            current_arm_side = "right" if current_arm_side == "left" else "left"
        elif choice == "1":
            _toggle_motor_state(robot, ec)
        elif choice == "2":
            _toggle_operate_mode(robot, ec)
        elif choice == "3":
            _toggle_drag(connected_arm)
        elif choice == "4":
            _cartesian_control_loop(connected_arm)
        elif choice == "5":
            _joint_control_loop(connected_arm)
        elif choice == "6":
            _single_joint_control_loop(connected_arm)
        elif choice == "7":
            _loop_predefined_joint_motion(connected_arm)
        elif choice == "8":
            _recover_estop(robot, ec)
        elif choice == "9":
            _loop_both_arms_predefined_joint_motion(connected_arms)
        elif choice == "q":
            return
        else:
            print("无效选择")


def main() -> int:
    """程序入口。"""

    arm_configs = [
        ("left", LEFT_ARM_CONTROLLER_IP),
        ("right", RIGHT_ARM_CONTROLLER_IP),
    ]
    connected_arms = _connect_arms(arm_configs)
    try:
        _main_menu(connected_arms)
        return 0
    finally:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)


if __name__ == "__main__":
    raise SystemExit(main())
