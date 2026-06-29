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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from sdk.xcoresdk import xCoreSDK_python
from common import DEFAULT_PORT, GRIPPER_PORT, create_wuyou_channel, stop_ssh_process
from src.wuji.dahuan_gripper_client import DahuanGripperClient
from src.wuji.right_hand_client import WujiRightHandClient


LEFT_ARM_IP = "192.168.1.161"
RIGHT_ARM_IP = "192.168.1.160"
MM_PER_M = 1000.0
DEFAULT_CARTESIAN_SPEED = 50.0
DEFAULT_CARTESIAN_ZONE = 1.0
DEFAULT_JOINT_SPEED = 1000.0
DEFAULT_JOINT_ZONE = 10.0
DEFAULT_PREDEFINED_JOINT_SPEED = 500.0
DEFAULT_PREDEFINED_JOINT_ZONE = 10.0
DEFAULT_POWER_ON_TIMEOUT_S = 3.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
POSITION_POLL_INTERVAL_S = 0.2
EXPECTED_ARM_TYPES = {
    "left": "AR5-5_0.8L-W4C1C9-ZY2",
    "right": "AR5-5_0.8R-W4C1C9-ZY2",
}


@dataclass(slots=True)
class ConnectedArm:
    """单台已连接机械臂的运行上下文。"""

    arm_side: str
    "机械臂侧别，取值为 `left` 或 `right`。"

    robot_ip: str
    "机器人控制器 IP 地址。"

    robot: xCoreSDK_python.xMateErProRobot
    "SDK 机器人对象。"

    robot_type: str
    "控制器上报的机器人型号。"

    robot_uid: str
    "控制器上报的机器人唯一标识。"

    ec: dict[str, object]
    "该机械臂独立复用的 SDK 错误码字典。"


@dataclass(slots=True)
class PersistentHandClients:
    """CLI 生命周期内复用的手部客户端。"""

    gripper_process: object
    gripper: DahuanGripperClient
    right_hand_process: object
    right_hand: WujiRightHandClient


@dataclass(frozen=True, slots=True)
class InterlockTarget:
    """联调硬编码目标。"""

    target_type: str
    "目标类型，arm 或 hand。"

    target_id: int
    "目标编号。"

    arm_joint_deg: list[float]
    "机械臂关节角，单位 deg。"

    arm_xyzrpye: list[float] | None
    "机械臂笛卡尔位姿，单位 mm/deg。"

    gripper_pos: int | None
    "左手夹爪位置。"

    hand_root_tip: dict[str, list[float]] | None
    "右手根部和指尖的目标值。"


# endregion

INTERLOCK_LEFT_TARGETS: tuple[InterlockTarget, ...] = (
    InterlockTarget(
        target_type="arm",
        target_id=1,
        arm_joint_deg=[-85.00, -100.00, 45.00, -50.00, -10.00, -15.00, -5.00],
        arm_xyzrpye=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_pos=1000,
        hand_root_tip=None,
    ),
    InterlockTarget(
        target_type="arm",
        target_id=2,
        arm_joint_deg=[-75.00, -90.00, 50.00, -45.00, -8.00, -12.00, -2.00],
        arm_xyzrpye=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_pos=0,
        hand_root_tip=None,
    ),
)
INTERLOCK_RIGHT_TARGETS: tuple[InterlockTarget, ...] = (
    InterlockTarget(
        target_type="arm",
        target_id=1,
        arm_joint_deg=[-100.00, 100.00, 135.00, -55.00, 0.00, -15.00, 10.00],
        arm_xyzrpye=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_pos=None,
        hand_root_tip={"root": [0.0, 0.0, 0.0], "tip": [1.0, 1.0, 1.0]},
    ),
    InterlockTarget(
        target_type="arm",
        target_id=2,
        arm_joint_deg=[-90.00, 95.00, 120.00, -50.00, 3.00, -10.00, 8.00],
        arm_xyzrpye=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_pos=None,
        hand_root_tip={"root": [1.0, 1.0, 1.0], "tip": [0.0, 0.0, 0.0]},
    ),
)

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


def _prompt_motion_speed(current_speed: float, label: str) -> float:
    """调整当前模式下的速度参数。"""

    while True:
        print(f"当前{label}速度: {current_speed:.2f}")
        raw_text = input(f"请输入新的{label}速度，或输入 q 返回: ").strip().lower()
        if raw_text == "q":
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
        print(f"当前{label}速度: {speed:.2f}")
        return
    print(f"当前{label}速度: {speed:.2f}, zone: {zone:.2f}")


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
    print(f"当前机械臂: {connected_arm.arm_side} (ip={connected_arm.robot_ip}, type={connected_arm.robot_type}, uid={connected_arm.robot_uid})")
    print(f"当前模式/状态/电机: {operate_mode} / {operation_state} / {power_state} ({_describe_power_state(power_state)})")


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
    hand_clients: PersistentHandClients,
) -> list[dict[str, str]]:
    """拖动开启后的记录模式。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    records: list[dict[str, str]] = []
    print("已进入记录模式。")
    try:
        while True:
            print("直接回车记录当前臂的 joints / pose，输入 s 进入手部手动记录，输入 q 退出并保存。")
            raw_text = input("请输入: ").strip().lower()
            if raw_text == "q":
                print("退出记录模式")
                return records
            if raw_text == "s":
                if connected_arm.arm_side == "left":
                    _manual_gripper_record(hand_clients.gripper, records)
                else:
                    _manual_m11_record(hand_clients.right_hand, records)
                continue
            if raw_text != "":
                print("无效输入，请直接回车、输入 s 或输入 q")
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
            print(f"  type: {record['type']}, 时间: {record['timestamp']}")
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


def _manual_gripper_record(client: DahuanGripperClient, records: list[dict[str, str]]) -> None:
    """左臂拖动记录时的夹爪手动控制。"""

    status = client.get_status()
    print(f"当前 gripper 值: {status.position}")
    raw_value = input("请输入单个 gripper 值并回车: ").strip()
    if raw_value == "":
        print("已取消 gripper 手动记录")
        return
    target_value = int(raw_value)
    if not client.set_pos(target_value):
        raise RuntimeError("gripper 手动下发失败")
    time.sleep(2.0)
    current_status = client.get_status()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "gripper",
        "joints": "NaN",
        "pose": f"{float(current_status.position or 0):.2f}",
    }
    records.append(record)
    print(f"已记录第 {len(records)} 条")
    print(f"  type: {record['type']}, pose: {record['pose']}")


def _manual_m11_record(hand: WujiRightHandClient, records: list[dict[str, str]]) -> None:
    """右臂拖动记录时的 m11 手动控制。"""

    group_choice = input("请选择 root/tip 并回车: ").strip().lower()
    if group_choice not in {"root", "tip"}:
        print("已取消 m11 手动记录")
        return
    state = hand.get_hand_state(include_tactile=False)
    if state is None:
        raise RuntimeError("右手状态不可用")
    positions = [float(item["position"]) for item in state["actuators"]]
    selected_ids = (0, 1, 2, 3, 4) if group_choice == "root" else (5, 6, 7, 8, 9, 10)
    print("当前选中四指/关节值:")
    print(_format_joint_values([positions[index] for index in selected_ids]))
    raw_value = input("请输入统一值并回车: ").strip()
    if raw_value == "":
        print("已取消 m11 手动记录")
        return
    target_value = float(raw_value)
    for index in selected_ids:
        positions[index] = target_value
    if not hand.set_hand_state(positions):
        raise RuntimeError("m11 手动下发失败")
    time.sleep(2.0)
    current_state = hand.get_hand_state(include_tactile=False)
    if current_state is None:
        raise RuntimeError("右手状态不可用")
    current_positions = [float(item["position"]) for item in current_state["actuators"]]
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "m11",
        "joints": _format_joint_values(current_positions),
        "pose": "NaN",
    }
    records.append(record)
    print(f"已记录第 {len(records)} 条")
    print(f"  type: {record['type']}, joints: {record['joints']}")


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
    raise ValueError(f"未识别的机器人型号: {robot_type}")


def _connect_arms() -> dict[str, ConnectedArm]:
    """连接多台机械臂，并按控制器上报的型号归类左右臂。"""

    configs = [
        ("left", LEFT_ARM_IP),
        ("right", RIGHT_ARM_IP),
    ]
    connected_arms: dict[str, ConnectedArm] = {}
    try:
        for expected_side, robot_ip in configs:
            ec: dict[str, object] = {}
            robot = xCoreSDK_python.xMateErProRobot(robot_ip)
            robot_info = robot.robotInfo(ec)
            _print_sdk_result(f"robotInfo({robot_ip})", ec)
            if ec.get("ec", 0) != 0:
                raise RuntimeError(f"读取机器人信息失败: ip={robot_ip}")
            arm_side = _detect_arm_side(robot_info.type)
            if arm_side != expected_side:
                raise RuntimeError(f"机器人型号与预期侧别不匹配: ip={robot_ip}, expected={expected_side}, actual={arm_side}")
            if arm_side in connected_arms:
                raise RuntimeError(
                    f"检测到重复的 {arm_side} 机械臂: "
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
            print(
                f"已连接 {arm_side} arm: ip={robot_ip}, "
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


def _toggle_drag(connected_arm: ConnectedArm, hand_clients: PersistentHandClients) -> None:
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
        records = _drag_record_loop(connected_arm, hand_clients)
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

    cartesian_speed = DEFAULT_CARTESIAN_SPEED
    cartesian_zone = DEFAULT_CARTESIAN_ZONE
    while True:
        _print_cartesian_pose(robot, ec)
        _print_motion_speed_status("笛卡尔", cartesian_speed, cartesian_zone)
        print("输入新的 xyzrpy，单位分别为 mm 和 deg")
        print("输入 s 调整速度，输入 q 返回主菜单")
        raw_text = input("目标 xyzrpy: ").strip()
        if raw_text.lower() == "s":
            cartesian_speed = _prompt_motion_speed(cartesian_speed, "笛卡尔")
            continue
        if raw_text.lower() == "q":
            return
        try:
            target_values = _parse_float_list(raw_text, expected_len=6)
        except ValueError as exc:
            print(f"输入格式错误: {exc}")
            continue
        current_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
        target_pose = xCoreSDK_python.CartesianPosition(_mm_to_m(target_values[:3]) + _deg_to_rad(target_values[3:]))
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
            f"speed={cartesian_speed:.2f}, "
            f"zone={cartesian_zone:.2f}, "
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
                [xCoreSDK_python.MoveLCommand(target_pose, cartesian_speed, cartesian_zone)],
                cmd_id,
                ec,
            )
            _print_sdk_result("moveAppend(MoveL)", ec)
        else:
            robot.moveAppend(
                [xCoreSDK_python.MoveJCommand(target_pose, cartesian_speed, cartesian_zone)],
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

    joint_speed = DEFAULT_JOINT_SPEED
    joint_zone = DEFAULT_JOINT_ZONE
    while True:
        joint_values = robot.jointPos(ec)
        print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
        _print_motion_speed_status("关节", joint_speed, joint_zone)
        print("输入新的关节值，单位 deg，支持空格、英文逗号或中文逗号分隔")
        print("输入 s 调整速度，输入 q 返回主菜单")
        raw_text = input("目标关节值: ").strip()
        if raw_text.lower() == "s":
            joint_speed = _prompt_motion_speed(joint_speed, "关节")
            continue
        if raw_text.lower() == "q":
            return
        try:
            target_values = _parse_float_list(raw_text, expected_len=len(joint_values))
        except ValueError as exc:
            print(f"输入格式错误: {exc}")
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

    single_joint_speed = DEFAULT_JOINT_SPEED
    single_joint_zone = DEFAULT_JOINT_ZONE
    while True:
        joint_values = robot.jointPos(ec)
        print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
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
            target_joint = xCoreSDK_python.JointPosition(target_joint_values)
            cmd_id = xCoreSDK_python.PyString()
        try:
            robot.moveReset(ec)
            _print_sdk_result("moveReset", ec)
            if ec.get("ec", 0) != 0:
                continue
            robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, single_joint_speed, single_joint_zone)], cmd_id, ec)
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

    predefined_joint_speed = DEFAULT_PREDEFINED_JOINT_SPEED
    predefined_joint_zone = DEFAULT_PREDEFINED_JOINT_ZONE
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
            _print_motion_speed_status("循环关节", predefined_joint_speed, predefined_joint_zone)
            print("输入 s 调整速度，或直接按回车继续执行预设轨迹")
            raw_text = input("继续/调整: ").strip().lower()
            if raw_text == "s":
                predefined_joint_speed = _prompt_motion_speed(predefined_joint_speed, "循环关节")
                continue
            for target_values in waypoints:
                print(f"移动到关节值(deg): {_format_sequence(target_values)}")
                target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(list(target_values)))
                cmd_id = xCoreSDK_python.PyString()
                robot.moveReset(ec)
                _print_sdk_result("moveReset", ec)
                if ec.get("ec", 0) != 0:
                    return
                robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, predefined_joint_speed, predefined_joint_zone)], cmd_id, ec)
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
def _toggle_motor_state(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """根据当前电机状态翻转上/下电。"""

    current_power_state = robot.powerState(ec)
    target_power_state = current_power_state != xCoreSDK_python.PowerState.on
    print(f"当前电机状态: {current_power_state} ({_describe_power_state(current_power_state)})")
    print(f"切换目标状态: {target_power_state}")
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
    print(f"当前模式: {current_mode}")
    print(f"切换目标模式: {target_mode}")
    robot.setOperateMode(target_mode, ec)
    _print_sdk_result("setOperateMode(toggle)", ec)


def _read_nonblocking_key() -> str | None:
    """读取一个非阻塞按键。"""

    if sys.platform == "win32":
        if not msvcrt.kbhit():
            return None
        key = msvcrt.getwch()
        if key in {"\r", "\n"}:
            return "\n"
        return key.lower()
    try:
        import select
    except Exception:
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    key = sys.stdin.read(1)
    if key in {"\r", "\n"}:
        return "\n"
    return key.lower()


def _poll_gripper_until_idle(client: DahuanGripperClient, target_position: int) -> None:
    deadline = time.monotonic() + DEFAULT_REQUEST_TIMEOUT_S
    while True:
        status = client.get_status()
        print(f"gripper pos={status.position}/r")
        if int(status.position or 0) == int(target_position):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("夹爪联调执行超时")
        time.sleep(POSITION_POLL_INTERVAL_S)


def _poll_right_hand_until_idle(hand: WujiRightHandClient, target_name: str) -> None:
    deadline = time.monotonic() + DEFAULT_REQUEST_TIMEOUT_S
    while True:
        state = hand.get_hand_state(include_tactile=False)
        if state is None:
            raise RuntimeError("右手状态不可用")
        positions = [float(item["position"]) for item in state["actuators"]]
        print(
            f"hand {target_name}: "
            f"root={positions[0]:.6f},{positions[1]:.6f},{positions[2]:.6f}/r "
            f"tip={positions[3]:.6f},{positions[4]:.6f},{positions[5]:.6f}/r"
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("右手联调执行超时")
        time.sleep(POSITION_POLL_INTERVAL_S)


def _move_arm_to_joint(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    target_joint_deg: list[float],
    speed: float,
) -> None:
    """按关节角移动机械臂。"""

    target_joint = xCoreSDK_python.JointPosition(_deg_to_rad(target_joint_deg))
    cmd_id = xCoreSDK_python.PyString()
    robot.moveReset(ec)
    _print_sdk_result("moveReset(interlock)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("联调 moveReset 失败")
    robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, speed, DEFAULT_JOINT_ZONE)], cmd_id, ec)
    _print_sdk_result("moveAppend(MoveAbsJCommand)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("联调 moveAppend 失败")
    robot.moveStart(ec)
    _print_sdk_result("moveStart(interlock)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("联调 moveStart 失败")
    _wait_until_idle(robot, ec, "联调机械臂运动")


def _run_interlock_sequence(connected_arm: ConnectedArm, hand_clients: PersistentHandClients) -> None:
    """硬编码联调：左手夹爪、右手手部。"""

    print("进入手臂联调。按回车执行当前目标，输入 s 调整速度，输入 e 急停。")
    if not _prepare_predefined_joint_motion_loop(connected_arm.robot, connected_arm.ec):
        print("联调前置状态未准备完成")
        return
    speed = DEFAULT_JOINT_SPEED
    if connected_arm.arm_side == "left":
        client = hand_clients.gripper
        for target in INTERLOCK_LEFT_TARGETS:
            print(
                f"下一个目标: {target.target_type}-{target.target_id}, "
                f"关节角度(deg)={_format_sequence(target.arm_joint_deg)}, "
                f"xyzrpye(mm/deg)={_format_sequence(target.arm_xyzrpye or [])}, "
                f"pos={target.gripper_pos}, speed={speed:.2f}"
            )
            while True:
                key = _read_nonblocking_key()
                if key == "e":
                    connected_arm.robot.stop(connected_arm.ec)
                    _print_sdk_result("stop(interlock)", connected_arm.ec)
                    break
                if key == "s":
                    speed = _prompt_motion_speed(speed, "联调")
                    print(f"联调速度已更新: {speed:.2f}")
                    continue
                if key == "\n":
                    _move_arm_to_joint(connected_arm.robot, connected_arm.ec, target.arm_joint_deg, speed)
                    if target.gripper_pos is None:
                        raise RuntimeError("左手夹爪目标缺失")
                    if not client.set_pos(target.gripper_pos):
                        raise RuntimeError("夹爪联调下发失败")
                    _poll_gripper_until_idle(client, target.gripper_pos)
                    break
                time.sleep(0.05)
        return

    hand = hand_clients.right_hand
    for target in INTERLOCK_RIGHT_TARGETS:
        print(
            f"下一个目标: {target.target_type}-{target.target_id}, "
            f"关节角度(deg)={_format_sequence(target.arm_joint_deg)}, "
            f"xyzrpye(mm/deg)={_format_sequence(target.arm_xyzrpye or [])}, "
            f"pos={target.hand_root_tip}, speed={speed:.2f}"
        )
        while True:
            key = _read_nonblocking_key()
            if key == "e":
                connected_arm.robot.stop(connected_arm.ec)
                _print_sdk_result("stop(interlock)", connected_arm.ec)
                break
            if key == "s":
                speed = _prompt_motion_speed(speed, "联调")
                print(f"联调速度已更新: {speed:.2f}")
                continue
            if key == "\n":
                _move_arm_to_joint(connected_arm.robot, connected_arm.ec, target.arm_joint_deg, speed)
                state = hand.get_hand_state(include_tactile=False)
                if state is None or target.hand_root_tip is None:
                    raise RuntimeError("右手状态不可用")
                current_positions = [float(item["position"]) for item in state["actuators"]]
                root_values = target.hand_root_tip["root"]
                tip_values = target.hand_root_tip["tip"]
                for actuator_id in (0, 1, 2):
                    current_positions[actuator_id] = root_values[actuator_id]
                for actuator_id in (3, 4, 5):
                    current_positions[actuator_id] = tip_values[actuator_id - 3]
                if not hand.set_hand_state(current_positions):
                    raise RuntimeError("右手联调下发失败")
                _poll_right_hand_until_idle(hand, f"{target.target_type}-{target.target_id}")
                break
            time.sleep(0.05)


def _main_menu(connected_arms: dict[str, ConnectedArm], hand_clients: PersistentHandClients) -> None:
    """主菜单循环。"""

    current_arm_side = "left"
    while True:
        connected_arm = connected_arms[current_arm_side]
        robot = connected_arm.robot
        ec = connected_arm.ec
        print("")
        _print_current_arm_state(connected_arm)
        print("可选操作:")
        print("  0. 切换机械臂到另一侧 1. 电机开/关切换 2. 手动/自动模式切换 3. 开关拖动")
        print("  4. 笛卡尔空间控制 5. 关节空间控制 6. 单关节控制 7. 硬编码关节值循环移动")
        print("  8. 急停复位 9. 手臂联调 q. 退出")
        choice = input("请选择: ").strip().lower()
        if choice == "0":
            current_arm_side = "right" if current_arm_side == "left" else "left"
        elif choice == "1":
            _toggle_motor_state(robot, ec)
        elif choice == "2":
            _toggle_operate_mode(robot, ec)
        elif choice == "3":
            _toggle_drag(connected_arm, hand_clients)
        elif choice == "4":
            _cartesian_control_loop(robot, ec)
        elif choice == "5":
            _joint_control_loop(robot, ec)
        elif choice == "6":
            _single_joint_control_loop(robot, ec)
        elif choice == "7":
            _loop_predefined_joint_motion(robot, ec)
        elif choice == "8":
            _recover_estop(robot, ec)
        elif choice == "9":
            _run_interlock_sequence(connected_arm, hand_clients)
        elif choice == "q":
            return
        else:
            print("无效选择")


def main() -> int:
    """程序入口。"""

    connected_arms = _connect_arms()
    gripper_process, gripper_channel = create_wuyou_channel(GRIPPER_PORT)
    right_hand_process, right_hand_channel = create_wuyou_channel(DEFAULT_PORT)
    hand_clients = PersistentHandClients(
        gripper_process=gripper_process,
        gripper=DahuanGripperClient(gripper_channel),
        right_hand_process=right_hand_process,
        right_hand=WujiRightHandClient(right_hand_channel),
    )
    try:
        _main_menu(connected_arms, hand_clients)
        return 0
    finally:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)
        stop_ssh_process(hand_clients.gripper_process)
        stop_ssh_process(hand_clients.right_hand_process)


if __name__ == "__main__":
    raise SystemExit(main())
