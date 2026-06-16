#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"
ROBOT_IP = "192.168.1.129"
LOCAL_IP = "192.168.1.91"
MM_PER_M = 1000.0
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from xcoresdk import xCoreSDK_python  # type: ignore[import-not-found]


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

    pose = xCoreSDK_python.CartesianPosition()
    pose.trans = [float(values[0]), float(values[1]), float(values[2])]
    pose.rpy = [float(values[3]), float(values[4]), float(values[5])]
    return pose


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
def _format_sequence(values: list[float] | tuple[float, ...]) -> str:
    return ", ".join(f"{float(value):.6f}" for value in values)


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


def _print_robot_snapshot(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
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
    motion_mode = robot.motionControlMode(ec)
    joint_values = robot.jointPos(ec)
    cart_pose = robot.cartPosture(xCoreSDK_python.CoordinateType.flange, ec)

    print("当前模式:", operate_mode)
    print("当前状态:", operation_state)
    print("电机状态:", power_state)
    print("运动控制模式:", motion_mode)
    print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
    print(f"当前笛卡尔位姿(mm/deg): trans={_format_sequence(_m_to_mm(cart_pose.trans))} rpy={_format_sequence(_rad_to_deg(cart_pose.rpy))}")


def _print_cartesian_pose(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    """打印当前笛卡尔空间位姿。"""

    pose = robot.cartPosture(xCoreSDK_python.CoordinateType.flange, ec)
    print("当前笛卡尔空间位姿:")
    print(f"  trans(mm): {_format_sequence(_m_to_mm(pose.trans))}")
    print(f"  rpy(deg): {_format_sequence(_rad_to_deg(pose.rpy))}")


def _wait_until_idle(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object], prompt: str) -> None:
    """轮询等待机器人运动结束。

    Parameters
    ----------
    robot:
        已连接的 SDK 机器人对象。
    ec:
        SDK 错误码字典，由调用方复用。
    prompt:
        轮询期间显示的提示文本。
    """

    while True:
        state = robot.operationState(ec)
        if state in (xCoreSDK_python.OperationState.idle, xCoreSDK_python.OperationState.unknown):
            print("运动已结束")
            return
        print(f"{prompt}: {state}")
        time.sleep(0.2)


# endregion


# region 机器人控制
def _connect_robot(config: RobotConnectionConfig) -> xCoreSDK_python.xMateRobot:
    """创建并连接机器人对象。"""

    if config.local_ip:
        robot = xCoreSDK_python.xMateRobot(config.robot_ip, config.local_ip)
    else:
        robot = xCoreSDK_python.xMateRobot(config.robot_ip)
    return robot


def _default_connection_config() -> RobotConnectionConfig:
    """返回脚本内置的默认连接配置。"""

    return RobotConnectionConfig(robot_ip=ROBOT_IP, local_ip=LOCAL_IP)


def _prepare_robot(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    """完成基础上电前准备。"""

    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    robot.setPowerState(False, ec)


def _set_motor_state(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object], on: bool) -> None:
    robot.setPowerState(on, ec)
    print(f"电机已{'打开' if on else '关闭'}")


def _switch_mode(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
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


def _toggle_drag(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    print("可选拖动开关:")
    print("  1. 打开拖动")
    print("  2. 关闭拖动")
    choice = input("请选择: ").strip()
    if choice == "1":
        print("拖动空间:")
        print("  1. 轴空间")
        print("  2. 笛卡尔空间")
        space_choice = input("请选择拖动空间: ").strip()
        print("拖动类型:")
        print("  1. 仅平移")
        print("  2. 仅旋转")
        print("  3. 自由拖拽")
        type_choice = input("请选择拖动类型: ").strip()
        if space_choice == "1":
            space = xCoreSDK_python.DragParameterSpace.jointSpace
        elif space_choice == "2":
            space = xCoreSDK_python.DragParameterSpace.cartesianSpace
        else:
            raise ValueError("无效拖动空间")
        if type_choice == "1":
            drag_type = xCoreSDK_python.DragParameterType.translationOnly
        elif type_choice == "2":
            drag_type = xCoreSDK_python.DragParameterType.rotationOnly
        elif type_choice == "3":
            drag_type = xCoreSDK_python.DragParameterType.freely
        else:
            raise ValueError("无效拖动类型")
        robot.enableDrag(space, drag_type, ec, True)
        print("拖动已打开")
    elif choice == "2":
        robot.disableDrag(ec)
        print("拖动已关闭")
    else:
        raise ValueError("无效选择")


def _cartesian_control_loop(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    """笛卡尔空间控制循环。"""

    while True:
        _print_cartesian_pose(robot, ec)
        print("输入新的 xyzrpy，单位分别为 mm 和 deg")
        print("输入 q 返回主菜单")
        raw_text = input("目标 xyzrpy: ").strip()
        if raw_text.lower() == "q":
            return
        target_values = _parse_float_list(raw_text, expected_len=6)
        target_pose = _make_cartesian_position(_mm_to_m(target_values[:3]) + _deg_to_rad(target_values[3:]))
        cmd_id = xCoreSDK_python.PyString()
        robot.moveReset(ec)
        robot.moveAppend([xCoreSDK_python.MoveLCommand(target_pose, 1000, 10)], cmd_id, ec)
        robot.moveStart(ec)
        print(f"已下发笛卡尔运动，cmd_id={cmd_id.content()}")
        _wait_until_idle(robot, ec, "等待笛卡尔运动")


def _joint_control_loop(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    """关节空间控制循环。"""

    while True:
        joint_values = robot.jointPos(ec)
        print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
        print("输入新的关节值，单位 deg，支持空格、英文逗号或中文逗号分隔")
        print("输入 q 返回主菜单")
        raw_text = input("目标关节值: ").strip()
        if raw_text.lower() == "q":
            return
        target_values = _parse_float_list(raw_text, expected_len=len(joint_values))
        target_joint = _make_joint_position(_deg_to_rad(target_values))
        cmd_id = xCoreSDK_python.PyString()
        robot.moveReset(ec)
        robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, 1000, 10)], cmd_id, ec)
        robot.moveStart(ec)
        print(f"已下发关节运动，cmd_id={cmd_id.content()}")
        _wait_until_idle(robot, ec, "等待关节运动")


def _single_joint_control_loop(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    """单关节控制循环。"""

    joint_values = robot.jointPos(ec)
    print(f"当前关节值(deg): {_format_sequence(_rad_to_deg(joint_values))}")
    axis_text = input(f"请选择轴编号 1-{len(joint_values)}: ").strip()
    axis_index = int(axis_text)
    if not 1 <= axis_index <= len(joint_values):
        raise ValueError("轴编号超出范围")

    while True:
        print(f"当前所选轴 J{axis_index} 值(deg): {math.degrees(joint_values[axis_index - 1]):.6f}")
        print("输入当前轴的目标值，单位 deg")
        print("输入 q 返回主菜单")
        raw_text = input("目标轴值: ").strip()
        if raw_text.lower() == "q":
            return
        target_value = math.radians(float(raw_text))
        target_joint_values = list(joint_values)
        target_joint_values[axis_index - 1] = target_value
        target_joint = _make_joint_position(target_joint_values)
        cmd_id = xCoreSDK_python.PyString()
        robot.moveReset(ec)
        robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, 1000, 10)], cmd_id, ec)
        robot.moveStart(ec)
        print(f"已下发单关节运动，cmd_id={cmd_id.content()}")
        _wait_until_idle(robot, ec, "等待单关节运动")
        joint_values = robot.jointPos(ec)


def _loop_predefined_joint_motion(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    """按硬编码关节值循环移动，直到用户中断。"""

    waypoints = (
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [11.4591559, -11.4591559, 17.1887339, -22.9183118, 28.6478898, -34.3774677],
        [-5.72957795, 17.1887339, -11.4591559, 22.9183118, -28.6478898, 34.3774677],
    )
    print("开始循环移动。按 Ctrl+C 中断并退出。")
    try:
        while True:
            for target_values in waypoints:
                print(f"移动到关节值(deg): {_format_sequence(target_values)}")
                target_joint = _make_joint_position(_deg_to_rad(list(target_values)))
                cmd_id = xCoreSDK_python.PyString()
                robot.moveReset(ec)
                robot.moveAppend([xCoreSDK_python.MoveAbsJCommand(target_joint, 1000, 10)], cmd_id, ec)
                robot.moveStart(ec)
                print(f"已下发循环关节运动，cmd_id={cmd_id.content()}")
                _wait_until_idle(robot, ec, "等待循环关节运动")
    except KeyboardInterrupt:
        print("用户中断，停止循环移动")


# endregion


# region 主菜单
def _main_menu(robot: xCoreSDK_python.xMateRobot, ec: dict[str, object]) -> None:
    """主菜单循环。"""

    while True:
        print("\n可选操作:")
        print("  1. 打开电机")
        print("  2. 关闭电机")
        print("  3. 切换模式")
        print("  4. 开关拖动")
        print("  5. 笛卡尔空间控制")
        print("  6. 关节空间控制")
        print("  7. 单关节控制")
        print("  8. 硬编码关节值循环移动")
        print("  q. 退出")
        choice = input("请选择: ").strip().lower()
        if choice == "1":
            _set_motor_state(robot, ec, True)
        elif choice == "2":
            _set_motor_state(robot, ec, False)
        elif choice == "3":
            _switch_mode(robot, ec)
        elif choice == "4":
            _toggle_drag(robot, ec)
        elif choice == "5":
            _cartesian_control_loop(robot, ec)
        elif choice == "6":
            _joint_control_loop(robot, ec)
        elif choice == "7":
            _single_joint_control_loop(robot, ec)
        elif choice == "8":
            _loop_predefined_joint_motion(robot, ec)
        elif choice == "q":
            return
        else:
            print("无效选择")


def main() -> int:
    """程序入口。"""

    config = _default_connection_config()
    ec: dict[str, object] = {}
    robot = _connect_robot(config)
    try:
        _prepare_robot(robot, ec)
        _print_robot_snapshot(robot, ec)
        _main_menu(robot, ec)
        return 0
    finally:
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


if __name__ == "__main__":
    raise SystemExit(main())
