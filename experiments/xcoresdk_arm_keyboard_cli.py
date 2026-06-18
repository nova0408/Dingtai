#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from pynput import keyboard  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.text import Text  # noqa: E402

# region 顶部默认常量
# 本机网卡 IP 地址；若现场不需要双网卡显式指定，可改为 `None`。
DEFAULT_LOCAL_IP: str | None = "192.168.1.116"
EXPECTED_ARM_TYPES = {
    "left": "AR5-5_0.8L-W4C1C9-ZY2",
    "right": "AR5-5_0.8R-W4C1C9-ZY2",
}
# 面板刷新周期，单位 s。
DEFAULT_REFRESH_INTERVAL_S = 0.05
# 平移 jog API rate，范围 0.01-1.00。
DEFAULT_TRANSLATION_RATE = 0.08
# 旋转 jog API rate，范围 0.01-1.00。
DEFAULT_ROTATION_RATE = 0.08
# 平移 jog 步长，单位 mm；主要作为持续 jog 的单次上限。
DEFAULT_TRANSLATION_STEP_MM = 1000.0
# 旋转 jog 步长，单位 deg；主要作为持续 jog 的单次上限。
DEFAULT_ROTATION_STEP_DEG = 360.0
# 轻点按键时的最短动作时长，单位 s。
DEFAULT_TAP_MIN_DURATION_S = 0.06
# 启动时默认速度，单位 mm/s。
DEFAULT_CARTESIAN_SPEED = 50.0
# 启动时默认转弯区，单位 mm。
DEFAULT_CARTESIAN_ZONE = 1.0
# 启动上电确认超时时间，单位 s。
DEFAULT_POWER_ON_TIMEOUT_S = 3.0
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class RobotConnectionConfig:
    """机器人连接配置。"""

    robot_ip: str
    "机器人控制器 IP 地址。"

    local_ip: str | None
    "本机网卡 IP 地址；若不需要显式指定则为 `None`。"


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


@dataclass(frozen=True, slots=True)
class JogBinding:
    """单个按键对应的 Jog 配置。"""

    key: str
    "触发按键。"

    axis_name: str
    "用于显示的人类可读轴名称。"

    index: int
    "SDK Jog 轴序号。"

    direction: bool
    "SDK Jog 方向；`True` 为正向，`False` 为负向。"

    step: float
    "Jog 步长；平移单位 mm，旋转单位 deg。"

    rate: float
    "Jog 速率，范围 0.01-1.00。"


@dataclass(slots=True)
class ActiveJogState:
    """当前正在执行的 Jog 状态。"""

    binding: JogBinding
    "当前生效的按键绑定。"

    pressed_at: float
    "按下时间戳，单位 s。"

    pending_stop_at: float | None
    "轻点补偿停止时间；长按或正常松开后通常为 `None`。"


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    """用于控制台展示的机器人快照。"""

    operation_state: object
    "当前操作状态。"

    operate_mode: object
    "当前操作模式。"

    power_state: object
    "当前电机状态。"

    xyz_mm: tuple[float, float, float]
    "当前 xyz，单位 mm。"

    rpy_deg: tuple[float, float, float]
    "当前 rpy，单位 deg。"

    joint_deg: tuple[float, ...]
    "当前关节角，单位 deg。"

    base_frame_mm_deg: tuple[float, ...]
    "当前基坐标系，单位 mm/deg。"

    tool_end_mm_deg: tuple[float, ...]
    "当前工具坐标系 end，单位 mm/deg。"

    wobj_ref_mm_deg: tuple[float, ...]
    "当前工件参考坐标系 ref，单位 mm/deg。"


@dataclass(frozen=True, slots=True)
class KeyEvent:
    """键盘事件。"""

    event_type: str
    "事件类型，仅允许 `press` / `release` / `exit`。"

    key: str | None
    "按键字符；退出事件时允许为 `None`。"


@dataclass(slots=True)
class ActiveArmState:
    """当前键盘控制所选中的机械臂状态。"""

    arm_side: str
    "当前活动机械臂侧别。"

    last_notice: str
    "当前需要在界面上持续显示的最近提示。"

    jog_mode: str
    "当前 Jog 控制空间，取值为 `cartesian` 或 `joint`。"

    joint_rate: float
    "当前关节 Jog API rate，范围 0.01-1.00。"

    xyz_rate: float
    "当前 xyz Jog API rate，范围 0.01-1.00。"

    rpy_rate: float
    "当前 rpy Jog API rate，范围 0.01-1.00。"


KEY_BINDINGS_CARTESIAN: dict[str, JogBinding] = {
    "w": JogBinding("w", "X+", 0, True, DEFAULT_TRANSLATION_STEP_MM, DEFAULT_TRANSLATION_RATE),
    "s": JogBinding("s", "X-", 0, False, DEFAULT_TRANSLATION_STEP_MM, DEFAULT_TRANSLATION_RATE),
    "a": JogBinding("a", "Y+", 1, True, DEFAULT_TRANSLATION_STEP_MM, DEFAULT_TRANSLATION_RATE),
    "d": JogBinding("d", "Y-", 1, False, DEFAULT_TRANSLATION_STEP_MM, DEFAULT_TRANSLATION_RATE),
    "r": JogBinding("r", "Z+", 2, True, DEFAULT_TRANSLATION_STEP_MM, DEFAULT_TRANSLATION_RATE),
    "f": JogBinding("f", "Z-", 2, False, DEFAULT_TRANSLATION_STEP_MM, DEFAULT_TRANSLATION_RATE),
    "j": JogBinding("j", "Rx+", 3, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "l": JogBinding("l", "Rx-", 3, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "i": JogBinding("i", "Ry+", 4, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "k": JogBinding("k", "Ry-", 4, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "y": JogBinding("y", "Rz+", 5, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "h": JogBinding("h", "Rz-", 5, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
}

KEY_BINDINGS_JOINT: dict[str, JogBinding] = {
    "a": JogBinding("a", "J1+", 0, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "z": JogBinding("z", "J1-", 0, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "s": JogBinding("s", "J2+", 1, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "x": JogBinding("x", "J2-", 1, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "d": JogBinding("d", "J3+", 2, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "c": JogBinding("c", "J3-", 2, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "f": JogBinding("f", "J4+", 3, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "v": JogBinding("v", "J4-", 3, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "g": JogBinding("g", "J5+", 4, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "b": JogBinding("b", "J5-", 4, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "h": JogBinding("h", "J6+", 5, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "n": JogBinding("n", "J6-", 5, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "j": JogBinding("j", "J7+", 6, True, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
    "m": JogBinding("m", "J7-", 6, False, DEFAULT_ROTATION_STEP_DEG, DEFAULT_ROTATION_RATE),
}

CONSOLE = Console()
# endregion


# region 基础工具
def _default_connection_configs() -> list[RobotConnectionConfig]:
    """返回脚本内置的候选连接参数。"""

    return [
        RobotConnectionConfig(robot_ip="192.168.1.161", local_ip=DEFAULT_LOCAL_IP),
        RobotConnectionConfig(robot_ip="192.168.1.160", local_ip=DEFAULT_LOCAL_IP),
    ]


def _connect_robot(config: RobotConnectionConfig) -> xCoreSDK_python.xMateErProRobot:
    """创建并连接机械臂对象。"""

    if config.local_ip:
        return xCoreSDK_python.xMateErProRobot(config.robot_ip, config.local_ip)
    return xCoreSDK_python.xMateErProRobot(config.robot_ip)


def _detect_arm_side(robot_type: str) -> str:
    """根据控制器上报的机型名称判断左右臂。"""

    for arm_side, expected_robot_type in EXPECTED_ARM_TYPES.items():
        if robot_type == expected_robot_type:
            return arm_side
    raise ValueError(f"未识别的机器人型号: {robot_type}")


def _print_sdk_result(action: str, ec: dict[str, object]) -> None:
    """打印 SDK 错误码，便于现场快速定位问题。"""

    message_text = f"{action}: ec={ec.get('ec', 0)} message={ec.get('message', '')}"
    logger.info(message_text)
    CONSOLE.print(message_text)


def _describe_power_state(power_state: object) -> str:
    """把 SDK 电机状态转为更直观的文本。"""

    if power_state == xCoreSDK_python.PowerState.on:
        return "上电"
    if power_state == xCoreSDK_python.PowerState.off:
        return "下电"
    if power_state == xCoreSDK_python.PowerState.estop:
        return "急停按下"
    if power_state == xCoreSDK_python.PowerState.gstop:
        return "安全门打开"
    return "未知"


def _rad_to_deg(values_rad: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """把弧度序列转换为角度序列。"""

    return tuple(math.degrees(float(value)) for value in values_rad)


def _m_to_mm(values_m: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """把米序列转换为毫米序列。"""

    return tuple(float(value) * 1000.0 for value in values_m)


def _format_values(values: tuple[float, ...], decimals: int = 2) -> str:
    """把数值序列格式化为控制台字符串。"""

    return ", ".join(f"{value:.{decimals}f}" for value in values)


def _to_xyz_tuple(values: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    """把前三个数值显式整理为三元组。"""

    return float(values[0]), float(values[1]), float(values[2])


def _print_banner() -> None:
    """输出脚本说明和按键映射。"""

    logger.info("启动机械臂交互式 Jog CLI")
    logger.info("按键说明：W/S=X+/X-，A/D=Y+/Y-，R/F=Z+/Z-")
    logger.info("按键说明：J/L=Rx+/Rx-，I/K=Ry+/Ry-，Y/H=Rz+/Rz-")
    logger.info("按键说明：O=切换当前机械臂，P=切换笛卡尔/关节 Jog，Q=退出")
    logger.info("当前版本使用非实时单轴 jog，不支持组合键。")
    logger.warning("本脚本依赖真实机械臂与现场安全条件；本次只做代码与静态检查，未连接硬件实测。")


def _build_snapshot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> RobotSnapshot:
    """读取一帧机器人状态并转为展示对象。"""

    operation_state = robot.operationState(ec)
    operate_mode = robot.operateMode(ec)
    power_state = robot.powerState(ec)
    cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    joint_values = robot.jointPos(ec)
    base_frame = tuple(robot.baseFrame(ec))
    toolset = robot.toolset(ec)
    xyz_mm = _to_xyz_tuple(_m_to_mm(tuple(cart_pose.trans)))
    rpy_deg = _to_xyz_tuple(_rad_to_deg(tuple(cart_pose.rpy)))
    return RobotSnapshot(
        operation_state=operation_state,
        operate_mode=operate_mode,
        power_state=power_state,
        xyz_mm=xyz_mm,
        rpy_deg=rpy_deg,
        joint_deg=_rad_to_deg(tuple(joint_values)),
        base_frame_mm_deg=(
            *_m_to_mm(base_frame[:3]),
            *_rad_to_deg(base_frame[3:]),
        ),
        tool_end_mm_deg=(
            *_m_to_mm(tuple(toolset.end.trans)),
            *_rad_to_deg(tuple(toolset.end.rpy)),
        ),
        wobj_ref_mm_deg=(
            *_m_to_mm(tuple(toolset.ref.trans)),
            *_rad_to_deg(tuple(toolset.ref.rpy)),
        ),
    )


def _log_jog_failure_diagnostics(
    connected_arm: ConnectedArm,
    binding: JogBinding,
    jog_mode: str,
    failure_code: object,
    failure_message: str,
) -> str:
    """打印 Jog 失败时的关键上下文，便于现场判断是否与奇异点相关。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    operation_state = robot.operationState(ec)
    operate_mode = robot.operateMode(ec)
    power_state = robot.powerState(ec)
    joint_values = tuple(robot.jointPos(ec))
    cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    failure_summary = (
        f"Jog失败 arm={connected_arm.arm_side} mode={jog_mode} axis={binding.axis_name} "
        f"ec={failure_code} message={failure_message}"
    )
    logger.warning(
        "Jog 失败 arm={} ip={} axis={} ec={} message={}",
        connected_arm.arm_side,
        connected_arm.config.robot_ip,
        binding.axis_name,
        failure_code,
        failure_message,
    )
    logger.warning(
        "Jog 失败上下文 mode={} state={} power={}",
        operate_mode,
        operation_state,
        _describe_power_state(power_state),
    )
    logger.warning("Jog 失败关节值(deg)=[{}]", _format_values(_rad_to_deg(joint_values)))
    logger.warning(
        "Jog 失败笛卡尔位姿 xyz(mm)=[{}] rpy(deg)=[{}]",
        _format_values(_m_to_mm(tuple(cart_pose.trans))),
        _format_values(_rad_to_deg(tuple(cart_pose.rpy))),
    )
    CONSOLE.print(failure_summary)
    CONSOLE.print(
        "失败上下文: "
        f"mode={operate_mode} state={operation_state} "
        f"power={_describe_power_state(power_state)}"
    )
    CONSOLE.print(f"失败关节值(deg): [{_format_values(_rad_to_deg(joint_values))}]")
    CONSOLE.print(
        "失败笛卡尔位姿: "
        f"xyz(mm)=[{_format_values(_m_to_mm(tuple(cart_pose.trans)))}] "
        f"rpy(deg)=[{_format_values(_rad_to_deg(tuple(cart_pose.rpy)))}]"
    )
    return failure_summary


def _render_status(
    connected_arm: ConnectedArm,
    snapshot: RobotSnapshot,
    active_jog: ActiveJogState | None,
    last_notice: str,
    jog_mode: str,
) -> Panel:
    """构造固定三行的 `rich` 状态面板。"""

    active_key = "None" if active_jog is None else active_jog.binding.key.upper()
    active_axis = "None" if active_jog is None else active_jog.binding.axis_name
    power_style = "green" if snapshot.power_state == xCoreSDK_python.PowerState.on else "yellow"
    mode_style = "green" if snapshot.operate_mode == xCoreSDK_python.OperateMode.manual else "yellow"
    jog_hint = (
        "cart: WS AD RF JL IK YH"
        if jog_mode == "cartesian"
        else "joint: AZ SX DC FV GB HN JM"
    )
    line_extra = Text()
    line_extra.append("INFO ", style="bold cyan")
    line_extra.append(f"Q exit | O switch arm | P toggle cart/joint jog | ] set rate | {jog_hint}", style="white")
    line_extra.append(" | arm=", style="dim")
    line_extra.append(connected_arm.arm_side.upper(), style="bright_green")
    line_extra.append(" | ip=", style="dim")
    line_extra.append(connected_arm.config.robot_ip, style="cyan")
    line_extra.append(" | jog=", style="dim")
    line_extra.append(jog_mode, style="bright_yellow")
    line_extra.append(" | held=", style="dim")
    line_extra.append(active_key, style="magenta" if active_jog is not None else "cyan")
    line_extra.append(" | axis=", style="dim")
    line_extra.append(active_axis, style="magenta" if active_jog is not None else "cyan")
    line_extra.append(" | mode=", style="dim")
    line_extra.append(str(snapshot.operate_mode), style=mode_style)
    line_extra.append(" | state=", style="dim")
    line_extra.append(str(snapshot.operation_state), style="white")
    line_extra.append(" | power=", style="dim")
    line_extra.append(_describe_power_state(snapshot.power_state), style=power_style)

    line_cartesian = Text()
    line_cartesian.append("CART ", style="bold cyan")
    line_cartesian.append("xyz(mm)=", style="dim")
    line_cartesian.append(f"[{_format_values(snapshot.xyz_mm)}]", style="bright_cyan")
    line_cartesian.append(" | rpy(deg)=", style="dim")
    line_cartesian.append(f"[{_format_values(snapshot.rpy_deg)}]", style="bright_cyan")

    line_joint = Text()
    line_joint.append("JOINT ", style="bold cyan")
    line_joint.append("deg=", style="dim")
    line_joint.append(f"[{_format_values(snapshot.joint_deg)}]", style="bright_cyan")

    line_frame = Text()
    line_frame.append("FRAME ", style="bold cyan")
    line_frame.append("base(mm/deg)=", style="dim")
    line_frame.append(f"[{_format_values(snapshot.base_frame_mm_deg)}]", style="bright_cyan")
    line_frame.append(" | tool end(mm/deg)=", style="dim")
    line_frame.append(f"[{_format_values(snapshot.tool_end_mm_deg)}]", style="bright_cyan")
    line_frame.append(" | wobj ref(mm/deg)=", style="dim")
    line_frame.append(f"[{_format_values(snapshot.wobj_ref_mm_deg)}]", style="bright_cyan")

    line_notice = Text()
    line_notice.append("NOTICE ", style="bold yellow")
    line_notice.append(last_notice, style="yellow" if last_notice else "dim")

    body = Group(line_extra, line_cartesian, line_joint, line_frame, line_notice)
    return Panel(body, title=f"xCoreSDK Arm CLI [{connected_arm.robot_type}]", border_style="cyan")


def _wait_for_power_on(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    timeout_s: float = DEFAULT_POWER_ON_TIMEOUT_S,
) -> bool:
    """轮询确认机械臂已经真正进入上电状态。"""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current_power_state = robot.powerState(ec)
        if current_power_state == xCoreSDK_python.PowerState.on:
            return True
        time.sleep(0.1)
    return False


def _ensure_nrt_jog_ready(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """启动时强制切到可 jog 的非实时状态。"""

    robot.stop(ec)
    _print_sdk_result("stop", ec)
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _print_sdk_result("setMotionControlMode(NrtCommandMode)", ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _print_sdk_result("setOperateMode(manual)", ec)
    robot.setPowerState(True, ec)
    _print_sdk_result("setPowerState(True)", ec)
    if not _wait_for_power_on(robot, ec):
        robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
        _print_sdk_result("setOperateMode(manual, retry)", ec)
        robot.setPowerState(True, ec)
        _print_sdk_result("setPowerState(True, retry)", ec)
    robot.setDefaultConfOpt(False, ec)
    _print_sdk_result("setDefaultConfOpt(False)", ec)
    robot.setDefaultSpeed(DEFAULT_CARTESIAN_SPEED, ec)
    _print_sdk_result(f"setDefaultSpeed({DEFAULT_CARTESIAN_SPEED:.2f})", ec)
    robot.setDefaultZone(DEFAULT_CARTESIAN_ZONE, ec)
    _print_sdk_result(f"setDefaultZone({DEFAULT_CARTESIAN_ZONE:.2f})", ec)

    current_power_state = robot.powerState(ec)
    current_operate_mode = robot.operateMode(ec)
    _print_sdk_result(
        f"startup state mode={current_operate_mode} power={_describe_power_state(current_power_state)}",
        ec,
    )
    if current_power_state != xCoreSDK_python.PowerState.on:
        raise RuntimeError("jog 准备后未成功进入上电状态，请检查现场使能、急停和安全门。")


def _shutdown_robot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """安全停止并断开单台机械臂连接。"""

    try:
        robot.stop(ec)
    except Exception as exc:
        logger.warning("清理 stop 失败：{}", exc)
    try:
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    except Exception as exc:
        logger.warning("清理 setMotionControlMode(NrtCommandMode) 失败：{}", exc)
    try:
        robot.setPowerState(False, ec)
    except Exception as exc:
        logger.warning("清理 setPowerState(False) 失败：{}", exc)
    try:
        robot.disconnectFromRobot(ec)
    except Exception as exc:
        logger.warning("清理 disconnectFromRobot 失败：{}", exc)


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
            _ensure_nrt_jog_ready(robot, ec)
            connected_arms[arm_side] = connected_arm
            logger.info(
                "已连接 {} arm: ip={} type={} uid={}",
                arm_side,
                config.robot_ip,
                robot_info.type,
                robot_info.id,
            )
        missing_arm_sides = [arm_side for arm_side in EXPECTED_ARM_TYPES if arm_side not in connected_arms]
        if missing_arm_sides:
            raise RuntimeError(f"缺少目标机械臂连接: {', '.join(missing_arm_sides)}")
        return connected_arms
    except Exception:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)
        raise


def _resolve_jog_rate(binding: JogBinding, jog_mode: str, active_arm_state: ActiveArmState) -> float:
    """根据当前控制空间与轴类型选择实际 Jog 速率。"""

    if jog_mode == "joint":
        return active_arm_state.joint_rate
    if binding.axis_name.startswith("R"):
        return active_arm_state.rpy_rate
    return active_arm_state.xyz_rate


def _parse_rate_input(prompt: str, current_value: float) -> float | None:
    """读取单个 Jog API rate，支持直接回车保留当前值。"""

    raw_text = CONSOLE.input(f"{prompt} [{current_value:.2f}]: ").strip()
    if raw_text == "":
        return current_value
    try:
        parsed = float(raw_text)
    except ValueError:
        return None
    if not 0.01 <= parsed <= 1.00:
        return None
    return parsed


def _configure_jog_rates(active_arm_state: ActiveArmState) -> None:
    """交互式设置关节、xyz 与 rpy 三类 Jog API rate。"""

    CONSOLE.print("设置 startJog 的实际 rate 参数，范围 0.01-1.00，直接回车保留当前值")
    joint_rate = _parse_rate_input("joint api rate", active_arm_state.joint_rate)
    xyz_rate = _parse_rate_input("xyz api rate", active_arm_state.xyz_rate)
    rpy_rate = _parse_rate_input("rpy api rate", active_arm_state.rpy_rate)
    if joint_rate is None or xyz_rate is None or rpy_rate is None:
        active_arm_state.last_notice = "rate 设置失败：请输入 0.01-1.00 范围内数字"
        CONSOLE.print(active_arm_state.last_notice)
        return
    active_arm_state.joint_rate = joint_rate
    active_arm_state.xyz_rate = xyz_rate
    active_arm_state.rpy_rate = rpy_rate
    active_arm_state.last_notice = (
        f"api rate 已更新 joint={joint_rate:.2f} xyz={xyz_rate:.2f} rpy={rpy_rate:.2f}"
    )
    CONSOLE.print(active_arm_state.last_notice)


def _start_jog(
    connected_arm: ConnectedArm,
    binding: JogBinding,
    jog_mode: str,
    active_arm_state: ActiveArmState,
) -> tuple[bool, str]:
    """按绑定配置启动一次 Jog。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    jog_space = (
        xCoreSDK_python.JogOptSpace.jointSpace
        if jog_mode == "joint"
        else xCoreSDK_python.JogOptSpace.baseFrame
    )
    jog_rate = _resolve_jog_rate(binding, jog_mode, active_arm_state)
    robot.startJog(jog_space, jog_rate, binding.step, binding.index, binding.direction, ec)
    if ec.get("ec", 0) != 0:
        failure_code = ec.get("ec", 0)
        failure_message = str(ec.get("message", ""))
        _print_sdk_result(f"startJog({binding.axis_name}, {jog_mode}, rate={jog_rate:.2f})", ec)
        failure_summary = _log_jog_failure_diagnostics(
            connected_arm,
            binding,
            jog_mode,
            failure_code,
            failure_message,
        )
        if jog_mode == "cartesian":
            return False, f"{failure_summary} | 可按 P 切换到 joint 模式后重试"
        return False, failure_summary
    return True, f"Jog启动成功 axis={binding.axis_name} mode={jog_mode} rate={jog_rate:.2f}"


def _stop_jog(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object], active_jog: ActiveJogState | None) -> None:
    """停止当前 Jog。"""

    if active_jog is None:
        return
    robot.stop(ec)
    if ec.get("ec", 0) != 0:
        _print_sdk_result(f"stop({active_jog.binding.axis_name})", ec)


def _handle_key_event(
    connected_arm: ConnectedArm,
    event: KeyEvent,
    active_jog: ActiveJogState | None,
    active_arm_state: ActiveArmState,
) -> tuple[ActiveJogState | None, bool]:
    """处理单个按键事件。"""

    robot = connected_arm.robot
    ec = connected_arm.ec
    if event.event_type == "exit":
        return active_jog, True

    key = event.key
    if key is None:
        return active_jog, False

    if key == "o":
        return active_jog, False
    if key == "p":
        active_arm_state.jog_mode = "joint" if active_arm_state.jog_mode == "cartesian" else "cartesian"
        active_arm_state.last_notice = f"Jog模式已切换为 {active_arm_state.jog_mode}"
        return active_jog, False
    if key == "]":
        return active_jog, False

    key_bindings = KEY_BINDINGS_JOINT if active_arm_state.jog_mode == "joint" else KEY_BINDINGS_CARTESIAN
    binding = key_bindings.get(key)
    if binding is None:
        return active_jog, False

    now = time.monotonic()
    if event.event_type == "press":
        if active_jog is not None and active_jog.binding.key == key:
            return active_jog, False
        if active_jog is not None:
            _stop_jog(robot, ec, active_jog)
            active_jog = None
        jog_started, notice = _start_jog(connected_arm, binding, active_arm_state.jog_mode, active_arm_state)
        active_arm_state.last_notice = notice
        if not jog_started:
            return None, False
        return ActiveJogState(binding=binding, pressed_at=now, pending_stop_at=None), False

    if event.event_type == "release":
        if active_jog is None or active_jog.binding.key != key:
            return active_jog, False
        if now - active_jog.pressed_at >= DEFAULT_TAP_MIN_DURATION_S:
            _stop_jog(robot, ec, active_jog)
            active_arm_state.last_notice = "Jog已停止"
            return None, False
        active_jog.pending_stop_at = active_jog.pressed_at + DEFAULT_TAP_MIN_DURATION_S
        return active_jog, False

    return active_jog, False


def _normalize_listener_key(key: keyboard.Key | keyboard.KeyCode | None) -> str | None:
    """把 `pynput` 按键对象转为小写字符。"""

    if key is None:
        return None
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        return key.char.lower()
    return None


def _start_keyboard_listener(event_queue: SimpleQueue[KeyEvent]) -> keyboard.Listener:
    """启动 `pynput` 键盘监听。"""

    pressed_keys: set[str] = set()
    pressed_lock = threading.Lock()

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        normalized = _normalize_listener_key(key)
        if normalized is None:
            return
        if normalized == "q":
            event_queue.put(KeyEvent(event_type="exit", key="q"))
            return
        if normalized == "o":
            event_queue.put(KeyEvent(event_type="press", key="o"))
            return
        if normalized == "p":
            event_queue.put(KeyEvent(event_type="press", key="p"))
            return
        if normalized == "]":
            event_queue.put(KeyEvent(event_type="press", key="]"))
            return
        if normalized not in KEY_BINDINGS_CARTESIAN and normalized not in KEY_BINDINGS_JOINT:
            return
        with pressed_lock:
            if normalized in pressed_keys:
                return
            pressed_keys.add(normalized)
        event_queue.put(KeyEvent(event_type="press", key=normalized))

    def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        normalized = _normalize_listener_key(key)
        if normalized is None or normalized in ("o", "p", "]"):
            return
        if normalized not in KEY_BINDINGS_CARTESIAN and normalized not in KEY_BINDINGS_JOINT:
            return
        with pressed_lock:
            if normalized not in pressed_keys:
                return
            pressed_keys.remove(normalized)
        event_queue.put(KeyEvent(event_type="release", key=normalized))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def _switch_active_arm(connected_arms: dict[str, ConnectedArm], active_arm_state: ActiveArmState) -> None:
    """切换当前活动机械臂。"""

    active_arm_state.arm_side = "right" if active_arm_state.arm_side == "left" else "left"
    connected_arm = connected_arms[active_arm_state.arm_side]
    logger.info(
        "已切换到 {} arm: ip={} type={} uid={}",
        connected_arm.arm_side,
        connected_arm.config.robot_ip,
        connected_arm.robot_type,
        connected_arm.robot_uid,
    )
    active_arm_state.last_notice = (
        f"已切换到 {connected_arm.arm_side} arm "
        f"ip={connected_arm.config.robot_ip}"
    )


def _run_keyboard_loop(connected_arms: dict[str, ConnectedArm]) -> None:
    """运行交互式键盘控制循环。"""

    event_queue: SimpleQueue[KeyEvent] = SimpleQueue()
    listener = _start_keyboard_listener(event_queue)
    active_jog: ActiveJogState | None = None
    active_arm_state = ActiveArmState(
        arm_side="left",
        last_notice="等待按键输入",
        jog_mode="cartesian",
        joint_rate=DEFAULT_ROTATION_RATE,
        xyz_rate=DEFAULT_TRANSLATION_RATE,
        rpy_rate=DEFAULT_ROTATION_RATE,
    )

    try:
        with Live(console=CONSOLE, refresh_per_second=20, transient=False) as live:
            while True:
                connected_arm = connected_arms[active_arm_state.arm_side]
                robot = connected_arm.robot
                ec = connected_arm.ec
                while not event_queue.empty():
                    event = event_queue.get()
                    if event.key == "o" and event.event_type == "press":
                        _stop_jog(robot, ec, active_jog)
                        active_jog = None
                        _switch_active_arm(connected_arms, active_arm_state)
                        connected_arm = connected_arms[active_arm_state.arm_side]
                        robot = connected_arm.robot
                        ec = connected_arm.ec
                        continue
                    if event.key == "]" and event.event_type == "press":
                        _stop_jog(robot, ec, active_jog)
                        active_jog = None
                        live.stop()
                        _configure_jog_rates(active_arm_state)
                        live.start()
                        continue
                    active_jog, should_exit = _handle_key_event(connected_arm, event, active_jog, active_arm_state)
                    if should_exit:
                        _stop_jog(robot, ec, active_jog)
                        logger.success("收到退出指令，CLI 已结束。")
                        return

                if (
                    active_jog is not None
                    and active_jog.pending_stop_at is not None
                    and time.monotonic() >= active_jog.pending_stop_at
                ):
                    _stop_jog(robot, ec, active_jog)
                    active_jog = None
                    active_arm_state.last_notice = "Jog已停止"

                snapshot = _build_snapshot(robot, ec)
                live.update(
                    _render_status(
                        connected_arm,
                        snapshot,
                        active_jog,
                        active_arm_state.last_notice,
                        active_arm_state.jog_mode,
                    )
                )
                time.sleep(DEFAULT_REFRESH_INTERVAL_S)
    finally:
        listener.stop()


# endregion


# region 程序入口
def main() -> int:
    """程序入口。"""

    _print_banner()
    configs = _default_connection_configs()
    for config in configs:
        logger.info("连接目标：robot_ip {} local_ip {}", config.robot_ip, config.local_ip)
    connected_arms = _connect_arms(configs)
    try:
        _run_keyboard_loop(connected_arms)
        return 0
    finally:
        for connected_arm in connected_arms.values():
            _shutdown_robot(connected_arm.robot, connected_arm.ec)


if __name__ == "__main__":
    raise SystemExit(main())
# endregion
