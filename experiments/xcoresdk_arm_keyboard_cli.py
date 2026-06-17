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
DEFAULT_ROBOT_IP = "192.168.0.160"
# 本机网卡 IP 地址；若现场不需要双网卡显式指定，可改为 `None`。
DEFAULT_LOCAL_IP: str | None = "192.168.0.1"
# 面板刷新周期，单位 s。
DEFAULT_REFRESH_INTERVAL_S = 0.05
# 平移 jog 速率，范围 0.01-1.00。
DEFAULT_TRANSLATION_RATE = 0.08
# 旋转 jog 速率，范围 0.01-1.00。
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


@dataclass(frozen=True, slots=True)
class KeyEvent:
    """键盘事件。"""

    event_type: str
    "事件类型，仅允许 `press` / `release` / `exit`。"

    key: str | None
    "按键字符；退出事件时允许为 `None`。"


KEY_BINDINGS: dict[str, JogBinding] = {
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

CONSOLE = Console()
# endregion


# region 基础工具
def _default_connection_config() -> RobotConnectionConfig:
    """返回脚本内置的默认连接参数。"""

    return RobotConnectionConfig(robot_ip=DEFAULT_ROBOT_IP, local_ip=DEFAULT_LOCAL_IP)


def _connect_robot(config: RobotConnectionConfig) -> xCoreSDK_python.xMateErProRobot:
    """创建并连接机械臂对象。"""

    if config.local_ip:
        return xCoreSDK_python.xMateErProRobot(config.robot_ip, config.local_ip)
    return xCoreSDK_python.xMateErProRobot(config.robot_ip)


def _print_sdk_result(action: str, ec: dict[str, object]) -> None:
    """打印 SDK 错误码，便于现场快速定位问题。"""

    logger.info("{}: ec={} message={}", action, ec.get("ec", 0), ec.get("message", ""))


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
    logger.info("当前版本使用非实时单轴 jog，不支持组合键。")
    logger.warning("本脚本依赖真实机械臂与现场安全条件；本次只做代码与静态检查，未连接硬件实测。")


def _build_snapshot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> RobotSnapshot:
    """读取一帧机器人状态并转为展示对象。"""

    operation_state = robot.operationState(ec)
    operate_mode = robot.operateMode(ec)
    power_state = robot.powerState(ec)
    cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    joint_values = robot.jointPos(ec)
    xyz_mm = _to_xyz_tuple(_m_to_mm(tuple(cart_pose.trans)))
    rpy_deg = _to_xyz_tuple(_rad_to_deg(tuple(cart_pose.rpy)))
    return RobotSnapshot(
        operation_state=operation_state,
        operate_mode=operate_mode,
        power_state=power_state,
        xyz_mm=xyz_mm,
        rpy_deg=rpy_deg,
        joint_deg=_rad_to_deg(tuple(joint_values)),
    )


def _render_status(snapshot: RobotSnapshot, active_jog: ActiveJogState | None) -> Panel:
    """构造固定三行的 `rich` 状态面板。"""

    active_key = "None" if active_jog is None else active_jog.binding.key.upper()
    active_axis = "None" if active_jog is None else active_jog.binding.axis_name
    power_style = "green" if snapshot.power_state == xCoreSDK_python.PowerState.on else "yellow"
    mode_style = "green" if snapshot.operate_mode == xCoreSDK_python.OperateMode.manual else "yellow"

    line_extra = Text()
    line_extra.append("INFO ", style="bold cyan")
    line_extra.append("Q exit | move WS AD RF | rot JL IK YH", style="white")
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

    body = Group(line_extra, line_cartesian, line_joint)
    return Panel(body, title="xCoreSDK Arm CLI", border_style="cyan")


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


def _start_jog(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    binding: JogBinding,
) -> bool:
    """按绑定配置启动一次 Jog。"""

    robot.startJog(xCoreSDK_python.JogOptSpace.baseFrame, binding.rate, binding.step, binding.index, binding.direction, ec)
    if ec.get("ec", 0) != 0:
        _print_sdk_result(f"startJog({binding.axis_name})", ec)
        return False
    return True


def _stop_jog(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object], active_jog: ActiveJogState | None) -> None:
    """停止当前 Jog。"""

    if active_jog is None:
        return
    robot.stop(ec)
    if ec.get("ec", 0) != 0:
        _print_sdk_result(f"stop({active_jog.binding.axis_name})", ec)


def _handle_key_event(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    event: KeyEvent,
    active_jog: ActiveJogState | None,
) -> tuple[ActiveJogState | None, bool]:
    """处理单个按键事件。"""

    if event.event_type == "exit":
        return active_jog, True

    key = event.key
    if key is None:
        return active_jog, False

    binding = KEY_BINDINGS.get(key)
    if binding is None:
        return active_jog, False

    now = time.monotonic()
    if event.event_type == "press":
        if active_jog is not None and active_jog.binding.key == key:
            return active_jog, False
        if active_jog is not None:
            _stop_jog(robot, ec, active_jog)
            active_jog = None
        if not _start_jog(robot, ec, binding):
            return None, False
        return ActiveJogState(binding=binding, pressed_at=now, pending_stop_at=None), False

    if event.event_type == "release":
        if active_jog is None or active_jog.binding.key != key:
            return active_jog, False
        if now - active_jog.pressed_at >= DEFAULT_TAP_MIN_DURATION_S:
            _stop_jog(robot, ec, active_jog)
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
        if normalized not in KEY_BINDINGS:
            return
        with pressed_lock:
            if normalized in pressed_keys:
                return
            pressed_keys.add(normalized)
        event_queue.put(KeyEvent(event_type="press", key=normalized))

    def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        normalized = _normalize_listener_key(key)
        if normalized is None or normalized not in KEY_BINDINGS:
            return
        with pressed_lock:
            if normalized not in pressed_keys:
                return
            pressed_keys.remove(normalized)
        event_queue.put(KeyEvent(event_type="release", key=normalized))

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def _run_keyboard_loop(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    """运行交互式键盘控制循环。"""

    _ensure_nrt_jog_ready(robot, ec)
    event_queue: SimpleQueue[KeyEvent] = SimpleQueue()
    listener = _start_keyboard_listener(event_queue)
    active_jog: ActiveJogState | None = None

    try:
        with Live(console=CONSOLE, refresh_per_second=20, transient=False) as live:
            while True:
                while not event_queue.empty():
                    event = event_queue.get()
                    active_jog, should_exit = _handle_key_event(robot, ec, event, active_jog)
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

                snapshot = _build_snapshot(robot, ec)
                live.update(_render_status(snapshot, active_jog))
                time.sleep(DEFAULT_REFRESH_INTERVAL_S)
    finally:
        listener.stop()


# endregion


# region 程序入口
def main() -> int:
    """程序入口。"""

    config = _default_connection_config()
    ec: dict[str, object] = {}
    robot = _connect_robot(config)
    _print_banner()
    logger.info("连接目标：robot_ip {} local_ip {}", config.robot_ip, config.local_ip)
    try:
        _run_keyboard_loop(robot, ec)
        return 0
    finally:
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


if __name__ == "__main__":
    raise SystemExit(main())
# endregion
