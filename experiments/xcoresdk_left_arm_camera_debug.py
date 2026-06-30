#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_WUJI_ROOT = PROJECT_ROOT / "test" / "wuji"
if str(TEST_WUJI_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_WUJI_ROOT))

from sdk.xcoresdk import xCoreSDK_python  # noqa: E402
from src.wuji import SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL, WujiCameraName, WujiZmqCameraClient  # noqa: E402

WUYOU_HOST = "192.168.100.60"
WUYOU_SSH_ALIAS = "wuyou"
DEFAULT_LOCAL_IP: str | None = "192.168.1.116"
DEFAULT_LEFT_ARM_IP = "192.168.1.161"
DEFAULT_CAMERA_NAME: WujiCameraName = "left_hand_camera"
DEFAULT_CONTROL_PORT = 5570
DEFAULT_LEFT_HAND_STREAM_PORT = 5562
DEFAULT_REQUEST_TIMEOUT_MS = 3000
DEFAULT_STREAM_TIMEOUT_MS = 5000
DEFAULT_FORWARD_WAIT_S = 1.0
DEFAULT_REFRESH_INTERVAL_S = 0.03
DEFAULT_TRANSLATION_RATE = 0.08
DEFAULT_ROTATION_RATE = 0.08
DEFAULT_TRANSLATION_STEP_MM = 1000.0
DEFAULT_ROTATION_STEP_DEG = 360.0
DEFAULT_TAP_MIN_DURATION_S = 0.06
DEFAULT_POWER_ON_TIMEOUT_S = 3.0
DEFAULT_TOOL_NAME = "g_tool_0"
DEFAULT_WOBJ_NAME = "g_wobj_0"
DEFAULT_FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")
DEFAULT_FONT_SIZE = 22


@dataclass(frozen=True, slots=True)
class RobotConnectionConfig:
    robot_ip: str
    local_ip: str | None


@dataclass(slots=True)
class ConnectedArm:
    arm_side: str
    config: RobotConnectionConfig
    robot: xCoreSDK_python.xMateErProRobot
    robot_type: str
    robot_uid: str
    ec: dict[str, object]


@dataclass(frozen=True, slots=True)
class JogBinding:
    key: str
    axis_name: str
    index: int
    direction: bool
    step: float
    rate: float


@dataclass(slots=True)
class ActiveJogState:
    binding: JogBinding
    pressed_at: float
    pending_stop_at: float | None


@dataclass(frozen=True, slots=True)
class KeyEvent:
    event_type: str
    key: str | None


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    joint_deg: tuple[float, ...]
    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    power_state: object
    operate_mode: object
    operation_state: object
    tool_frame: str
    wobj_frame: str


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


def _connect_robot(config: RobotConnectionConfig) -> xCoreSDK_python.xMateErProRobot:
    if config.local_ip:
        return xCoreSDK_python.xMateErProRobot(config.robot_ip, config.local_ip)
    return xCoreSDK_python.xMateErProRobot(config.robot_ip)


def _print_sdk_result(action: str, ec: dict[str, object]) -> None:
    logger.info("{}: ec={} message={}", action, ec.get("ec", 0), ec.get("message", ""))


def _describe_power_state(power_state: object) -> str:
    if power_state == xCoreSDK_python.PowerState.on:
        return "上电"
    if power_state == xCoreSDK_python.PowerState.off:
        return "下电"
    if power_state == xCoreSDK_python.PowerState.estop:
        return "急停按下"
    if power_state == xCoreSDK_python.PowerState.gstop:
        return "安全门打开"
    return "未知"


def _m_to_mm(values_m: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(float(value) * 1000.0 for value in values_m)


def _rad_to_deg(values_rad: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(math.degrees(float(value)) for value in values_rad)


def _format_values(values: tuple[float, ...], decimals: int = 2) -> str:
    return ", ".join(f"{value:.{decimals}f}" for value in values)


def _to_xyz_tuple(values: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _apply_named_toolset(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> bool:
    robot.setToolset(DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME, ec)
    _print_sdk_result(f"setToolset({DEFAULT_TOOL_NAME}, {DEFAULT_WOBJ_NAME})", ec)
    return ec.get("ec", 0) == 0


def start_ssh_tunnel(remote_port: int, remote_host: str, ssh_alias: str) -> subprocess.Popen[bytes]:
    command = [
        "ssh",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "TCPKeepAlive=yes",
        "-N",
        "-L",
        f"127.0.0.1:{remote_port - 1}:{remote_host}:{remote_port}",
        ssh_alias,
    ]
    logger.info("启动 SSH 隧道: local=127.0.0.1:{} remote={}:{} alias={}", remote_port - 1, remote_host, remote_port, ssh_alias)
    return subprocess.Popen(command, stderr=subprocess.PIPE)


def stop_ssh_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=3.0)
    except Exception:  # noqa: BLE001
        process.kill()


def _wait_for_power_on(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> bool:
    deadline = time.monotonic() + DEFAULT_POWER_ON_TIMEOUT_S
    while time.monotonic() < deadline:
        if robot.powerState(ec) == xCoreSDK_python.PowerState.on:
            return True
        time.sleep(0.1)
    return False


def _ensure_nrt_jog_ready(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> bool:
    robot.stop(ec)
    _print_sdk_result("stop", ec)
    robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    _print_sdk_result("setMotionControlMode(NrtCommandMode)", ec)
    robot.setOperateMode(xCoreSDK_python.OperateMode.manual, ec)
    _print_sdk_result("setOperateMode(manual)", ec)
    robot.setPowerState(True, ec)
    _print_sdk_result("setPowerState(True)", ec)
    if not _wait_for_power_on(robot, ec):
        return False
    robot.setDefaultConfOpt(False, ec)
    _print_sdk_result("setDefaultConfOpt(False)", ec)
    if not _apply_named_toolset(robot, ec):
        return False
    return robot.powerState(ec) == xCoreSDK_python.PowerState.on


def _shutdown_robot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> None:
    try:
        robot.stop(ec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("清理 stop 失败: {}", exc)
    try:
        robot.setMotionControlMode(xCoreSDK_python.MotionControlMode.NrtCommandMode, ec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("清理 setMotionControlMode 失败: {}", exc)
    try:
        robot.setPowerState(False, ec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("清理 setPowerState 失败: {}", exc)
    try:
        robot.disconnectFromRobot(ec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("清理 disconnectFromRobot 失败: {}", exc)


def _connect_left_arm() -> ConnectedArm:
    config = RobotConnectionConfig(robot_ip=DEFAULT_LEFT_ARM_IP, local_ip=DEFAULT_LOCAL_IP)
    ec: dict[str, object] = {}
    robot = _connect_robot(config)
    robot_info = robot.robotInfo(ec)
    _print_sdk_result("robotInfo", ec)
    arm = ConnectedArm(
        arm_side="left",
        config=config,
        robot=robot,
        robot_type=robot_info.type,
        robot_uid=robot_info.id,
        ec=ec,
    )
    if not _ensure_nrt_jog_ready(robot, ec):
        raise RuntimeError("左臂 jog 准备失败")
    return arm


def _format_frame_values(frame: xCoreSDK_python.Frame) -> str:
    return f"trans(m)=[{_format_values(tuple(frame.trans), decimals=4)}], rpy(deg)=[{_format_values(_rad_to_deg(tuple(frame.rpy)))}]"


def _build_snapshot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> RobotSnapshot:
    joint_deg = _rad_to_deg(tuple(robot.jointPos(ec)))
    cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    toolset = robot.toolset(ec)
    return RobotSnapshot(
        joint_deg=joint_deg,
        xyz_mm=_to_xyz_tuple(_m_to_mm(tuple(cart_pose.trans))),
        rpy_deg=_to_xyz_tuple(_rad_to_deg(tuple(cart_pose.rpy))),
        power_state=robot.powerState(ec),
        operate_mode=robot.operateMode(ec),
        operation_state=robot.operationState(ec),
        tool_frame=_format_frame_values(toolset.end),
        wobj_frame=_format_frame_values(toolset.ref),
    )


def _load_font(size: int = DEFAULT_FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(DEFAULT_FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_stroked_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    stroke_fill: tuple[int, int, int],
    stroke_width: int,
) -> None:
    draw.text(position, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)


def _draw_camera_overlay(frame: Any, snapshot: RobotSnapshot, jog_mode: str) -> Any:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_title = _load_font(20)
    font_body = _load_font(19)
    title_color = (255, 255, 255)
    x0, y0 = 18, 18
    left_stroke = (255, 110, 0) if jog_mode == "cartesian" else (0, 180, 255)
    _draw_stroked_text(draw, (x0, y0), "左臂实时相机调试", font=font_title, fill=title_color, stroke_fill=left_stroke, stroke_width=2)
    _draw_stroked_text(draw, (x0, y0 + 28), "Q 退出  |  P 切换笛卡尔 / 轴坐标", font=font_body, fill=title_color, stroke_fill=(90, 90, 90), stroke_width=2)

    if jog_mode == "cartesian":
        _draw_stroked_text(draw, (x0, y0 + 62), "W A R J I Y", font=font_body, fill=title_color, stroke_fill=(255, 110, 0), stroke_width=2)
        _draw_stroked_text(draw, (x0, y0 + 89), "X+ Y+ Z+ Rx+ Ry+ Rz+", font=font_body, fill=title_color, stroke_fill=(0, 180, 255), stroke_width=2)
        _draw_stroked_text(draw, (x0, y0 + 116), "S D F L K H", font=font_body, fill=title_color, stroke_fill=(255, 110, 0), stroke_width=2)
        _draw_stroked_text(draw, (x0, y0 + 143), "X- Y- Z- Rx- Ry- Rz-", font=font_body, fill=title_color, stroke_fill=(0, 180, 255), stroke_width=2)
    else:
        _draw_stroked_text(draw, (x0, y0 + 62), "A S D F G H J", font=font_body, fill=title_color, stroke_fill=(0, 180, 255), stroke_width=2)
        _draw_stroked_text(draw, (x0, y0 + 89), "J1+ J2+ J3+ J4+ J5+ J6+ J7+", font=font_body, fill=title_color, stroke_fill=(80, 220, 255), stroke_width=2)
        _draw_stroked_text(draw, (x0, y0 + 116), "Z X C V B N M", font=font_body, fill=title_color, stroke_fill=(0, 180, 255), stroke_width=2)
        _draw_stroked_text(draw, (x0, y0 + 143), "J1- J2- J3- J4- J5- J6- J7-", font=font_body, fill=title_color, stroke_fill=(80, 220, 255), stroke_width=2)

    _, w = frame.shape[:2]
    right_x = max(18, w - 560)
    right_y = 18
    _draw_stroked_text(
        draw,
        (right_x, right_y),
        f"{snapshot.joint_deg[0]:.1f} {snapshot.joint_deg[1]:.1f} {snapshot.joint_deg[2]:.1f} {snapshot.joint_deg[3]:.1f} {snapshot.joint_deg[4]:.1f} {snapshot.joint_deg[5]:.1f} {snapshot.joint_deg[6]:.1f}",
        font=font_body,
        fill=title_color,
        stroke_fill=(255, 80, 80),
        stroke_width=2,
    )
    _draw_stroked_text(
        draw,
        (right_x, right_y + 28),
        f"{snapshot.xyz_mm[0]:.2f} {snapshot.xyz_mm[1]:.2f} {snapshot.xyz_mm[2]:.2f}",
        font=font_body,
        fill=title_color,
        stroke_fill=(80, 255, 120),
        stroke_width=2,
    )
    _draw_stroked_text(
        draw,
        (right_x, right_y + 56),
        f"{snapshot.rpy_deg[0]:.2f} {snapshot.rpy_deg[1]:.2f} {snapshot.rpy_deg[2]:.2f}",
        font=font_body,
        fill=title_color,
        stroke_fill=(80, 180, 255),
        stroke_width=2,
    )

    image = image.convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _start_camera_session() -> tuple[WujiZmqCameraClient, tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]]]:
    control_tunnel = start_ssh_tunnel(DEFAULT_CONTROL_PORT, remote_host=WUYOU_HOST, ssh_alias=WUYOU_SSH_ALIAS)
    stream_tunnel = start_ssh_tunnel(
        DEFAULT_LEFT_HAND_STREAM_PORT,
        remote_host=WUYOU_HOST,
        ssh_alias=WUYOU_SSH_ALIAS,
    )
    time.sleep(DEFAULT_FORWARD_WAIT_S)
    client = WujiZmqCameraClient(
        host="127.0.0.1",
        control_port=DEFAULT_CONTROL_PORT - 1,
        request_timeout_ms=DEFAULT_REQUEST_TIMEOUT_MS,
        stream_timeout_ms=DEFAULT_STREAM_TIMEOUT_MS,
        camera_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
    )
    return client, (control_tunnel, stream_tunnel)


def _camera_preview_loop(
    client: WujiZmqCameraClient,
    connected_arm: ConnectedArm,
    stop_event: threading.Event,
    jog_mode_ref: list[str],
) -> None:
    window_name = "Wuji left-hand camera debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while not stop_event.is_set():
            frame = next(client.stream_camera_rgb_frames(DEFAULT_CAMERA_NAME))
            preview = frame.color_bgr.copy()
            snapshot = _build_snapshot(connected_arm.robot, connected_arm.ec)
            preview = _draw_camera_overlay(preview, snapshot, jog_mode_ref[0])
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                stop_event.set()
                return
            time.sleep(DEFAULT_REFRESH_INTERVAL_S)
    finally:
        cv2.destroyWindow(window_name)


def _normalize_key(key: object) -> str | None:
    try:
        from pynput import keyboard
    except Exception:  # noqa: BLE001
        return None
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        return str(key.char).lower()
    return None


def _start_keyboard_listener(event_queue: SimpleQueue[KeyEvent]) -> Any:
    try:
        from pynput import keyboard
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"pynput 不可用: {exc}") from exc

    pressed_keys: set[str] = set()
    pressed_lock = threading.Lock()

    def on_press(key: object) -> None:
        normalized = _normalize_key(key)
        if normalized is None:
            return
        if normalized == "q":
            event_queue.put(KeyEvent("exit", "q"))
            return
        if normalized not in KEY_BINDINGS_CARTESIAN:
            return
        with pressed_lock:
            if normalized in pressed_keys:
                return
            pressed_keys.add(normalized)
        event_queue.put(KeyEvent("press", normalized))

    def on_release(key: object) -> None:
        normalized = _normalize_key(key)
        if normalized is None or normalized not in KEY_BINDINGS_CARTESIAN:
            return
        with pressed_lock:
            if normalized not in pressed_keys:
                return
            pressed_keys.remove(normalized)
        event_queue.put(KeyEvent("release", normalized))

    listener: keyboard.Listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def _start_jog(
    connected_arm: ConnectedArm,
    binding: JogBinding,
    jog_mode: str,
) -> tuple[bool, str]:
    robot = connected_arm.robot
    ec = connected_arm.ec
    jog_space = (
        xCoreSDK_python.JogOptSpace.jointSpace
        if jog_mode == "joint"
        else xCoreSDK_python.JogOptSpace.wobjFrame
    )
    robot.startJog(
        jog_space,
        binding.rate,
        binding.step,
        binding.index,
        binding.direction,
        ec,
    )
    if ec.get("ec", 0) != 0:
        _print_sdk_result(f"startJog({binding.axis_name})", ec)
        return False, f"Jog 启动失败 axis={binding.axis_name} mode={jog_mode}"
    return True, f"Jog 启动成功 axis={binding.axis_name} mode={jog_mode}"


def _stop_jog(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object], active_jog: ActiveJogState | None) -> None:
    if active_jog is None:
        return
    robot.stop(ec)
    if ec.get("ec", 0) != 0:
        _print_sdk_result(f"stop({active_jog.binding.axis_name})", ec)


def _build_robot_snapshot_text(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> str:
    operate_mode = robot.operateMode(ec)
    operation_state = robot.operationState(ec)
    power_state = robot.powerState(ec)
    cart_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
    toolset = robot.toolset(ec)
    return (
        f"mode={operate_mode} state={operation_state} power={power_state} ({_describe_power_state(power_state)})\n"
        f"xyz(mm)=[{_format_values(_to_xyz_tuple(_m_to_mm(tuple(cart_pose.trans))))}] "
        f"rpy(deg)=[{_format_values(_to_xyz_tuple(_rad_to_deg(tuple(cart_pose.rpy))))}]\n"
        f"tool={DEFAULT_TOOL_NAME} wobj={DEFAULT_WOBJ_NAME}\n"
        f"tool frame: {_format_frame_values(toolset.end)}\n"
        f"wobj frame: {_format_frame_values(toolset.ref)}"
    )


def _run_jog_loop(connected_arm: ConnectedArm, stop_event: threading.Event, jog_mode_ref: list[str]) -> None:
    event_queue: SimpleQueue[KeyEvent] = SimpleQueue()
    listener = _start_keyboard_listener(event_queue)
    active_jog: ActiveJogState | None = None
    logger.info("按键: W/S A/D R/F J/L I/K Y/H，Q 退出")
    try:
        while not stop_event.is_set():
            while not event_queue.empty():
                event = event_queue.get()
                if event.event_type == "exit":
                    stop_event.set()
                    break
                if event.key == "p" and event.event_type == "press":
                    jog_mode_ref[0] = "joint" if jog_mode_ref[0] == "cartesian" else "cartesian"
                    logger.info("已切换到 {} 模式", jog_mode_ref[0])
                    continue
                if event.key is None:
                    continue
                bindings = KEY_BINDINGS_JOINT if jog_mode_ref[0] == "joint" else KEY_BINDINGS_CARTESIAN
                binding = bindings.get(event.key)
                if binding is None:
                    continue
                if event.event_type == "press":
                    if active_jog is not None:
                        _stop_jog(connected_arm.robot, connected_arm.ec, active_jog)
                        active_jog = None
                    started, notice = _start_jog(connected_arm, binding, jog_mode_ref[0])
                    logger.info(notice)
                    if started:
                        active_jog = ActiveJogState(binding=binding, pressed_at=time.monotonic(), pending_stop_at=None)
                elif event.event_type == "release" and active_jog is not None and active_jog.binding.key == event.key:
                    if time.monotonic() - active_jog.pressed_at >= DEFAULT_TAP_MIN_DURATION_S:
                        _stop_jog(connected_arm.robot, connected_arm.ec, active_jog)
                        active_jog = None
                    else:
                        active_jog.pending_stop_at = active_jog.pressed_at + DEFAULT_TAP_MIN_DURATION_S
            if active_jog is not None and active_jog.pending_stop_at is not None:
                if time.monotonic() >= active_jog.pending_stop_at:
                    _stop_jog(connected_arm.robot, connected_arm.ec, active_jog)
                    active_jog = None
            logger.info("\n{}", _build_robot_snapshot_text(connected_arm.robot, connected_arm.ec))
            time.sleep(0.5)
    finally:
        listener.stop()


def main() -> int:
    logger.info("左臂实时相机 + 笛卡尔 jog 调试脚本")
    logger.info("左臂 IP: {} | 相机: {}", DEFAULT_LEFT_ARM_IP, DEFAULT_CAMERA_NAME)
    logger.info("笛卡尔 jog 基坐标系固定为 tool={} wobj={}", DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME)

    arm = _connect_left_arm()
    camera_client: WujiZmqCameraClient | None = None
    tunnels: tuple[subprocess.Popen[bytes], subprocess.Popen[bytes]] | None = None
    stop_event = threading.Event()
    jog_mode_ref = ["cartesian"]
    camera_thread: threading.Thread | None = None
    jog_thread: threading.Thread | None = None
    try:
        camera_client, tunnels = _start_camera_session()
        camera_thread = threading.Thread(
            target=_camera_preview_loop,
            args=(camera_client, arm, stop_event, jog_mode_ref),
            name="left-hand-camera-preview",
            daemon=True,
        )
        jog_thread = threading.Thread(
            target=_run_jog_loop,
            args=(arm, stop_event, jog_mode_ref),
            name="left-arm-jog",
            daemon=True,
        )
        camera_thread.start()
        jog_thread.start()
        while not stop_event.is_set():
            time.sleep(0.1)
        return 0
    finally:
        stop_event.set()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)
        if jog_thread is not None:
            jog_thread.join(timeout=2.0)
        if camera_client is not None:
            camera_client.close()
        if tunnels is not None:
            stop_ssh_process(tunnels[0])
            stop_ssh_process(tunnels[1])
        _shutdown_robot(arm.robot, arm.ec)


if __name__ == "__main__":
    raise SystemExit(main())
