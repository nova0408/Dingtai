"""通过 RobotControl HTTP 服务访问双臂 xCoreSDK 能力。"""

from __future__ import annotations

import json
import time
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from loguru import logger

from .settings import ReplayArmSettings, ReplayServiceSettings

_ResultT = TypeVar("_ResultT")
_DIRECT_OPENER = build_opener(ProxyHandler({}))


@dataclass(slots=True)
class ConnectedArm:
    """RecordReplay 持有的一侧 RobotControl 逻辑连接。"""

    arm_side: str
    robot_ip: str
    base_url: str
    robot_type: str
    robot_uid: str
    command_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class CartesianPose:
    """xCoreSDK CartesianPosition 的显式 HTTP 数据。"""

    trans_m: tuple[float, ...]
    rpy_rad: tuple[float, ...]
    has_elbow: bool
    elbow_rad: float
    conf_data: tuple[int, ...]


def retry_non_motion_call(
    label: str,
    operation: Callable[[], _ResultT],
    retry_count: int,
    retry_delay_s: float,
    stop_event: threading.Event | None = None,
) -> _ResultT:
    """重试不直接提交运动的 RobotControl HTTP 调用。"""

    last_error: BaseException | None = None
    for attempt in range(1, retry_count + 1):
        _raise_if_stopped(stop_event)
        try:
            result = operation()
            _raise_if_stopped(stop_event)
            return result
        except BaseException as error:
            last_error = error
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError(f"检测到停止请求，终止 {label} 重试") from error
            logger.warning(
                "{} 失败，准备重试 attempt={}/{} delay={:.1f}s exc={}",
                label,
                attempt,
                retry_count,
                retry_delay_s,
                error,
            )
            if attempt < retry_count:
                if stop_event is not None and stop_event.wait(timeout=retry_delay_s):
                    raise RuntimeError(f"检测到停止请求，终止 {label} 重试") from error
                if stop_event is None:
                    time.sleep(retry_delay_s)
    raise RuntimeError(f"{label} 连续失败 {retry_count} 次") from last_error


def connect_arm(
    arm_side: str,
    robot_ip: str,
    settings: ReplayServiceSettings,
    stop_event: threading.Event | None = None,
) -> ConnectedArm:
    """通过 RobotControl 读取一侧机械臂身份并应用固定 tool/wobj。"""

    if arm_side not in {"left", "right"}:
        raise ValueError(f"不支持的机械臂侧别：{arm_side}")
    base_url = settings.arm.robot_control_base_url.rstrip("/")
    payload = retry_non_motion_call(
        f"RobotControl robotInfo({arm_side})",
        lambda: _request(base_url, arm_side, "robot-info", timeout_s=settings.arm.http_timeout_s),
        settings.non_motion_retry_count,
        settings.non_motion_retry_delay_s,
        stop_event,
    )
    robot_type = _string(payload, "robot_type")
    robot_uid = _string(payload, "robot_uid")
    detected_arm_side = detect_arm_side(robot_type, settings.arm)
    if detected_arm_side != arm_side:
        raise RuntimeError(
            f"连接到的机械臂侧别不匹配：expected={arm_side}, ip={robot_ip}, actual={detected_arm_side}"
        )
    arm = ConnectedArm(arm_side, robot_ip, base_url, robot_type, robot_uid)
    apply_named_toolset(arm, settings, stop_event)
    logger.info(
        "RobotControl 机械臂连接与身份校验完成 arm_side={} ip={} robot_type={} robot_uid={}",
        arm_side,
        robot_ip,
        robot_type,
        robot_uid,
    )
    return arm


def prepare_nrt_motion(
    connected_arm: ConnectedArm,
    settings: ReplayServiceSettings,
    stop_event: threading.Event | None = None,
) -> None:
    """保持原回放顺序，通过显式 HTTP 接口准备 NRT 运动。"""

    arm_settings = settings.arm
    with connected_arm.command_lock:
        _prepare_nrt_motion_locked(connected_arm, arm_settings, stop_event)
        _post(connected_arm, "set-default-conf-opt", {"enabled": False})
        apply_named_toolset(connected_arm, settings, stop_event)
        _post(
            connected_arm,
            "set-default-speed",
            {"speed_mm_s": arm_settings.default_cartesian_speed_mm_s},
        )
        _post(
            connected_arm,
            "set-default-zone",
            {"zone_mm": arm_settings.default_cartesian_zone_mm},
        )
        if read_power_state(connected_arm) != "on":
            raise RuntimeError(f"{connected_arm.arm_side} 臂 NRT 准备完成后电机未处于 on 状态")


def restore_nrt_motion_state_locked(
    connected_arm: ConnectedArm,
    settings: ReplayServiceSettings,
    stop_event: threading.Event | None = None,
) -> None:
    """在已持有命令锁时恢复 NRT、自动模式和电机上电状态。

    该恢复流程不清空已排入的 MoveAbsJ 队列，供 ``moveStart`` 因电机状态拒绝
    后在同一命令临界区内使用。
    """

    _prepare_nrt_motion_locked(connected_arm, settings.arm, stop_event)


def _prepare_nrt_motion_locked(
    connected_arm: ConnectedArm,
    arm_settings: ReplayArmSettings,
    stop_event: threading.Event | None,
) -> None:
    """执行必须在 ``connected_arm.command_lock`` 内完成的 NRT 状态准备。"""

    _raise_if_stopped(stop_event)
    _post(connected_arm, "set-motion-control-mode")
    _post(connected_arm, "set-operate-mode")
    _post(connected_arm, "set-power-state", {"enabled": True})
    _wait_power_on(connected_arm, arm_settings, stop_event)


def close_arm(connected_arm: ConnectedArm) -> None:
    """尽力停止、退出拖动并下电；SDK 连接继续由 RobotControl 管理。"""

    with connected_arm.command_lock:
        for operation, body in (
            ("stop", None),
            ("disable-drag", None),
            ("set-power-state", {"enabled": False}),
        ):
            try:
                _post(connected_arm, operation, body)
            except Exception as error:
                logger.warning("关闭 {} 臂时发生异常 operation={} error={}", connected_arm.arm_side, operation, error)


def stop_arm(connected_arm: ConnectedArm) -> None:
    """停止一侧 AR5 当前规划运动。"""

    with connected_arm.command_lock:
        _post(connected_arm, "stop")


def apply_named_toolset(
    connected_arm: ConnectedArm,
    settings: ReplayServiceSettings,
    stop_event: threading.Event | None = None,
) -> None:
    """应用当前回放固定的 tool/wobj。"""

    _raise_if_stopped(stop_event)
    _post(
        connected_arm,
        "set-toolset",
        {"tool_name": settings.arm.tool_name, "wobj_name": settings.arm.wobj_name},
    )


def read_cart_posture(connected_arm: ConnectedArm) -> CartesianPose:
    payload = _request(connected_arm.base_url, connected_arm.arm_side, "cart-posture")
    return CartesianPose(
        _float_tuple(payload, "trans_m", 3),
        _float_tuple(payload, "rpy_rad", 3),
        _bool(payload, "has_elbow"),
        _float(payload, "elbow_rad"),
        _int_tuple(payload, "conf_data"),
    )


def calculate_ik(connected_arm: ConnectedArm, pose: CartesianPose) -> tuple[float, ...]:
    payload = _post(
        connected_arm,
        "calc-ik",
        {
            "trans_m": pose.trans_m,
            "rpy_rad": pose.rpy_rad,
            "has_elbow": pose.has_elbow,
            "elbow_rad": pose.elbow_rad,
            "conf_data": pose.conf_data,
        },
    )
    return _float_tuple(payload, "joints_rad", 7)


def move_reset(connected_arm: ConnectedArm) -> None:
    _post(connected_arm, "move-reset")


def move_append_abs_j(
    connected_arm: ConnectedArm,
    targets: Sequence[tuple[Sequence[float], float, float]],
) -> None:
    _post(
        connected_arm,
        "move-append",
        {
            "targets": [
                {"joints_rad": list(joints), "speed_mm_s": speed, "zone_mm": zone}
                for joints, speed, zone in targets
            ]
        },
    )


def move_start(connected_arm: ConnectedArm) -> None:
    _post(connected_arm, "move-start")


def read_operation_state(connected_arm: ConnectedArm) -> str:
    return _string(_request(connected_arm.base_url, connected_arm.arm_side, "operation-state"), "state")


def read_operate_mode(connected_arm: ConnectedArm) -> str:
    return _string(_request(connected_arm.base_url, connected_arm.arm_side, "operate-mode"), "mode")


def read_power_state(connected_arm: ConnectedArm) -> str:
    return _string(_request(connected_arm.base_url, connected_arm.arm_side, "power-state"), "state")


def read_arm_diagnostic(
    base_url: str, arm_side: str
) -> tuple[str, str, str, str, str]:
    """只读获取机械臂身份、工作模式、运行状态和电机状态。"""

    identity = _request(base_url.rstrip("/"), arm_side, "robot-info")
    operation = _request(base_url.rstrip("/"), arm_side, "operation-state")
    operate = _request(base_url.rstrip("/"), arm_side, "operate-mode")
    power = _request(base_url.rstrip("/"), arm_side, "power-state")
    return (
        _string(identity, "robot_type"),
        _string(identity, "robot_uid"),
        _string(operate, "mode"),
        _string(operation, "state"),
        _string(power, "state"),
    )


def detect_arm_side(robot_type: str, settings: ReplayArmSettings) -> str:
    for arm_side, expected in (("left", settings.left_arm_type), ("right", settings.right_arm_type)):
        if robot_type == expected:
            return arm_side
    raise ValueError(f"未识别的机器人型号：{robot_type}")


def _wait_power_on(
    arm: ConnectedArm, settings: ReplayArmSettings, stop_event: threading.Event | None
) -> None:
    deadline = time.monotonic() + settings.power_on_timeout_s
    while time.monotonic() < deadline:
        _raise_if_stopped(stop_event)
        if read_power_state(arm) == "on":
            return
        if stop_event is not None:
            stop_event.wait(settings.power_on_poll_interval_s)
        else:
            time.sleep(settings.power_on_poll_interval_s)
    raise RuntimeError("机械臂未在超时内确认上电")


def _post(arm: ConnectedArm, operation: str, body: Mapping[str, object] | None = None) -> dict[str, object]:
    return _request(arm.base_url, arm.arm_side, operation, body=body, method="POST")


def _request(
    base_url: str,
    side: str,
    operation: str,
    *,
    body: Mapping[str, object] | None = None,
    method: str = "GET",
    timeout_s: float = 10.0,
) -> dict[str, object]:
    url = f"{base_url}/api/v1/ar5/{side}/{operation}"
    data = None if body is None else json.dumps(dict(body), ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with _DIRECT_OPENER.open(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"RobotControl HTTP {error.code} operation={operation} detail={detail}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"RobotControl 请求失败 operation={operation} url={url}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"RobotControl 响应不是 JSON object operation={operation}")
    return {str(key): value for key, value in payload.items()}


def _string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"RobotControl 响应字段 {name} 不是非空字符串：{value!r}")
    return value


def _float(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"RobotControl 响应字段 {name} 不是数值：{value!r}")
    return float(value)


def _bool(payload: Mapping[str, object], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise RuntimeError(f"RobotControl 响应字段 {name} 不是 bool：{value!r}")
    return value


def _float_tuple(payload: Mapping[str, object], name: str, length: int) -> tuple[float, ...]:
    value = payload.get(name)
    if not isinstance(value, list) or len(value) != length:
        raise RuntimeError(f"RobotControl 响应字段 {name} 长度不是 {length}：{value!r}")
    return tuple(_float({name: item}, name) for item in value)


def _int_tuple(payload: Mapping[str, object], name: str) -> tuple[int, ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise RuntimeError(f"RobotControl 响应字段 {name} 不是数组：{value!r}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise RuntimeError(f"RobotControl 响应字段 {name} 包含非整数：{value!r}")
    return tuple(value)


def _raise_if_stopped(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("检测到停止请求，禁止继续准备机械臂")
