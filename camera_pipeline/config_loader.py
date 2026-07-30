from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeAlias, cast

from .camera_local.config import LocalStreamProfileConfig
from .pipeline_context import (
    DEFAULT_CAMERA_ENDPOINTS,
    CameraEndpointConfig,
    PipelineContextConfig,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_CAMERA_IDS = ("LEFT", "RIGHT", "HEAD", "CHEST")


def load_pipeline_context_config(path: Path) -> PipelineContextConfig:
    """从 UTF-8 JSON 文件读取相机输入模式和四个安装位配置。

    Parameters
    ----------
    path:
        配置文件路径，通常为 `camera_pipeline/config.json`。

    Returns
    -------
    PipelineContextConfig
        已完成字段类型与固定安装位校验的上下文配置。

    Raises
    ------
    ValueError
        JSON 根对象、模式、安装位或字段类型不合法。
    """

    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    root = _require_object(raw, "root")
    mode_value = _require_string(root, "camera_source_mode")
    if mode_value not in ("zmq", "usb"):
        raise ValueError("camera_source_mode must be 'zmq' or 'usb'")
    mode = cast(Literal["zmq", "usb"], mode_value)

    zmq_config = _require_object(root.get("zmq"), "zmq")
    usb_config = _require_object(root.get("usb"), "usb")
    camera_config = _require_object(root.get("cameras"), "cameras")
    if set(camera_config) != set(_CAMERA_IDS):
        raise ValueError("cameras must contain exactly LEFT, RIGHT, HEAD and CHEST")

    default_by_id = {item.camera_id: item for item in DEFAULT_CAMERA_ENDPOINTS}
    endpoints: list[CameraEndpointConfig] = []
    for camera_id in _CAMERA_IDS:
        item = _require_object(camera_config.get(camera_id), f"cameras.{camera_id}")
        default = default_by_id[camera_id]
        enabled_key = "zmq_enabled" if mode == "zmq" else "usb_enabled"
        endpoints.append(
            CameraEndpointConfig(
                camera_name=_require_string(item, "camera_name"),
                camera_id=camera_id,
                stream_port=_require_int(item, "zmq_stream_port"),
                stable_frame_config=default.stable_frame_config,
                connected=_require_bool(item, enabled_key),
                serial_number=_require_string(item, "serial_number"),
                color_profile=_read_stream_profile(item, "color"),
                depth_profile=_read_stream_profile(item, "depth"),
            )
        )

    # 默认调用相机由 PipelineContextConfig 的代码常量决定，部署 JSON 不允许覆盖。
    context_defaults = PipelineContextConfig()
    default_endpoint = next(
        item for item in endpoints if item.camera_id == context_defaults.camera_id
    )
    if default_endpoint.camera_name != context_defaults.camera_name:
        raise ValueError(
            "configured default camera_name does not match PipelineContextConfig"
        )
    if not default_endpoint.connected:
        raise ValueError("context default camera must be enabled for the selected mode")

    reconnect_initial_interval_s = _require_number(
        usb_config, "reconnect_initial_interval_s"
    )
    reconnect_max_interval_s = _require_number(
        usb_config, "reconnect_max_interval_s"
    )
    if reconnect_max_interval_s < reconnect_initial_interval_s:
        raise ValueError(
            "reconnect_max_interval_s must be greater than or equal to "
            "reconnect_initial_interval_s"
        )

    return PipelineContextConfig(
        camera_source_mode=mode,
        camera_host=_require_string(zmq_config, "host"),
        camera_control_port=_require_int(zmq_config, "control_port"),
        camera_stream_port=default_endpoint.stream_port,
        camera_request_timeout_ms=_require_int(zmq_config, "request_timeout_ms"),
        camera_stream_timeout_ms=_require_int(zmq_config, "stream_timeout_ms"),
        camera_stale_frame_timeout_s=_require_number(
            zmq_config, "stale_frame_timeout_s"
        ),
        camera_frame_cache_size=_require_int(zmq_config, "frame_cache_size"),
        camera_endpoints=tuple(endpoints),
        usb_frame_timeout_ms=_require_int(usb_config, "frame_timeout_ms"),
        usb_reconnect_initial_interval_s=reconnect_initial_interval_s,
        usb_reconnect_max_interval_s=reconnect_max_interval_s,
    )


def _read_stream_profile(
    camera_config: dict[str, JsonValue], key: str
) -> LocalStreamProfileConfig:
    """读取一个严格的 USB 视频流 profile。"""

    profile = _require_object(camera_config.get(key), key)
    return LocalStreamProfileConfig(
        width=_require_int(profile, "width"),
        height=_require_int(profile, "height"),
        fps=_require_int(profile, "fps"),
        format_name=_require_string(profile, "format"),
    )


def _require_object(value: JsonValue, field_name: str) -> dict[str, JsonValue]:
    """校验 JSON 对象字段。"""

    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_string(payload: dict[str, JsonValue], key: str) -> str:
    """读取字符串字段。"""

    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _require_bool(payload: dict[str, JsonValue], key: str) -> bool:
    """读取布尔字段。"""

    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _require_int(payload: dict[str, JsonValue], key: str) -> int:
    """读取正整数字段。"""

    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _require_number(payload: dict[str, JsonValue], key: str) -> float:
    """读取正数值字段。"""

    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)
