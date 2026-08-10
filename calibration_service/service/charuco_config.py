"""手眼标定 ChArUco 板参数和 OpenCV ArUco 字典适配。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import cast

import cv2


@dataclass(frozen=True, slots=True)
class HandEyeCharucoConfig:
    """手眼采样使用的 ChArUco 板和稳定帧检测默认参数。

    所有长度单位均为 mm，配置只在当前 Calibration Service 进程内生效。
    """

    dictionary_name: str = "DICT_APRILTAG_16H5"
    """OpenCV `cv2.aruco` 可用的预定义字典名称。"""

    squares_x: int = 4
    """ChArUco 板横向方格数。"""

    squares_y: int = 4
    """ChArUco 板纵向方格数。"""

    square_length_mm: float = 20.0
    """ChArUco 方格边长，单位 mm。"""

    marker_length_mm: float = 14.0
    """ArUco marker 边长，单位 mm。"""

    min_charuco_corners: int = 6
    """进入 PnP 的最少 ChArUco 角点数量。"""

    max_frames: int = 300
    """稳定帧检测最多尝试的帧数。"""

    stable_timeout_s: float = 10.0
    """每次等待稳定帧的超时时间，单位 s。"""

    enable_debug: bool = False
    """是否让 CameraPipeline 返回调试叠加图。"""

    def __post_init__(self) -> None:
        """校验配置并确保字典名称来自当前 OpenCV。"""

        if isinstance(self.dictionary_name, str):
            object.__setattr__(self, "dictionary_name", self.dictionary_name.upper())
        _validate_config(self)

    def to_dict(self) -> dict[str, object]:
        """转换为 HTTP JSON data 中的配置对象。"""

        return {
            "dictionary_name": self.dictionary_name,
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_mm": self.square_length_mm,
            "marker_length_mm": self.marker_length_mm,
            "min_charuco_corners": self.min_charuco_corners,
            "max_frames": self.max_frames,
            "stable_timeout_s": self.stable_timeout_s,
            "enable_debug": self.enable_debug,
        }

    def with_updates(self, payload: dict[str, object]) -> "HandEyeCharucoConfig":
        """按 PATCH body 更新部分字段并返回新的不可变配置。"""

        allowed = set(self.to_dict())
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"不支持的手眼 ChArUco 配置字段：{', '.join(unknown)}")
        return replace(self, **payload)


def available_aruco_dictionary_names() -> tuple[str, ...]:
    """返回当前 OpenCV 暴露的去重、规范化预定义字典名称。"""

    names = {
        name.upper()
        for name, value in vars(cv2.aruco).items()
        if name.startswith("DICT_") and isinstance(value, int)
    }
    return tuple(sorted(names))


def aruco_dictionary_id(dictionary_name: str) -> int:
    """解析当前 OpenCV 中预定义 ArUco 字典的整数 ID。"""

    normalized_name = dictionary_name.upper()
    dictionary_ids = {
        name.upper(): int(value)
        for name, value in vars(cv2.aruco).items()
        if name.startswith("DICT_") and isinstance(value, int)
    }
    try:
        return dictionary_ids[normalized_name]
    except KeyError as error:
        raise ValueError(
            f"dictionary_name 必须是当前 OpenCV 可用字典之一：{', '.join(available_aruco_dictionary_names())}"
        ) from error


def _validate_config(config: HandEyeCharucoConfig) -> None:
    if not isinstance(config.dictionary_name, str):
        raise ValueError("dictionary_name 必须是 string")
    for field_name, value in (
        ("squares_x", config.squares_x),
        ("squares_y", config.squares_y),
        ("min_charuco_corners", config.min_charuco_corners),
        ("max_frames", config.max_frames),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field_name} 必须是 integer")
    for field_name, value in (
        ("square_length_mm", config.square_length_mm),
        ("marker_length_mm", config.marker_length_mm),
        ("stable_timeout_s", config.stable_timeout_s),
    ):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field_name} 必须是 number")
    if config.dictionary_name.upper() not in available_aruco_dictionary_names():
        raise ValueError(
            f"dictionary_name 必须是当前 OpenCV 可用字典之一：{', '.join(available_aruco_dictionary_names())}"
        )
    if config.squares_x < 2 or config.squares_y < 2:
        raise ValueError("squares_x 和 squares_y 必须至少为 2")
    if not isfinite(config.square_length_mm) or config.square_length_mm <= 0.0:
        raise ValueError("square_length_mm 必须是大于 0 的有限数值")
    if (
        not isfinite(config.marker_length_mm)
        or not 0.0 < config.marker_length_mm < config.square_length_mm
    ):
        raise ValueError("marker_length_mm 必须满足 0 < marker_length_mm < square_length_mm")
    if config.min_charuco_corners < 4:
        raise ValueError("min_charuco_corners 必须至少为 4")
    if config.max_frames <= 0 or not isfinite(config.stable_timeout_s) or config.stable_timeout_s <= 0.0:
        raise ValueError("max_frames 和 stable_timeout_s 必须大于 0")
    if not isinstance(config.enable_debug, bool):
        raise ValueError("enable_debug 必须是 boolean")


def parse_hand_eye_config_update(payload: dict[str, object]) -> dict[str, object]:
    """校验 PATCH body 的标量类型，避免 replace 隐式接受错误类型。"""

    integer_fields = {"squares_x", "squares_y", "min_charuco_corners", "max_frames"}
    number_fields = {"square_length_mm", "marker_length_mm", "stable_timeout_s"}
    for field_name in integer_fields:
        if field_name in payload and (
            isinstance(payload[field_name], bool) or not isinstance(payload[field_name], int)
        ):
            raise ValueError(f"{field_name} 必须是 integer")
    for field_name in number_fields:
        value = payload.get(field_name)
        if field_name in payload and (
            isinstance(value, bool) or not isinstance(value, int | float)
        ):
            raise ValueError(f"{field_name} 必须是 number")
    if "dictionary_name" in payload and not isinstance(payload["dictionary_name"], str):
        raise ValueError("dictionary_name 必须是 string")
    if "enable_debug" in payload and not isinstance(payload["enable_debug"], bool):
        raise ValueError("enable_debug 必须是 boolean")
    return cast(dict[str, object], payload)
