"""CameraPipeline 对外 HTTP 与 WebSocket 的数据编码工具。

内部 ZMQ 使用 ``CPW1``，只供同一部署包内的 Python client 使用。本模块定义的
``CPWS1`` 仅用于 WebSocket 图像帧，避免把内部 dataclass 编解码格式直接暴露给
其它项目。
"""

from __future__ import annotations

import base64
import json
import math
import struct
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import TypeAlias

import numpy as np

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

_STREAM_MAGIC = b"CPWS1"
_STREAM_HEADER = struct.Struct("!5sI")


def to_json_value(value: object) -> JsonValue:
    """把协议 dataclass、枚举、tuple 和 NumPy 数组转换为 JSON 值。

    HTTP debug 图像使用 base64，适合控制接口返回；高频图像流使用下面的
    ``encode_stream_packet``，不会经过 base64。
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else str(value.value)
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "encoding": "base64",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, tuple):
        return [to_json_value(item) for item in value]
    if isinstance(value, list):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return to_json_value(asdict(value))  # pyright: ignore[reportArgumentType]
    raise TypeError(f"unsupported external JSON value: {type(value).__name__}")


def encode_stream_packet(packet_type: str, fields: dict[str, object]) -> bytes:
    """编码一个 ``CPWS1`` WebSocket 二进制帧。

    metadata 中的数组 ``offset`` 和 ``nbytes`` 相对于 metadata 后面的二进制区，
    数组 dtype 和 shape 显式传输，GUI 不需要导入 CameraPipeline 的 Python 包。
    """

    blobs: list[bytes] = []
    encoded_fields = {
        name: _encode_stream_value(value, blobs)
        for name, value in fields.items()
    }
    metadata = {
        "protocol": "camera_pipeline.websocket",
        "protocol_version": 1,
        "packet_type": packet_type,
        "fields": encoded_fields,
    }
    metadata_bytes = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        (_STREAM_HEADER.pack(_STREAM_MAGIC, len(metadata_bytes)), metadata_bytes, *blobs)
    )


def _encode_stream_value(value: object, blobs: list[bytes]) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_encode_stream_value(item, blobs) for item in value]
    if isinstance(value, list):
        return [_encode_stream_value(item, blobs) for item in value]
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        raw = contiguous.tobytes(order="C")
        offset = sum(len(blob) for blob in blobs)
        blobs.append(raw)
        return {
            "encoding": "raw",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "offset": offset,
            "nbytes": len(raw),
        }
    raise TypeError(f"unsupported stream value: {type(value).__name__}")
