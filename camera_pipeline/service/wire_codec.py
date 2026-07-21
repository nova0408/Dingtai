from __future__ import annotations

import json
import struct
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import TypeAlias, TypeVar

import numpy as np

from ..ball_pose_detection.protocol import (
    BallDetectionInfo,
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPosePriorInfo,
)
from ..protocol import CameraColorFramePacket, CameraDepthFramePacket, CameraFramePacket, CameraName
from .protocol import (
    CameraColorFrameSubscribeResponse,
    CameraDepthFrameSubscribeResponse,
    CameraFrameSubscribeResponse,
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraPipelineServiceResponse,
    CameraStatusResponse,
    CameraSummaryResponse,
    StableFrameResponse,
    CharucoDetectionRequest,
    CharucoDetectionResponse,
)

_MAGIC = b"CPW1"
_HEADER = struct.Struct("!4sI")
_MAX_METADATA_BYTES = 64 * 1024 * 1024

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

WireDataclass: TypeAlias = (
    CameraFramePacket
    | CameraColorFramePacket
    | CameraDepthFramePacket
    | BallPosePriorInfo
    | BallPoseDetectionRequest
    | BallDetectionInfo
    | BallPoseDetectionDebugArtifacts
    | BallPoseDetectionResponse
    | CameraSummaryResponse
    | CameraIntrinsicsResponse
    | CameraStatusResponse
    | StableFrameResponse
    | CameraFrameSubscribeResponse
    | CameraColorFrameSubscribeResponse
    | CameraDepthFrameSubscribeResponse
    | CameraPipelineServiceRequest
    | CameraPipelineServiceResponse
    | CharucoDetectionRequest
    | CharucoDetectionResponse
)
WireValue: TypeAlias = (
    JsonScalar
    | CameraName
    | np.generic
    | np.ndarray
    | tuple["WireValue", ...]
    | list["WireValue"]
    | dict[str, "WireValue"]
    | WireDataclass
)
DecodedValue: TypeAlias = (
    JsonScalar
    | CameraName
    | np.ndarray
    | tuple["DecodedValue", ...]
    | list["DecodedValue"]
    | dict[str, "DecodedValue"]
    | WireDataclass
)
DecodedT = TypeVar("DecodedT", bound=DecodedValue)

_DATACLASS_TYPES = (
    CameraFramePacket,
    CameraColorFramePacket,
    CameraDepthFramePacket,
    BallPosePriorInfo,
    BallPoseDetectionRequest,
    BallDetectionInfo,
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionResponse,
    CameraSummaryResponse,
    CameraIntrinsicsResponse,
    CameraStatusResponse,
    StableFrameResponse,
    CameraFrameSubscribeResponse,
    CameraColorFrameSubscribeResponse,
    CameraDepthFrameSubscribeResponse,
    CameraPipelineServiceRequest,
    CameraPipelineServiceResponse,
    CharucoDetectionRequest,
    CharucoDetectionResponse,
)
_TYPE_TO_ID: dict[type[WireDataclass], str] = {
    item: item.__name__ for item in _DATACLASS_TYPES
}
_ID_TO_FACTORY: dict[str, Callable[..., WireDataclass]] = {
    item.__name__: item for item in _DATACLASS_TYPES
}


class WireCodecError(RuntimeError):
    """显式二进制协议编码或解码失败。"""


def encode_wire(value: WireValue) -> bytes:
    """把白名单协议对象编码为 JSON 元数据和 NumPy 原始字节块。"""

    blobs: list[bytes] = []
    metadata = _encode_value(value, blobs)
    metadata_bytes = json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(metadata_bytes) > _MAX_METADATA_BYTES:
        raise WireCodecError("wire metadata is too large")
    return b"".join((_HEADER.pack(_MAGIC, len(metadata_bytes)), metadata_bytes, *blobs))


def decode_wire(payload: bytes, expected_type: type[DecodedT]) -> DecodedT:
    """解码显式二进制协议，并校验最外层协议类型。"""

    if len(payload) < _HEADER.size:
        raise WireCodecError("wire payload is shorter than header")
    magic, metadata_size = _HEADER.unpack_from(payload)
    if magic != _MAGIC:
        raise WireCodecError("unsupported wire protocol magic")
    if metadata_size > _MAX_METADATA_BYTES:
        raise WireCodecError("wire metadata is too large")
    metadata_end = _HEADER.size + metadata_size
    if metadata_end > len(payload):
        raise WireCodecError("wire metadata is truncated")
    try:
        metadata: JsonValue = json.loads(
            payload[_HEADER.size:metadata_end].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WireCodecError(f"invalid wire metadata: {exc}") from exc
    value = _decode_value(metadata, memoryview(payload)[metadata_end:])
    if not isinstance(value, expected_type):
        raise WireCodecError(
            f"unexpected wire type {type(value).__name__}; expected {expected_type.__name__}"
        )
    return value


def _encode_value(value: WireValue, blobs: list[bytes]) -> JsonValue:
    if isinstance(value, CameraName):
        return {"@camera_name": value.value}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise WireCodecError("object dtype arrays are not supported")
        contiguous = np.ascontiguousarray(value)
        offset = sum(len(blob) for blob in blobs)
        raw = contiguous.tobytes(order="C")
        blobs.append(raw)
        return {
            "@array": {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        }
    if isinstance(value, tuple):
        return {"@tuple": [_encode_value(item, blobs) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item, blobs) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise WireCodecError("wire dictionaries require string keys")
        return {key: _encode_value(item, blobs) for key, item in value.items()}
    value_type = type(value)
    type_id = _TYPE_TO_ID.get(value_type)
    if type_id is None or not is_dataclass(value):
        raise WireCodecError(f"unsupported wire value type: {value_type.__name__}")
    return {
        "@type": type_id,
        "fields": {
            field.name: _encode_value(getattr(value, field.name), blobs)
            for field in fields(value)
        },
    }


def _decode_value(value: JsonValue, binary: memoryview) -> DecodedValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_decode_value(item, binary) for item in value]
    if not isinstance(value, dict):
        raise WireCodecError(f"unsupported wire metadata type: {type(value).__name__}")
    if "@array" in value:
        return _decode_array(value["@array"], binary)
    if "@camera_name" in value:
        camera_name = value["@camera_name"]
        if not isinstance(camera_name, str):
            raise WireCodecError("camera name must be a string")
        try:
            return CameraName(camera_name)
        except ValueError as exc:
            raise WireCodecError(f"unsupported camera name: {camera_name}") from exc
    if "@tuple" in value:
        items = value["@tuple"]
        if not isinstance(items, list):
            raise WireCodecError("tuple metadata must be a list")
        return tuple(_decode_value(item, binary) for item in items)
    if "@type" in value:
        return _decode_dataclass(value, binary)
    return {key: _decode_value(item, binary) for key, item in value.items()}


def _decode_array(metadata: JsonValue, binary: memoryview) -> np.ndarray:
    if not isinstance(metadata, dict):
        raise WireCodecError("array metadata must be a dictionary")
    dtype_value = metadata.get("dtype")
    shape_value = metadata.get("shape")
    offset_value = metadata.get("offset")
    nbytes_value = metadata.get("nbytes")
    if not isinstance(dtype_value, str):
        raise WireCodecError("array dtype must be a string")
    if not isinstance(shape_value, list):
        raise WireCodecError("array shape must contain integers")
    shape_items: list[int] = []
    for item in shape_value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise WireCodecError("array shape must contain integers")
        shape_items.append(item)
    if not isinstance(offset_value, int) or isinstance(offset_value, bool):
        raise WireCodecError("array offset must be an integer")
    if not isinstance(nbytes_value, int) or isinstance(nbytes_value, bool):
        raise WireCodecError("array nbytes must be an integer")
    try:
        dtype = np.dtype(dtype_value)
    except TypeError as exc:
        raise WireCodecError(f"invalid array dtype: {exc}") from exc
    shape = tuple(shape_items)
    offset = offset_value
    nbytes = nbytes_value
    if dtype.hasobject or offset < 0 or nbytes < 0:
        raise WireCodecError("invalid array dtype or byte range")
    expected_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected_nbytes != nbytes or offset + nbytes > len(binary):
        raise WireCodecError("array byte range does not match dtype and shape")
    return np.frombuffer(binary[offset : offset + nbytes], dtype=dtype).reshape(shape).copy()


def _decode_dataclass(
    metadata: dict[str, JsonValue], binary: memoryview
) -> WireDataclass:
    type_id = metadata.get("@type")
    field_values = metadata.get("fields")
    if not isinstance(type_id, str) or not isinstance(field_values, dict):
        raise WireCodecError("invalid dataclass metadata")
    dataclass_factory = _ID_TO_FACTORY.get(type_id)
    if dataclass_factory is None:
        raise WireCodecError(f"wire dataclass is not allowed: {type_id}")
    decoded_fields = {
        key: _decode_value(item, binary) for key, item in field_values.items()
    }
    try:
        return dataclass_factory(**decoded_fields)
    except TypeError as exc:
        raise WireCodecError(f"invalid {type_id} fields: {exc}") from exc
