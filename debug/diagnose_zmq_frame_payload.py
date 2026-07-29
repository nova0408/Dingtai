from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
from dataclasses import dataclass, field
import struct
import time

import cv2
import lz4.block
import numpy as np
import numpy.typing as npt
import zmq

_FRAME_HEADER = struct.Struct("<4sBBBBIIIIIIIQI")
_CAMERA_PORTS = {
    "HEAD": 5560,
    "CHEST": 5561,
    "LEFT": 5562,
}


@dataclass(slots=True)
class StreamStats:
    """单路 ZMQ 相机载荷诊断统计。"""

    received: int = 0
    valid: int = 0
    invalid: int = 0
    sequence_gaps: int = 0
    last_sequence: int | None = None
    last_error: str = ""
    previous_timestamp_ms: float | None = None
    previous_gray: npt.NDArray[np.float32] | None = None
    previous_depth_mm: npt.NDArray[np.float32] | None = None
    frame_gaps_ms: list[float] = field(default_factory=list)
    color_mean_deltas: list[float] = field(default_factory=list)
    color_changed_ratios: list[float] = field(default_factory=list)
    valid_depth_ratios: list[float] = field(default_factory=list)
    depth_median_deltas_mm: list[float] = field(default_factory=list)
    depth_percentile_deltas_mm: list[float] = field(default_factory=list)


def _validate_message(
    camera_id: str,
    raw_message: bytes,
) -> tuple[
    int,
    float,
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint16],
]:
    """校验单条 ZMQ RGBD 消息并返回帧号与时间戳。"""

    if len(raw_message) < _FRAME_HEADER.size:
        raise RuntimeError(
            f"message too short actual={len(raw_message)} header={_FRAME_HEADER.size}"
        )
    (
        magic,
        version,
        camera_index,
        color_format,
        depth_format,
        color_width,
        color_height,
        color_size,
        depth_width,
        depth_height,
        depth_size,
        depth_original_size,
        timestamp_us,
        sequence,
    ) = _FRAME_HEADER.unpack_from(raw_message)
    if magic != b"ZCAM":
        raise RuntimeError(f"invalid magic={magic!r}")
    expected_index = {"HEAD": 4, "CHEST": 5, "LEFT": 6}[camera_id]
    if camera_index != expected_index:
        raise RuntimeError(
            f"camera index mismatch expected={expected_index} actual={camera_index}"
        )
    if version != 1 or color_format != 0:
        raise RuntimeError(
            f"unsupported format version={version} color_format={color_format}"
        )

    expected_message_size = _FRAME_HEADER.size + color_size + depth_size
    if len(raw_message) != expected_message_size:
        raise RuntimeError(
            "message size mismatch "
            f"actual={len(raw_message)} expected={expected_message_size} "
            f"color={color_size} depth={depth_size} sequence={sequence}"
        )
    if color_width <= 0 or color_height <= 0 or color_size <= 0:
        raise RuntimeError(
            "invalid color payload "
            f"size={color_width}x{color_height} bytes={color_size}"
        )
    if depth_format != 1:
        raise RuntimeError(
            f"missing LZ4 depth depth_format={depth_format} sequence={sequence}"
        )

    expected_depth_size = depth_width * depth_height * 2
    if depth_original_size != expected_depth_size:
        raise RuntimeError(
            "depth original size mismatch "
            f"header={depth_original_size} dimensions={expected_depth_size} "
            f"size={depth_width}x{depth_height} sequence={sequence}"
        )
    depth_start = _FRAME_HEADER.size + color_size
    depth_payload = raw_message[depth_start:]
    try:
        decompressed = lz4.block.decompress(
            depth_payload,
            uncompressed_size=depth_original_size,
        )
    except lz4.block.LZ4BlockError as exc:
        raise RuntimeError(
            "LZ4 decode failed "
            f"message={len(raw_message)} color={color_size} "
            f"depth={depth_size} original={depth_original_size} "
            f"size={depth_width}x{depth_height} sequence={sequence}"
        ) from exc
    if len(decompressed) != depth_original_size:
        raise RuntimeError(
            "LZ4 output size mismatch "
            f"actual={len(decompressed)} expected={depth_original_size} "
            f"sequence={sequence}"
        )
    color_bgr = cv2.imdecode(
        np.frombuffer(
            raw_message,
            dtype=np.uint8,
            count=color_size,
            offset=_FRAME_HEADER.size,
        ),
        cv2.IMREAD_COLOR,
    )
    if color_bgr is None:
        raise RuntimeError(f"JPEG decode failed sequence={sequence}")
    depth_mm = np.frombuffer(decompressed, dtype=np.uint16).reshape(
        (depth_height, depth_width)
    )
    return (
        sequence,
        timestamp_us / 1000.0,
        np.asarray(color_bgr, dtype=np.uint8),
        depth_mm,
    )


def _update_stability_metrics(
    stats: StreamStats,
    timestamp_ms: float,
    color_bgr: npt.NDArray[np.uint8],
    depth_mm: npt.NDArray[np.uint16],
) -> None:
    """按正式稳定帧检测器的缩放和公式记录相邻帧指标。"""

    color_small = cv2.resize(
        color_bgr,
        dsize=None,
        fx=0.25,
        fy=0.25,
        interpolation=cv2.INTER_AREA,
    )
    depth_small = cv2.resize(
        depth_mm,
        dsize=(color_small.shape[1], color_small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    gray = cv2.cvtColor(color_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    depth_float = depth_small.astype(np.float32)
    if (
        stats.previous_timestamp_ms is not None
        and stats.previous_gray is not None
        and stats.previous_depth_mm is not None
    ):
        stats.frame_gaps_ms.append(timestamp_ms - stats.previous_timestamp_ms)
        gray_delta = gray - stats.previous_gray
        gray_delta -= float(np.median(gray_delta))
        absolute_gray_delta = np.abs(gray_delta)
        stats.color_mean_deltas.append(float(np.mean(absolute_gray_delta)))
        stats.color_changed_ratios.append(
            float(np.mean(absolute_gray_delta > 12))
        )
        valid_depth = (stats.previous_depth_mm > 0.0) & (depth_float > 0.0)
        stats.valid_depth_ratios.append(float(np.mean(valid_depth)))
        if np.any(valid_depth):
            depth_delta = np.abs(
                depth_float[valid_depth] - stats.previous_depth_mm[valid_depth]
            )
            stats.depth_median_deltas_mm.append(float(np.median(depth_delta)))
            stats.depth_percentile_deltas_mm.append(
                float(np.percentile(depth_delta, 75.0))
            )
    stats.previous_timestamp_ms = timestamp_ms
    stats.previous_gray = gray
    stats.previous_depth_mm = depth_float


def _distribution(values: list[float]) -> str:
    """生成一组指标的中位数、95 分位和最大值摘要。"""

    if not values:
        return "-"
    array = np.asarray(values, dtype=np.float64)
    return (
        f"p50={np.percentile(array, 50):.3f} "
        f"p95={np.percentile(array, 95):.3f} max={np.max(array):.3f}"
    )


def diagnose(host: str, duration_s: float) -> int:
    """同时订阅三路相机并统计上游消息载荷完整性。"""

    context = zmq.Context()
    poller = zmq.Poller()
    socket_to_camera: dict[zmq.Socket, str] = {}
    stats = {camera_id: StreamStats() for camera_id in _CAMERA_PORTS}
    try:
        for camera_id, port in _CAMERA_PORTS.items():
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVHWM, 2)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            socket.connect(f"tcp://{host}:{port}")
            poller.register(socket, zmq.POLLIN)
            socket_to_camera[socket] = camera_id

        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            for socket, _ in poller.poll(timeout=200):
                camera_id = socket_to_camera[socket]
                raw_message = socket.recv()
                current = stats[camera_id]
                current.received += 1
                try:
                    sequence, timestamp_ms, color_bgr, depth_mm = _validate_message(
                        camera_id,
                        raw_message,
                    )
                    current.valid += 1
                    if (
                        current.last_sequence is not None
                        and sequence > current.last_sequence + 1
                    ):
                        current.sequence_gaps += sequence - current.last_sequence - 1
                    current.last_sequence = sequence
                    _update_stability_metrics(
                        current,
                        timestamp_ms,
                        color_bgr,
                        depth_mm,
                    )
                except RuntimeError as exc:
                    current.invalid += 1
                    current.last_error = str(exc)
                    print(f"{camera_id} invalid: {exc}")
    finally:
        for socket in socket_to_camera:
            socket.close(linger=0)
        context.term()

    print("\nsummary")
    exit_code = 0
    for camera_id, current in stats.items():
        print(
            f"{camera_id}: received={current.received} valid={current.valid} "
            f"invalid={current.invalid} sequence_gaps={current.sequence_gaps} "
            f"last_error={current.last_error or '-'}"
        )
        print(f"  frame_gap_ms: {_distribution(current.frame_gaps_ms)}")
        print(f"  color_mean_delta: {_distribution(current.color_mean_deltas)}")
        print(
            "  color_changed_ratio: "
            f"{_distribution(current.color_changed_ratios)}"
        )
        print(
            "  valid_depth_ratio: "
            f"{_distribution(current.valid_depth_ratios)}"
        )
        print(
            "  depth_median_delta_mm: "
            f"{_distribution(current.depth_median_deltas_mm)}"
        )
        print(
            "  depth_p75_delta_mm: "
            f"{_distribution(current.depth_percentile_deltas_mm)}"
        )
        if camera_id == "HEAD":
            print(
                "  HEAD threshold failures: "
                f"gap={sum(value <= 0.0 or value > 150.0 for value in current.frame_gaps_ms)} "
                f"color_mean={sum(value > 2.5 for value in current.color_mean_deltas)} "
                f"color_ratio={sum(value > 0.02 for value in current.color_changed_ratios)} "
                f"valid_depth={sum(value < 0.20 for value in current.valid_depth_ratios)} "
                f"depth_median={sum(value > 8.0 for value in current.depth_median_deltas_mm)} "
                f"depth_p75={sum(value > 25.0 for value in current.depth_percentile_deltas_mm)}"
            )
        if current.invalid > 0 or current.received == 0:
            exit_code = 1
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw wuyou ZMQ camera frame payloads."
    )
    parser.add_argument("--host", default="192.168.100.60")
    parser.add_argument("--duration-s", type=float, default=15.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(diagnose(host=args.host, duration_s=args.duration_s))
