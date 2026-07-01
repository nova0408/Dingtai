from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import cv2
import numpy as np
from loguru import logger
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orin.opening_detection_pipeline.protocol import OpeningDetectionPipelineRequest  # noqa: E402
from orin.opening_detection_pipeline.transport import OpeningDetectionPipelineRpcClient, ZmqSocketOptions  # noqa: E402
from src.wuji import SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL, WujiZmqCameraClient  # noqa: E402
from src.wuji.camera_protocol import WujiCameraFrame, WujiCameraName  # noqa: E402


DEFAULT_SERVICE_ADDR = "tcp://192.168.1.116:6220"
DEFAULT_CAMERA_NAME: WujiCameraName = "left_hand_camera"
DEFAULT_CAMERA_HOST = "192.168.100.60"
DEFAULT_CONTROL_PORT = 5570
DEFAULT_STREAM_PORT = 5562
DEFAULT_TARGET_TRAY_INDEX = 0
DEFAULT_COMPUTE_INTERVAL_MS = 120
DEFAULT_RPC_TIMEOUT_MS = 30_000
DEFAULT_WINDOW_NAME = "opening_detection cv2"
DEFAULT_MIN_WINDOW_LONG_SIDE = 900


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    frame: WujiCameraFrame


@dataclass(frozen=True)
class ResultPacket:
    frame_id: int
    response: object | None
    error: str | None


class FrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None

    def put(self, packet: FramePacket) -> None:
        with self._lock:
            self._latest = packet

    def latest(self) -> Optional[FramePacket]:
        with self._lock:
            return self._latest


class ResultBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[ResultPacket] = None

    def put(self, packet: ResultPacket) -> None:
        with self._lock:
            self._latest = packet

    def latest(self) -> Optional[ResultPacket]:
        with self._lock:
            return self._latest


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: WujiCameraName = DEFAULT_CAMERA_NAME,
    target_tray_index: int = DEFAULT_TARGET_TRAY_INDEX,
    compute_interval_ms: int = DEFAULT_COMPUTE_INTERVAL_MS,
) -> int:
    logger.info("启动 opening_detection cv2 viewer")
    frame_buffer = FrameBuffer()
    result_buffer = ResultBuffer()
    stop_event = threading.Event()

    camera_thread = threading.Thread(
        target=_camera_loop,
        args=(frame_buffer, stop_event, str(camera_name)),
        name="opening-detection-camera",
        daemon=True,
    )
    camera_thread.start()

    rpc_thread = threading.Thread(
        target=_rpc_loop,
        args=(frame_buffer, result_buffer, stop_event, service_addr, str(camera_name), int(target_tray_index), int(compute_interval_ms)),
        name="opening-detection-rpc",
        daemon=True,
    )
    rpc_thread.start()

    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_ready = False
    shown_frames = 0
    fps_t0 = time.perf_counter()

    try:
        while True:
            frame_packet = frame_buffer.latest()
            result_packet = result_buffer.latest()
            if frame_packet is None:
                canvas = _build_waiting_canvas(result_packet)
            else:
                canvas = _build_canvas(frame_packet.frame, result_packet)
            if not window_ready:
                win_w, win_h = _compute_preview_window_size(canvas.shape[1], canvas.shape[0], DEFAULT_MIN_WINDOW_LONG_SIDE)
                cv2.resizeWindow(DEFAULT_WINDOW_NAME, win_w, win_h)
                window_ready = True
            cv2.imshow(DEFAULT_WINDOW_NAME, canvas)
            shown_frames += 1
            if shown_frames % 30 == 0:
                elapsed = max(1e-6, time.perf_counter() - fps_t0)
                logger.info("preview_fps={:.1f}", shown_frames / elapsed)
            key = cv2.waitKey(1)
            if key in (27, ord("q"), ord("Q")):
                break
            if _cv_window_closed(DEFAULT_WINDOW_NAME):
                break
    finally:
        stop_event.set()
        camera_thread.join(timeout=1.0)
        rpc_thread.join(timeout=1.0)
        _safe_destroy_cv_window(DEFAULT_WINDOW_NAME)
    return 0


def _camera_loop(frame_buffer: FrameBuffer, stop_event: threading.Event, camera_name: str) -> None:
    client = WujiZmqCameraClient(
        host=DEFAULT_CAMERA_HOST,
        control_port=DEFAULT_CONTROL_PORT,
        request_timeout_ms=3000,
        stream_timeout_ms=8000,
        camera_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
    )
    try:
        status = client.get_camera_status(camera_name)
        logger.info("camera online={} color_enabled={} depth_enabled={}", status.online, status.color_enabled, status.depth_enabled)
        for frame in client.stream_camera_rgbd_frames(camera_name):
            if stop_event.is_set():
                break
            frame_buffer.put(FramePacket(frame_id=_resolve_frame_id(frame), frame=frame))
    except Exception as exc:  # noqa: BLE001
        logger.error("camera loop failed: {}", exc)
    finally:
        client.close()


def _rpc_loop(
    frame_buffer: FrameBuffer,
    result_buffer: ResultBuffer,
    stop_event: threading.Event,
    service_addr: str,
    camera_name: str,
    target_tray_index: int,
    compute_interval_ms: int,
) -> None:
    client = OpeningDetectionPipelineRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=DEFAULT_RPC_TIMEOUT_MS, send_timeout_ms=DEFAULT_RPC_TIMEOUT_MS),
    )
    last_sent_frame_id = -1
    last_submit_ts = 0.0
    try:
        while not stop_event.is_set():
            packet = frame_buffer.latest()
            if packet is None:
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            if packet.frame_id == last_sent_frame_id or now - last_submit_ts < max(0.02, compute_interval_ms / 1000.0):
                time.sleep(0.01)
                continue
            request = OpeningDetectionPipelineRequest(
                request_id=packet.frame_id,
                camera_name=str(camera_name),
                frame_id=packet.frame_id,
                target_tray_index=int(target_tray_index),
                enable_debug=True,
            )
            try:
                response = client.call(request)
                result_buffer.put(ResultPacket(frame_id=packet.frame_id, response=response, error=response.error))
            except Exception as exc:  # noqa: BLE001
                result_buffer.put(ResultPacket(frame_id=packet.frame_id, response=None, error=f"{type(exc).__name__}: {exc}"))
            last_sent_frame_id = packet.frame_id
            last_submit_ts = now
    finally:
        client.close()


def _build_waiting_canvas(result_packet: Optional[ResultPacket]) -> np.ndarray:
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(canvas, "waiting camera / opening_detection result...", (48, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (240, 240, 240), 2, cv2.LINE_AA)
    if result_packet is not None and result_packet.error:
        cv2.putText(canvas, result_packet.error[:120], (48, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 120, 255), 1, cv2.LINE_AA)
    return canvas


def _build_canvas(frame: WujiCameraFrame, result_packet: Optional[ResultPacket]) -> np.ndarray:
    left = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
    right = _build_depth_view(frame.depth) if frame.depth is not None else left.copy()
    if result_packet is not None and result_packet.response is not None:
        response = result_packet.response
        if getattr(response, "debug", None) is not None:
            left = _draw_remote_overlay(left, response)
            right = _build_remote_debug_view(response)
    merged = np.hstack([left, right])
    title = f"frame_id { _resolve_frame_id(frame) }"
    if result_packet is not None:
        if result_packet.response is not None:
            title += f" remote_frame {result_packet.response.frame_id} tray_count {result_packet.response.tray_count}"
        if result_packet.error:
            title += f" err {result_packet.error[:90]}"
    cv2.putText(merged, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    return merged


def _draw_remote_overlay(base_bgr: np.ndarray, response) -> np.ndarray:
    out = np.asarray(base_bgr, dtype=np.uint8).copy()
    debug = response.debug
    if debug is not None:
        for tray_id, tray_mask in enumerate(debug.tray_instance_masks):
            _draw_mask_outline(out, tray_mask, _tray_color_bgr(tray_id))
        if debug.selected_tray_mask is not None:
            _blend_mask(out, debug.selected_tray_mask, (0, 180, 180), 0.18)
        if debug.near_plane_mask is not None:
            _blend_mask(out, debug.near_plane_mask, (0, 140, 255), 0.25)
        if debug.no_hole_mask is not None:
            _blend_mask(out, debug.no_hole_mask, (0, 220, 0), 0.18)
    for tray in response.tray_results:
        x, y, w, h = tray.bbox_xywh
        color = _tray_color_bgr(int(tray.tray_id))
        cv2.rectangle(out, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), color, 2, cv2.LINE_AA)
        cv2.putText(out, f"tray_{tray.tray_id}", (int(x), max(18, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return out


def _build_remote_debug_view(response) -> np.ndarray:
    debug = response.debug
    if debug is None or debug.color_bgr is None:
        return np.zeros((720, 1280, 3), dtype=np.uint8)
    color_bgr = np.asarray(debug.color_bgr, dtype=np.uint8)
    if debug.depth_mm is None:
        return color_bgr
    return np.hstack([color_bgr, _build_depth_view(np.asarray(debug.depth_mm, dtype=np.uint16))])


def _build_depth_view(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1.0)
    hsv = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if np.any(valid):
        z_min = float(np.percentile(depth[valid], 2))
        z_max = float(np.percentile(depth[valid], 98))
        norm = np.clip((depth - z_min) / max(1e-6, z_max - z_min), 0.0, 1.0)
        hsv[..., 0] = np.where(valid, np.rint((1.0 - norm) * 120.0), 0).astype(np.uint8)
        hsv[..., 1] = np.where(valid, 255, 0).astype(np.uint8)
        hsv[..., 2] = np.where(valid, 255, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _blend_mask(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(mask_bool):
        return
    base = np.asarray(base_bgr, dtype=np.float32)
    color = np.asarray(color_bgr, dtype=np.float32)
    base[mask_bool] = base[mask_bool] * (1.0 - float(alpha)) + color * float(alpha)
    base_bgr[:, :, :] = np.clip(base, 0.0, 255.0).astype(np.uint8)


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _tray_color_bgr(index: int) -> tuple[int, int, int]:
    palette = ((0, 220, 255), (80, 200, 120), (255, 170, 0), (255, 110, 180), (140, 180, 255), (200, 120, 255))
    return palette[int(index) % len(palette)]


def _resolve_frame_id(frame: WujiCameraFrame) -> int:
    if frame.sequence_id is not None:
        return int(frame.sequence_id)
    timestamp = frame.timestamp
    if isinstance(timestamp, (int, np.integer)):
        return int(timestamp)
    return int(time.time() * 1000.0)


def _compute_preview_window_size(src_w: int, src_h: int, min_long_side: int) -> tuple[int, int]:
    long_side = max(1, src_w, src_h)
    if long_side >= min_long_side:
        return max(1, src_w), max(1, src_h)
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _safe_destroy_cv_window(window_name: str) -> None:
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def _cv_window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV viewer for Orin opening detection pipeline")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--target-tray-index", type=int, default=DEFAULT_TARGET_TRAY_INDEX)
    parser.add_argument("--compute-interval-ms", type=int, default=DEFAULT_COMPUTE_INTERVAL_MS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=args.camera_name,
            target_tray_index=int(args.target_tray_index),
            compute_interval_ms=int(args.compute_interval_ms),
        )
    )
