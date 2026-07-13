from __future__ import annotations

# pyright: reportMissingImports=false

import queue
import logging
import struct
import threading
import time
from dataclasses import dataclass

import cv2
import lz4.block
import numpy as np
import zmq

from ..protocol import CameraFramePacket

LOGGER = logging.getLogger("..camera_stream.runtime")


# region 数据结构
@dataclass(frozen=True, slots=True)
class CameraStreamRuntimeConfig:
    """共享相机流运行配置。"""

    host: str = "192.168.100.60"
    "wuyou 相机服务主机地址。"

    control_port: int = 5570
    "相机控制口端口号。"

    stream_port: int = 5562
    "相机数据流端口号。"

    camera_id: str = "LEFT"
    "远端相机控制标识。"

    camera_name: str = "left_hand_camera"
    "项目内逻辑相机名。"

    request_timeout_ms: int = 3000
    "控制命令超时，单位 ms。"

    stream_timeout_ms: int = 8000
    "数据流接收超时，单位 ms。"

    cache_size: int = 16
    "按帧号缓存最近帧数量。"

    max_consecutive_timeouts: int = 3
    "连续收流超时次数上限；超过后触发自愈。"

    recover_retry_interval_s: float = 2.0
    "自愈后再次尝试前的最小间隔，单位 s。"


# endregion


# region 运行时
class CameraStreamRuntime:
    """共享相机流运行时。

    职责边界：
    - 持续从 `wuyou` 拉取真实 RGBD 流。
    - 维护最新帧和最近帧号索引缓存。
    - 不负责托盘检测和抓取位姿估计。
    """

    _FRAME_HEADER_STRUCT = struct.Struct("<4sBBBBIIIIIIIQI")
    _FRAME_HEADER_SIZE = _FRAME_HEADER_STRUCT.size

    def __init__(self, config: CameraStreamRuntimeConfig | None = None) -> None:
        self._config = CameraStreamRuntimeConfig() if config is None else config
        self._context = zmq.Context()
        self._control_socket = self._create_control_socket()
        self._stream_socket: zmq.Socket | None = None
        self._lock = threading.Lock()
        self._latest_frame: CameraFramePacket | None = None
        self._frame_cache: dict[int, CameraFramePacket] = {}
        self._frame_order: queue.Queue[int] = queue.Queue(
            maxsize=max(1, int(self._config.cache_size))
        )
        self._running = False
        self._thread: threading.Thread | None = None
        self._cached_intrinsics: tuple[float, float, float, float] | None = None
        self._last_recover_time = 0.0

    def start(self) -> None:
        """启动后台采流线程。"""

        if self._running:
            return
        try:
            self._send_control_command("set_depth_enabled", {"enable": True})
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("set_depth_enabled failed during camera start: %s", exc)
        try:
            self._cached_intrinsics = self._get_intrinsics_from_control()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("get_intrinsics failed during camera start: %s", exc)
            self._cached_intrinsics = None
        self._stream_socket = self._create_stream_socket()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="orin-camera-stream", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """停止后台采流并释放资源。"""

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self._send_control_command("set_depth_enabled", {"enable": False})
        except Exception:
            pass
        if self._stream_socket is not None:
            self._stream_socket.close(linger=0)
            self._stream_socket = None
        self._control_socket.close(linger=0)
        self._context.term()
        self._cached_intrinsics = None

    def get_latest_frame(self) -> CameraFramePacket | None:
        """获取最新一帧缓存。"""

        with self._lock:
            return self._latest_frame

    def get_frame_by_id(self, frame_id: int) -> CameraFramePacket | None:
        """按帧号查询缓存帧。"""

        with self._lock:
            return self._frame_cache.get(int(frame_id))

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        """等待第一帧就绪。"""

        deadline = time.perf_counter() + max(0.1, float(timeout_s))
        while time.perf_counter() < deadline:
            if self.get_latest_frame() is not None:
                return True
            time.sleep(0.05)
        return False

    def _capture_loop(self) -> None:
        consecutive_timeouts = 0
        while self._running:
            if self._stream_socket is None:
                self._recover_stream_runtime("stream socket missing")
                time.sleep(0.05)
                continue
            try:
                raw_message = self._stream_socket.recv()
                packet = self._decode_frame(raw_message)
            except zmq.error.Again:
                consecutive_timeouts += 1
                if consecutive_timeouts >= max(
                    1, int(self._config.max_consecutive_timeouts)
                ):
                    self._recover_stream_runtime(
                        "stream recv timeout x{0}".format(consecutive_timeouts)
                    )
                    consecutive_timeouts = 0
                continue
            except Exception as exc:
                LOGGER.warning("decode camera frame failed: %s", exc)
                self._recover_stream_runtime("frame decode failure")
                consecutive_timeouts = 0
                continue
            consecutive_timeouts = 0
            with self._lock:
                self._latest_frame = packet
                self._frame_cache[int(packet.frame_id)] = packet
                if self._frame_order.full():
                    try:
                        expired = self._frame_order.get_nowait()
                        self._frame_cache.pop(int(expired), None)
                    except queue.Empty:
                        pass
                self._frame_order.put_nowait(int(packet.frame_id))

    def _recover_stream_runtime(self, reason: str) -> None:
        now = time.perf_counter()
        min_interval_s = max(0.2, float(self._config.recover_retry_interval_s))
        if now - self._last_recover_time < min_interval_s:
            return
        self._last_recover_time = now
        LOGGER.warning("camera stream runtime recovering: %s", reason)
        try:
            self._send_control_command("set_depth_enabled", {"enable": True})
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("set_depth_enabled failed during recover: %s", exc)
        try:
            self._cached_intrinsics = self._get_intrinsics_from_control()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("get_intrinsics failed during recover: %s", exc)
            self._cached_intrinsics = None
        self._recreate_stream_socket()

    def _recreate_stream_socket(self) -> None:
        old_socket = self._stream_socket
        self._stream_socket = None
        if old_socket is not None:
            try:
                old_socket.close(linger=0)
            except Exception:
                pass
        try:
            self._stream_socket = self._create_stream_socket()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("recreate stream socket failed: %s", exc)

    def _send_control_command(
        self,
        command_name: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = {
            "cmd": command_name,
            "camera": self._config.camera_id,
            "params": {} if params is None else params,
        }
        try:
            self._control_socket.send_json(payload)
            response = self._control_socket.recv_json()
        except Exception:
            self._reset_control_socket()
            raise
        if not isinstance(response, dict):
            raise RuntimeError("invalid camera control response")
        if not bool(response.get("success", False)):
            raise RuntimeError(
                str(response.get("error", "unknown camera control error"))
            )
        return dict(response)

    def _decode_frame(self, raw_message: bytes) -> CameraFramePacket:
        if len(raw_message) < self._FRAME_HEADER_SIZE:
            raise RuntimeError("ZMQ camera frame too short")
        if raw_message[:4] != b"ZCAM":
            raise RuntimeError("invalid ZMQ camera frame magic")
        frame_header = self._FRAME_HEADER_STRUCT.unpack(
            raw_message[: self._FRAME_HEADER_SIZE]
        )
        color_data_size = int(frame_header[7])
        depth_width = int(frame_header[8])
        depth_height = int(frame_header[9])
        depth_data_size = int(frame_header[10])
        depth_original_size = int(frame_header[11])
        timestamp_us = int(frame_header[12])
        sequence = int(frame_header[13])
        color_start = self._FRAME_HEADER_SIZE
        color_end = color_start + color_data_size
        color_jpeg = raw_message[color_start:color_end]
        color_bgr = cv2.imdecode(
            np.frombuffer(color_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if color_bgr is None:
            raise RuntimeError("camera jpeg decode failed")
        depth_start = color_end
        depth_end = depth_start + depth_data_size
        depth_bytes = raw_message[depth_start:depth_end]
        depth_raw = lz4.block.decompress(
            depth_bytes, uncompressed_size=depth_original_size
        )
        depth_mm = (
            np.frombuffer(depth_raw, dtype=np.uint16)
            .reshape((depth_height, depth_width))
            .copy()
        )
        fx, fy, cx, cy = self._get_intrinsics()
        return CameraFramePacket(
            frame_id=int(sequence),
            camera_name=str(self._config.camera_name),
            timestamp_ms=float(timestamp_us) / 1000.0,
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        )

    def _get_intrinsics(self) -> tuple[float, float, float, float]:
        if self._cached_intrinsics is not None:
            return self._cached_intrinsics
        self._cached_intrinsics = self._get_intrinsics_from_control()
        return self._cached_intrinsics

    def _get_intrinsics_from_control(self) -> tuple[float, float, float, float]:
        payload = self._send_control_command("get_intrinsics")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("invalid get_intrinsics payload")
        return (
            float(data.get("fx", 910.0)),
            float(data.get("fy", 910.0)),
            float(data.get("cx", 640.0)),
            float(data.get("cy", 360.0)),
        )

    def _tcp_addr(self, port: int) -> str:
        return "tcp://{0}:{1}".format(self._config.host, int(port))

    def _create_stream_socket(self) -> zmq.Socket:
        socket_obj = self._context.socket(zmq.SUB)
        socket_obj.setsockopt(zmq.CONFLATE, 1)
        socket_obj.setsockopt(zmq.RCVHWM, 1)
        socket_obj.setsockopt_string(zmq.SUBSCRIBE, "")
        socket_obj.setsockopt(zmq.RCVTIMEO, int(self._config.stream_timeout_ms))
        socket_obj.connect(self._tcp_addr(self._config.stream_port))
        return socket_obj

    def _create_control_socket(self) -> zmq.Socket:
        socket_obj = self._context.socket(zmq.REQ)
        socket_obj.setsockopt(zmq.RCVTIMEO, int(self._config.request_timeout_ms))
        socket_obj.setsockopt(zmq.SNDTIMEO, int(self._config.request_timeout_ms))
        socket_obj.connect(self._tcp_addr(self._config.control_port))
        return socket_obj

    def _reset_control_socket(self) -> None:
        try:
            self._control_socket.close(linger=0)
        except Exception:
            pass
        self._control_socket = self._create_control_socket()


# endregion
