from __future__ import annotations

import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import lz4.block
import numpy as np
import zmq


# region 数据结构
@dataclass(frozen=True)
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


@dataclass(frozen=True)
class CameraFramePacket:
    """共享相机流中的单帧数据包。"""

    frame_id: int
    "真实相机帧号。优先使用流 sequence。"

    camera_name: str
    "逻辑相机名。"

    timestamp_ms: float
    "帧时间戳，单位 ms。"

    color_bgr: np.ndarray
    "彩色图像，形状 `(H, W, 3)`，dtype `uint8`。"

    depth_mm: np.ndarray
    "深度图，形状 `(H, W)`，dtype `uint16`，单位 mm。"

    fx: float
    "X 方向焦距，单位 像素。"

    fy: float
    "Y 方向焦距，单位 像素。"

    cx: float
    "主点 X 坐标，单位 像素。"

    cy: float
    "主点 Y 坐标，单位 像素。"

    source_meta: Dict[str, str] = field(default_factory=dict)
    "来源元信息。"


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

    def __init__(self, config: Optional[CameraStreamRuntimeConfig] = None) -> None:
        self._config = CameraStreamRuntimeConfig() if config is None else config
        self._context = zmq.Context()
        self._control_socket = self._context.socket(zmq.REQ)
        self._control_socket.setsockopt(zmq.RCVTIMEO, int(self._config.request_timeout_ms))
        self._control_socket.setsockopt(zmq.SNDTIMEO, int(self._config.request_timeout_ms))
        self._control_socket.connect(self._tcp_addr(self._config.control_port))
        self._stream_socket: Optional[zmq.Socket] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[CameraFramePacket] = None
        self._frame_cache: Dict[int, CameraFramePacket] = {}
        self._frame_order: queue.Queue[int] = queue.Queue(maxsize=max(1, int(self._config.cache_size)))
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """启动后台采流线程。"""

        if self._running:
            return
        self._send_control_command("set_depth_enabled", {"enable": True})
        self._stream_socket = self._create_stream_socket()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="orin-camera-stream", daemon=True)
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

    def get_latest_frame(self) -> Optional[CameraFramePacket]:
        """获取最新一帧缓存。"""

        with self._lock:
            return self._latest_frame

    def get_frame_by_id(self, frame_id: int) -> Optional[CameraFramePacket]:
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
        if self._stream_socket is None:
            return
        while self._running:
            try:
                raw_message = self._stream_socket.recv()
                packet = self._decode_frame(raw_message)
            except zmq.error.Again:
                continue
            except Exception:
                continue
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

    def _send_control_command(self, command_name: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        payload = {
            "cmd": command_name,
            "camera": self._config.camera_id,
            "params": {} if params is None else params,
        }
        self._control_socket.send_json(payload)
        response = self._control_socket.recv_json()
        if not isinstance(response, dict):
            raise RuntimeError("invalid camera control response")
        if not bool(response.get("success", False)):
            raise RuntimeError(str(response.get("error", "unknown camera control error")))
        return dict(response)

    def _decode_frame(self, raw_message: bytes) -> CameraFramePacket:
        if len(raw_message) < self._FRAME_HEADER_SIZE:
            raise RuntimeError("ZMQ camera frame too short")
        if raw_message[:4] != b"ZCAM":
            raise RuntimeError("invalid ZMQ camera frame magic")
        frame_header = self._FRAME_HEADER_STRUCT.unpack(raw_message[: self._FRAME_HEADER_SIZE])
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
        color_bgr = cv2.imdecode(np.frombuffer(color_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError("camera jpeg decode failed")
        depth_start = color_end
        depth_end = depth_start + depth_data_size
        depth_bytes = raw_message[depth_start:depth_end]
        depth_raw = lz4.block.decompress(depth_bytes, uncompressed_size=depth_original_size)
        depth_mm = np.frombuffer(depth_raw, dtype=np.uint16).reshape((depth_height, depth_width)).copy()
        fx, fy, cx, cy = self._get_intrinsics()
        return CameraFramePacket(
            frame_id=int(sequence),
            camera_name=str(self._config.camera_name),
            timestamp_ms=float(timestamp_us) / 1000.0,
            color_bgr=np.asarray(color_bgr, dtype=np.uint8).copy(),
            depth_mm=np.asarray(depth_mm, dtype=np.uint16),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            source_meta={"source": "wuyou", "camera_stream": str(self._config.camera_name)},
        )

    def _get_intrinsics(self) -> tuple[float, float, float, float]:
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


# endregion
