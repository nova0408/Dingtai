from __future__ import annotations

# pyright: reportMissingImports=false

import queue
import json
import struct
import threading
import time
from dataclasses import dataclass
from typing import TypeAlias

import cv2
import lz4.block
import numpy as np
import zmq
from loguru import logger

from ..protocol import CameraFramePacket

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


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

    @property
    def keeps_frame_history(self) -> bool:
        """表示 ZMQ 模式维护有限历史帧缓存。"""

        return True

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
        self._cached_intrinsics: tuple[
            float, float, float, float, tuple[float, ...]
        ] | None = None
        self._last_recover_time = 0.0

    def start(self) -> None:
        """启动后台采流线程。"""

        if self._running:
            logger.warning(
                "camera stream start ignored because runtime is already running camera_name={}",
                self._config.camera_name,
            )
            return
        logger.info(
            "camera stream starting camera_name={} camera_id={} host={} control_port={} stream_port={}",
            self._config.camera_name,
            self._config.camera_id,
            self._config.host,
            self._config.control_port,
            self._config.stream_port,
        )
        try:
            self._send_control_command("set_depth_enabled", {"enable": True})
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "set_depth_enabled failed during camera start camera_name={} error={}",
                self._config.camera_name,
                exc,
            )
        try:
            self._cached_intrinsics = self._get_intrinsics_from_control()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_intrinsics failed during camera start camera_name={} error={}",
                self._config.camera_name,
                exc,
            )
            self._cached_intrinsics = None
        self._stream_socket = self._create_stream_socket()
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="orin-camera-stream", daemon=True
        )
        self._thread.start()
        logger.info("camera stream started camera_name={}", self._config.camera_name)

    def stop(self) -> None:
        """停止后台采流并释放资源。"""

        was_running = self._running
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self._send_control_command("set_depth_enabled", {"enable": False})
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "set_depth_enabled failed during camera stop camera_name={} error={}",
                self._config.camera_name,
                exc,
            )
        if self._stream_socket is not None:
            self._stream_socket.close(linger=0)
            self._stream_socket = None
        self._control_socket.close(linger=0)
        self._context.term()
        self._cached_intrinsics = None
        if was_running:
            logger.info("camera stream stopped camera_name={}", self._config.camera_name)

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
                logger.error(
                    "camera frame decode failed camera_name={} error={}",
                    self._config.camera_name,
                    exc,
                )
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
        logger.warning(
            "camera stream recovering camera_name={} reason={}",
            self._config.camera_name,
            reason,
        )
        try:
            self._send_control_command("set_depth_enabled", {"enable": True})
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "set_depth_enabled failed during recovery camera_name={} error={}",
                self._config.camera_name,
                exc,
            )
        try:
            self._cached_intrinsics = self._get_intrinsics_from_control()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "get_intrinsics failed during recovery camera_name={} error={}",
                self._config.camera_name,
                exc,
            )
            self._cached_intrinsics = None
        self._recreate_stream_socket()

    def _recreate_stream_socket(self) -> None:
        old_socket = self._stream_socket
        self._stream_socket = None
        if old_socket is not None:
            try:
                old_socket.close(linger=0)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "closing stale stream socket failed camera_name={} error={}",
                    self._config.camera_name,
                    exc,
                )
        try:
            self._stream_socket = self._create_stream_socket()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "recreate stream socket failed camera_name={} error={}",
                self._config.camera_name,
                exc,
            )

    def _send_control_command(
        self,
        command_name: str,
        params: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        payload = {
            "cmd": command_name,
            "camera": self._config.camera_id,
            "params": {} if params is None else params,
        }
        try:
            self._control_socket.send_json(payload)
            response: JsonValue = json.loads(
                self._control_socket.recv().decode("utf-8")
            )
        except Exception:
            self._reset_control_socket()
            raise
        if not isinstance(response, dict):
            raise RuntimeError("invalid camera control response")
        success = response.get("success", False)
        if not isinstance(success, bool):
            raise RuntimeError("invalid camera control success flag")
        if not success:
            error = response.get("error", "unknown camera control error")
            if not isinstance(error, str):
                raise RuntimeError("invalid camera control error text")
            raise RuntimeError(error)
        return response

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
        # OpenCV 类型声明只保证 MatLike，在解码边界收窄为协议要求的 uint8 图像。
        color_bgr = np.asarray(color_bgr, dtype=np.uint8)
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
        fx, fy, cx, cy, distortion = self._get_intrinsics()
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
            distortion=distortion,
        )

    def _get_intrinsics(
        self,
    ) -> tuple[float, float, float, float, tuple[float, ...]]:
        if self._cached_intrinsics is not None:
            return self._cached_intrinsics
        self._cached_intrinsics = self._get_intrinsics_from_control()
        return self._cached_intrinsics

    def _get_intrinsics_from_control(
        self,
    ) -> tuple[float, float, float, float, tuple[float, ...]]:
        payload = self._send_control_command("get_intrinsics")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("invalid get_intrinsics payload")
        return (
            _read_json_number(data, "fx", 910.0),
            _read_json_number(data, "fy", 910.0),
            _read_json_number(data, "cx", 640.0),
            _read_json_number(data, "cy", 360.0),
            _read_zmq_distortion(data),
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "reset control socket close failed camera_name={} error={}",
                self._config.camera_name,
                exc,
            )
        self._control_socket = self._create_control_socket()


# endregion


def _read_json_number(
    payload: dict[str, JsonValue], key: str, default: float
) -> float:
    """读取并校验相机控制响应中的数值字段。"""

    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"invalid camera control numeric field: {key}")
    return float(value)


def _read_zmq_distortion(payload: dict[str, JsonValue]) -> tuple[float, ...]:
    """读取远端 SDK 顺序的彩色相机畸变参数并转换为 OpenCV 顺序。"""

    distortion_raw = payload.get("dist", ())
    if not isinstance(distortion_raw, (list, tuple)):
        raise RuntimeError("invalid get_intrinsics dist")
    if len(distortion_raw) != 8:
        raise RuntimeError("get_intrinsics dist must contain 8 coefficients")

    distortion_values: list[float] = []
    for item in distortion_raw:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise RuntimeError("invalid get_intrinsics dist value")
        distortion_values.append(float(item))

    k1, k2, k3, k4, k5, k6, p1, p2 = distortion_values
    return k1, k2, p1, p2, k3, k4, k5, k6
