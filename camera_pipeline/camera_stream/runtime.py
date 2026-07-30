from __future__ import annotations

# pyright: reportMissingImports=false

import json
import math
import queue
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

from ..protocol import CameraColorFramePacket, CameraFramePacket

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

    stale_frame_timeout_s: float = 3.0
    "帧号或时间戳持续不递增时触发流恢复的超时，单位 s。"

    cache_size: int = 16
    "按帧号缓存最近帧数量。"

    max_consecutive_timeouts: int = 3
    "连续收流超时次数上限；超过后触发自愈。"

    recover_retry_interval_s: float = 2.0
    "上游控制失败后的初始重试间隔，单位 s。"

    recover_retry_max_interval_s: float = 30.0
    "上游控制连续失败时指数退避的最大间隔，单位 s。"

    max_consecutive_color_only_frames: int = 30
    "连续无深度帧数量上限；达到后重新请求上游开启深度。"


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
        if self._config.recover_retry_interval_s <= 0.0:
            raise ValueError("recover_retry_interval_s must be greater than zero")
        if (
            self._config.recover_retry_max_interval_s
            < self._config.recover_retry_interval_s
        ):
            raise ValueError(
                "recover_retry_max_interval_s must be greater than or equal to "
                "recover_retry_interval_s"
            )
        if self._config.max_consecutive_color_only_frames <= 0:
            raise ValueError(
                "max_consecutive_color_only_frames must be greater than zero"
            )
        if self._config.stale_frame_timeout_s <= 0.0:
            raise ValueError("stale_frame_timeout_s must be greater than zero")
        self._context = zmq.Context()
        self._control_socket = self._create_control_socket()
        self._stream_socket: zmq.Socket | None = None
        self._lock = threading.Lock()
        self._latest_frame: CameraFramePacket | None = None
        self._frame_cache: dict[int, CameraFramePacket] = {}
        self._frame_order: queue.Queue[int] = queue.Queue(
            maxsize=max(1, int(self._config.cache_size))
        )
        self._latest_color_frame: CameraColorFramePacket | None = None
        self._color_frame_cache: dict[int, CameraColorFramePacket] = {}
        self._color_frame_order: queue.Queue[int] = queue.Queue(
            maxsize=max(1, int(self._config.cache_size))
        )
        self._running = False
        self._thread: threading.Thread | None = None
        self._cached_intrinsics: tuple[
            float, float, float, float, tuple[float, ...]
        ] | None = None
        self._last_recover_time = 0.0
        self._depth_stream_confirmed = False
        self._depth_reset_required = False
        self._consecutive_color_only_frames = 0
        self._control_retry_failures = 0
        self._next_control_retry_at = 0.0
        self._last_stream_frame_id: int | None = None
        self._last_stream_timestamp_ms: float | None = None
        self._last_stream_progress_at = time.perf_counter()

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
        self._stream_socket = self._create_stream_socket()
        self._depth_stream_confirmed = False
        self._depth_reset_required = False
        self._consecutive_color_only_frames = 0
        self._control_retry_failures = 0
        self._next_control_retry_at = 0.0
        self._reset_stream_progress()
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

    def get_latest_color_frame(self) -> CameraColorFramePacket | None:
        """返回最新彩色帧，不要求同一消息包含深度载荷。"""

        with self._lock:
            return self._latest_color_frame

    def get_color_frame_by_id(
        self,
        frame_id: int,
    ) -> CameraColorFramePacket | None:
        """按帧号查询彩色帧缓存。"""

        with self._lock:
            return self._color_frame_cache.get(int(frame_id))

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        """等待第一帧就绪。"""

        deadline = time.perf_counter() + max(0.1, float(timeout_s))
        while time.perf_counter() < deadline:
            if self.get_latest_frame() is not None:
                return True
            time.sleep(0.05)
        return False

    def _capture_loop(self) -> None:
        """持续接收相机帧，并在上游任意启动顺序下收敛到 RGBD 可用状态。

        Notes
        -----
        采集线程同时持有控制 REQ 与数据 SUB socket。控制服务尚未启动时使用指数
        退避持续重试；数据服务尚未启动时由 ZMQ SUB 自动重连，并由接收超时路径
        重建本地 socket。只要进程未停止，启动阶段的暂时失败不会成为永久状态。
        """

        consecutive_timeouts = 0
        while self._running:
            if not self._maintain_upstream_control():
                time.sleep(0.05)
                continue
            if self._stream_socket is None:
                self._recover_stream_runtime("stream socket missing")
                time.sleep(0.05)
                continue
            try:
                raw_message = self._stream_socket.recv()
                color_packet, rgbd_packet = self._decode_frame(raw_message)
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
            if not self._accept_stream_progress(color_packet):
                continue
            with self._lock:
                self._latest_color_frame = color_packet
                self._color_frame_cache[int(color_packet.frame_id)] = color_packet
                if self._color_frame_order.full():
                    try:
                        expired = self._color_frame_order.get_nowait()
                        self._color_frame_cache.pop(int(expired), None)
                    except queue.Empty:
                        pass
                self._color_frame_order.put_nowait(int(color_packet.frame_id))
                if rgbd_packet is None:
                    self._record_color_only_frame()
                    continue
                self._record_rgbd_frame()
                self._latest_frame = rgbd_packet
                self._frame_cache[int(rgbd_packet.frame_id)] = rgbd_packet
                if self._frame_order.full():
                    try:
                        expired = self._frame_order.get_nowait()
                        self._frame_cache.pop(int(expired), None)
                    except queue.Empty:
                        pass
                self._frame_order.put_nowait(int(rgbd_packet.frame_id))

    def _maintain_upstream_control(self) -> bool:
        """按退避计划恢复深度开关和相机内参。

        Returns
        -------
        bool
            `True` 表示已有内参，可继续解码数据流；`False` 表示当前只能等待下一次
            控制重试。深度是否真正恢复由收到 RGBD 帧确认，不能仅相信控制响应。
        """

        needs_depth = not self._depth_stream_confirmed
        reset_depth = needs_depth and self._depth_reset_required
        if reset_depth:
            # 深度流与控制状态不一致时，上游相机可能已经重新初始化。旧内参不能
            # 继续随恢复后的帧使用，必须与深度开关一起重新读取。
            self._cached_intrinsics = None
        needs_intrinsics = self._cached_intrinsics is None
        if not needs_depth and not needs_intrinsics:
            return True

        now = time.monotonic()
        if now < self._next_control_retry_at:
            return not needs_intrinsics
        try:
            if needs_depth:
                if reset_depth:
                    self._send_control_command(
                        "set_depth_enabled",
                        {"enable": False},
                    )
                self._send_control_command("set_depth_enabled", {"enable": True})
                self._depth_reset_required = False
            if needs_intrinsics:
                self._cached_intrinsics = self._get_intrinsics_from_control()
        except Exception as exc:  # noqa: BLE001
            self._control_retry_failures += 1
            retry_interval_s = _calculate_retry_interval_s(
                self._control_retry_failures,
                self._config.recover_retry_interval_s,
                self._config.recover_retry_max_interval_s,
            )
            self._next_control_retry_at = now + retry_interval_s
            logger.warning(
                "camera upstream control unavailable; retry scheduled "
                "camera_name={} failures={} retry_s={:.3f} error={}",
                self._config.camera_name,
                self._control_retry_failures,
                retry_interval_s,
                exc,
            )
            return self._cached_intrinsics is not None

        if self._control_retry_failures > 0:
            logger.success(
                "camera upstream control recovered camera_name={} failures={}",
                self._config.camera_name,
                self._control_retry_failures,
            )
        self._control_retry_failures = 0
        self._next_control_retry_at = now + self._config.recover_retry_interval_s
        return self._cached_intrinsics is not None

    def _record_color_only_frame(self) -> None:
        """记录无深度帧，并在持续退化时重新进入深度恢复状态。"""

        self._consecutive_color_only_frames += 1
        threshold = self._config.max_consecutive_color_only_frames
        if self._consecutive_color_only_frames < threshold:
            return
        if self._depth_stream_confirmed:
            logger.warning(
                "camera depth stream lost; control reset scheduled "
                "camera_name={} color_only_frames={}",
                self._config.camera_name,
                self._consecutive_color_only_frames,
            )
            self._depth_stream_confirmed = False
        if self._consecutive_color_only_frames % threshold != 0:
            return
        logger.warning(
            "camera depth stream still missing; disable-enable reset scheduled "
            "camera_name={} color_only_frames={}",
            self._config.camera_name,
            self._consecutive_color_only_frames,
        )
        self._depth_reset_required = True
        self._next_control_retry_at = 0.0

    def _record_rgbd_frame(self) -> None:
        """以实际 RGBD 帧确认深度恢复，并清空连续退化计数。"""

        if not self._depth_stream_confirmed:
            logger.success(
                "camera depth stream confirmed camera_name={}",
                self._config.camera_name,
            )
        self._depth_stream_confirmed = True
        self._depth_reset_required = False
        self._consecutive_color_only_frames = 0
        self._control_retry_failures = 0

    def _accept_stream_progress(self, frame: CameraColorFramePacket) -> bool:
        """仅接受帧号与时间戳都向前推进的数据包，并恢复持续陈旧的流。"""

        now = time.perf_counter()
        frame_id = frame.frame_id
        timestamp_ms = frame.timestamp_ms
        if (
            self._last_stream_frame_id is None
            or self._last_stream_timestamp_ms is None
        ):
            self._last_stream_frame_id = frame_id
            self._last_stream_timestamp_ms = timestamp_ms
            self._last_stream_progress_at = now
            return True
        if (
            frame_id > self._last_stream_frame_id
            and timestamp_ms > self._last_stream_timestamp_ms
        ):
            self._last_stream_frame_id = frame_id
            self._last_stream_timestamp_ms = timestamp_ms
            self._last_stream_progress_at = now
            return True

        stale_duration_s = now - self._last_stream_progress_at
        if stale_duration_s < self._config.stale_frame_timeout_s:
            return False

        previous_frame_id = self._last_stream_frame_id
        previous_timestamp_ms = self._last_stream_timestamp_ms
        self._last_stream_progress_at = now
        self._recover_stream_runtime(
            "frame identity did not advance "
            f"for {stale_duration_s:.3f}s "
            f"previous_frame_id={previous_frame_id} frame_id={frame_id} "
            f"previous_timestamp_ms={previous_timestamp_ms:.3f} "
            f"timestamp_ms={timestamp_ms:.3f}"
        )
        return False

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
        self._depth_stream_confirmed = False
        self._depth_reset_required = False
        self._consecutive_color_only_frames = 0
        self._cached_intrinsics = None
        self._next_control_retry_at = 0.0
        self._clear_cached_frames()
        self._reset_stream_progress()
        self._recreate_stream_socket()

    def _clear_cached_frames(self) -> None:
        """恢复收流前清除旧帧，避免查询接口继续返回陈旧缓存。"""

        with self._lock:
            self._latest_frame = None
            self._frame_cache.clear()
            while not self._frame_order.empty():
                self._frame_order.get_nowait()
            self._latest_color_frame = None
            self._color_frame_cache.clear()
            while not self._color_frame_order.empty():
                self._color_frame_order.get_nowait()

    def _reset_stream_progress(self) -> None:
        """重置帧身份推进状态，让重连后的首帧建立新基线。"""

        self._last_stream_frame_id = None
        self._last_stream_timestamp_ms = None
        self._last_stream_progress_at = time.perf_counter()

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

    def _decode_frame(
        self,
        raw_message: bytes,
    ) -> tuple[CameraColorFramePacket, CameraFramePacket | None]:
        if len(raw_message) < self._FRAME_HEADER_SIZE:
            raise RuntimeError(
                "ZMQ camera frame too short "
                f"actual={len(raw_message)} header={self._FRAME_HEADER_SIZE}"
            )
        if raw_message[:4] != b"ZCAM":
            raise RuntimeError("invalid ZMQ camera frame magic")
        frame_header = self._FRAME_HEADER_STRUCT.unpack(
            raw_message[: self._FRAME_HEADER_SIZE]
        )
        protocol_version = int(frame_header[1])
        depth_format = int(frame_header[4])
        color_data_size = int(frame_header[7])
        depth_width = int(frame_header[8])
        depth_height = int(frame_header[9])
        depth_data_size = int(frame_header[10])
        depth_original_size = int(frame_header[11])
        timestamp_us = int(frame_header[12])
        sequence = int(frame_header[13])
        if protocol_version != 1:
            raise RuntimeError(
                f"unsupported ZMQ camera protocol version {protocol_version}"
            )
        expected_message_size = (
            self._FRAME_HEADER_SIZE + color_data_size + depth_data_size
        )
        if len(raw_message) != expected_message_size:
            raise RuntimeError(
                "ZMQ camera frame size mismatch "
                f"actual={len(raw_message)} expected={expected_message_size} "
                f"color={color_data_size} depth={depth_data_size} "
                f"sequence={sequence}"
            )
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
        fx, fy, cx, cy, distortion = self._get_intrinsics()
        color_packet = CameraColorFramePacket(
            frame_id=sequence,
            camera_name=self._config.camera_name,
            timestamp_ms=timestamp_us / 1000.0,
            color_bgr=color_bgr,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion=distortion,
        )
        if (
            depth_format == 0
            and depth_data_size == 0
            and depth_original_size == 0
        ):
            return color_packet, None
        if depth_format != 1:
            raise RuntimeError(
                "unsupported ZMQ camera depth format "
                f"depth_format={depth_format} sequence={sequence}"
            )
        expected_depth_size = depth_width * depth_height * 2
        if depth_original_size != expected_depth_size:
            raise RuntimeError(
                "ZMQ camera depth size mismatch "
                f"header={depth_original_size} dimensions={expected_depth_size} "
                f"size={depth_width}x{depth_height} sequence={sequence}"
            )
        depth_start = color_end
        depth_end = depth_start + depth_data_size
        depth_bytes = raw_message[depth_start:depth_end]
        depth_raw = lz4.block.decompress(
            depth_bytes, uncompressed_size=depth_original_size
        )
        if len(depth_raw) != depth_original_size:
            raise RuntimeError(
                "ZMQ camera depth decompressed size mismatch "
                f"actual={len(depth_raw)} expected={depth_original_size} "
                f"sequence={sequence}"
            )
        depth_mm = (
            np.frombuffer(depth_raw, dtype=np.uint16)
            .reshape((depth_height, depth_width))
            .copy()
        )
        return color_packet, CameraFramePacket(
            frame_id=sequence,
            camera_name=self._config.camera_name,
            timestamp_ms=timestamp_us / 1000.0,
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion=distortion,
        )

    def _get_intrinsics(
        self,
    ) -> tuple[float, float, float, float, tuple[float, ...]]:
        """返回控制维护流程已经缓存的内参，不在解码路径发起 ZMQ 请求。"""

        if self._cached_intrinsics is None:
            raise RuntimeError("camera intrinsics are not ready")
        return self._cached_intrinsics

    def _get_intrinsics_from_control(
        self,
    ) -> tuple[float, float, float, float, tuple[float, ...]]:
        payload = self._send_control_command("get_intrinsics")
        data = payload.get("data", {})
        if not isinstance(data, dict):
            raise RuntimeError("invalid get_intrinsics payload")
        intrinsics = (
            _read_json_number(data, "fx", 910.0),
            _read_json_number(data, "fy", 910.0),
            _read_json_number(data, "cx", 640.0),
            _read_json_number(data, "cy", 360.0),
            _read_zmq_distortion(data),
        )
        fx, fy, cx, cy, _distortion = intrinsics
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(
                "invalid get_intrinsics focal length: "
                f"fx={fx:.6f} fy={fy:.6f} cx={cx:.6f} cy={cy:.6f}"
            )
        return intrinsics

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
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite camera control numeric field: {key}")
    return number


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
        coefficient = float(item)
        if not math.isfinite(coefficient):
            raise RuntimeError("non-finite get_intrinsics dist value")
        distortion_values.append(coefficient)

    k1, k2, k3, k4, k5, k6, p1, p2 = distortion_values
    return k1, k2, p1, p2, k3, k4, k5, k6


def _calculate_retry_interval_s(
    failure_count: int,
    initial_interval_s: float,
    max_interval_s: float,
) -> float:
    """计算上游控制连续失败后的指数退避时间。

    Parameters
    ----------
    failure_count:
        当前连续失败次数，从 1 开始。
    initial_interval_s:
        第一次失败后的等待时间，单位 s。
    max_interval_s:
        退避等待上限，单位 s。

    Returns
    -------
    float
        本次失败后的等待时间，单位 s。
    """

    retry_interval_s = initial_interval_s
    for _ in range(max(0, failure_count - 1)):
        retry_interval_s = min(retry_interval_s * 2.0, max_interval_s)
        if retry_interval_s >= max_interval_s:
            break
    return retry_interval_s
