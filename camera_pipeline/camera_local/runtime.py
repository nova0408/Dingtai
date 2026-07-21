from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from loguru import logger
from pyorbbecsdk import (
    Config,
    Context,
    DepthFrame,
    Device,
    DeviceList,
    OBAlignMode,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    Pipeline,
    VideoFrame,
    VideoStreamProfile,
)

from ..protocol import CameraFramePacket
from .config import LocalCameraRuntimeConfig, LocalStreamProfileConfig

# pyright: reportMissingImports=false


_FORMAT_BY_NAME: dict[str, OBFormat] = {
    "MJPG": OBFormat.MJPG,
    "RGB": OBFormat.RGB,
    "BGR": OBFormat.BGR,
    "YUYV": OBFormat.YUYV,
    "Y16": OBFormat.Y16,
    "Z16": OBFormat.Z16,
}


# region USB 运行时


class LocalCameraRuntime:
    """持续采集单台本机 USB Orbbec 相机的运行时。

    职责边界：
    - 按 SN 枚举设备，按配置启动 RGBD 流并读取彩色相机标定参数。
    - 后台持续获取当前帧，断线或 SDK 异常后释放 pipeline 并自动重连。
    - 只持有最新帧，不维护历史缓存，不负责算法和网络传输。

    生命周期与线程语义：
    - `start()` 创建一个守护采集线程，线程独占所有 pyorbbecsdk 资源。
    - 调用线程仅在锁内读取最新不可变 packet；`stop()` 请求退出并等待资源释放。
    - 类不继承业务基类，通过 `CameraRuntimeProtocol` 参与上下文编排。
    """

    def __init__(self, config: LocalCameraRuntimeConfig) -> None:
        if config.reconnect_initial_interval_s <= 0.0:
            raise ValueError("reconnect_initial_interval_s must be greater than zero")
        if config.reconnect_max_interval_s < config.reconnect_initial_interval_s:
            raise ValueError(
                "reconnect_max_interval_s must be greater than or equal to " "reconnect_initial_interval_s"
            )
        self._config = config
        self._lock = threading.Lock()
        self._latest_frame: CameraFramePacket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_id = 0
        self._session_received_frame = False

    @property
    def keeps_frame_history(self) -> bool:
        """表示 USB 模式只保存最新帧。"""

        return False

    def start(self) -> None:
        """启动长期采集与自动重连线程。"""

        if self._thread is not None:
            logger.warning(
                "usb camera start ignored because runtime is already running camera_name={}",
                self._config.camera_name,
            )
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"usb-camera-{self._config.camera_id.lower()}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "usb camera runtime started camera_name={} camera_id={} serial_number={}",
            self._config.camera_name,
            self._config.camera_id,
            self._config.serial_number or "<empty>",
        )

    def stop(self) -> None:
        """停止采集线程；SDK pipeline 由采集线程的 finally 块释放。"""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(3.0, self._config.frame_timeout_ms / 1000.0 + 1.0))
            self._thread = None
        with self._lock:
            self._latest_frame = None
        logger.info("usb camera runtime stopped camera_name={}", self._config.camera_name)

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        """等待本轮连接产生第一帧。

        Parameters
        ----------
        timeout_s:
            最长等待时间，单位 s。

        Returns
        -------
        bool
            `True` 表示当前已有有效帧，`False` 表示超时。
        """

        deadline = time.monotonic() + max(0.1, timeout_s)
        while time.monotonic() < deadline:
            if self.get_latest_frame() is not None:
                return True
            self._stop_event.wait(0.05)
        return False

    def get_latest_frame(self) -> CameraFramePacket | None:
        """返回当前最新帧；尚未成功采集时返回 `None`。"""

        with self._lock:
            return self._latest_frame

    def get_frame_by_id(self, frame_id: int) -> CameraFramePacket | None:
        """仅在请求帧仍是当前帧时返回，绝不回取历史帧。"""

        with self._lock:
            if self._latest_frame is None or self._latest_frame.frame_id != frame_id:
                return None
            return self._latest_frame

    def _run(self) -> None:
        """循环执行连接、采集、释放和低频指数退避重试。"""

        if not self._config.serial_number:
            logger.warning(
                "usb camera serial number is empty; runtime remains idle camera_name={} camera_id={}",
                self._config.camera_name,
                self._config.camera_id,
            )
            # 配置在进程生命周期内不可变，空 SN 重复枚举不会产生有效结果。
            self._stop_event.wait()
            return

        consecutive_failures = 0
        while not self._stop_event.is_set():
            self._session_received_frame = False
            try:
                self._capture_session()
            except Exception:
                with self._lock:
                    self._latest_frame = None
                if self._session_received_frame:
                    consecutive_failures = 1
                else:
                    consecutive_failures += 1
                retry_interval_s = _calculate_retry_interval_s(
                    consecutive_failures,
                    self._config.reconnect_initial_interval_s,
                    self._config.reconnect_max_interval_s,
                )
                logger.exception(
                    "usb camera session failed; reconnect scheduled camera_name={} camera_id={} serial_number={} retry_s={:.3f}",
                    self._config.camera_name,
                    self._config.camera_id,
                    self._config.serial_number,
                    retry_interval_s,
                )
                self._stop_event.wait(retry_interval_s)

    def _capture_session(self) -> None:
        """建立一次 SDK pipeline 会话并持续读取帧，直到停止或发生错误。"""

        context = Context()
        device: Device = self._find_device(context)
        pipeline = Pipeline(device)
        stream_config, color_profile = self._build_stream_config(pipeline)
        started = False
        try:
            pipeline.start(stream_config)
            started = True
            intrinsic = color_profile.get_intrinsic()
            distortion = color_profile.get_distortion()
            distortion_values = (
                float(distortion.k1),
                float(distortion.k2),
                float(distortion.p1),
                float(distortion.p2),
                float(distortion.k3),
                float(distortion.k4),
                float(distortion.k5),
                float(distortion.k6),
            )
            logger.info(
                "usb camera connected camera_name={} serial_number={} color={}x{}@{} {} depth={}x{}@{} {}",
                self._config.camera_name,
                self._config.serial_number,
                self._config.color.width,
                self._config.color.height,
                self._config.color.fps,
                self._config.color.format_name,
                self._config.depth.width,
                self._config.depth.height,
                self._config.depth.fps,
                self._config.depth.format_name,
            )
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(self._config.frame_timeout_ms)
                if frames is None:
                    raise RuntimeError("USB camera frame wait timed out")
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    logger.warning(
                        "usb camera incomplete frameset camera_name={} color={} depth={}",
                        self._config.camera_name,
                        color_frame is not None,
                        depth_frame is not None,
                    )
                    continue
                color_bgr = _color_frame_to_bgr(color_frame)
                depth_mm = _depth_frame_to_mm(depth_frame)
                self._frame_id += 1
                packet = CameraFramePacket(
                    frame_id=self._frame_id,
                    camera_name=self._config.camera_name,
                    timestamp_ms=float(frames.get_timestamp_us()) / 1000.0,
                    color_bgr=color_bgr,
                    depth_mm=depth_mm,
                    fx=float(intrinsic.fx),
                    fy=float(intrinsic.fy),
                    cx=float(intrinsic.cx),
                    cy=float(intrinsic.cy),
                    distortion=distortion_values,
                )
                with self._lock:
                    self._latest_frame = packet
                self._session_received_frame = True
        finally:
            if started:
                pipeline.stop()
                logger.info(
                    "usb camera disconnected camera_name={} serial_number={}",
                    self._config.camera_name,
                    self._config.serial_number,
                )

    def _find_device(self, context: Context) -> Device:
        """按配置 SN 查找设备，不使用枚举顺序代替身份。"""

        device_list: DeviceList = context.query_devices()
        for index in range(device_list.get_count()):
            device: Device = device_list.get_device_by_index(index)
            if device.get_device_info().get_serial_number() == self._config.serial_number:
                return device
        raise RuntimeError(f"USB camera not found: {self._config.serial_number}")

    def _build_stream_config(self, pipeline: Pipeline) -> tuple[Config, VideoStreamProfile]:
        """按配置精确选择彩色和深度 profile，并启用 D2C 软件对齐。"""

        color_profile = self._select_profile(pipeline, OBSensorType.COLOR_SENSOR, self._config.color)
        depth_profile = self._select_profile(pipeline, OBSensorType.DEPTH_SENSOR, self._config.depth)
        config = Config()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_align_mode(OBAlignMode.SW_MODE)
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        return config, color_profile

    @staticmethod
    def _select_profile(
        pipeline: Pipeline,
        sensor_type: OBSensorType,
        profile_config: LocalStreamProfileConfig,
    ) -> VideoStreamProfile:
        """从设备 profile 列表精确选择一项配置。"""

        format_value = _FORMAT_BY_NAME.get(profile_config.format_name)
        if format_value is None:
            raise ValueError(f"unsupported USB stream format: {profile_config.format_name}")
        profiles = pipeline.get_stream_profile_list(sensor_type)
        return profiles.get_video_stream_profile(
            profile_config.width,
            profile_config.height,
            format_value,
            profile_config.fps,
        )


# endregion


# region 帧转换


def _calculate_retry_interval_s(
    consecutive_failures: int,
    initial_interval_s: float,
    max_interval_s: float,
) -> float:
    """计算有上限的 2 倍指数退避间隔，避免构造过大的指数值。"""

    retry_interval_s = initial_interval_s
    for _ in range(max(0, consecutive_failures - 1)):
        retry_interval_s = min(retry_interval_s * 2.0, max_interval_s)
        if retry_interval_s >= max_interval_s:
            break
    return retry_interval_s


def _color_frame_to_bgr(frame: VideoFrame) -> np.ndarray:
    """将 SDK 彩色帧转换成 `(H, W, 3)` 的 `uint8` BGR 图像。"""

    width = frame.get_width()
    height = frame.get_height()
    data = np.asarray(frame.get_data())
    frame_format = frame.get_format()
    if frame_format == OBFormat.MJPG:
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("USB camera MJPG decode failed")
        return np.asarray(decoded, dtype=np.uint8)
    if frame_format == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if frame_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3)).copy()
    if frame_format == OBFormat.YUYV:
        yuyv = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    raise RuntimeError(f"unsupported USB color frame format: {frame_format}")


def _depth_frame_to_mm(frame: DepthFrame) -> np.ndarray:
    """将 SDK 深度帧转换成 `(H, W)` 的 `uint16` 毫米深度图。"""

    depth_raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((frame.get_height(), frame.get_width()))
    # SDK 原始深度乘 depth scale 后单位为 mm；饱和裁剪后收窄为协议要求的 uint16。
    depth_mm = depth_raw.astype(np.float32) * frame.get_depth_scale()
    return np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)


# endregion
