from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import cv2
from loguru import logger

from .camera_local.config import LocalCameraRuntimeConfig, LocalStreamProfileConfig
from .camera_runtime_protocol import CameraRuntimeProtocol
from .camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig
from .charuco_detection import (
    CharucoDetectionConfig,
    CharucoDetectionResult,
    CharucoDetector,
)
from .protocol import RgbdFrameProtocol
from .stable_frame import StableFrameConfig, StableFrameDetector

# region 数据结构


@dataclass(frozen=True, slots=True)
class CameraEndpointConfig:
    """单路远端相机端点配置。

    该对象只描述安装位、控制标识、数据端口与现场连接状态。
    它不建立 socket，不持有帧缓存，可作为不可变配置跨线程只读复用。
    类不继承业务基类，由 `PipelineContext` 将已连接端点组装为运行时。
    """

    camera_name: str
    "项目内逻辑相机名，例如 `head_camera`。"

    camera_id: str
    "上游控制口相机标识，例如 `HEAD`。"

    stream_port: int
    "上游 ZMQ 数据流端口，单位 TCP 端口号。"

    stable_frame_config: StableFrameConfig = StableFrameConfig()
    "按安装位实测标定的稳定帧判定配置。"

    connected: bool = True
    "现场是否已连接该相机；为 `False` 时保留 API 但不启动运行时。"

    serial_number: str = ""
    "本机 USB 模式使用的 Orbbec 设备 SN；ZMQ 模式忽略。"

    color_profile: LocalStreamProfileConfig = LocalStreamProfileConfig(
        1280, 720, 30, "MJPG"
    )
    "本机 USB 彩色流 profile。"

    depth_profile: LocalStreamProfileConfig = LocalStreamProfileConfig(
        848, 480, 30, "Y16"
    )
    "本机 USB 深度流 profile。"


DEFAULT_CAMERA_ENDPOINTS: tuple[CameraEndpointConfig, ...] = (
    CameraEndpointConfig(
        "head_camera",
        "HEAD",
        5560,
        stable_frame_config=StableFrameConfig(
            depth_median_delta_threshold_mm=8.0,
            depth_percentile_delta_threshold_mm=25.0,
        ),
    ),
    CameraEndpointConfig(
        "chest_camera",
        "CHEST",
        5561,
        stable_frame_config=StableFrameConfig(
            depth_median_delta_threshold_mm=6.0,
            depth_percentile_delta_threshold_mm=35.0,
        ),
    ),
    CameraEndpointConfig(
        "left_hand_camera",
        "LEFT",
        5562,
        stable_frame_config=StableFrameConfig(
            depth_median_delta_threshold_mm=40.0,
            depth_percentile_delta_threshold_mm=160.0,
        ),
    ),
    CameraEndpointConfig("right_hand_camera", "RIGHT", 5563, connected=False),
)
"现场四个相机安装位与上游数据端口映射，右臂安装位当前未连接。"


@dataclass(frozen=True, slots=True)
class PipelineContextConfig:
    """总流程上下文配置。

    职责边界：
    - 只保存相机流运行时所需配置。
    - 不负责建立连接，不负责缓存帧，不负责执行算法。

    设计思想：
    - 将相机访问参数收口到上下文层，由上下文统一组装底层 runtime。
    - 让服务入口只处理命令行与配置映射，不直接拼接相机访问细节。

    生命周期：
    - 作为纯配置对象可跨线程传递。
    - 不持有硬件、socket 或队列资源。

    继承关系：
    - 不继承业务基类，仅作为上下文配置数据结构。
    """

    camera_source_mode: Literal["zmq", "usb"] = "zmq"
    "相机输入模式；默认 `zmq`，本机 USB 直连时使用 `usb`。"

    camera_host: str = "192.168.100.60"
    "ZMQ 相机服务主机地址。"

    camera_control_port: int = 5570
    "相机控制口端口号，单位 端口号。"

    camera_stream_port: int = 5562
    "相机数据流端口号，单位 端口号。"

    camera_id: str = "LEFT"
    "远端相机控制标识。"

    camera_name: str = "left_hand_camera"
    "项目内逻辑相机名。"

    camera_request_timeout_ms: int = 3000
    "相机控制命令超时，单位 ms。"

    camera_stream_timeout_ms: int = 8000
    "相机数据流接收超时，单位 ms。"

    camera_frame_cache_size: int = 64
    "最近相机帧缓存数量，需要覆盖稳定时间窗中点帧的回取。"

    camera_endpoints: tuple[CameraEndpointConfig, ...] = DEFAULT_CAMERA_ENDPOINTS
    "所有相机安装位配置；主相机对应项由上方单路参数覆盖。"

    usb_frame_timeout_ms: int = 2000
    "USB 模式单次等待帧组超时，单位 ms。"

    usb_reconnect_initial_interval_s: float = 5.0
    "USB 设备断线或首次连接失败后的重试间隔，单位 s。"

    usb_reconnect_max_interval_s: float = 60.0
    "USB 连续连接失败时指数退避的最大间隔，单位 s。"


class PipelineContext:
    """统一管理相机流和帧数据输入输出的上下文。

    职责边界：
    - 负责根据配置组装相机运行时，并提供统一的帧访问入口。
    - 不负责算法本身，不负责服务协议细节，不负责 GUI 展示。

    设计思想：
    - 把相机访问参数固定在上下文层，避免上层脚本直接拼 runtime 配置。
    - 由上下文统一协调启动、等待就绪、关闭资源，减少调用方重复代码。

    生命周期：
    - 可随服务进程长期存在。
    - 持有相机运行时，必须显式调用 `close()` 释放。

    继承关系：
    - 不继承业务基类。
    """

    def __init__(self, config: PipelineContextConfig) -> None:
        self._config = config
        self._camera_endpoints = self._resolve_camera_endpoints(config)
        self._frame_runtimes: dict[str, CameraRuntimeProtocol] = {
            endpoint.camera_name: self._build_camera_runtime(endpoint)
            for endpoint in self._camera_endpoints
            if endpoint.connected
        }

    def start(self) -> None:
        """启动已连接相机的流运行时。"""

        for runtime in self._frame_runtimes.values():
            runtime.start()

    def close(self) -> None:
        """关闭所有已启动的相机流运行时。"""

        for runtime in self._frame_runtimes.values():
            runtime.stop()

    def wait_until_ready(
        self,
        timeout_s: float = 8.0,
        camera_name: str | None = None,
    ) -> bool:
        """等待指定相机首帧就绪。"""

        return self.get_camera_runtime(camera_name).wait_until_ready(timeout_s=timeout_s)

    def get_camera_runtime(
        self,
        camera_name: str | None = None,
    ) -> CameraRuntimeProtocol:
        """返回指定相机运行时，未指定时返回主相机运行时。"""

        resolved_name = self._config.camera_name if camera_name is None else camera_name
        runtime = self._frame_runtimes.get(resolved_name)
        if runtime is not None:
            return runtime
        endpoint = self._find_camera_endpoint(resolved_name)
        if not endpoint.connected:
            raise RuntimeError(f"camera {resolved_name} is configured but not connected")
        raise RuntimeError(f"camera runtime {resolved_name} is unavailable")

    def get_camera_id(self, camera_name: str | None = None) -> str:
        """返回指定相机控制标识。"""

        resolved_name = self._config.camera_name if camera_name is None else camera_name
        return self._find_camera_endpoint(resolved_name).camera_id

    def get_camera_name(self) -> str:
        """返回默认算法相机逻辑名称。"""

        return self._config.camera_name

    def get_connected_camera_names(self) -> tuple[str, ...]:
        """返回当前已组装运行时的相机名。"""

        return tuple(self._frame_runtimes)

    def get_latest_frame(
        self,
        camera_name: str | None = None,
    ) -> RgbdFrameProtocol | None:
        """返回指定相机最新缓存帧；尚未收到首帧时返回 `None`。"""

        return self.get_camera_runtime(camera_name).get_latest_frame()

    def get_frame_by_id(
        self,
        frame_id: int,
        camera_name: str | None = None,
    ) -> RgbdFrameProtocol | None:
        """按相机和帧号查询缓存帧；帧不存在或已淘汰时返回 `None`。"""

        return self.get_camera_runtime(camera_name).get_frame_by_id(frame_id)

    def resolve_frame(self, frame_id: int) -> RgbdFrameProtocol:
        """按请求帧号选择相机帧，未指定帧号时默认等待稳定帧。"""

        if frame_id > 0:
            frame = self.get_frame_by_id(frame_id)
            if frame is not None:
                return frame
            raise RuntimeError(f"camera frame {frame_id} is not available")
        return self.wait_for_stable_frame()

    def wait_for_stable_frame(
        self,
        timeout_s: float = 10.0,
        camera_name: str | None = None,
    ) -> RgbdFrameProtocol:
        """等待指定相机画面连续稳定，并返回稳定时间窗中点帧。

        Parameters
        ----------
        timeout_s:
            等待稳定帧的最长时间，单位 s。
        camera_name:
            逻辑相机名；未指定时使用默认左臂相机。

        Returns
        -------
        RgbdFrameProtocol
            稳定时间窗中点附近的缓存相机帧。

        Raises
        ------
        ValueError
            `timeout_s` 不大于零。
        RuntimeError
            超时前没有形成稳定时间窗，或中点帧已无法从缓存取回。
        """

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be greater than zero")

        resolved_name = self._config.camera_name if camera_name is None else camera_name
        logger.info(
            "stable frame wait started camera_name={} timeout_s={:.3f}",
            resolved_name,
            timeout_s,
        )
        endpoint = self._find_camera_endpoint(resolved_name)
        detector = StableFrameDetector(config=endpoint.stable_frame_config)
        deadline = time.monotonic() + timeout_s
        last_frame_id = -1
        missing_stable_frame_id: int | None = None
        while time.monotonic() < deadline:
            frame = self.get_latest_frame(resolved_name)
            if frame is None or frame.frame_id == last_frame_id:
                time.sleep(0.01)
                continue

            last_frame_id = frame.frame_id
            stable_frame_id = detector.update(frame)
            if stable_frame_id is None:
                continue
            if self._config.camera_source_mode == "usb":
                logger.info(
                    "stable frame detected in current-frame mode camera_name={} frame_id={} evidence_frame_id={}",
                    resolved_name,
                    frame.frame_id,
                    frame.frame_id,
                )
                return frame
            stable_frame = self.get_frame_by_id(stable_frame_id, resolved_name)
            if stable_frame is not None:
                logger.info(
                    "stable frame detected camera_name={} frame_id={} evidence_frame_id={}",
                    resolved_name,
                    stable_frame.frame_id,
                    frame.frame_id,
                )
                return stable_frame
            missing_stable_frame_id = stable_frame_id

        if missing_stable_frame_id is not None:
            logger.warning(
                "stable frame evicted before retrieval camera_name={} frame_id={}",
                resolved_name,
                missing_stable_frame_id,
            )
            raise RuntimeError(
                f"stable frame {missing_stable_frame_id} is no longer available in camera frame cache"
            )
        logger.warning(
            "stable frame wait timed out camera_name={} timeout_s={:.3f} last_frame_id={}",
            resolved_name,
            timeout_s,
            last_frame_id,
        )
        raise RuntimeError(f"camera did not become stable within {timeout_s:.1f}s")

    def detect_charuco(
        self,
        board: cv2.aruco.CharucoBoard,
        *,
        camera_name: str | None = None,
        config: CharucoDetectionConfig | None = None,
        enable_debug: bool = False,
        max_frames: int = 5,
        stable_timeout_s: float = 10.0,
    ) -> CharucoDetectionResult:
        """使用连续稳定帧检测 ChArUco 标定板。

        Parameters
        ----------
        board:
            原生 OpenCV ChArUco 板对象。方格和 marker 长度统一使用 mm。
        camera_name:
            逻辑相机名；未指定时使用默认算法相机。
        config:
            单帧 ChArUco 检测和图像增强配置；为空时使用默认配置。
        enable_debug:
            是否为每次检测构造 marker、ChArUco 和 pose 叠加图。仅最终结果返回。
        max_frames:
            单次调用允许尝试的稳定帧数量，单位 帧，默认 5。
        stable_timeout_s:
            每次等待下一稳定帧的最长时间，单位 s。

        Returns
        -------
        CharucoDetectionResult
            首个有效位姿结果；达到帧数上限仍失败时返回最后一帧的 `missing` 结果。

        Raises
        ------
        ValueError
            帧数上限或稳定帧超时不大于零。
        RuntimeError
            相机不可用、稳定等待超时或稳定帧已被缓存淘汰。

        Notes
        -----
        本方法只负责稳定帧获取与重试编排。单帧预处理、融合和 PnP 全部由
        `CharucoDetector` 完成。每次重新调用 `wait_for_stable_frame` 都创建新的稳定
        时间窗，因此失败后输入的是后续稳定帧，不重复使用同一证据帧。
        """

        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        if stable_timeout_s <= 0.0:
            raise ValueError("stable_timeout_s must be greater than zero")

        detector = CharucoDetector(board=board, config=config)
        last_result: CharucoDetectionResult | None = None
        resolved_name = self._config.camera_name if camera_name is None else camera_name
        logger.info(
            "charuco detection started camera_name={} max_frames={} stable_timeout_s={:.3f} debug_enabled={}",
            resolved_name,
            max_frames,
            stable_timeout_s,
            enable_debug,
        )
        for attempt_index in range(1, max_frames + 1):
            frame = self.wait_for_stable_frame(
                timeout_s=stable_timeout_s,
                camera_name=camera_name,
            )
            last_result = detector.detect(frame, enable_debug=enable_debug)
            logger.info(
                "charuco detection attempt camera_name={} attempt={}/{} frame_id={} status={} marker_count={} charuco_count={} error_px={:.6f}",
                resolved_name,
                attempt_index,
                max_frames,
                frame.frame_id,
                last_result.status,
                last_result.marker_num,
                last_result.charuco_num,
                last_result.error_px,
            )
            if last_result.status == "detected":
                logger.info(
                    "charuco detection completed camera_name={} frame_id={} attempts={} status=detected",
                    resolved_name,
                    frame.frame_id,
                    attempt_index,
                )
                return last_result

        if last_result is None:
            raise RuntimeError("charuco detection did not receive a stable frame")
        logger.warning(
            "charuco detection completed camera_name={} attempts={} status=missing marker_count={} charuco_count={}",
            resolved_name,
            max_frames,
            last_result.marker_num,
            last_result.charuco_num,
        )
        return last_result

    # region 相机配置

    @staticmethod
    def _resolve_camera_endpoints(
        config: PipelineContextConfig,
    ) -> tuple[CameraEndpointConfig, ...]:
        """合并多相机端点表与默认算法相机覆盖参数。"""

        resolved: list[CameraEndpointConfig] = []
        primary_found = False
        for endpoint in config.camera_endpoints:
            if endpoint.camera_name != config.camera_name:
                resolved.append(endpoint)
                continue
            primary_found = True
            resolved.append(
                CameraEndpointConfig(
                    camera_name=config.camera_name,
                    camera_id=config.camera_id,
                    stream_port=config.camera_stream_port,
                    stable_frame_config=endpoint.stable_frame_config,
                    connected=True,
                    serial_number=endpoint.serial_number,
                    color_profile=endpoint.color_profile,
                    depth_profile=endpoint.depth_profile,
                )
            )
        if not primary_found:
            raise ValueError(
                f"primary camera {config.camera_name} is missing from camera endpoints"
            )
        return tuple(resolved)

    def _find_camera_endpoint(self, camera_name: str) -> CameraEndpointConfig:
        """查找指定逻辑相机的端点配置。"""

        for endpoint in self._camera_endpoints:
            if endpoint.camera_name == camera_name:
                return endpoint
        raise ValueError(f"unsupported camera: {camera_name}")

    def _build_camera_runtime(
        self, endpoint: CameraEndpointConfig
    ) -> CameraRuntimeProtocol:
        """按上下文模式为一个安装位构造采集运行时。"""

        if self._config.camera_source_mode == "zmq":
            return CameraStreamRuntime(
                CameraStreamRuntimeConfig(
                    host=self._config.camera_host,
                    control_port=self._config.camera_control_port,
                    stream_port=endpoint.stream_port,
                    camera_id=endpoint.camera_id,
                    camera_name=endpoint.camera_name,
                    request_timeout_ms=self._config.camera_request_timeout_ms,
                    stream_timeout_ms=self._config.camera_stream_timeout_ms,
                    cache_size=self._config.camera_frame_cache_size,
                )
            )
        if self._config.camera_source_mode == "usb":
            # 仅 USB 模式加载 pyorbbecsdk，默认 ZMQ 部署不承担本机 SDK 依赖。
            from .camera_local.runtime import LocalCameraRuntime

            return LocalCameraRuntime(
                LocalCameraRuntimeConfig(
                    camera_name=endpoint.camera_name,
                    camera_id=endpoint.camera_id,
                    serial_number=endpoint.serial_number,
                    color=endpoint.color_profile,
                    depth=endpoint.depth_profile,
                    frame_timeout_ms=self._config.usb_frame_timeout_ms,
                    reconnect_initial_interval_s=self._config.usb_reconnect_initial_interval_s,
                    reconnect_max_interval_s=self._config.usb_reconnect_max_interval_s,
                )
            )
        raise ValueError(f"unsupported camera source mode: {self._config.camera_source_mode}")

    # endregion


# endregion
