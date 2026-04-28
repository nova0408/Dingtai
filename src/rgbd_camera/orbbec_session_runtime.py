from __future__ import annotations

import subprocess
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBCameraParam,
    OBError,
    OBFormat,
    OBFrameType,
    OBFrameAggregateOutputMode,
    OBLogLevel,
    OBSensorType,
    OBStreamType,
    Pipeline,
    PointCloudFilter,
)

from .orbbec_models import OrbbecImuSample, SensorFrustumConfig, SessionOptions
from .orbbec_pointcloud_utils import (
    filter_points_in_sensor_frustum,
    filter_valid_points,
    normalize_points,
    set_point_cloud_filter_format,
    voxel_downsample_points_numpy,
)

if TYPE_CHECKING:
    from pyorbbecsdk import FrameSet


class OrbbecSession:
    """Orbbec 通用会话，负责采集流、对齐和点云计算。"""

    def __init__(
        self,
        options: SessionOptions | None = None,
        sensor_frustum: SensorFrustumConfig | None = None,
    ) -> None:
        self.options = options or SessionOptions()
        self.sensor_frustum = sensor_frustum or self.get_default_sensor_frustum()

        # 关键：不要在 __init__ 里直接触发 Pipeline()。
        # 某些 pyorbbecsdk 版本在无设备时会原生崩溃，Python 层无法捕获。
        self.context: Context | None = None
        self.pipeline: Pipeline | None = None
        self.config: Config | None = None

        self.has_color_sensor = False
        self.has_accel_sensor = False
        self.has_gyro_sensor = False
        self.align_filter: AlignFilter | None = None
        self._depth_stream_fps: float = 0.0
        self._started = False

    @classmethod
    def get_default_sensor_frustum(cls) -> SensorFrustumConfig:
        """
        返回默认传感器视锥参数。

        Returns
        -------
        SensorFrustumConfig
            基础会话使用的默认视锥配置。
        """
        return SensorFrustumConfig()

    def __enter__(self) -> "OrbbecSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        """
        启动相机会话。

        Notes
        -----
        该方法会按 `SessionOptions` 配置启用流、帧同步与对齐过滤器。
        若会话已启动，将直接返回。
        """
        if self._started:
            return

        self._init_runtime_objects()
        self._set_log_level()
        self._configure_streams()

        if self.has_color_sensor and self.options.enable_frame_sync:
            try:
                self.pipeline.enable_frame_sync()  # type: ignore[union-attr]
            except OBError:
                pass

        try:
            self.pipeline.start(self.config)  # type: ignore[union-attr]
        except OBError as exc:
            self._raise_device_runtime_error(
                stage="pipeline.start",
                exc=exc,
                extra_hint="可能是设备未连接、被其他进程占用，或 USB 链路异常。",
            )
        if self.has_color_sensor:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        self._started = True

    def stop(self) -> None:
        """
        停止相机会话并释放运行态资源。

        Notes
        -----
        若会话未启动，该方法为幂等空操作。
        """
        if not self._started:
            return
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        finally:
            self._started = False
            self.align_filter = None

    def wait_for_frames(self, timeout_ms: int | None = None) -> "FrameSet | None":
        """
        等待一组帧数据。

        Parameters
        ----------
        timeout_ms : int | None, optional
            超时时间（毫秒）。为 `None` 时使用 `SessionOptions.timeout_ms`。

        Returns
        -------
        FrameSet | None
            获取到的数据帧集合；超时或失败时返回 `None`。
        """
        if self.pipeline is None:
            raise RuntimeError("会话尚未初始化。请先调用 start()。")
        timeout = self.options.timeout_ms if timeout_ms is None else timeout_ms
        return self.pipeline.wait_for_frames(timeout)

    def get_imu_sample_from_frames(self, frames: "FrameSet") -> OrbbecImuSample:
        """
        从当前帧集合提取 Orbbec IMU 数据。

        Parameters
        ----------
        frames : FrameSet
            `wait_for_frames()` 返回的 SDK 帧集合。

        Returns
        -------
        OrbbecImuSample
            加速度与陀螺仪结构化数据。设备不支持或当前帧缺失时，对应字段为 `None`。
        """
        accel_frame = None
        gyro_frame = None

        if self.has_accel_sensor:
            try:
                frame = frames.get_frame(OBFrameType.ACCEL_FRAME)
                if frame is not None:
                    accel_frame = frame.as_accel_frame()
            except Exception:
                accel_frame = None

        if self.has_gyro_sensor:
            try:
                frame = frames.get_frame(OBFrameType.GYRO_FRAME)
                if frame is not None:
                    gyro_frame = frame.as_gyro_frame()
            except Exception:
                gyro_frame = None

        return OrbbecImuSample(
            accel_mps2=_vector_from_imu_frame(accel_frame),
            gyro_rad_s=_vector_from_imu_frame(gyro_frame),
            accel_temperature_c=_temperature_from_imu_frame(accel_frame),
            gyro_temperature_c=_temperature_from_imu_frame(gyro_frame),
            accel_timestamp_us=_timestamp_us_from_frame(accel_frame),
            gyro_timestamp_us=_timestamp_us_from_frame(gyro_frame),
        )

    def get_camera_param(self) -> OBCameraParam:
        """
        读取当前管线相机参数。

        Returns
        -------
        OBCameraParam
            SDK 原生相机参数对象。
        """
        if self.pipeline is None:
            raise RuntimeError("会话尚未初始化。请先调用 start()。")
        return self.pipeline.get_camera_param()

    def create_point_cloud_filter(
        self, camera_param: OBCameraParam | None = None
    ) -> PointCloudFilter:
        """
        创建点云过滤器并可选写入相机参数。

        Parameters
        ----------
        camera_param : OBCameraParam | None, optional
            若提供则写入过滤器，确保点云计算使用指定参数。

        Returns
        -------
        PointCloudFilter
            已初始化的点云过滤器。
        """
        point_filter = PointCloudFilter()
        point_filter.set_color_data_normalization(False)
        if camera_param is not None:
            point_filter.set_camera_param(camera_param)
        return point_filter

    def _resolve_frustum_config(
        self,
        *,
        min_depth_mm: float | None = None,
        frustum_max_depth_mm: float | None = None,
        near_width_mm: float | None = None,
        near_height_mm: float | None = None,
        far_width_mm: float | None = None,
        far_height_mm: float | None = None,
    ) -> SensorFrustumConfig:
        """解析本次调用的视锥参数（调用参数优先，实例默认次之）。"""
        cfg = self.sensor_frustum
        return SensorFrustumConfig(
            min_depth_mm=(
                cfg.min_depth_mm if min_depth_mm is None else float(min_depth_mm)
            ),
            max_depth_mm=(
                cfg.max_depth_mm
                if frustum_max_depth_mm is None
                else float(frustum_max_depth_mm)
            ),
            near_width_mm=(
                cfg.near_width_mm if near_width_mm is None else float(near_width_mm)
            ),
            near_height_mm=(
                cfg.near_height_mm if near_height_mm is None else float(near_height_mm)
            ),
            far_width_mm=(
                cfg.far_width_mm if far_width_mm is None else float(far_width_mm)
            ),
            far_height_mm=(
                cfg.far_height_mm if far_height_mm is None else float(far_height_mm)
            ),
        )

    def filter_points_for_sensor(
        self,
        points: np.ndarray,
        *,
        max_depth_mm: float | None = None,
        apply_sensor_frustum: bool = True,
        min_depth_mm: float | None = None,
        frustum_max_depth_mm: float | None = None,
        near_width_mm: float | None = None,
        near_height_mm: float | None = None,
        far_width_mm: float | None = None,
        far_height_mm: float | None = None,
    ) -> np.ndarray:
        """
        对点云进行归一化、有效性过滤与可选视锥切割。

        Parameters
        ----------
        points : np.ndarray
            输入点云，支持 `Nx3` 或 `Nx6`（含 RGB）。
        max_depth_mm : float | None, optional
            深度上限。`None` 时使用实例视锥配置中的 `max_depth_mm`。
        apply_sensor_frustum : bool, optional
            是否应用视锥切割。

        Returns
        -------
        np.ndarray
            过滤后的点云，保持输入列数（3 或 6）。
        """
        frustum_cfg = self._resolve_frustum_config(
            min_depth_mm=min_depth_mm,
            frustum_max_depth_mm=frustum_max_depth_mm,
            near_width_mm=near_width_mm,
            near_height_mm=near_height_mm,
            far_width_mm=far_width_mm,
            far_height_mm=far_height_mm,
        )

        effective_depth_limit = (
            frustum_cfg.max_depth_mm if max_depth_mm is None else float(max_depth_mm)
        )
        normalized = normalize_points(points)
        valid_points, _ = filter_valid_points(
            normalized, max_depth_mm=effective_depth_limit
        )
        if len(valid_points) == 0:
            return valid_points

        if not apply_sensor_frustum:
            return valid_points

        frustum_depth = min(frustum_cfg.max_depth_mm, effective_depth_limit)
        return filter_points_in_sensor_frustum(
            valid_points,
            min_depth_mm=frustum_cfg.min_depth_mm,
            max_depth_mm=frustum_depth,
            near_width_mm=frustum_cfg.near_width_mm,
            near_height_mm=frustum_cfg.near_height_mm,
            far_width_mm=frustum_cfg.far_width_mm,
            far_height_mm=frustum_cfg.far_height_mm,
        )

    def calculate_points_from_frames(
        self,
        *,
        frames: "FrameSet",
        point_filter: PointCloudFilter,
        max_depth_mm: float | None = None,
        apply_sensor_frustum: bool = True,
        min_depth_mm: float | None = None,
        frustum_max_depth_mm: float | None = None,
        near_width_mm: float | None = None,
        near_height_mm: float | None = None,
        far_width_mm: float | None = None,
        far_height_mm: float | None = None,
    ) -> np.ndarray:
        """
        从帧集合计算点云，并应用统一过滤流程。

        Returns
        -------
        np.ndarray
            输出点云，形状为 `Nx3` 或 `Nx6`。
        """
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            width = 6 if self.has_color_sensor else 3
            return np.empty((0, width), dtype=np.float32)

        point_frames, use_color = self.prepare_frame_for_point_cloud(frames)
        set_point_cloud_filter_format(
            point_filter,
            depth_scale=float(depth_frame.get_depth_scale()),
            use_color=use_color,
        )
        cloud_frame = point_filter.process(point_frames)
        if cloud_frame is None:
            width = 6 if use_color else 3
            return np.empty((0, width), dtype=np.float32)

        raw_points = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
        return self.filter_points_for_sensor(
            raw_points,
            max_depth_mm=max_depth_mm,
            apply_sensor_frustum=apply_sensor_frustum,
            min_depth_mm=min_depth_mm,
            frustum_max_depth_mm=frustum_max_depth_mm,
            near_width_mm=near_width_mm,
            near_height_mm=near_height_mm,
            far_width_mm=far_width_mm,
            far_height_mm=far_height_mm,
        )

    def estimate_fusion_frame_count(self, fusion_interval_s: float = 1.0) -> int:
        """根据深度流帧率估算融合采样的目标帧数。"""
        if fusion_interval_s <= 0:
            raise ValueError("fusion_interval_s must be > 0")
        fps = self._depth_stream_fps if self._depth_stream_fps > 0 else 30.0
        return max(1, int(round(fps * fusion_interval_s)))

    def capture_fused_points_by_interval(
        self, fusion_interval_s: float = 1.0
    ) -> np.ndarray:
        """在指定时间窗口内采样并融合多帧点云。"""
        if not self._started:
            raise RuntimeError("Session must be started before capturing fused points.")
        target_frames = self.estimate_fusion_frame_count(
            fusion_interval_s=fusion_interval_s
        )

        point_filter = self.create_point_cloud_filter(
            camera_param=self.get_camera_param()
        )
        fused_parts: list[np.ndarray] = []
        sampled = 0
        deadline = time.monotonic() + max(fusion_interval_s * 3.0, 1.0)

        while sampled < target_frames:
            if time.monotonic() > deadline:
                break

            frames = self.wait_for_frames()
            if frames is None:
                continue
            valid_points = self.calculate_points_from_frames(
                frames=frames,
                point_filter=point_filter,
                max_depth_mm=self.sensor_frustum.max_depth_mm,
                apply_sensor_frustum=True,
            )
            if len(valid_points) == 0:
                continue

            fused_parts.append(valid_points)
            sampled += 1

        if not fused_parts:
            width = 6 if self.has_color_sensor else 3
            return np.empty((0, width), dtype=np.float32)

        fused = np.concatenate(fused_parts, axis=0)
        return voxel_downsample_points_numpy(fused, voxel_size_mm=1.0)

    def prepare_frame_for_point_cloud(
        self, frames: "FrameSet"
    ) -> tuple["FrameSet", bool]:
        """将帧集合转换为可用于点云计算的输入。"""
        color_frame = frames.get_color_frame()
        use_color = self.has_color_sensor and color_frame is not None

        if use_color and self.align_filter is not None:
            try:
                aligned = self.align_filter.process(frames)
                if aligned is not None:
                    return aligned, True
            except Exception:
                pass
        return frames, use_color

    def _set_log_level(self) -> None:
        """设置 SDK 日志级别为静默，减少终端噪音。"""
        try:
            Context.set_logger_level(OBLogLevel.NONE)
        except Exception:
            pass

    def _configure_streams(self) -> None:
        """配置深度/彩色流并选择合适 profile。"""
        if self.pipeline is None or self.config is None:
            raise RuntimeError("会话运行对象未初始化。请先调用 start()。")
        depth_profile_list = None
        try:
            depth_profile_list = self.pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
        except OBError as exc:
            self._raise_device_runtime_error(
                stage="get_stream_profile_list(DEPTH_SENSOR)",
                exc=exc,
                extra_hint="未检测到可用深度设备，或设备枚举失败。",
            )
        if depth_profile_list is None:
            raise RuntimeError(
                "未获取到深度流 profile。请检查相机是否通过 USB 正常连接，"
                "并确认未被其他程序占用。"
            )
        depth_profile = _select_profile_with_preferred_fps(
            profile_list=depth_profile_list,
            preferred_fps=self.options.preferred_capture_fps,
            preferred_format=OBFormat.Y16,
        )
        self.config.enable_stream(depth_profile)
        self._depth_stream_fps = _safe_profile_fps(depth_profile, fallback=30.0)

        self.has_color_sensor = False
        self.has_accel_sensor = False
        self.has_gyro_sensor = False
        try:
            color_profile_list = self.pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
            if color_profile_list is not None:
                color_profile = _select_profile_with_preferred_fps(
                    profile_list=color_profile_list,
                    preferred_fps=self.options.preferred_capture_fps,
                    preferred_format=OBFormat.YUYV,
                )
                self.config.enable_stream(color_profile)
                if self.options.require_full_frame_when_color:
                    self.config.set_frame_aggregate_output_mode(
                        OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                    )
                self.has_color_sensor = True
        except OBError:
            self.has_color_sensor = False

        if self.options.enable_imu:
            self._configure_imu_streams()

    def _configure_imu_streams(self) -> None:
        """按设备支持情况启用 Orbbec 内置 IMU 流。"""
        if self.pipeline is None or self.config is None:
            raise RuntimeError("会话运行对象未初始化。请先调用 start()。")

        device = self.pipeline.get_device()
        try:
            device.get_sensor(OBSensorType.ACCEL_SENSOR)
            self.config.enable_accel_stream()
            self.has_accel_sensor = True
        except Exception:
            self.has_accel_sensor = False

        try:
            device.get_sensor(OBSensorType.GYRO_SENSOR)
            self.config.enable_gyro_stream()
            self.has_gyro_sensor = True
        except Exception:
            self.has_gyro_sensor = False

        if not (self.has_accel_sensor or self.has_gyro_sensor):
            raise RuntimeError(
                "当前 Orbbec 设备未检测到可用 ACCEL/GYRO IMU 传感器。"
                " 若设备型号确认支持 IMU，请先运行 pyorbbecsdk/examples/imu.py "
                "确认 SDK 与固件侧是否能枚举 ACCEL/GYRO_SENSOR。"
            )

        if self.options.require_full_frame_when_imu:
            self.config.set_frame_aggregate_output_mode(
                OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
            )

    def _raise_device_runtime_error(
        self, stage: str, exc: Exception, extra_hint: str
    ) -> None:
        """
        将 SDK 设备异常统一包装为可读的运行时错误。

        Parameters
        ----------
        stage : str
            出错阶段标识。
        exc : Exception
            原始异常对象。
        extra_hint : str
            附加排查提示。

        Raises
        ------
        RuntimeError
            统一错误类型，便于上层脚本输出清晰提示。
        """
        message = (
            f"Orbbec 会话初始化失败（阶段：{stage}）。{extra_hint} " f"原始错误：{exc}"
        )
        raise RuntimeError(message) from exc

    def _init_runtime_objects(self) -> None:
        """
        初始化 SDK 运行对象（带子进程构造预检）。

        Notes
        -----
        当 pyorbbecsdk 在无设备时原生崩溃，主进程无法直接捕获。
        先在子进程探测 Pipeline() 构造，可把“直接退出”转成可读错误。
        """
        if (
            self.pipeline is not None
            and self.config is not None
            and self.context is not None
        ):
            return

        self._probe_pipeline_ctor_in_subprocess()
        self.context = Context()
        self.pipeline = Pipeline()
        self.config = Config()

    def _probe_pipeline_ctor_in_subprocess(self) -> None:
        """
        在子进程探测 Pipeline 构造，避免主进程被原生崩溃直接带走。

        Raises
        ------
        RuntimeError
            当子进程异常退出（常见于无 USB 设备、驱动异常、设备占用）时抛出。
        """
        cmd = [
            sys.executable,
            "-c",
            "from pyorbbecsdk import Pipeline; Pipeline(); print('ok')",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Orbbec 子进程预检失败，无法验证 Pipeline 构造。原始错误：{exc}"
            ) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr if stderr else stdout
            raise RuntimeError(
                "Orbbec 设备预检失败：子进程在构造 Pipeline 时异常退出。"
                " 可能是 USB 未连接、设备被占用或驱动异常。"
                f" 子进程退出码={proc.returncode}。输出：{detail}"
            )


class Gemini305(OrbbecSession):
    """Gemini305 专用会话，集中维护该型号默认视锥参数。"""

    @classmethod
    def get_default_sensor_frustum(cls) -> SensorFrustumConfig:
        """返回 Gemini305 的默认视锥参数。"""
        return SensorFrustumConfig(
            min_depth_mm=70.0,
            max_depth_mm=430.0,
            near_width_mm=117.0,
            near_height_mm=89.0,
            far_width_mm=839.0,
            far_height_mm=637.0,
        )


def _safe_profile_fps(profile, fallback: float) -> float:
    """安全读取 profile 帧率。"""
    try:
        return float(profile.get_fps())
    except Exception:
        return float(fallback)


def _vector_from_imu_frame(frame) -> tuple[float, float, float] | None:
    """从 Orbbec IMU 帧中读取三轴数据。"""
    if frame is None:
        return None
    try:
        return (float(frame.get_x()), float(frame.get_y()), float(frame.get_z()))
    except Exception:
        return None


def _temperature_from_imu_frame(frame) -> float | None:
    """从 Orbbec IMU 帧中读取温度。"""
    if frame is None:
        return None
    try:
        return float(frame.get_temperature())
    except Exception:
        return None


def _timestamp_us_from_frame(frame) -> int | None:
    """从 SDK 帧中读取微秒时间戳。"""
    if frame is None:
        return None
    try:
        return int(frame.get_timestamp_us())
    except Exception:
        try:
            return int(frame.get_timestamp()) * 1000
        except Exception:
            return None


def _select_profile_with_preferred_fps(
    profile_list, preferred_fps: int | None, preferred_format: OBFormat
):
    """按“格式优先 + 帧率优先”策略选择流 profile。"""
    default_profile = profile_list.get_default_video_stream_profile()
    if preferred_fps is None:
        return default_profile

    count = int(profile_list.get_count())
    candidates: list = []
    for idx in range(count):
        profile = profile_list.get_stream_profile_by_index(idx)
        if hasattr(profile, "get_fps") and hasattr(profile, "get_format"):
            candidates.append(profile)

    if not candidates:
        return default_profile

    for profile in candidates:
        if (
            int(profile.get_fps()) == int(preferred_fps)
            and profile.get_format() == preferred_format
        ):
            return profile

    for profile in candidates:
        if int(profile.get_fps()) == int(preferred_fps):
            return profile

    return default_profile
