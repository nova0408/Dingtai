from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBCameraDistortion,
    OBCameraIntrinsic,
    OBCameraParam,
    OBError,
    OBExtrinsic,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBLogLevel,
    OBSensorType,
    OBStreamType,
    Pipeline,
    PointCloudFilter,
)

if TYPE_CHECKING:
    from pyorbbecsdk import FrameSet


@dataclass(frozen=True)
class SessionOptions:
    timeout_ms: int = 120
    enable_frame_sync: bool = True
    require_full_frame_when_color: bool = True
    preferred_capture_fps: int | None = None


@dataclass(frozen=True)
class IntrinsicPatch:
    fx_scale: float = 1.0
    fy_scale: float = 1.0
    cx_offset: float = 0.0
    cy_offset: float = 0.0


@dataclass(frozen=True)
class DistortionPatch:
    k1_offset: float = 0.0
    k2_offset: float = 0.0
    p1_offset: float = 0.0
    p2_offset: float = 0.0


@dataclass(frozen=True)
class CameraParamPatch:
    depth: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    color: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    depth_dist: DistortionPatch = field(default_factory=DistortionPatch)
    color_dist: DistortionPatch = field(default_factory=DistortionPatch)
    d2c_translation_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


class OrbbecSession:
    def __init__(self, options: SessionOptions | None = None) -> None:
        self.options = options or SessionOptions()
        self.context = Context()
        self.pipeline = Pipeline()
        self.config = Config()

        self.has_color_sensor = False
        self.align_filter: AlignFilter | None = None
        self._depth_stream_fps: float = 0.0
        self._started = False

    def __enter__(self) -> "OrbbecSession":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return

        self._set_log_level()
        self._configure_streams()

        if self.has_color_sensor and self.options.enable_frame_sync:
            try:
                self.pipeline.enable_frame_sync()
            except OBError:
                pass

        self.pipeline.start(self.config)
        if self.has_color_sensor:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.pipeline.stop()
        finally:
            self._started = False
            self.align_filter = None

    def wait_for_frames(self, timeout_ms: int | None = None) -> "FrameSet | None":
        timeout = self.options.timeout_ms if timeout_ms is None else timeout_ms
        return self.pipeline.wait_for_frames(timeout)

    def get_camera_param(self) -> OBCameraParam:
        return self.pipeline.get_camera_param()

    def create_point_cloud_filter(self, camera_param: OBCameraParam | None = None) -> PointCloudFilter:
        point_filter = PointCloudFilter()
        point_filter.set_color_data_normalization(False)
        if camera_param is not None:
            point_filter.set_camera_param(camera_param)
        return point_filter

    def estimate_fusion_frame_count(self, fusion_interval_s: float = 1.0) -> int:
        if fusion_interval_s <= 0:
            raise ValueError("fusion_interval_s must be > 0")
        fps = self._depth_stream_fps if self._depth_stream_fps > 0 else 30.0
        return max(1, int(round(fps * fusion_interval_s)))

    def capture_fused_points_by_interval(self, fusion_interval_s: float = 1.0) -> np.ndarray:
        if not self._started:
            raise RuntimeError("Session must be started before capturing fused points.")
        target_frames = self.estimate_fusion_frame_count(fusion_interval_s=fusion_interval_s)

        point_filter = self.create_point_cloud_filter(camera_param=self.get_camera_param())
        fused_parts: list[np.ndarray] = []
        sampled = 0
        deadline = time.monotonic() + max(fusion_interval_s * 3.0, 1.0)

        while sampled < target_frames:
            if time.monotonic() > deadline:
                break

            frames = self.wait_for_frames()
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            point_frames, use_color = self.prepare_frame_for_point_cloud(frames)
            set_point_cloud_filter_format(
                point_filter,
                depth_scale=float(depth_frame.get_depth_scale()),
                use_color=use_color,
            )
            cloud_frame = point_filter.process(point_frames)
            if cloud_frame is None:
                continue

            raw_points = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
            normalized = normalize_points(raw_points)
            valid_points, _ = filter_valid_points(normalized, max_depth_mm=DEFAULT_SENSOR_MAX_DEPTH_MM)
            valid_points = filter_points_in_sensor_frustum(
                valid_points,
                min_depth_mm=DEFAULT_SENSOR_MIN_DEPTH_MM,
                max_depth_mm=DEFAULT_SENSOR_MAX_DEPTH_MM,
                near_width_mm=DEFAULT_SENSOR_NEAR_WIDTH_MM,
                near_height_mm=DEFAULT_SENSOR_NEAR_HEIGHT_MM,
                far_width_mm=DEFAULT_SENSOR_FAR_WIDTH_MM,
                far_height_mm=DEFAULT_SENSOR_FAR_HEIGHT_MM,
            )
            if len(valid_points) == 0:
                continue

            fused_parts.append(valid_points)
            sampled += 1

        if not fused_parts:
            width = 6 if self.has_color_sensor else 3
            return np.empty((0, width), dtype=np.float32)

        fused = np.concatenate(fused_parts, axis=0)
        return _voxel_downsample_points_numpy(fused, voxel_size_mm=1.0)

    def prepare_frame_for_point_cloud(self, frames: "FrameSet") -> tuple["FrameSet", bool]:
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
        try:
            Context.set_logger_level(OBLogLevel.NONE)
        except Exception:
            pass

    def _configure_streams(self) -> None:
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            raise RuntimeError("No depth stream profile available.")
        depth_profile = _select_profile_with_preferred_fps(
            profile_list=depth_profile_list,
            preferred_fps=self.options.preferred_capture_fps,
            preferred_format=OBFormat.Y16,
        )
        self.config.enable_stream(depth_profile)
        self._depth_stream_fps = _safe_profile_fps(depth_profile, fallback=30.0)

        self.has_color_sensor = False
        try:
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profile_list is not None:
                color_profile = _select_profile_with_preferred_fps(
                    profile_list=color_profile_list,
                    preferred_fps=self.options.preferred_capture_fps,
                    preferred_format=OBFormat.YUYV,
                )
                self.config.enable_stream(color_profile)
                if self.options.require_full_frame_when_color:
                    self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                self.has_color_sensor = True
        except OBError:
            self.has_color_sensor = False


def clone_camera_param(source: OBCameraParam) -> OBCameraParam:
    cloned = OBCameraParam()
    cloned.depth_intrinsic = _clone_intrinsic(source.depth_intrinsic)
    cloned.rgb_intrinsic = _clone_intrinsic(source.rgb_intrinsic)
    cloned.depth_distortion = _clone_distortion(source.depth_distortion)
    cloned.rgb_distortion = _clone_distortion(source.rgb_distortion)
    cloned.transform = _clone_extrinsic(source.transform)
    return cloned


def apply_camera_param_patch(camera_param: OBCameraParam, patch: CameraParamPatch) -> OBCameraParam:
    depth_intrinsic = _apply_intrinsic_patch(camera_param.depth_intrinsic, patch.depth)
    color_intrinsic = _apply_intrinsic_patch(camera_param.rgb_intrinsic, patch.color)
    depth_distortion = _apply_distortion_patch(camera_param.depth_distortion, patch.depth_dist)
    color_distortion = _apply_distortion_patch(camera_param.rgb_distortion, patch.color_dist)
    transform = _apply_extrinsic_patch(camera_param.transform, patch.d2c_translation_offset_mm)

    camera_param.depth_intrinsic = depth_intrinsic
    camera_param.rgb_intrinsic = color_intrinsic
    camera_param.depth_distortion = depth_distortion
    camera_param.rgb_distortion = color_distortion
    camera_param.transform = transform
    return camera_param


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 2 and points.shape[1] in (3, 6):
        return points

    flat = points.reshape(-1)
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if flat.size % 6 == 0:
        return flat.reshape(-1, 6)
    if flat.size % 3 == 0:
        return flat.reshape(-1, 3)
    raise RuntimeError(f"Unsupported point array shape: {points.shape}, flat_size={flat.size}")


DEFAULT_SENSOR_MIN_DEPTH_MM = 70.0
DEFAULT_SENSOR_MAX_DEPTH_MM = 430.0
DEFAULT_SENSOR_NEAR_WIDTH_MM = 117.0
DEFAULT_SENSOR_NEAR_HEIGHT_MM = 89.0
DEFAULT_SENSOR_FAR_WIDTH_MM = 839.0
DEFAULT_SENSOR_FAR_HEIGHT_MM = 637.0


def filter_valid_points(points: np.ndarray, max_depth_mm: float | None) -> tuple[np.ndarray, float]:
    if points.size == 0:
        return points.reshape(0, 3), 0.0

    xyz = points[:, :3]
    valid = np.isfinite(xyz).all(axis=1)
    valid &= xyz[:, 2] > 0.0
    if max_depth_mm is not None:
        valid &= xyz[:, 2] <= max_depth_mm

    valid_ratio = float(np.count_nonzero(valid)) / float(len(points))
    return points[valid], valid_ratio


def filter_points_in_sensor_frustum(
    points: np.ndarray,
    min_depth_mm: float,
    max_depth_mm: float,
    near_width_mm: float,
    near_height_mm: float,
    far_width_mm: float,
    far_height_mm: float,
) -> np.ndarray:
    if points.size == 0:
        return points[0:0]
    if max_depth_mm <= min_depth_mm:
        raise ValueError("max_depth_mm must be > min_depth_mm")

    xyz = points[:, :3]
    z = xyz[:, 2]
    valid = (z >= float(min_depth_mm)) & (z <= float(max_depth_mm))
    if not np.any(valid):
        return points[valid]

    t = np.clip((z - float(min_depth_mm)) / float(max_depth_mm - min_depth_mm), 0.0, 1.0)
    width = float(near_width_mm) + t * float(far_width_mm - near_width_mm)
    height = float(near_height_mm) + t * float(far_height_mm - near_height_mm)

    valid &= np.abs(xyz[:, 0]) <= (0.5 * width)
    valid &= np.abs(xyz[:, 1]) <= (0.5 * height)
    return points[valid]


def set_point_cloud_filter_format(point_filter: PointCloudFilter, depth_scale: float, use_color: bool) -> None:
    point_filter.set_position_data_scaled(depth_scale)
    point_filter.set_create_point_format(OBFormat.RGB_POINT if use_color else OBFormat.POINT)


def camera_param_summary(name: str, camera_param: OBCameraParam) -> str:
    di = camera_param.depth_intrinsic
    ci = camera_param.rgb_intrinsic
    dd = camera_param.depth_distortion
    d2c_t = np.asarray(camera_param.transform.transform, dtype=np.float32).reshape(3)
    return (
        f"[{name}] depth(fx={di.fx:.2f}, fy={di.fy:.2f}, cx={di.cx:.2f}, cy={di.cy:.2f}) "
        f"color(fx={ci.fx:.2f}, fy={ci.fy:.2f}, cx={ci.cx:.2f}, cy={ci.cy:.2f}) "
        f"depth_dist(k1={dd.k1:.5f}, k2={dd.k2:.5f}) "
        f"d2c_t=({d2c_t[0]:.2f}, {d2c_t[1]:.2f}, {d2c_t[2]:.2f})mm"
    )


def _clone_intrinsic(source: OBCameraIntrinsic) -> OBCameraIntrinsic:
    cloned = OBCameraIntrinsic()
    cloned.fx = float(source.fx)
    cloned.fy = float(source.fy)
    cloned.cx = float(source.cx)
    cloned.cy = float(source.cy)
    cloned.width = int(source.width)
    cloned.height = int(source.height)
    return cloned


def _clone_distortion(source: OBCameraDistortion) -> OBCameraDistortion:
    cloned = OBCameraDistortion()
    cloned.k1 = float(source.k1)
    cloned.k2 = float(source.k2)
    cloned.k3 = float(source.k3)
    cloned.k4 = float(source.k4)
    cloned.k5 = float(source.k5)
    cloned.k6 = float(source.k6)
    cloned.p1 = float(source.p1)
    cloned.p2 = float(source.p2)
    return cloned


def _clone_extrinsic(source: OBExtrinsic) -> OBExtrinsic:
    cloned = OBExtrinsic()
    cloned.rot = np.asarray(source.rot, dtype=np.float32).copy().reshape(3, 3)
    cloned.transform = np.asarray(source.transform, dtype=np.float32).copy().reshape(3)
    return cloned


def _apply_intrinsic_patch(intrinsic: OBCameraIntrinsic, patch: IntrinsicPatch) -> OBCameraIntrinsic:
    intrinsic.fx = float(intrinsic.fx * patch.fx_scale)
    intrinsic.fy = float(intrinsic.fy * patch.fy_scale)
    intrinsic.cx = float(intrinsic.cx + patch.cx_offset)
    intrinsic.cy = float(intrinsic.cy + patch.cy_offset)
    return intrinsic


def _apply_distortion_patch(distortion: OBCameraDistortion, patch: DistortionPatch) -> OBCameraDistortion:
    distortion.k1 = float(distortion.k1 + patch.k1_offset)
    distortion.k2 = float(distortion.k2 + patch.k2_offset)
    distortion.p1 = float(distortion.p1 + patch.p1_offset)
    distortion.p2 = float(distortion.p2 + patch.p2_offset)
    return distortion


def _apply_extrinsic_patch(extrinsic: OBExtrinsic, translation_offset_mm: tuple[float, float, float]) -> OBExtrinsic:
    translation = np.asarray(extrinsic.transform, dtype=np.float32).reshape(3)
    translation += np.asarray(translation_offset_mm, dtype=np.float32).reshape(3)
    extrinsic.transform = translation
    return extrinsic


def _voxel_downsample_points_numpy(points: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    if points.size == 0:
        return points
    if voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be > 0")

    xyz = points[:, :3]
    voxel = np.floor(xyz / float(voxel_size_mm)).astype(np.int64)
    _, unique_idx = np.unique(voxel, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx]


def _safe_profile_fps(profile, fallback: float) -> float:
    try:
        return float(profile.get_fps())
    except Exception:
        return float(fallback)


def _select_profile_with_preferred_fps(profile_list, preferred_fps: int | None, preferred_format: OBFormat):
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
        if int(profile.get_fps()) == int(preferred_fps) and profile.get_format() == preferred_format:
            return profile

    for profile in candidates:
        if int(profile.get_fps()) == int(preferred_fps):
            return profile

    return default_profile
