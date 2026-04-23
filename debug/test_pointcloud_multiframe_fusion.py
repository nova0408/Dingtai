from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from loguru import logger
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBError,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBLogLevel,
    OBSensorType,
    OBStreamType,
    Pipeline,
    PointCloudFilter,
)

# region 默认参数（优先在此硬编码修改）
DEFAULT_TIMEOUT_MS = 120
DEFAULT_PREFERRED_CAPTURE_FPS = 30
DEFAULT_FUSION_FRAMES = 60
DEFAULT_FUSION_VOXEL_MM = 1.0
DEFAULT_MAX_DEPTH_MM = 5000.0
DEFAULT_MAX_POINTS_PER_FRAME = 120_000
DEFAULT_POINT_SIZE = 1.5
DEFAULT_COORDINATE_FRAME_SIZE = 200.0
DEFAULT_WINDOW_NAME = "Orbbec 多帧滑窗融合点云预览"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
DEFAULT_ENABLE_OUTLIER_FILTER = True
DEFAULT_OUTLIER_NB_NEIGHBORS = 20
DEFAULT_OUTLIER_STD_RATIO = 2.0
DEFAULT_VIS = True
# endregion


# region 数据结构
@dataclass(frozen=True)
class FusionPreviewOptions:
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    preferred_capture_fps: int = DEFAULT_PREFERRED_CAPTURE_FPS
    fusion_frames: int = DEFAULT_FUSION_FRAMES
    fusion_voxel_mm: float = DEFAULT_FUSION_VOXEL_MM
    max_depth_mm: float | None = DEFAULT_MAX_DEPTH_MM
    max_points_per_frame: int = DEFAULT_MAX_POINTS_PER_FRAME
    point_size: float = DEFAULT_POINT_SIZE
    coordinate_frame_size: float = DEFAULT_COORDINATE_FRAME_SIZE
    window_name: str = DEFAULT_WINDOW_NAME
    window_width: int = DEFAULT_WINDOW_WIDTH
    window_height: int = DEFAULT_WINDOW_HEIGHT
    enable_outlier_filter: bool = DEFAULT_ENABLE_OUTLIER_FILTER
    outlier_nb_neighbors: int = DEFAULT_OUTLIER_NB_NEIGHBORS
    outlier_std_ratio: float = DEFAULT_OUTLIER_STD_RATIO
    vis: bool = DEFAULT_VIS


@dataclass(frozen=True)
class FusionResult:
    batch_idx: int
    window_frames: int
    points: np.ndarray
    display_interval_s: float
    display_fps: float
    compute_cost_ms: float


DEFAULT_OPTIONS = FusionPreviewOptions()
# endregion


class OrbbecMultiFrameFusionPreview:
    def __init__(self, options: FusionPreviewOptions | None = None) -> None:
        self.options = options or DEFAULT_OPTIONS
        self.fusion_frames = max(1, int(self.options.fusion_frames))
        self.actual_capture_fps = float(self.options.preferred_capture_fps)
        self.expected_display_fps = self.actual_capture_fps / float(self.fusion_frames)

        self.context = Context()
        self.pipeline = Pipeline()
        self.config = Config()

        self.has_color_sensor = False
        self.align_filter: AlignFilter | None = None
        self.point_cloud_filter = PointCloudFilter()

        self.vis: o3d.visualization.Visualizer | None = None
        self.fused_pcd = o3d.geometry.PointCloud()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.options.coordinate_frame_size,
            origin=[0.0, 0.0, 0.0],
        )

        self._window_frames: deque[np.ndarray] = deque(maxlen=self.fusion_frames)
        self._collected_frames = 0
        self._output_batch_idx = 0
        self._last_emit_time: float | None = None

        self._fusion_queue: queue.Queue[FusionResult] = queue.Queue(maxsize=1)
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._worker_error: Exception | None = None

    # region 生命周期
    def run(self) -> None:
        try:
            self._set_sdk_log_level()
            self._configure_streams()
            self._start_pipeline()
            self._configure_point_cloud_filter()
            if self.options.vis:
                self._create_visualizer()
            self._start_worker()

            logger.info(
                f"采集帧率 {self.actual_capture_fps:.3f} fps，"
                f"融合帧数 {self.fusion_frames} 帧，"
                f"预期显示帧率 {self.expected_display_fps:.3f} fps"
            )

            if self.options.vis:
                self._preview_loop()
            else:
                self._headless_loop()
        finally:
            self._shutdown()

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(target=self._fusion_worker_loop, name="fusion-worker", daemon=True)
        self._worker_thread.start()

    def _shutdown(self) -> None:
        self._stop_event.set()

        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

        try:
            self.pipeline.stop()
        except Exception:
            pass

        try:
            if self.vis is not None:
                self.vis.destroy_window()
        except Exception:
            pass
    # endregion

    # region 配置与启动
    def _set_sdk_log_level(self) -> None:
        try:
            self.context.set_logger_level(OBLogLevel.NONE)
        except Exception:
            pass

    def _configure_streams(self) -> None:
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profiles is None:
            raise RuntimeError("未找到深度流配置。")

        depth_profile = self._pick_profile(
            profile_list=depth_profiles,
            preferred_fps=self.options.preferred_capture_fps,
            preferred_format=OBFormat.Y16,
            stream_name="深度",
        )
        self.config.enable_stream(depth_profile)
        self.actual_capture_fps = self._safe_get_fps(depth_profile, fallback=self.options.preferred_capture_fps)
        self.expected_display_fps = self.actual_capture_fps / float(self.fusion_frames)

        self.has_color_sensor = False
        try:
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profiles is not None:
                color_profile = self._pick_profile(
                    profile_list=color_profiles,
                    preferred_fps=self.options.preferred_capture_fps,
                    preferred_format=OBFormat.YUYV,
                    stream_name="彩色",
                )
                self.config.enable_stream(color_profile)
                self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                self.has_color_sensor = True
        except OBError as exc:
            logger.warning(f"彩色流不可用，将仅显示深度点云：{exc}")
            self.has_color_sensor = False

    def _pick_profile(self, profile_list, preferred_fps: int, preferred_format: OBFormat, stream_name: str):
        candidates: list = []
        count = int(profile_list.get_count())
        for idx in range(count):
            p = profile_list.get_stream_profile_by_index(idx)
            if hasattr(p, "get_fps") and hasattr(p, "get_format"):
                candidates.append(p)

        if not candidates:
            return profile_list.get_default_video_stream_profile()

        for p in candidates:
            if int(p.get_fps()) == int(preferred_fps) and p.get_format() == preferred_format:
                logger.info(
                    f"{stream_name}流 选中分辨率 {p.get_width()}x{p.get_height()} 像素，"
                    f"格式 {p.get_format()}，帧率 {p.get_fps()} fps"
                )
                return p

        for p in candidates:
            if int(p.get_fps()) == int(preferred_fps):
                logger.info(
                    f"{stream_name}流 选中分辨率 {p.get_width()}x{p.get_height()} 像素，"
                    f"格式 {p.get_format()}，帧率 {p.get_fps()} fps（格式回退）"
                )
                return p

        selected = profile_list.get_default_video_stream_profile()
        try:
            logger.warning(
                f"{stream_name}流 使用默认分辨率 {selected.get_width()}x{selected.get_height()} 像素，"
                f"格式 {selected.get_format()}，帧率 {selected.get_fps()} fps"
            )
        except Exception:
            logger.warning(f"{stream_name}流 使用默认 profile")
        return selected

    def _safe_get_fps(self, profile, fallback: int) -> float:
        try:
            return float(profile.get_fps())
        except Exception:
            return float(fallback)

    def _start_pipeline(self) -> None:
        if self.has_color_sensor:
            try:
                self.pipeline.enable_frame_sync()
                logger.success("已启用 frame sync")
            except OBError as exc:
                logger.warning(f"frame sync 不可用，继续非同步模式：{exc}")

        self.pipeline.start(self.config)
        if self.has_color_sensor:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def _configure_point_cloud_filter(self) -> None:
        try:
            camera_param = self.pipeline.get_camera_param()
            self.point_cloud_filter.set_camera_param(camera_param)
        except Exception as exc:
            logger.warning(f"设置 camera param 失败，继续默认行为：{exc}")

    def _create_visualizer(self) -> None:
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=(
                f"{self.options.window_name}（融合 {self.fusion_frames} 帧 / "
                f"显示帧率约 {self.expected_display_fps:.3f} fps）"
            ),
            width=self.options.window_width,
            height=self.options.window_height,
        )
        self.vis.add_geometry(self.axis)
        self.vis.add_geometry(self.fused_pcd)

        render_option = self.vis.get_render_option()
        if render_option is not None:
            render_option.point_size = self.options.point_size
            render_option.background_color = np.asarray([0.0, 0.0, 0.0])
    # endregion

    # region 线程与融合
    def _fusion_worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                points = self._capture_one_points_frame()
                if points is None:
                    continue

                self._window_frames.append(points)
                self._collected_frames += 1

                if len(self._window_frames) < self.fusion_frames:
                    continue
                if self._collected_frames % self.fusion_frames != 0:
                    continue

                start = time.monotonic()
                fused = np.concatenate(list(self._window_frames), axis=0)
                fused = self._voxel_downsample_points(fused, voxel_mm=self.options.fusion_voxel_mm)
                fused = self._remove_outliers_if_needed(fused)
                compute_cost_ms = (time.monotonic() - start) * 1000.0

                now = time.monotonic()
                if self._last_emit_time is None:
                    interval = float("nan")
                    fps = float("nan")
                else:
                    interval = now - self._last_emit_time
                    fps = 1.0 / max(interval, 1e-6)
                self._last_emit_time = now

                self._output_batch_idx += 1
                result = FusionResult(
                    batch_idx=self._output_batch_idx,
                    window_frames=len(self._window_frames),
                    points=fused,
                    display_interval_s=interval,
                    display_fps=fps,
                    compute_cost_ms=compute_cost_ms,
                )
                self._push_latest_result(result)
        except Exception as exc:
            self._worker_error = exc

    def _push_latest_result(self, result: FusionResult) -> None:
        while True:
            try:
                self._fusion_queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    _ = self._fusion_queue.get_nowait()
                except queue.Empty:
                    return

    def _drain_latest_result(self) -> FusionResult | None:
        latest: FusionResult | None = None
        while True:
            try:
                latest = self._fusion_queue.get_nowait()
            except queue.Empty:
                return latest
    # endregion

    # region 前台显示
    def _preview_loop(self) -> None:
        if self.vis is None:
            raise RuntimeError("Visualizer 未创建")

        while True:
            if not self._refresh_window():
                break

            if self._worker_error is not None:
                raise RuntimeError(f"后台融合线程异常：{self._worker_error}") from self._worker_error

            latest = self._drain_latest_result()
            if latest is not None:
                self._update_open3d_point_cloud(latest.points)
                self._refresh_window()
                if np.isfinite(latest.display_interval_s):
                    logger.info(
                        f"融合批次 {latest.batch_idx} 次，窗口帧数 {latest.window_frames} 帧，"
                        f"点数 {len(latest.points)} 点，显示间隔 {latest.display_interval_s:.3f} 秒，"
                        f"显示帧率 {latest.display_fps:.3f} fps，融合耗时 {latest.compute_cost_ms:.2f} ms"
                    )
                else:
                    logger.info(
                        f"融合批次 {latest.batch_idx} 次，窗口帧数 {latest.window_frames} 帧，"
                        f"点数 {len(latest.points)} 点，融合耗时 {latest.compute_cost_ms:.2f} ms"
                    )

            time.sleep(0.005)

    def _headless_loop(self) -> None:
        logger.warning("当前为无可视化模式，仅输出融合统计日志。")
        while not self._stop_event.is_set():
            if self._worker_error is not None:
                raise RuntimeError(f"后台融合线程异常：{self._worker_error}") from self._worker_error

            latest = self._drain_latest_result()
            if latest is not None:
                logger.info(
                    f"融合批次 {latest.batch_idx} 次，窗口帧数 {latest.window_frames} 帧，"
                    f"点数 {len(latest.points)} 点，融合耗时 {latest.compute_cost_ms:.2f} ms"
                )
            time.sleep(0.02)

    def _refresh_window(self) -> bool:
        if self.vis is None:
            return False
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return bool(alive)
    # endregion

    # region 点云处理
    def _capture_one_points_frame(self) -> np.ndarray | None:
        frames = self.pipeline.wait_for_frames(self.options.timeout_ms)
        if frames is None:
            return None

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            return None

        point_frames = frames
        color_frame = frames.get_color_frame()
        use_color = self.has_color_sensor and color_frame is not None
        if use_color and self.align_filter is not None:
            try:
                aligned = self.align_filter.process(frames)
                if aligned is not None:
                    point_frames = aligned
            except Exception:
                pass

        depth_scale = float(depth_frame.get_depth_scale())
        self.point_cloud_filter.set_position_data_scaled(depth_scale)
        self.point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT if use_color else OBFormat.POINT)

        cloud_frame = self.point_cloud_filter.process(point_frames)
        if cloud_frame is None:
            return None

        points = np.asarray(self.point_cloud_filter.calculate(cloud_frame), dtype=np.float32)
        points = self._normalize_points(points)
        if points.size == 0:
            return None

        points = self._filter_points(points)
        points = self._downsample_points(points, self.options.max_points_per_frame)
        if points.size == 0:
            return None
        return points

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
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
        raise RuntimeError(f"无法识别点云数组形状：shape={points.shape}, flat_size={flat.size}")

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        xyz = points[:, :3]
        valid_mask = np.isfinite(xyz).all(axis=1)
        valid_mask &= xyz[:, 2] > 0.0

        max_depth_mm = self.options.max_depth_mm
        if max_depth_mm is not None:
            valid_mask &= xyz[:, 2] <= max_depth_mm
        return points[valid_mask]

    def _downsample_points(self, points: np.ndarray, max_points: int) -> np.ndarray:
        if len(points) <= max_points:
            return points
        step = max(1, len(points) // max_points)
        return points[::step]

    def _voxel_downsample_points(self, points: np.ndarray, voxel_mm: float) -> np.ndarray:
        if points.size == 0:
            return points
        if voxel_mm <= 0:
            raise ValueError("voxel_mm must be > 0")

        pcd = o3d.geometry.PointCloud()
        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        has_color = points.shape[1] >= 6
        if has_color:
            rgb = np.ascontiguousarray(points[:, 3:6], dtype=np.float32)
            if rgb.size > 0 and float(np.max(rgb)) > 1.0:
                rgb = rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0.0, 1.0).astype(np.float64))

        down = pcd.voxel_down_sample(voxel_size=float(voxel_mm))
        down_xyz = np.asarray(down.points, dtype=np.float32)
        if down_xyz.size == 0:
            width = 6 if has_color else 3
            return np.empty((0, width), dtype=np.float32)

        if has_color and len(down.colors) == len(down.points):
            down_rgb = np.asarray(down.colors, dtype=np.float32)
            return np.concatenate([down_xyz, down_rgb], axis=1)
        return down_xyz

    def _remove_outliers_if_needed(self, points: np.ndarray) -> np.ndarray:
        if not self.options.enable_outlier_filter or len(points) < 100:
            return points

        pcd = o3d.geometry.PointCloud()
        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(xyz)

        _, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=max(2, int(self.options.outlier_nb_neighbors)),
            std_ratio=max(0.1, float(self.options.outlier_std_ratio)),
        )
        if len(inlier_idx) == 0:
            return points
        return points[np.asarray(inlier_idx, dtype=np.int64)]

    def _update_open3d_point_cloud(self, points: np.ndarray) -> None:
        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        self.fused_pcd.points = o3d.utility.Vector3dVector(xyz)

        if points.shape[1] >= 6:
            rgb = np.ascontiguousarray(points[:, 3:6], dtype=np.float32)
            if rgb.size > 0 and float(np.max(rgb)) > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            self.fused_pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        else:
            gray = np.full((len(points), 3), 0.85, dtype=np.float64)
            self.fused_pcd.colors = o3d.utility.Vector3dVector(gray)

        if self.vis is not None:
            self.vis.update_geometry(self.fused_pcd)
    # endregion


# region CLI（仅用于覆盖调参）
def _parse_cli() -> FusionPreviewOptions:
    parser = argparse.ArgumentParser(description="Orbbec 多帧滑窗融合点云预览（CLI 仅用于覆盖调参）")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames timeout in ms")
    parser.add_argument(
        "--preferred-capture-fps", type=int, default=DEFAULT_PREFERRED_CAPTURE_FPS, help="preferred stream fps"
    )
    parser.add_argument(
        "--fusion-frames", type=int, default=DEFAULT_FUSION_FRAMES, help="sliding window size and output stride"
    )
    parser.add_argument("--fusion-voxel-mm", type=float, default=DEFAULT_FUSION_VOXEL_MM, help="voxel size for fused cloud")
    parser.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM, help="max valid depth (mm)")
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=DEFAULT_MAX_POINTS_PER_FRAME,
        help="max points kept for each input frame",
    )
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE, help="visual point size")
    parser.add_argument(
        "--enable-outlier-filter",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_OUTLIER_FILTER,
        help="enable statistical outlier removal on fused cloud",
    )
    parser.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=DEFAULT_OUTLIER_NB_NEIGHBORS,
        help="neighbors for statistical outlier removal",
    )
    parser.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=DEFAULT_OUTLIER_STD_RATIO,
        help="std ratio for statistical outlier removal",
    )
    parser.add_argument(
        "--vis",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_VIS,
        help="enable visualizer window",
    )
    args = parser.parse_args()

    max_depth = None if float(args.max_depth_mm) <= 0 else float(args.max_depth_mm)
    return FusionPreviewOptions(
        timeout_ms=int(args.timeout_ms),
        preferred_capture_fps=max(1, int(args.preferred_capture_fps)),
        fusion_frames=max(1, int(args.fusion_frames)),
        fusion_voxel_mm=max(0.01, float(args.fusion_voxel_mm)),
        max_depth_mm=max_depth,
        max_points_per_frame=int(args.max_points_per_frame),
        point_size=float(args.point_size),
        enable_outlier_filter=bool(args.enable_outlier_filter),
        outlier_nb_neighbors=max(2, int(args.outlier_nb_neighbors)),
        outlier_std_ratio=max(0.1, float(args.outlier_std_ratio)),
        vis=bool(args.vis),
    )
# endregion


def main(options: FusionPreviewOptions | None = None) -> None:
    preview = OrbbecMultiFrameFusionPreview(options=options)
    preview.run()


if __name__ == "__main__":
    try:
        # IDE 直跑默认用顶部常量；需要调参时再传 CLI 覆盖。
        cli_override = _parse_cli() if len(sys.argv) > 1 else DEFAULT_OPTIONS
        main(cli_override)
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        traceback.print_exc()
