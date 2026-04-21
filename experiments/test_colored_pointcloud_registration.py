from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import traceback
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

from src.utils.Datas.Kinematics.SE3 import SE3_string


@dataclass(frozen=True)
class CaptureOptions:
    timeout_ms: int = 100
    min_depth_mm: float = 70.0
    max_depth_mm: float = 430.0
    frustum_min_width_mm: float = 117.0
    frustum_min_height_mm: float = 89.0
    frustum_max_width_mm: float = 839.0
    frustum_max_height_mm: float = 637.0
    max_points: int = 180_000
    window_width: int = 1280
    window_height: int = 720
    point_size: float = 1.5
    fusion_frame_count: int = 8
    fusion_min_frames: int = 3
    fusion_voxel_mm: float = 0.8


class RGBPointCloudCollector:
    def __init__(self, options: CaptureOptions | None = None) -> None:
        self.options = options or CaptureOptions()
        self.context = Context()
        self.pipeline = Pipeline()
        self.config = Config()
        self.align_filter: AlignFilter | None = None
        self.point_filter = PointCloudFilter()

        self._configure_streams()
        self._start_pipeline()
        self._configure_point_filter()

    def _configure_streams(self) -> None:
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profiles is None:
            raise RuntimeError("未找到深度流配置。")
        self.config.enable_stream(depth_profiles.get_default_video_stream_profile())

        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profiles is None:
            raise RuntimeError("未找到彩色流配置，RGB 点云采集不可用。")
        self.config.enable_stream(color_profiles.get_default_video_stream_profile())
        self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    def _start_pipeline(self) -> None:
        try:
            self.context.set_logger_level(OBLogLevel.NONE)
        except Exception:
            pass
        try:
            self.pipeline.enable_frame_sync()
        except OBError:
            pass

        self.pipeline.start(self.config)
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def _configure_point_filter(self) -> None:
        camera_param = self.pipeline.get_camera_param()
        self.point_filter.set_camera_param(camera_param)
        self.point_filter.set_create_point_format(OBFormat.RGB_POINT)
        self.point_filter.set_color_data_normalization(False)

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def capture_once(self, title: str) -> o3d.geometry.PointCloud:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=title,
            width=self.options.window_width,
            height=self.options.window_height,
        )

        render_opt = vis.get_render_option()
        if render_opt is not None:
            render_opt.point_size = self.options.point_size
            render_opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=200,
            origin=[0.0, 0.0, 0.0],
        )
        vis.add_geometry(axis)
        vis.add_geometry(pcd)
        view = vis.get_view_control()
        view.set_lookat([0.0, 0.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])

        latest_xyz: np.ndarray | None = None
        latest_rgb: np.ndarray | None = None
        recent_clouds: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=max(1, self.options.fusion_frame_count))
        while True:
            frames = self.pipeline.wait_for_frames(self.options.timeout_ms)
            if frames is not None:
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if depth_frame is not None and color_frame is not None:
                    aligned = frames
                    if self.align_filter is not None:
                        try:
                            maybe_aligned = self.align_filter.process(frames)
                            if maybe_aligned is not None:
                                aligned = maybe_aligned
                        except Exception:
                            pass

                    self.point_filter.set_position_data_scaled(depth_frame.get_depth_scale())
                    cloud_frame = self.point_filter.process(aligned)
                    if cloud_frame is not None:
                        raw_points = np.asarray(self.point_filter.calculate(cloud_frame), dtype=np.float32)
                        rgb_pcd = self._to_open3d_rgb_point_cloud(raw_points)
                        if len(rgb_pcd.points) > 0:
                            if len(rgb_pcd.points) > self.options.max_points:
                                step = max(1, len(rgb_pcd.points) // self.options.max_points)
                                rgb_pcd = rgb_pcd.select_by_index(np.arange(0, len(rgb_pcd.points), step))
                            pcd.points = rgb_pcd.points
                            pcd.colors = rgb_pcd.colors
                            vis.update_geometry(pcd)
                            latest_xyz = np.asarray(rgb_pcd.points, dtype=np.float64).copy()
                            latest_rgb = np.asarray(rgb_pcd.colors, dtype=np.float64).copy()
                            if latest_rgb.shape[0] == latest_xyz.shape[0]:
                                recent_clouds.append((latest_xyz, latest_rgb))

            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break

        vis.destroy_window()
        if latest_xyz is None or latest_xyz.size == 0:
            raise RuntimeError(f"{title} 未采集到有效 RGB 点云。")
        latest = o3d.geometry.PointCloud()
        latest.points = o3d.utility.Vector3dVector(latest_xyz)
        if latest_rgb is not None and latest_rgb.shape[0] == latest_xyz.shape[0]:
            latest.colors = o3d.utility.Vector3dVector(latest_rgb)
        if len(recent_clouds) < max(1, self.options.fusion_min_frames):
            return latest
        fused = self._fuse_recent_clouds(recent_clouds)
        if len(fused.points) == 0:
            return latest
        logger.info(
            f"多帧融合完成：frames={len(recent_clouds)}, raw_last={len(latest.points)}, fused={len(fused.points)}"
        )
        return fused

    def _to_open3d_rgb_point_cloud(self, points: np.ndarray) -> o3d.geometry.PointCloud:
        flat = np.asarray(points, dtype=np.float32).reshape(-1)
        if flat.size == 0:
            return o3d.geometry.PointCloud()

        if flat.size % 6 == 0:
            data = flat.reshape(-1, 6)
        elif flat.size % 3 == 0:
            data = flat.reshape(-1, 3)
        else:
            return o3d.geometry.PointCloud()

        xyz = data[:, :3]
        valid = np.isfinite(xyz).all(axis=1)
        valid &= self._valid_in_sensor_frustum(xyz)
        if not np.any(valid):
            return o3d.geometry.PointCloud()

        xyz = np.ascontiguousarray(xyz[valid], dtype=np.float64)
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(xyz)

        if data.shape[1] >= 6:
            rgb = np.ascontiguousarray(data[valid, 3:6], dtype=np.float32)
            if float(np.max(rgb)) > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            out.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        else:
            gray = np.full((xyz.shape[0], 3), 0.8, dtype=np.float64)
            out.colors = o3d.utility.Vector3dVector(gray)
        return out

    def _valid_in_sensor_frustum(self, xyz_mm: np.ndarray) -> np.ndarray:
        z = xyz_mm[:, 2]
        z_min = float(self.options.min_depth_mm)
        z_max = float(self.options.max_depth_mm)
        valid = (z >= z_min) & (z <= z_max)
        if not np.any(valid):
            return valid

        # 根据产品手册：z=70mm 时 FOV 约 117x89mm，z=430mm 时 FOV 约 839x637mm，按 z 线性插值。
        denom = max(z_max - z_min, 1e-9)
        t = np.clip((z - z_min) / denom, 0.0, 1.0)
        width = self.options.frustum_min_width_mm + t * (
            self.options.frustum_max_width_mm - self.options.frustum_min_width_mm
        )
        height = self.options.frustum_min_height_mm + t * (
            self.options.frustum_max_height_mm - self.options.frustum_min_height_mm
        )

        half_w = 0.5 * width
        half_h = 0.5 * height
        valid &= np.abs(xyz_mm[:, 0]) <= half_w
        valid &= np.abs(xyz_mm[:, 1]) <= half_h
        return valid

    def _fuse_recent_clouds(self, clouds: deque[tuple[np.ndarray, np.ndarray]]) -> o3d.geometry.PointCloud:
        xyz_list = [c[0] for c in clouds if c[0].size > 0]
        rgb_list = [c[1] for c in clouds if c[1].size > 0]
        if not xyz_list or len(xyz_list) != len(rgb_list):
            return o3d.geometry.PointCloud()
        merged_xyz = np.ascontiguousarray(np.vstack(xyz_list), dtype=np.float64)
        merged_rgb = np.ascontiguousarray(np.vstack(rgb_list), dtype=np.float64)
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(merged_xyz)
        merged.colors = o3d.utility.Vector3dVector(np.clip(merged_rgb, 0.0, 1.0))
        if self.options.fusion_voxel_mm > 0.0:
            merged = merged.voxel_down_sample(self.options.fusion_voxel_mm)
        return merged


def _preprocess_for_feature(
    pcd: o3d.geometry.PointCloud, voxel_size_mm: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    if len(pcd.points) == 0:
        raise RuntimeError("输入点云为空，无法做粗配准。")
    down = pcd.voxel_down_sample(voxel_size_mm)
    if len(down.points) == 0:
        down = pcd
    normal_radius = voxel_size_mm * 2.0
    feature_radius = voxel_size_mm * 5.0
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100),
    )
    return down, feature


def coarse_register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    min_scale_mm: float = 10.0,
    max_scale_mm: float = 80.0,
) -> o3d.pipelines.registration.RegistrationResult:
    voxel_size_mm = float(np.clip(max_scale_mm * 0.25, min_scale_mm, max_scale_mm))
    src_down, src_fpfh = _preprocess_for_feature(source, voxel_size_mm)
    tgt_down, tgt_fpfh = _preprocess_for_feature(target, voxel_size_mm)
    logger.debug(f"粗配准：voxel_size_mm={voxel_size_mm:.2f}")
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        max_scale_mm,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_scale_mm),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(80_000, 0.999),
    )


def fine_register_colored_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_error_mm: float = 5.0,
) -> o3d.pipelines.registration.RegistrationResult:
    src = o3d.geometry.PointCloud(source)
    tgt = o3d.geometry.PointCloud(target)
    normal_radius_mm = max(2.0 * max_error_mm, 2.0)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_mm, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius_mm, max_nn=30))
    return o3d.pipelines.registration.registration_colored_icp(
        src,
        tgt,
        max_error_mm,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )


def fine_register_multiscale_robust_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    voxel_sizes_mm: tuple[float, ...] = (2.0, 1.0, 0.5),
    max_corr_mm: tuple[float, ...] = (6.0, 3.0, 1.5),
    max_iters: tuple[int, ...] = (50, 40, 30),
) -> o3d.pipelines.registration.RegistrationResult:
    if not (len(voxel_sizes_mm) == len(max_corr_mm) == len(max_iters)):
        raise ValueError("多尺度 ICP 参数长度不一致。")
    transform = np.asarray(init_transform, dtype=np.float64).copy()
    last_result: o3d.pipelines.registration.RegistrationResult | None = None
    for level, (vox, corr, iters) in enumerate(zip(voxel_sizes_mm, max_corr_mm, max_iters), start=1):
        src_lvl = source.voxel_down_sample(vox)
        tgt_lvl = target.voxel_down_sample(vox)
        if len(src_lvl.points) == 0:
            src_lvl = o3d.geometry.PointCloud(source)
        if len(tgt_lvl.points) == 0:
            tgt_lvl = o3d.geometry.PointCloud(target)
        normal_radius = max(vox * 2.5, corr)
        src_lvl.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        tgt_lvl.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        kernel = o3d.pipelines.registration.TukeyLoss(k=max(corr * 0.5, 0.5))
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(kernel)
        last_result = o3d.pipelines.registration.registration_icp(
            src_lvl,
            tgt_lvl,
            corr,
            transform,
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iters)),
        )
        transform = last_result.transformation
        logger.debug(
            f"鲁棒 ICP level={level} vox={vox:.2f} corr={corr:.2f} iters={iters} "
            f"fitness={last_result.fitness:.4f} rmse={last_result.inlier_rmse:.4f}"
        )
    if last_result is None:
        raise RuntimeError("多尺度 ICP 未产生结果。")
    return last_result


def save_point_cloud(pcd: o3d.geometry.PointCloud, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=True)
    if not ok:
        raise RuntimeError(f"点云落盘失败：{path}")
    logger.info(f"点云已保存：{path}")


def visualize_registration_before_after(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transform_source_to_target: np.ndarray,
) -> None:
    before_src = o3d.geometry.PointCloud(source)
    before_tgt = o3d.geometry.PointCloud(target)

    after_src = o3d.geometry.PointCloud(source)
    after_tgt = o3d.geometry.PointCloud(target)
    after_src.transform(transform_source_to_target)
    offset = np.array([0.0, 1000.0, 0.0], dtype=np.float64)
    after_src.translate(offset)
    after_tgt.translate(offset)

    axis_before = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    axis_after = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=offset.tolist())

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("配准对比：原始 (下方) 与 配准后 (+Y 1000mm)", 1440, 900)
    vis.show_settings = True

    point_material = o3d.visualization.rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"
    point_material.point_size = 1.5

    vis.add_geometry("before_target", before_tgt, point_material)
    vis.add_geometry("before_source", before_src, point_material)
    vis.add_geometry("after_target", after_tgt, point_material)
    vis.add_geometry("after_source", after_src, point_material)
    vis.add_geometry("axis_before", axis_before)
    vis.add_geometry("axis_after", axis_after)
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


def main() -> None:
    collector = RGBPointCloudCollector()
    try:
        logger.info("第一次采集：关闭窗口后将锁定第一幅 RGB 点云。")
        pcd_1 = collector.capture_once("第一次采集：调整视角后关闭窗口")
        logger.debug(f"第一幅点云点数：{len(pcd_1.points)}")
        logger.info("第二次采集：关闭窗口后将锁定第二幅 RGB 点云。")
        pcd_2 = collector.capture_once("第二次采集：调整视角后关闭窗口")
        logger.debug(f"第二幅点云点数：{len(pcd_2.points)}")
    finally:
        collector.close()

    if len(pcd_1.points) == 0 or len(pcd_2.points) == 0:
        raise RuntimeError("采集结果含空点云，停止配准。")

    script_dir = Path(__file__).resolve().parent
    pcd_1_path = script_dir / "pcd1.pcd"
    pcd_2_path = script_dir / "pcd2.pcd"
    save_point_cloud(pcd_1, pcd_1_path)
    save_point_cloud(pcd_2, pcd_2_path)

    coarse = coarse_register(pcd_2, pcd_1, min_scale_mm=10.0, max_scale_mm=80.0)
    fine = fine_register_multiscale_robust_icp(pcd_2, pcd_1, coarse.transformation)

    logger.success(f"粗配准 T(PCD2->PCD1): {SE3_string(coarse.transformation)}")
    logger.success(f"精配准 T(PCD2->PCD1): {SE3_string(fine.transformation)}")

    visualize_registration_before_after(pcd_2, pcd_1, fine.transformation)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断。")
    except Exception as exc:
        logger.error(f"运行失败：{exc}")
        traceback.print_exc()
