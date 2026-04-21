from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from open3d.cpu.pybind.visualization import Visualizer

    o3d.visualization.Visualizer = Visualizer

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
    OBCameraDistortion,
    OBCameraIntrinsic,
    OBCameraParam,
    OBExtrinsic,
    Pipeline,
    PointCloudFilter,
)


@dataclass(frozen=True)
class PreviewOptions:
    timeout_ms: int = 100
    max_points: int = 120_000
    max_depth_mm: float | None = 5000.0
    point_size: float = 1.5
    coordinate_frame_size: float = 200.0
    window_width: int = 960
    window_height: int = 640


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


CAMERA_PARAM_PRESETS: dict[str, CameraParamPatch] = {
    "baseline": CameraParamPatch(),
    "depth_fx_fy_plus10": CameraParamPatch(
        depth=IntrinsicPatch(fx_scale=1.10, fy_scale=1.10),
    ),
    "depth_fx_fy_minus10": CameraParamPatch(
        depth=IntrinsicPatch(fx_scale=0.90, fy_scale=0.90),
    ),
    "depth_cx_plus25": CameraParamPatch(
        depth=IntrinsicPatch(cx_offset=25.0),
    ),
    "depth_cy_minus25": CameraParamPatch(
        depth=IntrinsicPatch(cy_offset=-25.0),
    ),
    "color_fx_fy_plus10": CameraParamPatch(
        color=IntrinsicPatch(fx_scale=1.10, fy_scale=1.10),
    ),
    "depth_k1_plus_0p02": CameraParamPatch(
        depth_dist=DistortionPatch(k1_offset=0.02),
    ),
    "d2c_tx_plus10mm": CameraParamPatch(
        d2c_translation_offset_mm=(10.0, 0.0, 0.0),
    ),
}


@dataclass
class PreviewNode:
    name: str
    patch: CameraParamPatch
    point_filter: PointCloudFilter
    vis: o3d.visualization.Visualizer
    pcd: o3d.geometry.PointCloud
    alive: bool = True


class MultiCameraParamPreview:
    def __init__(self, options: PreviewOptions | None = None) -> None:
        self.options = options or PreviewOptions()
        self.context = Context()
        self.pipeline = Pipeline()
        self.config = Config()

        self.has_color_sensor = False
        self.align_filter: AlignFilter | None = None
        self.nodes: list[PreviewNode] = []

    def run(self) -> None:
        try:
            self._set_sdk_log_level()
            self._configure_streams()
            self._start_pipeline()
            self._create_preview_nodes()
            print("已创建多参数对比窗口。关闭全部窗口即可退出。")
            self._preview_loop()
        finally:
            self._shutdown()

    def _set_sdk_log_level(self) -> None:
        try:
            self.context.set_logger_level(OBLogLevel.NONE)
        except Exception:
            pass

    def _configure_streams(self) -> None:
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            raise RuntimeError("未找到可用的深度流配置，无法生成点云。")
        self.config.enable_stream(depth_profile_list.get_default_video_stream_profile())

        try:
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profile_list is not None:
                self.config.enable_stream(color_profile_list.get_default_video_stream_profile())
                self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                self.has_color_sensor = True
        except OBError as exc:
            print(f"读取彩色传感器失败，将仅显示深度点云：{exc}")
            self.has_color_sensor = False

    def _start_pipeline(self) -> None:
        if self.has_color_sensor:
            try:
                self.pipeline.enable_frame_sync()
            except OBError:
                pass
        self.pipeline.start(self.config)
        if self.has_color_sensor:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def _create_preview_nodes(self) -> None:
        base_camera_param = self.pipeline.get_camera_param()
        for name, patch in CAMERA_PARAM_PRESETS.items():
            point_filter = PointCloudFilter()
            point_filter.set_color_data_normalization(False)
            camera_param = self._clone_camera_param(base_camera_param)
            self._apply_camera_param_patch(camera_param, patch)
            point_filter.set_camera_param(camera_param)
            print(self._camera_param_summary(name, camera_param))

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"RGBD 点云参数对比：{name}",
                width=self.options.window_width,
                height=self.options.window_height,
            )
            pcd = o3d.geometry.PointCloud()
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.options.coordinate_frame_size,
                origin=[0.0, 0.0, 0.0],
            )
            vis.add_geometry(axis)
            vis.add_geometry(pcd)

            view = vis.get_view_control()
            view.set_lookat([0.0, 0.0, 0.0])
            view.set_front([0.0, 0.0, -1.0])
            view.set_up([0.0, -1.0, 0.0])

            render_option = vis.get_render_option()
            if render_option is not None:
                render_option.point_size = self.options.point_size
                render_option.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

            self.nodes.append(
                PreviewNode(
                    name=name,
                    patch=patch,
                    point_filter=point_filter,
                    vis=vis,
                    pcd=pcd,
                )
            )

    def _preview_loop(self) -> None:
        while any(node.alive for node in self.nodes):
            frames = self.pipeline.wait_for_frames(self.options.timeout_ms)
            if frames is None:
                self._refresh_windows_only()
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                self._refresh_windows_only()
                continue

            color_frame = frames.get_color_frame()
            use_color = self.has_color_sensor and color_frame is not None

            frame_for_point_cloud = frames
            if use_color and self.align_filter is not None:
                try:
                    aligned = self.align_filter.process(frames)
                    if aligned is not None:
                        frame_for_point_cloud = aligned
                except Exception:
                    pass

            depth_scale = depth_frame.get_depth_scale()
            for node in self.nodes:
                if not node.alive:
                    continue
                node.point_filter.set_position_data_scaled(depth_scale)
                node.point_filter.set_create_point_format(OBFormat.RGB_POINT if use_color else OBFormat.POINT)

                cloud_frame = node.point_filter.process(frame_for_point_cloud)
                if cloud_frame is None:
                    continue

                points = np.asarray(node.point_filter.calculate(cloud_frame), dtype=np.float32)
                points = self._normalize_points(points)
                points = self._filter_points(points)
                points = self._downsample_points(points)
                if points.size == 0:
                    continue

                self._update_open3d_point_cloud(node.pcd, points)
                node.vis.update_geometry(node.pcd)

            self._refresh_windows_only()

    def _refresh_windows_only(self) -> None:
        for node in self.nodes:
            if not node.alive:
                continue
            alive = node.vis.poll_events()
            node.vis.update_renderer()
            node.alive = bool(alive)

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
        raise RuntimeError(f"无法识别点云数组形状：{points.shape}, flat_size={flat.size}")

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        xyz = points[:, :3]
        valid = np.isfinite(xyz).all(axis=1)
        valid &= xyz[:, 2] > 0.0
        max_depth_mm = self.options.max_depth_mm
        if max_depth_mm is not None:
            valid &= xyz[:, 2] <= max_depth_mm
        return points[valid]

    def _downsample_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= self.options.max_points:
            return points
        step = max(1, len(points) // self.options.max_points)
        return points[::step]

    def _update_open3d_point_cloud(self, pcd: o3d.geometry.PointCloud, points: np.ndarray) -> None:
        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(xyz)

        if points.shape[1] >= 6:
            rgb = np.ascontiguousarray(points[:, 3:6], dtype=np.float32)
            max_rgb = float(np.max(rgb)) if rgb.size > 0 else 0.0
            if max_rgb > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        else:
            gray = np.full((len(points), 3), 0.85, dtype=np.float64)
            pcd.colors = o3d.utility.Vector3dVector(gray)

    def _clone_camera_param(self, source: OBCameraParam) -> OBCameraParam:
        cloned = OBCameraParam()
        cloned.depth_intrinsic = self._clone_intrinsic(source.depth_intrinsic)
        cloned.rgb_intrinsic = self._clone_intrinsic(source.rgb_intrinsic)
        cloned.depth_distortion = self._clone_distortion(source.depth_distortion)
        cloned.rgb_distortion = self._clone_distortion(source.rgb_distortion)
        cloned.transform = self._clone_extrinsic(source.transform)
        return cloned

    def _clone_intrinsic(self, source: OBCameraIntrinsic) -> OBCameraIntrinsic:
        cloned = OBCameraIntrinsic()
        cloned.fx = float(source.fx)
        cloned.fy = float(source.fy)
        cloned.cx = float(source.cx)
        cloned.cy = float(source.cy)
        cloned.width = int(source.width)
        cloned.height = int(source.height)
        return cloned

    def _clone_distortion(self, source: OBCameraDistortion) -> OBCameraDistortion:
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

    def _clone_extrinsic(self, source: OBExtrinsic) -> OBExtrinsic:
        cloned = OBExtrinsic()
        cloned.rot = np.asarray(source.rot, dtype=np.float32).copy().reshape(3, 3)
        cloned.transform = np.asarray(source.transform, dtype=np.float32).copy().reshape(3)
        return cloned

    def _apply_camera_param_patch(self, camera_param: OBCameraParam, patch: CameraParamPatch) -> None:
        depth_intrinsic = self._apply_intrinsic_patch(camera_param.depth_intrinsic, patch.depth)
        color_intrinsic = self._apply_intrinsic_patch(camera_param.rgb_intrinsic, patch.color)
        depth_distortion = self._apply_distortion_patch(camera_param.depth_distortion, patch.depth_dist)
        color_distortion = self._apply_distortion_patch(camera_param.rgb_distortion, patch.color_dist)
        transform = self._apply_extrinsic_patch(camera_param.transform, patch.d2c_translation_offset_mm)

        camera_param.depth_intrinsic = depth_intrinsic
        camera_param.rgb_intrinsic = color_intrinsic
        camera_param.depth_distortion = depth_distortion
        camera_param.rgb_distortion = color_distortion
        camera_param.transform = transform

    def _apply_intrinsic_patch(self, intrinsic: OBCameraIntrinsic, patch: IntrinsicPatch) -> OBCameraIntrinsic:
        intrinsic.fx = float(intrinsic.fx * patch.fx_scale)
        intrinsic.fy = float(intrinsic.fy * patch.fy_scale)
        intrinsic.cx = float(intrinsic.cx + patch.cx_offset)
        intrinsic.cy = float(intrinsic.cy + patch.cy_offset)
        return intrinsic

    def _apply_distortion_patch(
        self,
        distortion: OBCameraDistortion,
        patch: DistortionPatch,
    ) -> OBCameraDistortion:
        distortion.k1 = float(distortion.k1 + patch.k1_offset)
        distortion.k2 = float(distortion.k2 + patch.k2_offset)
        distortion.p1 = float(distortion.p1 + patch.p1_offset)
        distortion.p2 = float(distortion.p2 + patch.p2_offset)
        return distortion

    def _apply_extrinsic_patch(
        self,
        extrinsic: OBExtrinsic,
        translation_offset_mm: tuple[float, float, float],
    ) -> OBExtrinsic:
        translation = np.asarray(extrinsic.transform, dtype=np.float32).reshape(3)
        translation += np.asarray(translation_offset_mm, dtype=np.float32).reshape(3)
        extrinsic.transform = translation
        return extrinsic

    def _camera_param_summary(self, name: str, camera_param: OBCameraParam) -> str:
        di = camera_param.depth_intrinsic
        ci = camera_param.rgb_intrinsic
        dd = camera_param.depth_distortion
        d2c_t = np.asarray(camera_param.transform.transform, dtype=np.float32).reshape(3)
        return (
            f"[{name}] "
            f"depth(fx={di.fx:.2f}, fy={di.fy:.2f}, cx={di.cx:.2f}, cy={di.cy:.2f}) "
            f"color(fx={ci.fx:.2f}, fy={ci.fy:.2f}, cx={ci.cx:.2f}, cy={ci.cy:.2f}) "
            f"depth_dist(k1={dd.k1:.5f}, k2={dd.k2:.5f}) "
            f"d2c_t=({d2c_t[0]:.2f}, {d2c_t[1]:.2f}, {d2c_t[2]:.2f})mm"
        )

    def _shutdown(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        for node in self.nodes:
            try:
                node.vis.destroy_window()
            except Exception:
                pass


def main() -> None:
    preview = MultiCameraParamPreview()
    preview.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断，程序退出。")
    except Exception as exc:
        print(f"程序异常退出：{exc}")
        traceback.print_exc()
