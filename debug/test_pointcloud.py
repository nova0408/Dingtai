from __future__ import annotations

import traceback
from dataclasses import dataclass

import numpy as np
import open3d as o3d
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


@dataclass(frozen=True)
class PreviewOptions:
    """实时点云预览配置。"""

    timeout_ms: int = 100
    max_points: int = 150_000
    max_depth_mm: float | None = 5000.0
    point_size: float = 1.5
    coordinate_frame_size: float = 200.0
    window_name: str = "Orbbec Open3D 实时点云预览"
    window_width: int = 1280
    window_height: int = 720


class OrbbecOpen3DPreview:
    """基于 pyorbbecsdk + Open3D 的实时点云预览器。"""

    def __init__(self, options: PreviewOptions | None = None) -> None:
        self.options = options or PreviewOptions()
        self.context = Context()
        self.pipeline = Pipeline()
        self.config = Config()

        self.has_color_sensor = False
        self.align_filter: AlignFilter | None = None
        self.point_cloud_filter = PointCloudFilter()

        self.vis: o3d.visualization.Visualizer = None
        self.pcd = o3d.geometry.PointCloud()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.options.coordinate_frame_size,
            origin=[0.0, 0.0, 0.0],
        )
        self._pcd_added = False

    def run(self) -> None:
        try:
            self._set_sdk_log_level()
            self._configure_streams()
            self._start_pipeline()
            self._configure_point_cloud_filter()
            self._create_visualizer()
            print("开始实时预览点云。关闭 Open3D 窗口即可退出。")
            self._preview_loop()
        finally:
            self._shutdown()

    def _set_sdk_log_level(self) -> None:
        """降低 SDK 控制台日志噪声。"""
        try:
            self.context.set_logger_level(OBLogLevel.NONE)
        except Exception:
            pass

    def _configure_streams(self) -> None:
        """配置深度流和可选彩色流。"""
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            raise RuntimeError("未找到可用的深度流配置，无法生成点云。")

        depth_profile = depth_profile_list.get_default_video_stream_profile()
        self.config.enable_stream(depth_profile)

        try:
            color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profile_list is not None:
                color_profile = color_profile_list.get_default_video_stream_profile()
                self.config.enable_stream(color_profile)
                self.config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
                self.has_color_sensor = True
        except OBError as exc:
            print(f"读取彩色传感器配置失败，将仅显示深度点云：{exc}")
            self.has_color_sensor = False

    def _start_pipeline(self) -> None:
        """启动相机流。"""
        if self.has_color_sensor:
            try:
                self.pipeline.enable_frame_sync()
                print("已启用 frame sync。")
            except OBError as exc:
                # 某些设备 / 固件组合不支持 frame sync。
                print(f"当前设备不支持 frame sync，继续以非同步模式运行：{exc}")

        self.pipeline.start(self.config)

        if self.has_color_sensor:
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def _configure_point_cloud_filter(self) -> None:
        """配置点云滤波器。"""
        try:
            camera_param = self.pipeline.get_camera_param()
            self.point_cloud_filter.set_camera_param(camera_param)
        except Exception as exc:
            print(f"设置 camera param 失败，继续使用默认行为：{exc}")

    def _create_visualizer(self) -> None:
        """创建 Open3D 可视化窗口。"""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=self.options.window_name,
            width=self.options.window_width,
            height=self.options.window_height,
        )
        self.vis.add_geometry(self.axis)
        self.vis.add_geometry(self.pcd)
        self._pcd_added = True
        view_control = self.vis.get_view_control()
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_up([0.0, -1.0, 0.0])

        render_option = self.vis.get_render_option()
        if render_option is not None:
            render_option.point_size = self.options.point_size
            render_option.background_color = np.asarray([0.0, 0.0, 0.0])

    def _preview_loop(self) -> None:
        """持续采集并刷新点云显示。"""
        if self.vis is None:
            raise RuntimeError("Visualizer 尚未创建。")

        while True:
            frames = self.pipeline.wait_for_frames(self.options.timeout_ms)
            if frames is None:
                if not self._refresh_window():
                    break
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                if not self._refresh_window():
                    break
                continue

            color_frame = frames.get_color_frame()
            use_color = self.has_color_sensor and color_frame is not None

            frame_for_point_cloud = frames
            if use_color and self.align_filter is not None:
                try:
                    aligned = self.align_filter.process(frames)
                    if aligned is not None:
                        frame_for_point_cloud = aligned
                except Exception as exc:
                    print(f"深度对齐到彩色失败，本帧退回未对齐模式：{exc}")

            depth_scale = depth_frame.get_depth_scale()
            self.point_cloud_filter.set_position_data_scaled(depth_scale)
            self.point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT if use_color else OBFormat.POINT)

            point_cloud_frame = self.point_cloud_filter.process(frame_for_point_cloud)
            if point_cloud_frame is None:
                if not self._refresh_window():
                    break
                continue

            points = np.asarray(
                self.point_cloud_filter.calculate(point_cloud_frame),
                dtype=np.float32,
            )
            points = self._normalize_points(points)
            if points.size == 0:
                if not self._refresh_window():
                    break
                continue

            points = self._filter_points(points)
            points = self._downsample_points(points)

            self._update_open3d_point_cloud(points)
            if not self._refresh_window():
                break

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """将 SDK 输出标准化为 (N,3) 或 (N,6)。"""
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

        raise RuntimeError(f"无法识别点云数组形状：original_shape={points.shape}, flat_size={flat.size}")

    def _filter_points(self, points: np.ndarray) -> np.ndarray:
        """移除非法点，并按最大深度裁剪。"""
        xyz = points[:, :3]
        valid_mask = np.isfinite(xyz).all(axis=1)
        valid_mask &= xyz[:, 2] > 0.0

        max_depth_mm = self.options.max_depth_mm
        if max_depth_mm is not None:
            valid_mask &= xyz[:, 2] <= max_depth_mm

        return points[valid_mask]

    def _downsample_points(self, points: np.ndarray) -> np.ndarray:
        """限制显示点数，避免 Open3D 刷新过慢。"""
        if len(points) <= self.options.max_points:
            return points

        step = max(1, len(points) // self.options.max_points)
        return points[::step]

    def _update_open3d_point_cloud(self, points: np.ndarray) -> None:
        """将 numpy 点云写回 Open3D。"""
        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        self.pcd.points = o3d.utility.Vector3dVector(xyz)

        if points.shape[1] >= 6:
            rgb = np.ascontiguousarray(points[:, 3:6], dtype=np.float32)
            max_rgb = float(np.max(rgb)) if rgb.size > 0 else 0.0
            if max_rgb > 1.0:
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
            self.pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        else:
            # 纯深度点云给一个浅灰色，避免全部显示成黑色。
            gray = np.full((len(points), 3), 0.85, dtype=np.float64)
            self.pcd.colors = o3d.utility.Vector3dVector(gray)

        if self.vis is not None and self._pcd_added:
            self.vis.update_geometry(self.pcd)

    def _refresh_window(self) -> bool:
        """刷新窗口；若窗口已关闭则返回 False。"""
        if self.vis is None:
            return False
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return bool(alive)

    def _shutdown(self) -> None:
        """释放资源。"""
        try:
            self.pipeline.stop()
        except Exception:
            pass

        try:
            if self.vis is not None:
                self.vis.destroy_window()
        except Exception:
            pass


def main() -> None:
    preview = OrbbecOpen3DPreview()
    preview.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断，程序退出。")
    except Exception as exc:
        print(f"程序异常退出：{exc}")
        traceback.print_exc()
