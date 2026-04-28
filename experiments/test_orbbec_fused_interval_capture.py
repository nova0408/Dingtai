from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import (
    Gemini305,
    SessionOptions,
    filter_valid_points,
    normalize_points,
    set_point_cloud_filter_format,
)


# region 默认参数（优先在这里硬编码修改）
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_FUSION_INTERVAL_S = 2.0
DEFAULT_MAX_DEPTH_MM = 5000.0
DEFAULT_OUTPUT_PCD1 = PROJECT_ROOT / "experiments" / "pcd1.pcd"
DEFAULT_OUTPUT_PCD2 = PROJECT_ROOT / "experiments" / "pcd2.pcd"
DEFAULT_MAX_PREVIEW_POINTS = 160_000
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
DEFAULT_POINT_SIZE = 1.5
# endregion


# region 主流程
def main(
    fusion_interval_s: float = DEFAULT_FUSION_INTERVAL_S,
    output_pcd1: Path = DEFAULT_OUTPUT_PCD1,
    output_pcd2: Path = DEFAULT_OUTPUT_PCD2,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
) -> None:
    if fusion_interval_s <= 0:
        raise ValueError("fusion_interval_s must be > 0")

    output_pcd1.parent.mkdir(parents=True, exist_ok=True)
    output_pcd2.parent.mkdir(parents=True, exist_ok=True)

    session_options = SessionOptions(
        timeout_ms=int(timeout_ms),
        preferred_capture_fps=max(1, int(capture_fps)),
    )

    with Gemini305(options=session_options) as session:
        estimated_frames = session.estimate_fusion_frame_count(fusion_interval_s=fusion_interval_s)
        logger.info(
            f"预览模式已启动：采样时长 {fusion_interval_s:.2f} 秒，目标采样帧数 {estimated_frames} 帧，"
            f"期望采集帧率 {capture_fps} fps"
        )
        logger.info("操作说明：空格键 触发一次融合采样；采满两次后自动退出并保存 pcd1/pcd2。")

        _preview_and_capture(
            session=session,
            fusion_interval_s=fusion_interval_s,
            output_paths=[output_pcd1, output_pcd2],
        )


# endregion


# region 交互预览
def _preview_and_capture(session: Gemini305, fusion_interval_s: float, output_paths: list[Path]) -> None:
    capture_requested = {"flag": False}
    capture_index = 0

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Orbbec 实时预览（空格触发采样）",
        width=DEFAULT_WINDOW_WIDTH,
        height=DEFAULT_WINDOW_HEIGHT,
    )

    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = DEFAULT_POINT_SIZE
        render_opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(axis)
    vis.add_geometry(pcd)

    view = vis.get_view_control()
    view.set_lookat([0.0, 0.0, 0.0])
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])

    def _on_space(_vis: o3d.visualization.Visualizer) -> bool:
        capture_requested["flag"] = True
        return False

    vis.register_key_callback(32, _on_space)

    preview_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())

    try:
        while True:
            points = _capture_preview_points_once(session=session, point_filter=preview_filter)
            if points is not None and len(points) > 0:
                _update_point_cloud(pcd, points)
                vis.update_geometry(pcd)

            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                logger.warning("预览窗口关闭，提前结束采样。")
                break

            if not capture_requested["flag"]:
                continue

            capture_requested["flag"] = False
            if capture_index >= len(output_paths):
                continue

            target_path = output_paths[capture_index]
            logger.info(
                f"开始第 {capture_index + 1} 次融合采样：采样时长 {fusion_interval_s:.2f} 秒，输出文件 {target_path.name}"
            )
            fused = session.capture_fused_points_by_interval(fusion_interval_s=fusion_interval_s)
            fused_pcd = _points_to_point_cloud(fused)
            o3d.io.write_point_cloud(str(target_path), fused_pcd)
            logger.success(f"第 {capture_index + 1} 次采样完成，保存路径：{target_path}")
            capture_index += 1

            if capture_index >= len(output_paths):
                logger.success("两组点云采样完成，程序退出。")
                break
    finally:
        vis.destroy_window()


# endregion


# region 点云工具
def _capture_preview_points_once(session: Gemini305, point_filter) -> np.ndarray | None:
    frames = session.wait_for_frames()
    if frames is None:
        return None

    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None

    point_frames, use_color = session.prepare_frame_for_point_cloud(frames)
    set_point_cloud_filter_format(
        point_filter,
        depth_scale=float(depth_frame.get_depth_scale()),
        use_color=use_color,
    )
    cloud_frame = point_filter.process(point_frames)
    if cloud_frame is None:
        return None

    raw_points = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
    normalized = normalize_points(raw_points)
    valid_points, _ = filter_valid_points(normalized, max_depth_mm=DEFAULT_MAX_DEPTH_MM)
    if len(valid_points) == 0:
        return None

    return _downsample_points(valid_points, max_points=DEFAULT_MAX_PREVIEW_POINTS)


def _update_point_cloud(pcd: o3d.geometry.PointCloud, points: np.ndarray) -> None:
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if points.shape[1] >= 6:
        rgb = np.ascontiguousarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    else:
        gray = np.full((len(points), 3), 0.85, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(gray)


def _points_to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if points.size == 0:
        return pcd

    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if points.shape[1] >= 6:
        rgb = np.ascontiguousarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    return pcd


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


# endregion


# region CLI（仅用于覆盖调参）
def _parse_cli() -> tuple[float, Path, Path, int, int]:
    parser = argparse.ArgumentParser(description="Orbbec 动态预览 + 手动触发两组融合采样（CLI 仅用于覆盖调参）")
    parser.add_argument("--fusion-interval-s", type=float, default=DEFAULT_FUSION_INTERVAL_S, help="fusion interval in seconds")
    parser.add_argument("--pcd1", type=Path, default=DEFAULT_OUTPUT_PCD1, help="output path for pcd1")
    parser.add_argument("--pcd2", type=Path, default=DEFAULT_OUTPUT_PCD2, help="output path for pcd2")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames timeout in ms")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="preferred capture fps")
    args = parser.parse_args()
    return (
        float(args.fusion_interval_s),
        Path(args.pcd1),
        Path(args.pcd2),
        int(args.timeout_ms),
        int(args.capture_fps),
    )


# endregion


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            interval_arg, pcd1_arg, pcd2_arg, timeout_arg, fps_arg = _parse_cli()
            main(
                fusion_interval_s=interval_arg,
                output_pcd1=pcd1_arg,
                output_pcd2=pcd2_arg,
                timeout_ms=timeout_arg,
                capture_fps=fps_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise

