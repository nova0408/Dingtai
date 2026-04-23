from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import (
    OrbbecSession,
    SessionOptions,
    filter_valid_points,
    normalize_points,
    set_point_cloud_filter_format,
)


# region 默认参数（优先在这里直接改）
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_MAX_DEPTH_MM = 5000.0
DEFAULT_MAX_PREVIEW_POINTS = 120_000
DEFAULT_MAX_SEGMENT_POINTS = 80_000
DEFAULT_PLANE_DISTANCE_MM = 3.0
DEFAULT_PLANE_RANSAC_ITER = 110
DEFAULT_MAX_PLANES = 6
DEFAULT_MIN_INLIERS = 320
DEFAULT_ALPHA = 0.38
DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_POINT_SIZE = 1.5
DEFAULT_BACKGROUND_COLOR = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)
DEFAULT_BASE_COLOR = np.asarray([0.25, 0.25, 0.25], dtype=np.float64)
DEFAULT_TRACK_MAX_MISSED_FRAMES = 20
DEFAULT_TRACK_ANGLE_THRESHOLD_DEG = 12.0
DEFAULT_TRACK_D_THRESHOLD_MM = 12.0
DEFAULT_TRACK_CENTROID_DISTANCE_MM = 70.0
DEFAULT_TRACK_SMOOTHING = 0.35
DEFAULT_PLANE_PALETTE = [
    np.asarray([1.00, 0.36, 0.24], dtype=np.float64),
    np.asarray([0.19, 0.73, 0.91], dtype=np.float64),
    np.asarray([0.31, 0.83, 0.44], dtype=np.float64),
    np.asarray([0.99, 0.79, 0.24], dtype=np.float64),
    np.asarray([0.73, 0.45, 0.96], dtype=np.float64),
    np.asarray([0.98, 0.52, 0.74], dtype=np.float64),
]
# endregion


# region 平面跟踪
@dataclass
class PlaneTrack:
    track_id: int
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    last_seen_frame: int
    seen_count: int


class PlaneTracker:
    def __init__(
        self,
        max_missed_frames: int = DEFAULT_TRACK_MAX_MISSED_FRAMES,
        angle_threshold_deg: float = DEFAULT_TRACK_ANGLE_THRESHOLD_DEG,
        d_threshold_mm: float = DEFAULT_TRACK_D_THRESHOLD_MM,
        centroid_distance_mm: float = DEFAULT_TRACK_CENTROID_DISTANCE_MM,
        smoothing: float = DEFAULT_TRACK_SMOOTHING,
    ) -> None:
        self.max_missed_frames = int(max(1, max_missed_frames))
        self.angle_threshold_deg = float(max(1.0, angle_threshold_deg))
        self.d_threshold_mm = float(max(0.1, d_threshold_mm))
        self.centroid_distance_mm = float(max(1.0, centroid_distance_mm))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self._tracks: dict[int, PlaneTrack] = {}
        self._next_track_id = 0

    def assign(
        self,
        frame_idx: int,
        plane_infos: list[dict[str, np.ndarray | float]],
    ) -> list[int]:
        assigned_track_ids: list[int] = []
        used_tracks: set[int] = set()

        for info in plane_infos:
            matched = self._match_track(info=info, used_tracks=used_tracks)
            if matched is None:
                matched = self._create_track(frame_idx=frame_idx, info=info)
            else:
                self._update_track(track_id=matched, frame_idx=frame_idx, info=info)
            used_tracks.add(matched)
            assigned_track_ids.append(matched)

        self._purge_old_tracks(frame_idx=frame_idx)
        return assigned_track_ids

    def _match_track(self, info: dict[str, np.ndarray | float], used_tracks: set[int]) -> int | None:
        normal = np.asarray(info["normal"], dtype=np.float64)
        centroid = np.asarray(info["centroid"], dtype=np.float64)
        d_value = float(info["d"])

        best_track_id: int | None = None
        best_score = float("inf")

        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue

            dot_val = float(np.clip(np.dot(track.normal, normal), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(abs(dot_val))))
            d_diff = abs(track.d - d_value)
            centroid_dist = float(np.linalg.norm(track.centroid - centroid))

            if angle_deg > self.angle_threshold_deg:
                continue
            if d_diff > self.d_threshold_mm:
                continue
            if centroid_dist > self.centroid_distance_mm:
                continue

            score = angle_deg + 0.1 * d_diff + 0.02 * centroid_dist
            if score < best_score:
                best_score = score
                best_track_id = track_id

        return best_track_id

    def _create_track(self, frame_idx: int, info: dict[str, np.ndarray | float]) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = PlaneTrack(
            track_id=track_id,
            normal=np.asarray(info["normal"], dtype=np.float64).copy(),
            d=float(info["d"]),
            centroid=np.asarray(info["centroid"], dtype=np.float64).copy(),
            last_seen_frame=int(frame_idx),
            seen_count=1,
        )
        return track_id

    def _update_track(self, track_id: int, frame_idx: int, info: dict[str, np.ndarray | float]) -> None:
        track = self._tracks[track_id]
        new_normal = np.asarray(info["normal"], dtype=np.float64)
        if float(np.dot(track.normal, new_normal)) < 0.0:
            new_normal = -new_normal

        updated_normal = (1.0 - self.smoothing) * track.normal + self.smoothing * new_normal
        norm_val = float(np.linalg.norm(updated_normal))
        if norm_val > 1e-9:
            updated_normal /= norm_val

        track.normal = updated_normal
        track.d = (1.0 - self.smoothing) * track.d + self.smoothing * float(info["d"])
        track.centroid = (1.0 - self.smoothing) * track.centroid + self.smoothing * np.asarray(
            info["centroid"], dtype=np.float64
        )
        track.last_seen_frame = int(frame_idx)
        track.seen_count += 1

    def _purge_old_tracks(self, frame_idx: int) -> None:
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if int(frame_idx) - int(track.last_seen_frame) > self.max_missed_frames
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]


# endregion


# region 主流程
def main(
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    plane_distance_mm: float = DEFAULT_PLANE_DISTANCE_MM,
    plane_ransac_iter: int = DEFAULT_PLANE_RANSAC_ITER,
    max_planes: int = DEFAULT_MAX_PLANES,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    if plane_distance_mm <= 0:
        raise ValueError("plane_distance_mm must be > 0")
    if plane_ransac_iter < 10:
        raise ValueError("plane_ransac_iter must be >= 10")
    if max_planes < 1:
        raise ValueError("max_planes must be >= 1")
    if min_inliers < 50:
        raise ValueError("min_inliers must be >= 50")

    alpha = float(np.clip(alpha, 0.0, 1.0))

    session_options = SessionOptions(
        timeout_ms=int(timeout_ms),
        preferred_capture_fps=max(1, int(capture_fps)),
    )

    with OrbbecSession(options=session_options) as session:
        logger.info(
            f"启动实时平面识别：平面阈值 {plane_distance_mm:.2f} mm，"
            f"RANSAC迭代 {plane_ransac_iter} 次，最多平面数 {max_planes} 个，最小平面点数 {min_inliers} 点"
        )
        logger.info("可视化说明：基底点云为灰色；识别平面使用不同颜色混色显示，ESC 或关闭窗口退出。")
        _preview_and_segment(
            session=session,
            plane_distance_mm=float(plane_distance_mm),
            plane_ransac_iter=int(plane_ransac_iter),
            max_planes=int(max_planes),
            min_inliers=int(min_inliers),
            alpha=alpha,
        )


# endregion


# region 实时预览与分割
def _preview_and_segment(
    session: OrbbecSession,
    plane_distance_mm: float,
    plane_ransac_iter: int,
    max_planes: int,
    min_inliers: int,
    alpha: float,
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="Orbbec 每帧平面识别预览",
        width=DEFAULT_WINDOW_WIDTH,
        height=DEFAULT_WINDOW_HEIGHT,
    )

    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = DEFAULT_POINT_SIZE
        render_opt.background_color = DEFAULT_BACKGROUND_COLOR

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    base_pcd = o3d.geometry.PointCloud()
    plane_pcds: list[o3d.geometry.PointCloud] = []

    vis.add_geometry(axis)
    vis.add_geometry(base_pcd)

    view = vis.get_view_control()
    view.set_lookat([0.0, 0.0, 0.0])
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])

    stop_requested = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop_requested["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)

    point_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())

    frame_idx = 0
    last_plane_count = -1
    tracker = PlaneTracker()

    try:
        while True:
            if stop_requested["flag"]:
                break

            points = _capture_preview_points_once(session=session, point_filter=point_filter)
            if points is None or len(points) == 0:
                alive = vis.poll_events()
                vis.update_renderer()
                if not alive:
                    break
                continue

            frame_idx += 1
            xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)

            labels, plane_infos = _segment_planes_labels(
                xyz=xyz,
                max_points=DEFAULT_MAX_SEGMENT_POINTS,
                distance_mm=plane_distance_mm,
                ransac_iter=plane_ransac_iter,
                max_planes=max_planes,
                min_inliers=min_inliers,
            )
            track_ids = tracker.assign(frame_idx=frame_idx, plane_infos=plane_infos)
            stable_labels = _map_plane_labels_to_track_labels(
                labels=labels,
                plane_infos=plane_infos,
                assigned_track_ids=track_ids,
            )

            _update_base_cloud(base_pcd=base_pcd, xyz=xyz, labels=stable_labels, alpha=alpha)
            vis.update_geometry(base_pcd)

            for geom in plane_pcds:
                vis.remove_geometry(geom, reset_bounding_box=False)
            plane_pcds.clear()

            detected_planes = len(track_ids)
            for track_id in track_ids:
                mask = stable_labels == track_id
                if not np.any(mask):
                    continue
                overlay = o3d.geometry.PointCloud()
                overlay.points = o3d.utility.Vector3dVector(xyz[mask])
                overlay_color = _palette_color(track_id)
                overlay_rgb = np.tile(overlay_color.reshape(1, 3), (int(np.sum(mask)), 1))
                overlay.colors = o3d.utility.Vector3dVector(overlay_rgb)
                vis.add_geometry(overlay, reset_bounding_box=False)
                plane_pcds.append(overlay)

            if detected_planes != last_plane_count or frame_idx % 30 == 0:
                logger.info(f"帧 {frame_idx}：识别平面数 {detected_planes} 个")
                last_plane_count = detected_planes

            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break
    finally:
        vis.destroy_window()


# endregion


# region 分割与点云工具
def _capture_preview_points_once(session: OrbbecSession, point_filter) -> np.ndarray | None:
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


def _segment_planes_labels(
    xyz: np.ndarray,
    max_points: int,
    distance_mm: float,
    ransac_iter: int,
    max_planes: int,
    min_inliers: int,
) -> tuple[np.ndarray, list[dict[str, np.ndarray | float]]]:
    n = int(xyz.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int32), []

    if n > max_points:
        step = max(1, n // max_points)
        sample_idx = np.arange(0, n, step, dtype=np.int32)
    else:
        sample_idx = np.arange(0, n, dtype=np.int32)

    sampled_xyz = xyz[sample_idx]
    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(sampled_xyz)

    sampled_labels = np.full((sampled_xyz.shape[0],), -1, dtype=np.int32)
    sampled_active = np.arange(sampled_xyz.shape[0], dtype=np.int32)
    plane_infos_sampled: list[dict[str, np.ndarray | float]] = []

    for plane_id in range(max_planes):
        if sampled_active.shape[0] < min_inliers:
            break

        active_cloud = work.select_by_index(sampled_active.tolist())
        if len(active_cloud.points) < min_inliers:
            break

        model, inliers = active_cloud.segment_plane(
            distance_threshold=distance_mm,
            ransac_n=3,
            num_iterations=ransac_iter,
        )
        if len(inliers) < min_inliers:
            break

        inliers_arr = np.asarray(inliers, dtype=np.int32)
        global_inliers = sampled_active[inliers_arr]
        sampled_labels[global_inliers] = plane_id
        model_arr = np.asarray(model, dtype=np.float64)
        normal = model_arr[:3]
        norm_val = float(np.linalg.norm(normal))
        if norm_val > 1e-9:
            normal = normal / norm_val
        d_value = float(model_arr[3] / max(norm_val, 1e-9))
        centroid = np.mean(sampled_xyz[global_inliers], axis=0)
        plane_infos_sampled.append({"normal": normal, "d": d_value, "centroid": centroid})

        keep_mask = np.ones(sampled_active.shape[0], dtype=bool)
        keep_mask[inliers_arr] = False
        sampled_active = sampled_active[keep_mask]

    labels = np.full((n,), -1, dtype=np.int32)
    labels[sample_idx] = sampled_labels
    plane_infos_full: list[dict[str, np.ndarray | float]] = []
    for plane_id, info in enumerate(plane_infos_sampled):
        full_mask = labels == plane_id
        if not np.any(full_mask):
            continue
        full_centroid = np.mean(xyz[full_mask], axis=0)
        plane_infos_full.append(
            {
                "normal": np.asarray(info["normal"], dtype=np.float64),
                "d": float(info["d"]),
                "centroid": np.asarray(full_centroid, dtype=np.float64),
            }
        )
    if len(plane_infos_full) == 0:
        return labels, plane_infos_full

    dense_labels = np.full((n,), -1, dtype=np.int32)
    best_dist = np.full((n,), np.inf, dtype=np.float64)
    assign_threshold = distance_mm * 1.5
    for plane_id, info in enumerate(plane_infos_full):
        normal = np.asarray(info["normal"], dtype=np.float64)
        d_value = float(info["d"])
        dist = np.abs(xyz @ normal + d_value)
        candidate_mask = dist <= assign_threshold
        better_mask = candidate_mask & (dist < best_dist)
        dense_labels[better_mask] = plane_id
        best_dist[better_mask] = dist[better_mask]

    return dense_labels, plane_infos_full


def _map_plane_labels_to_track_labels(
    labels: np.ndarray,
    plane_infos: list[dict[str, np.ndarray | float]],
    assigned_track_ids: list[int],
) -> np.ndarray:
    stable_labels = np.full_like(labels, fill_value=-1)
    if labels.size == 0:
        return stable_labels
    for plane_id, _ in enumerate(plane_infos):
        if plane_id >= len(assigned_track_ids):
            continue
        stable_labels[labels == plane_id] = int(assigned_track_ids[plane_id])
    return stable_labels


def _update_base_cloud(base_pcd: o3d.geometry.PointCloud, xyz: np.ndarray, labels: np.ndarray, alpha: float) -> None:
    base_rgb = np.tile(DEFAULT_BASE_COLOR.reshape(1, 3), (xyz.shape[0], 1))
    valid_track_ids = np.unique(labels[labels >= 0])
    if valid_track_ids.size > 0:
        for track_id in valid_track_ids:
            mask = labels == int(track_id)
            if not np.any(mask):
                continue
            plane_color = _palette_color(int(track_id))
            # PointCloud 不支持 alpha，这里通过颜色混合模拟“半透明”。
            base_rgb[mask] = (1.0 - alpha) * base_rgb[mask] + alpha * plane_color.reshape(1, 3)

    base_pcd.points = o3d.utility.Vector3dVector(xyz)
    base_pcd.colors = o3d.utility.Vector3dVector(np.clip(base_rgb, 0.0, 1.0))


def _palette_color(index: int) -> np.ndarray:
    return DEFAULT_PLANE_PALETTE[index % len(DEFAULT_PLANE_PALETTE)]


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


# endregion


# region CLI（仅覆盖默认参数）
def _parse_cli() -> tuple[int, int, float, int, int, int, float]:
    parser = argparse.ArgumentParser(description="Orbbec 每帧快速平面识别与预览（CLI 仅用于覆盖调参）")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames timeout in ms")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="preferred capture fps")
    parser.add_argument("--plane-distance-mm", type=float, default=DEFAULT_PLANE_DISTANCE_MM, help="RANSAC 平面距离阈值（毫米）")
    parser.add_argument("--plane-ransac-iter", type=int, default=DEFAULT_PLANE_RANSAC_ITER, help="RANSAC 迭代次数")
    parser.add_argument("--max-planes", type=int, default=DEFAULT_MAX_PLANES, help="每帧最多识别平面数")
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS, help="单平面最小内点数")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="平面颜色混合权重 0~1")
    args = parser.parse_args()
    return (
        int(args.timeout_ms),
        int(args.capture_fps),
        float(args.plane_distance_mm),
        int(args.plane_ransac_iter),
        int(args.max_planes),
        int(args.min_inliers),
        float(args.alpha),
    )


# endregion


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            timeout_arg, fps_arg, dist_arg, iter_arg, planes_arg, inliers_arg, alpha_arg = _parse_cli()
            main(
                timeout_ms=timeout_arg,
                capture_fps=fps_arg,
                plane_distance_mm=dist_arg,
                plane_ransac_iter=iter_arg,
                max_planes=planes_arg,
                min_inliers=inliers_arg,
                alpha=alpha_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
