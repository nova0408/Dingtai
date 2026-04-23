from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 opencv-python 才能运行该脚本。") from exc

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
DEFAULT_TIMEOUT_MS = 120  # 等待帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求相机帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 5000.0  # 最大深度裁剪，单位 mm
DEFAULT_MAX_PREVIEW_POINTS = 90_000  # 预览阶段最多保留点数，直接影响实时速度
DEFAULT_MAX_SEGMENT_POINTS = 45_000  # 分割阶段采样点数上限，越小越快
DEFAULT_WINDOW_WIDTH = 1440  # 3D窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 3D窗口高度，单位 像素
DEFAULT_POINT_SIZE = 1.5  # 3D点大小
DEFAULT_BACKGROUND_COLOR = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)  # 3D背景色
DEFAULT_BASE_COLOR = np.asarray([0.22, 0.22, 0.22], dtype=np.float64)  # 未识别点默认颜色
DEFAULT_ALPHA = 0.42  # 平面颜色半透明混合权重
DEFAULT_2D_WINDOW_NAME = "Orbbec 平面RGB四边形预览"  # 2D预览窗口名

DEFAULT_PLANE_DISTANCE_MM = 2.0  # 平面RANSAC距离阈值，单位 mm（小平面优先）
DEFAULT_PLANE_RANSAC_ITER = 140  # 平面RANSAC迭代次数
DEFAULT_MAX_PLANES = 12  # 每帧最大平面数量
DEFAULT_MIN_INLIERS = 90  # 单平面最小内点数（降低后更易识别细长平面）
DEFAULT_MIN_INLIER_RATIO = 0.30  # 区域平面内点占比阈值

DEFAULT_CV2_RGB_CANNY_LOW = 24  # RGB边缘阈值低
DEFAULT_CV2_RGB_CANNY_HIGH = 64  # RGB边缘阈值高
DEFAULT_CV2_DEPTH_CANNY_LOW = 20  # 深度边缘阈值低
DEFAULT_CV2_DEPTH_CANNY_HIGH = 52  # 深度边缘阈值高
DEFAULT_CV2_DILATE_KERNEL = 1  # 边缘膨胀核（减小以保留细小结构）
DEFAULT_CV2_MIN_REGION_PIXELS = 70  # 连通域最小像素面积（降低以保留细长区域）

DEFAULT_TRACK_MAX_MISSED_FRAMES = 24  # 轨迹允许丢失帧数
DEFAULT_TRACK_ANGLE_THRESHOLD_DEG = 12.0  # 轨迹匹配法向夹角阈值
DEFAULT_TRACK_D_THRESHOLD_MM = 12.0  # 轨迹匹配平面d阈值，单位 mm
DEFAULT_TRACK_CENTROID_DISTANCE_MM = 75.0  # 轨迹匹配质心距离阈值，单位 mm
DEFAULT_TRACK_SMOOTHING = 0.35  # 轨迹指数平滑系数

DEFAULT_PLANE_PALETTE = [
    np.asarray([1.00, 0.35, 0.22], dtype=np.float64),
    np.asarray([0.16, 0.74, 0.93], dtype=np.float64),
    np.asarray([0.29, 0.83, 0.43], dtype=np.float64),
    np.asarray([0.99, 0.79, 0.23], dtype=np.float64),
    np.asarray([0.72, 0.45, 0.96], dtype=np.float64),
    np.asarray([0.97, 0.50, 0.76], dtype=np.float64),
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

    def assign(self, frame_idx: int, plane_infos: list[dict[str, np.ndarray | float]]) -> list[int]:
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
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
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
    min_inlier_ratio = float(np.clip(min_inlier_ratio, 0.05, 0.99))

    session_options = SessionOptions(
        timeout_ms=int(timeout_ms),
        preferred_capture_fps=max(1, int(capture_fps)),
    )

    with OrbbecSession(options=session_options) as session:
        camera_param = session.get_camera_param()
        ci = camera_param.rgb_intrinsic if session.has_color_sensor else camera_param.depth_intrinsic
        width = int(max(32, ci.width))
        height = int(max(32, ci.height))
        fx = float(ci.fx)
        fy = float(ci.fy)
        cx = float(ci.cx)
        cy = float(ci.cy)

        logger.info(
            f"启动 cv2+3D 混合平面识别：平面阈值 {plane_distance_mm:.2f} mm，RANSAC迭代 {plane_ransac_iter} 次，"
            f"最小平面点数 {min_inliers} 点，最小内点占比 {min_inlier_ratio:.2f}"
        )
        logger.info(f"投影尺寸 {width}x{height}，焦距 fx={fx:.2f}, fy={fy:.2f}")

        _preview_and_segment_hybrid(
            session=session,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_w=width,
            img_h=height,
            plane_distance_mm=plane_distance_mm,
            plane_ransac_iter=plane_ransac_iter,
            max_planes=max_planes,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            alpha=alpha,
        )


# endregion


# region 实时预览与分割
def _preview_and_segment_hybrid(
    session: OrbbecSession,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    plane_distance_mm: float,
    plane_ransac_iter: int,
    max_planes: int,
    min_inliers: int,
    min_inlier_ratio: float,
    alpha: float,
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Orbbec cv2+3D 混合平面识别预览", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)

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
    tracker = PlaneTracker()

    frame_idx = 0
    last_plane_count = -1
    fps_tick = time.perf_counter()

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
            rgb = _extract_rgb(points)

            labels, plane_infos = _segment_planes_cv2_hybrid(
                xyz=xyz,
                rgb=rgb,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_w=img_w,
                img_h=img_h,
                plane_distance_mm=plane_distance_mm,
                plane_ransac_iter=plane_ransac_iter,
                max_planes=max_planes,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )

            track_ids = tracker.assign(frame_idx=frame_idx, plane_infos=plane_infos)
            stable_labels = _map_plane_labels_to_track_labels(labels=labels, assigned_track_ids=track_ids)

            _update_base_cloud(base_pcd=base_pcd, xyz=xyz, labels=stable_labels, alpha=alpha)
            vis.update_geometry(base_pcd)

            overlay_img = _render_plane_rgb_overlay(
                xyz=xyz,
                rgb=rgb,
                labels=stable_labels,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_w=img_w,
                img_h=img_h,
                alpha=alpha,
            )
            cv2.imshow(DEFAULT_2D_WINDOW_NAME, overlay_img)
            key = cv2.waitKey(1)
            if key == 27:
                stop_requested["flag"] = True
            if cv2.getWindowProperty(DEFAULT_2D_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                stop_requested["flag"] = True

            for geom in plane_pcds:
                vis.remove_geometry(geom, reset_bounding_box=False)
            plane_pcds.clear()

            detected_planes = len(track_ids)
            for track_id in track_ids:
                mask = stable_labels == int(track_id)
                if not np.any(mask):
                    continue
                overlay = o3d.geometry.PointCloud()
                overlay.points = o3d.utility.Vector3dVector(xyz[mask])
                overlay_color = _palette_color(int(track_id))
                overlay_rgb = np.tile(overlay_color.reshape(1, 3), (int(np.sum(mask)), 1))
                overlay.colors = o3d.utility.Vector3dVector(overlay_rgb)
                vis.add_geometry(overlay, reset_bounding_box=False)
                plane_pcds.append(overlay)

            if detected_planes != last_plane_count or frame_idx % 30 == 0:
                now = time.perf_counter()
                dt = max(1e-6, now - fps_tick)
                fps_tick = now
                logger.info(f"帧 {frame_idx}：识别平面数 {detected_planes} 个，主循环帧率 {1.0 / dt:.2f} fps")
                last_plane_count = detected_planes

            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break
    finally:
        vis.destroy_window()
        cv2.destroyWindow(DEFAULT_2D_WINDOW_NAME)


# endregion


# region cv2+3D 混合分割
def _segment_planes_cv2_hybrid(
    xyz: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    plane_distance_mm: float,
    plane_ransac_iter: int,
    max_planes: int,
    min_inliers: int,
    min_inlier_ratio: float,
) -> tuple[np.ndarray, list[dict[str, np.ndarray | float]]]:
    n = int(xyz.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int32), []

    if n > DEFAULT_MAX_SEGMENT_POINTS:
        step = max(1, n // DEFAULT_MAX_SEGMENT_POINTS)
        sample_idx = np.arange(0, n, step, dtype=np.int32)
    else:
        sample_idx = np.arange(0, n, dtype=np.int32)

    sampled_xyz = xyz[sample_idx]
    sampled_rgb = rgb[sample_idx]

    uv, valid_proj = _project_points_to_image(sampled_xyz, fx=fx, fy=fy, cx=cx, cy=cy, w=img_w, h=img_h)
    if not np.any(valid_proj):
        return np.full((n,), -1, dtype=np.int32), []

    _, depth_img, gray_img, point_index_img, valid_img = _rasterize_projected_points(
        xyz=sampled_xyz,
        rgb=sampled_rgb,
        uv=uv,
        valid_proj=valid_proj,
        w=img_w,
        h=img_h,
    )

    region_point_sets = _build_cv2_prior_regions(
        depth_img=depth_img,
        gray_img=gray_img,
        point_index_img=point_index_img,
        valid_img=valid_img,
    )

    labels_sampled = np.full((sampled_xyz.shape[0],), -1, dtype=np.int32)
    plane_infos: list[dict[str, np.ndarray | float]] = []

    for point_indices in region_point_sets:
        if len(plane_infos) >= max_planes:
            break
        if point_indices.size < min_inliers:
            continue

        candidate_xyz = sampled_xyz[point_indices]
        plane_info, inliers_local = _fit_plane_on_candidate(
            candidate_xyz=candidate_xyz,
            plane_distance_mm=plane_distance_mm,
            plane_ransac_iter=plane_ransac_iter,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
        )
        if plane_info is None:
            continue

        plane_id = len(plane_infos)
        global_inliers = point_indices[inliers_local]
        labels_sampled[global_inliers] = plane_id
        plane_infos.append(plane_info)

    # 回退：若 cv2 区域不足，继续在剩余点上全局补平面
    active = np.where(labels_sampled < 0)[0]
    while len(plane_infos) < max_planes and active.size >= min_inliers:
        work = o3d.geometry.PointCloud()
        work.points = o3d.utility.Vector3dVector(sampled_xyz[active])

        model, inliers = work.segment_plane(
            distance_threshold=plane_distance_mm,
            ransac_n=3,
            num_iterations=plane_ransac_iter,
        )
        if len(inliers) < min_inliers:
            break

        inliers_arr = np.asarray(inliers, dtype=np.int32)
        if float(inliers_arr.size) / float(active.size) < max(min_inlier_ratio * 0.4, 0.10):
            break

        model_arr = np.asarray(model, dtype=np.float64)
        normal = model_arr[:3]
        norm_val = float(np.linalg.norm(normal))
        if norm_val <= 1e-9:
            break
        normal = normal / norm_val
        d_value = float(model_arr[3] / norm_val)

        plane_id = len(plane_infos)
        global_inliers = active[inliers_arr]
        labels_sampled[global_inliers] = plane_id
        plane_infos.append(
            {
                "normal": normal,
                "d": d_value,
                "centroid": np.mean(sampled_xyz[global_inliers], axis=0),
            }
        )

        keep_mask = np.ones(active.shape[0], dtype=bool)
        keep_mask[inliers_arr] = False
        active = active[keep_mask]

    labels = np.full((n,), -1, dtype=np.int32)
    labels[sample_idx] = labels_sampled

    if len(plane_infos) == 0:
        return labels, []

    dense_labels = np.full((n,), -1, dtype=np.int32)
    best_dist = np.full((n,), np.inf, dtype=np.float64)
    assign_threshold = plane_distance_mm * 1.6

    dense_infos: list[dict[str, np.ndarray | float]] = []
    for plane_id, info in enumerate(plane_infos):
        normal = np.asarray(info["normal"], dtype=np.float64)
        d_value = float(info["d"])
        dist = np.abs(xyz @ normal + d_value)
        candidate_mask = dist <= assign_threshold
        better_mask = candidate_mask & (dist < best_dist)
        dense_labels[better_mask] = plane_id
        best_dist[better_mask] = dist[better_mask]

        plane_mask = dense_labels == plane_id
        if np.any(plane_mask):
            dense_infos.append(
                {
                    "normal": normal,
                    "d": d_value,
                    "centroid": np.mean(xyz[plane_mask], axis=0),
                }
            )

    return dense_labels, dense_infos


def _build_cv2_prior_regions(
    depth_img: np.ndarray,
    gray_img: np.ndarray,
    point_index_img: np.ndarray,
    valid_img: np.ndarray,
) -> list[np.ndarray]:
    rgb_edges = cv2.Canny(gray_img, DEFAULT_CV2_RGB_CANNY_LOW, DEFAULT_CV2_RGB_CANNY_HIGH)

    depth_u8 = _depth_to_u8(depth_img=depth_img, valid_img=valid_img)
    depth_edges = cv2.Canny(depth_u8, DEFAULT_CV2_DEPTH_CANNY_LOW, DEFAULT_CV2_DEPTH_CANNY_HIGH)

    edge = cv2.bitwise_or(rgb_edges, depth_edges)
    kernel = np.ones((DEFAULT_CV2_DILATE_KERNEL, DEFAULT_CV2_DILATE_KERNEL), dtype=np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)

    candidate = (valid_img > 0) & (edge == 0)
    binary = np.zeros_like(gray_img, dtype=np.uint8)
    binary[candidate] = 255

    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: list[np.ndarray] = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < DEFAULT_CV2_MIN_REGION_PIXELS:
            continue

        region_mask = label_img == label_id
        point_ids = point_index_img[region_mask]
        point_ids = point_ids[point_ids >= 0]
        if point_ids.size == 0:
            continue

        unique_ids = np.unique(point_ids)
        regions.append(unique_ids.astype(np.int32))

    thin_regions = _collect_thin_strip_regions(edge=edge, point_index_img=point_index_img, valid_img=valid_img)
    regions.extend(thin_regions)

    regions.sort(key=lambda arr: int(arr.size), reverse=True)
    dedup_regions: list[np.ndarray] = []
    seen_signatures: set[tuple[int, int]] = set()
    for reg in regions:
        if reg.size == 0:
            continue
        signature = (int(reg[0]), int(reg[-1]))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        dedup_regions.append(reg)
    return dedup_regions


def _collect_thin_strip_regions(edge: np.ndarray, point_index_img: np.ndarray, valid_img: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[np.ndarray] = []
    for contour in contours:
        if contour.shape[0] < 8:
            continue
        rect = cv2.minAreaRect(contour)
        (w_rect, h_rect) = rect[1]
        long_side = float(max(w_rect, h_rect))
        short_side = float(min(w_rect, h_rect))
        if short_side < 1.0:
            continue
        aspect = long_side / short_side
        if aspect < 3.0:
            continue
        if long_side < 18.0:
            continue

        box = cv2.boxPoints(rect).astype(np.int32)
        mask = np.zeros_like(valid_img, dtype=np.uint8)
        cv2.fillConvexPoly(mask, box, 255)
        point_ids = point_index_img[mask > 0]
        point_ids = point_ids[point_ids >= 0]
        if point_ids.size < DEFAULT_CV2_MIN_REGION_PIXELS:
            continue
        unique_ids = np.unique(point_ids).astype(np.int32)
        if unique_ids.size > 0:
            regions.append(unique_ids)
    return regions


def _fit_plane_on_candidate(
    candidate_xyz: np.ndarray,
    plane_distance_mm: float,
    plane_ransac_iter: int,
    min_inliers: int,
    min_inlier_ratio: float,
) -> tuple[dict[str, np.ndarray | float] | None, np.ndarray | None]:
    if candidate_xyz.shape[0] < min_inliers:
        return None, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(candidate_xyz)

    model, inliers = pcd.segment_plane(
        distance_threshold=plane_distance_mm,
        ransac_n=3,
        num_iterations=plane_ransac_iter,
    )
    inliers_arr = np.asarray(inliers, dtype=np.int32)
    if inliers_arr.size < min_inliers:
        return None, None

    inlier_ratio = float(inliers_arr.size) / float(candidate_xyz.shape[0])
    if inlier_ratio < min_inlier_ratio:
        return None, None

    model_arr = np.asarray(model, dtype=np.float64)
    normal = model_arr[:3]
    norm_val = float(np.linalg.norm(normal))
    if norm_val <= 1e-9:
        return None, None
    normal = normal / norm_val
    d_value = float(model_arr[3] / norm_val)

    info: dict[str, np.ndarray | float] = {
        "normal": normal,
        "d": d_value,
        "centroid": np.mean(candidate_xyz[inliers_arr], axis=0),
    }
    return info, inliers_arr


# endregion


# region 点云与投影工具
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


def _extract_rgb(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 6:
        rgb = np.asarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0)
    return np.full((points.shape[0], 3), 0.7, dtype=np.float32)


def _project_points_to_image(
    xyz: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    z = xyz[:, 2]
    valid_z = z > 1e-6

    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)

    if np.any(valid_z):
        x = xyz[valid_z, 0]
        y = xyz[valid_z, 1]
        zz = z[valid_z]
        uu = np.rint(fx * x / zz + cx).astype(np.int32)
        vv = np.rint(fy * y / zz + cy).astype(np.int32)

        in_bounds = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        valid_indices = np.where(valid_z)[0]
        target_indices = valid_indices[in_bounds]
        u[target_indices] = uu[in_bounds]
        v[target_indices] = vv[in_bounds]

    uv = np.stack([u, v], axis=1)
    valid_proj = (u >= 0) & (v >= 0)
    return uv, valid_proj


def _rasterize_projected_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    uv: np.ndarray,
    valid_proj: np.ndarray,
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    depth_img = np.zeros((h, w), dtype=np.float32)
    gray_img = np.zeros((h, w), dtype=np.uint8)
    point_index_img = np.full((h, w), -1, dtype=np.int32)
    valid_img = np.zeros((h, w), dtype=np.uint8)

    idxs = np.where(valid_proj)[0]
    if idxs.size == 0:
        return rgb_img, depth_img, gray_img, point_index_img, valid_img

    # 同像素保留最近深度点，使用向量化代替逐点循环提升速度。
    u = uv[idxs, 0].astype(np.int32)
    v = uv[idxs, 1].astype(np.int32)
    z = xyz[idxs, 2].astype(np.float32)
    linear = v * int(w) + u

    order = np.lexsort((z, linear))
    linear_sorted = linear[order]
    idxs_sorted = idxs[order]
    first = np.unique(linear_sorted, return_index=True)[1]
    chosen = idxs_sorted[first]

    u_c = uv[chosen, 0].astype(np.int32)
    v_c = uv[chosen, 1].astype(np.int32)
    depth_img[v_c, u_c] = xyz[chosen, 2].astype(np.float32)

    gray_vals = np.clip(
        255.0 * np.sum(rgb[chosen] * np.asarray([0.299, 0.587, 0.114], dtype=np.float32).reshape(1, 3), axis=1),
        0.0,
        255.0,
    ).astype(np.uint8)
    gray_img[v_c, u_c] = gray_vals

    rgb_u8 = np.clip(rgb[chosen] * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb_img[v_c, u_c, :] = rgb_u8[:, ::-1]  # BGR

    point_index_img[v_c, u_c] = chosen.astype(np.int32)
    valid_img[v_c, u_c] = 255
    return rgb_img, depth_img, gray_img, point_index_img, valid_img


def _depth_to_u8(depth_img: np.ndarray, valid_img: np.ndarray) -> np.ndarray:
    out = np.zeros_like(valid_img, dtype=np.uint8)
    valid_mask = valid_img > 0
    if not np.any(valid_mask):
        return out

    depth_values = depth_img[valid_mask]
    d_min = float(np.min(depth_values))
    d_max = float(np.max(depth_values))
    if d_max - d_min < 1e-6:
        out[valid_mask] = 128
        return out

    normalized = (depth_values - d_min) / (d_max - d_min)
    out[valid_mask] = np.asarray(np.clip(normalized * 255.0, 0.0, 255.0), dtype=np.uint8)
    return out


def _render_plane_rgb_overlay(
    xyz: np.ndarray,
    rgb: np.ndarray,
    labels: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    alpha: float,
) -> np.ndarray:
    uv, valid_proj = _project_points_to_image(xyz, fx=fx, fy=fy, cx=cx, cy=cy, w=img_w, h=img_h)
    rgb_img, _, _, _, valid_img = _rasterize_projected_points(
        xyz=xyz,
        rgb=rgb,
        uv=uv,
        valid_proj=valid_proj,
        w=img_w,
        h=img_h,
    )
    base = rgb_img.copy()
    overlay = rgb_img.copy()

    track_ids = np.unique(labels[labels >= 0])
    for track_id in track_ids:
        point_mask = (labels == int(track_id)) & valid_proj
        if int(np.count_nonzero(point_mask)) < 20:
            continue
        pts = uv[point_mask].astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.int32)
        color_bgr = tuple(int(v) for v in np.clip(_palette_color(int(track_id))[::-1] * 255.0, 0, 255).astype(np.uint8))
        cv2.fillConvexPoly(overlay, box, color_bgr)
        cv2.polylines(overlay, [box], True, (255, 255, 255), 1, cv2.LINE_AA)
        center = np.mean(box, axis=0).astype(np.int32)
        cv2.putText(
            overlay,
            f"P{int(track_id)}",
            (int(center[0]), int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    blended = cv2.addWeighted(overlay, float(alpha), base, float(1.0 - alpha), 0.0)
    blended[valid_img == 0] = 0
    return blended


def _map_plane_labels_to_track_labels(labels: np.ndarray, assigned_track_ids: list[int]) -> np.ndarray:
    stable_labels = np.full_like(labels, fill_value=-1)
    if labels.size == 0:
        return stable_labels

    for plane_id, track_id in enumerate(assigned_track_ids):
        stable_labels[labels == plane_id] = int(track_id)
    return stable_labels


def _update_base_cloud(base_pcd: o3d.geometry.PointCloud, xyz: np.ndarray, labels: np.ndarray, alpha: float) -> None:
    base_rgb = np.tile(DEFAULT_BASE_COLOR.reshape(1, 3), (xyz.shape[0], 1))
    track_ids = np.unique(labels[labels >= 0])

    for track_id in track_ids:
        mask = labels == int(track_id)
        if not np.any(mask):
            continue
        color = _palette_color(int(track_id))
        base_rgb[mask] = (1.0 - alpha) * base_rgb[mask] + alpha * color.reshape(1, 3)

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
def _parse_cli() -> tuple[int, int, float, int, int, int, float, float]:
    parser = argparse.ArgumentParser(description="Orbbec cv2+3D 混合平面识别（CLI 仅用于覆盖调参）")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames timeout in ms")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="preferred capture fps")
    parser.add_argument("--plane-distance-mm", type=float, default=DEFAULT_PLANE_DISTANCE_MM, help="RANSAC 平面距离阈值（毫米）")
    parser.add_argument("--plane-ransac-iter", type=int, default=DEFAULT_PLANE_RANSAC_ITER, help="RANSAC 迭代次数")
    parser.add_argument("--max-planes", type=int, default=DEFAULT_MAX_PLANES, help="每帧最多识别平面数")
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS, help="单平面最小内点数")
    parser.add_argument("--min-inlier-ratio", type=float, default=DEFAULT_MIN_INLIER_RATIO, help="候选区域最小内点占比")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="平面颜色混合权重 0~1")
    args = parser.parse_args()
    return (
        int(args.timeout_ms),
        int(args.capture_fps),
        float(args.plane_distance_mm),
        int(args.plane_ransac_iter),
        int(args.max_planes),
        int(args.min_inliers),
        float(args.min_inlier_ratio),
        float(args.alpha),
    )


# endregion


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            timeout_arg, fps_arg, dist_arg, iter_arg, planes_arg, inliers_arg, inlier_ratio_arg, alpha_arg = _parse_cli()
            main(
                timeout_ms=timeout_arg,
                capture_fps=fps_arg,
                plane_distance_mm=dist_arg,
                plane_ransac_iter=iter_arg,
                max_planes=planes_arg,
                min_inliers=inliers_arg,
                min_inlier_ratio=inlier_ratio_arg,
                alpha=alpha_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
