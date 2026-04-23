from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 opencv-python 才能运行该脚本。") from exc

from pyorbbecsdk import OBFormat

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
DEFAULT_LABEL_IMAGE = PROJECT_ROOT / "experiments" / "sampled_images" / "color" / "sample_0002_20260423_103758.png"  # 标注图路径
DEFAULT_TIMEOUT_MS = 120  # 等待帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 5000.0  # 深度截断上限，单位 mm
DEFAULT_MAX_PREVIEW_POINTS = 80_000  # 预览点上限，越小越快

DEFAULT_PLANE_DISTANCE_MM = 2.2  # 平面RANSAC距离阈值，单位 mm
DEFAULT_PLANE_RANSAC_ITER = 140  # 平面RANSAC迭代次数
DEFAULT_MIN_INLIERS = 90  # 单平面最小内点数
DEFAULT_MIN_INLIER_RATIO = 0.22  # 掩码区域最小内点占比
DEFAULT_NORMAL_ALIGN_MIN = 0.72  # 平面法线一致性最小阈值（点云辅助验证）
DEFAULT_MAX_NORMAL_DOT = 0.45  # 红绿平面法线最大点积绝对值（小于该值表示明显不平行）
DEFAULT_ALPHA = 0.42  # 平面叠加半透明权重

DEFAULT_MATCH_MIN_INLIERS = 18  # Homography 最小内点数
DEFAULT_MATCH_MIN_GOOD = 12  # Homography 最小有效匹配数
DEFAULT_MATCH_UPDATE_INTERVAL = 1  # 每 N 帧更新一次 Homography（1 表示每帧重识别）
DEFAULT_MASK_MIN_PIXELS = 120  # 掩码最小有效像素数

DEFAULT_WINDOW_WIDTH = 1440  # 3D窗口宽度
DEFAULT_WINDOW_HEIGHT = 900  # 3D窗口高度
DEFAULT_POINT_SIZE = 1.5  # 3D点大小
DEFAULT_BACKGROUND_COLOR = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)  # 3D背景色
DEFAULT_BASE_COLOR = np.asarray([0.22, 0.22, 0.22], dtype=np.float64)  # 基础点云颜色
DEFAULT_2D_WINDOW_NAME = "Orbbec 标注引导平面预览"  # 2D窗口名
# endregion


# region 模板对齐
class MaskGuidedWarper:
    def __init__(self, label_bgr: np.ndarray) -> None:
        self.label_bgr = label_bgr
        self.h, self.w = label_bgr.shape[:2]
        self.ref_gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)
        self.red_mask_ref, self.green_mask_ref = _extract_red_green_masks(label_bgr)
        self.orb = cv2.ORB_create(1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.red_feat_mask = self._build_feature_mask(self.red_mask_ref)
        self.green_feat_mask = self._build_feature_mask(self.green_mask_ref)
        self.ref_kp_red, self.ref_desc_red = self.orb.detectAndCompute(self.ref_gray, self.red_feat_mask)
        self.ref_kp_green, self.ref_desc_green = self.orb.detectAndCompute(self.ref_gray, self.green_feat_mask)

        self.h_ref_to_cur_red = np.eye(3, dtype=np.float64)
        self.h_ref_to_cur_green = np.eye(3, dtype=np.float64)
        self.last_inliers_red = 0
        self.last_inliers_green = 0
        self.red_ok = False
        self.green_ok = False

    def _build_feature_mask(self, plane_mask: np.ndarray) -> np.ndarray:
        # 特征优先取平面边界与周边上下文，降低纯色平面（尤其红色）特征不足问题。
        dilate_big = cv2.dilate(plane_mask, np.ones((31, 31), dtype=np.uint8), iterations=1)
        erode_small = cv2.erode(plane_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        ring = cv2.subtract(dilate_big, erode_small)
        feat_mask = cv2.bitwise_or(plane_mask, ring)
        return feat_mask

    def _update_plane_h(
        self,
        current_gray: np.ndarray,
        ref_kp,
        ref_desc,
    ) -> tuple[np.ndarray, int, bool]:
        if ref_desc is None or ref_kp is None or len(ref_kp) < 10:
            return np.eye(3, dtype=np.float64), 0, False

        kp2, desc2 = self.orb.detectAndCompute(current_gray, None)
        if desc2 is None or kp2 is None or len(kp2) < 10:
            return np.eye(3, dtype=np.float64), 0, False

        knn = self.matcher.knnMatch(ref_desc, desc2, k=2)
        good: list[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < DEFAULT_MATCH_MIN_GOOD:
            # 放宽一次匹配阈值，提高红平面等弱纹理区域的重定位概率。
            good = []
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < 0.85 * n.distance:
                    good.append(m)
        if len(good) < DEFAULT_MATCH_MIN_GOOD:
            return np.eye(3, dtype=np.float64), 0, False

        src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        h, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if h is None or inlier_mask is None:
            return np.eye(3, dtype=np.float64), 0, False
        inliers = int(np.sum(inlier_mask))
        if inliers < DEFAULT_MATCH_MIN_INLIERS:
            return np.eye(3, dtype=np.float64), inliers, False
        return h.astype(np.float64), inliers, True

    def update_homography(self, current_bgr: np.ndarray) -> bool:
        cur = cv2.resize(current_bgr, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        h_red, in_red, red_ok = self._update_plane_h(
            current_gray=cur_gray,
            ref_kp=self.ref_kp_red,
            ref_desc=self.ref_desc_red,
        )
        h_green, in_green, green_ok = self._update_plane_h(
            current_gray=cur_gray,
            ref_kp=self.ref_kp_green,
            ref_desc=self.ref_desc_green,
        )
        self.h_ref_to_cur_red = h_red
        self.h_ref_to_cur_green = h_green
        self.last_inliers_red = in_red
        self.last_inliers_green = in_green
        self.red_ok = red_ok
        self.green_ok = green_ok
        return self.red_ok or self.green_ok

    def warp_masks(self, target_w: int, target_h: int) -> tuple[np.ndarray, np.ndarray]:
        if self.red_ok:
            red = cv2.warpPerspective(
                self.red_mask_ref,
                self.h_ref_to_cur_red,
                (target_w, target_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            red = np.zeros((target_h, target_w), dtype=np.uint8)
        if self.green_ok:
            green = cv2.warpPerspective(
                self.green_mask_ref,
                self.h_ref_to_cur_green,
                (target_w, target_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            green = np.zeros((target_h, target_w), dtype=np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=1)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=1)
        return red, green


# endregion


# region 主流程
def main(
    label_image: Path = DEFAULT_LABEL_IMAGE,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    plane_distance_mm: float = DEFAULT_PLANE_DISTANCE_MM,
    plane_ransac_iter: int = DEFAULT_PLANE_RANSAC_ITER,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    label_image = Path(label_image)
    if not label_image.exists():
        raise FileNotFoundError(f"标注图不存在：{label_image}")

    label_bgr = cv2.imread(str(label_image), cv2.IMREAD_COLOR)
    if label_bgr is None:
        raise RuntimeError(f"无法读取标注图：{label_image}")

    warper = MaskGuidedWarper(label_bgr=label_bgr)
    if int(np.count_nonzero(warper.red_mask_ref)) < DEFAULT_MASK_MIN_PIXELS:
        raise RuntimeError("标注图中红色区域像素过少，无法作为平面先验。")
    if int(np.count_nonzero(warper.green_mask_ref)) < DEFAULT_MASK_MIN_PIXELS:
        raise RuntimeError("标注图中绿色区域像素过少，无法作为平面先验。")

    options = SessionOptions(timeout_ms=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))

    with OrbbecSession(options=options) as session:
        cam = session.get_camera_param()
        ci = cam.rgb_intrinsic if session.has_color_sensor else cam.depth_intrinsic
        img_w = int(max(32, ci.width))
        img_h = int(max(32, ci.height))
        fx = float(ci.fx)
        fy = float(ci.fy)
        cx = float(ci.cx)
        cy = float(ci.cy)

        logger.info(f"启动标注引导平面识别：标注图 {label_image.name}")
        logger.info(
            f"参数：plane_distance_mm {plane_distance_mm:.2f} mm, plane_ransac_iter {plane_ransac_iter}, "
            f"min_inliers {min_inliers}, min_inlier_ratio {min_inlier_ratio:.2f}"
        )

        _run_loop(
            session=session,
            warper=warper,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_w=img_w,
            img_h=img_h,
            plane_distance_mm=plane_distance_mm,
            plane_ransac_iter=plane_ransac_iter,
            min_inliers=min_inliers,
            min_inlier_ratio=min_inlier_ratio,
            alpha=float(np.clip(alpha, 0.0, 1.0)),
        )


# endregion


# region 实时循环
def _run_loop(
    session: OrbbecSession,
    warper: MaskGuidedWarper,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    plane_distance_mm: float,
    plane_ransac_iter: int,
    min_inliers: int,
    min_inlier_ratio: float,
    alpha: float,
) -> None:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Orbbec 标注引导3D平面结果", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = DEFAULT_POINT_SIZE
        render_opt.background_color = DEFAULT_BACKGROUND_COLOR

    base_pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(axis)
    vis.add_geometry(base_pcd)

    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)

    stop = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)

    point_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())

    frame_idx = 0
    try:
        while True:
            if stop["flag"]:
                break

            points, color_bgr = _capture_preview_with_color_once(session=session, point_filter=point_filter)
            if points is None or len(points) == 0:
                alive = vis.poll_events()
                vis.update_renderer()
                if not alive:
                    break
                continue

            frame_idx += 1
            xyz = np.asarray(points[:, :3], dtype=np.float64)
            rgb = _extract_rgb(points)
            uv, valid_proj = _project_points_to_image(xyz=xyz, fx=fx, fy=fy, cx=cx, cy=cy, w=img_w, h=img_h)
            rgb_img = _rasterize_rgb(xyz=xyz, rgb=rgb, uv=uv, valid_proj=valid_proj, w=img_w, h=img_h)

            if color_bgr is not None:
                base_2d = cv2.resize(color_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            else:
                base_2d = rgb_img

            # 纯识别：每帧都基于参考模板重识别（不使用跨帧跟踪状态）
            color_for_match = cv2.resize(base_2d, (warper.w, warper.h), interpolation=cv2.INTER_LINEAR)
            warper.update_homography(current_bgr=color_for_match)
            red_mask_prior, green_mask_prior = warper.warp_masks(target_w=img_w, target_h=img_h)

            labels = np.full((xyz.shape[0],), -1, dtype=np.int32)
            plane_quads: dict[int, np.ndarray] = {}
            source_tags: dict[int, str] = {}

            red_ids = _collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=red_mask_prior)
            red_fit = _fit_plane_indices(
                xyz=xyz,
                candidate_indices=red_ids,
                distance_mm=plane_distance_mm,
                ransac_iter=plane_ransac_iter,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )
            red_normal: np.ndarray | None = None
            if red_fit is not None:
                red_inliers, red_normal = red_fit
                labels[red_inliers] = 0
                plane_quads[0] = _sanitize_quad(_min_area_box_from_uv(uv[red_inliers]), w=img_w, h=img_h).astype(np.int32)
                source_tags[0] = "recognition"

            green_ids = _collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=green_mask_prior)
            green_fit = _fit_plane_indices(
                xyz=xyz,
                candidate_indices=green_ids,
                distance_mm=plane_distance_mm,
                ransac_iter=plane_ransac_iter,
                min_inliers=min_inliers,
                min_inlier_ratio=min_inlier_ratio,
            )
            green_normal: np.ndarray | None = None
            if green_fit is not None:
                green_inliers, green_normal = green_fit
                labels[green_inliers] = 1
                plane_quads[1] = _sanitize_quad(_min_area_box_from_uv(uv[green_inliers]), w=img_w, h=img_h).astype(np.int32)
                source_tags[1] = "recognition"

            if red_normal is not None and green_normal is not None:
                dot_val = float(abs(np.dot(red_normal, green_normal)))
                if dot_val > DEFAULT_MAX_NORMAL_DOT:
                    red_count = int(np.sum(labels == 0))
                    green_count = int(np.sum(labels == 1))
                    # 两平面不应近似平行：保留更可靠的一侧，另一侧判为无效等待下一帧重识别。
                    if red_count >= green_count:
                        labels[labels == 1] = -1
                        plane_quads.pop(1, None)
                        source_tags[1] = "rejected_parallel"
                    else:
                        labels[labels == 0] = -1
                        plane_quads.pop(0, None)
                        source_tags[0] = "rejected_parallel"

            _update_3d_cloud(base_pcd=base_pcd, xyz=xyz, labels=labels, alpha=alpha)
            vis.update_geometry(base_pcd)

            overlay = _draw_2d_overlay(rgb_img=base_2d, plane_quads=plane_quads, alpha=alpha)
            cv2.imshow(DEFAULT_2D_WINDOW_NAME, overlay)
            key = cv2.waitKey(1)
            if key == 27:
                stop["flag"] = True

            if frame_idx % 30 == 0:
                red_cnt = int(np.sum(labels == 0))
                green_cnt = int(np.sum(labels == 1))
                logger.info(
                    f"帧 {frame_idx}：red_inliers {red_cnt} ({source_tags.get(0, 'none')}), "
                    f"green_inliers {green_cnt} ({source_tags.get(1, 'none')}), "
                    f"match_inliers red={warper.last_inliers_red}, green={warper.last_inliers_green}"
                )

            alive = vis.poll_events()
            vis.update_renderer()
            if not alive:
                break
            if cv2.getWindowProperty(DEFAULT_2D_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        vis.destroy_window()
        cv2.destroyWindow(DEFAULT_2D_WINDOW_NAME)


# endregion


# region 识别工具
def _extract_red_green_masks(label_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2HSV)

    red_1 = cv2.inRange(hsv, np.asarray([0, 80, 60], dtype=np.uint8), np.asarray([12, 255, 255], dtype=np.uint8))
    red_2 = cv2.inRange(hsv, np.asarray([168, 80, 60], dtype=np.uint8), np.asarray([179, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(red_1, red_2)

    green = cv2.inRange(hsv, np.asarray([35, 60, 50], dtype=np.uint8), np.asarray([90, 255, 255], dtype=np.uint8))

    kernel = np.ones((3, 3), dtype=np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel, iterations=1)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel, iterations=1)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel, iterations=1)
    return red, green


def _collect_indices_in_mask(uv: np.ndarray, valid_proj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    idx = np.where(valid_proj)[0]
    if idx.size == 0:
        return np.empty((0,), dtype=np.int32)
    u = uv[idx, 0]
    v = uv[idx, 1]
    inside = mask[v, u] > 0
    return idx[inside].astype(np.int32)


def _fit_plane_indices(
    xyz: np.ndarray,
    candidate_indices: np.ndarray,
    distance_mm: float,
    ransac_iter: int,
    min_inliers: int,
    min_inlier_ratio: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if candidate_indices.size < min_inliers:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[candidate_indices])
    model, inliers = pcd.segment_plane(
        distance_threshold=distance_mm,
        ransac_n=3,
        num_iterations=ransac_iter,
    )
    inliers_arr = np.asarray(inliers, dtype=np.int32)
    if inliers_arr.size < min_inliers:
        return None
    if float(inliers_arr.size) / float(candidate_indices.size) < min_inlier_ratio:
        return None
    plane_normal = np.asarray(model[:3], dtype=np.float64)
    norm_val = float(np.linalg.norm(plane_normal))
    if norm_val <= 1e-9:
        return None
    plane_normal /= norm_val
    inlier_xyz = xyz[candidate_indices[inliers_arr]]
    if inlier_xyz.shape[0] >= 30:
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inlier_xyz)
        inlier_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=max(distance_mm * 6.0, 8.0), max_nn=30)
        )
        normals = np.asarray(inlier_pcd.normals, dtype=np.float64)
        if normals.shape[0] > 0:
            align = np.abs(normals @ plane_normal.reshape(3))
            if float(np.median(align)) < DEFAULT_NORMAL_ALIGN_MIN:
                return None
    return candidate_indices[inliers_arr], plane_normal


def _min_area_box_from_uv(uv: np.ndarray) -> np.ndarray:
    if uv.shape[0] < 4:
        return np.empty((0, 2), dtype=np.int32)
    rect = cv2.minAreaRect(uv.astype(np.float32))
    box = cv2.boxPoints(rect)
    return box.astype(np.int32)


def _sanitize_quad(quad: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return np.empty((0, 2), dtype=np.float32)

    hull = cv2.convexHull(pts).reshape(-1, 2)
    if hull.shape[0] != 4:
        rect = cv2.minAreaRect(pts)
        hull = cv2.boxPoints(rect).reshape(-1, 2).astype(np.float32)

    center = np.mean(hull, axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = hull[order]
    ordered[:, 0] = np.clip(ordered[:, 0], 0, max(0, w - 1))
    ordered[:, 1] = np.clip(ordered[:, 1], 0, max(0, h - 1))
    return ordered.astype(np.float32)


# endregion


# region 可视化工具
def _update_3d_cloud(base_pcd: o3d.geometry.PointCloud, xyz: np.ndarray, labels: np.ndarray, alpha: float) -> None:
    base_rgb = np.tile(DEFAULT_BASE_COLOR.reshape(1, 3), (xyz.shape[0], 1))
    palette = {
        0: np.asarray([1.0, 0.2, 0.2], dtype=np.float64),
        1: np.asarray([0.2, 0.95, 0.2], dtype=np.float64),
    }
    for plane_id, color in palette.items():
        mask = labels == plane_id
        if np.any(mask):
            base_rgb[mask] = (1.0 - alpha) * base_rgb[mask] + alpha * color.reshape(1, 3)

    base_pcd.points = o3d.utility.Vector3dVector(xyz)
    base_pcd.colors = o3d.utility.Vector3dVector(np.clip(base_rgb, 0.0, 1.0))


def _draw_2d_overlay(rgb_img: np.ndarray, plane_quads: dict[int, np.ndarray], alpha: float) -> np.ndarray:
    overlay = rgb_img.copy()
    color_map = {0: (40, 40, 235), 1: (40, 230, 40)}  # BGR
    h, w = rgb_img.shape[:2]

    for plane_id, quad in plane_quads.items():
        q = _sanitize_quad(quad, w=w, h=h).astype(np.int32)
        if q.shape[0] != 4:
            continue
        c = color_map[plane_id]
        cv2.fillConvexPoly(overlay, q, c)
        cv2.polylines(overlay, [q], True, (255, 255, 255), 1, cv2.LINE_AA)
        center = np.mean(q, axis=0).astype(np.int32)
        cv2.putText(overlay, f"P{plane_id}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    blended = cv2.addWeighted(overlay, float(alpha), rgb_img, float(1.0 - alpha), 0.0)
    return blended


# endregion


# region 采集/投影工具
def _capture_preview_with_color_once(session: OrbbecSession, point_filter) -> tuple[np.ndarray | None, np.ndarray | None]:
    frames = session.wait_for_frames()
    if frames is None:
        return None, None

    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None, None
    color_frame = frames.get_color_frame()
    color_bgr = _decode_color_frame_bgr(color_frame)

    point_frames, use_color = session.prepare_frame_for_point_cloud(frames)
    set_point_cloud_filter_format(
        point_filter,
        depth_scale=float(depth_frame.get_depth_scale()),
        use_color=use_color,
    )
    cloud_frame = point_filter.process(point_frames)
    if cloud_frame is None:
        return None, color_bgr

    raw = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
    normalized = normalize_points(raw)
    valid, _ = filter_valid_points(normalized, max_depth_mm=DEFAULT_MAX_DEPTH_MM)
    if len(valid) == 0:
        return None, color_bgr

    if len(valid) > DEFAULT_MAX_PREVIEW_POINTS:
        step = max(1, len(valid) // DEFAULT_MAX_PREVIEW_POINTS)
        valid = valid[::step]
    return valid, color_bgr


def _extract_rgb(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 6:
        rgb = np.asarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0)
    return np.full((points.shape[0], 3), 0.7, dtype=np.float32)


def _decode_color_frame_bgr(color_frame) -> np.ndarray | None:
    if color_frame is None:
        return None
    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    if width <= 0 or height <= 0:
        return None

    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())
    if data.size == 0:
        return None

    if color_format == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3)).copy()
    if color_format in (OBFormat.YUYV, OBFormat.YUY2):
        yuy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if color_format == OBFormat.I420:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return None


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
    valid = z > 1e-6
    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)

    if np.any(valid):
        x = xyz[valid, 0]
        y = xyz[valid, 1]
        zz = z[valid]
        uu = np.rint(fx * x / zz + cx).astype(np.int32)
        vv = np.rint(fy * y / zz + cy).astype(np.int32)
        in_bounds = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        idx = np.where(valid)[0][in_bounds]
        u[idx] = uu[in_bounds]
        v[idx] = vv[in_bounds]

    uv = np.stack([u, v], axis=1)
    return uv, (u >= 0) & (v >= 0)


def _rasterize_rgb(xyz: np.ndarray, rgb: np.ndarray, uv: np.ndarray, valid_proj: np.ndarray, w: int, h: int) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    idx = np.where(valid_proj)[0]
    if idx.size == 0:
        return out

    u = uv[idx, 0].astype(np.int32)
    v = uv[idx, 1].astype(np.int32)
    z = xyz[idx, 2].astype(np.float32)
    linear = v * int(w) + u
    order = np.lexsort((z, linear))
    linear_sorted = linear[order]
    idx_sorted = idx[order]
    first = np.unique(linear_sorted, return_index=True)[1]
    chosen = idx_sorted[first]

    u_c = uv[chosen, 0].astype(np.int32)
    v_c = uv[chosen, 1].astype(np.int32)
    out[v_c, u_c, :] = np.clip(rgb[chosen] * 255.0, 0.0, 255.0).astype(np.uint8)[:, ::-1]
    return out


# endregion


# region CLI
def _parse_cli() -> tuple[Path, int, int, float, int, int, float, float]:
    parser = argparse.ArgumentParser(description="Orbbec 标注引导平面识别（红/绿掩码）")
    parser.add_argument("--label-image", type=Path, default=DEFAULT_LABEL_IMAGE, help="标注图路径")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames 超时（ms）")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="请求采集帧率（fps）")
    parser.add_argument("--plane-distance-mm", type=float, default=DEFAULT_PLANE_DISTANCE_MM, help="平面距离阈值（mm）")
    parser.add_argument("--plane-ransac-iter", type=int, default=DEFAULT_PLANE_RANSAC_ITER, help="RANSAC迭代次数")
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS, help="单平面最小内点数")
    parser.add_argument("--min-inlier-ratio", type=float, default=DEFAULT_MIN_INLIER_RATIO, help="掩码平面最小内点占比")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="半透明叠加权重 0~1")
    args = parser.parse_args()
    return (
        Path(args.label_image),
        int(args.timeout_ms),
        int(args.capture_fps),
        float(args.plane_distance_mm),
        int(args.plane_ransac_iter),
        int(args.min_inliers),
        float(args.min_inlier_ratio),
        float(args.alpha),
    )


# endregion


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            label_arg, timeout_arg, fps_arg, dist_arg, iter_arg, inliers_arg, ratio_arg, alpha_arg = _parse_cli()
            main(
                label_image=label_arg,
                timeout_ms=timeout_arg,
                capture_fps=fps_arg,
                plane_distance_mm=dist_arg,
                plane_ransac_iter=iter_arg,
                min_inliers=inliers_arg,
                min_inlier_ratio=ratio_arg,
                alpha=alpha_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
