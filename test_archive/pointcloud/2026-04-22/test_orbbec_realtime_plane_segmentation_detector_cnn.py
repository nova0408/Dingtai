from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from loguru import logger

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 opencv-python 才能运行该脚本。") from exc

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

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
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments" / "models" / "plane_yolov8n_seg.pt"  # YOLOv8-seg 权重路径
DEFAULT_CLASS_RED = "red"  # 红平面类别名（需与模型类别一致）
DEFAULT_CLASS_GREEN = "green"  # 绿平面类别名（需与模型类别一致）
DEFAULT_DEVICE = "0"  # 推理设备：默认使用第 0 块 GPU，可改为 1 / 0,1
DEFAULT_DET_CONF = 0.25  # 检测置信度阈值
DEFAULT_DET_IOU = 0.45  # NMS IoU 阈值
DEFAULT_DET_IMGSZ = 960  # 推理尺寸，越大越准但越慢

DEFAULT_TIMEOUT_MS = 120  # 等待帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 5000.0  # 深度裁剪上限，单位 mm
DEFAULT_MAX_PREVIEW_POINTS = 90_000  # 预览点上限

DEFAULT_PLANE_DISTANCE_MM = 2.0  # 平面 RANSAC 距离阈值，单位 mm
DEFAULT_PLANE_RANSAC_ITER = 150  # 平面 RANSAC 迭代次数
DEFAULT_MIN_INLIERS = 80  # 单平面最小内点数
DEFAULT_MIN_INLIER_RATIO = 0.14  # 最小内点占比
DEFAULT_NORMAL_ALIGN_MIN = 0.70  # 点云法线一致性阈值
DEFAULT_MAX_NORMAL_DOT = 0.45  # 红绿法线最大点积绝对值
DEFAULT_MASK_MIN_PIXELS = 260  # 2D 分割最小有效像素面积

DEFAULT_ALPHA = 0.42  # 2D/3D叠加半透明权重
DEFAULT_WINDOW_WIDTH = 1440  # 3D 窗口宽度
DEFAULT_WINDOW_HEIGHT = 900  # 3D 窗口高度
DEFAULT_POINT_SIZE = 1.5  # 3D 点大小
DEFAULT_BACKGROUND_COLOR = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)  # 3D 背景色
DEFAULT_BASE_COLOR = np.asarray([0.22, 0.22, 0.22], dtype=np.float64)  # 基础点云颜色
DEFAULT_2D_WINDOW_NAME = "Orbbec CNN 平面识别"  # 2D 窗口名
# endregion


# region 数据结构
@dataclass
class Detection2D:
    plane_id: int
    confidence_2d: float
    polygon: np.ndarray
    mask_warped: np.ndarray
    area_pixels: int


# endregion


# region CNN 检测器
class CNNPlaneDetector:
    def __init__(
        self,
        model_path: Path,
        class_red: str,
        class_green: str,
        det_conf: float,
        det_iou: float,
        det_imgsz: int,
        device: str,
    ) -> None:
        if YOLO is None:
            raise RuntimeError("缺少 ultralytics。请先安装：pip install ultralytics")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型权重不存在：{model_path}")

        self.model = YOLO(str(model_path))
        self.class_red = class_red.strip().lower()
        self.class_green = class_green.strip().lower()
        self.det_conf = float(det_conf)
        self.det_iou = float(det_iou)
        self.det_imgsz = int(det_imgsz)
        self.device = device

        names = self.model.names
        if isinstance(names, dict):
            self.class_map = {int(k): str(v).strip().lower() for k, v in names.items()}
        else:
            self.class_map = {i: str(v).strip().lower() for i, v in enumerate(names)}

        logger.info(f"模型类别：{self.class_map}")

    def detect(self, frame_bgr: np.ndarray) -> dict[int, Detection2D]:
        h, w = frame_bgr.shape[:2]
        pred = self.model.predict(
            source=frame_bgr,
            conf=self.det_conf,
            iou=self.det_iou,
            imgsz=self.det_imgsz,
            device=self.device,
            verbose=False,
        )
        if len(pred) == 0:
            return {}

        r0 = pred[0]
        if r0.boxes is None or r0.masks is None:
            return {}

        boxes_cls = r0.boxes.cls
        boxes_conf = r0.boxes.conf
        masks = r0.masks.data
        if boxes_cls is None or boxes_conf is None or masks is None:
            return {}

        cls_arr = boxes_cls.detach().cpu().numpy().astype(np.int32)
        conf_arr = boxes_conf.detach().cpu().numpy().astype(np.float32)
        masks_arr = masks.detach().cpu().numpy().astype(np.float32)

        results: dict[int, Detection2D] = {}
        for pid, cname in ((0, self.class_red), (1, self.class_green)):
            det_idx = self._pick_best_index(cls_arr=cls_arr, conf_arr=conf_arr, target_name=cname)
            if det_idx < 0:
                continue

            m = masks_arr[det_idx]
            mask_u8 = (m > 0.5).astype(np.uint8) * 255
            if mask_u8.shape[0] != h or mask_u8.shape[1] != w:
                mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
            area = int(np.count_nonzero(mask_u8))
            if area < DEFAULT_MASK_MIN_PIXELS:
                continue

            poly = _mask_to_quad(mask_u8)
            if poly.shape[0] != 4:
                continue

            results[pid] = Detection2D(
                plane_id=pid,
                confidence_2d=float(conf_arr[det_idx]),
                polygon=poly,
                mask_warped=mask_u8,
                area_pixels=area,
            )

        return results

    def _pick_best_index(self, cls_arr: np.ndarray, conf_arr: np.ndarray, target_name: str) -> int:
        idxs = np.where([self.class_map.get(int(c), "") == target_name for c in cls_arr])[0]
        if idxs.size == 0:
            return -1
        best_local = int(np.argmax(conf_arr[idxs]))
        return int(idxs[best_local])


# endregion


# region 主流程
def main(
    model_path: Path = DEFAULT_MODEL_PATH,
    class_red: str = DEFAULT_CLASS_RED,
    class_green: str = DEFAULT_CLASS_GREEN,
    device: str = DEFAULT_DEVICE,
    det_conf: float = DEFAULT_DET_CONF,
    det_iou: float = DEFAULT_DET_IOU,
    det_imgsz: int = DEFAULT_DET_IMGSZ,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    plane_distance_mm: float = DEFAULT_PLANE_DISTANCE_MM,
    plane_ransac_iter: int = DEFAULT_PLANE_RANSAC_ITER,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    min_inlier_ratio: float = DEFAULT_MIN_INLIER_RATIO,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    detector = CNNPlaneDetector(
        model_path=Path(model_path),
        class_red=class_red,
        class_green=class_green,
        det_conf=det_conf,
        det_iou=det_iou,
        det_imgsz=det_imgsz,
        device=device,
    )

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

        logger.info("启动 CNN 平面识别（每帧独立检测，无跟踪状态）")
        logger.info(
            f"参数：conf {det_conf:.2f}, iou {det_iou:.2f}, imgsz {det_imgsz}, "
            f"plane_distance_mm {plane_distance_mm:.2f} mm"
        )

        _run_loop(
            session=session,
            detector=detector,
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
    detector: CNNPlaneDetector,
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
    vis.create_window("Orbbec CNN 平面 3D 结果", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
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

            dets = detector.detect(base_2d)

            labels = np.full((xyz.shape[0],), -1, dtype=np.int32)
            plane_quads: dict[int, np.ndarray] = {}
            source_tags: dict[int, str] = {}
            normals: dict[int, np.ndarray] = {}

            for pid in (0, 1):
                det = dets.get(pid)
                if det is None:
                    continue
                ids = _collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=det.mask_warped)
                fit = _fit_plane_indices(
                    xyz=xyz,
                    candidate_indices=ids,
                    distance_mm=plane_distance_mm,
                    ransac_iter=plane_ransac_iter,
                    min_inliers=min_inliers,
                    min_inlier_ratio=min_inlier_ratio,
                )
                if fit is None:
                    continue
                inlier_ids, nrm = fit
                labels[inlier_ids] = pid
                plane_quads[pid] = det.polygon.astype(np.int32)
                source_tags[pid] = f"cnn(conf={det.confidence_2d:.2f},pix={det.area_pixels})"
                normals[pid] = nrm

            if 0 in normals and 1 in normals:
                dot_val = float(abs(np.dot(normals[0], normals[1])))
                if dot_val > DEFAULT_MAX_NORMAL_DOT:
                    c0 = int(np.sum(labels == 0))
                    c1 = int(np.sum(labels == 1))
                    if c0 >= c1:
                        labels[labels == 1] = -1
                        plane_quads.pop(1, None)
                        source_tags[1] = f"rejected_parallel(dot={dot_val:.2f})"
                    else:
                        labels[labels == 0] = -1
                        plane_quads.pop(0, None)
                        source_tags[0] = f"rejected_parallel(dot={dot_val:.2f})"

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
                    f"green_inliers {green_cnt} ({source_tags.get(1, 'none')})"
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


# region 工具函数
def _mask_to_quad(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.empty((0, 2), dtype=np.int32)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < DEFAULT_MASK_MIN_PIXELS:
        return np.empty((0, 2), dtype=np.int32)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    h, w = mask.shape[:2]
    return _sanitize_quad(box, w=w, h=h).astype(np.int32)


def _capture_preview_with_color_once(
    session: OrbbecSession, point_filter
) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def _rasterize_rgb(
    xyz: np.ndarray, rgb: np.ndarray, uv: np.ndarray, valid_proj: np.ndarray, w: int, h: int
) -> np.ndarray:
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
        inlier_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(distance_mm * 6.0, 8.0), max_nn=30))
        normals = np.asarray(inlier_pcd.normals, dtype=np.float64)
        if normals.shape[0] > 0:
            align = np.abs(normals @ plane_normal.reshape(3))
            if float(np.median(align)) < DEFAULT_NORMAL_ALIGN_MIN:
                return None

    return candidate_indices[inliers_arr], plane_normal


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
    h, w = rgb_img.shape[:2]
    color_map = {0: (40, 40, 235), 1: (40, 230, 40)}

    for plane_id, quad in plane_quads.items():
        q = _sanitize_quad(quad, w=w, h=h).astype(np.int32)
        if q.shape[0] != 4:
            continue
        c = color_map[plane_id]
        cv2.fillConvexPoly(overlay, q, c)
        cv2.polylines(overlay, [q], True, (255, 255, 255), 1, cv2.LINE_AA)
        center = np.mean(q, axis=0).astype(np.int32)
        cv2.putText(
            overlay, f"P{plane_id}", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    blended = cv2.addWeighted(overlay, float(alpha), rgb_img, float(1.0 - alpha), 0.0)
    return blended


# endregion


# region CLI
def _parse_cli() -> tuple[Path, str, str, str, float, float, int, int, int, float, int, int, float, float]:
    parser = argparse.ArgumentParser(description="CNN 平面识别（YOLOv8-seg + 点云法线验证）")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="YOLOv8-seg 权重路径")
    parser.add_argument("--class-red", type=str, default=DEFAULT_CLASS_RED, help="红平面类别名")
    parser.add_argument("--class-green", type=str, default=DEFAULT_CLASS_GREEN, help="绿平面类别名")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="推理设备：cpu / 0 / 0,1")
    parser.add_argument("--det-conf", type=float, default=DEFAULT_DET_CONF, help="检测置信度阈值")
    parser.add_argument("--det-iou", type=float, default=DEFAULT_DET_IOU, help="NMS IoU 阈值")
    parser.add_argument("--det-imgsz", type=int, default=DEFAULT_DET_IMGSZ, help="推理尺寸")

    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames 超时（ms）")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="请求采集帧率（fps）")
    parser.add_argument("--plane-distance-mm", type=float, default=DEFAULT_PLANE_DISTANCE_MM, help="平面距离阈值（mm）")
    parser.add_argument("--plane-ransac-iter", type=int, default=DEFAULT_PLANE_RANSAC_ITER, help="RANSAC 迭代次数")
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS, help="单平面最小内点数")
    parser.add_argument("--min-inlier-ratio", type=float, default=DEFAULT_MIN_INLIER_RATIO, help="最小内点占比")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="半透明叠加权重 0~1")
    args = parser.parse_args()

    return (
        Path(args.model_path),
        str(args.class_red),
        str(args.class_green),
        str(args.device),
        float(args.det_conf),
        float(args.det_iou),
        int(args.det_imgsz),
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
            (
                model_path_arg,
                class_red_arg,
                class_green_arg,
                device_arg,
                det_conf_arg,
                det_iou_arg,
                det_imgsz_arg,
                timeout_arg,
                fps_arg,
                dist_arg,
                iter_arg,
                inliers_arg,
                ratio_arg,
                alpha_arg,
            ) = _parse_cli()
            main(
                model_path=model_path_arg,
                class_red=class_red_arg,
                class_green=class_green_arg,
                device=device_arg,
                det_conf=det_conf_arg,
                det_iou=det_iou_arg,
                det_imgsz=det_imgsz_arg,
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
