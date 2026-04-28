from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from pyorbbecsdk import OBFormat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import Gemini305, SessionOptions, set_point_cloud_filter_format


# region 默认参数（优先在这里直接改）
DEFAULT_TIMEOUT_MS = 120  # 等待相机帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 5000.0  # filter_points_for_sensor 深度上限，单位 mm
DEFAULT_MAX_PREVIEW_POINTS = 120_000  # 3D 原始点云预览点数上限，单位 点
DEFAULT_MAX_RANSAC_POINTS = 60_000  # 平面拟合最大采样点数，单位 点
DEFAULT_PLANE_DISTANCE_MM = 4.0  # RANSAC 平面内点距离阈值，单位 mm
DEFAULT_PLANE_MIN_POINTS = 1200  # 单个平面最少内点数量，单位 点
DEFAULT_RANSAC_ITERATIONS = 1200  # RANSAC 最大迭代次数
DEFAULT_BOTTOM_AXIS = "y"  # 底面法向参考轴，可选 x/y/z；Orbbec 视图中 y 轴通常接近竖直方向
DEFAULT_ALPHA = 0.45  # 2D/3D 平面颜色叠加透明度
DEFAULT_WINDOW_WIDTH = 1440  # 3D 窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 3D 窗口高度，单位 像素
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 预览窗口最小长边，单位 像素
DEFAULT_POINT_SIZE = 1.5  # 3D 点大小
DEFAULT_2D_WINDOW_NAME = "Orbbec three plane detector"  # 2D 窗口名，ASCII
DEFAULT_3D_WINDOW_NAME = "Orbbec three plane detector 3D"  # 3D 窗口名，ASCII
# endregion


# region 数据结构
@dataclass(frozen=True)
class CaptureJob:
    frame_idx: int
    points: np.ndarray
    color_bgr: np.ndarray | None
    fx: float
    fy: float
    cx: float
    cy: float
    img_w: int
    img_h: int


@dataclass(frozen=True)
class PlanePatch:
    label: str
    model: np.ndarray
    color_rgb: np.ndarray
    contour: np.ndarray
    mesh_vertices: np.ndarray
    mesh_triangles: np.ndarray
    inlier_count: int


@dataclass(frozen=True)
class PlaneDetectionResult:
    frame_idx: int
    xyz: np.ndarray
    colors: np.ndarray
    labels: np.ndarray
    overlay_bgr: np.ndarray
    planes: list[PlanePatch]
    elapsed_ms: float


@dataclass(frozen=True)
class PlaneDetectorConfig:
    max_ransac_points: int
    plane_distance_mm: float
    plane_min_points: int
    ransac_iterations: int
    bottom_axis: str
    alpha: float


# endregion


# region 主流程
def main(
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
    max_preview_points: int = DEFAULT_MAX_PREVIEW_POINTS,
    max_ransac_points: int = DEFAULT_MAX_RANSAC_POINTS,
    plane_distance_mm: float = DEFAULT_PLANE_DISTANCE_MM,
    plane_min_points: int = DEFAULT_PLANE_MIN_POINTS,
    ransac_iterations: int = DEFAULT_RANSAC_ITERATIONS,
    bottom_axis: str = DEFAULT_BOTTOM_AXIS,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    cfg = PlaneDetectorConfig(
        max_ransac_points=max(500, int(max_ransac_points)),
        plane_distance_mm=max(0.1, float(plane_distance_mm)),
        plane_min_points=max(20, int(plane_min_points)),
        ransac_iterations=max(50, int(ransac_iterations)),
        bottom_axis=str(bottom_axis).strip().lower(),
        alpha=float(np.clip(alpha, 0.0, 1.0)),
    )
    if cfg.bottom_axis not in {"x", "y", "z"}:
        raise ValueError("bottom_axis must be one of: x, y, z")

    options = SessionOptions(timeout_ms=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))
    with Gemini305(options=options) as session:
        cam = session.get_camera_param()
        ci = cam.rgb_intrinsic if session.has_color_sensor else cam.depth_intrinsic
        img_w = int(max(32, ci.width))
        img_h = int(max(32, ci.height))
        point_filter = session.create_point_cloud_filter(camera_param=cam)
        logger.info(
            f"三平面实时检测启动：capture_fps {capture_fps} fps, max_depth_mm {float(max_depth_mm):.1f} mm, "
            f"plane_distance_mm {cfg.plane_distance_mm:.2f} mm, plane_min_points {cfg.plane_min_points} 点"
        )
        logger.info("计算线程使用单槽队列：上一帧计算未完成时，当前帧只参与预览，不进入计算。")
        _run_preview_loop(
            session=session,
            point_filter=point_filter,
            fx=float(ci.fx),
            fy=float(ci.fy),
            cx=float(ci.cx),
            cy=float(ci.cy),
            img_w=img_w,
            img_h=img_h,
            max_depth_mm=float(max_depth_mm),
            max_preview_points=max(1, int(max_preview_points)),
            cfg=cfg,
        )


# endregion


# region 实时预览与后台计算
def _run_preview_loop(
    session: Gemini305,
    point_filter,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    max_depth_mm: float,
    max_preview_points: int,
    cfg: PlaneDetectorConfig,
) -> None:
    job_queue: queue.Queue[CaptureJob | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[PlaneDetectionResult] = queue.Queue(maxsize=2)
    worker_busy = threading.Event()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_plane_worker,
        args=(job_queue, result_queue, worker_busy, stop_event, cfg),
        name="three_plane_worker",
        daemon=True,
    )
    worker.start()

    vis, stop_flag, raw_pcd, plane_pcd, plane_meshes = _init_3d_viewer()
    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)
    win_w, win_h = _compute_preview_window_size(
        src_w=img_w,
        src_h=img_h,
        min_long_side=DEFAULT_MIN_2D_WINDOW_LONG_SIDE,
    )
    cv2.resizeWindow(DEFAULT_2D_WINDOW_NAME, win_w, win_h)

    frame_idx = 0
    submitted = 0
    dropped = 0
    last_overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    last_log_ts = time.monotonic()

    try:
        while True:
            if stop_flag["flag"]:
                break

            points, color_bgr = _capture_cropped_points_once(
                session=session,
                point_filter=point_filter,
                max_depth_mm=max_depth_mm,
            )
            if points is None or len(points) == 0:
                _poll_viewers(vis=vis)
                continue

            frame_idx += 1
            preview_points = _downsample_points(points, max_points=max_preview_points)
            _update_raw_cloud(raw_pcd=raw_pcd, points=preview_points)
            vis.update_geometry(raw_pcd)

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break
                _apply_detection_result(
                    result=result,
                    plane_pcd=plane_pcd,
                    plane_meshes=plane_meshes,
                    vis=vis,
                )
                last_overlay = result.overlay_bgr
                logger.info(
                    f"帧 {result.frame_idx} 平面检测完成：planes {len(result.planes)}, "
                    f"耗时 {result.elapsed_ms:.1f} ms"
                )

            if not worker_busy.is_set() and job_queue.empty():
                job = CaptureJob(
                    frame_idx=frame_idx,
                    points=np.asarray(points, dtype=np.float32).copy(),
                    color_bgr=None if color_bgr is None else color_bgr.copy(),
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    img_w=img_w,
                    img_h=img_h,
                )
                try:
                    job_queue.put_nowait(job)
                    submitted += 1
                except queue.Full:
                    dropped += 1
            else:
                dropped += 1

            cv2.imshow(DEFAULT_2D_WINDOW_NAME, last_overlay)
            if cv2.waitKey(1) == 27:
                logger.warning("用户按下 ESC，提前结束。")
                break

            alive = _poll_viewers(vis=vis)
            if not alive or cv2.getWindowProperty(DEFAULT_2D_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            now = time.monotonic()
            if now - last_log_ts >= 3.0:
                logger.info(f"实时状态：frame {frame_idx}, submitted {submitted}, dropped {dropped}")
                last_log_ts = now
    finally:
        stop_event.set()
        with _suppress_queue_full():
            job_queue.put_nowait(None)
        worker.join(timeout=1.0)
        vis.destroy_window()
        cv2.destroyWindow(DEFAULT_2D_WINDOW_NAME)


def _plane_worker(
    job_queue: queue.Queue[CaptureJob | None],
    result_queue: queue.Queue[PlaneDetectionResult],
    worker_busy: threading.Event,
    stop_event: threading.Event,
    cfg: PlaneDetectorConfig,
) -> None:
    while not stop_event.is_set():
        job = job_queue.get()
        if job is None:
            break
        worker_busy.set()
        try:
            result = _detect_three_planes(job=job, cfg=cfg)
            _put_latest_result(result_queue, result)
        except Exception as exc:
            logger.exception(f"帧 {job.frame_idx} 平面检测失败：{exc}")
        finally:
            worker_busy.clear()
            job_queue.task_done()


# endregion


# region 平面检测
def _detect_three_planes(job: CaptureJob, cfg: PlaneDetectorConfig) -> PlaneDetectionResult:
    start = time.perf_counter()
    xyz = np.asarray(job.points[:, :3], dtype=np.float64)
    colors = _extract_rgb(job.points)
    plane_models = _segment_plane_models(
        xyz=xyz,
        max_ransac_points=cfg.max_ransac_points,
        distance_mm=cfg.plane_distance_mm,
        min_points=cfg.plane_min_points,
        ransac_iterations=cfg.ransac_iterations,
    )
    labels = _assign_points_to_planes(xyz=xyz, plane_models=plane_models, distance_mm=cfg.plane_distance_mm)
    ordered = _order_planes(plane_models=plane_models, labels=labels, xyz=xyz, bottom_axis=cfg.bottom_axis)
    labels = _remap_labels(labels=labels, ordered_old_ids=[old_id for old_id, _ in ordered])
    base_2d = _build_base_2d(job=job, xyz=xyz, colors=colors)
    uv, valid_proj = _project_points_to_image(
        xyz=xyz,
        fx=job.fx,
        fy=job.fy,
        cx=job.cx,
        cy=job.cy,
        w=job.img_w,
        h=job.img_h,
    )

    planes: list[PlanePatch] = []
    for new_id, (old_id, label) in enumerate(ordered):
        mask = labels == new_id
        if not np.any(mask):
            continue
        contour = _plane_contour_from_projection(
            uv=uv,
            valid_proj=valid_proj,
            mask=mask,
            h=job.img_h,
            w=job.img_w,
        )
        vertices, triangles = _make_plane_patch_mesh(xyz[mask], plane_models[old_id])
        planes.append(
            PlanePatch(
                label=label,
                model=plane_models[old_id],
                color_rgb=_plane_color_rgb(new_id),
                contour=contour,
                mesh_vertices=vertices,
                mesh_triangles=triangles,
                inlier_count=int(np.count_nonzero(mask)),
            )
        )

    overlay = _draw_2d_overlay(base_bgr=base_2d, planes=planes, labels=labels, alpha=cfg.alpha)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PlaneDetectionResult(
        frame_idx=job.frame_idx,
        xyz=xyz,
        colors=colors,
        labels=labels,
        overlay_bgr=overlay,
        planes=planes,
        elapsed_ms=elapsed_ms,
    )


def _segment_plane_models(
    xyz: np.ndarray,
    max_ransac_points: int,
    distance_mm: float,
    min_points: int,
    ransac_iterations: int,
) -> list[np.ndarray]:
    if xyz.shape[0] < min_points:
        return []

    sample = _downsample_points(xyz, max_points=max_ransac_points)
    remain = o3d.geometry.PointCloud()
    remain.points = o3d.utility.Vector3dVector(np.ascontiguousarray(sample, dtype=np.float64))
    out: list[np.ndarray] = []
    for _ in range(3):
        if len(remain.points) < min_points:
            break
        model, inliers = remain.segment_plane(
            distance_threshold=float(distance_mm),
            ransac_n=3,
            num_iterations=int(ransac_iterations),
        )
        if len(inliers) < min_points:
            break
        out.append(_normalize_plane_model(np.asarray(model, dtype=np.float64)))
        remain = remain.select_by_index(inliers, invert=True)
    return out


def _assign_points_to_planes(xyz: np.ndarray, plane_models: list[np.ndarray], distance_mm: float) -> np.ndarray:
    labels = np.full((xyz.shape[0],), -1, dtype=np.int32)
    if len(plane_models) == 0 or xyz.size == 0:
        return labels
    distances = []
    for model in plane_models:
        n = model[:3]
        d = float(model[3])
        distances.append(np.abs(xyz @ n + d))
    dist_mat = np.stack(distances, axis=1)
    best = np.argmin(dist_mat, axis=1)
    best_dist = dist_mat[np.arange(xyz.shape[0]), best]
    labels[best_dist <= float(distance_mm)] = best[best_dist <= float(distance_mm)].astype(np.int32)
    return labels


def _order_planes(
    plane_models: list[np.ndarray],
    labels: np.ndarray,
    xyz: np.ndarray,
    bottom_axis: str,
) -> list[tuple[int, str]]:
    if len(plane_models) == 0:
        return []
    axis_map = {
        "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    }
    axis = axis_map[bottom_axis]
    scores = [float(abs(np.dot(model[:3], axis))) for model in plane_models]
    bottom_old = int(np.argmax(scores))
    side_ids = [idx for idx in range(len(plane_models)) if idx != bottom_old]
    side_ids.sort(key=lambda idx: _plane_centroid_x(labels=labels, xyz=xyz, plane_id=idx))
    ordered = [(bottom_old, "bottom")]
    if len(side_ids) >= 1:
        ordered.append((side_ids[0], "left_side"))
    if len(side_ids) >= 2:
        ordered.append((side_ids[1], "right_side"))
    return ordered


# endregion


# region 可视化
def _init_3d_viewer() -> tuple[
    o3d.visualization.VisualizerWithKeyCallback,
    dict[str, bool],
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    list[o3d.geometry.TriangleMesh],
]:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(DEFAULT_3D_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    render_opt = vis.get_render_option()
    if render_opt is not None:
        render_opt.point_size = DEFAULT_POINT_SIZE
        render_opt.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)

    stop = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    raw_pcd = o3d.geometry.PointCloud()
    plane_pcd = o3d.geometry.PointCloud()
    plane_meshes = [_empty_mesh() for _ in range(3)]
    vis.add_geometry(axis)
    vis.add_geometry(raw_pcd)
    vis.add_geometry(plane_pcd)
    for mesh in plane_meshes:
        vis.add_geometry(mesh)
    _apply_reference_camera_view(vis)
    return vis, stop, raw_pcd, plane_pcd, plane_meshes


def _apply_detection_result(
    result: PlaneDetectionResult,
    plane_pcd: o3d.geometry.PointCloud,
    plane_meshes: list[o3d.geometry.TriangleMesh],
    vis: o3d.visualization.VisualizerWithKeyCallback,
) -> None:
    plane_colors = np.zeros_like(result.colors)
    valid = result.labels >= 0
    for idx in range(3):
        m = result.labels == idx
        if np.any(m):
            plane_colors[m] = _plane_color_rgb(idx)
    plane_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(result.xyz[valid], dtype=np.float64))
    plane_pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(plane_colors[valid], dtype=np.float64))
    vis.update_geometry(plane_pcd)

    for idx, mesh in enumerate(plane_meshes):
        if idx < len(result.planes):
            patch = result.planes[idx]
            mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(patch.mesh_vertices, dtype=np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(np.ascontiguousarray(patch.mesh_triangles, dtype=np.int32))
            mesh.paint_uniform_color(patch.color_rgb)
            mesh.compute_vertex_normals()
        else:
            empty = _empty_mesh()
            mesh.vertices = empty.vertices
            mesh.triangles = empty.triangles
            mesh.vertex_colors = empty.vertex_colors
        vis.update_geometry(mesh)


def _update_raw_cloud(raw_pcd: o3d.geometry.PointCloud, points: np.ndarray) -> None:
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    raw_pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = _extract_rgb(points)
    muted = 0.35 * rgb + 0.15
    raw_pcd.colors = o3d.utility.Vector3dVector(np.clip(muted, 0.0, 1.0))


def _draw_2d_overlay(base_bgr: np.ndarray, planes: list[PlanePatch], labels: np.ndarray, alpha: float) -> np.ndarray:
    overlay = base_bgr.copy()
    for idx, plane in enumerate(planes):
        bgr = _rgb_to_bgr_tuple(plane.color_rgb)
        if plane.contour.shape[0] >= 3:
            cv2.fillConvexPoly(overlay, plane.contour.astype(np.int32), bgr)
            cv2.polylines(overlay, [plane.contour.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
            center = np.mean(plane.contour, axis=0).astype(np.int32)
            cv2.putText(
                overlay,
                f"{plane.label} pts={plane.inlier_count}",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    blended = cv2.addWeighted(overlay, float(alpha), base_bgr, float(1.0 - alpha), 0.0)
    cv2.putText(
        blended,
        f"planes={len(planes)} assigned={int(np.count_nonzero(labels >= 0))}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return blended


# endregion


# region 捕获与图像投影
def _capture_cropped_points_once(
    session: Gemini305,
    point_filter,
    max_depth_mm: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    frames = session.wait_for_frames()
    if frames is None:
        return None, None
    color_bgr = _decode_color_frame_bgr(frames.get_color_frame())
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None, color_bgr

    point_frames, use_color = session.prepare_frame_for_point_cloud(frames)
    set_point_cloud_filter_format(
        point_filter,
        depth_scale=float(depth_frame.get_depth_scale()),
        use_color=use_color,
    )
    cloud_frame = point_filter.process(point_frames)
    if cloud_frame is None:
        return None, color_bgr

    raw_points = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
    points = session.filter_points_for_sensor(
        raw_points,
        max_depth_mm=float(max_depth_mm),
        apply_sensor_frustum=True,
    )
    if points.size == 0:
        return None, color_bgr
    return points, color_bgr


def _build_base_2d(job: CaptureJob, xyz: np.ndarray, colors: np.ndarray) -> np.ndarray:
    if job.color_bgr is not None:
        return cv2.resize(job.color_bgr, (job.img_w, job.img_h), interpolation=cv2.INTER_LINEAR)
    uv, valid_proj = _project_points_to_image(
        xyz=xyz,
        fx=job.fx,
        fy=job.fy,
        cx=job.cx,
        cy=job.cy,
        w=job.img_w,
        h=job.img_h,
    )
    return _rasterize_rgb(xyz=xyz, rgb=colors, uv=uv, valid_proj=valid_proj, w=job.img_w, h=job.img_h)


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
    xyz: np.ndarray,
    rgb: np.ndarray,
    uv: np.ndarray,
    valid_proj: np.ndarray,
    w: int,
    h: int,
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
    out[uv[chosen, 1], uv[chosen, 0], :] = np.clip(rgb[chosen] * 255.0, 0.0, 255.0).astype(np.uint8)[:, ::-1]
    return out


# endregion


# region 几何工具
def _plane_contour_from_projection(
    uv: np.ndarray,
    valid_proj: np.ndarray,
    mask: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    idx = np.where(valid_proj & mask)[0]
    if idx.size < 3:
        return np.empty((0, 2), dtype=np.int32)
    img = np.zeros((h, w), dtype=np.uint8)
    img[uv[idx, 1], uv[idx, 0]] = 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.empty((0, 2), dtype=np.int32)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 20.0:
        return np.empty((0, 2), dtype=np.int32)
    return cv2.convexHull(contour).reshape(-1, 2).astype(np.int32)


def _make_plane_patch_mesh(xyz: np.ndarray, model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] < 3:
        return _empty_mesh_arrays()
    normal = _normalize_vector(model[:3])
    basis_u = _normalize_vector(np.cross(normal, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)))
    if np.linalg.norm(basis_u) < 1e-8:
        basis_u = _normalize_vector(np.cross(normal, np.asarray([0.0, 1.0, 0.0], dtype=np.float64)))
    basis_v = _normalize_vector(np.cross(normal, basis_u))
    center = np.mean(xyz, axis=0)
    rel = xyz - center.reshape(1, 3)
    u = rel @ basis_u
    v = rel @ basis_v
    u_min, u_max = np.percentile(u, [2.0, 98.0])
    v_min, v_max = np.percentile(v, [2.0, 98.0])
    corners = np.asarray(
        [
            center + basis_u * u_min + basis_v * v_min,
            center + basis_u * u_max + basis_v * v_min,
            center + basis_u * u_max + basis_v * v_max,
            center + basis_u * u_min + basis_v * v_max,
        ],
        dtype=np.float64,
    )
    d = float(model[3])
    signed = corners @ normal + d
    corners = corners - signed.reshape(-1, 1) * normal.reshape(1, 3)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return corners, triangles


def _normalize_plane_model(model: np.ndarray) -> np.ndarray:
    n = model[:3]
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return model
    out = model / norm
    if out[2] < 0:
        out = -out
    return out


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return np.zeros_like(v, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / norm


def _plane_centroid_x(labels: np.ndarray, xyz: np.ndarray, plane_id: int) -> float:
    mask = labels == int(plane_id)
    if not np.any(mask):
        return 0.0
    return float(np.mean(xyz[mask, 0]))


def _remap_labels(labels: np.ndarray, ordered_old_ids: list[int]) -> np.ndarray:
    out = np.full_like(labels, -1)
    for new_id, old_id in enumerate(ordered_old_ids):
        out[labels == int(old_id)] = int(new_id)
    return out


def _empty_mesh() -> o3d.geometry.TriangleMesh:
    vertices, triangles = _empty_mesh_arrays()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([0.0, 0.0, 0.0])
    return mesh


def _empty_mesh_arrays() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [[0.0, 0.0, -10_000.0], [1.0, 0.0, -10_000.0], [0.0, 1.0, -10_000.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    return vertices, triangles


# endregion


# region 通用工具
def _extract_rgb(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 6:
        rgb = np.asarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0).astype(np.float64)
    return np.full((points.shape[0], 3), 0.72, dtype=np.float64)


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(1, int(np.ceil(len(points) / float(max_points))))
    return points[::step]


def _plane_color_rgb(idx: int) -> np.ndarray:
    colors = [
        np.asarray([0.10, 0.70, 1.00], dtype=np.float64),
        np.asarray([1.00, 0.35, 0.20], dtype=np.float64),
        np.asarray([0.25, 0.95, 0.35], dtype=np.float64),
    ]
    return colors[int(idx) % len(colors)]


def _rgb_to_bgr_tuple(rgb: np.ndarray) -> tuple[int, int, int]:
    c = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    return int(c[2]), int(c[1]), int(c[0])


def _compute_preview_window_size(src_w: int, src_h: int, min_long_side: int) -> tuple[int, int]:
    w = max(1, int(src_w))
    h = max(1, int(src_h))
    long_side = max(w, h)
    if long_side >= int(min_long_side):
        return w, h
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _apply_reference_camera_view(vis: o3d.visualization.VisualizerWithKeyCallback) -> None:
    view = vis.get_view_control()
    if view is None:
        return
    view.set_lookat([0.0, 0.0, 0.0])
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])


def _poll_viewers(vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    alive = vis.poll_events()
    vis.update_renderer()
    return bool(alive)


def _put_latest_result(result_queue: queue.Queue[PlaneDetectionResult], result: PlaneDetectionResult) -> None:
    while True:
        try:
            result_queue.put_nowait(result)
            return
        except queue.Full:
            try:
                result_queue.get_nowait()
            except queue.Empty:
                return


class _suppress_queue_full:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is queue.Full


# endregion


# region CLI
def _parse_cli() -> tuple[int, int, float, int, int, float, int, int, str, float]:
    parser = argparse.ArgumentParser(description="Orbbec Gemini 305 实时三平面检测测试脚本")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames timeout in ms")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="preferred capture fps")
    parser.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM, help="max depth for sensor filtering")
    parser.add_argument("--max-preview-points", type=int, default=DEFAULT_MAX_PREVIEW_POINTS, help="max preview points")
    parser.add_argument("--max-ransac-points", type=int, default=DEFAULT_MAX_RANSAC_POINTS, help="max sampled points for RANSAC")
    parser.add_argument("--plane-distance-mm", type=float, default=DEFAULT_PLANE_DISTANCE_MM, help="RANSAC plane distance threshold")
    parser.add_argument("--plane-min-points", type=int, default=DEFAULT_PLANE_MIN_POINTS, help="minimum points per plane")
    parser.add_argument("--ransac-iterations", type=int, default=DEFAULT_RANSAC_ITERATIONS, help="RANSAC iterations")
    parser.add_argument("--bottom-axis", choices=["x", "y", "z"], default=DEFAULT_BOTTOM_AXIS, help="bottom plane reference normal axis")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="overlay alpha")
    args = parser.parse_args()
    return (
        int(args.timeout_ms),
        int(args.capture_fps),
        float(args.max_depth_mm),
        int(args.max_preview_points),
        int(args.max_ransac_points),
        float(args.plane_distance_mm),
        int(args.plane_min_points),
        int(args.ransac_iterations),
        str(args.bottom_axis),
        float(args.alpha),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        (
            timeout_arg,
            fps_arg,
            max_depth_arg,
            max_preview_arg,
            max_ransac_arg,
            plane_distance_arg,
            plane_min_arg,
            ransac_iter_arg,
            bottom_axis_arg,
            alpha_arg,
        ) = _parse_cli()
        main(
            timeout_ms=timeout_arg,
            capture_fps=fps_arg,
            max_depth_mm=max_depth_arg,
            max_preview_points=max_preview_arg,
            max_ransac_points=max_ransac_arg,
            plane_distance_mm=plane_distance_arg,
            plane_min_points=plane_min_arg,
            ransac_iterations=ransac_iter_arg,
            bottom_axis=bottom_axis_arg,
            alpha=alpha_arg,
        )
    else:
        main()
# endregion
