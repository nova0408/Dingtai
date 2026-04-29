from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from pyorbbecsdk import OBFormat

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pointcloud import (
    CoordinateFramePose,
    PlanePoseConfig,
    PoseWindowStabilizer,
    TrayDetection,
    TrayDetectionConfig,
    TrayPointExcluder,
    estimate_three_plane_pose,
    project_points_to_image,
    relative_pose,
)
from src.rgbd_camera import (
    CameraIntrinsics,
    Gemini305,
    SessionOptions,
    set_point_cloud_filter_format,
)
from src.utils.datas import Color

# region 默认参数（优先在这里直接改）
DEFAULT_TIMEOUT_MS = 120  # 等待相机帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 5000.0  # filter_points_for_sensor 深度上限，单位 mm
DEFAULT_MAX_PREVIEW_POINTS = 100_000  # 3D 预览点数上限，单位 点
DEFAULT_COMPUTE_MIN_INTERVAL_S = 0.10  # 提交计算任务的最小间隔，单位 s
DEFAULT_ENABLE_TRAY_EXCLUSION = True  # 是否启用料盘识别排除
DEFAULT_TRAY_USE_SAM = True  # 默认使用 SAM 细化料盘 mask，2D 预览显示实际区域而非检测框
DEFAULT_POSE_SMOOTH_FRAMES = 5  # 位姿平滑使用最近 N 个实际计算帧，最大 15
DEFAULT_WINDOW_WIDTH = 1440  # 3D 窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 3D 窗口高度，单位 像素
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 预览窗口最小长边，单位 像素
DEFAULT_POINT_SIZE = 1.5  # 3D 点大小
DEFAULT_2D_WINDOW_NAME = "Orbbec optimized three plane pose"  # 2D 窗口名，ASCII
DEFAULT_3D_WINDOW_NAME = "Orbbec optimized three plane pose 3D"  # 3D 窗口名，ASCII
PLANE_PREVIEW_COLORS = (
    Color.from_rgb(0.10, 0.70, 1.00),
    Color.from_rgb(1.00, 0.35, 0.20),
    Color.from_rgb(0.25, 0.95, 0.35),
)
MARKER_PREVIEW_COLOR = Color.from_rgb(1.0, 1.0, 0.0)
TRAY_MASK_PREVIEW_COLOR = Color.from_rgb(1.0, 0.0, 0.0)
# endregion


# region 数据结构
@dataclass(frozen=True)
class PipelineConfig:
    enable_tray_exclusion: bool
    tray_use_sam: bool
    pose_smooth_frames: int
    compute_min_interval_s: float


@dataclass(frozen=True)
class CaptureJob:
    frame_idx: int
    points: np.ndarray
    color_bgr: np.ndarray | None
    intrinsics: CameraIntrinsics
    img_w: int
    img_h: int


@dataclass(frozen=True)
class PipelineResult:
    frame_idx: int
    xyz: np.ndarray
    rgb: np.ndarray
    labels: np.ndarray
    pose: CoordinateFramePose | None
    tray_detections: list[TrayDetection]
    tray_excluded_points: int
    overlay_bgr: np.ndarray
    timings_ms: dict[str, float]


# endregion


# region 主流程
def main(
    enable_tray_exclusion: bool = DEFAULT_ENABLE_TRAY_EXCLUSION,
    tray_use_sam: bool = DEFAULT_TRAY_USE_SAM,
    pose_smooth_frames: int = DEFAULT_POSE_SMOOTH_FRAMES,
    compute_min_interval_s: float = DEFAULT_COMPUTE_MIN_INTERVAL_S,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
) -> None:
    cfg = PipelineConfig(
        enable_tray_exclusion=bool(enable_tray_exclusion),
        tray_use_sam=bool(tray_use_sam),
        pose_smooth_frames=int(np.clip(pose_smooth_frames, 1, 15)),
        compute_min_interval_s=max(0.0, float(compute_min_interval_s)),
    )
    logger.info(
        f"优化全流程参数：tray_exclusion {cfg.enable_tray_exclusion}, tray_use_sam {cfg.tray_use_sam}, "
        f"pose_smooth_frames {cfg.pose_smooth_frames}, compute_min_interval_s {cfg.compute_min_interval_s:.3f} s"
    )
    options = SessionOptions(timeout_ms=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))
    with Gemini305(options=options) as session:
        cam = session.get_camera_param()
        projection_intrinsics = session.get_projection_intrinsics()
        point_filter = session.create_point_cloud_filter(camera_param=cam)
        _run_pipeline(
            session=session,
            point_filter=point_filter,
            projection_intrinsics=projection_intrinsics,
            cfg=cfg,
        )


# endregion


# region 实时管线
def _run_pipeline(
    session: Gemini305,
    point_filter,
    projection_intrinsics: CameraIntrinsics,
    cfg: PipelineConfig,
) -> None:
    img_w = int(projection_intrinsics.width)
    img_h = int(projection_intrinsics.height)
    job_queue: queue.Queue[CaptureJob | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[PipelineResult] = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    busy_event = threading.Event()
    worker = threading.Thread(
        target=_worker_loop,
        args=(job_queue, result_queue, stop_event, busy_event, cfg),
        name="optimized_three_plane_worker",
        daemon=True,
    )
    worker.start()

    vis, stop_flag, raw_pcd, plane_pcd, frame_mesh, marker_mesh = _init_3d_viewer()
    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)
    win_w, win_h = _compute_preview_window_size(img_w, img_h, DEFAULT_MIN_2D_WINDOW_LONG_SIDE)
    cv2.resizeWindow(DEFAULT_2D_WINDOW_NAME, win_w, win_h)

    frame_idx = 0
    submitted = 0
    dropped = 0
    completed = 0
    last_submit_ts = 0.0
    # last_overlay: (H, W, 3) uint8 BGR；计算线程未返回时重复显示上一帧结果，避免预览阻塞。
    last_overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    reference_pose: CoordinateFramePose | None = None
    stabilizer = PoseWindowStabilizer(max_frames=cfg.pose_smooth_frames)
    fps_t0 = time.perf_counter()
    preview_frames = 0

    try:
        while True:
            if stop_flag["flag"]:
                break
            points, color_bgr = _capture_points_once(session=session, point_filter=point_filter)
            if points is None or len(points) == 0:
                _poll_viewer(vis)
                continue
            frame_idx += 1
            preview_frames += 1
            _update_raw_cloud(raw_pcd, _downsample_points(points, DEFAULT_MAX_PREVIEW_POINTS))
            vis.update_geometry(raw_pcd)

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break
                completed += 1
                if result.pose is not None:
                    stable_pose = stabilizer.update(result.pose)
                    result = replace(
                        result,
                        pose=stable_pose,
                        overlay_bgr=_draw_overlay(
                            base_bgr=result.overlay_bgr,
                            labels=result.labels,
                            pose=stable_pose,
                            tray_detections=result.tray_detections,
                        ),
                    )
                    if reference_pose is None:
                        reference_pose = stable_pose
                        logger.success(_format_pose_log("参考坐标系已锁定", reference_pose))
                    logger.info(_format_pose_log(f"帧 {result.frame_idx} 当前坐标系", stable_pose))
                    logger.info(_format_delta_log(reference_pose, stable_pose))
                _update_result_3d(plane_pcd, frame_mesh, marker_mesh, result)
                vis.update_geometry(plane_pcd)
                vis.update_geometry(frame_mesh)
                vis.update_geometry(marker_mesh)
                last_overlay = result.overlay_bgr
                logger.info(
                    f"帧 {result.frame_idx} 完成：tray_excluded {result.tray_excluded_points} 点，"
                    f"total {result.timings_ms.get('total', 0.0):.1f} ms，"
                    f"tray {result.timings_ms.get('tray', 0.0):.1f} ms，"
                    f"pose {result.timings_ms.get('pose', 0.0):.1f} ms"
                )

            now = time.perf_counter()
            if (not busy_event.is_set()) and job_queue.empty() and (now - last_submit_ts >= cfg.compute_min_interval_s):
                try:
                    job_queue.put_nowait(
                        CaptureJob(
                            frame_idx=frame_idx,
                            # points 可能来自 SDK 内部缓冲，copy 后交给计算线程，避免下一帧覆盖同一块内存。
                            points=np.asarray(points, dtype=np.float32).copy(),
                            color_bgr=None if color_bgr is None else color_bgr.copy(),
                            intrinsics=projection_intrinsics,
                            img_w=img_w,
                            img_h=img_h,
                        )
                    )
                    submitted += 1
                    last_submit_ts = now
                except queue.Full:
                    dropped += 1
            else:
                dropped += 1

            cv2.imshow(DEFAULT_2D_WINDOW_NAME, last_overlay)
            if cv2.waitKey(1) == 27:
                break
            if not _poll_viewer(vis) or cv2.getWindowProperty(DEFAULT_2D_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if now - fps_t0 >= 3.0:
                preview_fps = preview_frames / max(1e-6, now - fps_t0)
                logger.info(
                    f"性能状态：preview_fps {preview_fps:.1f}, submitted {submitted}, completed {completed}, dropped {dropped}"
                )
                fps_t0 = now
                preview_frames = 0
    finally:
        stop_event.set()
        try:
            job_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=1.0)
        vis.destroy_window()
        cv2.destroyWindow(DEFAULT_2D_WINDOW_NAME)


def _worker_loop(
    job_queue: queue.Queue[CaptureJob | None],
    result_queue: queue.Queue[PipelineResult],
    stop_event: threading.Event,
    busy_event: threading.Event,
    cfg: PipelineConfig,
) -> None:
    tray_excluder = None
    if cfg.enable_tray_exclusion:
        tray_excluder = TrayPointExcluder(TrayDetectionConfig(use_sam=cfg.tray_use_sam))
    pose_cfg = PlanePoseConfig()
    while not stop_event.is_set():
        job = job_queue.get()
        if job is None:
            break
        busy_event.set()
        try:
            result = _run_compute_job(job, pose_cfg, tray_excluder)
            _put_latest_result(result_queue, result)
        except Exception as exc:
            logger.exception(f"帧 {job.frame_idx} 优化管线计算失败：{exc}")
        finally:
            busy_event.clear()
            job_queue.task_done()


def _run_compute_job(
    job: CaptureJob, pose_cfg: PlanePoseConfig, tray_excluder: TrayPointExcluder | None
) -> PipelineResult:
    t0 = time.perf_counter()
    # 计算线程只取 XYZ 三列参与几何计算，颜色单独归一化给 2D/3D 预览使用。
    # xyz: (N, 3) float64，单位 mm；rgb: (N, 3) float64，范围 0-1。
    xyz = np.asarray(job.points[:, :3], dtype=np.float64)
    rgb = _extract_rgb(job.points)
    # uv: (N, 2) int32，valid_proj: (N,) bool；二者都与 xyz 原始点顺序对齐。
    uv, valid_proj = project_points_to_image(xyz, job.intrinsics)
    base_bgr = (
        cv2.resize(job.color_bgr, (job.img_w, job.img_h), interpolation=cv2.INTER_LINEAR)
        if job.color_bgr is not None
        else _rasterize_rgb(xyz, rgb, uv, valid_proj, job.img_w, job.img_h)
    )

    tray_t0 = time.perf_counter()
    if tray_excluder is not None:
        tray_result = tray_excluder.exclude_points(base_bgr, uv, valid_proj, xyz.shape[0])
        # excluded: (N,) bool；True 的点在 estimate_three_plane_pose 内会被标记为 -2。
        excluded = tray_result.excluded_mask
        tray_dets = tray_result.detections
    else:
        excluded = np.zeros((xyz.shape[0],), dtype=bool)
        tray_dets = []
    tray_ms = (time.perf_counter() - tray_t0) * 1000.0

    pose_t0 = time.perf_counter()
    # pose_result.labels 与 xyz 逐点对齐，2D 和 3D 预览都复用同一份标签数组。
    pose_result = estimate_three_plane_pose(xyz, excluded_mask=excluded, config=pose_cfg)
    pose_ms = (time.perf_counter() - pose_t0) * 1000.0
    overlay = _draw_overlay(base_bgr.copy(), pose_result.labels, pose_result.pose, tray_dets)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return PipelineResult(
        frame_idx=job.frame_idx,
        xyz=xyz,
        rgb=rgb,
        labels=pose_result.labels,
        pose=pose_result.pose,
        tray_detections=tray_dets,
        tray_excluded_points=int(np.count_nonzero(excluded)),
        overlay_bgr=overlay,
        timings_ms={"total": total_ms, "tray": tray_ms, "pose": pose_ms},
    )


# endregion


# region Orbbec 与可视化工具
def _capture_points_once(session: Gemini305, point_filter) -> tuple[np.ndarray | None, np.ndarray | None]:
    frames = session.wait_for_frames()
    if frames is None:
        return None, None
    color_bgr = _decode_color_frame_bgr(frames.get_color_frame())
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None, color_bgr
    point_frames, use_color = session.prepare_frame_for_point_cloud(frames)
    set_point_cloud_filter_format(point_filter, depth_scale=float(depth_frame.get_depth_scale()), use_color=use_color)
    cloud_frame = point_filter.process(point_frames)
    if cloud_frame is None:
        return None, color_bgr
    raw_points = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
    # filter_points_for_sensor 返回裁切后的 (N, 3/6) 点云，XYZ 单位 mm，可直接进入 src.pointcloud 算法。
    points = session.filter_points_for_sensor(raw_points, max_depth_mm=DEFAULT_MAX_DEPTH_MM, apply_sensor_frustum=True)
    if points.size == 0:
        return None, color_bgr
    return points, color_bgr


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
    # pyorbbecsdk 返回的是一维缓冲区，这里按格式恢复为 OpenCV 使用的 BGR 图像。
    if color_format == OBFormat.RGB:
        return cv2.cvtColor(np.resize(data, (height, width, 3)), cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3)).copy()
    if color_format in (OBFormat.YUYV, OBFormat.YUY2):
        return cv2.cvtColor(np.resize(data, (height, width, 2)), cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        return cv2.cvtColor(np.resize(data, (height, width, 2)), cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.NV12:
        return cv2.cvtColor(np.resize(data, (height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        return cv2.cvtColor(np.resize(data, (height * 3 // 2, width)), cv2.COLOR_YUV2BGR_NV21)
    if color_format == OBFormat.I420:
        return cv2.cvtColor(np.resize(data, (height * 3 // 2, width)), cv2.COLOR_YUV2BGR_I420)
    return None


def _init_3d_viewer():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(DEFAULT_3D_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = DEFAULT_POINT_SIZE
        # Open3D 期望 background_color 为 (3,) float64 RGB。
        opt.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)
    stop = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    raw_pcd = o3d.geometry.PointCloud()
    plane_pcd = o3d.geometry.PointCloud()
    frame_mesh = _empty_mesh()
    marker_mesh = _empty_mesh()
    for geo in (axis, raw_pcd, plane_pcd, frame_mesh, marker_mesh):
        vis.add_geometry(geo)
    view = vis.get_view_control()
    if view is not None:
        view.set_lookat([0.0, 0.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
    return vis, stop, raw_pcd, plane_pcd, frame_mesh, marker_mesh


def _update_raw_cloud(pcd: o3d.geometry.PointCloud, points: np.ndarray) -> None:
    # Open3D 点云顶点和颜色都要求 (N, 3) float64，颜色范围 0-1。
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = _extract_rgb(points)
    # 原始点云颜色压暗，避免干扰后续三平面高亮点云。
    pcd.colors = o3d.utility.Vector3dVector(np.clip(0.35 * rgb + 0.15, 0.0, 1.0))


def _update_result_3d(
    plane_pcd: o3d.geometry.PointCloud,
    frame_mesh: o3d.geometry.TriangleMesh,
    marker_mesh: o3d.geometry.TriangleMesh,
    result: PipelineResult,
) -> None:
    # valid: (N,) bool；只有 0/1/2 三平面标签进入高亮点云。
    valid = result.labels >= 0
    # colors: (N, 3) float64；先全 0，再按布尔掩码批量写入每个平面的 RGB。
    colors = np.zeros_like(result.rgb)
    for idx, color in enumerate(PLANE_PREVIEW_COLORS):
        colors[result.labels == idx] = color.as_array(normalized=True)
    # 只把三平面内点写入高亮点云，未分配点和料盘排除点保持不可见。
    plane_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(result.xyz[valid], dtype=np.float64))
    plane_pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors[valid], dtype=np.float64))
    if result.pose is not None:
        _update_pose_frame_mesh(frame_mesh, result.pose, 80.0)
        _update_marker_mesh(marker_mesh, result.pose.axis.origin.as_array(), 7.0, MARKER_PREVIEW_COLOR)


def _update_pose_frame_mesh(mesh: o3d.geometry.TriangleMesh, pose: CoordinateFramePose, size_mm: float) -> None:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size_mm), origin=[0.0, 0.0, 0.0])
    # mat: (4, 4) 齐次变换；Open3D mesh.transform 直接应用该矩阵。
    frame.transform(np.asarray(pose.axis.to_transform().as_SE3(), dtype=np.float64))
    mesh.vertices = frame.vertices
    mesh.triangles = frame.triangles
    mesh.vertex_colors = frame.vertex_colors
    mesh.vertex_normals = frame.vertex_normals


def _update_marker_mesh(mesh: o3d.geometry.TriangleMesh, origin_mm: np.ndarray, radius_mm: float, color: Color) -> None:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius_mm), resolution=16)
    marker.translate(np.asarray(origin_mm, dtype=np.float64))
    marker.paint_uniform_color(color.as_array(normalized=True))
    marker.compute_vertex_normals()
    mesh.vertices = marker.vertices
    mesh.triangles = marker.triangles
    mesh.vertex_colors = marker.vertex_colors
    mesh.vertex_normals = marker.vertex_normals


def _draw_overlay(
    base_bgr: np.ndarray,
    labels: np.ndarray,
    pose: CoordinateFramePose | None,
    tray_detections: list[TrayDetection],
) -> np.ndarray:
    # out: (H, W, 3) BGR 图像副本；所有 2D 标注只写副本，不改原始输入。
    out = base_bgr.copy()
    for tray in tray_detections:
        _draw_tray_mask_overlay(out, tray)
    cv2.putText(
        out,
        f"planes assigned={int(np.count_nonzero(labels >= 0))} tray_excluded={sum(t.excluded_points for t in tray_detections)}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if pose is not None:
        origin = pose.axis.origin
        cv2.putText(
            out,
            f"XYZ {origin.x:.1f} {origin.y:.1f} {origin.z:.1f} mm | "
            f"RPY {pose.rpy_deg[0]:.1f} {pose.rpy_deg[1]:.1f} {pose.rpy_deg[2]:.1f} deg",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def _draw_tray_mask_overlay(out_bgr: np.ndarray, tray: TrayDetection) -> None:
    """按料盘实际 mask 区域绘制 2D 预览。

    Parameters
    ----------
    out_bgr:
        待绘制的 BGR 图像，形状为 `(H, W, 3)`，dtype 为 `uint8`。该函数原地修改图像。
    tray:
        料盘检测结果。`mask` 形状为 `(H, W)`，非零像素表示实际排除区域。

    Notes
    -----
    优化脚本不再只画 `contour` 外接线，而是优先使用 `mask` 做半透明填充。
    当 SAM 启用时，mask 是模型分割区域；当 SAM 关闭时，mask 会退化为检测框区域。
    """
    mask = np.asarray(tray.mask > 0, dtype=bool)
    if mask.shape[:2] != out_bgr.shape[:2] or not np.any(mask):
        return

    # out_bgr[mask]: (K, 3) uint8，高级索引取出实际料盘区域像素，做半透明红色覆盖。
    bgr = TRAY_MASK_PREVIEW_COLOR.as_array(normalized=True)[::-1] * 255.0
    out_bgr[mask] = np.clip(0.55 * out_bgr[mask].astype(np.float32) + 0.45 * bgr, 0.0, 255.0).astype(np.uint8)

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) >= 20.0:
            cv2.polylines(out_bgr, [contour.astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)

    moments = cv2.moments(mask.astype(np.uint8))
    if abs(float(moments["m00"])) <= 1e-6:
        return
    center_x = int(round(float(moments["m10"]) / float(moments["m00"])))
    center_y = int(round(float(moments["m01"]) / float(moments["m00"])))
    cv2.putText(
        out_bgr,
        f"excluded tray {tray.confidence_2d:.2f} pts={tray.excluded_points}",
        (center_x, center_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def _extract_rgb(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 6:
        # points[:, 3:6]: (N, 3)，SDK 可能给 0-255，也可能给 0-1。
        rgb = np.asarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        # SDK 颜色可能是 0-255 或 0-1，这里统一归一化到 Open3D 需要的 0-1。
        return np.clip(rgb, 0.0, 1.0).astype(np.float64)
    return np.full((points.shape[0], 3), 0.72, dtype=np.float64)


def _rasterize_rgb(
    xyz: np.ndarray, rgb: np.ndarray, uv: np.ndarray, valid_proj: np.ndarray, w: int, h: int
) -> np.ndarray:
    """把带颜色点云栅格化为 BGR 预览图。

    Parameters
    ----------
    xyz:
        点云坐标数组，形状为 `(N, 3)`，单位 mm。
    rgb:
        点云颜色数组，形状为 `(N, 3)`，范围 0-1，顺序为 RGB。
    uv:
        点云投影像素坐标，形状为 `(N, 2)`，dtype 为 `int32`。
    valid_proj:
        有效投影掩码，形状为 `(N,)`，dtype 为 `bool`。
    w, h:
        输出图像宽高，单位 像素。

    Returns
    -------
    image_bgr:
        BGR 图像，形状为 `(H, W, 3)`，dtype 为 `uint8`。
    """
    # out: (H, W, 3) uint8 BGR 图像，仅用于测试脚本没有彩色帧时的 2D 预览兜底。
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


def _format_pose_log(prefix: str, pose: CoordinateFramePose) -> str:
    origin = pose.axis.origin
    roll, pitch, yaw = pose.rpy_deg
    x_axis = pose.axis.x_axis
    z_axis = pose.axis.z_axis
    return (
        f"{prefix}: XYZ {origin.x:.3f}, {origin.y:.3f}, {origin.z:.3f} mm; "
        f"RPY {roll:.3f}, {pitch:.3f}, {yaw:.3f} deg; "
        f"X {x_axis.x:.4f}, {x_axis.y:.4f}, {x_axis.z:.4f}; "
        f"Z {z_axis.x:.4f}, {z_axis.y:.4f}, {z_axis.z:.4f}; "
        f"residual {pose.residual:.6f} mm"
    )


def _format_delta_log(reference: CoordinateFramePose, current: CoordinateFramePose) -> str:
    delta = relative_pose(reference, current)
    origin = delta.axis.origin
    roll, pitch, yaw = delta.rpy_deg
    return (
        f"相对参考 delta：XYZ {origin.x:.3f}, {origin.y:.3f}, {origin.z:.3f} mm；"
        f"RPY {roll:.3f}, {pitch:.3f}, {yaw:.3f} deg"
    )


def _compute_preview_window_size(src_w: int, src_h: int, min_long_side: int) -> tuple[int, int]:
    long_side = max(1, src_w, src_h)
    if long_side >= min_long_side:
        return max(1, src_w), max(1, src_h)
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= int(max_points):
        return points
    step = max(1, int(np.ceil(len(points) / float(max_points))))
    # 步长采样保持二维数组列结构不变，输出形状约为 (max_points, C)。
    return points[::step]


def _empty_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    # 用远处的单三角形占位，后续只替换 vertices/triangles，避免反复增删 geometry。
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0], [0.0, 1.0, -10000.0]])
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32))
    return mesh


def _poll_viewer(vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    alive = vis.poll_events()
    vis.update_renderer()
    return bool(alive)


def _put_latest_result(result_queue: queue.Queue[PipelineResult], result: PipelineResult) -> None:
    while True:
        try:
            result_queue.put_nowait(result)
            return
        except queue.Full:
            try:
                result_queue.get_nowait()
            except queue.Empty:
                return


# endregion


# region CLI
def _parse_cli() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Orbbec 三平面位姿性能优化全流程测试")
    parser.add_argument(
        "--tray-exclusion", action=argparse.BooleanOptionalAction, default=DEFAULT_ENABLE_TRAY_EXCLUSION
    )
    parser.add_argument("--tray-use-sam", action=argparse.BooleanOptionalAction, default=DEFAULT_TRAY_USE_SAM)
    parser.add_argument("--pose-smooth-frames", type=int, default=DEFAULT_POSE_SMOOTH_FRAMES)
    parser.add_argument("--compute-min-interval-s", type=float, default=DEFAULT_COMPUTE_MIN_INTERVAL_S)
    args = parser.parse_args()
    return PipelineConfig(
        enable_tray_exclusion=bool(args.tray_exclusion),
        tray_use_sam=bool(args.tray_use_sam),
        pose_smooth_frames=int(np.clip(args.pose_smooth_frames, 1, 15)),
        compute_min_interval_s=max(0.0, float(args.compute_min_interval_s)),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_cfg = _parse_cli()
        main(
            enable_tray_exclusion=cli_cfg.enable_tray_exclusion,
            tray_use_sam=cli_cfg.tray_use_sam,
            pose_smooth_frames=cli_cfg.pose_smooth_frames,
            compute_min_interval_s=cli_cfg.compute_min_interval_s,
        )
    else:
        main()
# endregion
