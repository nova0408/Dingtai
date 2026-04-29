from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
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
DEFAULT_REFINE_PLANE_MODELS = True  # RANSAC 后是否用整片平面内点 PCA 重新拟合法线
DEFAULT_PLANE_REFINE_MAX_POINTS = 80_000  # PCA 重新拟合单帧最多使用点数，单位 点
DEFAULT_BOTTOM_AXIS = "y"  # 底面法向参考轴，可选 x/y/z；Orbbec 视图中 y 轴通常接近竖直方向
DEFAULT_FRAME_SIZE_MM = 80.0  # 计算坐标系在 3D 预览中的坐标轴长度，单位 mm
DEFAULT_FRAME_Z_HINT = (0.0594, -0.9020, -0.4276)  # 固定 Z 轴大致方向；符号不确定时先任取一个
DEFAULT_FRAME_X_HINT = (0.9178, -0.1191, 0.3788)  # 固定 X 轴大致方向，用于消除绕 Z 轴的方向漂移
DEFAULT_USE_FIXED_X_HINT_AXIS = True  # True=直接使用固定 X 参考方向投影，避免侧面法线抖动导致 yaw 跳变
DEFAULT_POSE_SMOOTHING = True  # 是否对输出坐标系做时间平滑
DEFAULT_POSE_SMOOTH_FRAMES = 1  # 平滑最多使用最近 N 个实际计算帧，最大强制限制为 15
MAX_POSE_SMOOTH_FRAMES = 5  # 平滑窗口硬上限，避免历史缓存拖慢跟随
DEFAULT_EXCLUDE_TRAY_WITH_ZERO_SHOT = True  # 是否启用 zero-shot 料盘识别，并在拟合平面前剔除料盘点
DEFAULT_TRAY_PROMPT = "black tray,black pallet,rectangular black tray"  # 料盘 zero-shot 提示词
DEFAULT_TRAY_TARGET_KEYWORDS = "rectangular black tray,black tray,black pallet"  # 料盘目标关键词
DEFAULT_TRAY_MIN_CONFIDENCE = 0.35  # 料盘识别最低置信度
DEFAULT_TRAY_USE_SAM = True  # 是否使用 SAM 细化料盘 mask
DEFAULT_TRAY_DETECT_MAX_SIDE = 512  # 料盘识别输入最长边，单位 像素
DEFAULT_ALPHA = 0.45  # 2D/3D 平面颜色叠加透明度
DEFAULT_WINDOW_WIDTH = 1440  # 3D 窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 3D 窗口高度，单位 像素
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 预览窗口最小长边，单位 像素
DEFAULT_POINT_SIZE = 1.5  # 3D 点大小
DEFAULT_2D_WINDOW_NAME = "Orbbec three plane pose frame"  # 2D 窗口名，ASCII
DEFAULT_3D_WINDOW_NAME = "Orbbec three plane pose frame 3D"  # 3D 窗口名，ASCII
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
class CoordinateFramePose:
    origin_mm: np.ndarray
    rotation: np.ndarray
    rpy_deg: np.ndarray
    residual_mm: float


class PoseStabilizer:
    def __init__(self, max_frames: int) -> None:
        self.max_frames = int(np.clip(max_frames, 1, MAX_POSE_SMOOTH_FRAMES))
        self._poses: deque[CoordinateFramePose] = deque(maxlen=self.max_frames)

    def update(self, pose: CoordinateFramePose) -> CoordinateFramePose:
        self._poses.append(pose)
        if len(self._poses) == 1:
            return pose

        latest = self._poses[-1]
        ref_x = latest.rotation[:, 0]
        ref_z = latest.rotation[:, 2]
        origins: list[np.ndarray] = []
        x_axes: list[np.ndarray] = []
        z_axes: list[np.ndarray] = []
        for item in self._poses:
            x_axis_item = item.rotation[:, 0]
            z_axis_item = item.rotation[:, 2]
            if float(np.dot(x_axis_item, ref_x)) < 0.0:
                x_axis_item = -x_axis_item
            if float(np.dot(z_axis_item, ref_z)) < 0.0:
                z_axis_item = -z_axis_item
            origins.append(item.origin_mm)
            x_axes.append(x_axis_item)
            z_axes.append(z_axis_item)

        origin = np.mean(np.stack(origins, axis=0), axis=0)
        z_axis = _normalize_vector(np.mean(np.stack(z_axes, axis=0), axis=0))
        x_axis = _normalize_vector(np.mean(np.stack(x_axes, axis=0), axis=0))
        x_axis = _normalize_vector(x_axis - float(np.dot(x_axis, z_axis)) * z_axis)
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = ref_x
        y_axis = _normalize_vector(np.cross(z_axis, x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        rotation = np.column_stack([x_axis, y_axis, z_axis])

        out = CoordinateFramePose(
            origin_mm=np.asarray(origin, dtype=np.float64),
            rotation=np.asarray(rotation, dtype=np.float64),
            rpy_deg=_rotation_matrix_to_rpy_deg(rotation),
            residual_mm=pose.residual_mm,
        )
        return out


@dataclass(frozen=True)
class TrayExclusion:
    label_text: str
    confidence_2d: float
    contour: np.ndarray
    excluded_points: int


@dataclass(frozen=True)
class PlaneDetectionResult:
    frame_idx: int
    xyz: np.ndarray
    colors: np.ndarray
    base_bgr: np.ndarray
    labels: np.ndarray
    overlay_bgr: np.ndarray
    planes: list[PlanePatch]
    pose: CoordinateFramePose | None
    tray_exclusions: list[TrayExclusion]
    tray_excluded_points: int
    elapsed_ms: float


@dataclass(frozen=True)
class PlaneDetectorConfig:
    max_ransac_points: int
    plane_distance_mm: float
    plane_min_points: int
    ransac_iterations: int
    refine_plane_models: bool
    plane_refine_max_points: int
    bottom_axis: str
    frame_z_hint: np.ndarray
    frame_x_hint: np.ndarray
    use_fixed_x_hint_axis: bool
    pose_smoothing: bool
    pose_smooth_frames: int
    exclude_tray_with_zero_shot: bool
    tray_prompt: str
    tray_target_keywords: str
    tray_min_confidence: float
    tray_use_sam: bool
    tray_detect_max_side: int
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
    refine_plane_models: bool = DEFAULT_REFINE_PLANE_MODELS,
    plane_refine_max_points: int = DEFAULT_PLANE_REFINE_MAX_POINTS,
    bottom_axis: str = DEFAULT_BOTTOM_AXIS,
    frame_z_hint: tuple[float, float, float] | str = DEFAULT_FRAME_Z_HINT,
    frame_x_hint: tuple[float, float, float] | str = DEFAULT_FRAME_X_HINT,
    use_fixed_x_hint_axis: bool = DEFAULT_USE_FIXED_X_HINT_AXIS,
    pose_smoothing: bool = DEFAULT_POSE_SMOOTHING,
    pose_smooth_frames: int = DEFAULT_POSE_SMOOTH_FRAMES,
    exclude_tray_with_zero_shot: bool = DEFAULT_EXCLUDE_TRAY_WITH_ZERO_SHOT,
    tray_prompt: str = DEFAULT_TRAY_PROMPT,
    tray_target_keywords: str = DEFAULT_TRAY_TARGET_KEYWORDS,
    tray_min_confidence: float = DEFAULT_TRAY_MIN_CONFIDENCE,
    tray_use_sam: bool = DEFAULT_TRAY_USE_SAM,
    tray_detect_max_side: int = DEFAULT_TRAY_DETECT_MAX_SIDE,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    cfg = PlaneDetectorConfig(
        max_ransac_points=max(500, int(max_ransac_points)),
        plane_distance_mm=max(0.1, float(plane_distance_mm)),
        plane_min_points=max(20, int(plane_min_points)),
        ransac_iterations=max(50, int(ransac_iterations)),
        refine_plane_models=bool(refine_plane_models),
        plane_refine_max_points=max(100, int(plane_refine_max_points)),
        bottom_axis=str(bottom_axis).strip().lower(),
        frame_z_hint=_coerce_hint_vector(frame_z_hint, name="frame_z_hint"),
        frame_x_hint=_coerce_hint_vector(frame_x_hint, name="frame_x_hint"),
        use_fixed_x_hint_axis=bool(use_fixed_x_hint_axis),
        pose_smoothing=bool(pose_smoothing),
        pose_smooth_frames=int(np.clip(pose_smooth_frames, 1, MAX_POSE_SMOOTH_FRAMES)),
        exclude_tray_with_zero_shot=bool(exclude_tray_with_zero_shot),
        tray_prompt=str(tray_prompt).strip(),
        tray_target_keywords=str(tray_target_keywords).strip(),
        tray_min_confidence=float(np.clip(tray_min_confidence, 0.0, 1.0)),
        tray_use_sam=bool(tray_use_sam),
        tray_detect_max_side=max(128, int(tray_detect_max_side)),
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
            f"三平面坐标系实时检测启动：capture_fps {capture_fps} fps, max_depth_mm {float(max_depth_mm):.1f} mm, "
            f"plane_distance_mm {cfg.plane_distance_mm:.2f} mm, plane_min_points {cfg.plane_min_points} 点"
        )
        logger.info(
            f"平面法线稳定：refine_plane_models {cfg.refine_plane_models}, "
            f"plane_refine_max_points {cfg.plane_refine_max_points} 点。"
        )
        logger.info(
            "坐标系定义：原点=三个平面交点，Z 轴=底面法线并对齐固定近似方向，X 轴=两个侧面法线叉乘后投影到底面并对齐固定近似方向，RPY 采用 roll(X)/pitch(Y)/yaw(Z)，单位 deg。"
        )
        logger.info(
            f"固定方向约束：frame_z_hint {cfg.frame_z_hint.tolist()}，frame_x_hint {cfg.frame_x_hint.tolist()}。"
        )
        logger.info(
            f"稳定策略：use_fixed_x_hint_axis {cfg.use_fixed_x_hint_axis}, "
            f"pose_smoothing {cfg.pose_smoothing}, pose_smooth_frames {cfg.pose_smooth_frames} 帧。"
        )
        tray_detector = _build_tray_exclusion_detector(cfg)
        logger.info(
            f"料盘排除：enabled {cfg.exclude_tray_with_zero_shot}, prompt {cfg.tray_prompt}, "
            f"target_keywords {cfg.tray_target_keywords}, min_confidence {cfg.tray_min_confidence:.2f}, use_sam {cfg.tray_use_sam}"
        )
        logger.info("第一次成功计算出的坐标系会被固定为参考坐标系，用于后续视角转动后的 delta 位姿对比。")
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
            tray_detector=tray_detector,
        )


# endregion


# region 料盘识别排除
def _build_tray_exclusion_detector(cfg: PlaneDetectorConfig):
    if not cfg.exclude_tray_with_zero_shot:
        return None

    from test.pointcloud import test_orbbec_realtime_plane_segmentation_zero_shot as zs

    logger.info("加载 zero-shot 料盘识别器，用于平面拟合前剔除料盘点。")
    return zs.ZeroShotObjectPartitionDetector(
        gd_model_id=zs.DEFAULT_GD_MODEL_ID,
        sam_model_id=zs.DEFAULT_SAM_MODEL_ID,
        hf_cache_dir=zs.DEFAULT_HF_CACHE_DIR,
        hf_local_files_only=zs.DEFAULT_HF_LOCAL_FILES_ONLY,
        device=zs.DEFAULT_DEVICE,
        proxy_url=zs.DEFAULT_PROXY_URL,
        prompt=cfg.tray_prompt,
        target_keywords=cfg.tray_target_keywords,
        strict_target_filter=True,
        max_targets=zs.DEFAULT_MAX_TARGETS,
        use_sam=cfg.tray_use_sam,
        box_threshold=zs.DEFAULT_BOX_THRESHOLD,
        text_threshold=zs.DEFAULT_TEXT_THRESHOLD,
        min_target_conf=cfg.tray_min_confidence,
        topk_objects=zs.DEFAULT_TOPK_OBJECTS,
        sam_max_boxes=zs.DEFAULT_SAM_MAX_BOXES,
        sam_primary_only=zs.DEFAULT_SAM_PRIMARY_ONLY,
        sam_secondary_conf_threshold=zs.DEFAULT_SAM_SECONDARY_CONF_THRESHOLD,
        combine_prompts_forward=zs.DEFAULT_COMBINE_PROMPTS_FORWARD,
        min_mask_pixels=zs.DEFAULT_MIN_MASK_PIXELS,
        mask_iou_suppress=zs.DEFAULT_MASK_IOU_SUPPRESS,
        detect_max_side=cfg.tray_detect_max_side,
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
    tray_detector,
) -> None:
    job_queue: queue.Queue[CaptureJob | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[PlaneDetectionResult] = queue.Queue(maxsize=2)
    worker_busy = threading.Event()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_plane_worker,
        args=(job_queue, result_queue, worker_busy, stop_event, cfg, tray_detector),
        name="three_plane_worker",
        daemon=True,
    )
    worker.start()

    (
        vis,
        stop_flag,
        raw_pcd,
        plane_pcd,
        plane_meshes,
        frame_mesh,
        current_marker,
    ) = _init_3d_viewer()
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
    reference_pose: CoordinateFramePose | None = None
    pose_stabilizer = PoseStabilizer(max_frames=cfg.pose_smooth_frames) if cfg.pose_smoothing else None
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
                if result.pose is not None and pose_stabilizer is not None:
                    stable_pose = pose_stabilizer.update(result.pose)
                    stable_overlay = _draw_2d_overlay(
                        base_bgr=result.base_bgr,
                        planes=result.planes,
                        labels=result.labels,
                        pose=stable_pose,
                        tray_exclusions=result.tray_exclusions,
                        alpha=cfg.alpha,
                    )
                    result = replace(result, pose=stable_pose, overlay_bgr=stable_overlay)
                _apply_detection_result(
                    result=result,
                    plane_pcd=plane_pcd,
                    plane_meshes=plane_meshes,
                    frame_mesh=frame_mesh,
                    current_marker=current_marker,
                    vis=vis,
                )
                if result.pose is not None:
                    if reference_pose is None:
                        reference_pose = result.pose
                        logger.success(_format_pose_log("参考坐标系已锁定", reference_pose))
                    logger.info(_format_pose_log(f"帧 {result.frame_idx} 当前坐标系", result.pose))
                    logger.info(_format_delta_log(reference=reference_pose, current=result.pose))
                last_overlay = result.overlay_bgr
                logger.info(
                    f"帧 {result.frame_idx} 平面检测完成：planes {len(result.planes)}, "
                    f"tray_excluded {result.tray_excluded_points} 点，耗时 {result.elapsed_ms:.1f} ms"
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
    tray_detector,
) -> None:
    while not stop_event.is_set():
        job = job_queue.get()
        if job is None:
            break
        worker_busy.set()
        try:
            result = _detect_three_planes(job=job, cfg=cfg, tray_detector=tray_detector)
            _put_latest_result(result_queue, result)
        except Exception as exc:
            logger.exception(f"帧 {job.frame_idx} 平面检测失败：{exc}")
        finally:
            worker_busy.clear()
            job_queue.task_done()


# endregion


# region 平面检测
def _detect_three_planes(job: CaptureJob, cfg: PlaneDetectorConfig, tray_detector) -> PlaneDetectionResult:
    start = time.perf_counter()
    xyz = np.asarray(job.points[:, :3], dtype=np.float64)
    colors = _extract_rgb(job.points)
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
    tray_excluded, tray_exclusions = _detect_tray_exclusion(
        detector=tray_detector,
        base_bgr=base_2d,
        uv=uv,
        valid_proj=valid_proj,
        total_points=xyz.shape[0],
    )
    plane_candidate_xyz = xyz[~tray_excluded]
    plane_models = _segment_plane_models(
        xyz=plane_candidate_xyz,
        max_ransac_points=cfg.max_ransac_points,
        distance_mm=cfg.plane_distance_mm,
        min_points=cfg.plane_min_points,
        ransac_iterations=cfg.ransac_iterations,
    )
    labels = _assign_points_to_planes(xyz=xyz, plane_models=plane_models, distance_mm=cfg.plane_distance_mm)
    labels[tray_excluded] = -2
    if cfg.refine_plane_models:
        plane_models = _refine_plane_models_by_pca(
            xyz=xyz,
            labels=labels,
            plane_models=plane_models,
            max_points=cfg.plane_refine_max_points,
        )
        labels = _assign_points_to_planes(xyz=xyz, plane_models=plane_models, distance_mm=cfg.plane_distance_mm)
        labels[tray_excluded] = -2
    ordered = _order_planes(plane_models=plane_models, labels=labels, xyz=xyz, bottom_axis=cfg.bottom_axis)
    labels = _remap_labels(labels=labels, ordered_old_ids=[old_id for old_id, _ in ordered])

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

    pose = _compute_coordinate_frame_pose(planes=planes, cfg=cfg)
    overlay = _draw_2d_overlay(
        base_bgr=base_2d,
        planes=planes,
        labels=labels,
        pose=pose,
        tray_exclusions=tray_exclusions,
        alpha=cfg.alpha,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PlaneDetectionResult(
        frame_idx=job.frame_idx,
        xyz=xyz,
        colors=colors,
        base_bgr=base_2d,
        labels=labels,
        overlay_bgr=overlay,
        planes=planes,
        pose=pose,
        tray_exclusions=tray_exclusions,
        tray_excluded_points=int(np.count_nonzero(tray_excluded)),
        elapsed_ms=elapsed_ms,
    )


def _detect_tray_exclusion(
    detector,
    base_bgr: np.ndarray,
    uv: np.ndarray,
    valid_proj: np.ndarray,
    total_points: int,
) -> tuple[np.ndarray, list[TrayExclusion]]:
    excluded = np.zeros((int(total_points),), dtype=bool)
    if detector is None:
        return excluded, []

    try:
        detections = detector.detect(base_bgr)
    except Exception as exc:
        logger.exception(f"料盘识别失败，本帧不执行料盘排除：{exc}")
        return excluded, []
    out: list[TrayExclusion] = []
    for det in detections:
        ids = _collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=det.mask)
        if ids.size == 0:
            continue
        excluded[ids] = True
        out.append(
            TrayExclusion(
                label_text=str(det.label_text),
                confidence_2d=float(det.confidence_2d),
                contour=np.asarray(det.contour, dtype=np.int32),
                excluded_points=int(ids.size),
            )
        )
    return excluded, out


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


def _refine_plane_models_by_pca(
    xyz: np.ndarray,
    labels: np.ndarray,
    plane_models: list[np.ndarray],
    max_points: int,
) -> list[np.ndarray]:
    refined: list[np.ndarray] = []
    for plane_id, seed_model in enumerate(plane_models):
        mask = labels == int(plane_id)
        pts = xyz[mask]
        if pts.shape[0] < 3:
            refined.append(seed_model)
            continue
        pts_sample = _downsample_points(pts, max_points=max_points)
        model = _fit_plane_model_pca(pts_sample, seed_model=seed_model)
        refined.append(model if model is not None else seed_model)
    return refined


def _fit_plane_model_pca(points: np.ndarray, seed_model: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    pts = np.asarray(points[:, :3], dtype=np.float64)
    center = np.mean(pts, axis=0)
    centered = pts - center.reshape(1, 3)
    cov = centered.T @ centered / max(1, pts.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = _normalize_vector(eigvecs[:, int(np.argmin(eigvals))])
    if np.linalg.norm(normal) < 1e-8:
        return None
    seed_normal = _normalize_vector(seed_model[:3])
    if float(np.dot(normal, seed_normal)) < 0.0:
        normal = -normal
    d = -float(np.dot(normal, center))
    return _normalize_plane_model(np.asarray([normal[0], normal[1], normal[2], d], dtype=np.float64))


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


def _collect_indices_in_mask(uv: np.ndarray, valid_proj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    idx = np.where(valid_proj)[0]
    if idx.size == 0:
        return np.empty((0,), dtype=np.int32)
    u = uv[idx, 0]
    v = uv[idx, 1]
    inside = mask[v, u] > 0
    return idx[inside].astype(np.int32)


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


def _compute_coordinate_frame_pose(planes: list[PlanePatch], cfg: PlaneDetectorConfig) -> CoordinateFramePose | None:
    if len(planes) < 3:
        return None

    by_label = {p.label: p for p in planes}
    bottom = by_label.get("bottom")
    left = by_label.get("left_side")
    right = by_label.get("right_side")
    if bottom is None or left is None or right is None:
        return None

    models = [bottom.model, left.model, right.model]
    origin, residual = _intersect_three_planes(models)
    if origin is None:
        return None

    z_axis = _orient_axis_to_hint(_normalize_vector(bottom.model[:3]), cfg.frame_z_hint)
    x_hint = _normalize_vector(cfg.frame_x_hint - float(np.dot(cfg.frame_x_hint, z_axis)) * z_axis)
    x_axis = np.zeros(3, dtype=np.float64)
    if cfg.use_fixed_x_hint_axis and np.linalg.norm(x_hint) >= 1e-8:
        x_axis = x_hint
    else:
        x_axis = _normalize_vector(np.cross(left.model[:3], right.model[:3]))
        x_axis = _normalize_vector(x_axis - float(np.dot(x_axis, z_axis)) * z_axis)
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = _normalize_vector(left.model[:3] - float(np.dot(left.model[:3], z_axis)) * z_axis)
    if np.linalg.norm(x_hint) >= 1e-8 and not cfg.use_fixed_x_hint_axis:
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = x_hint
        elif float(np.dot(x_axis, x_hint)) < 0.0:
            x_axis = -x_axis
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = _normalize_vector(left.model[:3] - float(np.dot(left.model[:3], z_axis)) * z_axis)
    y_axis = _normalize_vector(np.cross(z_axis, x_axis))
    x_axis = _normalize_vector(np.cross(y_axis, z_axis))

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(rotation) < 0.0:
        y_axis = -y_axis
        rotation = np.column_stack([x_axis, y_axis, z_axis])
    rpy_deg = _rotation_matrix_to_rpy_deg(rotation)
    return CoordinateFramePose(
        origin_mm=np.asarray(origin, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        rpy_deg=np.asarray(rpy_deg, dtype=np.float64),
        residual_mm=float(residual),
    )


def _orient_axis_to_hint(axis: np.ndarray, hint: np.ndarray) -> np.ndarray:
    out = _normalize_vector(axis)
    hint_n = _normalize_vector(hint)
    if np.linalg.norm(out) < 1e-8 or np.linalg.norm(hint_n) < 1e-8:
        return out
    if float(np.dot(out, hint_n)) < 0.0:
        return -out
    return out


def _intersect_three_planes(models: list[np.ndarray]) -> tuple[np.ndarray | None, float]:
    a = np.asarray([m[:3] for m in models], dtype=np.float64)
    b = -np.asarray([float(m[3]) for m in models], dtype=np.float64)
    if np.linalg.matrix_rank(a) < 3:
        return None, float("inf")
    origin = np.linalg.solve(a, b)
    residual = float(np.linalg.norm(a @ origin - b))
    return origin, residual


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    sy = float(np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0]))
    singular = sy < 1e-8
    if not singular:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.asarray([roll, pitch, yaw], dtype=np.float64))


def _pose_to_matrix(pose: CoordinateFramePose) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = pose.rotation
    out[:3, 3] = pose.origin_mm
    return out


def _relative_pose(reference: CoordinateFramePose, current: CoordinateFramePose) -> CoordinateFramePose:
    ref_t = _pose_to_matrix(reference)
    cur_t = _pose_to_matrix(current)
    delta = np.linalg.inv(ref_t) @ cur_t
    return CoordinateFramePose(
        origin_mm=delta[:3, 3],
        rotation=delta[:3, :3],
        rpy_deg=_rotation_matrix_to_rpy_deg(delta[:3, :3]),
        residual_mm=current.residual_mm,
    )


# endregion


# region 可视化
def _init_3d_viewer() -> tuple[
    o3d.visualization.VisualizerWithKeyCallback,
    dict[str, bool],
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    list[o3d.geometry.TriangleMesh],
    o3d.geometry.TriangleMesh,
    o3d.geometry.TriangleMesh,
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
    frame_mesh = _empty_mesh()
    current_marker = _empty_mesh()
    vis.add_geometry(axis)
    vis.add_geometry(raw_pcd)
    vis.add_geometry(plane_pcd)
    for mesh in plane_meshes:
        vis.add_geometry(mesh)
    vis.add_geometry(frame_mesh)
    vis.add_geometry(current_marker)
    _apply_reference_camera_view(vis)
    return vis, stop, raw_pcd, plane_pcd, plane_meshes, frame_mesh, current_marker


def _apply_detection_result(
    result: PlaneDetectionResult,
    plane_pcd: o3d.geometry.PointCloud,
    plane_meshes: list[o3d.geometry.TriangleMesh],
    frame_mesh: o3d.geometry.TriangleMesh,
    current_marker: o3d.geometry.TriangleMesh,
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

    if result.pose is not None:
        _update_pose_frame_mesh(frame_mesh, result.pose, DEFAULT_FRAME_SIZE_MM)
        _update_marker_mesh(current_marker, result.pose.origin_mm, radius_mm=7.0, color_rgb=[1.0, 1.0, 0.0])
        vis.update_geometry(frame_mesh)
        vis.update_geometry(current_marker)


def _update_raw_cloud(raw_pcd: o3d.geometry.PointCloud, points: np.ndarray) -> None:
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    raw_pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = _extract_rgb(points)
    muted = 0.35 * rgb + 0.15
    raw_pcd.colors = o3d.utility.Vector3dVector(np.clip(muted, 0.0, 1.0))


def _update_pose_frame_mesh(mesh: o3d.geometry.TriangleMesh, pose: CoordinateFramePose, size_mm: float) -> None:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=float(size_mm),
        origin=[0.0, 0.0, 0.0],
    )
    transform = _pose_to_matrix(pose)
    frame.transform(transform)
    mesh.vertices = frame.vertices
    mesh.triangles = frame.triangles
    mesh.vertex_colors = frame.vertex_colors
    mesh.vertex_normals = frame.vertex_normals


def _update_marker_mesh(
    mesh: o3d.geometry.TriangleMesh,
    origin_mm: np.ndarray,
    radius_mm: float,
    color_rgb: list[float],
) -> None:
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius_mm), resolution=16)
    marker.translate(np.asarray(origin_mm, dtype=np.float64))
    marker.paint_uniform_color(color_rgb)
    marker.compute_vertex_normals()
    mesh.vertices = marker.vertices
    mesh.triangles = marker.triangles
    mesh.vertex_colors = marker.vertex_colors
    mesh.vertex_normals = marker.vertex_normals


def _draw_2d_overlay(
    base_bgr: np.ndarray,
    planes: list[PlanePatch],
    labels: np.ndarray,
    pose: CoordinateFramePose | None,
    tray_exclusions: list[TrayExclusion],
    alpha: float,
) -> np.ndarray:
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
    for tray in tray_exclusions:
        if tray.contour.shape[0] >= 3:
            cv2.polylines(overlay, [tray.contour.astype(np.int32)], True, (0, 0, 255), 3, cv2.LINE_AA)
            center = np.mean(tray.contour, axis=0).astype(np.int32)
            cv2.putText(
                overlay,
                f"excluded tray {tray.confidence_2d:.2f} pts={tray.excluded_points}",
                (int(center[0]), int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
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
    if len(tray_exclusions) > 0:
        excluded_points = sum(t.excluded_points for t in tray_exclusions)
        cv2.putText(
            blended,
            f"tray excluded objects={len(tray_exclusions)} points={excluded_points}",
            (16, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if pose is not None:
        cv2.putText(
            blended,
            "XYZRPY "
            f"{pose.origin_mm[0]:.1f} {pose.origin_mm[1]:.1f} {pose.origin_mm[2]:.1f} mm | "
            f"{pose.rpy_deg[0]:.1f} {pose.rpy_deg[1]:.1f} {pose.rpy_deg[2]:.1f} deg",
            (16, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        blended,
        "3D legend: large RGB=world | small RGB+yellow dot=current frame",
        (16, blended.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
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


def _coerce_hint_vector(value: tuple[float, float, float] | list[float] | np.ndarray | str, name: str) -> np.ndarray:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if len(p.strip()) > 0]
        if len(parts) != 3:
            raise ValueError(f"{name} must be a 3D vector like (0, 1, 0) or '0,1,0'")
        vec = np.asarray([float(p) for p in parts], dtype=np.float64)
    else:
        vec = np.asarray(value, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError(f"{name} must be a 3D vector like (0, 1, 0) or '0,1,0'")
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise ValueError(f"{name} must not be zero")
    return vec / norm


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


def _format_pose_log(prefix: str, pose: CoordinateFramePose) -> str:
    x, y, z = pose.origin_mm
    roll, pitch, yaw = pose.rpy_deg
    x_axis = pose.rotation[:, 0]
    z_axis = pose.rotation[:, 2]
    return (
        f"{prefix}: XYZ {x:.3f}, {y:.3f}, {z:.3f} mm."
        f"RPY {roll:.3f}, {pitch:.3f}, {yaw:.3f} deg."
        f"X 轴 {x_axis[0]:.4f}, {x_axis[1]:.4f}, {x_axis[2]:.4f}；"
        f"Z 轴 {z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f}；"
        f"plane_intersection_residual {pose.residual_mm:.6f} mm"
    )


def _format_delta_log(reference: CoordinateFramePose, current: CoordinateFramePose) -> str:
    delta = _relative_pose(reference=reference, current=current)
    x, y, z = delta.origin_mm
    roll, pitch, yaw = delta.rpy_deg
    return f"相对参考 delta：XYZ {x:.3f}, {y:.3f}, {z:.3f} mm；" f"RPY {roll:.3f}, {pitch:.3f}, {yaw:.3f} deg"


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
def _parse_cli() -> tuple[
    int,
    int,
    float,
    int,
    int,
    float,
    int,
    int,
    bool,
    int,
    str,
    tuple[float, float, float],
    tuple[float, float, float],
    bool,
    bool,
    int,
    bool,
    str,
    str,
    float,
    bool,
    int,
    float,
]:
    parser = argparse.ArgumentParser(description="Orbbec Gemini 305 实时三平面坐标系 XYZRPY 测试脚本")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames timeout in ms")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="preferred capture fps")
    parser.add_argument(
        "--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM, help="max depth for sensor filtering"
    )
    parser.add_argument("--max-preview-points", type=int, default=DEFAULT_MAX_PREVIEW_POINTS, help="max preview points")
    parser.add_argument(
        "--max-ransac-points", type=int, default=DEFAULT_MAX_RANSAC_POINTS, help="max sampled points for RANSAC"
    )
    parser.add_argument(
        "--plane-distance-mm", type=float, default=DEFAULT_PLANE_DISTANCE_MM, help="RANSAC plane distance threshold"
    )
    parser.add_argument(
        "--plane-min-points", type=int, default=DEFAULT_PLANE_MIN_POINTS, help="minimum points per plane"
    )
    parser.add_argument("--ransac-iterations", type=int, default=DEFAULT_RANSAC_ITERATIONS, help="RANSAC iterations")
    parser.add_argument(
        "--refine-plane-models",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REFINE_PLANE_MODELS,
        help="refit each RANSAC plane normal by PCA over its inlier points",
    )
    parser.add_argument(
        "--plane-refine-max-points",
        type=int,
        default=DEFAULT_PLANE_REFINE_MAX_POINTS,
        help="max points per frame for PCA plane refinement",
    )
    parser.add_argument(
        "--bottom-axis", choices=["x", "y", "z"], default=DEFAULT_BOTTOM_AXIS, help="bottom plane reference normal axis"
    )
    parser.add_argument(
        "--frame-z-hint",
        type=_parse_cli_hint_tuple,
        default=DEFAULT_FRAME_Z_HINT,
        help="fixed approximate Z direction, e.g. 0,1,0",
    )
    parser.add_argument(
        "--frame-x-hint",
        type=_parse_cli_hint_tuple,
        default=DEFAULT_FRAME_X_HINT,
        help="fixed approximate X direction, e.g. 1,0,0",
    )
    parser.add_argument(
        "--use-fixed-x-hint-axis",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_FIXED_X_HINT_AXIS,
        help="use fixed X hint projected onto current Z instead of side-plane cross product",
    )
    parser.add_argument(
        "--pose-smoothing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_POSE_SMOOTHING,
        help="smooth output pose over time",
    )
    parser.add_argument(
        "--pose-smooth-frames",
        type=int,
        default=DEFAULT_POSE_SMOOTH_FRAMES,
        help="pose smoothing window in actual computed frames, hard-capped at 15",
    )
    parser.add_argument(
        "--exclude-tray-with-zero-shot",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXCLUDE_TRAY_WITH_ZERO_SHOT,
        help="exclude tray points by zero-shot detector before plane fitting",
    )
    parser.add_argument("--tray-prompt", type=str, default=DEFAULT_TRAY_PROMPT, help="zero-shot tray prompt")
    parser.add_argument(
        "--tray-target-keywords", type=str, default=DEFAULT_TRAY_TARGET_KEYWORDS, help="zero-shot tray target keywords"
    )
    parser.add_argument(
        "--tray-min-confidence",
        type=float,
        default=DEFAULT_TRAY_MIN_CONFIDENCE,
        help="minimum tray detection confidence",
    )
    parser.add_argument(
        "--tray-use-sam",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TRAY_USE_SAM,
        help="use SAM to refine tray mask",
    )
    parser.add_argument(
        "--tray-detect-max-side",
        type=int,
        default=DEFAULT_TRAY_DETECT_MAX_SIDE,
        help="tray detector max input side in pixels",
    )
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
        bool(args.refine_plane_models),
        int(args.plane_refine_max_points),
        str(args.bottom_axis),
        tuple(args.frame_z_hint),
        tuple(args.frame_x_hint),
        bool(args.use_fixed_x_hint_axis),
        bool(args.pose_smoothing),
        int(args.pose_smooth_frames),
        bool(args.exclude_tray_with_zero_shot),
        str(args.tray_prompt),
        str(args.tray_target_keywords),
        float(args.tray_min_confidence),
        bool(args.tray_use_sam),
        int(args.tray_detect_max_side),
        float(args.alpha),
    )


def _parse_cli_hint_tuple(value: str) -> tuple[float, float, float]:
    vec = _coerce_hint_vector(value, name="hint")
    return (float(vec[0]), float(vec[1]), float(vec[2]))


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
            refine_plane_models_arg,
            plane_refine_max_points_arg,
            bottom_axis_arg,
            frame_z_hint_arg,
            frame_x_hint_arg,
            use_fixed_x_hint_axis_arg,
            pose_smoothing_arg,
            pose_smooth_frames_arg,
            exclude_tray_arg,
            tray_prompt_arg,
            tray_target_keywords_arg,
            tray_min_conf_arg,
            tray_use_sam_arg,
            tray_detect_max_side_arg,
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
            refine_plane_models=refine_plane_models_arg,
            plane_refine_max_points=plane_refine_max_points_arg,
            bottom_axis=bottom_axis_arg,
            frame_z_hint=frame_z_hint_arg,
            frame_x_hint=frame_x_hint_arg,
            use_fixed_x_hint_axis=use_fixed_x_hint_axis_arg,
            pose_smoothing=pose_smoothing_arg,
            pose_smooth_frames=pose_smooth_frames_arg,
            exclude_tray_with_zero_shot=exclude_tray_arg,
            tray_prompt=tray_prompt_arg,
            tray_target_keywords=tray_target_keywords_arg,
            tray_min_confidence=tray_min_conf_arg,
            tray_use_sam=tray_use_sam_arg,
            tray_detect_max_side=tray_detect_max_side_arg,
            alpha=alpha_arg,
        )
    else:
        main()
# endregion
