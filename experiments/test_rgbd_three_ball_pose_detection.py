from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import itertools
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from loguru import logger
from pyorbbecsdk import OBFormat

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 opencv-python 才能运行三色小球位姿检测脚本。") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import Gemini305, OrbbecSession, SessionOptions, set_point_cloud_filter_format  # noqa: E402


# region 默认参数（优先在这里直接改）
DEFAULT_TIMEOUT_MS = 120  # 等待相机帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_BALL_DIAMETER_MM = 20.0  # 小球物理直径，单位 mm；可改为 8 或 12
DEFAULT_DIAMETER_TOLERANCE_RATIO = 0.45  # 允许直径误差比例，避免轻微遮挡直接丢失
DEFAULT_MAX_DEPTH_MM = 5000.0  # 点云深度过滤上限，单位 mm
DEFAULT_MIN_DEPTH_POINTS = 18  # 单个候选区域最少有效 3D 点数量，单位 点
DEFAULT_MAX_COLOR_COMPONENTS = 6  # 每种颜色保留的最大候选连通域数量
DEFAULT_MIN_COMPONENT_AREA_PX = 28  # 颜色连通域最小面积，单位 像素
DEFAULT_MIN_CIRCULARITY = 0.46  # 2D 候选最低圆度，1.0 为理想圆
DEFAULT_MIN_MASK_FILL_RATIO = 0.34  # 候选 mask 在外接圆中的最低填充比例
DEFAULT_MIN_CENTER_DISTANCE_RATIO = 1.35  # 不同球中心最小距离相对直径倍率，单位 倍
DEFAULT_RELATIVE_DISTANCE_TOLERANCE_RATIO = 0.30  # 相对边长允许误差比例，约束后续帧候选组合
DEFAULT_DEPTH_TRIM_RATIO = 0.18  # 3D 点云深度裁剪比例，去掉前后离群点
DEFAULT_MAX_CENTER_JUMP_MM = 80.0  # 开启稳定时允许单帧中心跳变，单位 mm
DEFAULT_SMOOTH_ALPHA = 0.55  # 中心指数平滑权重；越大越跟手
DEFAULT_COMPUTE_MIN_INTERVAL_S = 0.04  # 提交检测任务最小间隔，单位 s
DEFAULT_MAX_PREVIEW_POINTS = 120_000  # 3D 原始点云预览最大点数，单位 点
DEFAULT_2D_WINDOW_NAME = "Orbbec three ball pose"  # 2D 窗口名，ASCII
DEFAULT_3D_WINDOW_NAME = "Orbbec three ball pose 3D"  # 3D 窗口名，ASCII
DEFAULT_3D_WINDOW_WIDTH = 1440  # 3D 窗口宽度，单位 像素
DEFAULT_3D_WINDOW_HEIGHT = 900  # 3D 窗口高度，单位 像素
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 预览窗口最小长边，单位 像素
DEFAULT_POINT_SIZE = 1.5  # 3D 点云显示点大小
DEFAULT_RAW_POINT_DIM_FACTOR = 0.42  # 原始点云颜色压暗系数
DEFAULT_USE_FIRST_VALID_FRAME_AS_REFERENCE = True  # 是否用首个有效三球检测建立测试绝对参考
DEFAULT_REQUIRE_KNOWN_MODEL_POSE = False  # True 时必须使用下方模型坐标做刚体位姿估计

# 三个小球在物体坐标系下的已知相对位置，单位 mm。
# 若暂无真实标定值，保持 DEFAULT_USE_FIRST_VALID_FRAME_AS_REFERENCE=True 先用首帧实测关系测试算法链路。
DEFAULT_MODEL_POINTS_MM: dict[str, tuple[float, float, float]] = {
    "red": (0.0, 0.0, 0.0),
    "blue": (80.0, 0.0, 0.0),
    "yellow": (0.0, 60.0, 0.0),
}

# OpenCV HSV 色相范围：H 为 [0, 179]，S/V 为 [0, 255]。
DEFAULT_COLOR_RANGES: dict[str, tuple[tuple[int, int, int, int, int, int], ...]] = {
    "red": ((0, 75, 55, 10, 255, 255), (170, 75, 55, 179, 255, 255)),
    "blue": ((90, 55, 35, 130, 255, 255),),
    "yellow": ((18, 60, 70, 42, 255, 255),),
}

BALL_DRAW_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "blue": (255, 80, 0),
    "yellow": (0, 220, 255),
}

BALL_DRAW_RGB_FLOAT: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.05, 0.05),
    "blue": (0.05, 0.25, 1.0),
    "yellow": (1.0, 0.85, 0.05),
}

BALL_ORDER = ("red", "blue", "yellow")
# endregion


# region 数据结构
@dataclass(frozen=True)
class BallDetectorConfig:
    """三色小球检测配置。"""

    ball_diameter_mm: float
    diameter_tolerance_ratio: float
    max_depth_mm: float
    min_depth_points: int
    max_color_components: int
    min_component_area_px: int
    min_circularity: float
    min_mask_fill_ratio: float
    min_center_distance_ratio: float
    relative_distance_tolerance_ratio: float
    depth_trim_ratio: float
    max_center_jump_mm: float
    smooth_alpha: float
    compute_min_interval_s: float
    use_first_valid_frame_as_reference: bool
    require_known_model_pose: bool
    model_points_mm: dict[str, np.ndarray]
    color_ranges: dict[str, tuple[tuple[int, int, int, int, int, int], ...]]


@dataclass(frozen=True)
class ColorCandidate:
    """单个 2D 颜色连通域候选。"""

    color_name: str
    contour: np.ndarray
    mask: np.ndarray
    center_px: tuple[float, float]
    radius_px: float
    area_px: int
    circularity: float
    fill_ratio: float


@dataclass(frozen=True)
class BallDetection:
    """单个小球检测结果。"""

    color_name: str
    detected: bool
    status: str
    center_mm: np.ndarray | None
    center_px: tuple[float, float] | None
    radius_px: float
    physical_radius_mm: float
    depth_points: int
    score: float
    contour: np.ndarray | None
    mask: np.ndarray | None
    failure_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PoseEstimate:
    """三球心构造出的物体坐标系位姿。"""

    origin_mm: np.ndarray
    rotation: np.ndarray
    transform: np.ndarray
    residual_mm: float
    source: str


@dataclass(frozen=True)
class DetectionResult:
    """一帧检测结果。"""

    frame_idx: int
    detections: dict[str, BallDetection]
    pose: PoseEstimate | None
    reference_pose: PoseEstimate | None
    reference_distances_mm: dict[tuple[str, str], float] | None
    overlay_bgr: np.ndarray
    preview_points: np.ndarray
    intrinsics: Any
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class CaptureJob:
    """提交给后台检测线程的一帧数据。"""

    frame_idx: int
    color_bgr: np.ndarray
    points: np.ndarray
    intrinsics: Any


class CenterStabilizer:
    """按颜色对小球中心做轻量指数平滑。"""

    def __init__(self, max_jump_mm: float, alpha: float) -> None:
        self._max_jump_mm = float(max_jump_mm)
        self._alpha = float(np.clip(alpha, 0.0, 1.0))
        self._centers: dict[str, np.ndarray] = {}

    def update(self, detections: dict[str, BallDetection]) -> dict[str, BallDetection]:
        updated: dict[str, BallDetection] = {}
        for color_name, detection in detections.items():
            if detection.center_mm is None:
                updated[color_name] = detection
                continue

            current = np.asarray(detection.center_mm, dtype=np.float64)
            previous = self._centers.get(color_name)
            if previous is None:
                self._centers[color_name] = current
                updated[color_name] = detection
                continue

            jump = float(np.linalg.norm(current - previous))
            if jump > self._max_jump_mm:
                self._centers[color_name] = current
                updated[color_name] = detection
                continue

            stable = self._alpha * current + (1.0 - self._alpha) * previous
            self._centers[color_name] = stable
            updated[color_name] = replace(detection, center_mm=stable)
        return updated


# endregion


# region 主流程
def main(
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    ball_diameter_mm: float = DEFAULT_BALL_DIAMETER_MM,
    use_first_valid_frame_as_reference: bool = DEFAULT_USE_FIRST_VALID_FRAME_AS_REFERENCE,
    require_known_model_pose: bool = DEFAULT_REQUIRE_KNOWN_MODEL_POSE,
) -> None:
    config = _build_detector_config(
        ball_diameter_mm=ball_diameter_mm,
        use_first_valid_frame_as_reference=use_first_valid_frame_as_reference,
        require_known_model_pose=require_known_model_pose,
    )
    session_options = SessionOptions(
        timeout=int(timeout_ms),
        preferred_capture_fps=max(1, int(capture_fps)),
    )
    logger.info(
        f"三色小球检测启动：ball_diameter {config.ball_diameter_mm:.1f} mm，"
        f"capture_fps {capture_fps} fps，first_frame_reference {config.use_first_valid_frame_as_reference}"
    )

    with Gemini305(options=session_options) as session:
        point_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())
        projection_intrinsics = session.get_projection_intrinsics()
        _run_realtime_detection(
            session=session,
            point_filter=point_filter,
            projection_intrinsics=projection_intrinsics,
            config=config,
        )


def _run_realtime_detection(
    session: OrbbecSession,
    point_filter: Any,
    projection_intrinsics: Any,
    config: BallDetectorConfig,
) -> None:
    job_queue: queue.Queue[CaptureJob | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[DetectionResult] = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    busy_event = threading.Event()
    worker = threading.Thread(
        target=_worker_loop,
        args=(job_queue, result_queue, stop_event, busy_event, config),
        name="three-ball-pose-worker",
        daemon=True,
    )
    worker.start()

    vis, stop_flag, raw_pcd, ball_mesh, center_frame_mesh, pose_frame_mesh, line_set = _init_3d_viewer()
    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)
    win_w, win_h = _compute_preview_window_size(
        int(projection_intrinsics.width),
        int(projection_intrinsics.height),
        DEFAULT_MIN_2D_WINDOW_LONG_SIDE,
    )
    cv2.resizeWindow(DEFAULT_2D_WINDOW_NAME, win_w, win_h)

    frame_idx = 0
    last_submit_ts = 0.0
    submitted = 0
    completed = 0
    dropped = 0
    last_overlay = np.zeros(
        (int(projection_intrinsics.height), int(projection_intrinsics.width), 3),
        dtype=np.uint8,
    )
    fps_t0 = time.perf_counter()
    preview_frames = 0

    try:
        while True:
            if stop_flag["flag"]:
                break

            points, color_bgr = _capture_rgbd_points_once(
                session=session,
                point_filter=point_filter,
                max_depth_mm=config.max_depth_mm,
            )
            if color_bgr is None:
                _poll_viewer(vis)
                continue

            frame_idx += 1
            preview_frames += 1
            if points is not None and points.size > 0:
                preview_points = _downsample_points(points, DEFAULT_MAX_PREVIEW_POINTS)
                _update_raw_cloud(raw_pcd, preview_points)
                vis.update_geometry(raw_pcd)

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break
                completed += 1
                last_overlay = result.overlay_bgr
                _update_detection_3d(
                    ball_mesh=ball_mesh,
                    center_frame_mesh=center_frame_mesh,
                    pose_frame_mesh=pose_frame_mesh,
                    line_set=line_set,
                    result=result,
                )
                vis.update_geometry(ball_mesh)
                vis.update_geometry(center_frame_mesh)
                vis.update_geometry(pose_frame_mesh)
                vis.update_geometry(line_set)
                _log_result(result)

            now = time.perf_counter()
            if (
                points is not None
                and points.size > 0
                and (not busy_event.is_set())
                and job_queue.empty()
                and now - last_submit_ts >= config.compute_min_interval_s
            ):
                job = CaptureJob(
                    frame_idx=frame_idx,
                    color_bgr=color_bgr.copy(),
                    points=np.asarray(points, dtype=np.float32).copy(),
                    intrinsics=projection_intrinsics,
                )
                if _put_latest(job_queue, job):
                    submitted += 1
                    last_submit_ts = now
                else:
                    dropped += 1
            else:
                dropped += 1

            cv2.imshow(DEFAULT_2D_WINDOW_NAME, last_overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                logger.warning("收到退出指令，结束三色小球检测。")
                break
            if cv2.getWindowProperty(DEFAULT_2D_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                logger.warning("2D 预览窗口关闭，结束三色小球检测。")
                break
            if not _poll_viewer(vis):
                logger.warning("3D 预览窗口关闭，结束三色小球检测。")
                break

            if now - fps_t0 >= 3.0:
                fps = preview_frames / max(1e-6, now - fps_t0)
                logger.info(
                    f"性能状态：preview_fps {fps:.1f}，submitted {submitted}，"
                    f"completed {completed}，dropped {dropped}"
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
    result_queue: queue.Queue[DetectionResult],
    stop_event: threading.Event,
    busy_event: threading.Event,
    config: BallDetectorConfig,
) -> None:
    stabilizer = CenterStabilizer(
        max_jump_mm=config.max_center_jump_mm,
        alpha=config.smooth_alpha,
    )
    reference_pose: PoseEstimate | None = None
    reference_distances_mm: dict[tuple[str, str], float] | None = None
    while not stop_event.is_set():
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break
        busy_event.set()
        try:
            result, reference_pose, reference_distances_mm = _run_detection_job(
                job=job,
                config=config,
                stabilizer=stabilizer,
                reference_pose=reference_pose,
                reference_distances_mm=reference_distances_mm,
            )
            _put_latest(result_queue, result)
        except Exception as exc:
            logger.exception(f"帧 {job.frame_idx} 三色小球检测失败：{exc}")
        finally:
            busy_event.clear()
            job_queue.task_done()


def _run_detection_job(
    job: CaptureJob,
    config: BallDetectorConfig,
    stabilizer: CenterStabilizer,
    reference_pose: PoseEstimate | None,
    reference_distances_mm: dict[tuple[str, str], float] | None,
) -> tuple[DetectionResult, PoseEstimate | None, dict[tuple[str, str], float] | None]:
    t0 = time.perf_counter()
    timings_ms: dict[str, float] = {}

    step_t0 = time.perf_counter()
    masks = _build_color_masks(job.color_bgr, config)
    timings_ms["color_mask"] = (time.perf_counter() - step_t0) * 1000.0

    step_t0 = time.perf_counter()
    xyz = np.asarray(job.points[:, :3], dtype=np.float64)
    uv, valid_proj = _project_points_to_image(xyz, job.intrinsics)
    timings_ms["project_points"] = (time.perf_counter() - step_t0) * 1000.0

    step_t0 = time.perf_counter()
    detections = _detect_balls_from_masks(
        color_bgr=job.color_bgr,
        masks=masks,
        xyz=xyz,
        uv=uv,
        valid_proj=valid_proj,
        intrinsics=job.intrinsics,
        config=config,
        reference_distances_mm=reference_distances_mm,
    )
    detections = stabilizer.update(detections)
    timings_ms["detect_balls"] = (time.perf_counter() - step_t0) * 1000.0

    step_t0 = time.perf_counter()
    pose = _estimate_pose(detections, config, reference_pose)
    if reference_pose is None and pose is not None and config.use_first_valid_frame_as_reference:
        reference_pose = pose
        reference_distances_mm = _make_relative_distances(detections)
        logger.success("首个有效三球检测已作为测试绝对参考坐标系。")
    timings_ms["pose"] = (time.perf_counter() - step_t0) * 1000.0

    step_t0 = time.perf_counter()
    overlay = _draw_overlay(
        color_bgr=job.color_bgr,
        masks=masks,
        detections=detections,
        pose=pose,
        intrinsics=job.intrinsics,
        timings_ms=timings_ms,
    )
    timings_ms["draw_overlay"] = (time.perf_counter() - step_t0) * 1000.0
    timings_ms["total"] = (time.perf_counter() - t0) * 1000.0

    result = DetectionResult(
        frame_idx=job.frame_idx,
        detections=detections,
        pose=pose,
        reference_pose=reference_pose,
        reference_distances_mm=reference_distances_mm,
        overlay_bgr=overlay,
        preview_points=_downsample_points(job.points, DEFAULT_MAX_PREVIEW_POINTS),
        intrinsics=job.intrinsics,
        timings_ms=timings_ms,
    )
    return result, reference_pose, reference_distances_mm


# endregion


# region 检测算法
def _build_detector_config(
    ball_diameter_mm: float,
    use_first_valid_frame_as_reference: bool,
    require_known_model_pose: bool,
) -> BallDetectorConfig:
    model_points = {
        name: np.asarray(point, dtype=np.float64)
        for name, point in DEFAULT_MODEL_POINTS_MM.items()
    }
    color_ranges: dict[str, tuple[tuple[int, int, int, int, int, int], ...]] = {
        name: tuple(_normalize_hsv_range(bound) for bound in ranges)
        for name, ranges in DEFAULT_COLOR_RANGES.items()
    }
    return BallDetectorConfig(
        ball_diameter_mm=float(ball_diameter_mm),
        diameter_tolerance_ratio=float(DEFAULT_DIAMETER_TOLERANCE_RATIO),
        max_depth_mm=float(DEFAULT_MAX_DEPTH_MM),
        min_depth_points=int(DEFAULT_MIN_DEPTH_POINTS),
        max_color_components=int(DEFAULT_MAX_COLOR_COMPONENTS),
        min_component_area_px=int(DEFAULT_MIN_COMPONENT_AREA_PX),
        min_circularity=float(DEFAULT_MIN_CIRCULARITY),
        min_mask_fill_ratio=float(DEFAULT_MIN_MASK_FILL_RATIO),
        min_center_distance_ratio=float(DEFAULT_MIN_CENTER_DISTANCE_RATIO),
        relative_distance_tolerance_ratio=float(DEFAULT_RELATIVE_DISTANCE_TOLERANCE_RATIO),
        depth_trim_ratio=float(DEFAULT_DEPTH_TRIM_RATIO),
        max_center_jump_mm=float(DEFAULT_MAX_CENTER_JUMP_MM),
        smooth_alpha=float(DEFAULT_SMOOTH_ALPHA),
        compute_min_interval_s=float(DEFAULT_COMPUTE_MIN_INTERVAL_S),
        use_first_valid_frame_as_reference=bool(use_first_valid_frame_as_reference),
        require_known_model_pose=bool(require_known_model_pose),
        model_points_mm=model_points,
        color_ranges=color_ranges,
    )


def _build_color_masks(
    color_bgr: np.ndarray,
    config: BallDetectorConfig,
) -> dict[str, np.ndarray]:
    blurred = cv2.GaussianBlur(color_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    masks: dict[str, np.ndarray] = {}
    kernel = np.ones((5, 5), dtype=np.uint8)
    for color_name, ranges in config.color_ranges.items():
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for range_item in ranges:
            lower, upper = _hsv_range_to_bounds(range_item)
            partial = cv2.inRange(hsv, lower, upper)
            combined = cv2.bitwise_or(combined, partial)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        masks[color_name] = combined
    return masks


def _normalize_hsv_range(range_item: tuple[int, ...]) -> tuple[int, int, int, int, int, int]:
    if len(range_item) != 6:
        raise ValueError("HSV range must contain 6 integers")
    return (
        int(range_item[0]),
        int(range_item[1]),
        int(range_item[2]),
        int(range_item[3]),
        int(range_item[4]),
        int(range_item[5]),
    )


def _hsv_range_to_bounds(range_item: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if len(range_item) != 6:
        raise ValueError("HSV range must contain 6 integers")
    lower = np.asarray(range_item[:3], dtype=np.uint8)
    upper = np.asarray(range_item[3:], dtype=np.uint8)
    return lower, upper


def _detect_balls_from_masks(
    color_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    xyz: np.ndarray,
    uv: np.ndarray,
    valid_proj: np.ndarray,
    intrinsics: Any,
    config: BallDetectorConfig,
    reference_distances_mm: dict[tuple[str, str], float] | None,
) -> dict[str, BallDetection]:
    ranked_by_color: dict[str, list[BallDetection]] = {}
    for color_name in BALL_ORDER:
        candidates = _collect_color_candidates(
            color_name=color_name,
            mask=masks[color_name],
            config=config,
        )
        ranked_by_color[color_name] = _rank_ball_candidates(
            color_name=color_name,
            candidates=candidates,
            xyz=xyz,
            uv=uv,
            valid_proj=valid_proj,
            intrinsics=intrinsics,
            config=config,
            image_shape=color_bgr.shape[:2],
        )
    detections = _select_relative_consistent_detections(
        ranked_by_color=ranked_by_color,
        reference_distances_mm=reference_distances_mm,
        config=config,
    )
    return _resolve_duplicate_detections(detections, config)


def _collect_color_candidates(
    color_name: str,
    mask: np.ndarray,
    config: BallDetectorConfig,
) -> list[ColorCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[ColorCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_component_area_px:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < config.min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius <= 1.0:
            continue
        circle_area = float(np.pi * radius * radius)
        fill_ratio = float(area / max(1.0, circle_area))
        if fill_ratio < config.min_mask_fill_ratio:
            continue
        candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=cv2.FILLED)
        candidates.append(
            ColorCandidate(
                color_name=color_name,
                contour=contour,
                mask=candidate_mask,
                center_px=(float(cx), float(cy)),
                radius_px=float(radius),
                area_px=int(round(area)),
                circularity=circularity,
                fill_ratio=fill_ratio,
            )
        )
    candidates.sort(key=lambda item: item.area_px, reverse=True)
    return candidates[: config.max_color_components]


def _rank_ball_candidates(
    color_name: str,
    candidates: list[ColorCandidate],
    xyz: np.ndarray,
    uv: np.ndarray,
    valid_proj: np.ndarray,
    intrinsics: Any,
    config: BallDetectorConfig,
    image_shape: tuple[int, int],
) -> list[BallDetection]:
    if not candidates:
        return [_missing_detection(color_name, "no_color_component")]

    detections: list[BallDetection] = []
    for candidate in candidates:
        indices = _collect_indices_in_mask(uv=uv, valid_proj=valid_proj, mask=candidate.mask)
        ball_points = _trim_points_by_depth(xyz[indices], config.depth_trim_ratio)
        if ball_points.shape[0] < config.min_depth_points:
            continue

        center_mm = _robust_center(ball_points)
        physical_radius = _estimate_physical_radius_mm(
            center_mm=center_mm,
            radius_px=candidate.radius_px,
            intrinsics=intrinsics,
        )
        diameter_error = abs(physical_radius * 2.0 - config.ball_diameter_mm) / max(
            1e-6,
            config.ball_diameter_mm,
        )
        if diameter_error > config.diameter_tolerance_ratio:
            status = "diameter_out_of_range"
        else:
            status = "detected"

        expected_radius_px = _estimate_projected_radius_px(
            center_mm=center_mm,
            ball_radius_mm=config.ball_diameter_mm * 0.5,
            intrinsics=intrinsics,
        )
        radius_score = max(
            0.0,
            1.0 - abs(candidate.radius_px - expected_radius_px) / max(1.0, expected_radius_px),
        )
        depth_score = min(1.0, ball_points.shape[0] / max(1, config.min_depth_points * 3))
        circle_score = float(np.clip(candidate.circularity, 0.0, 1.0))
        fill_score = float(np.clip(candidate.fill_ratio, 0.0, 1.0))
        border_score = _score_inside_image(candidate.center_px, candidate.radius_px, image_shape)
        score = (
            0.32 * radius_score
            + 0.24 * depth_score
            + 0.20 * circle_score
            + 0.14 * fill_score
            + 0.10 * border_score
        )
        if status != "detected":
            score *= 0.72
        detections.append(
            BallDetection(
                color_name=color_name,
                detected=status == "detected",
                status=status,
                center_mm=center_mm,
                center_px=candidate.center_px,
                radius_px=candidate.radius_px,
                physical_radius_mm=physical_radius,
                depth_points=int(ball_points.shape[0]),
                score=float(score),
                contour=candidate.contour,
                mask=candidate.mask,
                failure_reasons=[] if status == "detected" else [status],
            )
        )
    if not detections:
        return [_missing_detection(color_name, "no_depth_candidate")]
    detections.sort(key=lambda item: item.score, reverse=True)
    return detections


def _select_relative_consistent_detections(
    ranked_by_color: dict[str, list[BallDetection]],
    reference_distances_mm: dict[tuple[str, str], float] | None,
    config: BallDetectorConfig,
) -> dict[str, BallDetection]:
    fallback = {color_name: ranked_by_color[color_name][0] for color_name in BALL_ORDER}
    if reference_distances_mm is None:
        return fallback

    candidate_groups = [
        [item for item in ranked_by_color[color_name] if item.detected and item.center_mm is not None]
        for color_name in BALL_ORDER
    ]
    if any(len(group) == 0 for group in candidate_groups):
        return fallback

    best_combo: tuple[BallDetection, ...] | None = None
    best_score = -1.0
    for combo in itertools.product(*candidate_groups):
        geometry_score, max_error_ratio = _score_relative_distances(combo, reference_distances_mm)
        detection_score = float(np.mean([item.score for item in combo]))
        score = 0.42 * detection_score + 0.58 * geometry_score
        if max_error_ratio > config.relative_distance_tolerance_ratio:
            score *= 0.35
        if score > best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        return fallback
    return {detection.color_name: detection for detection in best_combo}


def _score_relative_distances(
    detections: tuple[BallDetection, ...],
    reference_distances_mm: dict[tuple[str, str], float],
) -> tuple[float, float]:
    by_color = {detection.color_name: detection for detection in detections}
    scores: list[float] = []
    max_error_ratio = 0.0
    for pair, reference_distance in reference_distances_mm.items():
        left = by_color[pair[0]]
        right = by_color[pair[1]]
        if left.center_mm is None or right.center_mm is None:
            return 0.0, 1.0
        observed_distance = float(np.linalg.norm(left.center_mm - right.center_mm))
        error_ratio = abs(observed_distance - reference_distance) / max(1.0, reference_distance)
        max_error_ratio = max(max_error_ratio, error_ratio)
        scores.append(max(0.0, 1.0 - error_ratio))
    return float(np.mean(scores)), max_error_ratio


def _make_relative_distances(
    detections: dict[str, BallDetection],
) -> dict[tuple[str, str], float] | None:
    centers = _collect_detected_centers(detections)
    if centers is None:
        return None
    distances: dict[tuple[str, str], float] = {}
    for left_index, left_name in enumerate(BALL_ORDER):
        for right_name in BALL_ORDER[left_index + 1 :]:
            distances[(left_name, right_name)] = float(
                np.linalg.norm(centers[left_name] - centers[right_name])
            )
    return distances


def _trim_points_by_depth(points: np.ndarray, trim_ratio: float) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-6)
    valid_points = points[valid]
    if valid_points.shape[0] < 4:
        return valid_points
    z = valid_points[:, 2]
    low, high = np.quantile(z, [float(trim_ratio), 1.0 - float(trim_ratio)])
    return valid_points[(z >= low) & (z <= high)]


def _robust_center(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((3,), dtype=np.float64)
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center.reshape(1, 3), axis=1)
    keep_limit = np.quantile(distances, 0.78)
    kept = points[distances <= keep_limit]
    if kept.shape[0] == 0:
        return center.astype(np.float64)
    return np.mean(kept, axis=0).astype(np.float64)


def _estimate_physical_radius_mm(
    center_mm: np.ndarray,
    radius_px: float,
    intrinsics: Any,
) -> float:
    focal = 0.5 * (float(intrinsics.fx) + float(intrinsics.fy))
    return float(radius_px) * float(center_mm[2]) / max(1e-6, focal)


def _estimate_projected_radius_px(
    center_mm: np.ndarray,
    ball_radius_mm: float,
    intrinsics: Any,
) -> float:
    focal = 0.5 * (float(intrinsics.fx) + float(intrinsics.fy))
    return float(ball_radius_mm) * focal / max(1e-6, float(center_mm[2]))


def _score_inside_image(
    center_px: tuple[float, float],
    radius_px: float,
    image_shape: tuple[int, int],
) -> float:
    h, w = image_shape
    cx, cy = center_px
    margin = float(radius_px) * 0.8
    if cx < margin or cy < margin or cx > w - margin or cy > h - margin:
        return 0.35
    return 1.0


def _resolve_duplicate_detections(
    detections: dict[str, BallDetection],
    config: BallDetectorConfig,
) -> dict[str, BallDetection]:
    resolved = dict(detections)
    for left_index, left_name in enumerate(BALL_ORDER):
        left = resolved[left_name]
        if not left.detected:
            continue
        for right_name in BALL_ORDER[left_index + 1 :]:
            right = resolved[right_name]
            if not right.detected:
                continue
            if not _is_same_physical_ball(left, right, config):
                continue
            if left.score >= right.score:
                resolved[right_name] = _duplicate_detection(right_name, "duplicate_physical_ball")
            else:
                resolved[left_name] = _duplicate_detection(left_name, "duplicate_physical_ball")
                break
    return resolved


def _is_same_physical_ball(
    left: BallDetection,
    right: BallDetection,
    config: BallDetectorConfig,
) -> bool:
    if left.center_px is None or right.center_px is None:
        return False
    left_px = np.asarray(left.center_px, dtype=np.float64)
    right_px = np.asarray(right.center_px, dtype=np.float64)
    radius_px = max(2.0, min(float(left.radius_px), float(right.radius_px)))
    if float(np.linalg.norm(left_px - right_px)) <= radius_px * float(config.min_center_distance_ratio):
        return True
    if left.center_mm is None or right.center_mm is None:
        return False
    distance_mm = float(
        np.linalg.norm(
            np.asarray(left.center_mm, dtype=np.float64)
            - np.asarray(right.center_mm, dtype=np.float64)
        )
    )
    return distance_mm <= float(config.ball_diameter_mm) * float(config.min_center_distance_ratio)


def _duplicate_detection(color_name: str, reason: str) -> BallDetection:
    result = _missing_detection(color_name, reason)
    return replace(result, status="duplicate")


def _missing_detection(color_name: str, reason: str) -> BallDetection:
    return BallDetection(
        color_name=color_name,
        detected=False,
        status="missing",
        center_mm=None,
        center_px=None,
        radius_px=0.0,
        physical_radius_mm=0.0,
        depth_points=0,
        score=0.0,
        contour=None,
        mask=None,
        failure_reasons=[reason],
    )


# endregion


# region 位姿估计
def _estimate_pose(
    detections: dict[str, BallDetection],
    config: BallDetectorConfig,
    reference_pose: PoseEstimate | None,
) -> PoseEstimate | None:
    centers = _collect_detected_centers(detections)
    if centers is None:
        return None

    if config.require_known_model_pose:
        return _estimate_pose_from_model(centers=centers, config=config)

    if reference_pose is not None or config.use_first_valid_frame_as_reference:
        return _pose_from_current_ball_centers(centers, source="current_centers")
    return _estimate_pose_from_model(centers=centers, config=config)


def _collect_detected_centers(
    detections: dict[str, BallDetection],
) -> dict[str, np.ndarray] | None:
    centers: dict[str, np.ndarray] = {}
    for color_name in BALL_ORDER:
        detection = detections[color_name]
        if not detection.detected or detection.center_mm is None:
            return None
        centers[color_name] = np.asarray(detection.center_mm, dtype=np.float64)
    return centers


def _pose_from_current_ball_centers(
    centers: dict[str, np.ndarray],
    source: str,
) -> PoseEstimate | None:
    red = centers["red"]
    blue = centers["blue"]
    yellow = centers["yellow"]
    x_axis = _normalize(blue - red)
    yellow_vec = yellow - red
    z_axis = _normalize(np.cross(x_axis, yellow_vec))
    if np.linalg.norm(x_axis) < 1e-8 or np.linalg.norm(z_axis) < 1e-8:
        return None
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    transform = _make_transform(origin=red, rotation=rotation)
    plane_residual = abs(float(np.dot(yellow_vec, z_axis)))
    return PoseEstimate(
        origin_mm=red,
        rotation=rotation,
        transform=transform,
        residual_mm=plane_residual,
        source=source,
    )


def _estimate_pose_from_model(
    centers: dict[str, np.ndarray],
    config: BallDetectorConfig,
) -> PoseEstimate | None:
    model = np.stack([config.model_points_mm[name] for name in BALL_ORDER], axis=0)
    observed = np.stack([centers[name] for name in BALL_ORDER], axis=0)
    if _triangle_area(model) < 1e-6 or _triangle_area(observed) < 1e-6:
        return None
    model_centroid = np.mean(model, axis=0)
    observed_centroid = np.mean(observed, axis=0)
    model_centered = model - model_centroid
    observed_centered = observed - observed_centroid
    covariance = model_centered.T @ observed_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0.0:
        correction[2, 2] = -1.0
    rotation = vt.T @ correction @ u.T
    origin = observed_centroid - rotation @ model_centroid
    transformed = (rotation @ model.T).T + origin.reshape(1, 3)
    residual = float(np.mean(np.linalg.norm(transformed - observed, axis=1)))
    return PoseEstimate(
        origin_mm=origin,
        rotation=rotation,
        transform=_make_transform(origin=origin, rotation=rotation),
        residual_mm=residual,
        source="known_model_points",
    )


def _triangle_area(points: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros((3,), dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _make_transform(origin: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(origin, dtype=np.float64).reshape(3)
    return transform


# endregion


# region Orbbec 与点云工具
def _capture_rgbd_points_once(
    session: OrbbecSession,
    point_filter: Any,
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
        use_color=bool(use_color),
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


def _decode_color_frame_bgr(color_frame: Any) -> np.ndarray | None:
    if color_frame is None:
        return None
    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())
    if width <= 0 or height <= 0 or data.size == 0:
        return None

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

    logger.warning(f"当前 color format 暂未支持直接预览：{color_format}")
    return None


def _project_points_to_image(
    xyz: np.ndarray,
    intrinsics: Any,
) -> tuple[np.ndarray, np.ndarray]:
    z = xyz[:, 2]
    valid_depth = z > 1e-6
    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)
    if np.any(valid_depth):
        x = xyz[valid_depth, 0]
        y = xyz[valid_depth, 1]
        zz = z[valid_depth]
        uu = np.rint(float(intrinsics.fx) * x / zz + float(intrinsics.cx)).astype(np.int32)
        vv = np.rint(float(intrinsics.fy) * y / zz + float(intrinsics.cy)).astype(np.int32)
        in_bounds = (
            (uu >= 0)
            & (uu < int(intrinsics.width))
            & (vv >= 0)
            & (vv < int(intrinsics.height))
        )
        original_idx = np.where(valid_depth)[0][in_bounds]
        u[original_idx] = uu[in_bounds]
        v[original_idx] = vv[in_bounds]
    return np.stack([u, v], axis=1), (u >= 0) & (v >= 0)


def _collect_indices_in_mask(
    uv: np.ndarray,
    valid_proj: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    idx = np.where(valid_proj)[0]
    if idx.size == 0:
        return np.empty((0,), dtype=np.int32)
    u = uv[idx, 0]
    v = uv[idx, 1]
    inside = mask[v, u] > 0
    return idx[inside].astype(np.int32)


def _extract_rgb(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 6:
        rgb = np.asarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0).astype(np.float64)
    return np.full((points.shape[0], 3), 0.72, dtype=np.float64)


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= int(max_points):
        return points
    step = max(1, int(np.ceil(points.shape[0] / float(max_points))))
    return points[::step]


# endregion


# region 2D 可视化
def _draw_overlay(
    color_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    detections: dict[str, BallDetection],
    pose: PoseEstimate | None,
    intrinsics: Any,
    timings_ms: dict[str, float],
) -> np.ndarray:
    canvas = color_bgr.copy()
    _draw_detected_links(canvas, detections)
    for color_name in BALL_ORDER:
        _draw_detection(canvas, detections[color_name], intrinsics)
    _draw_status_text(canvas, detections, pose, timings_ms)
    mask_panel = _draw_mask_diagnostic_panel(color_bgr.shape, masks)
    return np.hstack([canvas, mask_panel])


def _draw_mask_diagnostic_panel(
    image_shape: tuple[int, ...],
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    h, w = image_shape[:2]
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[:] = (18, 18, 18)
    for color_name, mask in masks.items():
        color = np.asarray(BALL_DRAW_BGR[color_name], dtype=np.uint8)
        active = mask > 0
        if np.any(active):
            panel[active] = color
        _draw_mask_panel_stats(panel, color_name, mask)
    cv2.putText(
        panel,
        "Mask diagnostic",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _draw_mask_panel_stats(
    panel: np.ndarray,
    color_name: str,
    mask: np.ndarray,
) -> None:
    order_index = BALL_ORDER.index(color_name)
    y = 62 + order_index * 28
    pixels = int(np.count_nonzero(mask))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0.0
    if contours:
        largest_area = float(max(cv2.contourArea(contour) for contour in contours))
    text = f"{color_name}: px={pixels} max_area={largest_area:.0f}"
    cv2.putText(
        panel,
        text,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        BALL_DRAW_BGR[color_name],
        1,
        cv2.LINE_AA,
    )

def _draw_detected_links(
    canvas: np.ndarray,
    detections: dict[str, BallDetection],
) -> None:
    points: list[tuple[int, int]] = []
    for color_name in BALL_ORDER:
        center_px = detections[color_name].center_px
        if center_px is None:
            return
        points.append((int(round(center_px[0])), int(round(center_px[1]))))
    cv2.line(canvas, points[0], points[1], BALL_DRAW_BGR["blue"], 1, cv2.LINE_AA)
    cv2.line(canvas, points[1], points[2], BALL_DRAW_BGR["yellow"], 1, cv2.LINE_AA)


def _draw_detection(canvas: np.ndarray, detection: BallDetection, intrinsics: Any) -> None:
    color = BALL_DRAW_BGR[detection.color_name]
    if detection.contour is not None:
        cv2.drawContours(canvas, [detection.contour], -1, color, 1, cv2.LINE_AA)
    if detection.center_px is not None:
        center = (int(round(detection.center_px[0])), int(round(detection.center_px[1])))
        radius = max(1, int(round(_project_detection_radius_px(detection, intrinsics))))
        cv2.circle(canvas, center, radius, color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            center,
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        text = (
            f"{detection.color_name} {detection.status} "
            f"r={detection.physical_radius_mm:.1f}mm pts={detection.depth_points}"
        )
        cv2.putText(
            canvas,
            text,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )


def _project_detection_radius_px(detection: BallDetection, intrinsics: Any) -> float:
    if detection.center_mm is None or detection.physical_radius_mm <= 0.0:
        return float(detection.radius_px)
    z_mm = float(np.asarray(detection.center_mm, dtype=np.float64)[2])
    if z_mm <= 1e-6:
        return float(detection.radius_px)
    focal_px = 0.5 * (float(intrinsics.fx) + float(intrinsics.fy))
    projected_radius = float(detection.physical_radius_mm) * focal_px / z_mm
    if not np.isfinite(projected_radius) or projected_radius <= 0.0:
        return float(detection.radius_px)
    return projected_radius


def _draw_status_text(
    canvas: np.ndarray,
    detections: dict[str, BallDetection],
    pose: PoseEstimate | None,
    timings_ms: dict[str, float],
) -> None:
    detected_count = sum(1 for item in detections.values() if item.detected)
    lines = [
        f"detected {detected_count}/3 total={timings_ms.get('total', 0.0):.1f}ms",
    ]
    if pose is not None:
        origin = pose.origin_mm
        lines.append(
            f"pose {pose.source} XYZ={origin[0]:.1f},{origin[1]:.1f},{origin[2]:.1f}mm "
            f"res={pose.residual_mm:.2f}mm"
        )
    else:
        lines.append("pose unavailable")
    for idx, text in enumerate(lines):
        y = 30 + idx * 28
        cv2.putText(canvas, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)


def _compute_preview_window_size(
    src_w: int,
    src_h: int,
    min_long_side: int,
) -> tuple[int, int]:
    long_side = max(1, int(src_w), int(src_h))
    if long_side >= int(min_long_side):
        return max(1, int(src_w)), max(1, int(src_h))
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


# endregion


# region Open3D 可视化
def _init_3d_viewer() -> tuple[
    o3d.visualization.VisualizerWithKeyCallback,
    dict[str, bool],
    o3d.geometry.PointCloud,
    o3d.geometry.TriangleMesh,
    o3d.geometry.TriangleMesh,
    o3d.geometry.TriangleMesh,
    o3d.geometry.LineSet,
]:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(DEFAULT_3D_WINDOW_NAME, DEFAULT_3D_WINDOW_WIDTH, DEFAULT_3D_WINDOW_HEIGHT)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = DEFAULT_POINT_SIZE
        opt.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)

    stop = {"flag": False}

    def _on_escape(_vis: o3d.visualization.Visualizer) -> bool:
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)
    camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=120.0, origin=[0.0, 0.0, 0.0])
    raw_pcd = o3d.geometry.PointCloud()
    ball_mesh = _empty_mesh()
    center_frame_mesh = _empty_mesh()
    pose_frame_mesh = _empty_mesh()
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.zeros((2, 3), dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64))

    for geometry in (camera_axis, raw_pcd, ball_mesh, center_frame_mesh, pose_frame_mesh, line_set):
        vis.add_geometry(geometry)

    view = vis.get_view_control()
    if view is not None:
        view.set_lookat([0.0, 0.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
    return vis, stop, raw_pcd, ball_mesh, center_frame_mesh, pose_frame_mesh, line_set


def _update_raw_cloud(pcd: o3d.geometry.PointCloud, points: np.ndarray) -> None:
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = _extract_rgb(points)
    dimmed = np.clip(DEFAULT_RAW_POINT_DIM_FACTOR * rgb + 0.08, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(dimmed)


def _update_detection_3d(
    ball_mesh: o3d.geometry.TriangleMesh,
    center_frame_mesh: o3d.geometry.TriangleMesh,
    pose_frame_mesh: o3d.geometry.TriangleMesh,
    line_set: o3d.geometry.LineSet,
    result: DetectionResult,
) -> None:
    centers = _collect_visual_centers(result.detections)
    _replace_mesh(ball_mesh, _build_ball_marker_mesh(result.detections))
    _replace_mesh(center_frame_mesh, _build_center_frame_mesh(centers, result.pose))
    _replace_mesh(pose_frame_mesh, _build_pose_frame_mesh(result.pose, result.reference_pose))
    _update_line_set(line_set, centers)


def _collect_visual_centers(detections: dict[str, BallDetection]) -> dict[str, np.ndarray]:
    centers: dict[str, np.ndarray] = {}
    for color_name, detection in detections.items():
        if detection.center_mm is not None:
            centers[color_name] = np.asarray(detection.center_mm, dtype=np.float64)
    return centers


def _build_ball_marker_mesh(detections: dict[str, BallDetection]) -> o3d.geometry.TriangleMesh:
    merged = o3d.geometry.TriangleMesh()
    for color_name in BALL_ORDER:
        detection = detections[color_name]
        if detection.center_mm is None:
            continue
        radius = max(2.0, min(12.0, detection.physical_radius_mm or DEFAULT_BALL_DIAMETER_MM * 0.5))
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius), resolution=16)
        mesh.translate(np.asarray(detection.center_mm, dtype=np.float64))
        mesh.paint_uniform_color(BALL_DRAW_RGB_FLOAT[color_name])
        mesh.compute_vertex_normals()
        merged += mesh
    return merged if len(merged.vertices) > 0 else _empty_mesh()


def _build_center_frame_mesh(
    centers: dict[str, np.ndarray],
    pose: PoseEstimate | None,
) -> o3d.geometry.TriangleMesh:
    merged = o3d.geometry.TriangleMesh()
    rotation = np.eye(3, dtype=np.float64) if pose is None else pose.rotation
    for center in centers.values():
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0, origin=[0.0, 0.0, 0.0])
        frame.transform(_make_transform(origin=center, rotation=rotation))
        merged += frame
    return merged if len(merged.vertices) > 0 else _empty_mesh()


def _build_pose_frame_mesh(
    pose: PoseEstimate | None,
    reference_pose: PoseEstimate | None,
) -> o3d.geometry.TriangleMesh:
    merged = o3d.geometry.TriangleMesh()
    if reference_pose is not None:
        reference_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=70.0, origin=[0.0, 0.0, 0.0])
        reference_frame.transform(reference_pose.transform)
        merged += reference_frame
    if pose is not None:
        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=105.0, origin=[0.0, 0.0, 0.0])
        pose_frame.transform(pose.transform)
        merged += pose_frame
    return merged if len(merged.vertices) > 0 else _empty_mesh()


def _update_line_set(line_set: o3d.geometry.LineSet, centers: dict[str, np.ndarray]) -> None:
    if not all(name in centers for name in BALL_ORDER):
        line_set.points = o3d.utility.Vector3dVector(np.zeros((2, 3), dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64))
        return
    points = np.stack([centers[name] for name in BALL_ORDER], axis=0)
    lines = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
    colors = np.asarray(
        [
            BALL_DRAW_RGB_FLOAT["blue"],
            BALL_DRAW_RGB_FLOAT["yellow"],
        ],
        dtype=np.float64,
    )
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)


def _replace_mesh(target: o3d.geometry.TriangleMesh, source: o3d.geometry.TriangleMesh) -> None:
    target.vertices = source.vertices
    target.triangles = source.triangles
    target.vertex_colors = source.vertex_colors
    target.vertex_normals = source.vertex_normals


def _empty_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0], [0.0, 1.0, -10000.0]], dtype=np.float64)
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32))
    return mesh


def _poll_viewer(vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    alive = vis.poll_events()
    vis.update_renderer()
    return bool(alive)


# endregion


# region 日志与 CLI
def _log_result(result: DetectionResult) -> None:
    if result.pose is None:
        missing = [name for name, detection in result.detections.items() if not detection.detected]
        logger.info(
            f"帧 {result.frame_idx}：pose unavailable，missing {missing}，"
            f"total {result.timings_ms.get('total', 0.0):.1f} ms"
        )
        return
    origin = result.pose.origin_mm
    logger.info(
        f"帧 {result.frame_idx}：pose {result.pose.source}，"
        f"XYZ {origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f} mm，"
        f"residual {result.pose.residual_mm:.3f} mm，"
        f"total {result.timings_ms.get('total', 0.0):.1f} ms"
    )


def _put_latest(target_queue: queue.Queue, item: object) -> bool:
    while True:
        try:
            target_queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                target_queue.get_nowait()
                target_queue.task_done()
            except queue.Empty:
                return False


def _parse_cli() -> tuple[int, int, float, bool, bool]:
    parser = argparse.ArgumentParser(description="Orbbec RGBD 三色小球位姿检测实验脚本")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames 超时时间，单位 ms")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="请求采集帧率，单位 fps")
    parser.add_argument("--ball-diameter-mm", type=float, default=DEFAULT_BALL_DIAMETER_MM, help="小球直径，单位 mm")
    parser.add_argument(
        "--first-frame-reference",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_FIRST_VALID_FRAME_AS_REFERENCE,
        help="是否用首个有效三球检测建立测试绝对参考",
    )
    parser.add_argument(
        "--known-model-pose",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_REQUIRE_KNOWN_MODEL_POSE,
        help="是否强制使用 DEFAULT_MODEL_POINTS_MM 估计模型到相机的刚体位姿",
    )
    args = parser.parse_args()
    return (
        int(args.timeout_ms),
        int(args.capture_fps),
        float(args.ball_diameter_mm),
        bool(args.first_frame_reference),
        bool(args.known_model_pose),
    )


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            timeout_arg, fps_arg, diameter_arg, first_ref_arg, known_model_arg = _parse_cli()
            main(
                timeout_ms=timeout_arg,
                capture_fps=fps_arg,
                ball_diameter_mm=diameter_arg,
                use_first_valid_frame_as_reference=first_ref_arg,
                require_known_model_pose=known_model_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
# endregion
