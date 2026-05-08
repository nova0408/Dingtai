from __future__ import annotations

import csv
import concurrent.futures
import faulthandler
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from loguru import logger
from pyorbbecsdk import OBFormat

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pointcloud.grasp_pose.opening_pipeline import OpeningDetectionPipeline
from src.pointcloud.grasp_pose.pose_pipeline import GraspPoseEstimator, TemporalFilterState
from src.pointcloud.grasp_pose.types import GraspResult, OpeningDetection
from src.pointcloud.tray_detection import TrayDetectionPipeline, TrayRuntimeState
from src.rgbd_camera import Gemini305, SessionOptions, set_point_cloud_filter_format

# region 默认参数
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_MAX_DEPTH_MM = 5000.0
DEFAULT_MAX_PREVIEW_POINTS = 100_000
DEFAULT_COMPUTE_MIN_INTERVAL_S = 0.10
DEFAULT_OPENING_MIN_POINTS = 80
DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 900
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800
DEFAULT_POINT_SIZE = 1.5
DEFAULT_2D_WINDOW_NAME = "Orbbec realtime estimate grasp"
DEFAULT_2D_MERGED_WINDOW_NAME = "Orbbec realtime estimate grasp merged"
DEFAULT_3D_WINDOW_NAME = "Orbbec realtime estimate grasp 3D"
DEFAULT_TRAY_USE_SAM = False
DEFAULT_TRAY_DETECT_EVERY_N = 6
DEFAULT_TRAY_DETECT_MAX_SIDE = 384
DEFAULT_TRAY_COMBINE_PROMPTS_FORWARD = True
DEFAULT_MASK_SYNC_BUDGET_MS = 60.0
DEFAULT_TRAY_MOTION_DOWNSAMPLE = 0.25
DEFAULT_TRAY_MOTION_SMOOTH_ALPHA = 0.60
DEFAULT_TRAY_MOTION_MAX_SHIFT_PX = 36.0
DEFAULT_CONTRAST_SIGMA = 2.6
DEFAULT_CONTRAST_HP_A = 1.90
DEFAULT_CONTRAST_HP_B = -0.90
DEFAULT_CONTRAST_BILATERAL_D = 7
DEFAULT_CONTRAST_BILATERAL_SIGMA_COLOR = 42
DEFAULT_CONTRAST_BILATERAL_SIGMA_SPACE = 42
DEFAULT_CONTRAST_CANNY_LOW = 42
DEFAULT_CONTRAST_CANNY_HIGH = 118
DEFAULT_NEAR_GROW_DIFF = 18
DEFAULT_NEAR_GROW_MAX_PIXELS = 26000
DEFAULT_NEAR_GROW_LOCAL_DIFF = 14
DEFAULT_NEAR_GROW_GLOBAL_DIFF = 30
DEFAULT_TOP_NORMAL_EMA_ALPHA = 0.35
DEFAULT_TOP_NORMAL_MAX_ANGLE_DEG = 12.0
DEFAULT_GRASP_POINT_EMA_ALPHA = 0.30
DEFAULT_GRASP_AXIS_EMA_ALPHA = 0.28
DEFAULT_GRASP_MAX_TRANSLATION_MM = 40.0
DEFAULT_GRASP_MAX_AXIS_ANGLE_DEG = 15.0
DEFAULT_GRASP_TRANSLATION_SOFT_MM = 1.5
DEFAULT_GRASP_AXIS_SOFT_DEG = 1.2
DEFAULT_GRASP_MEDIAN_WINDOW = 5
DEFAULT_FAULT_LOG_PATH = PROJECT_ROOT / "logs" / "grasp_pose_pipeline_fault.log"
DEFAULT_TIMING_CSV_PATH = PROJECT_ROOT / "logs" / "grasp_pose_pipeline_timing.csv"
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
class PipelineResult:
    frame_idx: int
    xyz: np.ndarray
    rgb: np.ndarray
    tray_mask: np.ndarray | None
    tray_detect_ok: bool
    near_plane_mask: np.ndarray | None
    no_hole_mask: np.ndarray | None
    top_plane_quad_uv: np.ndarray | None
    opening: OpeningDetection | None
    grasp: GraspResult | None
    overlay_bgr: np.ndarray
    contrast_bgr: np.ndarray
    elapsed_ms: float
    error: str | None
    failed_step: str | None
    completed_steps: tuple[str, ...]


@dataclass(frozen=True)
class StepTiming:
    step_name: str
    start_ts: str
    end_ts: str
    elapsed_ms: float
    status: str
    error: str


@dataclass
class TrayDetectState:
    runtime: TrayRuntimeState = field(default_factory=TrayRuntimeState)


# endregion


# region 主流程
def _enable_fault_logging(log_path: Path) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "a", encoding="utf-8")
        faulthandler.enable(file=f, all_threads=True)
        logger.info(f"faulthandler 已启用，故障日志输出到：{log_path}")
    except Exception as exc:
        logger.warning(f"faulthandler 启用失败：{exc}")

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logger.exception(
            f"线程未捕获异常 thread={args.thread.name if args.thread is not None else 'unknown'}: {args.exc_value}"
        )

    threading.excepthook = _thread_excepthook


def main(
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
    compute_min_interval_s: float = DEFAULT_COMPUTE_MIN_INTERVAL_S,
) -> None:
    _enable_fault_logging(DEFAULT_FAULT_LOG_PATH)
    options = SessionOptions(timeout=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))
    with Gemini305(options=options) as session:
        cam = session.get_camera_param()
        color_intr = cam.rgb_intrinsic if session.has_color_sensor else cam.depth_intrinsic
        logger.info(
            f"内参：fx={color_intr.fx:.3f}, fy={color_intr.fy:.3f}, cx={color_intr.cx:.3f}, cy={color_intr.cy:.3f}, "
            f"w={color_intr.width}, h={color_intr.height}"
        )
        point_filter = session.create_point_cloud_filter(camera_param=cam)
        _run_pipeline(
            session=session,
            point_filter=point_filter,
            fx=float(color_intr.fx),
            fy=float(color_intr.fy),
            cx=float(color_intr.cx),
            cy=float(color_intr.cy),
            img_w=int(max(32, color_intr.width)),
            img_h=int(max(32, color_intr.height)),
            max_depth_mm=float(max_depth_mm),
            compute_min_interval_s=max(0.0, float(compute_min_interval_s)),
        )


def _run_pipeline(
    session: Gemini305,
    point_filter,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    max_depth_mm: float,
    compute_min_interval_s: float,
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_writer = _CsvTimingWriter(DEFAULT_TIMING_CSV_PATH, run_id)
    logger.info(f"步骤耗时 CSV 输出：{DEFAULT_TIMING_CSV_PATH}")
    job_queue: queue.Queue[CaptureJob | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[PipelineResult] = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    busy_event = threading.Event()
    worker = threading.Thread(
        target=_worker_loop,
        args=(job_queue, result_queue, stop_event, busy_event, timing_writer),
        name="estimate_grasp_worker",
        daemon=False,
    )
    worker.start()

    vis, stop_flag, raw_pcd, frame_mesh, grasp_line = _init_3d_viewer()
    cv2.namedWindow(DEFAULT_2D_MERGED_WINDOW_NAME, cv2.WINDOW_NORMAL)
    win_w, win_h = _compute_preview_window_size(img_w, img_h, DEFAULT_MIN_2D_WINDOW_LONG_SIDE)
    cv2.resizeWindow(DEFAULT_2D_MERGED_WINDOW_NAME, win_w * 2, win_h)

    frame_idx = 0
    submitted = 0
    completed = 0
    dropped = 0
    last_submit_ts = 0.0
    fps_t0 = time.perf_counter()
    preview_frames = 0
    last_overlay = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    last_contrast = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    latest_result_for_preview: PipelineResult | None = None

    try:
        while True:
            if stop_flag["flag"]:
                break
            points, color_bgr = _capture_points_once(session, point_filter, max_depth_mm)
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
                _update_result_3d(frame_mesh, grasp_line, result)
                vis.update_geometry(frame_mesh)
                vis.update_geometry(grasp_line)
                latest_result_for_preview = result
                if result.error is None and result.grasp is not None:
                    p = result.grasp.grasp_point
                    rpy = _rotation_matrix_to_rpy_deg(result.grasp.rotation)
                    logger.info(
                        f"帧 {result.frame_idx} grasp XYZ {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f} mm; "
                        f"RPY {rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f} deg; elapsed {result.elapsed_ms:.1f} ms"
                    )
                elif result.error is not None:
                    kind = _classify_failure_kind(result.failed_step)
                    logger.warning(
                        f"帧 {result.frame_idx} 计算失败[{kind}] step={result.failed_step} err={result.error}; "
                        f"completed={'>'.join(result.completed_steps) if len(result.completed_steps) > 0 else 'none'}"
                    )

            now = time.perf_counter()
            if (not busy_event.is_set()) and job_queue.empty() and (now - last_submit_ts >= compute_min_interval_s):
                try:
                    job_queue.put_nowait(
                        CaptureJob(
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
                    )
                    submitted += 1
                    last_submit_ts = now
                except queue.Full:
                    dropped += 1
            else:
                dropped += 1

            # 2D 预览以“当前实时帧”为底图，再叠加最近一次计算结果，避免显示历史帧画面。
            if color_bgr is not None:
                base_live = cv2.resize(color_bgr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            else:
                xyz_live = np.asarray(points[:, :3], dtype=np.float64)
                rgb_live = _extract_rgb(points)
                base_live = _rasterize_rgb(xyz_live, rgb_live, fx, fy, cx, cy, img_w, img_h)
            if latest_result_for_preview is not None:
                result = latest_result_for_preview
                try:
                    last_overlay = _safe_draw_overlay_partial(
                        base_live,
                        result.tray_mask,
                        result.tray_detect_ok,
                        result.near_plane_mask,
                        result.no_hole_mask,
                        result.top_plane_quad_uv,
                        result.opening,
                        result.grasp,
                        result.error,
                        result.failed_step,
                    )
                    _, live_hp_gray, live_hp_edge = _compute_high_contrast_domain(base_live)
                    last_contrast = _safe_build_contrast_preview_partial(
                        live_hp_gray,
                        live_hp_edge,
                        result.near_plane_mask,
                        result.no_hole_mask,
                        result.opening,
                    )
                except cv2.error as exc:
                    logger.warning(f"帧 {frame_idx} 预览绘制失败（已忽略，继续流）：{exc}")
                    last_overlay = _draw_status_only(base_live, f"preview draw cv2 err: {exc}")
                    _, live_hp_gray, live_hp_edge = _compute_high_contrast_domain(base_live)
                    last_contrast = _safe_build_contrast_preview_partial(live_hp_gray, live_hp_edge, None, None, None)
            else:
                last_overlay = base_live
                _, live_hp_gray, live_hp_edge = _compute_high_contrast_domain(base_live)
                last_contrast = _safe_build_contrast_preview_partial(live_hp_gray, live_hp_edge, None, None, None)

            merged = np.hstack([last_overlay, last_contrast])
            cv2.imshow(DEFAULT_2D_MERGED_WINDOW_NAME, merged)
            if cv2.waitKey(1) == 27:
                break
            if (not _poll_viewer(vis)) or cv2.getWindowProperty(
                DEFAULT_2D_MERGED_WINDOW_NAME, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

            if now - fps_t0 >= 3.0:
                preview_fps = preview_frames / max(1e-6, now - fps_t0)
                logger.info(
                    f"性能状态：preview_fps {preview_fps:.1f}, submitted {submitted}, completed {completed}, dropped {dropped}"
                )
                fps_t0 = now
                preview_frames = 0
    except Exception as exc:
        logger.exception(f"主循环异常退出：{exc}")
        raise
    finally:
        stop_event.set()
        # 等待后台任务完成并确保最后一帧计时写入 CSV 后再退出。
        job_queue.put(None)
        worker.join()
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break
            completed += 1
            _update_result_3d(frame_mesh, grasp_line, result)
        vis.destroy_window()
        _safe_destroy_cv_window(DEFAULT_2D_MERGED_WINDOW_NAME)


# endregion


# region 后台计算


def _worker_loop(
    job_queue: queue.Queue[CaptureJob | None],
    result_queue: queue.Queue[PipelineResult],
    stop_event: threading.Event,
    busy_event: threading.Event,
    timing_writer: "_CsvTimingWriter",
) -> None:
    tray_pipeline = TrayDetectionPipeline(TrayDetectionPipeline.build_default_detector())
    opening_pipeline = OpeningDetectionPipeline()
    pose_estimator = GraspPoseEstimator()
    tray_state = TrayDetectState()
    temporal_state = TemporalFilterState()
    mask_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mask_pipeline")
    while not stop_event.is_set():
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break
        busy_event.set()
        try:
            result = _run_compute_job(
                job=job,
                tray_pipeline=tray_pipeline,
                opening_pipeline=opening_pipeline,
                pose_estimator=pose_estimator,
                tray_state=tray_state,
                temporal_state=temporal_state,
                timing_writer=timing_writer,
                mask_executor=mask_executor,
            )
            _put_latest_result(result_queue, result)
        except Exception as exc:
            logger.exception(f"帧 {job.frame_idx} 计算线程异常：{exc}")
        finally:
            busy_event.clear()
            job_queue.task_done()
    mask_executor.shutdown(wait=True)


def _run_compute_job(
    job: CaptureJob,
    tray_pipeline: TrayDetectionPipeline,
    opening_pipeline: OpeningDetectionPipeline,
    pose_estimator: GraspPoseEstimator,
    tray_state: TrayDetectState,
    temporal_state: TemporalFilterState,
    timing_writer: "_CsvTimingWriter",
    mask_executor: concurrent.futures.ThreadPoolExecutor,
) -> PipelineResult:
    t0 = time.perf_counter()
    timings: list[StepTiming] = []
    xyz = np.asarray(job.points[:, :3], dtype=np.float64)
    rgb = _extract_rgb(job.points)
    base_bgr = (
        cv2.resize(job.color_bgr, (job.img_w, job.img_h), interpolation=cv2.INTER_LINEAR)
        if job.color_bgr is not None
        else _rasterize_rgb(xyz, rgb, job.fx, job.fy, job.cx, job.cy, job.img_w, job.img_h)
    )
    hp_bgr, hp_gray, hp_edge = _time_call(timings, "high_contrast_domain", _compute_high_contrast_domain, base_bgr)
    uv, valid = _time_call(
        timings,
        "project_points_to_image",
        _project_points_to_image,
        xyz,
        job.fx,
        job.fy,
        job.cx,
        job.cy,
        job.img_w,
        job.img_h,
    )

    tray_mask: np.ndarray | None = None
    tray_detect_ok = False
    near_plane_mask: np.ndarray | None = None
    no_hole_mask: np.ndarray | None = None
    top_quad: np.ndarray | None = None
    opening: OpeningDetection | None = None
    grasp: GraspResult | None = None
    failed_step: str | None = None
    err: str | None = None

    try:
        tray_mask, tray_detect_ok = _time_call(
            timings,
            "segment_tray",
            tray_pipeline.segment_tray,
            base_bgr,
            tray_state.runtime,
        )
        opening = _time_call(
            timings,
            "detect_opening",
            opening_pipeline.detect_opening,
            base_bgr,
            tray_mask,
            hp_gray,
        )
        mask_future = mask_executor.submit(
            opening_pipeline.compute_mask_pipeline,
            tray_mask,
            tray_detect_ok,
            opening,
            hp_gray,
            hp_edge,
        )
        xyz_local = _time_call(
            timings,
            "filter_local_points_fast",
            opening_pipeline.filter_opening_local_points,
            xyz,
            rgb,
            opening,
            job.img_w,
            job.img_h,
            uv,
            valid,
        )
        if xyz_local.shape[0] < DEFAULT_OPENING_MIN_POINTS:
            raise RuntimeError(f"开口局部点过少：{xyz_local.shape[0]}")
        plane = _time_call(timings, "estimate_opening_plane", pose_estimator.estimate_plane, xyz_local)
        grasp = _time_call(
            timings, "compute_grasp_fast", pose_estimator.compute_grasp, opening, plane, job.fx, job.fy, job.cx, job.cy, None
        )

        try:
            near_plane_mask, no_hole_mask = _time_call(
                timings,
                "mask_pipeline_sync",
                mask_future.result,
                timeout=max(1e-3, float(DEFAULT_MASK_SYNC_BUDGET_MS)) / 1000.0,
            )
            top_quad = _time_call(timings, "fit_top_quad", opening_pipeline.fit_rotated_quad, no_hole_mask)
            top_normal = _time_call(
                timings,
                "estimate_top_plane_normal",
                opening_pipeline.estimate_top_plane_normal,
                xyz,
                no_hole_mask,
                uv,
                valid,
            )
            top_normal = _time_call(
                timings, "stabilize_top_plane_normal", pose_estimator.stabilize_top_normal, top_normal, temporal_state
            )
            grasp = _time_call(
                timings,
                "compute_grasp_refined",
                pose_estimator.compute_grasp,
                opening,
                plane,
                job.fx,
                job.fy,
                job.cx,
                job.cy,
                top_normal,
            )
        except concurrent.futures.TimeoutError:
            failed_step = "mask_pipeline_sync_timeout"
            logger.debug(f"帧 {job.frame_idx} mask pipeline timeout，使用快速抓取结果")
        except Exception as exc:
            failed_step = "mask_pipeline_sync_error"
            logger.debug(f"帧 {job.frame_idx} mask pipeline error: {exc}")
    except Exception as exc:
        failed_step = _infer_failed_step_from_timings(timings)
        if isinstance(exc, cv2.error):
            err = f"OpenCV error: {exc}"
        else:
            err = str(exc)
    grasp = _time_call(timings, "stabilize_grasp_pose", pose_estimator.stabilize_grasp_result, grasp, temporal_state)

    overlay = _safe_draw_overlay_partial(
        base_bgr,
        tray_mask,
        tray_detect_ok,
        near_plane_mask,
        no_hole_mask,
        top_quad,
        opening,
        grasp,
        err,
        failed_step,
    )
    contrast = _safe_build_contrast_preview_partial(
        hp_gray,
        hp_edge,
        near_plane_mask,
        no_hole_mask,
        opening,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    timing_writer.write_rows(frame_idx=job.frame_idx, total_elapsed_ms=elapsed_ms, step_timings=timings, error=err)
    completed_steps = tuple(s.step_name for s in timings if s.status == "ok")
    return PipelineResult(
        frame_idx=job.frame_idx,
        xyz=xyz,
        rgb=rgb,
        tray_mask=tray_mask,
        tray_detect_ok=tray_detect_ok,
        near_plane_mask=near_plane_mask,
        no_hole_mask=no_hole_mask,
        top_plane_quad_uv=top_quad,
        opening=opening,
        grasp=grasp,
        overlay_bgr=overlay,
        contrast_bgr=contrast,
        elapsed_ms=elapsed_ms,
        error=err,
        failed_step=failed_step,
        completed_steps=completed_steps,
    )


# region 相机采集与可视化


def _capture_points_once(
    session: Gemini305, point_filter, max_depth_mm: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
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
    points = session.filter_points_for_sensor(raw_points, max_depth_mm=max_depth_mm, apply_sensor_frustum=True)
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
        opt.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)

    stop = {"flag": False}

    def _on_escape(_vis):
        stop["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    raw_pcd = o3d.geometry.PointCloud()
    frame_mesh = _empty_mesh()
    grasp_line = o3d.geometry.LineSet()
    grasp_line.points = o3d.utility.Vector3dVector(np.asarray([[0, 0, -10000], [1, 0, -10000]], dtype=np.float64))
    grasp_line.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    grasp_line.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.8, 1.0]], dtype=np.float64))
    for geo in (axis, raw_pcd, frame_mesh, grasp_line):
        vis.add_geometry(geo)
    view = vis.get_view_control()
    if view is not None:
        view.set_lookat([0.0, 0.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
    return vis, stop, raw_pcd, frame_mesh, grasp_line


def _update_raw_cloud(pcd: o3d.geometry.PointCloud, points: np.ndarray) -> None:
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = _extract_rgb(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(0.35 * rgb + 0.15, 0.0, 1.0))


def _update_result_3d(
    frame_mesh: o3d.geometry.TriangleMesh, grasp_line: o3d.geometry.LineSet, result: PipelineResult
) -> None:
    if result.grasp is None:
        return
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40.0, origin=[0.0, 0.0, 0.0])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = result.grasp.rotation
    mat[:3, 3] = result.grasp.grasp_point
    frame.transform(mat)
    frame_mesh.vertices = frame.vertices
    frame_mesh.triangles = frame.triangles
    frame_mesh.vertex_colors = frame.vertex_colors
    frame_mesh.vertex_normals = frame.vertex_normals

    line_points = np.vstack([result.grasp.grasp_point, result.grasp.pre_grasp_point]).astype(np.float64)
    grasp_line.points = o3d.utility.Vector3dVector(line_points)
    grasp_line.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    grasp_line.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.8, 1.0]], dtype=np.float64))


def _draw_overlay(
    base_bgr: np.ndarray,
    tray_mask: np.ndarray | None,
    tray_detect_ok: bool,
    near_plane_mask: np.ndarray | None,
    no_hole_mask: np.ndarray | None,
    top_plane_quad_uv: np.ndarray | None,
    opening: OpeningDetection | None,
    grasp: GraspResult | None,
) -> np.ndarray:
    out = base_bgr.copy()
    tray_m = None if tray_mask is None else (np.asarray(tray_mask) > 0)
    near_m = None if near_plane_mask is None else (np.asarray(near_plane_mask) > 0)
    top_m = None if no_hole_mask is None else (np.asarray(no_hole_mask) > 0)
    # 显示层互斥：严禁区域重合。
    if near_m is not None and top_m is not None:
        top_m = top_m & (~near_m)
    if tray_m is not None and near_m is not None:
        tray_m = tray_m & (~near_m)
    if tray_m is not None and top_m is not None:
        tray_m = tray_m & (~top_m)

    if tray_m is not None:
        out[tray_m] = (0.65 * out[tray_m] + 0.35 * np.array([0, 180, 180], dtype=np.float64)).astype(np.uint8)
        tx, ty, tw, th = _mask_bbox_xywh((tray_m.astype(np.uint8) * 255))
        cv2.putText(out, "Tray", (tx, max(14, ty - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 220), 1, cv2.LINE_AA)
    if near_m is not None:
        out[near_m] = (0.58 * out[near_m] + 0.42 * np.array([255, 120, 0], dtype=np.float64)).astype(np.uint8)
        near_u8 = near_m.astype(np.uint8) * 255
        if np.count_nonzero(near_u8) > 0:
            ctn, _ = cv2.findContours(near_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, ctn, -1, (255, 170, 0), 1, cv2.LINE_AA)
            mx, my, mw, mh = cv2.boundingRect(np.vstack(ctn))
            cv2.putText(
                out,
                "Opening Plane",
                (mx, max(14, my - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 170, 0),
                1,
                cv2.LINE_AA,
            )
    if top_m is not None:
        out[top_m] = (0.55 * out[top_m] + 0.45 * np.array([0, 200, 0], dtype=np.float64)).astype(np.uint8)
        if top_plane_quad_uv is not None:
            tq = np.round(np.asarray(top_plane_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(out, [tq], True, (0, 255, 0), 1, cv2.LINE_AA)
            mx2, my2, mw2, mh2 = cv2.boundingRect(tq)
            cv2.putText(
                out, "Top Plane", (mx2, max(14, my2 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA
            )
    if opening is not None:
        quad = np.round(opening.quad_uv).astype(np.int32)
        cv2.polylines(out, [quad], True, (0, 0, 255), 1, cv2.LINE_AA)
        qx, qy, qw, qh = cv2.boundingRect(quad)
        cv2.putText(out, "Opening", (qx, max(14, qy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
        u, v = int(round(opening.center_uv[0])), int(round(opening.center_uv[1]))
        cv2.circle(out, (u, v), 2, (0, 255, 0), -1)
        cv2.putText(out, "Center", (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
    if grasp is not None:
        rpy = _rotation_matrix_to_rpy_deg(grasp.rotation)
        txt1 = f"grasp XYZ {grasp.grasp_point[0]:.1f}, {grasp.grasp_point[1]:.1f}, {grasp.grasp_point[2]:.1f} mm"
        txt2 = f"RPY {rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f} deg"
        cv2.putText(out, txt1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, txt2, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"tray={'zero-shot' if tray_detect_ok else 'fallback'}",
        (12, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (60, 255, 60) if tray_detect_ok else (0, 200, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "near-plane(orange) -> top-plane(green)",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x1 = int(np.min(xs))
    x2 = int(np.max(xs))
    y1 = int(np.min(ys))
    y2 = int(np.max(ys))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def _build_contrast_preview(
    high_contrast_gray: np.ndarray,
    edge_mask: np.ndarray,
    near_plane_mask: np.ndarray | None,
    no_hole_mask: np.ndarray | None,
    opening: OpeningDetection | None,
) -> np.ndarray:
    """高反差保留预览：增强边缘与邻接边界可见性。"""
    out = cv2.cvtColor(high_contrast_gray, cv2.COLOR_GRAY2BGR)
    out[edge_mask > 0] = (255, 255, 255)
    if near_plane_mask is not None:
        c1, _ = cv2.findContours(near_plane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, c1, -1, (0, 165, 255), 1, cv2.LINE_AA)
    if no_hole_mask is not None:
        c2, _ = cv2.findContours(no_hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, c2, -1, (0, 255, 0), 1, cv2.LINE_AA)
    if opening is not None:
        q = np.round(opening.quad_uv).astype(np.int32)
        cv2.polylines(out, [q], True, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(
        out, "High-contrast retain + edges", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA
    )
    return out


def _compute_high_contrast_domain(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造高反差保留域，并返回 (highpass_bgr, filtered_gray, edge_mask)。"""
    blur = cv2.GaussianBlur(bgr, (0, 0), DEFAULT_CONTRAST_SIGMA)
    highpass = cv2.addWeighted(bgr, DEFAULT_CONTRAST_HP_A, blur, DEFAULT_CONTRAST_HP_B, 0.0)
    gray = cv2.cvtColor(highpass, cv2.COLOR_BGR2GRAY)
    # 过滤：先双边保边去噪，再轻微闭运算连边，提升平面/边界区分。
    gray_f = cv2.bilateralFilter(
        gray,
        d=DEFAULT_CONTRAST_BILATERAL_D,
        sigmaColor=DEFAULT_CONTRAST_BILATERAL_SIGMA_COLOR,
        sigmaSpace=DEFAULT_CONTRAST_BILATERAL_SIGMA_SPACE,
    )
    gray_f = cv2.morphologyEx(gray_f, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge = cv2.Canny(gray_f, DEFAULT_CONTRAST_CANNY_LOW, DEFAULT_CONTRAST_CANNY_HIGH)
    return highpass, gray_f, edge


def _draw_status_only(base_bgr: np.ndarray, message: str) -> np.ndarray:
    out = base_bgr.copy()
    cv2.putText(out, "estimate_grasp failed", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, message[:110], (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def _infer_failed_step_from_timings(timings: list[StepTiming]) -> str | None:
    for s in reversed(timings):
        if s.status == "error":
            return s.step_name
    return None


def _classify_failure_kind(failed_step: str | None) -> str:
    if failed_step is None:
        return "other"
    if failed_step == "segment_tray":
        return "tray"
    if failed_step == "detect_opening":
        return "opening"
    return "other"


def _safe_draw_overlay_partial(
    base_bgr: np.ndarray,
    tray_mask: np.ndarray | None,
    tray_detect_ok: bool,
    near_plane_mask: np.ndarray | None,
    no_hole_mask: np.ndarray | None,
    top_plane_quad_uv: np.ndarray | None,
    opening: OpeningDetection | None,
    grasp: GraspResult | None,
    error: str | None,
    failed_step: str | None,
) -> np.ndarray:
    out = _draw_overlay(
        base_bgr,
        tray_mask,
        tray_detect_ok,
        near_plane_mask,
        no_hole_mask,
        top_plane_quad_uv,
        opening,
        grasp,
    )
    if error is not None:
        kind = _classify_failure_kind(failed_step)
        cv2.putText(
            out,
            f"pipeline partial fail[{kind}] step={failed_step}",
            (12, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            error[:120],
            (12, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _safe_build_contrast_preview_partial(
    high_contrast_gray: np.ndarray,
    edge_mask: np.ndarray,
    near_plane_mask: np.ndarray | None,
    no_hole_mask: np.ndarray | None,
    opening: OpeningDetection | None,
) -> np.ndarray:
    try:
        return _build_contrast_preview(high_contrast_gray, edge_mask, near_plane_mask, no_hole_mask, opening)
    except cv2.error:
        return cv2.cvtColor(high_contrast_gray, cv2.COLOR_GRAY2BGR)


class _CsvTimingWriter:
    def __init__(self, csv_path: Path, run_id: str) -> None:
        self._csv_path = Path(csv_path)
        self._run_id = str(run_id)
        self._lock = threading.Lock()
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._csv_path.exists():
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "run_id",
                        "frame_idx",
                        "step_name",
                        "start_ts",
                        "end_ts",
                        "elapsed_ms",
                        "status",
                        "error",
                        "frame_total_elapsed_ms",
                    ]
                )

    def write_rows(
        self, frame_idx: int, total_elapsed_ms: float, step_timings: list[StepTiming], error: str | None
    ) -> None:
        frame_status = "ok" if error is None else "error"
        frame_error = "" if error is None else str(error)
        rows = [
            [
                self._run_id,
                int(frame_idx),
                s.step_name,
                s.start_ts,
                s.end_ts,
                f"{s.elapsed_ms:.3f}",
                s.status,
                s.error,
                f"{total_elapsed_ms:.3f}",
            ]
            for s in step_timings
        ]
        rows.append(
            [
                self._run_id,
                int(frame_idx),
                "frame_total",
                "",
                "",
                f"{total_elapsed_ms:.3f}",
                frame_status,
                frame_error,
                f"{total_elapsed_ms:.3f}",
            ]
        )
        with self._lock:
            with self._csv_path.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)


def _time_call(step_timings: list[StepTiming], step_name: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    start_ts = datetime.now().isoformat(timespec="milliseconds")
    try:
        out = fn(*args, **kwargs)
        status = "ok"
        err = ""
        return out
    except Exception as exc:
        status = "error"
        err = str(exc)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        end_ts = datetime.now().isoformat(timespec="milliseconds")
        step_timings.append(
            StepTiming(
                step_name=str(step_name),
                start_ts=start_ts,
                end_ts=end_ts,
                elapsed_ms=float(elapsed_ms),
                status=status,
                error=err,
            )
        )


# endregion


# region 数学与通用工具


def _extract_rgb(points: np.ndarray) -> np.ndarray:
    if points.shape[1] >= 6:
        rgb = np.asarray(points[:, 3:6], dtype=np.float32)
        if rgb.size > 0 and float(np.max(rgb)) > 1.0:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0).astype(np.float64)
    return np.full((points.shape[0], 3), 0.72, dtype=np.float64)


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
    valid = np.isfinite(z) & (z > 1e-6)
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
    xyz: np.ndarray, rgb: np.ndarray, fx: float, fy: float, cx: float, cy: float, w: int, h: int
) -> np.ndarray:
    uv, valid = _project_points_to_image(xyz, fx, fy, cx, cy, w, h)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    idx = np.where(valid)[0]
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


def _rotation_matrix_to_rpy_deg(rot: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.asarray([roll, pitch, yaw], dtype=np.float64))


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
    return points[::step]


def _empty_mesh() -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0], [0.0, 1.0, -10000.0]])
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32))
    return mesh


def _poll_viewer(vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    alive = vis.poll_events()
    vis.update_renderer()
    return bool(alive)


def _safe_destroy_cv_window(window_name: str) -> None:
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        # 用户可能已手动关闭窗口，此时无需再次销毁。
        pass


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


if __name__ == "__main__":
    main()

