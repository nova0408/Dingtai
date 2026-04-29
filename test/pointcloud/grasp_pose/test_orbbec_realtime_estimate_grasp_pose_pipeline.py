from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test.pointcloud import test_orbbec_realtime_plane_segmentation_zero_shot as zs

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
DEFAULT_TRAY_USE_SAM = True
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
class OpeningDetection:
    center_uv: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    quad_uv: np.ndarray
    score: float


@dataclass(frozen=True)
class PlaneResult:
    normal: np.ndarray
    d: float


@dataclass(frozen=True)
class GraspResult:
    grasp_point: np.ndarray
    pre_grasp_point: np.ndarray
    rotation: np.ndarray


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


# endregion


# region 主流程

def main(
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
    compute_min_interval_s: float = DEFAULT_COMPUTE_MIN_INTERVAL_S,
) -> None:
    options = SessionOptions(timeout_ms=int(timeout_ms), preferred_capture_fps=max(1, int(capture_fps)))
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
    job_queue: queue.Queue[CaptureJob | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[PipelineResult] = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    busy_event = threading.Event()
    worker = threading.Thread(
        target=_worker_loop,
        args=(job_queue, result_queue, stop_event, busy_event),
        name="estimate_grasp_worker",
        daemon=True,
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
                last_overlay = result.overlay_bgr
                last_contrast = result.contrast_bgr
                if result.error is None and result.grasp is not None:
                    p = result.grasp.grasp_point
                    rpy = _rotation_matrix_to_rpy_deg(result.grasp.rotation)
                    logger.info(
                        f"帧 {result.frame_idx} grasp XYZ {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f} mm; "
                        f"RPY {rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f} deg; elapsed {result.elapsed_ms:.1f} ms"
                    )
                elif result.error is not None:
                    logger.warning(f"帧 {result.frame_idx} 计算失败：{result.error}")

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

            merged = np.hstack([last_overlay, last_contrast])
            cv2.imshow(DEFAULT_2D_MERGED_WINDOW_NAME, merged)
            if cv2.waitKey(1) == 27:
                break
            if (
                (not _poll_viewer(vis))
                or cv2.getWindowProperty(DEFAULT_2D_MERGED_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            ):
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
        cv2.destroyWindow(DEFAULT_2D_MERGED_WINDOW_NAME)


# endregion


# region 后台计算

def _worker_loop(
    job_queue: queue.Queue[CaptureJob | None],
    result_queue: queue.Queue[PipelineResult],
    stop_event: threading.Event,
    busy_event: threading.Event,
) -> None:
    tray_detector = _build_tray_zero_shot_detector()
    while not stop_event.is_set():
        job = job_queue.get()
        if job is None:
            break
        busy_event.set()
        try:
            result = _run_compute_job(job, tray_detector)
            _put_latest_result(result_queue, result)
        except Exception as exc:
            logger.exception(f"帧 {job.frame_idx} 计算线程异常：{exc}")
        finally:
            busy_event.clear()
            job_queue.task_done()


def _run_compute_job(job: CaptureJob, tray_detector) -> PipelineResult:
    t0 = time.perf_counter()
    xyz = np.asarray(job.points[:, :3], dtype=np.float64)
    rgb = _extract_rgb(job.points)
    base_bgr = (
        cv2.resize(job.color_bgr, (job.img_w, job.img_h), interpolation=cv2.INTER_LINEAR)
        if job.color_bgr is not None
        else _rasterize_rgb(xyz, rgb, job.fx, job.fy, job.cx, job.cy, job.img_w, job.img_h)
    )

    try:
        tray_mask, tray_detect_ok = _segment_tray_by_zero_shot(base_bgr, tray_detector)
        opening = _detect_rect_opening_auto(base_bgr, tray_mask)
        near_plane_mask = _build_near_dark_plane_mask(base_bgr, tray_mask, opening) if tray_detect_ok else None
        no_hole_mask = _build_no_hole_top_plane_mask(base_bgr, tray_mask, opening, near_plane_mask)
        near_plane_mask, no_hole_mask = _enforce_disjoint_region_masks(near_plane_mask, no_hole_mask)
        top_quad = _fit_rotated_quad_from_mask(no_hole_mask)
        xyz_local = _filter_local_points(
            xyz,
            rgb,
            opening,
            job.fx,
            job.fy,
            job.cx,
            job.cy,
            job.img_w,
            job.img_h,
            no_hole_mask=no_hole_mask,
        )
        if xyz_local.shape[0] < DEFAULT_OPENING_MIN_POINTS:
            raise RuntimeError(f"开口局部点过少：{xyz_local.shape[0]}")
        plane = _estimate_plane(xyz_local)
        top_normal = _estimate_mask_plane_normal(xyz, no_hole_mask, job.fx, job.fy, job.cx, job.cy, job.img_w, job.img_h)
        grasp = _compute_grasp(opening, plane, job.fx, job.fy, job.cx, job.cy, top_normal)
        overlay = _draw_overlay(base_bgr, tray_mask, tray_detect_ok, near_plane_mask, no_hole_mask, top_quad, opening, grasp)
        contrast = _build_contrast_preview(base_bgr, near_plane_mask, no_hole_mask, opening)
        err = None
    except cv2.error as exc:
        tray_mask = None
        tray_detect_ok = False
        near_plane_mask = None
        no_hole_mask = None
        top_quad = None
        opening = None
        grasp = None
        overlay = _draw_status_only(base_bgr, f"OpenCV error: {exc}")
        contrast = _build_contrast_preview(base_bgr, None, None, None)
        err = f"OpenCV error: {exc}"
    except Exception as exc:
        tray_mask = None
        tray_detect_ok = False
        near_plane_mask = None
        no_hole_mask = None
        top_quad = None
        opening = None
        grasp = None
        overlay = _draw_status_only(base_bgr, str(exc))
        contrast = _build_contrast_preview(base_bgr, None, None, None)
        err = str(exc)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
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
    )


def _enforce_disjoint_region_masks(
    near_plane_mask: np.ndarray | None,
    top_plane_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """强制区域互斥：top 与 near 严禁重合。"""
    if near_plane_mask is None or top_plane_mask is None:
        return near_plane_mask, top_plane_mask
    near = (near_plane_mask > 0).astype(np.uint8) * 255
    top = (top_plane_mask > 0).astype(np.uint8) * 255
    if near.shape != top.shape:
        return near_plane_mask, top_plane_mask
    # 留一圈安全边距，避免视觉上贴边重合。
    near_guard = cv2.dilate(near, np.ones((3, 3), dtype=np.uint8), iterations=1)
    top = cv2.bitwise_and(top, cv2.bitwise_not(near_guard))
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if np.count_nonzero(top) < 20:
        top = np.zeros_like(top)
    return near, top


def _fit_rotated_quad_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    m = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if np.count_nonzero(m) < 40:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    pts = np.vstack(cnts)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float64)
    return box


def _estimate_mask_plane_normal(
    xyz: np.ndarray,
    mask: np.ndarray | None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    if mask is None:
        return None
    m = (np.asarray(mask) > 0)
    if np.count_nonzero(m) < 60:
        return None
    uv, valid = _project_points_to_image(xyz, fx, fy, cx, cy, img_w, img_h)
    idx = np.where(valid)[0]
    if idx.size < 120:
        return None
    u = uv[idx, 0]
    v = uv[idx, 1]
    keep = m[v, u]
    sel = idx[keep]
    if sel.size < 120:
        return None
    # 使用 top plane 的全部对应点拟合法线，减少局部抽样抖动。
    pts = np.asarray(xyz[sel], dtype=np.float64)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 120:
        return None
    # 稳健去极值：按到中值点距离截断上 3% 离群点
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med.reshape(1, 3), axis=1)
    d_thr = np.percentile(d, 97.0)
    pts = pts[d <= d_thr]
    if pts.shape[0] < 100:
        return None
    c = np.mean(pts, axis=0)
    q = pts - c.reshape(1, 3)
    cov = (q.T @ q) / max(1, q.shape[0])
    vals, vecs = np.linalg.eigh(cov)
    n = np.asarray(vecs[:, int(np.argmin(vals))], dtype=np.float64)
    n = _normalize(n)
    if np.dot(n, np.array([0.0, 0.0, -1.0], dtype=np.float64)) < 0.0:
        n = -n
    return n


# endregion


# region 估计抓取核心逻辑

def _detect_rect_opening_auto(rgb_bgr: np.ndarray, tray_mask: np.ndarray | None) -> OpeningDetection:
    h, w = rgb_bgr.shape[:2]
    if tray_mask is None or np.count_nonzero(tray_mask) < 0.01 * h * w:
        raise RuntimeError("托盘掩码无效，无法执行严格开口检测")

    tx, ty, tw, th = _mask_bbox_xywh(tray_mask)
    # 严格限定前立面区域（托盘内）
    x1 = int(max(0, tx + 0.08 * tw))
    x2 = int(min(w - 1, tx + 0.92 * tw))
    y1 = int(max(0, ty + 0.72 * th))
    y2 = int(min(h - 1, ty + 0.97 * th))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("前立面 ROI 无效")

    roi = rgb_bgr[y1:y2, x1:x2]
    roi_tray = tray_mask[y1:y2, x1:x2]
    if roi.size == 0 or np.count_nonzero(roi_tray) < 50:
        raise RuntimeError("前立面托盘像素不足")
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _hp, hp_gray, _hp_edge = _compute_high_contrast_domain(roi)
    del _hp, _hp_edge
    thresholds = sorted(set(int(np.clip(t, 20, 180)) for t in np.percentile(gray_blur, [4, 6, 8, 12, 16, 20, 25, 30])))

    candidates: list[tuple[float, np.ndarray]] = []
    for thr in thresholds:
        mask = np.zeros_like(gray_blur, dtype=np.uint8)
        mask[(gray_blur <= thr) & (roi_tray > 0)] = 255
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, gray_blur.shape[1] // 12), 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 40.0:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), _ = rect
            long_side = max(float(rw), float(rh))
            short_side = max(1.0, min(float(rw), float(rh)))
            aspect = long_side / short_side
            if not (2.8 <= aspect <= 20.0):
                continue
            wr = long_side / max(1.0, float(tw))
            hr = short_side / max(1.0, float(th))
            if not (0.10 <= wr <= 0.62 and 0.010 <= hr <= 0.095):
                continue
            y_pref = (y1 + cy - ty) / max(1.0, float(th))
            if y_pref < 0.72:
                continue
            box = cv2.boxPoints(rect)
            patch_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.fillConvexPoly(patch_mask, np.round(box).astype(np.int32), 255)
            in_roi = (patch_mask > 0) & (roi_tray > 0)
            if np.count_nonzero(in_roi) < 20:
                continue
            patch_raw = gray[in_roi]
            patch_hp = hp_gray[in_roi]
            dark_ratio_raw = float(np.mean(patch_raw <= thr))
            dark_ratio_hp = float(np.mean(patch_hp <= np.percentile(hp_gray[roi_tray > 0], 16)))
            ring = _patch_ring(gray_blur, int(cx - rw / 2), int(cy - rh / 2), int(max(1, rw)), int(max(1, rh)))
            slot_mean = float(np.mean(patch_raw))
            ring_mean = float(np.mean(ring)) if ring.size > 0 else slot_mean
            contrast_score = float(np.clip((ring_mean - slot_mean) / 45.0, 0.0, 1.5))
            x_center_pref = 1.0 - min(abs((x1 + cx) - (tx + 0.5 * tw)) / max(1.0, 0.5 * tw), 1.0)
            score = (
                2.4 * dark_ratio_raw
                + 1.5 * dark_ratio_hp
                + 1.5 * min(aspect / 8.0, 2.0)
                + 1.1 * contrast_score
                + 1.0 * x_center_pref
                + 0.7 * y_pref
            )
            box[:, 0] += x1
            box[:, 1] += y1
            candidates.append((float(score), box.astype(np.float64)))

    if len(candidates) == 0:
        raise RuntimeError("未检测到开口")

    candidates.sort(key=lambda x: x[0], reverse=True)
    quad = candidates[0][1]
    bx, by, bw, bh = cv2.boundingRect(np.round(quad).astype(np.int32))
    center_uv = np.mean(quad, axis=0).astype(np.float64)
    return OpeningDetection(center_uv=center_uv, bbox_xywh=(int(bx), int(by), int(bw), int(bh)), quad_uv=quad, score=float(candidates[0][0]))


def _filter_local_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opening: OpeningDetection,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    no_hole_mask: np.ndarray | None,
) -> np.ndarray:
    uv, valid = _project_points_to_image(xyz, fx, fy, cx, cy, img_w, img_h)
    local_roi = _build_opening_local_roi_mask((img_h, img_w), opening)
    uv_ok = valid & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
    mask = np.zeros_like(valid)
    idx0 = np.where(uv_ok)[0]
    if idx0.size > 0:
        u0 = uv[idx0, 0]
        v0 = uv[idx0, 1]
        mask[idx0] = local_roi[v0, u0] > 0
    if np.count_nonzero(mask) > 30:
        inten = np.max((rgb[mask] * 255.0).astype(np.float32), axis=1)
        thr = float(np.clip(np.percentile(inten, 78), 70, 185))
        idx = np.where(mask)[0]
        keep = inten <= thr
        mask2 = np.zeros_like(mask)
        mask2[idx[keep]] = True
        mask = mask2
    if no_hole_mask is not None:
        uv_ok = valid & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
        idx = np.where(uv_ok)[0]
        if idx.size > 0:
            u = uv[idx, 0]
            v = uv[idx, 1]
            keep = no_hole_mask[v, u] > 0
            mask3 = np.zeros_like(mask)
            mask3[idx[keep]] = True
            mix = mask & mask3
            if np.count_nonzero(mix) >= DEFAULT_OPENING_MIN_POINTS:
                mask = mix
    # 最终用稳健深度带收敛局部点，降低孔阵列/后景混入。
    if np.count_nonzero(mask) >= DEFAULT_OPENING_MIN_POINTS:
        zz = xyz[mask, 2]
        z_med = float(np.median(zz))
        z_abs = np.abs(xyz[:, 2] - z_med)
        mask &= z_abs <= 22.0
    return xyz[mask]


def _build_tray_zero_shot_detector():
    try:
        return zs.ZeroShotObjectPartitionDetector(
            gd_model_id=zs.DEFAULT_GD_MODEL_ID,
            sam_model_id=zs.DEFAULT_SAM_MODEL_ID,
            hf_cache_dir=zs.DEFAULT_HF_CACHE_DIR,
            hf_local_files_only=zs.DEFAULT_HF_LOCAL_FILES_ONLY,
            device=zs.DEFAULT_DEVICE,
            proxy_url=zs.DEFAULT_PROXY_URL,
            prompt=zs.DEFAULT_PROMPT,
            target_keywords=zs.DEFAULT_TARGET_KEYWORDS,
            strict_target_filter=True,
            max_targets=1,
            use_sam=DEFAULT_TRAY_USE_SAM,
            box_threshold=zs.DEFAULT_BOX_THRESHOLD,
            text_threshold=zs.DEFAULT_TEXT_THRESHOLD,
            min_target_conf=0.20,
            topk_objects=2,
            sam_max_boxes=1,
            sam_primary_only=True,
            sam_secondary_conf_threshold=zs.DEFAULT_SAM_SECONDARY_CONF_THRESHOLD,
            combine_prompts_forward=zs.DEFAULT_COMBINE_PROMPTS_FORWARD,
            min_mask_pixels=zs.DEFAULT_MIN_MASK_PIXELS,
            mask_iou_suppress=zs.DEFAULT_MASK_IOU_SUPPRESS,
            detect_max_side=zs.DEFAULT_DETECT_MAX_SIDE,
        )
    except Exception as exc:
        logger.warning(f"zero-shot 托盘分割器初始化失败，降级为传统分割：{exc}")
        return None


def _segment_tray_by_zero_shot(rgb_bgr: np.ndarray, tray_detector) -> tuple[np.ndarray, bool]:
    h, w = rgb_bgr.shape[:2]
    if tray_detector is not None:
        dets = tray_detector.detect(rgb_bgr)
        if len(dets) > 0:
            det = max(dets, key=lambda d: d.area_pixels)
            mask = np.asarray(det.mask, dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
            return mask, True
    # 兜底：zero-shot 空检时退回亮度阈值 + 连通域，避免整帧失败。
    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    base = (gray <= np.percentile(gray, 46)).astype(np.uint8) * 255
    base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
    # 只看下半区，降低背景盒子误选。
    upper_cut = int(0.38 * h)
    base[:upper_cut, :] = 0
    num, cc, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
    if num <= 1:
        return base, False
    best = 1
    best_score = -1e18
    tgt = np.array([0.5 * w, 0.78 * h], dtype=np.float64)
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        x = float(stats[idx, cv2.CC_STAT_LEFT] + 0.5 * stats[idx, cv2.CC_STAT_WIDTH])
        y = float(stats[idx, cv2.CC_STAT_TOP] + 0.5 * stats[idx, cv2.CC_STAT_HEIGHT])
        dist = float(np.linalg.norm(np.array([x, y], dtype=np.float64) - tgt))
        # 托盘通常在画面下方中间且连通域较大。
        score = area - 1.6 * dist
        if score > best_score:
            best_score = score
            best = idx
    out = np.zeros((h, w), dtype=np.uint8)
    out[cc == best] = 255
    return out, False


def _build_near_dark_plane_mask(
    rgb_bgr: np.ndarray, tray_mask: np.ndarray, opening: OpeningDetection
) -> np.ndarray | None:
    """分割开口附近暗色附近平面：以开口为种子做邻域生长，强制邻接。"""
    h, w = rgb_bgr.shape[:2]
    ring = _build_opening_surround_ring_mask((h, w), opening)
    work = cv2.bitwise_and(ring, tray_mask)
    if np.count_nonzero(work) == 0:
        return None
    _hp_bgr, hp_gray, hp_edge = _compute_high_contrast_domain(rgb_bgr)
    # 以开口四边形边界上的像素作为 seed，保证与开口邻接。
    seeds = _collect_opening_boundary_seeds(opening, work, hp_gray)
    if len(seeds) == 0:
        return None
    # 在高反差图域做区域生长：灰度差异小 + 不穿越强边缘 + 限制在 ring/tray。
    edge_block = cv2.dilate(hp_edge, np.ones((3, 3), dtype=np.uint8), iterations=1)
    grow = _region_grow_from_seeds(
        gray=hp_gray,
        allowed_mask=work,
        edge_block=edge_block,
        seeds=seeds,
        local_diff=int(DEFAULT_NEAR_GROW_LOCAL_DIFF),
        global_diff=int(DEFAULT_NEAR_GROW_GLOBAL_DIFF),
        max_pixels=int(DEFAULT_NEAR_GROW_MAX_PIXELS),
    )
    if np.count_nonzero(grow) < 40:
        return None
    grow = cv2.morphologyEx(grow, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    grow = cv2.morphologyEx(grow, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=2)
    return grow


def _build_no_hole_top_plane_mask(
    rgb_bgr: np.ndarray, tray_mask: np.ndarray, opening: OpeningDetection, near_plane_mask: np.ndarray | None
) -> np.ndarray | None:
    """构造开口上方无孔平面像素掩码（多候选 + 回退）。"""
    h, w = rgb_bgr.shape[:2]
    roi_poly = _build_no_hole_roi_poly((h, w), opening, tray_mask)
    if roi_poly is None:
        return None
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(poly_mask, np.round(roi_poly).astype(np.int32), 255)
    base_mask = cv2.bitwise_and(poly_mask, tray_mask)
    if np.count_nonzero(base_mask) == 0:
        return None
    hp_bgr, hp_gray, hp_edge = _compute_high_contrast_domain(rgb_bgr)
    _ = hp_bgr
    target = np.mean(roi_poly, axis=0)

    # 优先策略：从附近平面边界邻接带作为种子，向 base_mask 内生长
    if near_plane_mask is not None and np.count_nonzero(near_plane_mask) > 0:
        near_ring = cv2.dilate(near_plane_mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
        near_ring = cv2.bitwise_and(near_ring, cv2.bitwise_not(near_plane_mask))
        seed_band = cv2.bitwise_and(near_ring, base_mask)
        ys, xs = np.where(seed_band > 0)
        if xs.size > 0:
            step = max(1, xs.size // 140)
            seeds: list[tuple[int, int, int]] = []
            for i in range(0, xs.size, step):
                sx = int(xs[i])
                sy = int(ys[i])
                seeds.append((sx, sy, int(hp_gray[sy, sx])))
            edge_block = cv2.dilate(hp_edge, np.ones((3, 3), dtype=np.uint8), iterations=1)
            grown = _region_grow_from_seeds(
                gray=hp_gray,
                allowed_mask=base_mask,
                edge_block=edge_block,
                seeds=seeds,
                local_diff=max(10, int(DEFAULT_NEAR_GROW_LOCAL_DIFF) - 2),
                global_diff=max(18, int(DEFAULT_NEAR_GROW_GLOBAL_DIFF) - 6),
                max_pixels=max(12000, int(DEFAULT_NEAR_GROW_MAX_PIXELS)),
            )
            grown = cv2.morphologyEx(grown, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
            grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
            grown = _select_component_near_target(grown, target)
            if np.count_nonzero(grown) >= 180:
                return grown

    # 回退策略：几何 + 低边缘密度筛选
    edge_soft = cv2.GaussianBlur(hp_edge.astype(np.float32), (5, 5), 0)
    low_edge = np.zeros_like(base_mask)
    thr = float(np.percentile(edge_soft[base_mask > 0], 55)) if np.count_nonzero(base_mask) > 0 else 0.0
    low_edge[(edge_soft <= thr) & (base_mask > 0)] = 255
    fallback = cv2.morphologyEx(low_edge, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8), iterations=1)
    fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8), iterations=2)
    fallback = _select_component_near_target(fallback, target)
    if np.count_nonzero(fallback) < 180:
        return None
    return fallback


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    x1 = int(np.min(xs))
    x2 = int(np.max(xs))
    y1 = int(np.min(ys))
    y2 = int(np.max(ys))
    return x1, y1, max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)


def _select_component_near_target(mask: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    num, cc, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    best_id = 0
    best_score = -1e18
    tgt = np.asarray(target_xy, dtype=np.float64)
    for idx in range(1, num):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        c = np.asarray(cent[idx], dtype=np.float64)
        dist = float(np.linalg.norm(c - tgt))
        score = area - 1.3 * dist
        if score > best_score:
            best_score = score
            best_id = idx
    out = np.zeros_like(mask)
    if best_id > 0:
        out[cc == best_id] = 255
    return out


def _select_components_in_ring(mask: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """保留环带内的多个有效连通域，避免只剩一角。"""
    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    out = np.zeros_like(mask)
    ring_px = ring > 0
    for idx in range(1, num):
        comp = cc == idx
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 12.0:
            continue
        overlap = float(np.count_nonzero(comp & ring_px))
        if overlap <= 0.0:
            continue
        out[comp] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)
    return out


def _build_opening_local_roi_mask(image_shape: tuple[int, int], opening: OpeningDetection) -> np.ndarray:
    h, w = image_shape
    quad = np.asarray(opening.quad_uv, dtype=np.float64)
    c = np.mean(quad, axis=0)
    v_long, v_short = _opening_axes_from_quad(quad)
    long_len, short_len = _opening_long_short_lengths(quad)
    rect_c = c
    rect_w = max(16.0, long_len * 2.35)
    rect_h = max(12.0, short_len * 3.10)
    poly = _rot_rect_to_poly(rect_c, v_long, v_short, rect_w, rect_h)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)
    return mask


def _build_opening_surround_ring_mask(image_shape: tuple[int, int], opening: OpeningDetection) -> np.ndarray:
    """构造围绕开口四周的旋转环带区域（不是单侧小块）。"""
    h, w = image_shape
    quad = np.asarray(opening.quad_uv, dtype=np.float64)
    c = np.mean(quad, axis=0)
    v_long, v_short = _opening_axes_from_quad(quad)
    long_len, short_len = _opening_long_short_lengths(quad)

    outer_w = max(16.0, long_len * 3.20)
    outer_h = max(14.0, short_len * 4.20)
    inner_w = max(8.0, long_len * 1.08)
    inner_h = max(6.0, short_len * 1.35)

    outer = _rot_rect_to_poly(c, v_long, v_short, outer_w, outer_h)
    inner = _rot_rect_to_poly(c, v_long, v_short, inner_w, inner_h)
    outer[:, 0] = np.clip(outer[:, 0], 0, w - 1)
    outer[:, 1] = np.clip(outer[:, 1], 0, h - 1)
    inner[:, 0] = np.clip(inner[:, 0], 0, w - 1)
    inner[:, 1] = np.clip(inner[:, 1], 0, h - 1)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(outer).astype(np.int32), 255)
    cv2.fillConvexPoly(mask, np.round(inner).astype(np.int32), 0)
    return mask


def _collect_opening_boundary_seeds(
    opening: OpeningDetection, allowed_mask: np.ndarray, gray: np.ndarray
) -> list[tuple[int, int, int]]:
    """采集开口边界附近种子点 (x,y,gray)。"""
    h, w = allowed_mask.shape[:2]
    quad = np.round(opening.quad_uv).astype(np.int32)
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(edge_mask, [quad], True, 255, 2, cv2.LINE_AA)
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    seed_mask = cv2.bitwise_and(edge_mask, allowed_mask)
    ys, xs = np.where(seed_mask > 0)
    if xs.size == 0:
        return []
    seeds: list[tuple[int, int, int]] = []
    # 均匀抽样，避免 seed 过多拖慢实时性。
    step = max(1, xs.size // 120)
    for i in range(0, xs.size, step):
        x = int(xs[i])
        y = int(ys[i])
        seeds.append((x, y, int(gray[y, x])))
    return seeds


def _region_grow_from_seeds(
    gray: np.ndarray,
    allowed_mask: np.ndarray,
    edge_block: np.ndarray,
    seeds: list[tuple[int, int, int]],
    local_diff: int,
    global_diff: int,
    max_pixels: int,
) -> np.ndarray:
    """基于像素灰度差与边缘阻挡的 BFS 生长。"""
    h, w = gray.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=np.uint8)
    q: list[tuple[int, int, int, int]] = []
    for sx, sy, sgv in seeds:
        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            continue
        if allowed_mask[sy, sx] == 0:
            continue
        q.append((sx, sy, sgv, sgv))
        visited[sy, sx] = 1

    head = 0
    pix_count = 0
    while head < len(q):
        x, y, seed_ref, parent_gray = q[head]
        head += 1
        gv = int(gray[y, x])
        # 全局约束：不偏离种子太远；局部约束：相邻步长不过大
        if abs(gv - seed_ref) > global_diff:
            continue
        if abs(gv - parent_gray) > local_diff:
            continue
        # 边缘软约束：在强边缘上允许更小范围穿越，避免完全截断
        if edge_block[y, x] > 0 and abs(gv - parent_gray) > max(4, local_diff // 2):
            continue
        if out[y, x] == 0:
            out[y, x] = 255
            pix_count += 1
            if pix_count >= max_pixels:
                break
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                continue
            if visited[ny, nx] != 0:
                continue
            visited[ny, nx] = 1
            if allowed_mask[ny, nx] == 0:
                continue
            q.append((nx, ny, seed_ref, gv))
    return out


def _build_no_hole_roi_poly(
    image_shape: tuple[int, int], opening: OpeningDetection, tray_mask: np.ndarray
) -> np.ndarray | None:
    h, w = image_shape
    quad = np.asarray(opening.quad_uv, dtype=np.float64)
    c = np.mean(quad, axis=0)
    v_long, v_short = _opening_axes_from_quad(quad)
    # v_short 朝向托盘内部（远离开口）方向：通过 tray centroid 判定符号。
    ys, xs = np.where(tray_mask > 0)
    if xs.size == 0:
        return None
    tray_c = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
    if float(np.dot(v_short, tray_c - c)) < 0.0:
        v_short = -v_short
    long_len, short_len = _opening_long_short_lengths(quad)
    # 无孔平面 ROI：沿托盘内部方向偏移，保证与开口区域分离。
    center_shift = 2.05 * short_len
    rect_c = c + v_short * center_shift
    rect_w = max(18.0, long_len * 1.45)
    rect_h = max(14.0, short_len * 2.90)
    poly = _rot_rect_to_poly(rect_c, v_long, v_short, rect_w, rect_h)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    return poly


def _opening_axes_from_quad(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_i = 0
    best_len = -1.0
    for i in range(4):
        j = (i + 1) % 4
        ll = float(np.linalg.norm(quad[j] - quad[i]))
        if ll > best_len:
            best_len = ll
            best_i = i
    v_long = _normalize(quad[(best_i + 1) % 4] - quad[best_i])
    n2 = np.array([-v_long[1], v_long[0]], dtype=np.float64)
    v_short = _normalize(n2)
    return v_long, v_short


def _opening_long_short_lengths(quad: np.ndarray) -> tuple[float, float]:
    lens = []
    for i in range(4):
        j = (i + 1) % 4
        lens.append(float(np.linalg.norm(quad[j] - quad[i])))
    if len(lens) == 0:
        return 20.0, 6.0
    return max(lens), max(2.0, min(lens))


def _rot_rect_to_poly(center: np.ndarray, v_long: np.ndarray, v_short: np.ndarray, w: float, h: float) -> np.ndarray:
    hw = 0.5 * float(w)
    hh = 0.5 * float(h)
    c = np.asarray(center, dtype=np.float64)
    p1 = c - v_long * hw - v_short * hh
    p2 = c + v_long * hw - v_short * hh
    p3 = c + v_long * hw + v_short * hh
    p4 = c - v_long * hw + v_short * hh
    return np.asarray([p1, p2, p3, p4], dtype=np.float64)


def _patch_ring(gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """获取候选框外围一圈像素，用于评估开口黑槽对比度。"""
    ih, iw = gray.shape[:2]
    pad_x = max(3, int(round(0.35 * w)))
    pad_y = max(2, int(round(0.90 * h)))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(iw, x + w + pad_x)
    y2 = min(ih, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0,), dtype=np.uint8)
    outer = gray[y1:y2, x1:x2]
    inner = np.zeros_like(outer, dtype=np.uint8)
    ix1 = x - x1
    iy1 = y - y1
    ix2 = min(ix1 + w, outer.shape[1])
    iy2 = min(iy1 + h, outer.shape[0])
    if ix2 > ix1 and iy2 > iy1:
        inner[iy1:iy2, ix1:ix2] = 255
    ring_mask = inner == 0
    vals = outer[ring_mask]
    return vals


def _expand_opening_quad(
    quad_uv: np.ndarray, image_shape: tuple[int, int], long_scale: float, short_scale: float
) -> np.ndarray:
    """沿开口长短轴外扩四边形，避免只命中黑槽核心。"""
    h, w = image_shape
    quad = np.asarray(quad_uv, dtype=np.float64)
    c = np.mean(quad, axis=0)
    v_long, v_short = _opening_axes_from_quad(quad)
    long_len, short_len = _opening_long_short_lengths(quad)
    rect_w = max(8.0, float(long_len) * float(long_scale))
    rect_h = max(6.0, float(short_len) * float(short_scale))
    poly = _rot_rect_to_poly(c, v_long, v_short, rect_w, rect_h)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    return poly


def _refine_opening_quad_by_dark_band(
    quad_uv: np.ndarray,
    gray_full: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """用高置信暗槽带回裁开口框，防止过度膨胀。"""
    h, w = image_shape
    quad = np.asarray(quad_uv, dtype=np.float64)
    c = np.mean(quad, axis=0)
    v_long, v_short = _opening_axes_from_quad(quad)
    long_len, short_len = _opening_long_short_lengths(quad)

    # 在当前框附近仅保留最暗细长区域，重新估计宽高。
    roi_poly = _rot_rect_to_poly(c, v_long, v_short, long_len * 1.15, short_len * 1.35)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, np.round(roi_poly).astype(np.int32), 255)
    vals = gray_full[roi_mask > 0]
    if vals.size < 30:
        return quad
    thr = float(np.percentile(vals, 18))
    dark = np.zeros((h, w), dtype=np.uint8)
    dark[(gray_full <= thr) & (roi_mask > 0)] = 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 3), dtype=np.uint8), iterations=1)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return quad
    best = None
    best_score = -1e18
    for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        (_cx, _cy), (rw, rh), _ = rect
        l = max(float(rw), float(rh))
        s = max(1.0, min(float(rw), float(rh)))
        asp = l / s
        if asp < 2.0:
            continue
        score = l * min(asp, 20.0)
        if score > best_score:
            best_score = score
            best = rect
    if best is None:
        return quad
    b = cv2.boxPoints(best).astype(np.float64)
    # 仅做小幅放宽，避免再次膨胀。
    bc = np.mean(b, axis=0)
    bv_long, bv_short = _opening_axes_from_quad(b)
    bl, bs = _opening_long_short_lengths(b)
    out = _rot_rect_to_poly(bc, bv_long, bv_short, bl * 1.08, bs * 1.15)
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


def _estimate_plane(xyz_local: np.ndarray) -> PlaneResult:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz_local, dtype=np.float64))
    if len(pcd.points) >= 300:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(pcd.points) < 50:
        raise RuntimeError(f"滤波后点数太少：{len(pcd.points)}")

    model, inliers = pcd.segment_plane(distance_threshold=3.0, ransac_n=3, num_iterations=1500)
    if len(inliers) < 30:
        raise RuntimeError("平面内点不足")

    n = np.asarray(model[:3], dtype=np.float64)
    d = float(model[3])
    n = _normalize(n)
    if np.dot(n, np.array([0.0, 0.0, -1.0], dtype=np.float64)) < 0.0:
        n = -n
        d = -d
    return PlaneResult(normal=n, d=d)


def _compute_grasp(
    opening: OpeningDetection,
    plane: PlaneResult,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    top_ref_normal: np.ndarray | None = None,
) -> GraspResult:
    u, v = opening.center_uv
    grasp_point = _pixel_ray_intersect_plane(u, v, plane.normal, plane.d, fx, fy, cx, cy)

    quad = np.asarray(opening.quad_uv, dtype=np.float64)
    best_i = 0
    best_len = -1.0
    for i in range(4):
        j = (i + 1) % 4
        ll = float(np.linalg.norm(quad[j] - quad[i]))
        if ll > best_len:
            best_len = ll
            best_i = i
    p0 = quad[best_i]
    p1 = quad[(best_i + 1) % 4]
    edge_dir = _normalize(p1 - p0)
    half = 0.45 * best_len
    left_uv = opening.center_uv - edge_dir * half
    right_uv = opening.center_uv + edge_dir * half
    p_left = _pixel_ray_intersect_plane(left_uv[0], left_uv[1], plane.normal, plane.d, fx, fy, cx, cy)
    p_right = _pixel_ray_intersect_plane(right_uv[0], right_uv[1], plane.normal, plane.d, fx, fy, cx, cy)

    x_axis = _normalize(p_right - p_left)
    x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # X 正负号消歧：优先与参考方向同向
    if float(np.dot(x_axis, x_ref)) < 0.0:
        x_axis = -x_axis

    # 稳定策略：
    # 1) Y 由 top plane 法线主导（若可用）
    # 2) (0,-1,0) 仅用于 Y 正负号消歧
    # 3) 由 X 与 Y 回构稳定 Z
    y_ref = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    y_axis_candidate: np.ndarray | None = None
    if top_ref_normal is not None and np.linalg.norm(top_ref_normal) > 1e-9:
        y_axis_candidate = _normalize(np.asarray(top_ref_normal, dtype=np.float64))

    if y_axis_candidate is None:
        # top plane 法线缺失时，退回参考方向投影（仅此时使用）
        y_proj = y_ref - float(np.dot(y_ref, x_axis)) * x_axis
        if float(np.linalg.norm(y_proj)) < 1e-9:
            # 参考方向退化时，最后退回 near-plane 法线构造
            z_tmp = _normalize(plane.normal)
            y_axis = _normalize(np.cross(z_tmp, x_axis))
        else:
            y_axis = _normalize(y_proj)
    else:
        # 去掉 Y 在 X 上分量，保证正交
        y_axis = y_axis_candidate - float(np.dot(y_axis_candidate, x_axis)) * x_axis
        if float(np.linalg.norm(y_axis)) < 1e-9:
            z_tmp = _normalize(plane.normal)
            y_axis = _normalize(np.cross(z_tmp, x_axis))
        else:
            y_axis = _normalize(y_axis)

    # Y 正负号按参考方向消歧
    if float(np.dot(y_axis, y_ref)) < 0.0:
        y_axis = -y_axis

    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    # 重新确认 X 与参考同向（避免回构后符号翻转）
    if float(np.dot(x_axis, x_ref)) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis
        z_axis = _normalize(np.cross(x_axis, y_axis))

    rotation = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
    pre_grasp = grasp_point + z_axis * 80.0
    return GraspResult(grasp_point=grasp_point, pre_grasp_point=pre_grasp, rotation=rotation)


def _pixel_ray_intersect_plane(u: float, v: float, n: np.ndarray, d: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    ray = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=np.float64)
    ray = _normalize(ray)
    nn = _normalize(n)
    denom = float(np.dot(nn, ray))
    if abs(denom) < 1e-9:
        raise RuntimeError("射线与平面近平行")
    t = -float(d) / denom
    if t <= 0:
        raise RuntimeError(f"交点无效 t={t:.6f}")
    return t * ray


# endregion


# region 相机采集与可视化

def _capture_points_once(session: Gemini305, point_filter, max_depth_mm: float) -> tuple[np.ndarray | None, np.ndarray | None]:
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


def _update_result_3d(frame_mesh: o3d.geometry.TriangleMesh, grasp_line: o3d.geometry.LineSet, result: PipelineResult) -> None:
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
    opening: OpeningDetection,
    grasp: GraspResult,
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
        near_u8 = (near_m.astype(np.uint8) * 255)
        if np.count_nonzero(near_u8) > 0:
            ctn, _ = cv2.findContours(near_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, ctn, -1, (255, 170, 0), 1, cv2.LINE_AA)
            mx, my, mw, mh = cv2.boundingRect(np.vstack(ctn))
            cv2.putText(out, "Opening Plane", (mx, max(14, my - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 170, 0), 1, cv2.LINE_AA)
    if top_m is not None:
        out[top_m] = (0.55 * out[top_m] + 0.45 * np.array([0, 200, 0], dtype=np.float64)).astype(np.uint8)
        if top_plane_quad_uv is not None:
            tq = np.round(np.asarray(top_plane_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(out, [tq], True, (0, 255, 0), 1, cv2.LINE_AA)
            mx2, my2, mw2, mh2 = cv2.boundingRect(tq)
            cv2.putText(out, "Top Plane", (mx2, max(14, my2 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)
    quad = np.round(opening.quad_uv).astype(np.int32)
    cv2.polylines(out, [quad], True, (0, 0, 255), 1, cv2.LINE_AA)
    qx, qy, qw, qh = cv2.boundingRect(quad)
    cv2.putText(out, "Opening", (qx, max(14, qy - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
    u, v = int(round(opening.center_uv[0])), int(round(opening.center_uv[1]))
    cv2.circle(out, (u, v), 2, (0, 255, 0), -1)
    cv2.putText(out, "Center", (u + 4, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
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


def _build_contrast_preview(
    base_bgr: np.ndarray,
    near_plane_mask: np.ndarray | None,
    no_hole_mask: np.ndarray | None,
    opening: OpeningDetection | None,
) -> np.ndarray:
    """高反差保留预览：增强边缘与邻接边界可见性。"""
    highpass, gray, edge = _compute_high_contrast_domain(base_bgr)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out[edge > 0] = (255, 255, 255)
    if near_plane_mask is not None:
        c1, _ = cv2.findContours(near_plane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, c1, -1, (0, 165, 255), 1, cv2.LINE_AA)
    if no_hole_mask is not None:
        c2, _ = cv2.findContours(no_hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, c2, -1, (0, 255, 0), 1, cv2.LINE_AA)
    if opening is not None:
        q = np.round(opening.quad_uv).astype(np.int32)
        cv2.polylines(out, [q], True, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(out, "High-contrast retain + edges", (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
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


# endregion


# region 数学与通用工具

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise RuntimeError(f"向量归一化失败，norm={n}")
    return np.asarray(v, dtype=np.float64) / n


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


def _rasterize_rgb(xyz: np.ndarray, rgb: np.ndarray, fx: float, fy: float, cx: float, cy: float, w: int, h: int) -> np.ndarray:
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
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0], [0.0, 1.0, -10000.0]]))
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


if __name__ == "__main__":
    main()
