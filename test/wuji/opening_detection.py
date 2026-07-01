from __future__ import annotations

import argparse
import json
import sys
import importlib
import importlib.util
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_SERVICE_ADDR = "tcp://127.0.0.1:6220"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "opening_detection_capture"
DEFAULT_TARGET_TRAY_INDEX = 0

if (PROJECT_ROOT / "orin").is_dir():
    from orin.opening_detection_pipeline.protocol import OpeningDetectionPipelineRequest
    from orin.opening_detection_pipeline.transport import OpeningDetectionPipelineRpcClient, ZmqSocketOptions
else:
    from camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig
    from tray_detection.engine import OrinTrayDetectionExecutor, OrinTrayDetectionExecutorConfig
    from tray_detection.protocol import OrinTrayDetectionRequest


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_tray_index: int = DEFAULT_TARGET_TRAY_INDEX,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (PROJECT_ROOT / "orin").is_dir():
        return _run_service_mode(service_addr=service_addr, camera_name=camera_name, output_dir=output_dir, target_tray_index=target_tray_index)
    _ensure_orin_runtime_aliases()
    opening_engine = _load_orin_opening_engine()
    return _run_library_mode(
        camera_name=camera_name,
        output_dir=output_dir,
        target_tray_index=target_tray_index,
        opening_engine=opening_engine,
    )


def _run_service_mode(service_addr: str, camera_name: str, output_dir: Path, target_tray_index: int) -> int:
    logger.info("opening_detection service mode start")
    client = OpeningDetectionPipelineRpcClient(connect_addr=str(service_addr), options=ZmqSocketOptions(recv_timeout_ms=30_000, send_timeout_ms=30_000))
    try:
        response = client.call(OpeningDetectionPipelineRequest(request_id=1, camera_name=str(camera_name), frame_id=-1, target_tray_index=int(target_tray_index), enable_debug=True))
    finally:
        client.close()
    _save_opening_capture(output_dir, response)
    print(_summary_from_response(response))
    return 0


def _run_library_mode(camera_name: str, output_dir: Path, target_tray_index: int, opening_engine: Any) -> int:
    logger.info("opening_detection library mode start")
    runtime = CameraStreamRuntime(CameraStreamRuntimeConfig(camera_name=str(camera_name)))
    runtime.start()
    try:
        if not runtime.wait_until_ready(timeout_s=8.0):
            raise RuntimeError("camera stream not ready")
        tray_executor = OrinTrayDetectionExecutor(frame_runtime=runtime, config=OrinTrayDetectionExecutorConfig())
        opening_detector = opening_engine.OpeningDetector()
        pose_estimator = opening_engine.GraspPoseEstimator()
        temporal_state = opening_engine.TemporalFilterState()
        tray_request = OrinTrayDetectionRequest(request_id=1, camera_name=str(camera_name), frame_id=-1, enable_debug=True)
        tray_response = tray_executor.process_request(tray_request)
        print(f"[opening] tray frame_id={tray_response.frame_id} tray_count={tray_response.tray_count} error={tray_response.error}")
        for tray in tray_response.tray_results:
            print(f"[opening] tray_id={tray.tray_id} bbox={tray.bbox_xywh} center_uv={tray.center_uv} conf={tray.confidence_2d:.3f}")
        if tray_response.debug is None or len(tray_response.debug.tray_masks) == 0:
            raise RuntimeError("tray masks unavailable")
        if int(target_tray_index) >= len(tray_response.tray_results):
            raise RuntimeError("target tray index out of range")
        target_tray = tray_response.tray_results[int(target_tray_index)]
        target_mask = np.asarray(tray_response.debug.tray_masks[int(target_tray_index)], dtype=np.uint8)
        print(f"[opening] target tray_index={target_tray.tray_id} mask_pixels={int(np.count_nonzero(target_mask))}")
        frame = runtime.get_frame_by_id(int(tray_response.frame_id)) or runtime.get_latest_frame()
        if frame is None:
            raise RuntimeError("frame not ready for pose stage")
        frame = runtime.get_frame_by_id(int(tray_response.frame_id)) or runtime.get_latest_frame()
        if frame is None:
            raise RuntimeError("frame not ready for pose stage")
        color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
        depth_mm = np.asarray(frame.depth_mm, dtype=np.float32)
        hp_bgr, hp_gray, hp_edge = _compute_high_contrast_domain(color_bgr)
        opening_debug = _debug_detect_opening(opening_detector, color_bgr, target_mask, hp_gray)
        print(f"[opening] roi_bbox={opening_debug['roi_bbox']} roi_tray_pixels={opening_debug['roi_tray_pixels']} thresholds={opening_debug['thresholds']}")
        print(f"[opening] reject_counters={opening_debug['reject_counters']}")
        print(f"[opening] candidate_summary={opening_debug['candidate_summary']}")
        print(f"[opening] closest_candidates={opening_debug['closest_candidates'][:8]}")
        opening = opening_debug["opening"]
        if opening is None:
            raise RuntimeError("未检测到开口")
        near_plane_mask, no_hole_mask = opening_detector.compute_mask_pipeline(target_mask, opening, hp_gray, hp_edge)
        tray_mask_result = type(
            "TrayMaskResult",
            (),
            {
                "tray_mask": target_mask,
                "tray_detect_ok": True,
                "near_plane_mask": near_plane_mask,
                "no_hole_mask": no_hole_mask,
                "top_quad_uv": None if no_hole_mask is None else opening_detector.fit_rotated_quad(no_hole_mask),
            },
        )()
        print(f"[opening] opening center_uv={opening.center_uv.tolist()} bbox={opening.bbox_xywh} score={opening.score:.3f}")
        print(f"[opening] tray_mask_pixels={int(np.count_nonzero(tray_mask_result.tray_mask))} near_plane_pixels={0 if tray_mask_result.near_plane_mask is None else int(np.count_nonzero(tray_mask_result.near_plane_mask))} no_hole_pixels={0 if tray_mask_result.no_hole_mask is None else int(np.count_nonzero(tray_mask_result.no_hole_mask))}")
        xyzrgb = _rgbd_to_points(depth_mm, color_bgr, float(frame.fx), float(frame.fy), float(frame.cx), float(frame.cy))
        xyz = np.asarray(xyzrgb[:, :3], dtype=np.float64)
        rgb = np.asarray(xyzrgb[:, 3:], dtype=np.float64)
        uv, valid = _project_points_to_image(xyz, float(frame.fx), float(frame.fy), float(frame.cx), float(frame.cy), int(color_bgr.shape[1]), int(color_bgr.shape[0]))
        xyz_local = opening_detector.filter_opening_local_points(xyz, rgb, opening, int(color_bgr.shape[1]), int(color_bgr.shape[0]), uv, valid)
        print(f"[opening] xyz_local_count={int(xyz_local.shape[0])}")
        if xyz_local.shape[0] > 0:
            plane = pose_estimator.estimate_plane(xyz_local)
            print(f"[opening] plane_normal={plane.normal.tolist()} plane_d={float(plane.d):.3f}")
            top_normal = opening_detector.estimate_top_plane_normal(xyz, tray_mask_result.no_hole_mask, uv, valid)
            top_normal = pose_estimator.stabilize_top_normal(top_normal, temporal_state)
            print(f"[opening] top_normal={None if top_normal is None else top_normal.tolist()}")
            grasp = pose_estimator.compute_grasp(opening, plane, type("Intrinsics", (), {"fx": frame.fx, "fy": frame.fy, "cx": frame.cx, "cy": frame.cy})(), top_normal)
            grasp = pose_estimator.stabilize_grasp_result(grasp, temporal_state)
            print(f"[opening] grasp_point_mm={grasp.grasp_point.tolist() if grasp is not None else None}")
            print(f"[opening] pre_grasp_point_mm={grasp.pre_grasp_point.tolist() if grasp is not None else None}")
            print(f"[opening] rotation={None if grasp is None else grasp.rotation.tolist()}")
        else:
            print("[opening] skip pose stage because xyz_local_count=0")
            grasp = None
        direct_response = types.SimpleNamespace(
            frame_id=int(tray_response.frame_id),
            camera_name=str(tray_response.camera_name),
            tray_count=int(tray_response.tray_count),
            selected_tray_index=int(target_tray.tray_id),
            elapsed_ms=float(tray_response.elapsed_ms),
            error=None,
            selected_result=types.SimpleNamespace(
                tray_index=int(target_tray.tray_id),
                pose=None
                if grasp is None
                else types.SimpleNamespace(
                    grasp_point_mm=tuple(float(v) for v in grasp.grasp_point),
                    pre_grasp_point_mm=tuple(float(v) for v in grasp.pre_grasp_point),
                    rpy_deg=(0.0, 0.0, 0.0),
                ),
            ),
            all_tray_results=tuple(),
            debug=types.SimpleNamespace(
                overlay_bgr=None if tray_response.debug is None else tray_response.debug.overlay_bgr,
                contrast_bgr=None if tray_response.debug is None else tray_response.debug.mask_bgr,
                selected_tray_mask=target_mask,
                near_plane_mask=near_plane_mask,
                no_hole_mask=no_hole_mask,
            ),
        )
        _save_opening_capture(output_dir, direct_response)
    finally:
        runtime.stop()
    return 0


def _save_opening_capture(output_dir: Path, response: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_from_response(response)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if response is None or response.debug is None:
        return
    if response.debug.overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "overlay.jpg"), np.asarray(response.debug.overlay_bgr, dtype=np.uint8))
    if response.debug.contrast_bgr is not None:
        cv2.imwrite(str(output_dir / "contrast.jpg"), np.asarray(response.debug.contrast_bgr, dtype=np.uint8))
    if response.debug.selected_tray_mask is not None:
        cv2.imwrite(str(output_dir / "selected_tray_mask.png"), np.asarray(response.debug.selected_tray_mask, dtype=np.uint8))
    if response.debug.near_plane_mask is not None:
        cv2.imwrite(str(output_dir / "near_plane_mask.png"), np.asarray(response.debug.near_plane_mask, dtype=np.uint8))
    if response.debug.no_hole_mask is not None:
        cv2.imwrite(str(output_dir / "no_hole_mask.png"), np.asarray(response.debug.no_hole_mask, dtype=np.uint8))


def _summary_from_response(response: Any) -> dict[str, Any]:
    selected = None
    if response is not None and getattr(response, "selected_result", None) is not None:
        pose = response.selected_result.pose
        selected = {
            "tray_index": response.selected_result.tray_index,
            "pose": None if pose is None else {
                "grasp_point_mm": list(pose.grasp_point_mm),
                "pre_grasp_point_mm": list(pose.pre_grasp_point_mm),
                "rpy_deg": list(pose.rpy_deg),
            },
        }
    return {
        "frame_id": None if response is None else response.frame_id,
        "camera_name": None if response is None else response.camera_name,
        "tray_count": 0 if response is None else response.tray_count,
        "selected_tray_index": None if response is None else response.selected_tray_index,
        "elapsed_ms": None if response is None else response.elapsed_ms,
        "error": None if response is None else response.error,
        "selected_result": selected,
        "all_tray_count": 0 if response is None else len(response.all_tray_results),
        "has_debug": bool(response is not None and response.debug is not None),
    }


def _ensure_orin_runtime_aliases() -> None:
    if "orin" not in sys.modules:
        orin_pkg = types.ModuleType("orin")
        orin_pkg.__path__ = [str(PROJECT_ROOT)]  # type: ignore[attr-defined]
        sys.modules["orin"] = orin_pkg
    camera_stream_mod = importlib.import_module("camera_stream")
    sys.modules["orin.camera_stream"] = camera_stream_mod
    sys.modules["orin.opening_detection.camera_stream"] = camera_stream_mod
    sys.modules["orin.tray_detection"] = importlib.import_module("tray_detection")


def _load_orin_opening_engine():
    module_name = "orin.opening_detection.engine"
    if module_name in sys.modules:
        return sys.modules[module_name]
    package_dir = PROJECT_ROOT / "opening_detection"
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "engine.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create opening_detection engine spec")
    package_name = "orin.opening_detection"
    if package_name not in sys.modules:
        package_mod = types.ModuleType(package_name)
        package_mod.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package_mod
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compute_high_contrast_domain(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blur = cv2.GaussianBlur(bgr, (0, 0), 2.6)
    highpass = cv2.addWeighted(bgr, 1.90, blur, -0.90, 0.0)
    gray = cv2.cvtColor(highpass, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.bilateralFilter(gray, d=7, sigmaColor=42, sigmaSpace=42)
    gray_f = cv2.morphologyEx(gray_f, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge = cv2.Canny(gray_f, 42, 118)
    return highpass, gray_f, edge


def _debug_detect_opening(opening_detector: Any, rgb_bgr: np.ndarray, tray_mask: np.ndarray, hp_gray: np.ndarray) -> dict[str, Any]:
    height, width = rgb_bgr.shape[:2]
    tx, ty, tw, th = _mask_bbox_xywh(tray_mask)
    x1 = int(max(0, tx + 0.08 * tw))
    x2 = int(min(width - 1, tx + 0.92 * tw))
    y1 = int(max(0, ty + 0.72 * th))
    y2 = int(min(height - 1, ty + 0.97 * th))
    roi = rgb_bgr[y1:y2, x1:x2]
    roi_tray = tray_mask[y1:y2, x1:x2]
    if roi.size == 0 or np.count_nonzero(roi_tray) < 50:
        return {"roi_bbox": (x1, y1, x2, y2), "roi_tray_pixels": int(np.count_nonzero(roi_tray)), "thresholds": [], "candidate_summary": [], "opening": None}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hp = hp_gray[y1:y2, x1:x2]
    thresholds = sorted(set(int(np.clip(t, 20, 180)) for t in np.percentile(gray_blur, [4, 6, 8, 12, 16, 20, 25, 30])))
    candidate_summary = []
    opening = None
    best_score = -1e18
    reject_counters = {
        "area": 0,
        "aspect": 0,
        "ratio": 0,
        "y_pref": 0,
        "in_roi": 0,
        "thresholds": 0,
    }
    closest_candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        mask = np.zeros_like(gray_blur, dtype=np.uint8)
        mask[(gray_blur <= threshold) & (roi_tray > 0)] = 255
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, gray_blur.shape[1] // 12), 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [float(cv2.contourArea(contour)) for contour in contours]
        max_area = max(contour_areas) if contour_areas else 0.0
        max_aspect = 0.0
        best_contour_info = None
        if contours:
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                _center, (rw, rh), _angle = rect
                long_side = max(float(rw), float(rh))
                short_side = max(1.0, min(float(rw), float(rh)))
                max_aspect = max(max_aspect, float(long_side / short_side))
                box = cv2.boxPoints(rect)
                candidate = {
                    "thr": int(threshold),
                    "area": float(cv2.contourArea(contour)),
                    "aspect": float(long_side / short_side),
                    "wr": float(long_side / max(1.0, float(tw))),
                    "hr": float(short_side / max(1.0, float(th))),
                    "y_pref": float((y1 + _center[1] - ty) / max(1.0, float(th))),
                    "bbox": tuple(int(v) for v in cv2.boundingRect(np.round(box).astype(np.int32))),
                }
                if best_contour_info is None or candidate["area"] > best_contour_info["area"]:
                    best_contour_info = candidate
        local_candidates = 0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 40.0:
                reject_counters["area"] += 1
                continue
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), _angle = rect
            long_side = max(float(rw), float(rh))
            short_side = max(1.0, min(float(rw), float(rh)))
            aspect = long_side / short_side
            if not (2.0 <= aspect <= 48.0):
                reject_counters["aspect"] += 1
                continue
            wr = long_side / max(1.0, float(tw))
            hr = short_side / max(1.0, float(th))
            if not (0.05 <= wr <= 0.80 and 0.005 <= hr <= 0.16):
                reject_counters["ratio"] += 1
                continue
            y_pref = (y1 + cy - ty) / max(1.0, float(th))
            if y_pref < 0.60:
                reject_counters["y_pref"] += 1
                continue
            box = cv2.boxPoints(rect)
            patch_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.fillConvexPoly(patch_mask, np.round(box).astype(np.int32), 255)
            in_roi = (patch_mask > 0) & (roi_tray > 0)
            if np.count_nonzero(in_roi) < 20:
                reject_counters["in_roi"] += 1
                continue
            patch_raw = gray[in_roi]
            patch_hp = hp[in_roi]
            dark_ratio_raw = float(np.mean(patch_raw <= threshold))
            dark_ratio_hp = float(np.mean(patch_hp <= np.percentile(hp[roi_tray > 0], 16)))
            ring = _patch_ring(gray_blur, int(cx - rw / 2), int(cy - rh / 2), int(max(1, rw)), int(max(1, rh)))
            slot_mean = float(np.mean(patch_raw))
            ring_mean = float(np.mean(ring)) if ring.size > 0 else slot_mean
            contrast_score = float(np.clip((ring_mean - slot_mean) / 45.0, 0.0, 1.5))
            x_center_pref = 1.0 - min(abs((x1 + cx) - (tx + 0.5 * tw)) / max(1.0, 0.5 * tw), 1.0)
            score = 2.4 * dark_ratio_raw + 1.5 * dark_ratio_hp + 1.5 * min(aspect / 8.0, 2.0) + 1.1 * contrast_score + 1.0 * x_center_pref + 0.7 * y_pref
            box[:, 0] += x1
            box[:, 1] += y1
            candidate_summary.append(
                {
                    "thr": int(threshold),
                    "score": float(score),
                    "bbox": tuple(int(v) for v in cv2.boundingRect(np.round(box).astype(np.int32))),
                    "aspect": float(aspect),
                    "dark_ratio_raw": float(dark_ratio_raw),
                    "dark_ratio_hp": float(dark_ratio_hp),
                    "contrast_score": float(contrast_score),
                }
            )
            local_candidates += 1
            if float(score) > best_score:
                best_score = float(score)
                opening = type("Opening", (), {"center_uv": np.mean(box, axis=0).astype(np.float64), "bbox_xywh": cv2.boundingRect(np.round(box).astype(np.int32)), "quad_uv": box.astype(np.float64), "score": float(score)})()
        candidate_summary.append(
            {
                "thr": int(threshold),
                "contours": int(len(contours)),
                "max_area": float(max_area),
                "max_aspect": float(max_aspect),
                "local_candidates": int(local_candidates),
            }
        )
        if best_contour_info is not None:
            closest_candidates.append(best_contour_info)
    return {
        "roi_bbox": (x1, y1, x2, y2),
        "roi_tray_pixels": int(np.count_nonzero(roi_tray)),
        "thresholds": thresholds,
        "candidate_summary": candidate_summary,
        "reject_counters": reject_counters,
        "closest_candidates": closest_candidates,
        "opening": opening,
    }


def _rgbd_to_points(depth_mm: np.ndarray, color_bgr: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth_mm.shape[:2]
    v, u = np.indices((h, w))
    z = np.asarray(depth_mm, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1.0) & (z < 5000.0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    rgb = color_bgr.astype(np.float64) / 255.0
    pts = np.stack([x[valid], y[valid], z[valid], rgb[..., 0][valid], rgb[..., 1][valid], rgb[..., 2][valid]], axis=1)
    return pts


def _project_points_to_image(xyz: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    z = xyz[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)
    if np.any(valid):
        uu = np.rint(xyz[valid, 0] * fx / z[valid] + cx).astype(np.int32)
        vv = np.rint(xyz[valid, 1] * fy / z[valid] + cy).astype(np.int32)
        in_bounds = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        idx = np.where(valid)[0][in_bounds]
        u[idx] = uu[in_bounds]
        v[idx] = vv[in_bounds]
    return np.stack([u, v], axis=1), (u >= 0) & (v >= 0)


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if xs.size == 0:
        height, width = mask.shape[:2]
        return 0, 0, int(width), int(height)
    x1 = int(np.min(xs))
    x2 = int(np.max(xs))
    y1 = int(np.min(ys))
    y2 = int(np.max(ys))
    return x1, y1, x2 - x1 + 1, y2 - y1 + 1


def _patch_ring(gray: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    image_h, image_w = gray.shape[:2]
    pad_x = max(3, int(round(0.35 * width)))
    pad_y = max(2, int(round(0.90 * height)))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_w, x + width + pad_x)
    y2 = min(image_h, y + height + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0,), dtype=np.uint8)
    outer = gray[y1:y2, x1:x2]
    inner = np.zeros_like(outer, dtype=np.uint8)
    ix1 = x - x1
    iy1 = y - y1
    ix2 = min(ix1 + width, outer.shape[1])
    iy2 = min(iy1 + height, outer.shape[0])
    if ix2 > ix1 and iy2 > iy1:
        inner[iy1:iy2, ix1:ix2] = 255
    return outer[inner == 0]


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="opening detection smoke test")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-tray-index", type=int, default=DEFAULT_TARGET_TRAY_INDEX)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            output_dir=Path(args.output_dir),
            target_tray_index=int(args.target_tray_index),
        )
    )
