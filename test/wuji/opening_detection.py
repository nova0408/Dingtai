from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6220"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "opening_detection_capture"
DEFAULT_TARGET_TRAY_INDEX = 0

from camera_pipeline.opening_detection.protocol import OpeningDetectionPipelineRequest  # noqa: E402
from camera_pipeline.opening_detection.transport import OpeningDetectionPipelineRpcClient, ZmqSocketOptions  # noqa: E402


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_tray_index: int = DEFAULT_TARGET_TRAY_INDEX,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("opening_detection smoke test start")
    client = OpeningDetectionPipelineRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=30_000, send_timeout_ms=30_000),
    )
    response = None
    try:
        deadline = time.monotonic() + 60.0
        last_error: str | None = None
        request_id = 1
        while time.monotonic() < deadline:
            response = client.call(
                OpeningDetectionPipelineRequest(
                    request_id=request_id,
                    camera_name=str(camera_name),
                    frame_id=-1,
                    target_tray_index=int(target_tray_index),
                    enable_debug=True,
                )
            )
            request_id += 1
            if response.error is None and response.selected_result is not None and response.selected_result.pose is not None:
                break
            last_error = response.error or "opening detection returned no pose result"
            time.sleep(1.0)
        else:
            raise RuntimeError(last_error or "opening detection smoke test timed out")
    finally:
        client.close()
    if response is None:
        raise RuntimeError("opening detection returned no response")
    _save_capture(output_dir, response)
    print(_format_summary(response))
    return 0


def _save_capture(output_dir: Path, response: Any) -> None:
    selected = None if response.selected_result is None else {
        "tray_index": response.selected_result.tray_index,
        "tray_bbox_xywh": list(response.selected_result.tray_bbox_xywh),
        "tray_center_uv": list(response.selected_result.tray_center_uv),
        "opening_center_uv": None if response.selected_result.opening_center_uv is None else list(response.selected_result.opening_center_uv),
        "opening_quad_uv": None if response.selected_result.opening_quad_uv is None else [list(item) for item in response.selected_result.opening_quad_uv],
        "top_quad_uv": None if response.selected_result.top_quad_uv is None else [list(item) for item in response.selected_result.top_quad_uv],
        "pose": None if response.selected_result.pose is None else {
            "grasp_point_mm": list(response.selected_result.pose.grasp_point_mm),
            "pre_grasp_point_mm": list(response.selected_result.pose.pre_grasp_point_mm),
            "rotation": None if response.selected_result.pose.rotation is None else [list(row) for row in response.selected_result.pose.rotation],
            "rpy_deg": None if response.selected_result.pose.rpy_deg is None else list(response.selected_result.pose.rpy_deg),
        },
    }
    payload = {
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "tray_count": response.tray_count,
        "selected_tray_index": response.selected_tray_index,
        "elapsed_ms": response.elapsed_ms,
        "error": response.error,
        "selected_result": selected,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if response.debug is not None and response.debug.color_bgr is not None:
        cv2.imwrite(str(output_dir / "color_bgr.jpg"), np.asarray(response.debug.color_bgr, dtype=np.uint8))
    if response.debug is not None and response.debug.depth_mm is not None:
        depth_vis = _build_depth_view(np.asarray(response.debug.depth_mm))
        cv2.imwrite(str(output_dir / "depth.jpg"), depth_vis)
    if response.debug is not None and response.debug.overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "overlay.jpg"), np.asarray(response.debug.overlay_bgr, dtype=np.uint8))
    if response.debug is not None and response.debug.contrast_bgr is not None:
        cv2.imwrite(str(output_dir / "contrast.jpg"), np.asarray(response.debug.contrast_bgr, dtype=np.uint8))
    if response.debug is not None and response.debug.selected_tray_mask is not None:
        cv2.imwrite(str(output_dir / "selected_tray_mask.png"), np.asarray(response.debug.selected_tray_mask, dtype=np.uint8))
    if response.debug is not None and response.debug.near_plane_mask is not None:
        cv2.imwrite(str(output_dir / "near_plane_mask.png"), np.asarray(response.debug.near_plane_mask, dtype=np.uint8))
    if response.debug is not None and response.debug.no_hole_mask is not None:
        cv2.imwrite(str(output_dir / "no_hole_mask.png"), np.asarray(response.debug.no_hole_mask, dtype=np.uint8))
    if response.debug is not None and response.debug.selected_tray_mask is not None:
        cv2.imwrite(str(output_dir / "selected_tray_mask_overlay.jpg"), _build_mask_overlay(response.debug.color_bgr, response.debug.selected_tray_mask))


def _format_summary(response: Any) -> str:
    pose = response.selected_result.pose
    rpy_deg = _rotation_matrix_to_rpy_deg(pose.rotation)
    summary = {
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "tray_count": response.tray_count,
        "selected_tray_index": response.selected_tray_index,
        "elapsed_ms": response.elapsed_ms,
        "error": response.error,
        "opening_center_uv": None if response.selected_result is None or response.selected_result.opening_center_uv is None else [float(v) for v in response.selected_result.opening_center_uv],
        "opening_quad_uv": None if response.selected_result is None or response.selected_result.opening_quad_uv is None else [[float(v) for v in pt] for pt in response.selected_result.opening_quad_uv],
        "top_quad_uv": None if response.selected_result is None or response.selected_result.top_quad_uv is None else [[float(v) for v in pt] for pt in response.selected_result.top_quad_uv],
        "grasp_point_mm": [float(v) for v in pose.grasp_point_mm],
        "pre_grasp_point_mm": [float(v) for v in pose.pre_grasp_point_mm],
        "rotation": None if pose.rotation is None else [[float(v) for v in row] for row in pose.rotation],
        "rpy_deg": [float(v) for v in rpy_deg],
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _build_mask_overlay(color_bgr: Any, mask: Any) -> np.ndarray:
    base = np.asarray(color_bgr, dtype=np.uint8).copy()
    mask_u8 = np.asarray(mask, dtype=np.uint8) > 0
    base[mask_u8] = (base[mask_u8] * 0.55 + np.array([0, 160, 255], dtype=np.float32) * 0.45).astype(np.uint8)
    return base


def _build_depth_view(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1.0)
    hsv = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if np.any(valid):
        z_min = float(np.percentile(depth[valid], 2))
        z_max = float(np.percentile(depth[valid], 98))
        norm = np.clip((depth - z_min) / max(1e-6, z_max - z_min), 0.0, 1.0)
        hsv[..., 0] = np.where(valid, np.rint((1.0 - norm) * 120.0), 0).astype(np.uint8)
        hsv[..., 1] = np.where(valid, 255, 0).astype(np.uint8)
        hsv[..., 2] = np.where(valid, 255, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _rgbd_to_points(depth_mm: np.ndarray, color_bgr: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_mm.shape[:2]
    v, u = np.indices((h, w))
    z = np.asarray(depth_mm, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1.0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    rgb = np.asarray(color_bgr, dtype=np.float64) / 255.0
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1)
    colors = np.stack([rgb[..., 0][valid], rgb[..., 1][valid], rgb[..., 2][valid]], axis=1)
    return pts, colors


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


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation, dtype=np.float64)
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


def _load_local_compare_module() -> Any:
    module_name = "test.pointcloud.opening_detection.opening_detection_local"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = PROJECT_ROOT / "test" / "pointcloud" / "opening_detection" / "opening_detection_local.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
