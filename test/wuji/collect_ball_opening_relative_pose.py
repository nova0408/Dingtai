from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_WUJI_ROOT = PROJECT_ROOT / "test" / "wuji"
if str(TEST_WUJI_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_WUJI_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6220"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "collect_ball_opening_relative_pose"
DEFAULT_TARGET_TRAY_INDEX = 0
DEFAULT_BALL_RADIUS_MM = 20.0
DEFAULT_COLOR_ORDER = ("#ff0000", "#ffff00", "#ff00ff")

@dataclass(frozen=True)
class BallRelativePoseCapture:
    """单次采集结果。"""

    frame_id: int
    opening_pose_transform: np.ndarray | None
    ball_pose_transform: np.ndarray | None
    relative_transform: np.ndarray | None
    opening_response: Any | None
    ball_detection_result: Any | None
    camera_intrinsics: np.ndarray | None
    overlay_image_path: Path | None
    error: str | None


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    target_tray_index: int = DEFAULT_TARGET_TRAY_INDEX,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    logger.info("启动多球相对位姿采集")
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = _compute_once(service_addr=str(service_addr), camera_name=str(camera_name), target_tray_index=int(target_tray_index))
    overlay_path = _save_overlay(output_dir, capture)
    _save_capture(output_dir, capture, overlay_path)
    if capture.error is not None:
        raise RuntimeError(capture.error)
    print(json.dumps(_serialize_capture(capture, overlay_path), ensure_ascii=False, indent=2))
    return 0


def _compute_once(service_addr: str, camera_name: str, target_tray_index: int) -> BallRelativePoseCapture:
    from camera_pipeline.opening_detection.protocol import OpeningDetectionPipelineRequest
    from camera_pipeline.opening_detection.transport import OpeningDetectionPipelineRpcClient, ZmqSocketOptions

    client = OpeningDetectionPipelineRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=30_000, send_timeout_ms=30_000),
    )
    try:
        response = client.call(
            OpeningDetectionPipelineRequest(
                request_id=1,
                camera_name=str(camera_name),
                frame_id=-1,
                target_tray_index=int(target_tray_index),
                enable_debug=True,
            )
        )
    finally:
        client.close()

    if response.error is not None:
        return BallRelativePoseCapture(0, None, None, None, response, None, None, None, str(response.error))
    if response.selected_result is None or response.selected_result.pose is None:
        return BallRelativePoseCapture(int(response.frame_id), None, None, None, response, None, None, None, "opening detection returned no pose result")
    if response.debug is None or response.debug.color_bgr is None or response.debug.depth_mm is None:
        return BallRelativePoseCapture(int(response.frame_id), None, None, None, response, None, None, None, "opening detection debug frame is missing")

    opening_pose_transform = _pose_to_transform(response.selected_result.pose.rotation, response.selected_result.pose.grasp_point_mm)
    ball_result = _detect_balls_on_remote_frame(
        color_bgr=np.asarray(response.debug.color_bgr, dtype=np.uint8),
        depth_mm=np.asarray(response.debug.depth_mm, dtype=np.float64),
        camera_intrinsics=response.debug.camera_intrinsics,
    )
    ball_pose_transform = _ball_pose_from_detections(ball_result)
    relative_transform = None
    if opening_pose_transform is not None and ball_pose_transform is not None:
        relative_transform = np.linalg.inv(ball_pose_transform) @ opening_pose_transform

    return BallRelativePoseCapture(
        frame_id=int(response.frame_id),
        opening_pose_transform=opening_pose_transform,
        ball_pose_transform=ball_pose_transform,
        relative_transform=relative_transform,
        opening_response=response,
        ball_detection_result=ball_result,
        camera_intrinsics=None if response.debug.camera_intrinsics is None else np.asarray(response.debug.camera_intrinsics, dtype=np.float64),
        overlay_image_path=None,
        error=None,
    )


def _detect_balls_on_remote_frame(color_bgr: np.ndarray, depth_mm: np.ndarray, camera_intrinsics: Any) -> Any:
    ball_runtime = _load_ball_pose_runtime()
    if camera_intrinsics is None:
        raise RuntimeError("camera intrinsics is missing")
    fx, fy, cx, cy = (float(camera_intrinsics[0]), float(camera_intrinsics[1]), float(camera_intrinsics[2]), float(camera_intrinsics[3]))
    priors = [
        ball_runtime.BallPosePrior(color_hex="#ff0000", radius_mm=DEFAULT_BALL_RADIUS_MM, model_center_mm=np.asarray([0.0, 0.0, 0.0], dtype=np.float64)),
        ball_runtime.BallPosePrior(color_hex="#ffff00", radius_mm=DEFAULT_BALL_RADIUS_MM, model_center_mm=np.asarray([1.0, 0.0, 0.0], dtype=np.float64)),
        ball_runtime.BallPosePrior(color_hex="#ff00ff", radius_mm=DEFAULT_BALL_RADIUS_MM, model_center_mm=np.asarray([0.0, 1.0, 0.0], dtype=np.float64)),
    ]
    pipeline = ball_runtime.BallPoseDetectionPipeline()
    frame = SimpleNamespace(
        color_bgr=np.asarray(color_bgr, dtype=np.uint8),
        depth_mm=np.asarray(depth_mm, dtype=np.float64),
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
    )
    return pipeline.detect(frame, priors)


def _load_ball_pose_runtime() -> Any:
    import importlib.util
    from types import ModuleType

    package_name = "_ball_pose_detection_runtime"
    package_dir = PROJECT_ROOT / "src" / "pointcloud" / "ball_pose_detection"
    package_module = ModuleType(package_name)
    package_module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package_module

    types_spec = importlib.util.spec_from_file_location(f"{package_name}.types", package_dir / "types.py")
    priors_spec = importlib.util.spec_from_file_location(f"{package_name}.priors", package_dir / "priors.py")
    pipeline_spec = importlib.util.spec_from_file_location(f"{package_name}.pipeline", package_dir / "pipeline.py")
    if (
        types_spec is None
        or types_spec.loader is None
        or priors_spec is None
        or priors_spec.loader is None
        or pipeline_spec is None
        or pipeline_spec.loader is None
    ):
        raise RuntimeError("failed to load ball pose runtime")

    types_mod = importlib.util.module_from_spec(types_spec)
    priors_mod = importlib.util.module_from_spec(priors_spec)
    pipeline_mod = importlib.util.module_from_spec(pipeline_spec)
    sys.modules[f"{package_name}.types"] = types_mod
    sys.modules[f"{package_name}.priors"] = priors_mod
    sys.modules[f"{package_name}.pipeline"] = pipeline_mod
    types_spec.loader.exec_module(types_mod)
    priors_spec.loader.exec_module(priors_mod)
    pipeline_spec.loader.exec_module(pipeline_mod)
    return type(
        "BallPoseRuntime",
        (),
        {
            "BallPosePrior": priors_mod.BallPosePrior,
            "BallPoseDetectionPipeline": pipeline_mod.BallPoseDetectionPipeline,
        },
    )()


def _ball_pose_from_detections(result: Any) -> np.ndarray | None:
    centers = {
        item.color_hex: np.asarray(item.center_mm, dtype=np.float64)
        for item in result.detections
        if item.center_mm is not None and item.color_hex in DEFAULT_COLOR_ORDER
    }
    if not all(name in centers for name in DEFAULT_COLOR_ORDER):
        return None
    red = centers["#ff0000"]
    yellow = centers["#ffff00"]
    purple = centers["#ff00ff"]
    x_axis = yellow - red
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        return None
    x_axis = x_axis / x_norm
    plane_hint = purple - red
    z_axis = np.cross(x_axis, plane_hint)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6:
        return None
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-6:
        return None
    y_axis = y_axis / y_norm
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = red
    return transform


def _pose_to_transform(rotation: Any, translation_mm: Any) -> np.ndarray | None:
    if rotation is None or translation_mm is None:
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(translation_mm, dtype=np.float64)
    return transform


def _save_capture(output_dir: Path, capture: BallRelativePoseCapture, overlay_path: Path | None) -> None:
    payload = _serialize_capture(capture, overlay_path)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_overlay(output_dir: Path, capture: BallRelativePoseCapture) -> Path | None:
    if capture.opening_response is None or capture.opening_response.debug is None:
        return None
    if capture.opening_response.debug.color_bgr is None:
        return None
    overlay = np.asarray(capture.opening_response.debug.color_bgr, dtype=np.uint8).copy()
    ball_result = capture.ball_detection_result
    if ball_result is not None:
        _draw_ball_overlay(overlay, ball_result)
    _draw_opening_overlay(overlay, capture)
    overlay_path = output_dir / f"frame_{capture.frame_id:06d}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    return overlay_path


def _draw_ball_overlay(image_bgr: np.ndarray, result: Any) -> None:
    for item in result.detections:
        base_color = np.asarray(item.debug_bgr, dtype=np.uint8)
        contour_color = tuple(int(value) for value in base_color.tolist())
        fitted_color = tuple(int(value) for value in np.clip(base_color.astype(np.int16) * 0.55, 0, 255).tolist())
        if item.contour is not None:
            cv2.drawContours(image_bgr, [np.asarray(item.contour, dtype=np.int32)], -1, contour_color, 2)
        if item.center_px is not None:
            center = tuple(int(round(value)) for value in np.asarray(item.center_px, dtype=np.float64).tolist())
            cv2.circle(image_bgr, center, max(4, int(round(float(item.radius_px)))), fitted_color, 2)
            label = f"{item.color_hex}:{item.status}"
            cv2.putText(
                image_bgr,
                label,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                contour_color,
                2,
                cv2.LINE_AA,
            )
        if item.center_mm is not None and item.center_px is not None:
            norm_text = "" if item.center_norm is None else f" norm={np.asarray(item.center_norm, dtype=np.float64).round(1).tolist()}"
            mm_text = (
                f"{item.color_hex} r_px={float(item.radius_px):.1f} "
                f"r_mm={float(item.radius_mm):.1f}{norm_text} "
                f"{np.asarray(item.center_mm, dtype=np.float64).round(1).tolist()}"
            )
            center = tuple(int(round(value)) for value in np.asarray(item.center_px, dtype=np.float64).tolist())
            cv2.putText(
                image_bgr,
                mm_text,
                (center[0] + 8, center[1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                fitted_color,
                1,
                cv2.LINE_AA,
            )


def _draw_opening_overlay(image_bgr: np.ndarray, capture: BallRelativePoseCapture) -> None:
    if capture.opening_response is None or capture.opening_response.selected_result is None:
        return
    opening = capture.opening_response.selected_result
    if opening.opening_quad_uv is None:
        return
    quad = np.asarray(opening.opening_quad_uv, dtype=np.float64)
    if quad.shape != (4, 2):
        return
    quad_i32 = np.round(quad).astype(np.int32)
    cv2.polylines(image_bgr, [quad_i32], True, (0, 255, 255), 2, cv2.LINE_AA)
    center = np.mean(quad, axis=0)
    center_xy = tuple(int(round(v)) for v in center.tolist())
    cv2.drawMarker(image_bgr, center_xy, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(image_bgr, "opening", (center_xy[0] + 8, center_xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def _serialize_capture(capture: BallRelativePoseCapture, overlay_path: Path | None) -> dict[str, Any]:
    ballinfo = _serialize_ballinfo(capture.ball_detection_result)
    return {
        "frame_id": capture.frame_id,
        "camera": {
            "intrinsics": None if capture.camera_intrinsics is None else capture.camera_intrinsics.tolist(),
        },
        "balls": {
            "ballinfo": ballinfo,
        },
        "pose": {
            "opening_pose_camera_frame": None if capture.opening_pose_transform is None else capture.opening_pose_transform.tolist(),
            "ball_pose_camera_frame": None if capture.ball_pose_transform is None else capture.ball_pose_transform.tolist(),
            "relative_transform": None if capture.relative_transform is None else capture.relative_transform.tolist(),
        },
        "overlay_image_path": None if overlay_path is None else str(overlay_path),
        "error": capture.error,
    }


def _serialize_ball_priors(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    priors: list[dict[str, Any]] = []
    for item in result.detections:
        priors.append(
            {
                "color_hex": item.color_hex,
                "radius_mm": float(item.radius_mm),
                "position_camera_mm": None if item.center_mm is None else np.asarray(item.center_mm, dtype=np.float64).tolist(),
                "detected": bool(item.detected),
                "status": item.status,
            }
        )
    return priors


def _serialize_ballinfo(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    ordered = []
    for color_hex in DEFAULT_COLOR_ORDER:
        item = next((it for it in result.detections if it.color_hex == color_hex), None)
        if item is None:
            continue
        ordered.append(
            {
                "color_hex": item.color_hex,
                "radius_mm": float(item.radius_mm),
                "position_camera_mm": None if item.center_mm is None else np.asarray(item.center_mm, dtype=np.float64).tolist(),
                "detected": bool(item.detected),
                "status": item.status,
            }
        )
    return ordered


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集 opening pose 与多球相对位姿")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--target-tray-index", type=int, default=DEFAULT_TARGET_TRAY_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(main(service_addr=str(args.service_addr), camera_name=str(args.camera_name), target_tray_index=int(args.target_tray_index), output_dir=Path(args.output_dir)))
