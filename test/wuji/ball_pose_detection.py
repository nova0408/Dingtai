from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6230"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture"
DEFAULT_PRIOR_CAPTURE_PATH = PROJECT_ROOT / "test" / "wuji" / ".archive" / "collect_ball_opening_relative_pose" / "summary.json"

from camera_pipeline.ball_pose_detection.protocol import BallPoseDetectionRequest, BallPosePriorInfo  # noqa: E402
from camera_pipeline.ball_pose_detection.transport import BallPoseDetectionRpcClient, ZmqSocketOptions  # noqa: E402


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prior_capture_path: Path = DEFAULT_PRIOR_CAPTURE_PATH,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ball_pose_detection smoke test start")
    prior_capture = _load_prior_capture(prior_capture_path)
    priors = _build_priors_from_capture(prior_capture)
    reference_relative_transform = _load_reference_relative_transform(prior_capture)
    client = BallPoseDetectionRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=30_000, send_timeout_ms=30_000),
    )
    try:
        response = client.call(
            BallPoseDetectionRequest(
                request_id=1,
                camera_name=str(camera_name),
                frame_id=-1,
                enable_debug=True,
                priors=tuple(priors),
                reference_relative_transform_mm=reference_relative_transform,
            )
        )
    finally:
        client.close()
    if response.error is not None:
        raise RuntimeError(response.error)
    if response.matched_count < 3 or response.pose_transform is None:
        raise RuntimeError("ball pose detection returned insufficient pose result")
    _save_capture(output_dir, response)
    print(
        json.dumps(
            {
                "frame_id": response.frame_id,
                "camera_name": response.camera_name,
                "matched_count": response.matched_count,
                "elapsed_ms": response.elapsed_ms,
                "error": response.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _save_capture(output_dir: Path, response: Any) -> None:
    payload = {
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "matched_count": response.matched_count,
        "elapsed_ms": response.elapsed_ms,
        "error": response.error,
        "pose_transform": None if response.pose_transform is None else [list(row) for row in response.pose_transform],
        "pose_translation_mm": None if response.pose_translation_mm is None else list(response.pose_translation_mm),
        "pose_rotation": None if response.pose_rotation is None else [list(row) for row in response.pose_rotation],
        "detections": list(response.detections),
        "debug": None
        if response.debug is None
        else {
            "camera_intrinsics": None if response.debug.camera_intrinsics is None else list(response.debug.camera_intrinsics),
            "detections": list(response.debug.detections),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if response.debug is not None and response.debug.color_bgr is not None:
        cv2.imwrite(str(output_dir / "color_bgr.jpg"), np.asarray(response.debug.color_bgr, dtype=np.uint8))
    if response.debug is not None and response.debug.depth_mm is not None:
        cv2.imwrite(str(output_dir / "depth.jpg"), _build_depth_view(np.asarray(response.debug.depth_mm)))
    if response.debug is not None and response.debug.overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "overlay.jpg"), np.asarray(response.debug.overlay_bgr, dtype=np.uint8))
    if response.debug is not None and response.debug.detection_overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "detection_overlay.jpg"), np.asarray(response.debug.detection_overlay_bgr, dtype=np.uint8))


def _build_priors_from_capture(captured: dict[str, Any]) -> list[BallPosePriorInfo]:
    recorded_balls = captured.get("balls", {}).get("ballinfo", [])
    if not isinstance(recorded_balls, list) or len(recorded_balls) < 3:
        return _default_priors()
    ordered = recorded_balls[:3]
    origin = np.asarray(ordered[0].get("position_camera_mm"), dtype=np.float64)
    second = np.asarray(ordered[1].get("position_camera_mm"), dtype=np.float64)
    third = np.asarray(ordered[2].get("position_camera_mm"), dtype=np.float64)
    if origin.shape != (3,) or second.shape != (3,) or third.shape != (3,):
        return _default_priors()
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(second)) or not np.all(np.isfinite(third)):
        return _default_priors()
    x_axis = second - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        return _default_priors()
    x_axis = x_axis / x_norm
    plane_hint = third - origin
    z_axis = np.cross(x_axis, plane_hint)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6:
        return _default_priors()
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-6:
        return _default_priors()
    y_axis = y_axis / y_norm
    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    priors: list[BallPosePriorInfo] = []
    for item in ordered:
        position = np.asarray(item.get("position_camera_mm"), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return _default_priors()
        model_center = basis.T @ (position - origin)
        priors.append(
            BallPosePriorInfo(
                color_hex=str(item.get("color_hex")),
                radius_mm=float(item.get("radius_mm", 20.0)),
                model_center_mm=tuple(model_center.tolist()),
            )
        )
    return priors


def _load_reference_relative_transform(captured: dict[str, Any]) -> tuple[tuple[float, float, float, float], ...] | None:
    pose = captured.get("pose", {})
    relative_transform = pose.get("relative_transform")
    if not isinstance(relative_transform, list) or len(relative_transform) != 4:
        return None
    rows: list[tuple[float, float, float, float]] = []
    for row in relative_transform:
        if not isinstance(row, list) or len(row) != 4:
            return None
        rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return tuple(rows)


def _load_prior_capture(prior_capture_path: Path) -> dict[str, Any]:
    if not prior_capture_path.is_file():
        return {}
    return json.loads(prior_capture_path.read_text(encoding="utf-8"))


def _default_priors() -> list[BallPosePriorInfo]:
    return [
        BallPosePriorInfo(
            color_hex="#ff0000",
            radius_mm=20.0,
            model_center_mm=(0.0, 0.0, 0.0),
        ),
        BallPosePriorInfo(
            color_hex="#ffff00",
            radius_mm=20.0,
            model_center_mm=(1.0, 0.0, 0.0),
        ),
        BallPosePriorInfo(
            color_hex="#ff00ff",
            radius_mm=20.0,
            model_center_mm=(0.0, 1.0, 0.0),
        ),
    ]


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


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ball pose detection smoke test")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prior-capture-path", type=Path, default=DEFAULT_PRIOR_CAPTURE_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            output_dir=Path(args.output_dir),
            prior_capture_path=Path(args.prior_capture_path),
        )
    )
