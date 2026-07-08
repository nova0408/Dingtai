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
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6200"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture"
DEFAULT_PRIOR_CAPTURE_PATH = PROJECT_ROOT / "test" / "wuji" / ".archive" / "collect_ball_opening_relative_pose" / "summary.json"
DEFAULT_PRIOR_COMPARE_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_priori"

from camera_pipeline.ball_pose_detection.protocol import BallPoseDetectionRequest, BallPosePriorInfo  # noqa: E402
from camera_pipeline.client import CameraPipelineClient  # noqa: E402


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prior_capture_path: Path = DEFAULT_PRIOR_CAPTURE_PATH,
    prior_compare_dir: Path = DEFAULT_PRIOR_COMPARE_DIR,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ball_pose_detection smoke test start")
    prior_capture = _load_prior_capture(prior_capture_path)
    priors = _build_priors_from_capture(prior_capture)
    reference_relative_transform = _load_reference_relative_transform(prior_capture)
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=30_000)
    try:
        response = client.request_ball_pose_detection(
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
    _print_prior_comparison(prior_compare_dir=prior_compare_dir, output_dir=output_dir, response=response)
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


def _print_prior_comparison(prior_compare_dir: Path, output_dir: Path, response: Any) -> None:
    if not prior_compare_dir.is_dir():
        logger.info("未找到先验对比目录，跳过对比: {}", prior_compare_dir)
        return
    prior_summary_path = prior_compare_dir / "summary.json"
    if not prior_summary_path.is_file():
        logger.warning("先验对比目录缺少 summary.json，跳过对比: {}", prior_summary_path)
        return
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    current_translation = _as_translation_vector(response.pose_translation_mm)
    prior_translation = _as_translation_vector(prior_summary.get("pose_translation_mm"))
    if current_translation is None or prior_translation is None:
        logger.warning("当前结果或先验结果缺少 pose_translation_mm，跳过坐标偏移对比")
        return
    current_pose_transform = _as_transform_matrix(response.pose_transform)
    prior_pose_transform = _as_transform_matrix(prior_summary.get("pose_transform"))
    current_three_ball_transform = _build_three_ball_basis_transform(response.detections)
    prior_three_ball_transform = _build_three_ball_basis_transform(prior_summary.get("detections"))
    camera_intrinsics = _load_camera_intrinsics(prior_summary, response)
    delta_translation = current_translation - prior_translation
    delta_distance = float(np.linalg.norm(delta_translation))
    if current_pose_transform is not None and prior_pose_transform is not None and camera_intrinsics is not None:
        _draw_prior_comparison_overlay(
            output_dir=output_dir,
            current_pose_transform=current_pose_transform,
            prior_pose_transform=prior_pose_transform,
            camera_intrinsics=camera_intrinsics,
        )
    else:
        logger.warning("位姿矩阵或相机内参缺失，跳过坐标系绘制")
    three_ball_compare = _build_transform_comparison(current_three_ball_transform, prior_three_ball_transform)
    final_compare = _build_transform_comparison(current_pose_transform, prior_pose_transform)
    print(
        json.dumps(
            {
                "prior_compare": {
                    "prior_summary_path": str(prior_summary_path),
                    "current_pose_translation_mm": current_translation.tolist(),
                    "prior_pose_translation_mm": prior_translation.tolist(),
                    "delta_translation_mm": delta_translation.tolist(),
                    "delta_distance_mm": delta_distance,
                    "three_ball_basis_compare": three_ball_compare,
                    "final_pose_compare": final_compare,
                }
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _draw_prior_comparison_overlay(
    output_dir: Path,
    current_pose_transform: np.ndarray,
    prior_pose_transform: np.ndarray,
    camera_intrinsics: tuple[float, float, float, float],
) -> None:
    current_overlay_path = output_dir / "overlay.jpg"
    if not current_overlay_path.is_file():
        logger.info("当前输出目录缺少 overlay.jpg，跳过图像对比绘制: {}", current_overlay_path)
        return
    overlay_bgr = cv2.imread(str(current_overlay_path), cv2.IMREAD_COLOR)
    if overlay_bgr is None:
        logger.warning("当前 overlay.jpg 读取失败，跳过图像对比绘制: {}", current_overlay_path)
        return
    annotated = overlay_bgr.copy()
    _draw_pose_axes(
        image_bgr=annotated,
        pose_transform=prior_pose_transform,
        camera_intrinsics=camera_intrinsics,
        axis_length_mm=45.0,
        axis_colors=((0, 0, 180), (0, 180, 0), (180, 0, 0)),
        thickness=2,
    )
    _draw_pose_axes(
        image_bgr=annotated,
        pose_transform=current_pose_transform,
        camera_intrinsics=camera_intrinsics,
        axis_length_mm=45.0,
        axis_colors=((0, 0, 255), (0, 255, 0), (255, 0, 0)),
        thickness=3,
    )
    compare_overlay_path = output_dir / "prior_compare_overlay.jpg"
    cv2.imwrite(str(compare_overlay_path), annotated)


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


def _as_translation_vector(value: Any) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _as_transform_matrix(value: Any) -> np.ndarray | None:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def _build_three_ball_basis_transform(detections: Any) -> np.ndarray | None:
    if not isinstance(detections, list | tuple) or len(detections) < 3:
        return None
    centers: list[np.ndarray] = []
    for item in detections[:3]:
        if not isinstance(item, dict):
            return None
        center = np.asarray(item.get("center_mm"), dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            return None
        centers.append(center)
    origin = centers[0]
    x_axis = centers[1] - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        return None
    x_axis = x_axis / x_norm
    plane_hint = centers[2] - origin
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
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    transform[:3, 3] = origin
    return transform


def _build_transform_comparison(current_transform: np.ndarray | None, prior_transform: np.ndarray | None) -> dict[str, Any] | None:
    if current_transform is None or prior_transform is None:
        return None
    delta_transform = current_transform @ np.linalg.inv(prior_transform)
    delta_translation = delta_transform[:3, 3]
    rotation_trace = float(np.trace(delta_transform[:3, :3]))
    rotation_cos = float(np.clip((rotation_trace - 1.0) * 0.5, -1.0, 1.0))
    rotation_angle_deg = float(np.degrees(np.arccos(rotation_cos)))
    return {
        "current_translation_mm": current_transform[:3, 3].tolist(),
        "prior_translation_mm": prior_transform[:3, 3].tolist(),
        "delta_transform_translation_mm": delta_translation.tolist(),
        "delta_transform_distance_mm": float(np.linalg.norm(delta_translation)),
        "delta_rotation_deg": rotation_angle_deg,
    }


def _load_camera_intrinsics(prior_summary: dict[str, Any], response: Any) -> tuple[float, float, float, float] | None:
    prior_intrinsics = prior_summary.get("debug", {}).get("camera_intrinsics")
    vector = np.asarray(prior_intrinsics, dtype=np.float64)
    if vector.shape == (4,) and np.all(np.isfinite(vector)):
        return (float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3]))
    current_intrinsics = None if response.debug is None else response.debug.camera_intrinsics
    vector = np.asarray(current_intrinsics, dtype=np.float64)
    if vector.shape == (4,) and np.all(np.isfinite(vector)):
        return (float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3]))
    return None


def _draw_pose_axes(
    image_bgr: np.ndarray,
    pose_transform: np.ndarray,
    camera_intrinsics: tuple[float, float, float, float],
    axis_length_mm: float,
    axis_colors: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    thickness: int,
) -> None:
    rotation = pose_transform[:3, :3]
    translation = pose_transform[:3, 3]
    origin_px = _project_point_to_pixel(translation, camera_intrinsics)
    if origin_px is None:
        return
    axis_points = (
        translation + rotation[:, 0] * float(axis_length_mm),
        translation + rotation[:, 1] * float(axis_length_mm),
        translation + rotation[:, 2] * float(axis_length_mm),
    )
    projected_points = [_project_point_to_pixel(point, camera_intrinsics) for point in axis_points]
    cv2.circle(image_bgr, origin_px, 5, (255, 255, 255), -1, cv2.LINE_AA)
    for point_px, color in zip(projected_points, axis_colors, strict=False):
        if point_px is None:
            continue
        cv2.arrowedLine(image_bgr, origin_px, point_px, color, thickness, cv2.LINE_AA, tipLength=0.18)


def _project_point_to_pixel(
    point_mm: np.ndarray,
    camera_intrinsics: tuple[float, float, float, float],
) -> tuple[int, int] | None:
    if point_mm.shape != (3,) or not np.all(np.isfinite(point_mm)):
        return None
    z_mm = float(point_mm[2])
    if z_mm <= 1e-6:
        return None
    fx, fy, cx, cy = camera_intrinsics
    x_px = fx * float(point_mm[0]) / z_mm + cx
    y_px = fy * float(point_mm[1]) / z_mm + cy
    if not np.isfinite(x_px) or not np.isfinite(y_px):
        return None
    return (int(round(x_px)), int(round(y_px)))


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ball pose detection smoke test")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prior-capture-path", type=Path, default=DEFAULT_PRIOR_CAPTURE_PATH)
    parser.add_argument("--prior-compare-dir", type=Path, default=DEFAULT_PRIOR_COMPARE_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            output_dir=Path(args.output_dir),
            prior_capture_path=Path(args.prior_capture_path),
            prior_compare_dir=Path(args.prior_compare_dir),
        )
    )
