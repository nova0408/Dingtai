from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.121:6200"
DEFAULT_TIMEOUT_MS = 60_000
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture"
DEFAULT_PRIOR_CAPTURE_PATH = (
    PROJECT_ROOT / "test" / "wuji" / ".archive" / "collect_ball_opening_relative_pose" / "summary.json"
)
DEFAULT_PRIOR_COMPARE_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_priori"

from test.wuji.xcoresdk_arm_cli_test import (
    LEFT_ARM_IP,
    _print_sdk_result,
    _shutdown_robot,
)

from camera_pipeline.ball_pose_detection.protocol import (
    BallDetectionInfo,
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPosePriorInfo,
)
from camera_pipeline.client import CameraPipelineClient
from sdk.xcoresdk import xCoreSDK_python

DEFAULT_ARM_IP = LEFT_ARM_IP


@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    pose_matrix: np.ndarray
    translation_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


def _ensure_fixed_toolset(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
) -> None:
    robot.setToolset("g_tool_0", "g_wobj_0", ec)
    _print_sdk_result("setToolset(g_tool_0, g_wobj_0)", ec)
    if ec.get("ec", 0) != 0:
        raise RuntimeError("设置默认工具/工件失败")


def _pose_snapshot_from_sdk_pose(cartesian_pose: xCoreSDK_python.CartesianPosition) -> PoseSnapshot:
    rotation = Rotation.from_euler(
        "xyz",
        np.asarray(cartesian_pose.rpy, dtype=np.float64).reshape(3),
        degrees=False,
    ).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(cartesian_pose.trans, dtype=np.float64).reshape(3)
    return _matrix_to_pose_snapshot(matrix)


def _matrix_to_pose_snapshot(pose_matrix: np.ndarray) -> PoseSnapshot:
    matrix = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
    rpy_deg = Rotation.from_matrix(matrix[:3, :3]).as_euler("XYZ", degrees=True)
    return PoseSnapshot(
        pose_matrix=matrix,
        translation_mm=(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        rpy_deg=(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])),
    )


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prior_capture_path: Path = DEFAULT_PRIOR_CAPTURE_PATH,
    prior_compare_dir: Path = DEFAULT_PRIOR_COMPARE_DIR,
    arm_ip: str = DEFAULT_ARM_IP,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ball_pose_detection 先验采集与对比开始")
    prior_capture = _load_prior_capture(prior_capture_path)
    priors = _build_priors_from_capture(prior_capture)
    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(str(arm_ip))
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=DEFAULT_TIMEOUT_MS)
    try:
        _ensure_fixed_toolset(robot, ec)
        tcp_pose = robot.cartPosture(xCoreSDK_python.endInRef, ec)
        _print_sdk_result("cartPosture(endInRef)", ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"读取末端位姿失败: ip={arm_ip}")
        tcp_snapshot = _pose_snapshot_from_sdk_pose(tcp_pose)
        response = client.request_ball_pose_detection(
            BallPoseDetectionRequest(
                request_id=1,
                camera_name=str(camera_name),
                frame_id=-1,
                enable_debug=True,
                priors=tuple(priors),
            )
        )
    finally:
        client.close()
        _shutdown_robot(robot, ec)
    if response.matched_count < 3:
        raise RuntimeError("ball pose detection returned insufficient pose result")
    local_pose_transform = _build_three_ball_basis_transform(response.detections)
    if local_pose_transform is None:
        raise RuntimeError("failed to build local three-ball coordinate frame")
    _save_capture(output_dir, response, local_pose_transform, tcp_snapshot)
    _print_prior_comparison(
        prior_compare_dir=prior_compare_dir,
        output_dir=output_dir,
        response=response,
        local_pose_transform=local_pose_transform,
    )
    print(
        json.dumps(
            {
                "frame_id": response.frame_id,
                "camera_name": response.camera_name,
                "matched_count": response.matched_count,
                "elapsed_ms": response.elapsed_ms,
                "tcp_translation_mm": list(tcp_snapshot.translation_mm),
                "tcp_rpy_degrees": list(tcp_snapshot.rpy_deg),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _save_capture(
    output_dir: Path,
    response: BallPoseDetectionResponse,
    local_pose_transform: np.ndarray,
    tcp_snapshot: PoseSnapshot,
) -> None:
    local_pose_xyzrpy = _matrix_to_xyzrpy(local_pose_transform)
    local_overlay_bgr = _build_local_pose_overlay(response, local_pose_transform, local_pose_xyzrpy)
    payload = {
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "matched_count": response.matched_count,
        "elapsed_ms": response.elapsed_ms,
        "local_pose_transform": local_pose_transform.tolist(),
        "local_pose_translation_mm": local_pose_transform[:3, 3].tolist(),
        "local_pose_rotation": local_pose_transform[:3, :3].tolist(),
        "local_pose_xyzrpy": {
            "x_mm": float(local_pose_transform[0, 3]),
            "y_mm": float(local_pose_transform[1, 3]),
            "z_mm": float(local_pose_transform[2, 3]),
            "roll_deg": float(local_pose_xyzrpy[3]),
            "pitch_deg": float(local_pose_xyzrpy[4]),
            "yaw_deg": float(local_pose_xyzrpy[5]),
        },
        "detections": [_serialize_detection(item) for item in response.detections],
        "tcp_pose_matrix": tcp_snapshot.pose_matrix.tolist(),
        "tcp_translation_mm": list(tcp_snapshot.translation_mm),
        "tcp_rpy_degrees": list(tcp_snapshot.rpy_deg),
        "local_coordinate_frame": {
            "origin_ball": "yellow",
            "x_axis_ball": "red",
            "xoy_plane_ball": "purple",
        },
        "debug": _serialize_debug(response.debug_artifacts),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    debug = _get_debug_artifact(response)
    if debug is not None:
        cv2.imwrite(str(output_dir / "color_bgr.jpg"), np.asarray(debug.color_bgr, dtype=np.uint8))
        cv2.imwrite(str(output_dir / "depth.jpg"), _build_depth_view(np.asarray(debug.depth_mm)))
    if local_overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "overlay.jpg"), local_overlay_bgr)
        cv2.imwrite(str(output_dir / "local_pose_overlay.jpg"), local_overlay_bgr)
    if debug is not None:
        cv2.imwrite(
            str(output_dir / "detection_overlay.jpg"), np.asarray(debug.detection_overlay_bgr, dtype=np.uint8)
        )


def _serialize_detection(item: BallDetectionInfo) -> dict[str, Any]:
    return {
        "color_hex": item.color_hex,
        "detected": item.detected,
        "center_px": list(item.center_px),
        "center_mm": list(item.center_mm),
        "radius_mm": item.radius_mm,
        "radius_px": item.radius_px,
        "center_norm": list(item.center_norm),
        "radius_norm": item.radius_norm,
        "point_count": item.point_count,
        "status": item.status,
    }


def _serialize_debug(
    artifacts: tuple[BallPoseDetectionDebugArtifacts, ...],
) -> dict[str, Any] | None:
    if not artifacts:
        return None
    debug = artifacts[0]
    return {
        "camera_intrinsics": list(debug.camera_intrinsics),
        "detections": [_serialize_detection(item) for item in debug.detections],
    }


def _get_debug_artifact(
    response: BallPoseDetectionResponse,
) -> BallPoseDetectionDebugArtifacts | None:
    if not response.debug_artifacts:
        return None
    return response.debug_artifacts[0]


def _build_priors_from_capture(captured: dict[str, Any]) -> list[BallPosePriorInfo]:
    recorded_balls = captured.get("balls", {}).get("ballinfo", [])
    if not isinstance(recorded_balls, list) or len(recorded_balls) < 3:
        return _default_priors()
    lookup = {str(item.get("color_hex")): item for item in recorded_balls if isinstance(item, dict)}
    ordered_colors = ("#ffff00", "#ff0000", "#ff00ff")
    yellow_item = lookup.get(ordered_colors[0])
    red_item = lookup.get(ordered_colors[1])
    purple_item = lookup.get(ordered_colors[2])
    if yellow_item is None or red_item is None or purple_item is None:
        return _default_priors()
    ordered = (yellow_item, red_item, purple_item)
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
    for item, color_hex in zip(ordered, ordered_colors):
        position = np.asarray(item.get("position_camera_mm"), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return _default_priors()
        model_center = basis.T @ (position - origin)
        priors.append(
            BallPosePriorInfo(
                color_hex=color_hex,
                radius_mm=float(item.get("radius_mm", 20.0)),
                model_center_mm=tuple(model_center.tolist()),
            )
        )
    return priors


def _load_prior_capture(prior_capture_path: Path) -> dict[str, Any]:
    if not prior_capture_path.is_file():
        return {}
    return json.loads(prior_capture_path.read_text(encoding="utf-8"))


def _print_prior_comparison(
    prior_compare_dir: Path,
    output_dir: Path,
    response: Any,
    local_pose_transform: np.ndarray,
) -> None:
    if not prior_compare_dir.is_dir():
        logger.info("未找到先验对比目录，跳过对比: {}", prior_compare_dir)
        return
    prior_summary_path = prior_compare_dir / "summary.json"
    if not prior_summary_path.is_file():
        logger.warning("先验对比目录缺少 summary.json，跳过对比: {}", prior_summary_path)
        return
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8"))
    current_translation = _as_translation_vector(local_pose_transform[:3, 3])
    prior_translation = _as_translation_vector(
        prior_summary.get("local_pose_translation_mm", prior_summary.get("pose_translation_mm"))
    )
    if current_translation is None or prior_translation is None:
        logger.warning("当前结果或先验结果缺少本地坐标系平移，跳过坐标偏移对比")
        return
    current_pose_transform = local_pose_transform
    prior_pose_transform = _as_transform_matrix(
        prior_summary.get("local_pose_transform", prior_summary.get("pose_transform"))
    )
    current_three_ball_transform = local_pose_transform
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
                    "current_local_pose_translation_mm": current_translation.tolist(),
                    "prior_local_pose_translation_mm": prior_translation.tolist(),
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
            color_hex="#ffff00",
            radius_mm=20.0,
            model_center_mm=(0.0, 0.0, 0.0),
        ),
        BallPosePriorInfo(
            color_hex="#ff0000",
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


def _matrix_to_xyzrpy(transform: np.ndarray) -> tuple[float, float, float, float, float, float]:
    rotation = Rotation.from_matrix(np.asarray(transform[:3, :3], dtype=np.float64))
    roll_deg, pitch_deg, yaw_deg = rotation.as_euler("XYZ", degrees=True)
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(roll_deg),
        float(pitch_deg),
        float(yaw_deg),
    )


def _build_local_pose_overlay(
    response: BallPoseDetectionResponse,
    local_pose_transform: np.ndarray,
    local_pose_xyzrpy: tuple[float, float, float, float, float, float],
) -> np.ndarray | None:
    debug = _get_debug_artifact(response)
    if debug is None:
        return None
    overlay = np.asarray(debug.detection_overlay_bgr, dtype=np.uint8).copy()
    camera_intrinsics = debug.camera_intrinsics
    _draw_pose_axes(
        image_bgr=overlay,
        pose_transform=local_pose_transform,
        camera_intrinsics=camera_intrinsics,
        axis_length_mm=45.0,
        axis_colors=((0, 0, 255), (0, 255, 0), (255, 0, 0)),
        thickness=3,
    )
    x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = local_pose_xyzrpy
    lines = (
        "local xyzrpy",
        f"x={x_mm:.2f} mm  y={y_mm:.2f} mm  z={z_mm:.2f} mm",
        f"roll={roll_deg:.2f} deg  pitch={pitch_deg:.2f} deg  yaw={yaw_deg:.2f} deg",
        "frame: yellow origin, red x-axis, purple xoy plane",
    )
    _draw_text_block(overlay, lines)
    return overlay


def _draw_text_block(image_bgr: np.ndarray, lines: tuple[str, ...]) -> None:
    x0, y0 = 20, 30
    line_height = 24
    padding = 12
    width = 0
    for line in lines:
        (text_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        width = max(width, text_width)
    height = line_height * len(lines) + padding * 2
    cv2.rectangle(image_bgr, (x0 - 10, y0 - 22), (x0 + width + 20, y0 + height - 22), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        y = y0 + index * line_height
        cv2.putText(image_bgr, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


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
    if not isinstance(detections, (list, tuple)) or len(detections) < 3:
        return None
    by_color: dict[str, np.ndarray] = {}
    for item in detections:
        if isinstance(item, BallDetectionInfo):
            color_hex = item.color_hex
            center = np.asarray(item.center_mm, dtype=np.float64)
        elif isinstance(item, dict):
            color_hex = str(item.get("color_hex"))
            center = np.asarray(item.get("center_mm"), dtype=np.float64)
        else:
            continue
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            continue
        by_color[color_hex] = center
    origin = by_color.get("#ffff00")
    red = by_color.get("#ff0000")
    purple = by_color.get("#ff00ff")
    if origin is None or red is None or purple is None:
        return None
    x_axis = red - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        return None
    x_axis = x_axis / x_norm
    plane_hint = purple - origin
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


def _build_transform_comparison(
    current_transform: np.ndarray | None, prior_transform: np.ndarray | None
) -> dict[str, Any] | None:
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


def _load_camera_intrinsics(
    prior_summary: dict[str, Any],
    response: BallPoseDetectionResponse,
) -> tuple[float, float, float, float] | None:
    prior_debug = prior_summary.get("debug")
    prior_intrinsics = (
        prior_debug.get("camera_intrinsics")
        if isinstance(prior_debug, dict)
        else None
    )
    vector = np.asarray(prior_intrinsics, dtype=np.float64)
    if vector.shape == (4,) and np.all(np.isfinite(vector)):
        return (float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3]))
    debug = _get_debug_artifact(response)
    current_intrinsics = None if debug is None else debug.camera_intrinsics
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
    for point_px, color in zip(projected_points, axis_colors):
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
