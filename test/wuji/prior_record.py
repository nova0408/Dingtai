from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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

DEFAULT_BALL_CAMERA_NAME = "left_hand_camera"
DEFAULT_HEAD_CAMERA_NAME = "head_camera"
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.128:6200"
DEFAULT_TIMEOUT_MS = 60_000
DEFAULT_CAMERA_TIMEOUT_S = 10.0
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "prior_record"
DEFAULT_PRIOR_CAPTURE_PATH = (
    PROJECT_ROOT / "test" / "wuji" / ".archive" / "collect_ball_opening_relative_pose" / "summary.json"
)
DEFAULT_PRIOR_COMPARE_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_priori"
DEFAULT_HEAD_YAW_DEG = 60.0  # 头部固定 yaw 角度，单位 deg
DEFAULT_HEAD_PITCH_DEG = 45.0  # 头部固定 pitch 角度，单位 deg
DEFAULT_HEAD_SETTLE_S = 1.0  # 头部运动后的稳定等待时间，单位 s
DEFAULT_DICTIONARY_NAME = "DICT_APRILTAG_16H5"
DEFAULT_SQUARES_X = 4
DEFAULT_SQUARES_Y = 4
DEFAULT_SQUARE_LENGTH_MM = 20
DEFAULT_MARKER_LENGTH_MM = 14
DEFAULT_MIN_CHARUCO_CORNERS = 6
DEFAULT_WINDOW_WIDTH = 1440
DEFAULT_WINDOW_HEIGHT = 900
BALL_ORDERED_COLORS = ("#ffff00", "#ff0000", "#ff00ff")
BALL_COLOR_LABELS = ("yellow", "red", "purple")
BALL_DEFAULT_RADIUS_MM = 20.0
BALL_DEFAULT_MODEL_CENTERS_MM = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
BALL_POSE_AXIS_LENGTH_MM = 45.0
GEOMETRY_EPSILON = 1e-6
DEPTH_VALID_MIN_MM = 1.0
DEPTH_PERCENTILE_RANGE = (2.0, 98.0)

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
from camera_pipeline.protocol import CameraColorFramePacket
from common import (
    DEFAULT_PORT,
    SshTunnelGroup,
    close_wuyou_channel,
    create_wuyou_channel,
    stop_ssh_process,
)
from sdk.xcoresdk import xCoreSDK_python
from src.calibration import CharucoPoseEstimator, CharucoPoseResult
from src.wuji.head_client import WujiHeadClient

DEFAULT_ARM_IP = LEFT_ARM_IP


@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    pose_matrix: np.ndarray
    translation_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """头部相机内参与畸变参数。"""

    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


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
    rpy_deg = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    return PoseSnapshot(
        pose_matrix=matrix,
        translation_mm=(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        rpy_deg=(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])),
    )


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    ball_camera_name: str = DEFAULT_BALL_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    prior_capture_path: Path = DEFAULT_PRIOR_CAPTURE_PATH,
    prior_compare_dir: Path = DEFAULT_PRIOR_COMPARE_DIR,
    arm_ip: str = DEFAULT_ARM_IP,
    min_charuco_corners: int = DEFAULT_MIN_CHARUCO_CORNERS,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("开始记录三球先验与头部 ChArUco 板先验，输出目录：{}", output_dir)
    _record_ball_prior(
        service_addr=service_addr,
        camera_name=ball_camera_name,
        output_dir=output_dir,
        prior_capture_path=prior_capture_path,
        prior_compare_dir=prior_compare_dir,
        arm_ip=arm_ip,
    )
    _record_charuco_board_prior(
        service_addr=service_addr,
        output_dir=output_dir,
        min_charuco_corners=min_charuco_corners,
    )
    logger.success("先验记录完成：{}", output_dir)
    return 0


def _record_ball_prior(
    *,
    service_addr: str,
    camera_name: str,
    output_dir: Path,
    prior_capture_path: Path,
    prior_compare_dir: Path,
    arm_ip: str,
) -> None:
    """记录左臂三球坐标系先验。"""

    logger.info("开始记录左臂三球先验")
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
    _save_ball_capture(output_dir, response, local_pose_transform, tcp_snapshot)
    _print_prior_comparison(
        prior_compare_dir=prior_compare_dir,
        output_dir=output_dir,
        response=response,
        local_pose_transform=local_pose_transform,
    )
    logger.success(
        "三球先验已保存：frame_id={}，matched_count={}",
        response.frame_id,
        response.matched_count,
    )


def _save_ball_capture(
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
            "origin_ball": BALL_COLOR_LABELS[0],
            "x_axis_ball": BALL_COLOR_LABELS[1],
            "xoy_plane_ball": BALL_COLOR_LABELS[2],
        },
        "debug": _serialize_debug(response.debug_artifacts),
    }
    (output_dir / "ball_pose_prior.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    debug = _get_debug_artifact(response)
    if debug is not None:
        cv2.imwrite(str(output_dir / "ball_color_bgr.jpg"), np.asarray(debug.color_bgr, dtype=np.uint8))
        cv2.imwrite(str(output_dir / "ball_depth.jpg"), _build_depth_view(np.asarray(debug.depth_mm)))
    if local_overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "ball_pose_overlay.jpg"), local_overlay_bgr)
    if debug is not None:
        cv2.imwrite(
            str(output_dir / "ball_detection_overlay.jpg"),
            np.asarray(debug.detection_overlay_bgr, dtype=np.uint8),
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
    yellow_item = lookup.get(BALL_ORDERED_COLORS[0])
    red_item = lookup.get(BALL_ORDERED_COLORS[1])
    purple_item = lookup.get(BALL_ORDERED_COLORS[2])
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
    if x_norm <= GEOMETRY_EPSILON:
        return _default_priors()
    x_axis = x_axis / x_norm
    plane_hint = third - origin
    z_axis = np.cross(x_axis, plane_hint)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= GEOMETRY_EPSILON:
        return _default_priors()
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= GEOMETRY_EPSILON:
        return _default_priors()
    y_axis = y_axis / y_norm
    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    priors: list[BallPosePriorInfo] = []
    for item, color_hex in zip(ordered, BALL_ORDERED_COLORS, strict=True):
        position = np.asarray(item.get("position_camera_mm"), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return _default_priors()
        model_center = basis.T @ (position - origin)
        priors.append(
            BallPosePriorInfo(
                color_hex=color_hex,
                radius_mm=float(item.get("radius_mm", BALL_DEFAULT_RADIUS_MM)),
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
    current_overlay_path = output_dir / "ball_pose_overlay.jpg"
    if not current_overlay_path.is_file():
        logger.info("当前输出目录缺少三球位姿图，跳过图像对比绘制: {}", current_overlay_path)
        return
    overlay_bgr = cv2.imread(str(current_overlay_path), cv2.IMREAD_COLOR)
    if overlay_bgr is None:
        logger.warning("当前三球位姿图读取失败，跳过图像对比绘制: {}", current_overlay_path)
        return
    annotated = overlay_bgr.copy()
    _draw_pose_axes(
        image_bgr=annotated,
        pose_transform=prior_pose_transform,
        camera_intrinsics=camera_intrinsics,
        axis_length_mm=BALL_POSE_AXIS_LENGTH_MM,
        axis_colors=((0, 0, 180), (0, 180, 0), (180, 0, 0)),
        thickness=2,
    )
    _draw_pose_axes(
        image_bgr=annotated,
        pose_transform=current_pose_transform,
        camera_intrinsics=camera_intrinsics,
        axis_length_mm=BALL_POSE_AXIS_LENGTH_MM,
        axis_colors=((0, 0, 255), (0, 255, 0), (255, 0, 0)),
        thickness=3,
    )
    compare_overlay_path = output_dir / "ball_prior_compare_overlay.jpg"
    cv2.imwrite(str(compare_overlay_path), annotated)


def _default_priors() -> list[BallPosePriorInfo]:
    return [
        BallPosePriorInfo(
            color_hex=color_hex,
            radius_mm=BALL_DEFAULT_RADIUS_MM,
            model_center_mm=model_center_mm,
        )
        for color_hex, model_center_mm in zip(
            BALL_ORDERED_COLORS,
            BALL_DEFAULT_MODEL_CENTERS_MM,
            strict=True,
        )
    ]


def _build_depth_view(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > DEPTH_VALID_MIN_MM)
    hsv = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if np.any(valid):
        z_min = float(np.percentile(depth[valid], DEPTH_PERCENTILE_RANGE[0]))
        z_max = float(np.percentile(depth[valid], DEPTH_PERCENTILE_RANGE[1]))
        norm = np.clip(
            (depth - z_min) / max(GEOMETRY_EPSILON, z_max - z_min),
            0.0,
            1.0,
        )
        hsv[..., 0] = np.where(valid, np.rint((1.0 - norm) * 120.0), 0).astype(np.uint8)
        hsv[..., 1] = np.where(valid, 255, 0).astype(np.uint8)
        hsv[..., 2] = np.where(valid, 255, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _matrix_to_xyzrpy(transform: np.ndarray) -> tuple[float, float, float, float, float, float]:
    rotation = Rotation.from_matrix(np.asarray(transform[:3, :3], dtype=np.float64))
    roll_deg, pitch_deg, yaw_deg = rotation.as_euler("xyz", degrees=True)
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
        axis_length_mm=BALL_POSE_AXIS_LENGTH_MM,
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
    origin = by_color.get(BALL_ORDERED_COLORS[0])
    red = by_color.get(BALL_ORDERED_COLORS[1])
    purple = by_color.get(BALL_ORDERED_COLORS[2])
    if origin is None or red is None or purple is None:
        return None
    x_axis = red - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= GEOMETRY_EPSILON:
        return None
    x_axis = x_axis / x_norm
    plane_hint = purple - origin
    z_axis = np.cross(x_axis, plane_hint)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= GEOMETRY_EPSILON:
        return None
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= GEOMETRY_EPSILON:
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
    if z_mm <= GEOMETRY_EPSILON:
        return None
    fx, fy, cx, cy = camera_intrinsics
    x_px = fx * float(point_mm[0]) / z_mm + cx
    y_px = fy * float(point_mm[1]) / z_mm + cy
    if not np.isfinite(x_px) or not np.isfinite(y_px):
        return None
    return (int(round(x_px)), int(round(y_px)))


def _record_charuco_board_prior(
    *,
    service_addr: str,
    output_dir: Path,
    min_charuco_corners: int,
) -> None:
    """固定头部姿态，并交互记录一帧有效的 T_camera_board。"""

    head_tunnel: SshTunnelGroup | None = None
    head_channel: object | None = None
    try:
        head_tunnel, head_channel = create_wuyou_channel(DEFAULT_PORT)
        _set_head_fixed_pose(WujiHeadClient(head_channel))
        _capture_charuco_board_pose(
            service_addr=service_addr,
            output_dir=output_dir,
            min_charuco_corners=min_charuco_corners,
        )
    finally:
        if head_channel is not None:
            close_wuyou_channel(head_channel)
        if head_tunnel is not None:
            stop_ssh_process(head_tunnel)


def _set_head_fixed_pose(head: WujiHeadClient) -> None:
    logger.info(
        "固定头部姿态：yaw={:.1f} deg，pitch={:.1f} deg",
        DEFAULT_HEAD_YAW_DEG,
        DEFAULT_HEAD_PITCH_DEG,
    )
    head.set_head_yaw(DEFAULT_HEAD_YAW_DEG)
    head.set_head_pitch(DEFAULT_HEAD_PITCH_DEG)
    time.sleep(DEFAULT_HEAD_SETTLE_S)
    yaw_deg = float(head.get_head_yaw() or 0.0)
    pitch_deg = float(head.get_head_pitch() or 0.0)
    logger.success("头部已固定：yaw={:.1f} deg，pitch={:.1f} deg", yaw_deg, pitch_deg)


def _capture_charuco_board_pose(
    *,
    service_addr: str,
    output_dir: Path,
    min_charuco_corners: int,
) -> None:
    client = CameraPipelineClient(
        service_addr=service_addr,
        timeout_ms=int(DEFAULT_CAMERA_TIMEOUT_S * 1000.0),
    )
    estimator = CharucoPoseEstimator(_build_charuco_board())
    calibration = _read_head_camera_calibration(client)
    window_name = "Head Camera ChArUco Prior Record"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    logger.info("请保持标定板位姿不变；检测有效后按 Space/Enter/P 保存，按 Q/Esc 取消。")
    try:
        for frame_packet in client.subscribe_head_camera_color_frames():
            frame_bgr = np.asarray(frame_packet.color_bgr, dtype=np.uint8).copy()
            pose_result = estimator.estimate_pose(
                image_bgr=frame_bgr,
                camera_matrix=calibration.camera_matrix,
                dist_coeffs=calibration.dist_coeffs,
                min_charuco_corners=min_charuco_corners,
            )
            preview_bgr = _draw_charuco_preview(
                frame_bgr=frame_bgr,
                pose_result=pose_result,
                calibration=calibration,
            )
            cv2.imshow(window_name, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                raise RuntimeError("用户取消记录 ChArUco 板先验")
            if key in (13, 32, ord("p"), ord("P")):
                if pose_result.transform_se3 is None or pose_result.reprojection_error_px is None:
                    logger.warning("当前帧未获得有效 ChArUco 位姿，未保存。")
                    continue
                _save_charuco_board_prior(
                    output_dir=output_dir,
                    frame_packet=frame_packet,
                    frame_bgr=frame_bgr,
                    preview_bgr=preview_bgr,
                    pose_result=pose_result,
                )
                return
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("ChArUco 先验记录窗口已关闭，未保存结果")
    finally:
        client.close()
        cv2.destroyAllWindows()


def _build_charuco_board() -> cv2.aruco.CharucoBoard:
    if DEFAULT_DICTIONARY_NAME != "DICT_APRILTAG_16H5":
        raise ValueError(f"不支持的字典配置：{DEFAULT_DICTIONARY_NAME}")
    dictionary = cv2.aruco.getPredefinedDictionary(int(cv2.aruco.DICT_APRILTAG_16h5))
    return cv2.aruco.CharucoBoard(
        (DEFAULT_SQUARES_X, DEFAULT_SQUARES_Y),
        float(DEFAULT_SQUARE_LENGTH_MM),
        float(DEFAULT_MARKER_LENGTH_MM),
        dictionary,
    )


def _read_head_camera_calibration(client: CameraPipelineClient) -> CameraCalibration:
    response = client.get_head_camera_intrinsics(timeout_s=DEFAULT_CAMERA_TIMEOUT_S)
    distortion = np.asarray(response.distortion, dtype=np.float64).reshape(-1, 1)
    if distortion.size == 0:
        distortion = np.zeros((5, 1), dtype=np.float64)
    logger.info(
        "头部相机内参：camera={}，size={}x{}，fx={:.3f}，fy={:.3f}",
        response.camera_name,
        response.width,
        response.height,
        response.fx,
        response.fy,
    )
    return CameraCalibration(
        width=int(response.width),
        height=int(response.height),
        camera_matrix=np.asarray(
            [
                [float(response.fx), 0.0, float(response.cx)],
                [0.0, float(response.fy), float(response.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        dist_coeffs=distortion,
    )


def _draw_charuco_preview(
    *,
    frame_bgr: np.ndarray,
    pose_result: CharucoPoseResult,
    calibration: CameraCalibration,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    if pose_result.marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(canvas, pose_result.marker_corners_px, pose_result.marker_ids)
    if pose_result.charuco_corners_px is not None and pose_result.charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            canvas,
            pose_result.charuco_corners_px.reshape(-1, 1, 2).astype(np.float32),
            pose_result.charuco_ids,
        )
    if pose_result.rvec is not None and pose_result.tvec is not None:
        cv2.drawFrameAxes(
            canvas,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            pose_result.rvec,
            pose_result.tvec,
            float(DEFAULT_SQUARE_LENGTH_MM * 1.5),
            3,
        )
    status = "VALID" if pose_result.transform_se3 is not None else "INVALID"
    reprojection = (
        "NA"
        if pose_result.reprojection_error_px is None
        else f"{pose_result.reprojection_error_px:.3f}px"
    )
    lines = (
        f"ChArUco prior | {status}",
        f"head yaw={DEFAULT_HEAD_YAW_DEG:.1f}deg pitch={DEFAULT_HEAD_PITCH_DEG:.1f}deg",
        f"markers={pose_result.marker_count} charuco={pose_result.charuco_count} reproj={reprojection}",
        "Space/Enter/P save | Q/Esc cancel",
    )
    _draw_text_block(canvas, lines)
    return canvas


def _save_charuco_board_prior(
    *,
    output_dir: Path,
    frame_packet: CameraColorFramePacket,
    frame_bgr: np.ndarray,
    preview_bgr: np.ndarray,
    pose_result: CharucoPoseResult,
) -> None:
    if pose_result.transform_se3 is None or pose_result.reprojection_error_px is None:
        raise ValueError("ChArUco 位姿结果无效，不能保存先验")
    transform_mm = np.asarray(pose_result.transform_se3, dtype=np.float64).reshape(4, 4)
    translation_mm = transform_mm[:3, 3]
    rpy_deg = Rotation.from_matrix(transform_mm[:3, :3]).as_euler("xyz", degrees=True)
    payload = {
        "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
        "frame_id": int(frame_packet.frame_id),
        "camera_name": DEFAULT_HEAD_CAMERA_NAME,
        "camera_timestamp_ms": float(frame_packet.timestamp_ms),
        "pose_semantics": "T_camera_board",
        "translation_unit": "mm",
        "rotation_convention": 'scipy Rotation.as_euler("xyz", degrees=True)',
        "head_yaw_deg": DEFAULT_HEAD_YAW_DEG,
        "head_pitch_deg": DEFAULT_HEAD_PITCH_DEG,
        "dictionary": DEFAULT_DICTIONARY_NAME,
        "squares_x": DEFAULT_SQUARES_X,
        "squares_y": DEFAULT_SQUARES_Y,
        "square_length_mm": DEFAULT_SQUARE_LENGTH_MM,
        "marker_length_mm": DEFAULT_MARKER_LENGTH_MM,
        "marker_count": int(pose_result.marker_count),
        "charuco_count": int(pose_result.charuco_count),
        "reprojection_error_px": float(pose_result.reprojection_error_px),
        "camera_board_transform": transform_mm.tolist(),
        "translation_mm": translation_mm.tolist(),
        "rpy_deg": rpy_deg.tolist(),
    }
    result_path = output_dir / "charuco_board_prior.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not cv2.imwrite(str(output_dir / "charuco_board_raw.png"), frame_bgr):
        raise RuntimeError("保存 ChArUco 原始图像失败")
    if not cv2.imwrite(str(output_dir / "charuco_board_preview.png"), preview_bgr):
        raise RuntimeError("保存 ChArUco 预览图像失败")
    logger.success(
        "ChArUco 板先验已保存：{}，translation=({:.3f}, {:.3f}, {:.3f}) mm",
        result_path,
        translation_mm[0],
        translation_mm[1],
        translation_mm[2],
    )


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="记录左臂三球与头部 ChArUco 板先验")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--ball-camera-name", type=str, default=DEFAULT_BALL_CAMERA_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prior-capture-path", type=Path, default=DEFAULT_PRIOR_CAPTURE_PATH)
    parser.add_argument("--prior-compare-dir", type=Path, default=DEFAULT_PRIOR_COMPARE_DIR)
    parser.add_argument("--min-charuco-corners", type=int, default=DEFAULT_MIN_CHARUCO_CORNERS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = _parse_cli(sys.argv[1:])
        raise SystemExit(
            main(
                service_addr=str(args.service_addr),
                ball_camera_name=str(args.ball_camera_name),
                output_dir=Path(args.output_dir),
                prior_capture_path=Path(args.prior_capture_path),
                prior_compare_dir=Path(args.prior_compare_dir),
                min_charuco_corners=int(args.min_charuco_corners),
            )
        )
    raise SystemExit(main())
