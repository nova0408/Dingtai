from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rotation3D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.ball_pose_detection.protocol import BallPoseDetectionRequest
from camera_pipeline.client import CameraPipelineClient
from sdk.xcoresdk import xCoreSDK_python
from src.utils.datas import Quaternion, Transform, Translation
from test.wuji.ball_pose_detection import (
    DEFAULT_CAMERA_NAME,
    DEFAULT_PRIOR_CAPTURE_PATH,
    DEFAULT_SERVICE_ADDR,
    _build_priors_from_capture,
    _build_three_ball_basis_transform,
    _load_prior_capture,
)
from test.wuji.xcoresdk_arm_cli_test import (
    DEFAULT_CARTESIAN_ZONE,
    DEFAULT_TOOL_NAME,
    DEFAULT_WOBJ_NAME,
    LEFT_ARM_IP,
    _apply_named_toolset,
    _copy_cartesian_pose_context,
    _deg_to_rad,
    _ensure_nrt_motion_ready,
    _mm_to_m,
    _print_sdk_result,
    _shutdown_robot,
    _validate_cartesian_target,
    _wait_until_idle,
)


# region 默认参数

DEFAULT_WINDOW_NAME = "Ball Pose Offset Interactive Validation"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_offset_interactive"
DEFAULT_HAND_EYE_PATH = (
    PROJECT_ROOT / "experiments" / "hand_eye" / "runs" / "20260706_165649" / "tool_camera_extrinsic_result.txt"
)
DEFAULT_ARM_IP = LEFT_ARM_IP
DEFAULT_MOVE_SPEED_MM_S = 8.0
DEFAULT_TRANSLATION_STEP_MM = 5.0
DEFAULT_FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_TITLE_FONT_SIZE = 22
DEFAULT_BODY_FONT_SIZE = 18
DEFAULT_REFRESH_SLEEP_S = 0.03
DEFAULT_CAMERA_TIMEOUT_MS = 30_000
DEFAULT_FORCE_COMPARE_MATRIX = np.asarray(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
# endregion


# region 数据结构
@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    host_timestamp_iso: str
    flange_pose_base: Transform
    flange_translation_mm: tuple[float, float, float]
    flange_rpy_degrees: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    frame_id: int
    timestamp_ms: float
    pose_camera_ball: Transform
    pose_translation_mm: tuple[float, float, float]
    residual_mm: float | None
    matched_count: int


@dataclass(frozen=True, slots=True)
class ReferenceSnapshot:
    recorded_at_iso: str
    robot_snapshot: RobotSnapshot
    detection_snapshot: DetectionSnapshot


@dataclass(frozen=True, slots=True)
class ComparisonSnapshot:
    actual_robot_delta_mm: tuple[float, float, float]
    actual_robot_delta_distance_mm: float
    actual_pose_camera_ball: Transform
    actual_delta_camera_mm: tuple[float, float, float]
    actual_delta_distance_mm: float
    predicted_with_hand_eye_pose_camera_ball: Transform
    predicted_with_hand_eye_delta_camera_mm: tuple[float, float, float]
    predicted_with_hand_eye_error_mm: tuple[float, float, float]
    predicted_with_hand_eye_error_distance_mm: float
    predicted_without_hand_eye_pose_camera_ball: Transform
    predicted_without_hand_eye_delta_camera_mm: tuple[float, float, float]
    predicted_without_hand_eye_error_mm: tuple[float, float, float]
    predicted_without_hand_eye_error_distance_mm: float
    actual_delta_rotation_deg: float
    predicted_with_hand_eye_error_rotation_deg: float
    predicted_without_hand_eye_error_rotation_deg: float
    actual_robot_delta_force_m_mm: tuple[float, float, float]
    actual_delta_camera_force_m_mm: tuple[float, float, float]
    force_m_robot_vs_camera_error_mm: tuple[float, float, float]
    force_m_robot_vs_camera_error_distance_mm: float


@dataclass(frozen=True, slots=True)
class StepRecord:
    record_index: int
    action_name: str
    action_axis: str
    action_delta_mm: float
    host_timestamp_iso: str
    frame_id: int
    actual_robot_delta_mm: tuple[float, float, float]
    actual_delta_camera_mm: tuple[float, float, float]
    predicted_with_hand_eye_delta_camera_mm: tuple[float, float, float]
    predicted_without_hand_eye_delta_camera_mm: tuple[float, float, float]
    predicted_with_hand_eye_error_distance_mm: float
    predicted_without_hand_eye_error_distance_mm: float
    actual_delta_rotation_deg: float
    predicted_with_hand_eye_error_rotation_deg: float
    predicted_without_hand_eye_error_rotation_deg: float


@dataclass(frozen=True, slots=True)
class MoveAction:
    key: str
    axis_name: str
    index: int
    direction_sign: float


@dataclass(slots=True)
class RuntimeState:
    reference_snapshot: ReferenceSnapshot | None = None
    latest_comparison: ComparisonSnapshot | None = None
    last_action_text: str = "等待首个有效三球位姿作为基准"
    step_records: list[StepRecord] = field(default_factory=list)
    cumulative_command_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


# endregion


# region 常量映射
MOVE_ACTIONS: dict[int, MoveAction] = {
    ord("w"): MoveAction("w", "X+", 0, +1.0),
    ord("s"): MoveAction("s", "X-", 0, -1.0),
    ord("a"): MoveAction("a", "Y+", 1, +1.0),
    ord("d"): MoveAction("d", "Y-", 1, -1.0),
    ord("r"): MoveAction("r", "Z+", 2, +1.0),
    ord("f"): MoveAction("f", "Z-", 2, -1.0),
}
# endregion


# region 主流程
def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    arm_ip: str = DEFAULT_ARM_IP,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    prior_capture_path: Path = DEFAULT_PRIOR_CAPTURE_PATH,
    hand_eye_path: Path = DEFAULT_HAND_EYE_PATH,
    translation_step_mm: float = DEFAULT_TRANSLATION_STEP_MM,
    move_speed_mm_s: float = DEFAULT_MOVE_SPEED_MM_S,
) -> int:
    logger.info("交互式三球位姿偏移验证启动")
    logger.info("服务地址 {} 相机 {}", service_addr, camera_name)
    logger.info("机械臂 {} tool={} wobj={}", arm_ip, DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME)
    logger.info("验证边界：依赖真实相机、真实机械臂与 ball_pose_detection 服务")

    session_dir = _create_session_dir(output_root)
    records_csv_path = session_dir / "step_records.csv"
    reference_json_path = session_dir / "reference_snapshot.json"
    latest_json_path = session_dir / "latest_comparison.json"
    hand_eye_transform = _load_hand_eye_transform(hand_eye_path)
    priors_capture = _load_prior_capture(prior_capture_path)
    priors = tuple(_build_priors_from_capture(priors_capture))
    state = RuntimeState(step_records=[])

    ec: dict[str, object] = {}
    robot = xCoreSDK_python.xMateErProRobot(str(arm_ip))
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=DEFAULT_CAMERA_TIMEOUT_MS)
    try:
        robot_info = robot.robotInfo(ec)
        _print_sdk_result("robotInfo", ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(f"连接机械臂失败: ip={arm_ip}")
        logger.success("机械臂已连接 type={} uid={}", robot_info.type, robot_info.id)
        if not _ensure_nrt_motion_ready(robot, ec):
            raise RuntimeError("机械臂运动准备失败")
        if _apply_named_toolset(robot, ec) is None:
            raise RuntimeError("设置工具/工件坐标系失败")

        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            detection_response = _request_ball_pose_detection(
                client=client,
                camera_name=camera_name,
                priors=priors,
                reference_relative_transform=None,
            )
            robot_snapshot = _read_robot_snapshot(robot, ec)
            detection_snapshot = _build_detection_snapshot(detection_response)
            if state.reference_snapshot is None and detection_snapshot is not None:
                state.reference_snapshot = ReferenceSnapshot(
                    recorded_at_iso=datetime.now().isoformat(timespec="milliseconds"),
                    robot_snapshot=robot_snapshot,
                    detection_snapshot=detection_snapshot,
                )
                _write_reference_snapshot_json(reference_json_path, state.reference_snapshot)
                state.cumulative_command_offset_mm = (0.0, 0.0, 0.0)
                state.last_action_text = f"已自动记录首个有效基准 frame={detection_snapshot.frame_id}"
                logger.success(state.last_action_text)
            if state.reference_snapshot is not None and detection_snapshot is not None:
                state.latest_comparison = _build_comparison_snapshot(
                    reference_snapshot=state.reference_snapshot,
                    current_robot_snapshot=robot_snapshot,
                    current_detection_snapshot=detection_snapshot,
                    hand_eye_transform=hand_eye_transform,
                )
                _write_latest_comparison_json(latest_json_path, state.latest_comparison)
            preview_bgr = _build_preview_image(
                detection_response=detection_response,
                robot_snapshot=robot_snapshot,
                reference_snapshot=state.reference_snapshot,
                comparison_snapshot=state.latest_comparison,
                last_action_text=state.last_action_text,
                translation_step_mm=float(translation_step_mm),
                cumulative_command_offset_mm=state.cumulative_command_offset_mm,
            )
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                if detection_snapshot is None:
                    state.last_action_text = "基准重置失败：当前未拿到有效三球位姿"
                    logger.warning(state.last_action_text)
                else:
                    state.reference_snapshot = ReferenceSnapshot(
                        recorded_at_iso=datetime.now().isoformat(timespec="milliseconds"),
                        robot_snapshot=robot_snapshot,
                        detection_snapshot=detection_snapshot,
                    )
                    state.latest_comparison = None
                    state.cumulative_command_offset_mm = (0.0, 0.0, 0.0)
                    _write_reference_snapshot_json(reference_json_path, state.reference_snapshot)
                    state.last_action_text = f"已手动重置基准 frame={detection_snapshot.frame_id}"
                    logger.success(state.last_action_text)
            elif key in (ord("c"), ord("C")):
                state.reference_snapshot = None
                state.latest_comparison = None
                state.cumulative_command_offset_mm = (0.0, 0.0, 0.0)
                state.last_action_text = "已清空基准，等待下一个有效帧自动建基准"
                logger.info(state.last_action_text)
            elif key in MOVE_ACTIONS:
                move_action = MOVE_ACTIONS[key]
                state.last_action_text = _execute_single_step_move(
                    robot=robot,
                    ec=ec,
                    move_action=move_action,
                    translation_step_mm=float(translation_step_mm),
                    move_speed_mm_s=float(move_speed_mm_s),
                )
                logger.info(state.last_action_text)
                state.cumulative_command_offset_mm = _accumulate_command_offset(
                    current_offset_mm=state.cumulative_command_offset_mm,
                    move_action=move_action,
                    translation_step_mm=float(translation_step_mm),
                )
                if state.reference_snapshot is not None and state.latest_comparison is not None:
                    step_record = StepRecord(
                        record_index=len(state.step_records) + 1,
                        action_name=move_action.key,
                        action_axis=move_action.axis_name,
                        action_delta_mm=float(translation_step_mm) * float(move_action.direction_sign),
                        host_timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
                        frame_id=detection_snapshot.frame_id if detection_snapshot is not None else -1,
                        actual_robot_delta_mm=state.latest_comparison.actual_robot_delta_mm,
                        actual_delta_camera_mm=state.latest_comparison.actual_delta_camera_mm,
                        predicted_with_hand_eye_delta_camera_mm=state.latest_comparison.predicted_with_hand_eye_delta_camera_mm,
                        predicted_without_hand_eye_delta_camera_mm=state.latest_comparison.predicted_without_hand_eye_delta_camera_mm,
                        predicted_with_hand_eye_error_distance_mm=state.latest_comparison.predicted_with_hand_eye_error_distance_mm,
                        predicted_without_hand_eye_error_distance_mm=state.latest_comparison.predicted_without_hand_eye_error_distance_mm,
                        actual_delta_rotation_deg=state.latest_comparison.actual_delta_rotation_deg,
                        predicted_with_hand_eye_error_rotation_deg=state.latest_comparison.predicted_with_hand_eye_error_rotation_deg,
                        predicted_without_hand_eye_error_rotation_deg=state.latest_comparison.predicted_without_hand_eye_error_rotation_deg,
                    )
                    state.step_records.append(step_record)
                    _write_step_records_csv(records_csv_path, state.step_records)
            time.sleep(DEFAULT_REFRESH_SLEEP_S)
        return 0
    finally:
        client.close()
        cv2.destroyAllWindows()
        _shutdown_robot(robot, ec)


# endregion


# region 采样与检测
def _request_ball_pose_detection(
    client: CameraPipelineClient,
    camera_name: str,
    priors: tuple[Any, ...],
    reference_relative_transform: tuple[tuple[float, float, float, float], ...] | None,
) -> Any:
    response = client.request_ball_pose_detection(
        BallPoseDetectionRequest(
            request_id=1,
            camera_name=str(camera_name),
            frame_id=-1,
            enable_debug=True,
            priors=priors,
            reference_relative_transform_mm=reference_relative_transform,
        )
    )
    if response.error is not None:
        raise RuntimeError(f"ball pose detection 返回错误: {response.error}")
    return response


def _build_detection_snapshot(response: Any) -> DetectionSnapshot | None:
    pose_transform = _response_pose_transform(response)
    if pose_transform is None:
        return None
    return DetectionSnapshot(
        frame_id=int(response.frame_id),
        timestamp_ms=float(response.timestamp_ms),
        pose_camera_ball=pose_transform,
        pose_translation_mm=(
            float(pose_transform.translation.x),
            float(pose_transform.translation.y),
            float(pose_transform.translation.z),
        ),
        residual_mm=None if response.residual_mm is None else float(response.residual_mm),
        matched_count=int(response.matched_count),
    )


def _response_pose_transform(response: Any) -> Transform | None:
    matrix = np.asarray(response.pose_transform, dtype=np.float64)
    if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)):
        return Transform.from_SE3(matrix)
    fallback_matrix = _build_three_ball_basis_transform(list(response.detections))
    if fallback_matrix is None:
        return None
    return Transform.from_SE3(np.asarray(fallback_matrix, dtype=np.float64))


def _read_robot_snapshot(robot: xCoreSDK_python.xMateErProRobot, ec: dict[str, object]) -> RobotSnapshot:
    cartesian_pose = robot.cartPosture(xCoreSDK_python.flangeInBase, ec)
    if ec.get("ec", 0) != 0:
        _print_sdk_result("cartPosture(flangeInBase)", ec)
        raise RuntimeError("读取机械臂 flange 位姿失败")
    translation_mm = (
        float(cartesian_pose.trans[0]) * 1000.0,
        float(cartesian_pose.trans[1]) * 1000.0,
        float(cartesian_pose.trans[2]) * 1000.0,
    )
    rpy_degrees = (
        float(np.degrees(float(cartesian_pose.rpy[0]))),
        float(np.degrees(float(cartesian_pose.rpy[1]))),
        float(np.degrees(float(cartesian_pose.rpy[2]))),
    )
    return RobotSnapshot(
        host_timestamp_iso=datetime.now().isoformat(timespec="milliseconds"),
        flange_pose_base=_sdk_pose_to_transform(translation_mm, rpy_degrees),
        flange_translation_mm=translation_mm,
        flange_rpy_degrees=rpy_degrees,
    )


def _sdk_pose_to_transform(
    translation_mm: tuple[float, float, float],
    rpy_degrees: tuple[float, float, float],
) -> Transform:
    rotation_matrix = Rotation3D.from_euler("XYZ", list(rpy_degrees), degrees=True).as_matrix()
    return Transform(
        translation=Translation(*translation_mm),
        rotation=Quaternion.from_SO3(rotation_matrix),
    )


# endregion


# region 变换比较
def _build_comparison_snapshot(
    reference_snapshot: ReferenceSnapshot,
    current_robot_snapshot: RobotSnapshot,
    current_detection_snapshot: DetectionSnapshot,
    hand_eye_transform: Transform,
) -> ComparisonSnapshot:
    reference_pose_camera_ball = reference_snapshot.detection_snapshot.pose_camera_ball
    actual_pose_camera_ball = current_detection_snapshot.pose_camera_ball

    predicted_with_hand_eye_pose_camera_ball = _predict_current_camera_ball_pose(
        reference_robot_pose_base_flange=reference_snapshot.robot_snapshot.flange_pose_base,
        current_robot_pose_base_flange=current_robot_snapshot.flange_pose_base,
        reference_pose_camera_ball=reference_pose_camera_ball,
        tool_camera_transform=hand_eye_transform,
    )
    predicted_without_hand_eye_pose_camera_ball = _predict_current_camera_ball_pose(
        reference_robot_pose_base_flange=reference_snapshot.robot_snapshot.flange_pose_base,
        current_robot_pose_base_flange=current_robot_snapshot.flange_pose_base,
        reference_pose_camera_ball=reference_pose_camera_ball,
        tool_camera_transform=Transform.Identity(),
    )

    actual_robot_delta = _translation_delta(
        current_transform=current_robot_snapshot.flange_pose_base,
        reference_transform=reference_snapshot.robot_snapshot.flange_pose_base,
    )
    actual_delta_camera = _translation_delta(
        current_transform=actual_pose_camera_ball,
        reference_transform=reference_pose_camera_ball,
    )
    predicted_with_hand_eye_delta = _translation_delta(
        current_transform=predicted_with_hand_eye_pose_camera_ball,
        reference_transform=reference_pose_camera_ball,
    )
    predicted_without_hand_eye_delta = _translation_delta(
        current_transform=predicted_without_hand_eye_pose_camera_ball,
        reference_transform=reference_pose_camera_ball,
    )
    predicted_with_hand_eye_error = (
        float(predicted_with_hand_eye_delta[0] - actual_delta_camera[0]),
        float(predicted_with_hand_eye_delta[1] - actual_delta_camera[1]),
        float(predicted_with_hand_eye_delta[2] - actual_delta_camera[2]),
    )
    predicted_without_hand_eye_error = (
        float(predicted_without_hand_eye_delta[0] - actual_delta_camera[0]),
        float(predicted_without_hand_eye_delta[1] - actual_delta_camera[1]),
        float(predicted_without_hand_eye_delta[2] - actual_delta_camera[2]),
    )

    actual_delta_rotation_deg = _relative_rotation_deg(reference_pose_camera_ball, actual_pose_camera_ball)
    predicted_with_hand_eye_error_rotation_deg = _relative_rotation_error_deg(
        actual_transform=actual_pose_camera_ball,
        predicted_transform=predicted_with_hand_eye_pose_camera_ball,
    )
    predicted_without_hand_eye_error_rotation_deg = _relative_rotation_error_deg(
        actual_transform=actual_pose_camera_ball,
        predicted_transform=predicted_without_hand_eye_pose_camera_ball,
    )
    actual_robot_delta_force_m = _apply_force_compare_matrix(actual_robot_delta)
    actual_delta_camera_force_m = _apply_force_compare_matrix(actual_delta_camera)
    force_m_robot_vs_camera_error = (
        float(actual_robot_delta_force_m[0] - actual_delta_camera_force_m[0]),
        float(actual_robot_delta_force_m[1] - actual_delta_camera_force_m[1]),
        float(actual_robot_delta_force_m[2] - actual_delta_camera_force_m[2]),
    )
    return ComparisonSnapshot(
        actual_robot_delta_mm=actual_robot_delta,
        actual_robot_delta_distance_mm=_vector_norm(actual_robot_delta),
        actual_pose_camera_ball=actual_pose_camera_ball,
        actual_delta_camera_mm=actual_delta_camera,
        actual_delta_distance_mm=_vector_norm(actual_delta_camera),
        predicted_with_hand_eye_pose_camera_ball=predicted_with_hand_eye_pose_camera_ball,
        predicted_with_hand_eye_delta_camera_mm=predicted_with_hand_eye_delta,
        predicted_with_hand_eye_error_mm=predicted_with_hand_eye_error,
        predicted_with_hand_eye_error_distance_mm=_vector_norm(predicted_with_hand_eye_error),
        predicted_without_hand_eye_pose_camera_ball=predicted_without_hand_eye_pose_camera_ball,
        predicted_without_hand_eye_delta_camera_mm=predicted_without_hand_eye_delta,
        predicted_without_hand_eye_error_mm=predicted_without_hand_eye_error,
        predicted_without_hand_eye_error_distance_mm=_vector_norm(predicted_without_hand_eye_error),
        actual_delta_rotation_deg=actual_delta_rotation_deg,
        predicted_with_hand_eye_error_rotation_deg=predicted_with_hand_eye_error_rotation_deg,
        predicted_without_hand_eye_error_rotation_deg=predicted_without_hand_eye_error_rotation_deg,
        actual_robot_delta_force_m_mm=actual_robot_delta_force_m,
        actual_delta_camera_force_m_mm=actual_delta_camera_force_m,
        force_m_robot_vs_camera_error_mm=force_m_robot_vs_camera_error,
        force_m_robot_vs_camera_error_distance_mm=_vector_norm(force_m_robot_vs_camera_error),
    )


def _predict_current_camera_ball_pose(
    reference_robot_pose_base_flange: Transform,
    current_robot_pose_base_flange: Transform,
    reference_pose_camera_ball: Transform,
    tool_camera_transform: Transform,
) -> Transform:
    base_camera_reference = reference_robot_pose_base_flange @ tool_camera_transform
    base_camera_current = current_robot_pose_base_flange @ tool_camera_transform
    base_ball = base_camera_reference @ reference_pose_camera_ball
    current_camera_base = _inverse_transform(base_camera_current)
    return current_camera_base @ base_ball


def _inverse_transform(transform: Transform) -> Transform:
    return Transform.from_SE3(np.linalg.inv(transform.as_SE3()))


def _translation_delta(current_transform: Transform, reference_transform: Transform) -> tuple[float, float, float]:
    return (
        float(current_transform.translation.x - reference_transform.translation.x),
        float(current_transform.translation.y - reference_transform.translation.y),
        float(current_transform.translation.z - reference_transform.translation.z),
    )


def _relative_rotation_deg(reference_transform: Transform, current_transform: Transform) -> float:
    delta = _inverse_transform(reference_transform) @ current_transform
    return _rotation_angle_deg(delta.rotation.as_SE3()[:3, :3])


def _relative_rotation_error_deg(actual_transform: Transform, predicted_transform: Transform) -> float:
    delta = _inverse_transform(actual_transform) @ predicted_transform
    return _rotation_angle_deg(delta.rotation.as_SE3()[:3, :3])


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    trace_value = float(np.trace(np.asarray(rotation_matrix, dtype=np.float64)))
    cosine_value = float(np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_value)))


def _vector_norm(vector_xyz: tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.asarray(vector_xyz, dtype=np.float64)))


def _apply_force_compare_matrix(vector_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    row_vector = np.asarray([[float(vector_xyz[0]), float(vector_xyz[1]), float(vector_xyz[2])]], dtype=np.float64)
    transformed = row_vector @ DEFAULT_FORCE_COMPARE_MATRIX
    return float(transformed[0, 0]), float(transformed[0, 1]), float(transformed[0, 2])


# endregion


# region 机械臂步进
def _execute_single_step_move(
    robot: xCoreSDK_python.xMateErProRobot,
    ec: dict[str, object],
    move_action: MoveAction,
    translation_step_mm: float,
    move_speed_mm_s: float,
) -> str:
    current_pose = robot.cartPosture(xCoreSDK_python.flangeInBase, ec)
    if ec.get("ec", 0) != 0:
        _print_sdk_result("cartPosture(before-step flangeInBase)", ec)
        return f"读取当前位置失败 axis={move_action.axis_name}"
    target_xyz_mm = [
        float(current_pose.trans[0]) * 1000.0,
        float(current_pose.trans[1]) * 1000.0,
        float(current_pose.trans[2]) * 1000.0,
    ]
    target_rpy_deg = [
        float(np.degrees(float(current_pose.rpy[0]))),
        float(np.degrees(float(current_pose.rpy[1]))),
        float(np.degrees(float(current_pose.rpy[2]))),
    ]
    target_xyz_mm[move_action.index] += float(move_action.direction_sign) * float(translation_step_mm)
    target_pose = xCoreSDK_python.CartesianPosition(_mm_to_m(target_xyz_mm) + _deg_to_rad(target_rpy_deg))
    _copy_cartesian_pose_context(current_pose, target_pose)
    use_move_abs_j = not _validate_cartesian_target(robot, ec, target_pose)
    cmd_id = xCoreSDK_python.PyString()
    robot.moveReset(ec)
    _print_sdk_result("moveReset", ec)
    if ec.get("ec", 0) != 0:
        return f"moveReset 失败 axis={move_action.axis_name}"
    if use_move_abs_j:
        toolset = robot.toolset(ec)
        _print_sdk_result("toolset", ec)
        if ec.get("ec", 0) != 0:
            return f"读取 toolset 失败 axis={move_action.axis_name}"
        target_joint = robot.model().calcIk(target_pose, toolset, ec)
        _print_sdk_result("calcIk(step)", ec)
        if ec.get("ec", 0) != 0:
            return f"calcIk 失败 axis={move_action.axis_name}"
        robot.moveAppend(
            [xCoreSDK_python.MoveAbsJCommand(target_joint, max(1.0, float(move_speed_mm_s) * 20.0), DEFAULT_CARTESIAN_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveAbsJ)", ec)
    else:
        robot.moveAppend(
            [xCoreSDK_python.MoveLCommand(target_pose, float(move_speed_mm_s), DEFAULT_CARTESIAN_ZONE)],
            cmd_id,
            ec,
        )
        _print_sdk_result("moveAppend(MoveL)", ec)
    if ec.get("ec", 0) != 0:
        return f"下发运动失败 axis={move_action.axis_name}"
    robot.moveStart(ec)
    _print_sdk_result("moveStart", ec)
    if ec.get("ec", 0) != 0:
        return f"启动运动失败 axis={move_action.axis_name}"
    _wait_until_idle(robot, ec, f"等待单步移动 {move_action.axis_name}")
    return (
        f"已执行单步 axis={move_action.axis_name} "
        f"delta_mm={float(move_action.direction_sign) * float(translation_step_mm):.2f}"
    )


def _accumulate_command_offset(
    current_offset_mm: tuple[float, float, float],
    move_action: MoveAction,
    translation_step_mm: float,
) -> tuple[float, float, float]:
    values = [float(current_offset_mm[0]), float(current_offset_mm[1]), float(current_offset_mm[2])]
    values[move_action.index] += float(move_action.direction_sign) * float(translation_step_mm)
    return float(values[0]), float(values[1]), float(values[2])


# endregion


# region 文本绘制与预览
def _build_preview_image(
    detection_response: Any,
    robot_snapshot: RobotSnapshot,
    reference_snapshot: ReferenceSnapshot | None,
    comparison_snapshot: ComparisonSnapshot | None,
    last_action_text: str,
    translation_step_mm: float,
    cumulative_command_offset_mm: tuple[float, float, float],
) -> np.ndarray:
    cumulative_command_offset_force_m = _apply_force_compare_matrix(cumulative_command_offset_mm)
    if detection_response.debug is not None and detection_response.debug.overlay_bgr is not None:
        canvas = np.asarray(detection_response.debug.overlay_bgr, dtype=np.uint8).copy()
    elif detection_response.debug is not None and detection_response.debug.color_bgr is not None:
        canvas = np.asarray(detection_response.debug.color_bgr, dtype=np.uint8).copy()
    else:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    title_font = _load_font(DEFAULT_TITLE_FONT_SIZE)
    body_font = _load_font(DEFAULT_BODY_FONT_SIZE)

    lines = [
        "三球位姿偏移交互验证",
        f"首个有效帧自动建基准  空格=手动重置基准  C=清空  W/S/A/D/R/F=单步平移  Q=退出  step={translation_step_mm:.2f} mm",
        f"tool={DEFAULT_TOOL_NAME} wobj={DEFAULT_WOBJ_NAME}",
        f"当前 flange xyz(mm)=({_format_triplet(robot_snapshot.flange_translation_mm)})",
        f"当前 flange rpy(deg)=({_format_triplet(robot_snapshot.flange_rpy_degrees)})",
        f"当前检测 frame={int(detection_response.frame_id)} matched={int(detection_response.matched_count)} residual_mm={_fmt_optional(detection_response.residual_mm)}",
        "累计发送偏移(mm)=("
        f"{_format_triplet(cumulative_command_offset_mm)}) "
        f"|dist|={_vector_norm(cumulative_command_offset_mm):.3f}",
        f"状态: {last_action_text}",
    ]
    if reference_snapshot is None:
        lines.append("基准位姿: 未记录，等待有效三球检测")
    else:
        lines.append(
            "基准位姿: "
            f"frame={reference_snapshot.detection_snapshot.frame_id} "
            f"flange_xyz(mm)=({_format_triplet(reference_snapshot.robot_snapshot.flange_translation_mm)})"
        )
        lines.append(
            "参考三球 pose_camera_ball(mm)=("
            f"{_format_triplet(reference_snapshot.detection_snapshot.pose_translation_mm)})"
        )
    if comparison_snapshot is None:
        lines.append("比较结果: 记录参考后开始单步移动即可实时比较")
    else:
        lines.extend(
            [
                "命令累计偏移 robot_frame(mm)=("
                f"{_format_triplet(cumulative_command_offset_mm)}) "
                f"|dist|={_vector_norm(cumulative_command_offset_mm):.3f}",
                "机械臂实际累计偏移 flangeInBase(mm)=("
                f"{_format_triplet(comparison_snapshot.actual_robot_delta_mm)}) "
                f"|dist|={comparison_snapshot.actual_robot_delta_distance_mm:.3f}",
                "真实相机偏移 delta_camera(mm)=("
                f"{_format_triplet(comparison_snapshot.actual_delta_camera_mm)}) "
                f"|dist|={comparison_snapshot.actual_delta_distance_mm:.3f} "
                f"|rot|={comparison_snapshot.actual_delta_rotation_deg:.3f} deg",
                "手眼预测相机偏移 delta_camera(mm)=("
                f"{_format_triplet(comparison_snapshot.predicted_with_hand_eye_delta_camera_mm)}) "
                f"err(mm)=({_format_triplet(comparison_snapshot.predicted_with_hand_eye_error_mm)}) "
                f"|err|={comparison_snapshot.predicted_with_hand_eye_error_distance_mm:.3f} "
                f"rot_err={comparison_snapshot.predicted_with_hand_eye_error_rotation_deg:.3f} deg",
                "不加手眼错误映射 delta_camera(mm)=("
                f"{_format_triplet(comparison_snapshot.predicted_without_hand_eye_delta_camera_mm)}) "
                f"err(mm)=({_format_triplet(comparison_snapshot.predicted_without_hand_eye_error_mm)}) "
                f"|err|={comparison_snapshot.predicted_without_hand_eye_error_distance_mm:.3f} "
                f"rot_err={comparison_snapshot.predicted_without_hand_eye_error_rotation_deg:.3f} deg",
                "强制右乘 M 后 command@M(mm)=("
                f"{_format_triplet(cumulative_command_offset_force_m)}) "
                "robot@flangeInBase@M(mm)=("
                f"{_format_triplet(comparison_snapshot.actual_robot_delta_force_m_mm)})",
                "强制右乘 M 后 real_camera@M(mm)=("
                f"{_format_triplet(comparison_snapshot.actual_delta_camera_force_m_mm)}) "
                f"err(mm)=({_format_triplet(comparison_snapshot.force_m_robot_vs_camera_error_mm)}) "
                f"|err|={comparison_snapshot.force_m_robot_vs_camera_error_distance_mm:.3f}",
            ]
        )

    _draw_info_panel(draw, lines, title_font=title_font, body_font=body_font)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _draw_info_panel(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    title_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    body_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x0 = 18
    y0 = 18
    for index, text in enumerate(lines):
        font = title_font if index == 0 else body_font
        fill = (255, 255, 255) if index == 0 else (230, 230, 230)
        stroke_fill = (20, 20, 20)
        draw.text(
            (x0, y0 + index * 28),
            text,
            font=font,
            fill=fill,
            stroke_width=2,
            stroke_fill=stroke_fill,
        )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(DEFAULT_FONT_PATH), size=size)
    except OSError:
        return ImageFont.load_default()


def _format_triplet(values: tuple[float, float, float]) -> str:
    return ", ".join(f"{float(value):.3f}" for value in values)


def _fmt_optional(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


# endregion


# region 文件记录
def _create_session_dir(output_root: Path) -> Path:
    session_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _load_hand_eye_transform(hand_eye_path: Path) -> Transform:
    if not hand_eye_path.is_file():
        raise FileNotFoundError(f"手眼标定结果文件不存在: {hand_eye_path}")
    lines = hand_eye_path.read_text(encoding="utf-8").splitlines()
    matrix_rows: list[list[float]] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped == "tool_camera_matrix_se3=":
            collecting = True
            continue
        if collecting:
            if stripped == "":
                break
            values = [float(item.strip()) for item in stripped.split(",")]
            if len(values) != 4:
                raise ValueError(f"手眼矩阵行格式错误: {line}")
            matrix_rows.append(values)
            if len(matrix_rows) == 4:
                break
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"手眼矩阵维度错误: shape={matrix.shape}, path={hand_eye_path}")
    logger.info("已加载手眼标定结果 {}", hand_eye_path)
    return Transform.from_SE3(matrix)


def _write_reference_snapshot_json(path: Path, reference_snapshot: ReferenceSnapshot) -> None:
    payload = {
        "recorded_at_iso": reference_snapshot.recorded_at_iso,
        "robot_flange_translation_mm": list(reference_snapshot.robot_snapshot.flange_translation_mm),
        "robot_flange_rpy_degrees": list(reference_snapshot.robot_snapshot.flange_rpy_degrees),
        "robot_flange_pose_base": reference_snapshot.robot_snapshot.flange_pose_base.as_SE3().tolist(),
        "frame_id": reference_snapshot.detection_snapshot.frame_id,
        "timestamp_ms": reference_snapshot.detection_snapshot.timestamp_ms,
        "pose_camera_ball": reference_snapshot.detection_snapshot.pose_camera_ball.as_SE3().tolist(),
        "pose_translation_mm": list(reference_snapshot.detection_snapshot.pose_translation_mm),
        "residual_mm": reference_snapshot.detection_snapshot.residual_mm,
        "matched_count": reference_snapshot.detection_snapshot.matched_count,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_latest_comparison_json(path: Path, comparison_snapshot: ComparisonSnapshot) -> None:
    payload = {
        "actual_robot_delta_mm": list(comparison_snapshot.actual_robot_delta_mm),
        "actual_robot_delta_distance_mm": comparison_snapshot.actual_robot_delta_distance_mm,
        "actual_robot_delta_force_m_mm": list(comparison_snapshot.actual_robot_delta_force_m_mm),
        "actual_pose_camera_ball": comparison_snapshot.actual_pose_camera_ball.as_SE3().tolist(),
        "actual_delta_camera_mm": list(comparison_snapshot.actual_delta_camera_mm),
        "actual_delta_distance_mm": comparison_snapshot.actual_delta_distance_mm,
        "actual_delta_camera_force_m_mm": list(comparison_snapshot.actual_delta_camera_force_m_mm),
        "force_m_robot_vs_camera_error_mm": list(comparison_snapshot.force_m_robot_vs_camera_error_mm),
        "force_m_robot_vs_camera_error_distance_mm": comparison_snapshot.force_m_robot_vs_camera_error_distance_mm,
        "predicted_with_hand_eye_pose_camera_ball": comparison_snapshot.predicted_with_hand_eye_pose_camera_ball.as_SE3().tolist(),
        "predicted_with_hand_eye_delta_camera_mm": list(comparison_snapshot.predicted_with_hand_eye_delta_camera_mm),
        "predicted_with_hand_eye_error_mm": list(comparison_snapshot.predicted_with_hand_eye_error_mm),
        "predicted_with_hand_eye_error_distance_mm": comparison_snapshot.predicted_with_hand_eye_error_distance_mm,
        "predicted_without_hand_eye_pose_camera_ball": comparison_snapshot.predicted_without_hand_eye_pose_camera_ball.as_SE3().tolist(),
        "predicted_without_hand_eye_delta_camera_mm": list(comparison_snapshot.predicted_without_hand_eye_delta_camera_mm),
        "predicted_without_hand_eye_error_mm": list(comparison_snapshot.predicted_without_hand_eye_error_mm),
        "predicted_without_hand_eye_error_distance_mm": comparison_snapshot.predicted_without_hand_eye_error_distance_mm,
        "actual_delta_rotation_deg": comparison_snapshot.actual_delta_rotation_deg,
        "predicted_with_hand_eye_error_rotation_deg": comparison_snapshot.predicted_with_hand_eye_error_rotation_deg,
        "predicted_without_hand_eye_error_rotation_deg": comparison_snapshot.predicted_without_hand_eye_error_rotation_deg,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_step_records_csv(path: Path, step_records: list[StepRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_index",
                "action_name",
                "action_axis",
                "action_delta_mm",
                "host_timestamp_iso",
                "frame_id",
                "actual_robot_delta_x_mm",
                "actual_robot_delta_y_mm",
                "actual_robot_delta_z_mm",
                "actual_delta_camera_x_mm",
                "actual_delta_camera_y_mm",
                "actual_delta_camera_z_mm",
                "predicted_with_hand_eye_x_mm",
                "predicted_with_hand_eye_y_mm",
                "predicted_with_hand_eye_z_mm",
                "predicted_without_hand_eye_x_mm",
                "predicted_without_hand_eye_y_mm",
                "predicted_without_hand_eye_z_mm",
                "predicted_with_hand_eye_error_distance_mm",
                "predicted_without_hand_eye_error_distance_mm",
                "actual_delta_rotation_deg",
                "predicted_with_hand_eye_error_rotation_deg",
                "predicted_without_hand_eye_error_rotation_deg",
            ]
        )
        for item in step_records:
            writer.writerow(
                [
                    item.record_index,
                    item.action_name,
                    item.action_axis,
                    f"{item.action_delta_mm:.6f}",
                    item.host_timestamp_iso,
                    item.frame_id,
                    f"{item.actual_robot_delta_mm[0]:.6f}",
                    f"{item.actual_robot_delta_mm[1]:.6f}",
                    f"{item.actual_robot_delta_mm[2]:.6f}",
                    f"{item.actual_delta_camera_mm[0]:.6f}",
                    f"{item.actual_delta_camera_mm[1]:.6f}",
                    f"{item.actual_delta_camera_mm[2]:.6f}",
                    f"{item.predicted_with_hand_eye_delta_camera_mm[0]:.6f}",
                    f"{item.predicted_with_hand_eye_delta_camera_mm[1]:.6f}",
                    f"{item.predicted_with_hand_eye_delta_camera_mm[2]:.6f}",
                    f"{item.predicted_without_hand_eye_delta_camera_mm[0]:.6f}",
                    f"{item.predicted_without_hand_eye_delta_camera_mm[1]:.6f}",
                    f"{item.predicted_without_hand_eye_delta_camera_mm[2]:.6f}",
                    f"{item.predicted_with_hand_eye_error_distance_mm:.6f}",
                    f"{item.predicted_without_hand_eye_error_distance_mm:.6f}",
                    f"{item.actual_delta_rotation_deg:.6f}",
                    f"{item.predicted_with_hand_eye_error_rotation_deg:.6f}",
                    f"{item.predicted_without_hand_eye_error_rotation_deg:.6f}",
                ]
            )


# endregion


# region CLI
def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="interactive ball pose offset validation")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--arm-ip", type=str, default=DEFAULT_ARM_IP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--prior-capture-path", type=Path, default=DEFAULT_PRIOR_CAPTURE_PATH)
    parser.add_argument("--hand-eye-path", type=Path, default=DEFAULT_HAND_EYE_PATH)
    parser.add_argument("--translation-step-mm", type=float, default=DEFAULT_TRANSLATION_STEP_MM)
    parser.add_argument("--move-speed-mm-s", type=float, default=DEFAULT_MOVE_SPEED_MM_S)
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_args = _parse_cli(sys.argv[1:])
        raise SystemExit(
            main(
                service_addr=str(cli_args.service_addr),
                camera_name=str(cli_args.camera_name),
                arm_ip=str(cli_args.arm_ip),
                output_root=Path(cli_args.output_root),
                prior_capture_path=Path(cli_args.prior_capture_path),
                hand_eye_path=Path(cli_args.hand_eye_path),
                translation_step_mm=float(cli_args.translation_step_mm),
                move_speed_mm_s=float(cli_args.move_speed_mm_s),
            )
        )
    raise SystemExit(main())
