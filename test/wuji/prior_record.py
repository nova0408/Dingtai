from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import sys
import time
from dataclasses import dataclass, replace
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "record_replay" / "prior_data"
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
BALL_DEFAULT_DIAMETER_MM = 20.0
BALL_DEFAULT_MODEL_CENTERS_MM = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
BALL_POSE_AXIS_LENGTH_MM = 45.0
BALL_PRIOR_SAMPLE_COUNT = 30
"三球先验必须收集的完整且不同帧数量。"
BALL_PRIOR_MAX_ATTEMPTS = 90
"收集三球先验的最大请求次数，允许跳过未检出帧和重复帧。"
BALL_PRIOR_MIN_INLIER_COUNT = 24
"MAD 异常剔除后允许写入先验的最少帧数量。"
BALL_PRIOR_OUTLIER_MAD_SCALE = 3.5
"三球位置与直径联合异常距离的 MAD 倍数阈值。"
BALL_PRIOR_OUTLIER_MIN_THRESHOLD_MM = 2.0
"MAD 过小时使用的最小异常距离阈值，单位 mm。"
BALL_HUE_TOLERANCE_RANGE = (3.0, 8.0)
"标定 HSV Hue 半宽的最小值和最大值，OpenCV Hue 单位。"
BALL_SATURATION_TOLERANCE_RANGE = (15.0, 45.0)
"标定 HSV Saturation 半宽的最小值和最大值。"
BALL_VALUE_TOLERANCE_RANGE = (20.0, 55.0)
"标定 HSV Value 半宽的最小值和最大值。"
BALL_COLOR_OUTLIER_MIN_THRESHOLDS = (2.0, 8.0, 8.0)
"颜色帧异常剔除的 Hue、Saturation、Value 最小偏差阈值。"
BALL_COLOR_MIN_INLIER_RATIO = 0.8
"每个球生成精确颜色范围所需的最小颜色帧保留比例。"
GEOMETRY_EPSILON = 1e-6
DEPTH_VALID_MIN_MM = 1.0
DEPTH_PERCENTILE_RANGE = (2.0, 98.0)

from test.wuji.xcoresdk_arm_cli_test import (
    LEFT_ARM_IP,
    _print_sdk_result,
    _shutdown_robot,
)

from common import (
    DEFAULT_PORT,
    SshTunnelGroup,
    close_wuyou_channel,
    create_wuyou_channel,
    stop_ssh_process,
)

from camera_pipeline.ball_pose_detection.protocol import (
    BallDetectionInfo,
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPosePriorInfo,
)
from camera_pipeline.client import CameraName, CameraPipelineClient
from camera_pipeline.protocol import CameraColorFramePacket
from camera_pipeline.service.protocol import CharucoDetectionRequest
from sdk.xcoresdk import xCoreSDK_python
from src.calibration import CharucoPoseResult
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


HsvRange = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class BallPriorAggregation:
    """30 帧三球先验鲁棒聚合结果。"""

    response: BallPoseDetectionResponse
    "使用保留帧均值重建的检测响应，debug 图像来自最后一个完整有效帧。"
    hsv_ranges_by_color: dict[str, tuple[HsvRange, ...]]
    "按参考颜色标签索引的标定 HSV 窄范围。"
    observed_color_hex_by_color: dict[str, str]
    "按参考颜色标签索引的实测平均 RGB 十六进制颜色。"
    sample_count: int
    "聚合前完整且不同的采样帧数量。"
    inlier_count: int
    "位置与直径 MAD 异常剔除后保留的帧数量。"


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
            raise RuntimeError(f"读取末端位姿失败：ip={arm_ip}")
        tcp_snapshot = _pose_snapshot_from_sdk_pose(tcp_pose)
        samples, evidence_response = _capture_ball_prior_samples(
            client=client,
            camera_name=camera_name,
            priors=tuple(priors),
        )
    finally:
        client.close()
        _shutdown_robot(robot, ec)
    aggregation = _aggregate_ball_prior_samples(samples, evidence_response)
    response = aggregation.response
    local_pose_transform = _build_three_ball_basis_transform(response.detections)
    if local_pose_transform is None:
        raise RuntimeError("failed to build local three-ball coordinate frame")
    _save_ball_capture(
        output_dir,
        aggregation,
        local_pose_transform,
        tcp_snapshot,
    )
    _print_prior_comparison(
        prior_compare_dir=prior_compare_dir,
        output_dir=output_dir,
        response=response,
        local_pose_transform=local_pose_transform,
    )
    logger.success(
        "三球先验已保存：frame_id={}，采样帧={}，保留帧={}",
        response.frame_id,
        aggregation.sample_count,
        aggregation.inlier_count,
    )


def _capture_ball_prior_samples(
    *,
    client: CameraPipelineClient,
    camera_name: str,
    priors: tuple[BallPosePriorInfo, ...],
) -> tuple[list[BallPoseDetectionResponse], BallPoseDetectionResponse]:
    """收集 30 个不同且完整的三球检测帧。

    采样阶段只使用参考颜色、物理直径和占位模型中心，不使用尚未形成的精确颜色与
    相对位置先验。每个请求都必须携带 debug，完整帧的 debug overlay 用作采集证据；
    样本列表立即去除图像载荷，避免同时持有 30 份 RGBD 图。
    """

    reference_priors = tuple(
        replace(
            prior,
            model_center_mm=model_center_mm,
            hsv_ranges=(),
        )
        for prior, model_center_mm in zip(
            priors,
            BALL_DEFAULT_MODEL_CENTERS_MM,
            strict=True,
        )
    )
    samples: list[BallPoseDetectionResponse] = []
    seen_frame_ids: set[int] = set()
    evidence_response: BallPoseDetectionResponse | None = None
    for request_id in range(1, BALL_PRIOR_MAX_ATTEMPTS + 1):
        response = client.detect_ball(
            BallPoseDetectionRequest(
                request_id=request_id,
                camera_name=CameraName(camera_name),
                frame_id=-1,
                enable_debug=True,
                priors=reference_priors,
            )
        )
        if response.frame_id in seen_frame_ids:
            logger.warning("三球先验跳过重复帧：frame_id={}", response.frame_id)
            continue
        if not _is_complete_ball_response(response):
            logger.warning(
                "三球先验跳过未完整检出帧：frame_id={} matched_count={}",
                response.frame_id,
                response.matched_count,
            )
            continue
        seen_frame_ids.add(response.frame_id)
        samples.append(replace(response, debug_artifacts=()))
        evidence_response = response
        logger.info(
            "三球先验采样进度：有效帧={}/{} frame_id={}",
            len(samples),
            BALL_PRIOR_SAMPLE_COUNT,
            response.frame_id,
        )
        if len(samples) == BALL_PRIOR_SAMPLE_COUNT:
            break
    if len(samples) != BALL_PRIOR_SAMPLE_COUNT or evidence_response is None:
        raise RuntimeError(
            "三球先验未收集到足够完整且不同的帧 "
            f"samples={len(samples)}/{BALL_PRIOR_SAMPLE_COUNT} "
            f"attempts={BALL_PRIOR_MAX_ATTEMPTS}"
        )
    return samples, evidence_response


def _is_complete_ball_response(response: BallPoseDetectionResponse) -> bool:
    """判断响应是否包含三球中心、直径、实测 HSV 和 debug overlay。"""

    if response.matched_count != len(BALL_ORDERED_COLORS):
        return False
    if len(response.debug_artifacts) != 1:
        return False
    if np.asarray(response.debug_artifacts[0].overlay_bgr).size == 0:
        return False
    by_color = {item.color_hex: item for item in response.detections}
    return all(
        color in by_color
        and by_color[color].detected
        and len(by_color[color].center_mm) == 3
        and len(by_color[color].observed_hsv) == 3
        for color in BALL_ORDERED_COLORS
    )


def _aggregate_ball_prior_samples(
    samples: list[BallPoseDetectionResponse],
    evidence_response: BallPoseDetectionResponse,
) -> BallPriorAggregation:
    """对 30 帧三球位置、直径和颜色做异常剔除及均值聚合。"""

    if len(samples) != BALL_PRIOR_SAMPLE_COUNT:
        raise ValueError(
            f"三球先验必须恰好包含 {BALL_PRIOR_SAMPLE_COUNT} 帧，实际 {len(samples)}"
        )
    ordered_samples = [
        tuple(
            next(item for item in response.detections if item.color_hex == color)
            for color in BALL_ORDERED_COLORS
        )
        for response in samples
    ]
    features = np.asarray(
        [
            [
                value
                for detection in detections
                for value in (*detection.center_mm, detection.diameter_mm)
            ]
            for detections in ordered_samples
        ],
        dtype=np.float64,
    )
    median = np.median(features, axis=0)
    distances = np.linalg.norm(features - median.reshape(1, -1), axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    threshold = max(
        BALL_PRIOR_OUTLIER_MIN_THRESHOLD_MM,
        median_distance + BALL_PRIOR_OUTLIER_MAD_SCALE * mad,
    )
    keep_mask = distances <= threshold
    inlier_count = int(np.count_nonzero(keep_mask))
    if inlier_count < BALL_PRIOR_MIN_INLIER_COUNT:
        raise RuntimeError(
            "三球先验异常帧过多，拒绝写入 "
            f"inliers={inlier_count}/{BALL_PRIOR_SAMPLE_COUNT} threshold_mm={threshold:.3f}"
        )

    kept_samples = [
        detections
        for detections, keep in zip(ordered_samples, keep_mask, strict=True)
        if keep
    ]
    averaged_detections: list[BallDetectionInfo] = []
    hsv_ranges_by_color: dict[str, tuple[HsvRange, ...]] = {}
    observed_color_hex_by_color: dict[str, str] = {}
    for color_index, color_hex in enumerate(BALL_ORDERED_COLORS):
        detections = [items[color_index] for items in kept_samples]
        hsv_values = np.asarray(
            [item.observed_hsv for item in detections],
            dtype=np.float64,
        )
        observed_hsv, hsv_ranges = _aggregate_hsv_prior(hsv_values)
        template = detections[-1]
        averaged_detections.append(
            replace(
                template,
                center_px=_mean_tuple(item.center_px for item in detections),
                center_mm=_mean_tuple(item.center_mm for item in detections),
                diameter_mm=float(
                    np.mean([item.diameter_mm for item in detections])
                ),
                radius_px=float(np.mean([item.radius_px for item in detections])),
                center_norm=_mean_tuple(
                    item.center_norm for item in detections
                ),
                radius_norm=float(
                    np.mean([item.radius_norm for item in detections])
                ),
                point_count=int(
                    round(np.mean([item.point_count for item in detections]))
                ),
                observed_hsv=observed_hsv,
            )
        )
        hsv_ranges_by_color[color_hex] = hsv_ranges
        observed_color_hex_by_color[color_hex] = _hsv_to_hex(observed_hsv)

    averaged_tuple = tuple(averaged_detections)
    debug_artifacts = evidence_response.debug_artifacts
    if debug_artifacts:
        debug_artifacts = (
            replace(debug_artifacts[0], detections=averaged_tuple),
        )
    averaged_response = replace(
        evidence_response,
        elapsed_ms=float(np.mean([item.elapsed_ms for item in samples])),
        matched_count=len(averaged_tuple),
        detections=averaged_tuple,
        debug_artifacts=debug_artifacts,
    )
    logger.info(
        "三球先验异常剔除完成：采样帧={} 保留帧={} 剔除帧={} threshold_mm={:.3f}",
        len(samples),
        inlier_count,
        len(samples) - inlier_count,
        threshold,
    )
    return BallPriorAggregation(
        response=averaged_response,
        hsv_ranges_by_color=hsv_ranges_by_color,
        observed_color_hex_by_color=observed_color_hex_by_color,
        sample_count=len(samples),
        inlier_count=inlier_count,
    )


def _mean_tuple(values: Iterable[tuple[float, ...]]) -> tuple[float, ...]:
    """对等长数值元组求逐项均值。"""

    array = np.asarray(list(values), dtype=np.float64)
    return tuple(float(value) for value in np.mean(array, axis=0))


def _aggregate_hsv_prior(
    hsv_values: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[HsvRange, ...]]:
    """聚合多帧 HSV 中心并生成支持 Hue 环绕的窄范围。"""

    values = np.asarray(hsv_values, dtype=np.float64).reshape(-1, 3)
    initial_hue = _circular_hue_mean(values[:, 0])
    initial_center = np.asarray(
        [
            initial_hue,
            float(np.median(values[:, 1])),
            float(np.median(values[:, 2])),
        ],
        dtype=np.float64,
    )
    deviations = np.column_stack(
        [
            np.abs(
                ((values[:, 0] - initial_center[0] + 90.0) % 180.0) - 90.0
            ),
            np.abs(values[:, 1] - initial_center[1]),
            np.abs(values[:, 2] - initial_center[2]),
        ]
    )
    median_deviations = np.median(deviations, axis=0)
    deviation_mad = np.median(
        np.abs(deviations - median_deviations.reshape(1, 3)),
        axis=0,
    )
    thresholds = np.maximum(
        np.asarray(BALL_COLOR_OUTLIER_MIN_THRESHOLDS, dtype=np.float64),
        median_deviations + BALL_PRIOR_OUTLIER_MAD_SCALE * deviation_mad,
    )
    keep_mask = np.all(deviations <= thresholds.reshape(1, 3), axis=1)
    color_inlier_count = int(np.count_nonzero(keep_mask))
    minimum_color_inliers = int(
        np.ceil(values.shape[0] * BALL_COLOR_MIN_INLIER_RATIO)
    )
    if color_inlier_count < minimum_color_inliers:
        raise RuntimeError(
            "小球颜色异常帧过多，拒绝生成精确 HSV 范围 "
            f"inliers={color_inlier_count}/{values.shape[0]}"
        )
    values = values[keep_mask]
    hue = _circular_hue_mean(values[:, 0])
    saturation = float(np.mean(values[:, 1]))
    value = float(np.mean(values[:, 2]))
    hue_deviation = np.abs(((values[:, 0] - hue + 90.0) % 180.0) - 90.0)
    hue_tolerance = float(
        np.clip(
            np.quantile(hue_deviation, 0.90) + 2.0,
            *BALL_HUE_TOLERANCE_RANGE,
        )
    )
    saturation_tolerance = float(
        np.clip(
            np.quantile(np.abs(values[:, 1] - saturation), 0.90) + 8.0,
            *BALL_SATURATION_TOLERANCE_RANGE,
        )
    )
    value_tolerance = float(
        np.clip(
            np.quantile(np.abs(values[:, 2] - value), 0.90) + 10.0,
            *BALL_VALUE_TOLERANCE_RANGE,
        )
    )
    hsv_center = (hue, saturation, value)
    return hsv_center, _build_hsv_ranges(
        hsv_center,
        (hue_tolerance, saturation_tolerance, value_tolerance),
    )


def _circular_hue_mean(hue_values: np.ndarray) -> float:
    """计算 OpenCV Hue 周期为 180 的圆均值。"""

    hue_angles = np.asarray(hue_values, dtype=np.float64) * (
        2.0 * np.pi / 180.0
    )
    hue_angle = np.arctan2(
        np.mean(np.sin(hue_angles)),
        np.mean(np.cos(hue_angles)),
    )
    return float(
        (hue_angle % (2.0 * np.pi)) * 180.0 / (2.0 * np.pi)
    )


def _build_hsv_ranges(
    hsv_center: tuple[float, float, float],
    tolerances: tuple[float, float, float],
) -> tuple[HsvRange, ...]:
    """将 HSV 中心和半宽转换成 OpenCV 范围，必要时拆分 Hue 首尾。"""

    hue, saturation, value = hsv_center
    hue_tolerance, saturation_tolerance, value_tolerance = tolerances
    saturation_min = int(np.clip(np.floor(saturation - saturation_tolerance), 0, 255))
    saturation_max = int(np.clip(np.ceil(saturation + saturation_tolerance), 0, 255))
    value_min = int(np.clip(np.floor(value - value_tolerance), 0, 255))
    value_max = int(np.clip(np.ceil(value + value_tolerance), 0, 255))
    hue_min = hue - hue_tolerance
    hue_max = hue + hue_tolerance
    if hue_min < 0.0:
        return (
            (0, saturation_min, value_min, int(np.ceil(hue_max)), saturation_max, value_max),
            (int(np.floor(180.0 + hue_min)), saturation_min, value_min, 179, saturation_max, value_max),
        )
    if hue_max > 179.0:
        return (
            (0, saturation_min, value_min, int(np.ceil(hue_max - 180.0)), saturation_max, value_max),
            (int(np.floor(hue_min)), saturation_min, value_min, 179, saturation_max, value_max),
        )
    return (
        (
            int(np.floor(hue_min)),
            saturation_min,
            value_min,
            int(np.ceil(hue_max)),
            saturation_max,
            value_max,
        ),
    )


def _hsv_to_hex(hsv: tuple[float, float, float]) -> str:
    """将 OpenCV HSV 中心转换为便于人工查看的 RGB 十六进制颜色。"""

    pixel_hsv = np.asarray(
        [[[round(hsv[0]) % 180, round(hsv[1]), round(hsv[2])]]],
        dtype=np.uint8,
    )
    blue, green, red = cv2.cvtColor(pixel_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return f"#{int(red):02x}{int(green):02x}{int(blue):02x}"


def _save_ball_capture(
    output_dir: Path,
    aggregation: BallPriorAggregation,
    local_pose_transform: np.ndarray,
    tcp_snapshot: PoseSnapshot,
) -> None:
    response = aggregation.response
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
        "sample_count": aggregation.sample_count,
        "inlier_count": aggregation.inlier_count,
        "outlier_count": aggregation.sample_count - aggregation.inlier_count,
        "detections": [
            _serialize_detection(
                item,
                hsv_ranges=aggregation.hsv_ranges_by_color[item.color_hex],
                observed_color_hex=aggregation.observed_color_hex_by_color[
                    item.color_hex
                ],
            )
            for item in response.detections
        ],
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
    debug = _get_debug_artifact(response)
    if debug is None:
        raise RuntimeError("三球先验响应缺少 debug 产物，拒绝保存先验")
    if local_overlay_bgr is None:
        raise RuntimeError("三球先验无法构造本地坐标系 overlay，拒绝保存先验")
    _save_required_image(
        output_dir / "ball_color_bgr.jpg",
        np.asarray(debug.color_bgr, dtype=np.uint8),
    )
    _save_required_image(
        output_dir / "ball_depth.jpg",
        _build_depth_view(np.asarray(debug.depth_mm)),
    )
    _save_required_image(
        output_dir / "ball_debug_overlay.jpg",
        np.asarray(debug.overlay_bgr, dtype=np.uint8),
    )
    _save_required_image(
        output_dir / "ball_detection_overlay.jpg",
        np.asarray(debug.detection_overlay_bgr, dtype=np.uint8),
    )
    _save_required_image(output_dir / "ball_pose_overlay.jpg", local_overlay_bgr)
    (output_dir / "ball_pose_prior.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_required_image(path: Path, image: np.ndarray) -> None:
    """保存必须存在的先验核验图，失败时阻止先验 JSON 生效。"""

    if image.size == 0 or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"先验核验图片保存失败：{path}")


def _serialize_detection(
    item: BallDetectionInfo,
    *,
    hsv_ranges: tuple[HsvRange, ...] = (),
    observed_color_hex: str = "",
) -> dict[str, Any]:
    payload = {
        "color_hex": item.color_hex,
        "detected": item.detected,
        "center_px": list(item.center_px),
        "center_mm": list(item.center_mm),
        "diameter_mm": item.diameter_mm,
        "radius_px": item.radius_px,
        "center_norm": list(item.center_norm),
        "radius_norm": item.radius_norm,
        "point_count": item.point_count,
        "status": item.status,
        "observed_hsv": list(item.observed_hsv),
    }
    if hsv_ranges:
        payload["hsv_ranges"] = [list(item) for item in hsv_ranges]
        payload["observed_color_hex"] = observed_color_hex
    return payload


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


def _build_priors_from_capture(
    captured: dict[str, Any],
    *,
    require_recorded_prior: bool = False,
) -> list[BallPosePriorInfo]:
    """从采集文件构造三球先验；业务检测可要求完整相对位置和精确颜色。"""

    if require_recorded_prior:
        sample_count = captured.get("sample_count")
        inlier_count = captured.get("inlier_count")
        if sample_count != BALL_PRIOR_SAMPLE_COUNT:
            raise ValueError(
                "三球先验不是完整 30 帧记录："
                f"sample_count={sample_count!r}"
            )
        if not isinstance(inlier_count, int) or inlier_count < BALL_PRIOR_MIN_INLIER_COUNT:
            raise ValueError(
                "三球先验有效帧不足："
                f"inlier_count={inlier_count!r} "
                f"required={BALL_PRIOR_MIN_INLIER_COUNT}"
            )
    balls = captured.get("balls")
    recorded_balls = balls.get("ballinfo", []) if isinstance(balls, dict) else []
    if not isinstance(recorded_balls, list) or len(recorded_balls) < 3:
        recorded_balls = captured.get("detections", [])
    if not isinstance(recorded_balls, list) or len(recorded_balls) < 3:
        return _invalid_or_default_priors(
            require_recorded_prior,
            "三球先验缺少三个检测条目",
        )
    lookup = {str(item.get("color_hex")): item for item in recorded_balls if isinstance(item, dict)}
    yellow_item = lookup.get(BALL_ORDERED_COLORS[0])
    red_item = lookup.get(BALL_ORDERED_COLORS[1])
    purple_item = lookup.get(BALL_ORDERED_COLORS[2])
    if yellow_item is None or red_item is None or purple_item is None:
        return _invalid_or_default_priors(
            require_recorded_prior,
            "三球先验缺少黄、红、紫固定颜色条目",
        )
    ordered = (yellow_item, red_item, purple_item)
    origin = _prior_center_mm(ordered[0])
    second = _prior_center_mm(ordered[1])
    third = _prior_center_mm(ordered[2])
    if origin.shape != (3,) or second.shape != (3,) or third.shape != (3,):
        return _invalid_or_default_priors(
            require_recorded_prior,
            "三球先验球心维度无效",
        )
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(second)) or not np.all(np.isfinite(third)):
        return _invalid_or_default_priors(
            require_recorded_prior,
            "三球先验球心包含非有限值",
        )
    x_axis = second - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= GEOMETRY_EPSILON:
        return _invalid_or_default_priors(require_recorded_prior, "三球先验 X 轴退化")
    x_axis = x_axis / x_norm
    plane_hint = third - origin
    z_axis = np.cross(x_axis, plane_hint)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= GEOMETRY_EPSILON:
        return _invalid_or_default_priors(require_recorded_prior, "三球先验平面退化")
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= GEOMETRY_EPSILON:
        return _invalid_or_default_priors(require_recorded_prior, "三球先验 Y 轴退化")
    y_axis = y_axis / y_norm
    basis = np.stack([x_axis, y_axis, z_axis], axis=1)
    priors: list[BallPosePriorInfo] = []
    for item, color_hex in zip(ordered, BALL_ORDERED_COLORS, strict=True):
        position = _prior_center_mm(item)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            return _invalid_or_default_priors(
                require_recorded_prior,
                f"三球先验球心无效：color={color_hex}",
            )
        hsv_ranges = _parse_recorded_hsv_ranges(item.get("hsv_ranges"))
        if require_recorded_prior and not hsv_ranges:
            raise ValueError(f"三球先验缺少精确 HSV 范围：color={color_hex}")
        model_center = basis.T @ (position - origin)
        priors.append(
            BallPosePriorInfo(
                color_hex=color_hex,
                diameter_mm=float(
                    item.get("diameter_mm", BALL_DEFAULT_DIAMETER_MM)
                ),
                model_center_mm=tuple(model_center.tolist()),
                hsv_ranges=hsv_ranges,
            )
        )
    return priors


def _prior_center_mm(item: dict[str, Any]) -> np.ndarray:
    """读取旧采集摘要或当前先验文件中的相机系球心。"""

    return np.asarray(
        item.get("position_camera_mm", item.get("center_mm")),
        dtype=np.float64,
    )


def _parse_recorded_hsv_ranges(value: object) -> tuple[HsvRange, ...]:
    """读取先验文件中已经过聚合的每球 HSV 窄范围。"""

    if not isinstance(value, list):
        return ()
    ranges: list[HsvRange] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 6:
            return ()
        try:
            hsv_range: HsvRange = tuple(int(number) for number in item)  # type: ignore[assignment]
        except (TypeError, ValueError):
            return ()
        if not (
            0 <= hsv_range[0] <= hsv_range[3] <= 179
            and 0 <= hsv_range[1] <= hsv_range[4] <= 255
            and 0 <= hsv_range[2] <= hsv_range[5] <= 255
        ):
            return ()
        ranges.append(hsv_range)
    return tuple(ranges)


def _invalid_or_default_priors(
    require_recorded_prior: bool,
    message: str,
) -> list[BallPosePriorInfo]:
    """先验采集允许使用占位模型；业务检测必须显式暴露先验错误。"""

    if require_recorded_prior:
        raise ValueError(message)
    return _default_priors()


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
        logger.info("未找到先验对比目录，跳过对比：{}", prior_compare_dir)
        return
    prior_summary_path = prior_compare_dir / "summary.json"
    if not prior_summary_path.is_file():
        logger.warning("先验对比目录缺少 summary.json，跳过对比：{}", prior_summary_path)
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
        logger.info("当前输出目录缺少三球位姿图，跳过图像对比绘制：{}", current_overlay_path)
        return
    overlay_bgr = cv2.imread(str(current_overlay_path), cv2.IMREAD_COLOR)
    if overlay_bgr is None:
        logger.warning("当前三球位姿图读取失败，跳过图像对比绘制：{}", current_overlay_path)
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
            diameter_mm=BALL_DEFAULT_DIAMETER_MM,
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
    prior_intrinsics = prior_debug.get("camera_intrinsics") if isinstance(prior_debug, dict) else None
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
    client = CameraPipelineClient(service_addr=service_addr, timeout_ms=DEFAULT_TIMEOUT_MS)
    try:
        response = client.detect_charuco(
            CharucoDetectionRequest(
                camera_name=CameraName(DEFAULT_HEAD_CAMERA_NAME),
                dictionary_name=DEFAULT_DICTIONARY_NAME,
                squares_x=DEFAULT_SQUARES_X,
                squares_y=DEFAULT_SQUARES_Y,
                square_length_mm=float(DEFAULT_SQUARE_LENGTH_MM),
                marker_length_mm=float(DEFAULT_MARKER_LENGTH_MM),
                min_charuco_corners=min_charuco_corners,
                max_frames=300,
                stable_timeout_s=DEFAULT_CAMERA_TIMEOUT_S,
            )
        )
    finally:
        client.close()
    if response.status != "detected" or len(response.t_cam_board_mm) != 4:
        raise RuntimeError(
            "CameraPipeline 未检测到有效 ChArUco Board " f"markers={response.marker_num} charuco={response.charuco_num}"
        )
    transform_mm = np.asarray(response.t_cam_board_mm, dtype=np.float64).reshape(4, 4)
    translation_mm = transform_mm[:3, 3]
    rpy_deg = Rotation.from_matrix(transform_mm[:3, :3]).as_euler("xyz", degrees=True)
    payload = {
        "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
        "camera_name": response.camera_name,
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
        "marker_count": response.marker_num,
        "charuco_count": response.charuco_num,
        "reprojection_error_px": response.error_px,
        "camera_board_transform": transform_mm.tolist(),
        "translation_mm": translation_mm.tolist(),
        "rpy_deg": rpy_deg.tolist(),
    }
    result_path = output_dir / "charuco_board_prior.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("ChArUco Board 先验已由 CameraPipeline 检测并保存：{}", result_path)


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
    response = client.get_camera_intrinsics(
        CameraName.HEAD,
        timeout_s=DEFAULT_CAMERA_TIMEOUT_S,
    )
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
    reprojection = "NA" if pose_result.reprojection_error_px is None else f"{pose_result.reprojection_error_px:.3f}px"
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
