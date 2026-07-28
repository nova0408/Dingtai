from __future__ import annotations

from collections.abc import Callable, Iterable
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_pipeline.ball_pose_detection.protocol import (
    BallDetectionInfo,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPosePriorInfo,
)
from camera_pipeline.client import CameraName, CameraPipelineClient
from camera_pipeline.service.protocol import CharucoDetectionRequest
from src.wuji.ar5_client import Ar5Snapshot

# region 数据结构

HsvRange = tuple[int, int, int, int, int, int]
PriorProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class PriorCalibrationConfig:
    """先验标定固定配置。

    该结构集中定义三球采样、头部固定姿态、ChArUco 板规格和输出位置。配置本身不
    持有客户端或图像，允许后台任务安全共享。类不继承业务基类。
    """

    output_dir: Path
    "先验 JSON 与核验图片输出目录。"
    ball_sample_count: int = 30
    "三球先验所需完整且不同的帧数。"
    ball_max_attempts: int = 90
    "三球检测最大请求次数。"
    ball_min_inliers: int = 24
    "三球 MAD 异常剔除后最少保留帧数。"
    head_yaw_deg: float = 60.0
    "头部标定固定 Yaw，单位 deg。"
    head_pitch_deg: float = 45.0
    "头部标定固定 Pitch，单位 deg。"
    min_charuco_corners: int = 6
    "有效 ChArUco 检测所需最少角点数。"


@dataclass(frozen=True, slots=True)
class PriorCalibrationResult:
    """一次先验采集结果。

    结果只携带 GUI 展示和记录定位所需的信息，不持有 CameraPipeline 客户端。图像
    数组为 `(H, W, 3)`、`uint8` BGR；没有 overlay 时为 ``None``。
    """

    message: str
    "面向操作员的完成说明。"
    result_path: Path
    "已写入的先验 JSON 路径。"
    overlay_bgr: np.ndarray | None = None
    "三球先验核验图，形状 `(H, W, 3)`、dtype `uint8`、BGR。"


@dataclass(frozen=True, slots=True)
class _BallAggregation:
    """三球多帧聚合的内部结果。"""

    response: BallPoseDetectionResponse
    "聚合后的三球检测响应。"
    hsv_ranges: dict[str, tuple[HsvRange, ...]]
    "按参考颜色索引的 HSV 范围。"
    sample_count: int
    "聚合前完整帧数。"
    inlier_count: int
    "MAD 剔除后保留帧数。"


# endregion


# region 固定配置

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR_OUTPUT_DIR = PROJECT_ROOT / "record_replay" / "prior_data"
BALL_CAMERA_NAME = "left_hand_camera"
HEAD_CAMERA_NAME = "head_camera"
DEFAULT_BALL_COLORS = ("#ffff00", "#ff0000", "#00b3ff")
"GUI 与独立记录脚本共用的默认顺序：黄球、红球、蓝球。"
BALL_X_AXIS_INDEX = 0
"颜色顺序中用于指示坐标系 X 轴的球索引。"
BALL_ORIGIN_INDEX = 1
"颜色顺序中作为坐标原点的红球索引。"
BALL_PLANE_INDEX = 2
"颜色顺序中用于指示 XOY 平面的球索引。"
BALL_DIAMETER_MM = 20.0
BALL_MODEL_CENTERS_MM = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
OUTLIER_MAD_SCALE = 3.5
OUTLIER_MIN_THRESHOLD_MM = 2.0

# endregion


# region 先验记录器


class PriorCalibrationRecorder:
    """通过 CameraPipeline 记录左臂三球和头部 ChArUco 先验。

    职责边界：
    - 负责检测请求、三球多帧聚合、坐标系构造和结果落盘
    - 使用调用方提供的 AR5 快照记录采集时 TCP，不控制机械臂或头部运动
    - 不负责 Qt 控件、相机实时预览、SSH 转发和连接生命周期

    设计思想：
    - 复用 GUI 已连接的 CameraPipeline 客户端，避免重复建立硬件链路
    - 采用与 `test/wuji/prior_record.py` 相同的 30 帧完整采样和 MAD 异常剔除语义
    - 通过不可变结果对象把 overlay 交还 GUI

    生命周期与线程：
    - 不拥有传入客户端，``close`` 由连接 bundle 统一负责
    - 记录方法为阻塞调用，应由 GUI 后台线程执行
    - 类不继承业务基类
    """

    def __init__(
        self,
        client: CameraPipelineClient,
        config: PriorCalibrationConfig | None = None,
    ) -> None:
        """创建先验记录器。

        Parameters
        ----------
        client:
            已连接 CameraPipeline 服务的客户端，生命周期由调用方管理。
        config:
            采样与输出配置；不提供时写入项目标准先验目录。
        """

        self._client = client
        self._config = config or PriorCalibrationConfig(
            output_dir=DEFAULT_PRIOR_OUTPUT_DIR
        )

    def record_ball_prior(
        self,
        arm_snapshot: Ar5Snapshot,
        ball_colors: tuple[str, str, str] = DEFAULT_BALL_COLORS,
        progress: PriorProgressCallback | None = None,
    ) -> PriorCalibrationResult:
        """采集并保存左臂三球先验。

        Parameters
        ----------
        arm_snapshot:
            采集开始时的左 AR5 状态，关节角单位 deg、TCP 平移单位 mm。
        ball_colors:
            待检测球的 RGB HEX 颜色，顺序固定为 X 轴球、原点球、平面提示球。
        progress:
            后台进度回调，参数依次为有效帧数和目标帧数。

        Returns
        -------
        result:
            保存路径、完成说明和 BGR overlay。

        Raises
        ------
        RuntimeError
            完整帧不足、异常帧过多、坐标系退化或文件写入失败。
        """

        colors = _validate_ball_colors(ball_colors)
        samples, evidence = self._capture_ball_samples(colors, progress)
        aggregation = self._aggregate_ball_samples(samples, evidence, colors)
        transform = _build_ball_frame(aggregation.response.detections, colors)
        if transform is None:
            raise RuntimeError("三球坐标系退化，无法生成先验")
        return self._save_ball_prior(
            aggregation,
            transform,
            arm_snapshot,
            colors,
        )

    def record_head_prior(self) -> PriorCalibrationResult:
        """检测并保存头部相机 ChArUco 板先验。

        Returns
        -------
        result:
            ChArUco 先验保存路径和完成说明。

        Raises
        ------
        RuntimeError
            CameraPipeline 未返回有效板位姿。
        """

        cfg = self._config
        response = self._client.detect_charuco(
            CharucoDetectionRequest(
                camera_name=CameraName(HEAD_CAMERA_NAME),
                dictionary_name="DICT_APRILTAG_16H5",
                squares_x=4,
                squares_y=4,
                square_length_mm=20.0,
                marker_length_mm=14.0,
                min_charuco_corners=cfg.min_charuco_corners,
                max_frames=300,
                stable_timeout_s=10.0,
            )
        )
        if response.status != "detected" or len(response.t_cam_board_mm) != 4:
            raise RuntimeError(
                "未检测到有效 ChArUco Board："
                f"markers={response.marker_num}, charuco={response.charuco_num}"
            )
        transform = np.asarray(
            response.t_cam_board_mm,
            dtype=np.float64,
        ).reshape(4, 4)
        rpy_deg = Rotation.from_matrix(transform[:3, :3]).as_euler(
            "xyz",
            degrees=True,
        )
        payload = {
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "camera_name": str(response.camera_name),
            "pose_semantics": "T_camera_board",
            "translation_unit": "mm",
            "rotation_convention": 'scipy Rotation.as_euler("xyz", degrees=True)',
            "head_yaw_deg": cfg.head_yaw_deg,
            "head_pitch_deg": cfg.head_pitch_deg,
            "dictionary": "DICT_APRILTAG_16H5",
            "squares_x": 4,
            "squares_y": 4,
            "square_length_mm": 20,
            "marker_length_mm": 14,
            "marker_count": response.marker_num,
            "charuco_count": response.charuco_num,
            "reprojection_error_px": response.error_px,
            "camera_board_transform": transform.tolist(),
            "translation_mm": transform[:3, 3].tolist(),
            "rpy_deg": rpy_deg.tolist(),
        }
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = cfg.output_dir / "charuco_board_prior.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return PriorCalibrationResult(
            message=(
                "头部先验已保存 "
                f"(markers={response.marker_num}, charuco={response.charuco_num})"
            ),
            result_path=result_path,
        )

    def _capture_ball_samples(
        self,
        ball_colors: tuple[str, str, str],
        progress: PriorProgressCallback | None,
    ) -> tuple[list[BallPoseDetectionResponse], BallPoseDetectionResponse]:
        """采集完整且不同的三球检测帧。"""

        cfg = self._config
        priors = tuple(
            BallPosePriorInfo(
                color_hex=color,
                diameter_mm=BALL_DIAMETER_MM,
                model_center_mm=center,
            )
            for color, center in zip(
                ball_colors,
                BALL_MODEL_CENTERS_MM,
                strict=True,
            )
        )
        samples: list[BallPoseDetectionResponse] = []
        seen_ids: set[int] = set()
        evidence: BallPoseDetectionResponse | None = None
        for request_id in range(1, cfg.ball_max_attempts + 1):
            response = self._client.detect_ball(
                BallPoseDetectionRequest(
                    request_id=request_id,
                    camera_name=CameraName(BALL_CAMERA_NAME),
                    frame_id=-1,
                    enable_debug=True,
                    priors=priors,
                )
            )
            if response.frame_id in seen_ids or not _is_complete_ball_frame(
                response,
                ball_colors,
            ):
                continue
            seen_ids.add(response.frame_id)
            samples.append(replace(response, debug_artifacts=()))
            evidence = response
            if progress is not None:
                progress(len(samples), cfg.ball_sample_count)
            if len(samples) == cfg.ball_sample_count:
                break
        if len(samples) != cfg.ball_sample_count or evidence is None:
            raise RuntimeError(
                "三球先验完整帧不足："
                f"{len(samples)}/{cfg.ball_sample_count}，"
                f"最大尝试 {cfg.ball_max_attempts} 次"
            )
        return samples, evidence

    def _aggregate_ball_samples(
        self,
        samples: list[BallPoseDetectionResponse],
        evidence: BallPoseDetectionResponse,
        ball_colors: tuple[str, str, str],
    ) -> _BallAggregation:
        """对三球位置、直径和颜色执行 MAD 剔除及均值聚合。"""

        ordered = [
            tuple(
                next(item for item in response.detections if item.color_hex == color)
                for color in ball_colors
            )
            for response in samples
        ]
        # features: (N, 12) float64；每个球依次使用 XYZ(mm) 与直径(mm)。
        features = np.asarray(
            [
                [
                    value
                    for item in frame
                    for value in (*item.center_mm, item.diameter_mm)
                ]
                for frame in ordered
            ],
            dtype=np.float64,
        )
        median = np.median(features, axis=0)
        distances = np.linalg.norm(features - median.reshape(1, -1), axis=1)
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance))
        threshold = max(
            OUTLIER_MIN_THRESHOLD_MM,
            float(median_distance + OUTLIER_MAD_SCALE * mad),
        )
        keep_mask = distances <= threshold
        inlier_count = int(np.count_nonzero(keep_mask))
        if inlier_count < self._config.ball_min_inliers:
            raise RuntimeError(
                "三球先验异常帧过多："
                f"inliers={inlier_count}/{len(samples)}"
            )
        kept = [
            frame
            for frame, keep in zip(ordered, keep_mask, strict=True)
            if keep
        ]
        averaged: list[BallDetectionInfo] = []
        hsv_ranges: dict[str, tuple[HsvRange, ...]] = {}
        for color_index, color in enumerate(ball_colors):
            detections = [frame[color_index] for frame in kept]
            template = detections[-1]
            hsv_center = _mean_tuple(item.observed_hsv for item in detections)
            hsv_ranges[color] = _build_hsv_range(hsv_center)
            averaged.append(
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
                    observed_hsv=hsv_center,
                )
            )
        response = replace(
            evidence,
            detections=tuple(averaged),
            elapsed_ms=float(np.mean([item.elapsed_ms for item in samples])),
        )
        return _BallAggregation(
            response=response,
            hsv_ranges=hsv_ranges,
            sample_count=len(samples),
            inlier_count=inlier_count,
        )

    def _save_ball_prior(
        self,
        aggregation: _BallAggregation,
        transform: np.ndarray,
        arm_snapshot: Ar5Snapshot,
        ball_colors: tuple[str, str, str],
    ) -> PriorCalibrationResult:
        """保存三球 JSON 与 overlay 核验图。"""

        response = aggregation.response
        debug = response.debug_artifacts[0]
        overlay = np.asarray(debug.overlay_bgr, dtype=np.uint8).copy()
        rpy_deg = Rotation.from_matrix(transform[:3, :3]).as_euler(
            "xyz",
            degrees=True,
        )
        payload = {
            "frame_id": response.frame_id,
            "camera_name": response.camera_name,
            "matched_count": response.matched_count,
            "elapsed_ms": response.elapsed_ms,
            "sample_count": aggregation.sample_count,
            "inlier_count": aggregation.inlier_count,
            "outlier_count": aggregation.sample_count - aggregation.inlier_count,
            "ball_color_order": list(ball_colors),
            "local_coordinate_frame": {
                "origin_color": ball_colors[BALL_ORIGIN_INDEX],
                "x_axis_color": ball_colors[BALL_X_AXIS_INDEX],
                "xoy_plane_color": ball_colors[BALL_PLANE_INDEX],
            },
            "local_pose_transform": transform.tolist(),
            "local_pose_translation_mm": transform[:3, 3].tolist(),
            "local_pose_rotation": transform[:3, :3].tolist(),
            "local_pose_xyzrpy": [
                *transform[:3, 3].tolist(),
                *rpy_deg.tolist(),
            ],
            "detections": [
                {
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
                    "hsv_ranges": [
                        list(value)
                        for value in aggregation.hsv_ranges[item.color_hex]
                    ],
                }
                for item in response.detections
            ],
            "tcp_joint_deg": list(arm_snapshot.joint_deg),
            "tcp_translation_mm": list(arm_snapshot.xyz_mm),
            "tcp_rpy_degrees": list(arm_snapshot.rpy_deg),
            "tcp_elbow_deg": arm_snapshot.elbow_deg,
        }
        output_dir = self._config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = output_dir / "ball_debug_overlay.jpg"
        if overlay.size == 0 or not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(f"三球 overlay 保存失败：{overlay_path}")
        result_path = output_dir / "ball_pose_prior.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return PriorCalibrationResult(
            message=(
                "左臂三球先验已保存 "
                f"(有效 {aggregation.inlier_count}/{aggregation.sample_count} 帧)"
            ),
            result_path=result_path,
            overlay_bgr=overlay,
        )


# endregion


# region 聚合工具


def _validate_ball_colors(
    ball_colors: tuple[str, str, str],
) -> tuple[str, str, str]:
    """校验并规范化三球 RGB HEX 颜色。

    Parameters
    ----------
    ball_colors:
        RGB HEX 颜色，顺序为 X 轴球、原点球、平面提示球。

    Returns
    -------
    colors:
        转为小写后的三个 RGB HEX 颜色。

    Raises
    ------
    ValueError
        数量不是三个、格式不是 ``#rrggbb`` 或颜色重复。
    """

    if len(ball_colors) != 3:
        raise ValueError("三球先验必须提供三个颜色")
    colors = (
        ball_colors[0].lower(),
        ball_colors[1].lower(),
        ball_colors[2].lower(),
    )
    if any(re.fullmatch(r"#[0-9a-f]{6}", color) is None for color in colors):
        raise ValueError("三球颜色必须使用 #rrggbb 格式")
    if len(set(colors)) != 3:
        raise ValueError("三球颜色不能重复")
    return colors


def _is_complete_ball_frame(
    response: BallPoseDetectionResponse,
    ball_colors: tuple[str, str, str],
) -> bool:
    """判断响应是否包含调用方指定的三球完整检测和 overlay。

    Parameters
    ----------
    response:
        CameraPipeline 返回的球检测响应。
    ball_colors:
        RGB HEX 颜色，顺序为 X 轴球、原点球、平面提示球。

    Returns
    -------
    complete:
        三种颜色均得到三维位置且包含 overlay 时为 ``True``。
    """

    if response.matched_count != len(ball_colors) or len(response.debug_artifacts) != 1:
        return False
    if np.asarray(response.debug_artifacts[0].overlay_bgr).size == 0:
        return False
    by_color = {item.color_hex: item for item in response.detections}
    return all(
        color in by_color
        and by_color[color].detected
        and len(by_color[color].center_mm) == 3
        and len(by_color[color].observed_hsv) == 3
        for color in ball_colors
    )


def _mean_tuple(values: Iterable[tuple[float, ...]]) -> tuple[float, ...]:
    """对等长数值元组逐项求均值。"""

    array = np.asarray(list(values), dtype=np.float64)
    return tuple(float(value) for value in np.mean(array, axis=0))


def _build_hsv_range(hsv: tuple[float, ...]) -> tuple[HsvRange, ...]:
    """由聚合 HSV 中心生成保守的 OpenCV 检测范围。"""

    hue, saturation, value = hsv
    return (
        (
            int(np.clip(np.floor(hue - 5.0), 0, 179)),
            int(np.clip(np.floor(saturation - 30.0), 0, 255)),
            int(np.clip(np.floor(value - 35.0), 0, 255)),
            int(np.clip(np.ceil(hue + 5.0), 0, 179)),
            int(np.clip(np.ceil(saturation + 30.0), 0, 255)),
            int(np.clip(np.ceil(value + 35.0), 0, 255)),
        ),
    )


def _build_ball_frame(
    detections: tuple[BallDetectionInfo, ...],
    ball_colors: tuple[str, str, str],
) -> np.ndarray | None:
    """按原点球、X 轴球、平面提示球构造相机系变换。

    Parameters
    ----------
    detections:
        三球检测结果，球心坐标单位 mm。
    ball_colors:
        RGB HEX 颜色，顺序为 X 轴球、原点球、平面提示球。

    Returns
    -------
    transform:
        `T_camera_ball` 变换，平移单位 mm；三点退化或缺失时为 ``None``。
    """

    by_color = {
        item.color_hex: np.asarray(item.center_mm, dtype=np.float64)
        for item in detections
    }
    origin = by_color.get(ball_colors[BALL_ORIGIN_INDEX])
    x_axis_point = by_color.get(ball_colors[BALL_X_AXIS_INDEX])
    plane_point = by_color.get(ball_colors[BALL_PLANE_INDEX])
    if origin is None or x_axis_point is None or plane_point is None:
        return None
    x_axis = x_axis_point - origin
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-6:
        return None
    x_axis /= x_norm
    z_axis = np.cross(x_axis, plane_point - origin)
    z_norm = np.linalg.norm(z_axis)
    if z_norm <= 1e-6:
        return None
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack((x_axis, y_axis, z_axis), axis=1)
    transform[:3, 3] = origin
    return transform


# endregion
