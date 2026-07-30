"""基于 camera_pipeline 的三球检测适配器。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from camera_pipeline.ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPosePriorInfo,
)
from camera_pipeline.client import CameraName, CameraPipelineClient

from .offset_detection import ordered_three_ball_centers
from .settings import ReplayOffsetSettings

# region 接口

HsvRange = tuple[int, int, int, int, int, int]
PRIOR_SAMPLE_COUNT = 30
"先验记录要求的完整且不同帧数量。"
PRIOR_MIN_INLIER_COUNT = 24
"先验异常剔除后要求的最少保留帧数量。"
RUNTIME_NARROW_HUE_TOLERANCE = 8.0
"运行期窄 HSV Hue 半宽，单位 OpenCV Hue。"
RUNTIME_SATURATION_MIN = 140
"运行期窄 HSV 保留完整球面颜色的 Saturation 下限。"
RUNTIME_VALUE_MIN = 120
"运行期窄 HSV 保留完整球面明暗区域的 Value 下限。"


def load_three_ball_priors(
    prior_capture_path: Path,
) -> tuple[BallPosePriorInfo, ...]:
    """从 30 帧记录结果及其坐标语义重建三球模型先验。"""

    debug_overlay_path = prior_capture_path.with_name("ball_debug_overlay.jpg")
    if not debug_overlay_path.is_file():
        raise FileNotFoundError(f"三球先验缺少 debug overlay：{debug_overlay_path}")
    payload = json.loads(prior_capture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"三球先验根节点必须是对象：{prior_capture_path}")
    if payload.get("sample_count") != PRIOR_SAMPLE_COUNT:
        raise ValueError(
            "三球先验不是完整 30 帧记录："
            f"sample_count={payload.get('sample_count')!r} path={prior_capture_path}"
        )
    inlier_count = payload.get("inlier_count")
    if not isinstance(inlier_count, int) or inlier_count < PRIOR_MIN_INLIER_COUNT:
        raise ValueError(
            "三球先验有效帧不足："
            f"inlier_count={inlier_count!r} required={PRIOR_MIN_INLIER_COUNT} "
            f"path={prior_capture_path}"
        )
    recorded_balls = _extract_recorded_balls(payload)
    values = _extract_ball_values(recorded_balls)
    required_colors = _extract_coordinate_colors(payload)
    missing_colors = [color for color in required_colors if color not in values]
    if missing_colors:
        raise ValueError(
            f"三球先验缺少有效颜色条目 colors={missing_colors} path={prior_capture_path}"
        )
    missing_hsv = [color for color in required_colors if not values[color][2]]
    if missing_hsv:
        raise ValueError(
            f"三球先验缺少 30 帧标定 HSV 范围 colors={missing_hsv} "
            f"path={prior_capture_path}"
        )
    ordered = ordered_three_ball_centers(
        tuple((color, values[color][1]) for color in values),
        required_colors,
    )
    if ordered is None:
        raise ValueError(f"三球先验坐标无效：{prior_capture_path}")
    origin = ordered[0]
    x_axis = ordered[1] - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        raise ValueError(f"三球先验 X 轴退化：{prior_capture_path}")
    x_axis /= x_norm
    z_axis = np.cross(x_axis, ordered[2] - origin)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6:
        raise ValueError(f"三球先验平面退化：{prior_capture_path}")
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-6:
        raise ValueError(f"三球先验 Y 轴退化：{prior_capture_path}")
    y_axis /= y_norm
    basis = np.stack((x_axis, y_axis, z_axis), axis=1)
    return tuple(
        BallPosePriorInfo(
            color_hex=color,
            diameter_mm=values[color][0],
            model_center_mm=tuple(
                (basis.T @ (ordered[index] - origin)).tolist()
            ),
            hsv_ranges=values[color][2],
        )
        for index, color in enumerate(required_colors)
    )


def _extract_coordinate_colors(
    payload: dict[str, object],
) -> tuple[str, str, str]:
    """读取先验文件声明的原点、X 轴和平面提示球颜色。"""

    frame = payload.get("local_coordinate_frame")
    if not isinstance(frame, dict):
        raise ValueError("三球先验缺少 local_coordinate_frame")
    origin = frame.get("origin_color")
    x_axis = frame.get("x_axis_color")
    plane = frame.get("xoy_plane_color")
    if (
        not isinstance(origin, str)
        or not isinstance(x_axis, str)
        or not isinstance(plane, str)
    ):
        raise ValueError("三球先验坐标语义必须提供三个 HEX 颜色")
    colors = (origin, x_axis, plane)
    if len(set(colors)) != 3:
        raise ValueError("三球先验坐标语义中的颜色不能重复")
    return colors


def _extract_recorded_balls(payload: dict[str, object]) -> list[object]:
    """优先读取旧 `balls.ballinfo`，再兼容独立采集脚本的 `detections`。"""

    balls = payload.get("balls")
    if isinstance(balls, dict) and isinstance(balls.get("ballinfo"), list):
        return balls["ballinfo"]
    detections = payload.get("detections")
    return detections if isinstance(detections, list) else []


def _extract_ball_values(
    items: list[object],
) -> dict[
    str,
    tuple[float, tuple[float, float, float], tuple[HsvRange, ...]],
]:
    """从先验条目提取颜色、直径和相机系球心。"""

    values: dict[
        str,
        tuple[float, tuple[float, float, float], tuple[HsvRange, ...]],
    ] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        color = item.get("color_hex")
        center = item.get("position_camera_mm", item.get("center_mm"))
        if not isinstance(color, str) or not isinstance(center, list) or len(center) != 3:
            continue
        try:
            center_tuple = (float(center[0]), float(center[1]), float(center[2]))
            diameter_mm = float(item["diameter_mm"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(
            np.isfinite(np.asarray(center_tuple, dtype=np.float64))
        ) and np.isfinite(diameter_mm):
            values[color] = (
                diameter_mm,
                center_tuple,
                _runtime_hsv_ranges(item),
            )
    return values


def _runtime_hsv_ranges(item: dict[str, object]) -> tuple[HsvRange, ...]:
    """从实测 Hue 构造运行期窄范围，并保留完整球面的 S/V 区域。"""

    observed_hsv = item.get("observed_hsv")
    if isinstance(observed_hsv, list) and len(observed_hsv) == 3:
        try:
            hue = float(observed_hsv[0])
        except (TypeError, ValueError):
            hue = float("nan")
        if np.isfinite(hue) and 0.0 <= hue < 180.0:
            return _build_runtime_hue_ranges(hue)
    recorded = _parse_hsv_ranges(item.get("hsv_ranges"))
    return tuple(
        (
            hsv_range[0],
            min(hsv_range[1], RUNTIME_SATURATION_MIN),
            min(hsv_range[2], RUNTIME_VALUE_MIN),
            hsv_range[3],
            255,
            255,
        )
        for hsv_range in recorded
    )


def _build_runtime_hue_ranges(hue: float) -> tuple[HsvRange, ...]:
    """围绕实测 Hue 构造支持 179/0 环绕的一段或两段范围。"""

    hue_min = hue - RUNTIME_NARROW_HUE_TOLERANCE
    hue_max = hue + RUNTIME_NARROW_HUE_TOLERANCE
    if hue_min < 0.0:
        return (
            (
                0,
                RUNTIME_SATURATION_MIN,
                RUNTIME_VALUE_MIN,
                int(np.ceil(hue_max)),
                255,
                255,
            ),
            (
                int(np.floor(180.0 + hue_min)),
                RUNTIME_SATURATION_MIN,
                RUNTIME_VALUE_MIN,
                179,
                255,
                255,
            ),
        )
    if hue_max > 179.0:
        return (
            (
                0,
                RUNTIME_SATURATION_MIN,
                RUNTIME_VALUE_MIN,
                int(np.ceil(hue_max - 180.0)),
                255,
                255,
            ),
            (
                int(np.floor(hue_min)),
                RUNTIME_SATURATION_MIN,
                RUNTIME_VALUE_MIN,
                179,
                255,
                255,
            ),
        )
    return (
        (
            int(np.floor(hue_min)),
            RUNTIME_SATURATION_MIN,
            RUNTIME_VALUE_MIN,
            int(np.ceil(hue_max)),
            255,
            255,
        ),
    )


def _parse_hsv_ranges(value: object) -> tuple[HsvRange, ...]:
    """校验先验 JSON 中的每球专属 HSV 范围。"""

    if not isinstance(value, list):
        return ()
    parsed: list[HsvRange] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 6:
            return ()
        try:
            numbers: HsvRange = (
                int(item[0]),
                int(item[1]),
                int(item[2]),
                int(item[3]),
                int(item[4]),
                int(item[5]),
            )
        except (TypeError, ValueError):
            return ()
        if not (
            0 <= numbers[0] <= numbers[3] <= 179
            and 0 <= numbers[1] <= numbers[4] <= 255
            and 0 <= numbers[2] <= numbers[5] <= 255
        ):
            return ()
        parsed.append(numbers)
    return tuple(parsed)


class ThreeBallDetector(Protocol):
    """为 offset 计算提供三球 `(3, 3)` mm 样本的检测接口。"""

    def capture_samples(self, sample_count: int) -> list[tuple[tuple[float, float, float], ...]]:
        """采集多个按先验坐标语义排序的三球坐标样本。"""

        ...


@dataclass(frozen=True, slots=True)
class CameraPipelineThreeBallDetector:
    """将 camera_pipeline 的宽窄分级球位姿 RPC 适配为业务三球样本。"""

    camera_name: CameraName
    "逻辑相机名称。"
    priors: tuple[BallPosePriorInfo, ...]
    "检测请求使用的三球先验。"
    settings: ReplayOffsetSettings
    "三球检测与鲁棒聚合参数。"
    service_addr: str = "tcp://127.0.0.1:6200"
    "CameraPipeline 服务地址；正式服务固定使用 Orin 本机地址。"

    def capture_samples(self, sample_count: int) -> list[tuple[tuple[float, float, float], ...]]:
        """先用宽 HSV 确认三球，再以一致的窄 HSV 结果提高精度。

        窄范围未检出或与宽范围球心差异过大时回退宽范围结果。宽范围仍携带相同的
        物理直径和三球模型坐标，因此回退不会绕过尺寸与几何约束。
        """

        client = CameraPipelineClient(
            service_addr=self.service_addr,
            timeout_ms=self.settings.detection_timeout_ms,
        )
        samples: list[tuple[tuple[float, float, float], ...]] = []
        ordered_colors = tuple(prior.color_hex for prior in self.priors)
        if len(ordered_colors) != 3:
            raise RuntimeError("三球检测先验必须恰好包含三个颜色")
        wide_priors = tuple(
            replace(prior, hsv_ranges=())
            for prior in self.priors
        )
        try:
            for sample_index in range(1, sample_count + 1):
                wide_centers = self._detect_centers_with_retries(
                    client,
                    request_id_base=sample_index * 100,
                    priors=wide_priors,
                    ordered_colors=ordered_colors,
                    mode="wide",
                )
                if wide_centers is None:
                    logger.warning(
                        "宽 HSV 三球检测失败，跳过当前采样 index={}",
                        sample_index,
                    )
                    continue
                narrow_centers = self._detect_centers_with_retries(
                    client,
                    request_id_base=sample_index * 100 + 50,
                    priors=self.priors,
                    ordered_colors=ordered_colors,
                    mode="narrow",
                )
                selected_centers = wide_centers
                if narrow_centers is None:
                    logger.warning(
                        "窄 HSV 三球检测失败，回退宽 HSV 结果 index={}",
                        sample_index,
                    )
                else:
                    center_deltas_mm = np.linalg.norm(
                        narrow_centers - wide_centers,
                        axis=1,
                    )
                    max_delta_mm = float(np.max(center_deltas_mm))
                    if (
                        max_delta_mm
                        <= self.settings.narrow_consistency_tolerance_mm
                    ):
                        selected_centers = narrow_centers
                        logger.success(
                            "宽窄 HSV 三球检测一致，采用窄范围结果 "
                            "index={} max_center_delta_mm={:.3f}",
                            sample_index,
                            max_delta_mm,
                        )
                    else:
                        logger.warning(
                            "窄 HSV 球心与宽 HSV 不一致，回退宽范围结果 "
                            "index={} max_center_delta_mm={:.3f} tolerance_mm={:.3f}",
                            sample_index,
                            max_delta_mm,
                            self.settings.narrow_consistency_tolerance_mm,
                        )
                samples.append(_centers_to_sample(selected_centers))
        finally:
            client.close()
        if not samples:
            raise RuntimeError(
                "宽 HSV 三球检测在全部尝试中均未得到完整结果 "
                f"requested_samples={sample_count} "
                f"attempts_per_sample={self.settings.detection_attempts_per_sample}"
            )
        return samples

    def _detect_centers_with_retries(
        self,
        client: CameraPipelineClient,
        *,
        request_id_base: int,
        priors: tuple[BallPosePriorInfo, ...],
        ordered_colors: tuple[str, ...],
        mode: str,
    ) -> np.ndarray | None:
        """在限定次数内获取一次完整三球结果。"""

        for attempt in range(1, self.settings.detection_attempts_per_sample + 1):
            centers = self._detect_centers(
                client,
                request_id=request_id_base + attempt,
                priors=priors,
                ordered_colors=ordered_colors,
                mode=mode,
            )
            if centers is not None:
                return centers
            logger.warning(
                "三球检测未完整，准备重试 mode={} attempt={}/{}",
                mode,
                attempt,
                self.settings.detection_attempts_per_sample,
            )
        return None

    def _detect_centers(
        self,
        client: CameraPipelineClient,
        *,
        request_id: int,
        priors: tuple[BallPosePriorInfo, ...],
        ordered_colors: tuple[str, ...],
        mode: str,
    ) -> np.ndarray | None:
        """执行一次检测并按先验声明的颜色顺序返回 `(3, 3)` mm 球心。"""

        try:
            response = client.detect_ball(
                BallPoseDetectionRequest(
                    request_id=request_id,
                    camera_name=self.camera_name,
                    frame_id=-1,
                    enable_debug=False,
                    priors=priors,
                )
            )
        except RuntimeError as error:
            logger.warning(
                "三球检测请求失败 mode={} request_id={} error={}",
                mode,
                request_id,
                error,
            )
            return None
        detections = tuple(
            (
                item.color_hex,
                (
                    float(item.center_mm[0]),
                    float(item.center_mm[1]),
                    float(item.center_mm[2]),
                ),
            )
            for item in response.detections
            if item.detected and len(item.center_mm) == 3
        )
        colors = (
            ordered_colors[0],
            ordered_colors[1],
            ordered_colors[2],
        )
        centers = ordered_three_ball_centers(detections, colors)
        status_by_color = {
            item.color_hex: item.status
            for item in response.detections
        }
        logger.info(
            "三球检测响应 mode={} request_id={} frame_id={} matched_count={} "
            "complete={} status_by_color={}",
            mode,
            request_id,
            response.frame_id,
            response.matched_count,
            centers is not None,
            status_by_color,
        )
        return centers


def _centers_to_sample(
    centers: np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    """把 `(3, 3)` mm 数组转换为不可变三球样本。"""

    return (
        (float(centers[0, 0]), float(centers[0, 1]), float(centers[0, 2])),
        (float(centers[1, 0]), float(centers[1, 1]), float(centers[1, 2])),
        (float(centers[2, 0]), float(centers[2, 1]), float(centers[2, 2])),
    )


# endregion
