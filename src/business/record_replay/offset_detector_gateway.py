"""基于 camera_pipeline 的三球检测适配器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from loguru import logger

from camera_pipeline.ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPosePriorInfo,
)
from camera_pipeline.client import CameraPipelineClient

from .offset_detection import ordered_three_ball_centers
from .settings import ReplayOffsetSettings

# region 接口


def load_three_ball_priors(
    prior_capture_path: Path,
    settings: ReplayOffsetSettings,
) -> tuple[BallPosePriorInfo, ...]:
    """按旧先验优先级重建黄、红、紫三球的模型坐标先验。"""

    payload = json.loads(prior_capture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return _default_three_ball_priors()
    recorded_balls = _extract_recorded_balls(payload)
    values = _extract_ball_values(recorded_balls)
    ordered = ordered_three_ball_centers(
        tuple((color, values[color][1]) for color in values),
        settings,
    )
    if ordered is None:
        logger.warning("先验三球无效，使用旧流程默认先验 path={}", prior_capture_path)
        return _default_three_ball_priors()
    origin = ordered[0]
    x_axis = ordered[1] - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        return _default_three_ball_priors()
    x_axis /= x_norm
    z_axis = np.cross(x_axis, ordered[2] - origin)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6:
        return _default_three_ball_priors()
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-6:
        return _default_three_ball_priors()
    y_axis /= y_norm
    basis = np.stack((x_axis, y_axis, z_axis), axis=1)
    return tuple(
        BallPosePriorInfo(color, values[color][0], tuple((basis.T @ (ordered[index] - origin)).tolist()))
        for index, color in enumerate(("#ffff00", "#ff0000", "#ff00ff"))
    )


def _extract_recorded_balls(payload: dict[str, object]) -> list[object]:
    """优先读取旧 `balls.ballinfo`，再兼容独立采集脚本的 `detections`。"""

    balls = payload.get("balls")
    if isinstance(balls, dict) and isinstance(balls.get("ballinfo"), list):
        return balls["ballinfo"]
    detections = payload.get("detections")
    return detections if isinstance(detections, list) else []


def _extract_ball_values(items: list[object]) -> dict[str, tuple[float, tuple[float, float, float]]]:
    """从先验条目提取颜色、半径和相机系球心。"""

    values: dict[str, tuple[float, tuple[float, float, float]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        color = item.get("color_hex")
        center = item.get("position_camera_mm", item.get("center_mm"))
        if not isinstance(color, str) or not isinstance(center, list) or len(center) != 3:
            continue
        try:
            center_tuple = (float(center[0]), float(center[1]), float(center[2]))
            radius_mm = float(item.get("radius_mm", 20.0))
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(np.asarray(center_tuple, dtype=np.float64))) and np.isfinite(radius_mm):
            values[color] = (radius_mm, center_tuple)
    return values


def _default_three_ball_priors() -> tuple[BallPosePriorInfo, ...]:
    """返回旧三球检测流程在先验不可用时采用的默认模型。"""

    return (
        BallPosePriorInfo("#ffff00", 20.0, (0.0, 0.0, 0.0)),
        BallPosePriorInfo("#ff0000", 20.0, (1.0, 0.0, 0.0)),
        BallPosePriorInfo("#ff00ff", 20.0, (0.0, 1.0, 0.0)),
    )


class ThreeBallDetector(Protocol):
    """为 offset 计算提供三球 `(3, 3)` mm 样本的检测接口。"""

    def capture_samples(self, sample_count: int) -> list[tuple[tuple[float, float, float], ...]]:
        """采集多个按黄、红、紫排序的三球坐标样本。"""

        ...


@dataclass(frozen=True, slots=True)
class CameraPipelineThreeBallDetector:
    """将 camera_pipeline 的球位姿 RPC 适配为业务三球样本。"""

    service_addr: str
    "camera_pipeline RPC 地址。"
    camera_name: str
    "逻辑相机名称。"
    priors: tuple[BallPosePriorInfo, ...]
    "检测请求使用的三球先验。"
    settings: ReplayOffsetSettings
    "三球检测与鲁棒聚合参数。"

    def capture_samples(self, sample_count: int) -> list[tuple[tuple[float, float, float], ...]]:
        """连续请求三球检测并只返回完整有效的三球样本。"""

        client = CameraPipelineClient(self.service_addr, timeout_ms=self.settings.detection_timeout_ms)
        samples: list[tuple[tuple[float, float, float], ...]] = []
        try:
            for request_id in range(1, sample_count + 1):
                try:
                    response = client.request_ball_pose_detection(
                        BallPoseDetectionRequest(
                            request_id=request_id,
                            camera_name=self.camera_name,
                            frame_id=-1,
                            enable_debug=False,
                            priors=self.priors,
                        )
                    )
                except RuntimeError as error:
                    logger.warning("三球检测请求失败，跳过当前采样 index={} error={}", request_id, error)
                    continue
                detections = tuple(
                    (item.color_hex, (float(item.center_mm[0]), float(item.center_mm[1]), float(item.center_mm[2])))
                    for item in response.detections
                    if item.detected and len(item.center_mm) == 3
                )
                centers = ordered_three_ball_centers(detections, self.settings)
                if centers is not None:
                    samples.append(
                        (
                            (float(centers[0, 0]), float(centers[0, 1]), float(centers[0, 2])),
                            (float(centers[1, 0]), float(centers[1, 1]), float(centers[1, 2])),
                            (float(centers[2, 0]), float(centers[2, 1]), float(centers[2, 2])),
                        )
                    )
        finally:
            client.close()
        if not samples:
            raise RuntimeError("连续采样未得到可用三球检测结果")
        return samples


# endregion
