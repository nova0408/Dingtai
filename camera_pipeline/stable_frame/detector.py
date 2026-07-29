from __future__ import annotations

# pyright: reportMissingImports=false

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from ..protocol import StableColorFrameProtocol, StableRgbdFrameProtocol
from .types import StableFrameConfig


@dataclass(frozen=True, slots=True)
class _FrameFeature:
    """单帧稳定性特征。"""

    frame_id: int
    timestamp_ms: float
    gray: npt.NDArray[np.float32]
    depth_mm: npt.NDArray[np.float32] | None


class StableFrameDetector:
    """根据连续 RGBD 帧输出稳定时间段中点的帧号。

    本类只保存稳定性判定所需的短时图像特征，不连接相机、不等待帧、
    不读取文件，也不保证返回帧仍存在于外部缓存中。
    """

    def __init__(self, config: StableFrameConfig | None = None) -> None:
        self._config = config or StableFrameConfig()
        self._validate_config()
        self._stable_features: deque[_FrameFeature] = deque()

    def reset(self) -> None:
        """清除当前连续稳定区间。"""

        self._stable_features.clear()

    def update(self, frame: StableRgbdFrameProtocol) -> int | None:
        """输入一个新帧，稳定时间足够时返回区间中点帧号。

        Parameters
        ----------
        frame:
            按采集时间顺序输入的 RGBD 相机帧。

        Returns
        -------
        int | None
            连续稳定区间达到配置时长时，返回区间时间中点附近的
            `frame_id`；尚未稳定时返回 ``None``。
        """

        return self._update_feature(self._extract_rgbd_feature(frame))

    def update_color(self, frame: StableColorFrameProtocol) -> int | None:
        """输入新彩色帧，仅按时间和 RGB 变化判断稳定。"""

        return self._update_feature(self._extract_color_feature(frame))

    def _update_feature(self, feature: _FrameFeature) -> int | None:
        """将已提取特征加入连续稳定窗口。"""

        if not self._stable_features:
            self._stable_features.append(feature)
            return None

        previous = self._stable_features[-1]
        if not self._is_temporally_continuous(
            previous, feature
        ) or not self._is_stable_pair(previous, feature):
            self._stable_features.clear()
            self._stable_features.append(feature)
            return None

        self._stable_features.append(feature)
        required_duration_ms = self._config.stable_duration_s * 1000.0
        if (
            feature.timestamp_ms - self._stable_features[0].timestamp_ms
            < required_duration_ms
        ):
            return None

        self._trim_to_latest_complete_window(required_duration_ms)
        return self._select_midpoint_frame_id(required_duration_ms)

    def _extract_rgbd_feature(
        self,
        frame: StableRgbdFrameProtocol,
    ) -> _FrameFeature:
        """提取 RGBD 稳定性特征。"""

        color_bgr = np.asarray(frame.color_bgr)
        depth_mm = np.asarray(frame.depth_mm)
        color_small, gray = self._extract_color_arrays(color_bgr)
        if depth_mm.ndim != 2 or depth_mm.shape != color_bgr.shape[:2]:
            raise ValueError("depth_mm must have the same (H, W) as color_bgr")
        depth_small = cv2.resize(
            depth_mm,
            dsize=(color_small.shape[1], color_small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        return _FrameFeature(
            frame_id=int(frame.frame_id),
            timestamp_ms=float(frame.timestamp_ms),
            gray=gray,
            depth_mm=depth_small.astype(np.float32),
        )

    def _extract_color_feature(
        self,
        frame: StableColorFrameProtocol,
    ) -> _FrameFeature:
        """提取不依赖深度图的彩色稳定性特征。"""

        color_bgr = np.asarray(frame.color_bgr)
        _, gray = self._extract_color_arrays(color_bgr)
        return _FrameFeature(
            frame_id=int(frame.frame_id),
            timestamp_ms=float(frame.timestamp_ms),
            gray=gray,
            depth_mm=None,
        )

    def _extract_color_arrays(
        self,
        color_bgr: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        """校验并缩放彩色图，返回缩放图与灰度特征。"""

        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError("color_bgr must have shape (H, W, 3)")
        interpolation = (
            cv2.INTER_AREA if self._config.image_scale < 1.0 else cv2.INTER_LINEAR
        )
        color_small = np.asarray(
            cv2.resize(
                color_bgr,
                dsize=None,
                fx=self._config.image_scale,
                fy=self._config.image_scale,
                interpolation=interpolation,
            ),
            dtype=np.uint8,
        )
        gray = cv2.cvtColor(color_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return color_small, gray

    def _is_temporally_continuous(
        self, previous: _FrameFeature, current: _FrameFeature
    ) -> bool:
        delta_ms = current.timestamp_ms - previous.timestamp_ms
        return 0.0 < delta_ms <= self._config.max_frame_gap_ms

    def _is_stable_pair(self, previous: _FrameFeature, current: _FrameFeature) -> bool:
        gray_delta = current.gray - previous.gray
        gray_delta -= float(np.median(gray_delta))
        absolute_gray_delta = np.abs(gray_delta)
        color_mean_delta = float(np.mean(absolute_gray_delta))
        color_changed_ratio = float(
            np.mean(absolute_gray_delta > self._config.color_pixel_delta_threshold)
        )
        if color_mean_delta > self._config.color_mean_delta_threshold:
            return False
        if color_changed_ratio > self._config.color_changed_ratio_threshold:
            return False

        if previous.depth_mm is None or current.depth_mm is None:
            return previous.depth_mm is None and current.depth_mm is None
        valid_depth = (previous.depth_mm > 0.0) & (current.depth_mm > 0.0)
        valid_depth_ratio = float(np.mean(valid_depth))
        if valid_depth_ratio < self._config.min_valid_depth_ratio:
            return False
        depth_delta = np.abs(
            current.depth_mm[valid_depth] - previous.depth_mm[valid_depth]
        )
        if float(np.median(depth_delta)) > self._config.depth_median_delta_threshold_mm:
            return False
        return bool(
            np.percentile(depth_delta, self._config.depth_percentile)
            <= self._config.depth_percentile_delta_threshold_mm
        )

    def _trim_to_latest_complete_window(self, required_duration_ms: float) -> None:
        while (
            len(self._stable_features) >= 3
            and self._stable_features[-1].timestamp_ms
            - self._stable_features[1].timestamp_ms
            >= required_duration_ms
        ):
            self._stable_features.popleft()

    def _select_midpoint_frame_id(self, required_duration_ms: float) -> int:
        target_timestamp_ms = (
            self._stable_features[0].timestamp_ms + required_duration_ms / 2.0
        )
        midpoint = min(
            self._stable_features,
            key=lambda item: (
                abs(item.timestamp_ms - target_timestamp_ms),
                item.timestamp_ms,
            ),
        )
        return midpoint.frame_id

    def _validate_config(self) -> None:
        if self._config.stable_duration_s <= 0.0:
            raise ValueError("stable_duration_s must be greater than zero")
        if self._config.image_scale <= 0.0:
            raise ValueError("image_scale must be greater than zero")
        if self._config.max_frame_gap_ms <= 0.0:
            raise ValueError("max_frame_gap_ms must be greater than zero")
        if not 0.0 <= self._config.color_changed_ratio_threshold <= 1.0:
            raise ValueError("color_changed_ratio_threshold must be in [0, 1]")
        if not 0.0 <= self._config.min_valid_depth_ratio <= 1.0:
            raise ValueError("min_valid_depth_ratio must be in [0, 1]")
        if not 0.0 <= self._config.depth_percentile <= 100.0:
            raise ValueError("depth_percentile must be in [0, 100]")
