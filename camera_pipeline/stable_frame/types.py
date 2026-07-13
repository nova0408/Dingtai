from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StableFrameConfig:
    """稳定帧判定配置。

    所有图像阈值都作用于缩放后的检测图像。默认值用于形成可运行的
    初始版本，部署前应使用真实静止和运动片段重新标定。
    """

    stable_duration_s: float = 1.0
    "连续稳定时间，单位 s。"

    image_scale: float = 0.25
    "检测图像相对原图的缩放比例。"

    max_frame_gap_ms: float = 150.0
    "连续帧允许的最大时间间隔，单位 ms。"

    color_mean_delta_threshold: float = 2.5
    "去除全局亮度偏移后的灰度平均变化阈值，范围 0 至 255。"

    color_changed_ratio_threshold: float = 0.02
    "灰度变化超过单像素阈值的最大像素比例。"

    color_pixel_delta_threshold: int = 12
    "判定单个灰度像素发生变化的阈值，范围 0 至 255。"

    min_valid_depth_ratio: float = 0.20
    "两帧共同有效深度像素的最小比例。"

    depth_median_delta_threshold_mm: float = 4.0
    "共同有效深度的绝对差中位数阈值，单位 mm。"

    depth_percentile: float = 75.0
    "深度高分位差使用的百分位数。"

    depth_percentile_delta_threshold_mm: float = 25.0
    "共同有效深度的高分位绝对差阈值，单位 mm。"
