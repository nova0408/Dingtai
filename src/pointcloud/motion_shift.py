from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# region 数据结构
@dataclass(frozen=True)
class PhaseShiftEstimate:
    """图像平移估计结果。

    该结构是 `estimate_phase_shift` 的稳定返回契约，供托盘掩码补偿、
    位姿辅助跟踪等调用方统一消费。

    设计思想：
    - 使用不可变 dataclass，确保跨线程读取时不会被调用方意外改写。
    - 同时输出平移量、相关性响应和有效标记，避免调用方重复推断状态。
    - 字段不持有图像或缓存对象，仅承载轻量数值结果。

    继承关系：
    - 不继承业务基类，避免引入隐式生命周期或动态分发。
    - 仅依赖 dataclass 生成初始化与只读字段约束。
    """

    dx_px: float
    "X 方向平移，单位 像素。正值表示当前图像相对参考图像向右平移。"
    dy_px: float
    "Y 方向平移，单位 像素。正值表示当前图像相对参考图像向下平移。"
    response: float
    "phase correlation 响应值，范围通常为 0-1，越大表示匹配越可信。"
    valid: bool
    "平移估计是否有效。`False` 表示应忽略 `dx_px/dy_px`。"
# endregion


# region 核心算法
def prepare_tracking_gray(image_bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """生成用于平移估计的灰度图。

    Parameters
    ----------
    image_bgr:
        输入 BGR 图像，形状为 `(H, W, 3)`，dtype 通常为 `uint8`，像素坐标系。
    max_side:
        生成跟踪图的最长边上限，单位 像素。取值小于原图最长边时会触发缩放。

    Returns
    -------
    gray:
        灰度跟踪图，形状为 `(Hs, Ws)`，dtype 为 `float32`。
    scale:
        从原图到跟踪图的缩放比例。`scale=1.0` 表示未缩放。

    Notes
    -----
    该函数仅做尺寸压缩与灰度化，不做去噪、均衡化或直方图处理。
    调用方可用返回的 `scale` 把小图平移量还原到原图像素单位。
    """
    h, w = image_bgr.shape[:2]
    long_side = max(1, h, w)
    scale = min(1.0, float(max_side) / float(long_side))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image_bgr
    # gray: (Hs, Ws) float32；phaseCorrelate 输入要求单通道浮点数组。
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, scale


def estimate_phase_shift(
    ref_gray: np.ndarray,
    cur_gray: np.ndarray,
    scale: float,
    min_response: float = 0.02,
    max_shift_px: float | None = None,
) -> PhaseShiftEstimate:
    """基于 phase correlation 估计图像平移。

    Parameters
    ----------
    ref_gray:
        参考灰度图，形状为 `(H, W)`，dtype 推荐为 `float32`。
    cur_gray:
        当前灰度图，形状为 `(H, W)`，dtype 推荐为 `float32`。
    scale:
        跟踪图相对原图的缩放比例。用于把小图平移量恢复为原图像素单位。
    min_response:
        最小响应阈值。响应低于该值时结果视为无效。
    max_shift_px:
        平移量限幅阈值，单位 像素。为 `None` 时不做限幅。

    Returns
    -------
    estimate:
        平移估计结果。`valid=False` 时应忽略平移量。

    Notes
    -----
    该函数只建模二维平移，不建模旋转和尺度变化。
    若 `ref_gray` 与 `cur_gray` 形状不一致，直接返回无效结果。
    """
    if ref_gray.shape != cur_gray.shape:
        return PhaseShiftEstimate(dx_px=0.0, dy_px=0.0, response=0.0, valid=False)
    try:
        # dx_small/dy_small: 小图坐标系平移（像素）；response: 相关峰值强度。
        (dx_small, dy_small), response = cv2.phaseCorrelate(
            np.asarray(ref_gray, dtype=np.float32), np.asarray(cur_gray, dtype=np.float32)
        )
    except cv2.error:
        return PhaseShiftEstimate(dx_px=0.0, dy_px=0.0, response=0.0, valid=False)
    if not np.isfinite(dx_small) or not np.isfinite(dy_small):
        return PhaseShiftEstimate(
            dx_px=0.0,
            dy_px=0.0,
            response=float(response) if np.isfinite(response) else 0.0,
            valid=False,
        )
    response_value = float(response) if np.isfinite(response) else 0.0
    if response_value < float(min_response):
        return PhaseShiftEstimate(dx_px=0.0, dy_px=0.0, response=response_value, valid=False)
    scale_safe = max(1e-6, float(scale))
    # 从小图平移恢复到原图像素坐标系。
    dx_px = float(dx_small) / scale_safe
    dy_px = float(dy_small) / scale_safe
    if max_shift_px is not None:
        limit = abs(float(max_shift_px))
        # 对原图像素平移做对称限幅，避免异常峰值污染后续补偿。
        dx_px = float(np.clip(dx_px, -limit, limit))
        dy_px = float(np.clip(dy_px, -limit, limit))
    return PhaseShiftEstimate(dx_px=dx_px, dy_px=dy_px, response=response_value, valid=True)


def warp_mask(mask: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    """按像素平移二值掩码。

    Parameters
    ----------
    mask:
        输入掩码，形状为 `(H, W)`，dtype 通常为 `uint8`，非零表示有效区域。
    dx_px:
        X 方向平移，单位 像素。正值表示向右平移。
    dy_px:
        Y 方向平移，单位 像素。正值表示向下平移。

    Returns
    -------
    shifted_mask:
        平移后的掩码，形状为 `(H, W)`，dtype 为 `uint8`。

    Notes
    -----
    该函数使用最近邻插值，保证掩码标签不被插值污染。
    """
    h, w = mask.shape[:2]
    # mat: (2, 3) float32 仿射矩阵，仅含平移分量。
    mat = np.asarray([[1.0, 0.0, float(dx_px)], [0.0, 1.0, float(dy_px)]], dtype=np.float32)
    return cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8),
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
# endregion
