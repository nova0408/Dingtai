"""三球检测结果的排序、鲁棒聚合与坐标系构造。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .settings import ReplayOffsetSettings

# region 核心算法


def ordered_three_ball_centers(
    detections: tuple[tuple[str, tuple[float, float, float]], ...],
    settings: ReplayOffsetSettings,
) -> npt.NDArray[np.float64] | None:
    """将检测结果按黄、红、紫排序为 `(3, 3)` mm 坐标数组。"""

    by_color = {color: np.asarray(center, dtype=np.float64) for color, center in detections}
    centers: list[npt.NDArray[np.float64]] = []
    for color in settings.ordered_ball_colors:
        center = by_color.get(color)
        if center is None or center.shape != (3,) or not np.all(np.isfinite(center)):
            return None
        centers.append(center)
    return np.stack(centers, axis=0)


def build_three_ball_basis_transform(centers_mm: npt.ArrayLike) -> npt.NDArray[np.float64] | None:
    """以黄球为原点、红球为 X 轴、紫球为平面提示构造 4x4 变换。"""

    centers = np.asarray(centers_mm, dtype=np.float64)
    if centers.shape != (3, 3) or not np.all(np.isfinite(centers)):
        return None
    origin, red, purple = centers
    x_axis = red - origin
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1e-6:
        return None
    x_axis /= x_norm
    z_axis = np.cross(x_axis, purple - origin)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1e-6:
        return None
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm <= 1e-6:
        return None
    y_axis /= y_norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack((x_axis, y_axis, z_axis), axis=1)
    matrix[:3, 3] = origin
    return matrix


def robust_mean_three_ball_centers(
    samples_mm: Sequence[npt.ArrayLike],
    settings: ReplayOffsetSettings,
) -> npt.NDArray[np.float64]:
    """对多个 `(3, 3)` mm 三球样本做 9 维 MAD 剔除和均值聚合。"""

    if not samples_mm:
        raise ValueError("三球样本不能为空")
    stack = np.stack([np.asarray(sample, dtype=np.float64) for sample in samples_mm], axis=0)
    if stack.ndim != 3 or stack.shape[1:] != (3, 3) or not np.all(np.isfinite(stack)):
        raise ValueError(f"三球样本必须为 (N, 3, 3)，实际为 {stack.shape}")
    flattened = stack.reshape(stack.shape[0], 9)
    median = np.median(flattened, axis=0)
    distances = np.linalg.norm(flattened - median.reshape(1, 9), axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    keep_mask = distances <= max(
        settings.min_outlier_threshold_mm,
        median_distance + settings.mad_scale * mad,
    )
    if not np.any(keep_mask):
        keep_mask[int(np.argmin(distances))] = True
    return np.mean(stack[keep_mask], axis=0)


def camera_ball_transform_m(
    samples_mm: Sequence[npt.ArrayLike],
    settings: ReplayOffsetSettings,
) -> npt.NDArray[np.float64]:
    """将鲁棒聚合的三球坐标构造成平移单位为 m 的 T_cam_ball。"""

    transform_mm = build_three_ball_basis_transform(robust_mean_three_ball_centers(samples_mm, settings))
    if transform_mm is None:
        raise RuntimeError("均值三球基础坐标系构造失败")
    transform_m = transform_mm.copy()
    transform_m[:3, 3] *= 0.001
    return transform_m


# endregion
