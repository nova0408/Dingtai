# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import open3d as o3d
from loguru import logger
from matplotlib import pyplot as plt


# region 周期着色
def colorize_by_cycle(
    pcd: o3d.geometry.PointCloud,
    cycle: float = 2.0,
    axis: int = 2,
    color_map: str = "hsv",
    lut_size: int = 1024,
) -> None:
    """按指定轴做周期着色，放大小幅高度起伏。"""
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return
    target_coords = points[:, axis]
    normalized_cycle = (target_coords % cycle) / cycle
    cmap = plt.get_cmap(color_map)
    lut = cmap(np.linspace(0, 1, int(lut_size)))[:, :3].astype(np.float64)
    indices = (normalized_cycle * (int(lut_size) - 1)).astype(np.int32)
    colors = lut[indices]
    pcd.colors = o3d.utility.Vector3dVector(colors)
# endregion


def height_to_color(
    scalar_values: np.ndarray,
    color_map: str = "viridis",
    remove_outliers: bool = True,
    lower_percentile: float = 0.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """将一维标量数组映射为 RGB 颜色。"""
    if scalar_values.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    values = np.asarray(scalar_values, dtype=np.float64).reshape(-1)
    if remove_outliers:
        lower_bound = float(np.percentile(values, lower_percentile))
        upper_bound = float(np.percentile(values, upper_percentile))
        clipped = np.clip(values, lower_bound, upper_bound)
        normalized = (clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
    else:
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)

    if color_map == "viridis":
        from matplotlib.cm import viridis  # type: ignore

        colors = viridis(normalized)[:, :3]
    elif color_map == "cool":
        from matplotlib.cm import cool  # type: ignore

        colors = cool(normalized)[:, :3]
    else:
        from matplotlib.cm import plasma  # type: ignore

        colors = plasma(normalized)[:, :3]

    return colors


def colorize_by_height(pcd: o3d.geometry.PointCloud, color_map: str = "viridis") -> None:
    """按 Z 高度着色。"""
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        logger.warning("点云为空")
        return
    pcd.colors = o3d.utility.Vector3dVector(height_to_color(points[:, 2], color_map))


def _random_color() -> list[float]:
    return np.random.random(3).tolist()


def colorize_random(pcd: o3d.geometry.PointCloud) -> None:
    """随机统一着色。"""
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        logger.warning("点云为空")
        return
    pcd.paint_uniform_color(_random_color())
