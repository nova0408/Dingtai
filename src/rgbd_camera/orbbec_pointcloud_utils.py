from __future__ import annotations

import numpy as np
from pyorbbecsdk import OBFormat, PointCloudFilter


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    将任意点云数组标准化为 `Nx3` 或 `Nx6` 的 `float32` 数组。

    Parameters
    ----------
    points : np.ndarray
        原始点云数组。

    Returns
    -------
    np.ndarray
        标准化后的点云数组。
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 2 and points.shape[1] in (3, 6):
        return points

    flat = points.reshape(-1)
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if flat.size % 6 == 0:
        return flat.reshape(-1, 6)
    if flat.size % 3 == 0:
        return flat.reshape(-1, 3)
    raise RuntimeError(f"Unsupported point array shape: {points.shape}, flat_size={flat.size}")


def filter_valid_points(points: np.ndarray, max_depth_mm: float | None) -> tuple[np.ndarray, float]:
    """
    过滤无效点（非有限值、非正深度、超深度上限）。

    Parameters
    ----------
    points : np.ndarray
        输入点云（`Nx3`/`Nx6`）。
    max_depth_mm : float | None
        最大深度阈值，`None` 表示不限制。

    Returns
    -------
    tuple[np.ndarray, float]
        过滤后点云与有效点占比。
    """
    if points.size == 0:
        return points.reshape(0, 3), 0.0

    xyz = points[:, :3]
    valid = np.isfinite(xyz).all(axis=1)
    valid &= xyz[:, 2] > 0.0
    if max_depth_mm is not None:
        valid &= xyz[:, 2] <= max_depth_mm

    valid_ratio = float(np.count_nonzero(valid)) / float(len(points))
    return points[valid], valid_ratio


def filter_points_in_sensor_frustum(
    points: np.ndarray,
    min_depth_mm: float,
    max_depth_mm: float,
    near_width_mm: float,
    near_height_mm: float,
    far_width_mm: float,
    far_height_mm: float,
) -> np.ndarray:
    """
    按线性视锥模型切割点云。

    Parameters
    ----------
    points : np.ndarray
        输入点云（`Nx3`/`Nx6`）。
    min_depth_mm : float
        近端深度。
    max_depth_mm : float
        远端深度。
    near_width_mm, near_height_mm, far_width_mm, far_height_mm : float
        近端与远端的视场宽高（毫米）。

    Returns
    -------
    np.ndarray
        切割后的点云。
    """
    if points.size == 0:
        return points[0:0]
    if max_depth_mm <= min_depth_mm:
        raise ValueError("max_depth_mm must be > min_depth_mm")

    xyz = points[:, :3]
    z = xyz[:, 2]
    valid = (z >= float(min_depth_mm)) & (z <= float(max_depth_mm))
    if not np.any(valid):
        return points[valid]

    t = np.clip((z - float(min_depth_mm)) / float(max_depth_mm - min_depth_mm), 0.0, 1.0)
    width = float(near_width_mm) + t * float(far_width_mm - near_width_mm)
    height = float(near_height_mm) + t * float(far_height_mm - near_height_mm)

    valid &= np.abs(xyz[:, 0]) <= (0.5 * width)
    valid &= np.abs(xyz[:, 1]) <= (0.5 * height)
    return points[valid]


def set_point_cloud_filter_format(point_filter: PointCloudFilter, depth_scale: float, use_color: bool) -> None:
    """
    设置 SDK 点云过滤器的输出格式。

    Parameters
    ----------
    point_filter : PointCloudFilter
        SDK 点云过滤器。
    depth_scale : float
        深度尺度。
    use_color : bool
        是否输出彩色点云格式。
    """
    point_filter.set_position_data_scaled(depth_scale)
    point_filter.set_create_point_format(OBFormat.RGB_POINT if use_color else OBFormat.POINT)


def voxel_downsample_points_numpy(points: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    """
    纯 NumPy 实现的体素下采样。

    Parameters
    ----------
    points : np.ndarray
        输入点云（`Nx3`/`Nx6`）。
    voxel_size_mm : float
        体素边长（毫米）。

    Returns
    -------
    np.ndarray
        下采样后的点云。
    """
    if points.size == 0:
        return points
    if voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be > 0")

    xyz = points[:, :3]
    voxel = np.floor(xyz / float(voxel_size_mm)).astype(np.int64)
    _, unique_idx = np.unique(voxel, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx]
