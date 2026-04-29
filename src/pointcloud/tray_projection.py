from __future__ import annotations

import numpy as np

from src.rgbd_camera import CameraIntrinsics


# region 点云投影与图像索引
def project_points_to_image(
    xyz: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """将相机坐标系点云投影到图像平面。

    Parameters
    ----------
    xyz:
        点云坐标数组，形状为 `(N, 3)`，单位 mm，第 0/1/2 列分别为 X/Y/Z。
    intrinsics:
        相机针孔内参对象，包含图像宽高、焦距和主点坐标，单位均为 像素。

    Returns
    -------
    uv:
        像素坐标数组，形状为 `(N, 2)`，dtype 为 `int32`。无效投影点填 `-1`。
    valid_proj:
        有效投影掩码，形状为 `(N,)`，dtype 为 `bool`。True 表示该点可用于图像索引。

    Notes
    -----
    该函数只做针孔模型投影，不做畸变校正，也不改变输入点云顺序。
    `intrinsics.width/height` 决定输出有效投影范围。
    """
    # xyz: (N, 3)，单位 mm；z 是每个点的相机前向深度。
    z = xyz[:, 2]
    # valid: (N,) bool；z<=0 的点无法用针孔模型投影。
    valid = z > 1e-6
    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)
    if np.any(valid):
        # x/y/zz 形状均为 (M,)，M 是有效深度点数量。
        x = xyz[valid, 0]
        y = xyz[valid, 1]
        zz = z[valid]
        uu = np.rint(intrinsics.fx * x / zz + intrinsics.cx).astype(np.int32)
        vv = np.rint(intrinsics.fy * y / zz + intrinsics.cy).astype(np.int32)
        # in_bounds: (M,) bool；过滤投影到图像外的点。
        in_bounds = (uu >= 0) & (uu < intrinsics.width) & (vv >= 0) & (vv < intrinsics.height)
        # np.where(valid)[0] 把有效深度点映射回原始 N 点索引。
        idx = np.where(valid)[0][in_bounds]
        u[idx] = uu[in_bounds]
        v[idx] = vv[in_bounds]
    return np.stack([u, v], axis=1), (u >= 0) & (v >= 0)


def collect_indices_in_mask(uv: np.ndarray, valid_proj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """收集落入 2D mask 的原始点云索引。

    Parameters
    ----------
    uv:
        点云投影像素坐标，形状为 `(N, 2)`，dtype 为 `int32`。
    valid_proj:
        有效投影掩码，形状为 `(N,)`，dtype 为 `bool`。
    mask:
        2D 掩码图，形状为 `(H, W)`，非零像素表示目标区域。

    Returns
    -------
    indices:
        原始点云索引数组，形状为 `(M,)`，dtype 为 `int32`。
    """
    # idx: (M,) 原始点云索引，只包含已经投影进图像的点。
    idx = np.where(valid_proj)[0]
    if idx.size == 0:
        return np.empty((0,), dtype=np.int32)
    u = uv[idx, 0]
    v = uv[idx, 1]
    # NumPy 高级索引 mask[v, u] 会一次性取出 M 个像素值，inside 形状仍为 (M,)。
    inside = mask[v, u] > 0
    return idx[inside].astype(np.int32)


# endregion
