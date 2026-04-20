# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import open3d as o3d
from loguru import logger


def load_pcd(filepath: str | Path, ratio: float = 1.0, remove_outlier: bool = False) -> o3d.geometry.PointCloud:
    """读取点云文件，并可选执行随机下采样。

    参数说明：
    - ``filepath``：点云文件路径（通常为 ``.ply``）；
    - ``ratio``：随机下采样比例，``1.0`` 表示不下采样；
    - ``remove_outlier``：历史接口保留参数，当前实现不执行离群点去除。

    注意：
    - 本函数当前行为与旧实现保持一致：仅读取 + 随机下采样；
    - 若后续需要启用离群点去除，应在明确评估后单独扩展，不在此处静默改变行为。
    """
    # 占位保留：避免删除参数导致上层调用签名变化。
    _ = remove_outlier

    path = Path(filepath)
    pointcloud = o3d.io.read_point_cloud(str(path))

    if len(pointcloud.points) == 0:
        logger.warning("{} 点云为空", path)

    if ratio >= 1.0:
        return pointcloud

    return pointcloud.random_down_sample(sampling_ratio=ratio)
