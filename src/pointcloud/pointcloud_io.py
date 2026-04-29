# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import open3d as o3d
from loguru import logger


def load_pcd(filepath: str | Path, down_sample_ratio: float = 1.0) -> o3d.geometry.PointCloud:
    """读取点云文件，并可选执行随机下采样。

    参数说明：
    - ``filepath``：点云文件路径（通常为 ``.ply``）；
    - ``ratio``：随机下采样比例，``1.0`` 表示不下采样；

    """

    path = Path(filepath)
    pointcloud = o3d.io.read_point_cloud(path)

    if len(pointcloud.points) == 0:
        logger.warning("{} 点云为空", path)

    if down_sample_ratio >= 1.0:
        return pointcloud

    return pointcloud.random_down_sample(sampling_ratio=down_sample_ratio)
