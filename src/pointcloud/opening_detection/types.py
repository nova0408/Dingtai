from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# region 抓取流程数据契约


@dataclass(frozen=True)
class OpeningDetection:
    """开口检测结果。

    该结构定义托盘开口在 2D 图像中的几何表示，作为托盘检测模块与抓取位姿模块之间的
    固定数据契约。结构只承载结果数据，不持有检测器实例、线程句柄或图像资源。
    """

    center_uv: np.ndarray
    "开口中心像素坐标，形状 `(2,)`，dtype `float64`，顺序为 `[u, v]`，单位 像素。"
    bbox_xywh: tuple[int, int, int, int]
    "开口包围框 `(x, y, w, h)`，单位 像素。"
    quad_uv: np.ndarray
    "开口四边形顶点，形状 `(4, 2)`，dtype `float64`，坐标系为图像像素坐标。"
    score: float
    "开口检测置信分数，数值越大表示候选越可靠。"


@dataclass(frozen=True)
class PlaneResult:
    """平面拟合结果。

    用于表达开口附近平面模型 `n·x + d = 0`，供像素射线求交和姿态构造复用。
    """

    normal: np.ndarray
    "平面单位法向量，形状 `(3,)`，dtype `float64`，位于相机坐标系，单位 无量纲。"
    d: float
    "平面常数项，单位 mm。"


@dataclass(frozen=True)
class GraspResult:
    """抓取位姿结果。

    该结构封装抓取点、预抓取点与旋转矩阵。采用不可变 dataclass，便于跨线程安全传递。
    """

    grasp_point: np.ndarray
    "抓取点坐标，形状 `(3,)`，dtype `float64`，相机坐标系，单位 mm。"
    pre_grasp_point: np.ndarray
    "预抓取点坐标，形状 `(3,)`，dtype `float64`，相机坐标系，单位 mm。"
    rotation: np.ndarray
    "抓取旋转矩阵，形状 `(3, 3)`，dtype `float64`，列向量依次表示 X/Y/Z 轴方向。"


@dataclass(frozen=True)
class TrayMaskResult:
    """托盘掩码结果集合。

    该结构输出托盘流程产生的多层掩码，供后续可视化、顶面法线估计与抓取精修阶段复用。
    """

    tray_mask: np.ndarray
    "托盘主掩码，形状 `(H, W)`，dtype `uint8`，非零表示托盘像素。"
    tray_detect_ok: bool
    "托盘检测是否来自高置信检测器结果。`False` 时通常表示回退分割。"
    near_plane_mask: np.ndarray | None
    "开口邻近平面掩码，形状 `(H, W)`，dtype `uint8`；无结果时为 `None`。"
    no_hole_mask: np.ndarray | None
    "无孔顶面掩码，形状 `(H, W)`，dtype `uint8`；无结果时为 `None`。"
    top_quad_uv: np.ndarray | None
    "由 `no_hole_mask` 拟合得到的旋转四边形，形状 `(4, 2)`，dtype `float64`。"


# endregion

