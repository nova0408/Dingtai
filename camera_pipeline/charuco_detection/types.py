from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# region 配置
@dataclass(frozen=True, slots=True)
class CharucoDetectionConfig:
    """ChArUco 单帧检测配置。

    该不可变配置只承载图像增强、有效角点门槛和坐标轴显示参数，不持有
    `CharucoBoard`、图像或相机资源。检测器构造后可跨连续帧只读复用，
    但 OpenCV 检测器本身不保证并发调用安全。

    本类不继承业务基类，避免为纯算法参数引入生命周期和动态分发。
    """

    min_charuco_corners: int = 6
    "允许进入 PnP 位姿求解的最小唯一 ChArUco 角点数，单位 点，推荐不小于 6。"
    clahe_clip_limit: float = 2.0
    "CLAHE 对比度裁剪阈值，推荐范围 1.0 至 4.0。"
    clahe_grid_size: tuple[int, int] = (8, 8)
    "CLAHE 网格数量，顺序为 `(x, y)`，单位 格。"
    unsharp_sigma: float = 1.0
    "反锐化高斯模糊标准差，单位 pixel，推荐范围 0.6 至 1.5。"
    unsharp_amount: float = 0.5
    "反锐化增强权重，无量纲，推荐范围 0.2 至 0.8。"
    axis_length_scale: float = 1.5
    "debug pose 坐标轴长度相对棋盘方格边长的倍率。"


# endregion


# region 输出协议
@dataclass(frozen=True, slots=True)
class CharucoDebugArtifacts:
    """ChArUco 检测调试产物。

    该结构只在调用方显式启用 debug 时构造，保存最终融合角点及叠加图，
    不负责显示、传输或文件 IO。数组由检测器创建，调用方应按只读数据使用。

    本类不继承业务基类，不持有硬件、线程、文件句柄或 OpenCV 检测器。
    """

    overlay_bgr: np.ndarray
    "marker 边框、ChArUco 角点和 pose 坐标轴叠加图，形状 `(H, W, 3)`，dtype `uint8`。"
    marker_corners_px: tuple[np.ndarray, ...] = field(default_factory=tuple)
    "融合后的 marker 四角点，每项形状 `(4, 1, 2)`，dtype `float32`，图像像素坐标。"
    marker_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0, 1), dtype=np.int32)
    )
    "融合后的 marker ID，形状 `(M, 1)`，dtype `int32`；未检测到时为空数组。"
    charuco_corners_px: np.ndarray = field(
        default_factory=lambda: np.empty((0, 1, 2), dtype=np.float32)
    )
    "融合后的 ChArUco 角点，形状 `(N, 1, 2)`，dtype `float32`，图像像素坐标。"
    charuco_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0, 1), dtype=np.int32)
    )
    "融合后的 ChArUco ID，形状 `(N, 1)`，dtype `int32`；未检测到时为空数组。"


@dataclass(frozen=True, slots=True)
class CharucoDetectionResult:
    """ChArUco 标定板检测和位姿结果。

    该不可变结果只保存调用方需要的核心状态、位姿、误差、计数和可选调试产物。
    检测失败使用明确的 `status="missing"` 与空矩阵表达，不使用可空字段。

    本类不继承业务基类，不持有算法器、硬件或 IO 资源，可跨线程只读传递。
    """

    status: str
    "检测状态：`detected` 表示位姿有效，`missing` 表示当前帧未获得有效位姿。"
    t_cam_board_mm: np.ndarray
    "从标定板坐标系到相机坐标系的齐次矩阵，成功时形状 `(4, 4)`、dtype `float64`，平移单位 mm；失败时形状 `(0, 0)`。"
    error_px: float
    "有效位姿的平均重投影误差，单位 pixel；未获得位姿时为正无穷。"
    marker_num: int
    "融合后唯一 ArUco marker ID 数量，单位 个。"
    charuco_num: int
    "融合后唯一 ChArUco 角点 ID 数量，单位 点。"
    debug_artifacts: tuple[CharucoDebugArtifacts, ...] = field(default_factory=tuple)
    "可选调试产物；debug 关闭时为空元组，开启时包含一个元素。"


# endregion
