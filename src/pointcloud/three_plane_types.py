from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.utils.datas import Axis, Point, Transform, Vector


# region 数据结构
@dataclass(frozen=True)
class PlanePoseConfig:
    """三平面位姿估计配置。

    该配置对象描述从裁切点云中提取三个结构平面并构造测试坐标系所需的全部参数。
    它只保存算法阈值和坐标轴参考方向，不持有点云、Open3D 对象或相机资源。

    设计思想：
    - 使用不可变 dataclass 作为稳定算法契约，避免实时线程中被临时修改。
    - 参数以毫米、点数和 `Vector` 单位向量为主，直接对应点云算法的真实输入语义。
    - 底面轴、X/Z 方向 hint 均复用 `src.utils.datas.Vector`，避免在 pointcloud 内重复定义坐标轴。

    继承关系：
    - 不继承业务基类，不使用动态分发。
    - 仅依赖 dataclass 生成初始化和只读字段约束。
    """

    plane_distance_mm: float = 4.0
    "点到平面的最大归属距离，单位 mm。"
    plane_min_points: int = 1200
    "单个有效平面需要的最少内点数量，单位 点。"
    ransac_iterations: int = 1200
    "Open3D RANSAC 平面分割迭代次数，单位 次。"
    max_ransac_points: int = 60_000
    "进入 RANSAC 的最大采样点数，单位 点；超过时做固定步长降采样。"
    refine_plane_models: bool = True
    "是否在 RANSAC 后用 PCA 对每个平面模型做二次精修。"
    plane_refine_max_points: int = 80_000
    "PCA 精修单平面最多使用点数，单位 点。"
    bottom_axis: Vector = Vector.YAxis()
    "用于识别底面的相机坐标轴向量，在相机坐标系下表达，默认使用世界 Y 轴方向。"
    frame_z_hint: Vector = Vector(0.0594, -0.9020, -0.4276)
    "测试坐标系 Z 轴参考方向，在相机坐标系下表达，通常会被归一化后参与点积定向。"
    frame_x_hint: Vector = Vector(0.9178, -0.1191, 0.3788)
    "测试坐标系 X 轴参考方向，在相机坐标系下表达，通常会投影到底面切平面内使用。"
    use_fixed_x_hint_axis: bool = True
    "是否优先使用投影到切平面的 X hint 作为 X 轴，减少侧面法线抖动带来的旋转漂移。"


@dataclass(frozen=True)
class PlanePatch:
    """单个平面检测结果。

    该结构保存一个已经排序命名的平面模型，是 RANSAC/PCA 平面拟合与坐标系构造之间的
    数据契约。它不保存该平面的完整点集，也不保存预览颜色，避免把测试脚本的可视化策略
    固化到算法结果中。

    设计思想：
    - 用不可变 dataclass 固定平面语义，防止后续坐标系计算误改模型。
    - 平面模型采用标准 `ax + by + cz + d = 0` 形式，法线已归一化，距离单位保持 mm。
    - `label` 使用明确业务名：`bottom`、`left_side`、`right_side`。
    - 预览颜色由测试脚本按平面序号或标签自行定义，算法层只输出几何语义。

    继承关系：
    - 不继承业务基类。
    - 仅依赖 dataclass 的初始化和只读字段。
    """

    label: str
    "平面语义标签，例如 `bottom`、`left_side`、`right_side`。"
    model: np.ndarray
    "平面模型数组，形状为 `(4,)`，dtype 为 `float64`，表示 `ax + by + cz + d = 0`。"
    inlier_count: int
    "归属于该平面的点数量，单位 点。"


@dataclass(frozen=True)
class CoordinateFramePose:
    """由三个平面计算出的测试坐标系位姿。

    该结构是三平面算法的核心输出，描述测试坐标系在相机坐标系下的位置和方向。
    原点暂取三个平面的交点，坐标轴完全由 `Axis` 表达，避免在结果对象中保存多份可能
    不一致的位姿状态。

    设计思想：
    - 使用不可变 dataclass 便于跨线程传递到预览层和日志层。
    - 只持有 `Axis` 作为位姿源数据，`Transform`、原点数组和旋转矩阵都按需从 Axis 派生。
    - 派生旋转矩阵统一走 `Axis.to_transform().as_SE3()`，保持物理语义与项目 SE(3) 协议一致。

    继承关系：
    - 不继承业务基类，不绑定 Open3D 几何对象。
    - 持有 `Axis`、RPY 数组和残差标量；不重复持有 Transform 或旋转矩阵。
    """

    axis: Axis
    "项目统一右手坐标系对象，原点单位 mm，三条轴为相机坐标系下的单位方向向量。"
    rpy_deg: np.ndarray
    "欧拉角数组，形状为 `(3,)`，顺序为 roll/pitch/yaw，单位 deg。"
    residual: float
    "三平面交点线性方程残差，单位 mm。"

    @property
    def transform(self) -> Transform:
        """返回坐标系对应的 SE(3) 变换。

        Returns
        -------
        transform:
            项目统一 `Transform` 对象，平移单位 mm，旋转来自 `Axis` 的右手坐标系。

        Notes
        -----
        该属性不缓存结果，避免 `Axis` 与 `Transform` 成为两份独立状态。
        """
        return self.axis.to_transform()

    @property
    def origin_mm(self) -> np.ndarray:
        """返回坐标系原点数组。

        Returns
        -------
        origin:
            原点坐标数组，形状为 `(3,)`，dtype 为 `float64`，单位 mm，处于相机坐标系。
        """
        return self.axis.origin.as_array()

    @property
    def rotation(self) -> np.ndarray:
        """返回坐标系旋转矩阵。

        Returns
        -------
        rotation:
            旋转矩阵，形状为 `(3, 3)`，dtype 为 `float64`，列向量依次对应 X/Y/Z 轴。

        Notes
        -----
        旋转矩阵从 `Axis.to_transform().as_SE3()` 提取，而不是手工拼接三条轴向量。
        """
        return np.asarray(self.axis.to_transform().as_SE3()[:3, :3], dtype=np.float64)


@dataclass(frozen=True)
class ThreePlanePoseResult:
    """三平面位姿估计总结果。

    该结构封装一帧裁切点云的全部算法输出，包括逐点标签、排序后的平面列表和可选坐标系。
    它是测试脚本和后续业务模块读取三平面算法结果的单一入口。

    设计思想：
    - labels 与输入点云逐点对齐，避免额外复制分组点云。
    - planes 只保存平面模型和统计信息，坐标系计算失败时仍可用于可视化诊断。
    - pose 允许为 None，用于表达平面不足、退化或排序失败。

    继承关系：
    - 不继承业务基类。
    - 仅依赖 dataclass 生成不可变结果结构。
    """

    labels: np.ndarray
    "逐点标签数组，形状为 `(N,)`，dtype 为 `int32`；-2 料盘排除，-1 未分配，0/1/2 为三平面。"
    planes: list[PlanePatch]
    "排序后的平面列表，顺序通常为 bottom、left_side、right_side。"
    pose: CoordinateFramePose | None
    "计算出的测试坐标系位姿；平面不足或退化时为 None。"


# endregion


# region 位姿稳定
class PoseWindowStabilizer:
    """最近实际计算帧的位姿稳定器。

    该类用于实时预览中的轻量位姿平滑，只缓存最近 `max_frames` 个已经完成计算的位姿。
    它不接收相机帧、不启动线程、不参与平面检测，只对算法输出的坐标系做轴向符号对齐和均值稳定。

    设计思想：
    - 使用固定长度 deque，确保最多使用 15 个实际计算帧，避免跟随性明显变差。
    - 对 X/Z 轴先按最新帧对齐符号，再做均值，避免同一轴因法线符号翻转互相抵消。
    - 平滑后重新正交化 X/Y/Z，保证输出旋转矩阵仍可用于 Open3D 坐标系变换。

    继承关系：
    - 不继承业务基类。
    - 只持有 `CoordinateFramePose` 轻量结果，不持有点云、模型或硬件资源。

    线程语义：
    - 该类内部没有锁，建议只在预览线程或单个结果消费线程中使用。
    """

    def __init__(self, max_frames: int = 5) -> None:
        """初始化位姿稳定窗口。

        Parameters
        ----------
        max_frames:
            最多保留的实际计算帧数量，单位 帧。该值会被限制在 `[1, 15]`。
        """
        self.max_frames = int(np.clip(max_frames, 1, 15))
        self._poses: deque[CoordinateFramePose] = deque(maxlen=self.max_frames)

    def update(self, pose: CoordinateFramePose) -> CoordinateFramePose:
        """加入新位姿并返回平滑后的位姿。

        Parameters
        ----------
        pose:
            最新完成计算的测试坐标系位姿。位姿源数据为 `pose.axis`。

        Returns
        -------
        stable_pose:
            平滑后的测试坐标系位姿。`Axis` 和 `rpy_deg` 均重新计算；`residual` 保留最新帧残差。

        Notes
        -----
        该方法只使用实际完成计算的帧，不使用未完成或被丢弃的相机帧。
        """
        from .three_plane_pose import (
            make_coordinate_frame_pose,
            rotation_matrix_to_rpy_deg,
        )

        self._poses.append(pose)
        if len(self._poses) == 1:
            return pose

        latest = self._poses[-1]
        ref_x = latest.axis.x_axis
        ref_z = latest.axis.z_axis
        origins: list[Point] = []
        x_axes: list[Vector] = []
        z_axes: list[Vector] = []
        for item in self._poses:
            # Axis 已保证 X/Y/Z 是右手系单位轴；这里仅按最新帧做符号对齐后求均值。
            x_axis = item.axis.x_axis
            z_axis = item.axis.z_axis
            if x_axis.dot(ref_x) < 0.0:
                x_axis = x_axis.negated()
            if z_axis.dot(ref_z) < 0.0:
                z_axis = z_axis.negated()
            origins.append(item.axis.origin)
            x_axes.append(x_axis)
            z_axes.append(z_axis)

        origin = _mean_point(origins)
        z_axis = _mean_vector(z_axes).normalized()
        x_axis = _mean_vector(x_axes).normalized()
        x_axis = (x_axis - z_axis * x_axis.dot(z_axis)).normalized()
        if x_axis.length < 1e-8:
            x_axis = ref_x
        axis = Axis(origin=origin, x_axis=x_axis, z_axis=z_axis)
        rotation = np.asarray(axis.to_transform().as_SE3()[:3, :3], dtype=np.float64)
        return make_coordinate_frame_pose(
            origin=axis.origin.as_array(),
            rotation=np.asarray(rotation, dtype=np.float64),
            rpy_deg=rotation_matrix_to_rpy_deg(rotation),
            residual_mm=latest.residual,
        )


def _mean_vector(vectors: list[Vector]) -> Vector:
    """计算多个三维向量的逐分量均值。

    Parameters
    ----------
    vectors:
        三维向量列表，长度为 `M`，每个元素是项目统一 `Vector` 对象，单位为无量纲方向向量。

    Returns
    -------
    mean_vector:
        逐分量均值向量。该函数不做归一化，调用方可根据语义继续调用 `Vector.normalized()`。

    Notes
    -----
    该函数只承担 `Vector` 到 NumPy 批量均值的桥接职责。`data` 的形状为 `(M, 3)`，
    dtype 为 `float64`；返回值重新包装为 `Vector`，避免轴向计算继续扩散裸数组。
    """
    if len(vectors) == 0:
        return Vector.zero()
    # data: (M, 3) float64，每一行是一个已完成计算帧的单位轴向量。
    data = np.stack([axis.as_array() for axis in vectors], axis=0)
    return Vector.from_array(np.mean(data, axis=0))


def _mean_point(points: list[Point]) -> Point:
    """计算多个三维点的逐分量均值。

    Parameters
    ----------
    points:
        三维点列表，长度为 `M`，每个元素是项目统一 `Point` 对象，单位 mm。

    Returns
    -------
    mean_point:
        逐分量均值点，单位 mm。列表为空时返回坐标原点。

    Notes
    -----
    该函数只用于位姿稳定窗口的原点平滑。`data` 的形状为 `(M, 3)`，dtype 为
    `float64`，每一行对应一个实际完成计算帧的坐标系原点。
    """
    if len(points) == 0:
        return Point.Origin()
    # data: (M, 3) float64，每一行是一个已完成计算帧的原点坐标，单位 mm。
    data = np.stack([point.as_array() for point in points], axis=0)
    return Point.from_array(np.mean(data, axis=0))


# endregion
