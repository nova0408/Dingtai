from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils.datas import Transform


# region 相机标定数据结构
@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参。

    该结构用于把 SDK 原生相机内参转换为项目内部稳定数据契约。它只描述单个图像流
    的针孔投影参数，不持有 SDK 对象、帧对象或畸变参数。

    设计思想：
    - 使用不可变 dataclass，确保采集线程、计算线程和预览线程之间传递时不会被改写。
    - 将 `fx/fy/cx/cy/width/height` 封装为一个整体，避免投影函数散收多个无归属浮点参数。
    - 保留 `stream_name`，让调用侧明确当前参数来自 depth、color 或对齐后的投影流。

    继承关系：
    - 不继承业务基类，不绑定 pyorbbecsdk 类型。
    - 仅依赖 dataclass 生成初始化与只读字段约束。
    """

    stream_name: str
    "图像流名称，例如 `depth`、`color` 或 `projection`。"
    width: int
    "图像宽度，单位 像素。"
    height: int
    "图像高度，单位 像素。"
    fx: float
    "X 方向焦距，单位 像素。"
    fy: float
    "Y 方向焦距，单位 像素。"
    cx: float
    "主点 X 坐标，单位 像素。"
    cy: float
    "主点 Y 坐标，单位 像素。"

    def camera_matrix(self) -> np.ndarray:
        """返回 3x3 相机内参矩阵。

        Returns
        -------
        matrix:
            内参矩阵，形状为 `(3, 3)`，dtype 为 `float64`。矩阵形式为
            `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`。
        """
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """两个相机流之间的外参。

    该结构封装 SDK 给出的流间刚体变换，用项目统一 `Transform` 表达 SE(3)。它只保存
    源流、目标流和变换，不持有 SDK 原生外参对象。

    设计思想：
    - 使用 `Transform` 作为唯一位姿数据源，避免外参同时保存旋转矩阵和平移数组两份状态。
    - `source_stream` 与 `target_stream` 明确变换方向，减少深度到彩色、彩色到深度混用风险。
    - 按需通过属性返回矩阵、旋转和平移，所有派生结果都来自 `transform.as_SE3()`。

    继承关系：
    - 不继承业务基类。
    - 仅依赖 dataclass 与项目统一运动学数据结构。
    """

    source_stream: str
    "源图像流名称，例如 `depth`。"
    target_stream: str
    "目标图像流名称，例如 `color`。"
    transform: Transform
    "从源流坐标系到目标流坐标系的 SE(3) 变换，平移单位 mm。"

    @property
    def matrix(self) -> np.ndarray:
        """返回外参齐次矩阵。

        Returns
        -------
        matrix:
            齐次矩阵，形状为 `(4, 4)`，dtype 为 `float64`，平移单位 mm。
        """
        return np.asarray(self.transform.as_SE3(), dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        """返回外参旋转矩阵。

        Returns
        -------
        rotation:
            旋转矩阵，形状为 `(3, 3)`，dtype 为 `float64`。
        """
        return self.matrix[:3, :3]

    @property
    def translation_mm(self) -> np.ndarray:
        """返回外参平移向量。

        Returns
        -------
        translation:
            平移向量，形状为 `(3,)`，dtype 为 `float64`，单位 mm。
        """
        return self.matrix[:3, 3]


# endregion


@dataclass(frozen=True)
class SensorFrustumConfig:
    """传感器理论视锥参数（单位：毫米）。"""

    min_depth_mm: float = 70.0
    max_depth_mm: float = 430.0
    near_width_mm: float = 117.0
    near_height_mm: float = 89.0
    far_width_mm: float = 839.0
    far_height_mm: float = 637.0


@dataclass(frozen=True)
class SessionOptions:
    """Orbbec 会话运行参数。"""

    timeout_ms: int = 120
    enable_frame_sync: bool = True
    require_full_frame_when_color: bool = True
    preferred_capture_fps: int | None = None
    enable_imu: bool = False
    require_full_frame_when_imu: bool = True


@dataclass(frozen=True)
class OrbbecImuSample:
    """Orbbec IMU 单次采样数据。"""

    accel_mps2: tuple[float, float, float] | None = None
    gyro_rad_s: tuple[float, float, float] | None = None
    accel_temperature_c: float | None = None
    gyro_temperature_c: float | None = None
    accel_timestamp_us: int | None = None
    gyro_timestamp_us: int | None = None

    @property
    def has_any_data(self) -> bool:
        """是否包含至少一种 IMU 数据。"""
        return self.accel_mps2 is not None or self.gyro_rad_s is not None


@dataclass(frozen=True)
class IntrinsicPatch:
    """相机内参微调补丁。"""

    fx_scale: float = 1.0
    fy_scale: float = 1.0
    cx_offset: float = 0.0
    cy_offset: float = 0.0


@dataclass(frozen=True)
class DistortionPatch:
    """相机畸变参数微调补丁。"""

    k1_offset: float = 0.0
    k2_offset: float = 0.0
    p1_offset: float = 0.0
    p2_offset: float = 0.0


@dataclass(frozen=True)
class CameraParamPatch:
    """相机参数补丁聚合对象。"""

    depth: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    color: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    depth_dist: DistortionPatch = field(default_factory=DistortionPatch)
    color_dist: DistortionPatch = field(default_factory=DistortionPatch)
    d2c_translation_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
