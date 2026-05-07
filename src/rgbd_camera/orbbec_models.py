from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils.datas import Transform


# region 相机标定数据结构
@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参

    该结构用于把 SDK 原生相机内参转换为项目内部稳定数据契约它只描述单个图像流
    的针孔投影参数，不持有 SDK 对象、帧对象或畸变参数

    设计思想：
    - 使用不可变 dataclass，确保采集线程、计算线程和预览线程之间传递时不会被改写
    - 将 `fx/fy/cx/cy/width/height` 封装为一个整体，避免投影函数散收多个无归属浮点参数
    - 保留 `stream_name`，让调用侧明确当前参数来自 depth、color 或对齐后的投影流

    继承关系：
    - 不继承业务基类，不绑定 pyorbbecsdk 类型
    - 仅依赖 dataclass 生成初始化与只读字段约束
    """

    stream_name: str
    "图像流名称，例如 `depth`、`color` 或 `projection`"
    width: int
    "图像宽度，单位 像素"
    height: int
    "图像高度，单位 像素"
    fx: float
    "X 方向焦距，单位 像素"
    fy: float
    "Y 方向焦距，单位 像素"
    cx: float
    "主点 X 坐标，单位 像素"
    cy: float
    "主点 Y 坐标，单位 像素"

    def camera_matrix(self) -> np.ndarray:
        """返回 3x3 相机内参矩阵

        Returns
        -------
        matrix:
            内参矩阵，形状为 `(3, 3)`，dtype 为 `float64`矩阵形式为
            `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
        """
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """两个相机流之间的外参

    该结构封装 SDK 给出的流间刚体变换，用项目统一 `Transform` 表达 SE(3) 它只保存
    源流、目标流和变换，不持有 SDK 原生外参对象

    设计思想：
    - 使用 `Transform` 作为唯一位姿数据源，避免外参同时保存旋转矩阵和平移数组两份状态
    - `source_stream` 与 `target_stream` 明确变换方向，减少深度到彩色、彩色到深度混用风险
    - 按需通过属性返回矩阵、旋转和平移，所有派生结果都来自 `transform.as_SE3()`

    继承关系：
    - 不继承业务基类
    - 仅依赖 dataclass 与项目统一运动学数据结构
    """

    source_stream: str
    "源图像流名称，例如 `depth`"
    target_stream: str
    "目标图像流名称，例如 `color`"
    transform: Transform
    "从源流坐标系到目标流坐标系的 SE(3) 变换，平移单位 mm"

    @property
    def matrix(self) -> np.ndarray:
        """返回外参齐次矩阵

        Returns
        -------
        matrix:
            齐次矩阵，形状为 `(4, 4)`，dtype 为 `float64`，平移单位 mm
        """
        return np.asarray(self.transform.as_SE3(), dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        """返回外参旋转矩阵

        Returns
        -------
        rotation:
            旋转矩阵，形状为 `(3, 3)`，dtype 为 `float64`
        """
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """返回外参平移向量

        Returns
        -------
        translation:
            平移向量，形状为 `(3,)`，dtype 为 `float64`，单位 mm
        """
        return self.matrix[:3, 3]


# endregion


# region 会话与视锥配置数据结构
@dataclass(frozen=True)
class SensorFrustumConfig:
    """传感器理论视锥参数（单位：毫米）

    该结构用于应用层点云裁剪，不持有相机对象或动态状态调用方可按设备型号提供
    不同近远端几何参数，以保持实时过滤逻辑和硬件型号解耦
    """

    min_depth: float = 70.0
    "最小有效深度阈值，单位 mm"
    max_depth: float = 430.0
    "最大有效深度阈值，单位 mm"
    near_width: float = 117.0
    "近端平面的理论宽度，单位 mm"
    near_height: float = 89.0
    "近端平面的理论高度，单位 mm"
    far_width: float = 839.0
    "远端平面的理论宽度，单位 mm"
    far_height: float = 637.0
    "远端平面的理论高度，单位 mm"


@dataclass(frozen=True)
class SessionOptions:
    """Orbbec 会话运行参数

    该结构集中声明会话启动阶段的流配置、聚合模式与超时策略采用不可变 dataclass，
    避免运行中被多线程读写路径意外改写
    """

    timeout: int = 120
    "阻塞取帧超时时间，单位 ms"
    enable_frame_sync: bool = True
    "是否启用 SDK 帧同步`True` 启用，`False` 禁用"
    require_full_frame_when_color: bool = True
    "启用彩色流时是否要求 FULL_FRAME 聚合输出"
    preferred_capture_fps: int | None = None
    "期望采集帧率，单位 fps`None` 表示使用 SDK 默认 profile"
    enable_imu: bool = False
    "是否尝试启用设备 IMU`True` 启用，`False` 禁用"
    require_full_frame_when_imu: bool = True
    "启用 IMU 时是否要求 FULL_FRAME 聚合输出"
    auto_recover_on_disconnect: bool = True
    "运行中检测到掉线或取帧异常时，是否自动重建会话并继续采集"
    max_auto_recover_attempts: int = 6
    "单次异常后的最大自动恢复重试次数。超过后抛出异常交由上层决策"
    recover_retry_interval_s: float = 0.35
    "自动恢复重试间隔，单位 秒。用于热插拔等待设备重新枚举"
    recover_wait_timeout_s: float = 8.0
    "单次自动恢复允许的最长等待时间，单位 秒"
    recover_after_consecutive_timeouts: int = 10
    "连续取帧超时达到该阈值后触发自动恢复。0 表示仅在异常时恢复"


# endregion


# region IMU 数据结构
@dataclass(frozen=True)
class OrbbecImuSample:
    """Orbbec IMU 单次采样数据

    该结构是 SDK 帧读取与上层算法之间的数据契约字段允许 `None`，用于表达“设备不支持”
    或“本次帧集合未包含该类型数据”两类缺失场景
    """

    accel: tuple[float, float, float] | None = None
    "加速度三轴值 `(ax, ay, az)`，单位 m/s^2 设备不支持或缺帧时为 `None`"
    gyro: tuple[float, float, float] | None = None
    "角速度三轴值 `(gx, gy, gz)`，单位 rad/s设备不支持或缺帧时为 `None`"
    accel_temperature: float | None = None
    "加速度计温度，单位 摄氏度无数据时为 `None`"
    gyro_temperature: float | None = None
    "陀螺仪温度，单位 摄氏度无数据时为 `None`"
    accel_timestamp: int | None = None
    "加速度帧时间戳，单位 微秒无数据时为 `None`"
    gyro_timestamp: int | None = None
    "陀螺仪帧时间戳，单位 微秒无数据时为 `None`"

    @property
    def has_any_data(self) -> bool:
        """是否包含至少一种 IMU 数据"""
        return self.accel is not None or self.gyro is not None


# endregion


# region 相机参数补丁数据结构
@dataclass(frozen=True)
class IntrinsicPatch:
    """相机内参微调补丁

    补丁只描述缩放和偏移，不含应用流程控制该结构可被串联进标定调参脚本或运行时实验，
    但不会改变相机参数对象所有权
    """

    fx_scale: float = 1.0
    "X 焦距缩放系数`1.0` 表示不缩放"
    fy_scale: float = 1.0
    "Y 焦距缩放系数`1.0` 表示不缩放"
    cx_offset: float = 0.0
    "主点 X 偏移量，单位 像素"
    cy_offset: float = 0.0
    "主点 Y 偏移量，单位 像素"


@dataclass(frozen=True)
class DistortionPatch:
    """相机畸变参数微调补丁

    当前只覆盖 `k1/k2/p1/p2` 常用项，其他高阶畸变参数保持原值，避免在未知设备模型上
    引入过度校正风险
    """

    k1_offset: float = 0.0
    "径向畸变 `k1` 偏移量"
    k2_offset: float = 0.0
    "径向畸变 `k2` 偏移量"
    p1_offset: float = 0.0
    "切向畸变 `p1` 偏移量"
    p2_offset: float = 0.0
    "切向畸变 `p2` 偏移量"


@dataclass(frozen=True)
class CameraParamPatch:
    """相机参数补丁聚合对象

    该结构聚合深度/彩色内参与畸变补丁，以及 D2C 外参平移偏移用于一次性表达“相机参数
    微调意图”，降低调用链多参数透传的复杂度
    """

    depth: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    "深度相机内参补丁"
    color: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    "彩色相机内参补丁"
    depth_dist: DistortionPatch = field(default_factory=DistortionPatch)
    "深度相机畸变补丁"
    color_dist: DistortionPatch = field(default_factory=DistortionPatch)
    "彩色相机畸变补丁"
    d2c_translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    "深度到彩色外参平移偏移 `(tx, ty, tz)`，单位 mm"


# endregion
