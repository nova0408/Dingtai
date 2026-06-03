from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.servers.common import JointLimit, MotionLimit, coerce_float_tuple, validate_joint_values, validate_motion_values

ArmSide = Literal["left", "right"]


# region 数据结构


@dataclass(frozen=True, slots=True)
class ArmPhysicalParameters:
    """机械臂关键物理参数。

    职责边界：
    - 只描述接口层已知的机械臂物理信息。
    - 不表达连杆 DH 参数、惯量矩阵或真实电机参数，避免在协议未提供时猜测。

    设计思想：
    - 左右臂共享同一机械臂版本，通过整合层的镜像安装语义区分。
    - `joint_count` 用于调用方确认关节序列长度。

    生命周期：
    - 可跨线程读取，不持有硬件连接。

    继承关系：
    - 不继承业务基类，保持数据契约单一。
    """

    model_name: str
    "机械臂型号名称。"

    joint_count: int
    "机械臂关节数量，单位 个。"


@dataclass(frozen=True, slots=True)
class ArmConfig:
    """单只机械臂控制配置。

    职责边界：
    - 保存单只机械臂的可动范围、速度限制、加速度限制和物理参数。
    - 不包含左臂或右臂命名，左右实例由整合层添加侧别。

    设计思想：
    - 当前左右臂是镜像安装，同一 `ArmConfig` 可被左右两只手臂复用。
    - 配置对象集中承载默认值，避免服务类中散落参数。

    生命周期：
    - 可跨线程读取，不持有硬件连接。

    继承关系：
    - 不继承业务基类，保持配置职责。
    """

    movable_range: tuple[JointLimit, ...]
    "单只机械臂可动范围表，顺序与关节值序列一致。"

    speed_limit: tuple[MotionLimit, ...]
    "单只机械臂速度限制表，顺序与关节值序列一致。"

    acceleration_limit: tuple[MotionLimit, ...]
    "单只机械臂加减速度限制表，顺序与关节值序列一致。"

    physical_parameters: ArmPhysicalParameters
    "单只机械臂关键物理参数。"


@dataclass(slots=True)
class ArmState:
    """单只机械臂运行状态。

    职责边界：
    - 保存当前关节值、目标关节值、速度和加速度设置。
    - 不代表真实硬件反馈，也不负责运动插补。

    设计思想：
    - 状态与配置分离，配置负责约束，状态负责当前命令值。
    - 使用 tuple 保持关节序列不会被调用方原地修改。

    生命周期：
    - 当前状态仅在进程内存中维护。
    - 后续接入硬件时，应由硬件反馈刷新 `current_joint_values`。

    继承关系：
    - 不继承业务基类，保持服务内部状态职责。
    """

    current_joint_values: tuple[float, ...]
    "当前关节值序列，单位与 `ArmConfig.movable_range` 对齐。"

    target_joint_values: tuple[float, ...]
    "目标关节值序列，单位与 `ArmConfig.movable_range` 对齐。"

    speed: tuple[float, ...]
    "当前速度设置序列，单位与 `ArmConfig.speed_limit` 对齐。"

    acceleration: tuple[float, ...]
    "当前加速度设置序列，单位与 `ArmConfig.acceleration_limit` 对齐。"


# endregion


# region 配置


def default_casia_arm_config() -> ArmConfig:
    """创建 CASIA 机械臂默认配置。

    Returns
    -------
    ArmConfig
        单只 CASIA 机械臂默认配置。

    Notes
    -----
    当前文档未给出真实关节极限和动力学参数，因此这里保留最小可调用默认值。
    左右臂不在这里重复展开，整合层通过两个 `ArmServer` 实例表达镜像安装。
    """

    movable_range = tuple(JointLimit(f"j{idx}", -180.0, 180.0, "deg") for idx in range(1, 7))
    speed_limit = tuple(MotionLimit(item.name, 90.0, "deg/s") for item in movable_range)
    acceleration_limit = tuple(MotionLimit(item.name, 180.0, "deg/s^2") for item in movable_range)
    return ArmConfig(
        movable_range=movable_range,
        speed_limit=speed_limit,
        acceleration_limit=acceleration_limit,
        physical_parameters=ArmPhysicalParameters(model_name="casia_arm", joint_count=len(movable_range)),
    )


# endregion


# region 主入口


class ArmServer:
    """单只机械臂最小控制服务。

    职责边界：
    - 提供单只机械臂的范围、速度、加速度、物理参数与关节命令接口。
    - 不负责左右臂组合、镜像安装求解、gRPC 绑定、硬件通信、轨迹规划或碰撞检测。

    设计思想：
    - 左右臂共享同一配置，避免为镜像关系复制两套关节定义。
    - `side` 和 `mirror_sign` 只记录整机安装语义，不改变单臂关节限制。

    生命周期：
    - 实例可在单线程服务流程中复用。
    - 当前类不持有硬件连接、线程、协程或文件句柄。
    - 多线程场景需要在外层增加锁或消息队列。

    继承关系：
    - 不继承业务基类，便于后续按 gRPC 或硬件协议适配。
    """

    def __init__(self, side: ArmSide, config: ArmConfig | None = None, mirror_sign: int = 1) -> None:
        """初始化单只机械臂服务。

        Parameters
        ----------
        side:
            机械臂侧别，取值为 `left` 或 `right`。
        config:
            单只机械臂配置。为 `None` 时使用 CASIA 默认配置。
        mirror_sign:
            镜像安装方向标记。左臂通常为 `1`，右臂通常为 `-1`。
        """

        if mirror_sign not in {-1, 1}:
            raise ValueError("mirror_sign 只能为 -1 或 1")
        self.side = side
        self.mirror_sign = mirror_sign
        self._config = config or default_casia_arm_config()
        self._validate_config(self._config)
        zero_values = tuple(0.0 for _ in self._config.movable_range)
        self._state = ArmState(
            current_joint_values=zero_values,
            target_joint_values=zero_values,
            speed=zero_values,
            acceleration=zero_values,
        )

    def get_movable_range(self) -> tuple[JointLimit, ...]:
        """获取单只机械臂可动范围。

        Returns
        -------
        tuple[JointLimit, ...]
            单只机械臂可动范围表，顺序与关节值序列一致。
        """

        return self._config.movable_range

    def get_speed_limit(self) -> tuple[MotionLimit, ...]:
        """获取单只机械臂速度限制。

        Returns
        -------
        tuple[MotionLimit, ...]
            单只机械臂速度限制表，顺序与关节值序列一致。
        """

        return self._config.speed_limit

    def get_acceleration_limit(self) -> tuple[MotionLimit, ...]:
        """获取单只机械臂加减速度限制。

        Returns
        -------
        tuple[MotionLimit, ...]
            单只机械臂加减速度限制表，顺序与关节值序列一致。
        """

        return self._config.acceleration_limit

    def get_physical_parameters(self) -> ArmPhysicalParameters:
        """获取单只机械臂关键物理参数。

        Returns
        -------
        ArmPhysicalParameters
            单只机械臂关键物理参数。
        """

        return self._config.physical_parameters

    def get_current_joint_values(self) -> tuple[float, ...]:
        """获取当前关节值。

        Returns
        -------
        tuple[float, ...]
            当前关节值序列，单位与可动范围表一致。
        """

        return self._state.current_joint_values

    def set_target_joint_values(self, joint_values: tuple[float, ...]) -> None:
        """设置目标关节值。

        Parameters
        ----------
        joint_values:
            目标关节值序列，形状语义为 `(N,)`，N 必须等于单臂关节数量。

        Raises
        ------
        ValueError
            当关节数量不匹配或任一值超出可动范围时抛出。
        """

        values = coerce_float_tuple(joint_values)
        validate_joint_values(values, self._config.movable_range)
        self._state.target_joint_values = values
        self._state.current_joint_values = values

    def set_speed(self, speed: tuple[float, ...]) -> None:
        """设置机械臂速度。

        Parameters
        ----------
        speed:
            速度序列，形状语义为 `(N,)`，单位与速度限制表一致。

        Raises
        ------
        ValueError
            当数量不匹配或任一绝对值超过速度限制时抛出。
        """

        values = coerce_float_tuple(speed)
        validate_motion_values(values, self._config.speed_limit, "速度")
        self._state.speed = values

    def set_acceleration(self, acceleration: tuple[float, ...]) -> None:
        """设置机械臂加速度。

        Parameters
        ----------
        acceleration:
            加速度序列，形状语义为 `(N,)`，单位与加减速度限制表一致。

        Raises
        ------
        ValueError
            当数量不匹配或任一绝对值超过加减速度限制时抛出。
        """

        values = coerce_float_tuple(acceleration)
        validate_motion_values(values, self._config.acceleration_limit, "加速度")
        self._state.acceleration = values

    def get_target_joint_values(self) -> tuple[float, ...]:
        """获取目标关节值。

        Returns
        -------
        tuple[float, ...]
            目标关节值序列，单位与可动范围表一致。
        """

        return self._state.target_joint_values

    def get_speed(self) -> tuple[float, ...]:
        """获取当前速度设置。

        Returns
        -------
        tuple[float, ...]
            当前速度设置序列，单位与速度限制表一致。
        """

        return self._state.speed

    def get_acceleration(self) -> tuple[float, ...]:
        """获取当前加速度设置。

        Returns
        -------
        tuple[float, ...]
            当前加速度设置序列，单位与加减速度限制表一致。
        """

        return self._state.acceleration

    def _validate_config(self, config: ArmConfig) -> None:
        """校验机械臂配置。

        Parameters
        ----------
        config:
            待校验的单只机械臂配置。

        Raises
        ------
        ValueError
            当配置长度不一致或范围非法时抛出。
        """

        count = len(config.movable_range)
        if count == 0:
            raise ValueError("机械臂至少需要 1 个关节")
        if len(config.speed_limit) != count or len(config.acceleration_limit) != count:
            raise ValueError("机械臂可动范围、速度限制、加减速度限制长度必须一致")
        for item in config.movable_range:
            if item.minimum > item.maximum:
                raise ValueError(f"机械臂关节范围非法：{item.name}")


# endregion
