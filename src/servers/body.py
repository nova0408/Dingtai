from __future__ import annotations

from dataclasses import dataclass

from src.servers.common import JointLimit, MotionLimit, coerce_float_tuple, validate_joint_values, validate_motion_values


# region 数据结构


@dataclass(frozen=True, slots=True)
class BodyPhysicalParameters:
    """机器人非机械臂本体关键物理参数。

    职责边界：
    - 当前只表达接口文档中给出物理高度语义的升降柱参数。
    - 不猜测腰部、头部或升降机构的真实质量、惯量和安装尺寸。

    设计思想：
    - 只暴露协议文档明确存在的物理参数。
    - 使用不可变结构便于服务层安全返回。

    生命周期：
    - 可跨线程读取，不持有硬件连接。

    继承关系：
    - 不继承业务基类，保持协议数据职责。
    """

    lift_min_height_mm: int
    "升降柱最小物理高度，单位 mm。"

    lift_max_height_mm: int
    "升降柱最大物理高度，单位 mm。"


@dataclass(frozen=True, slots=True)
class BodyConfig:
    """非机械臂本体控制配置。

    职责边界：
    - 保存腰部、头部、升降柱的范围、速度、加速度和物理参数。
    - 不包含机械臂、底座、手部或相机接口。

    设计思想：
    - 将非机械臂本体从机械臂拆出，避免左右臂镜像关系污染本体配置。
    - 默认值集中在配置对象中，便于后续替换为设备描述文件。

    生命周期：
    - 可跨线程读取，不持有硬件连接。

    继承关系：
    - 不继承业务基类，保持配置职责。
    """

    movable_range: tuple[JointLimit, ...]
    "非机械臂本体可动范围表，顺序与关节值序列一致。"

    speed_limit: tuple[MotionLimit, ...]
    "非机械臂本体速度限制表，顺序与关节值序列一致。"

    acceleration_limit: tuple[MotionLimit, ...]
    "非机械臂本体加减速度限制表，顺序与关节值序列一致。"

    physical_parameters: BodyPhysicalParameters
    "非机械臂本体关键物理参数。"


@dataclass(slots=True)
class BodyState:
    """非机械臂本体运行状态。

    职责边界：
    - 保存腰部、头部、升降柱当前值、目标值、速度和加速度设置。
    - 不代表真实硬件反馈，也不负责运动插补。

    设计思想：
    - 状态对象与配置对象分离，配置负责约束，状态负责当前命令值。
    - 使用 tuple 保持序列不会被调用方原地修改。

    生命周期：
    - 当前状态仅在进程内存中维护。
    - 后续接入硬件时，应由硬件反馈刷新 `current_joint_values`。

    继承关系：
    - 不继承业务基类，保持服务内部状态职责。
    """

    current_joint_values: tuple[float, ...]
    "当前本体轴值序列，单位与 `BodyConfig.movable_range` 对齐。"

    target_joint_values: tuple[float, ...]
    "目标本体轴值序列，单位与 `BodyConfig.movable_range` 对齐。"

    speed: tuple[float, ...]
    "当前速度设置序列，单位与 `BodyConfig.speed_limit` 对齐。"

    acceleration: tuple[float, ...]
    "当前加速度设置序列，单位与 `BodyConfig.acceleration_limit` 对齐。"


# endregion


# region 配置


def default_body_config() -> BodyConfig:
    """创建非机械臂本体默认配置。

    Returns
    -------
    BodyConfig
        非机械臂本体默认配置。
    """

    movable_range = (
        JointLimit("waist_pitch", -30.0, 30.0, "deg"),
        JointLimit("head_pitch", -45.0, 45.0, "deg"),
        JointLimit("head_yaw", -90.0, 90.0, "deg"),
        JointLimit("lift_height", 0.0, 1.0, "ratio"),
    )
    speed_limit = tuple(
        MotionLimit(item.name, 90.0 if item.unit == "deg" else 0.5, "deg/s" if item.unit == "deg" else "ratio/s")
        for item in movable_range
    )
    acceleration_limit = tuple(
        MotionLimit(
            item.name,
            180.0 if item.unit == "deg" else 1.0,
            "deg/s^2" if item.unit == "deg" else "ratio/s^2",
        )
        for item in movable_range
    )
    return BodyConfig(
        movable_range=movable_range,
        speed_limit=speed_limit,
        acceleration_limit=acceleration_limit,
        physical_parameters=BodyPhysicalParameters(lift_min_height_mm=0, lift_max_height_mm=1000),
    )


# endregion


# region 主入口


class BodyServer:
    """非机械臂本体最小控制服务。

    职责边界：
    - 提供腰部、头部、升降柱的范围、速度、加速度、物理参数与目标值接口。
    - 不负责机械臂、底座、手部、gRPC 绑定、硬件通信、轨迹规划或碰撞检测。

    设计思想：
    - 将非机械臂本体与机械臂拆开，保持单一职责。
    - 所有写入先经过长度和范围校验，避免生成非法状态。

    生命周期：
    - 实例可在单线程服务流程中复用。
    - 当前类不持有硬件连接、线程、协程或文件句柄。
    - 多线程场景需要在外层增加锁或消息队列。

    继承关系：
    - 不继承业务基类，便于后续按 gRPC 或硬件协议适配。
    """

    def __init__(self, config: BodyConfig | None = None) -> None:
        """初始化非机械臂本体服务。

        Parameters
        ----------
        config:
            非机械臂本体配置。为 `None` 时使用默认配置。
        """

        self._config = config or default_body_config()
        self._validate_config(self._config)
        zero_values = tuple(0.0 for _ in self._config.movable_range)
        self._state = BodyState(
            current_joint_values=zero_values,
            target_joint_values=zero_values,
            speed=zero_values,
            acceleration=zero_values,
        )

    def get_movable_range(self) -> tuple[JointLimit, ...]:
        """获取非机械臂本体可动范围。

        Returns
        -------
        tuple[JointLimit, ...]
            非机械臂本体可动范围表，顺序与关节值序列一致。
        """

        return self._config.movable_range

    def get_speed_limit(self) -> tuple[MotionLimit, ...]:
        """获取非机械臂本体速度限制。

        Returns
        -------
        tuple[MotionLimit, ...]
            非机械臂本体速度限制表，顺序与关节值序列一致。
        """

        return self._config.speed_limit

    def get_acceleration_limit(self) -> tuple[MotionLimit, ...]:
        """获取非机械臂本体加减速度限制。

        Returns
        -------
        tuple[MotionLimit, ...]
            非机械臂本体加减速度限制表，顺序与关节值序列一致。
        """

        return self._config.acceleration_limit

    def get_physical_parameters(self) -> BodyPhysicalParameters:
        """获取非机械臂本体关键物理参数。

        Returns
        -------
        BodyPhysicalParameters
            非机械臂本体关键物理参数。
        """

        return self._config.physical_parameters

    def get_current_joint_values(self) -> tuple[float, ...]:
        """获取当前本体轴值。

        Returns
        -------
        tuple[float, ...]
            当前本体轴值序列，单位与可动范围表一致。
        """

        return self._state.current_joint_values

    def set_target_joint_values(self, joint_values: tuple[float, ...]) -> None:
        """设置非机械臂本体目标值。

        Parameters
        ----------
        joint_values:
            目标本体轴值序列，形状语义为 `(N,)`，N 必须等于本体轴数量。

        Raises
        ------
        ValueError
            当数量不匹配或任一值超出可动范围时抛出。
        """

        values = coerce_float_tuple(joint_values)
        validate_joint_values(values, self._config.movable_range)
        self._state.target_joint_values = values
        self._state.current_joint_values = values

    def set_speed(self, speed: tuple[float, ...]) -> None:
        """设置非机械臂本体速度。

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
        """设置非机械臂本体加速度。

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
        """获取目标本体轴值。

        Returns
        -------
        tuple[float, ...]
            目标本体轴值序列，单位与可动范围表一致。
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

    def _validate_config(self, config: BodyConfig) -> None:
        """校验非机械臂本体配置。

        Parameters
        ----------
        config:
            待校验的非机械臂本体配置。

        Raises
        ------
        ValueError
            当配置长度不一致或范围非法时抛出。
        """

        count = len(config.movable_range)
        if count == 0:
            raise ValueError("非机械臂本体至少需要 1 个受控轴")
        if len(config.speed_limit) != count or len(config.acceleration_limit) != count:
            raise ValueError("本体可动范围、速度限制、加减速度限制长度必须一致")
        for item in config.movable_range:
            if item.minimum > item.maximum:
                raise ValueError(f"本体轴范围非法：{item.name}")


# endregion
