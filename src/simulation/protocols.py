from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, TypeAlias

from src.robotics.kinematic_models import ArmKinematicProtocol, ArmMountState
from src.utils.datas import Axis, Color, Degree, Point, Radian, Transform

# region 数据结构


@dataclass(frozen=True, slots=True)
class JointUiSpec:
    """关节 UI 元信息

    该结构只承载交互层信息，不定义运动学自由度本体。
    自由度数量与当前关节值必须以 `ArmMountState.joint_positions` 为准。

    Parameters
    ----------
    name:
        关节展示名称
    min_value:
        滑条最小值
    max_value:
        滑条最大值
    default_value:
        滑条默认值
    """

    name: str
    """关节展示名称"""

    min_value: JointUiValue
    """滑条最小值"""

    max_value: JointUiValue
    """滑条最大值"""

    default_value: JointUiValue
    """滑条默认值"""


@dataclass(frozen=True, slots=True)
class JointAxisGlyph:
    """关节旋转副可视化信息"""

    axis: Axis
    """用于可视化的轴对象`axis.origin` 是起点，`axis.z_axis` 是箭头方向"""

    label: str
    """轴标签，例如 `j1`"""


@dataclass(frozen=True, slots=True)
class ChainSnapshot:
    """单条链绘制快照"""

    chain_name: str
    """链名称"""

    points: tuple[Point, ...]
    """链节点坐标序列"""

    color: Color
    """链绘制颜色"""

    joint_axes: tuple[JointAxisGlyph, ...] = ()
    """关节轴可视化序列"""


@dataclass(slots=True)
class ArmSimulationBinding:
    """单条机构链绑定

    职责边界：
    - `arm_state` 保存当前关节值；
    - `arm_model` 提供 FK/IK；
    - `base_transform/base_transform_solver` 决定该链在全局下的安装位姿

    Notes
    -----
    `link_point_solver` 仅负责渲染点序列，不参与真实关节定义与求解
    """

    chain_name: str
    """链名称"""

    arm_state: ArmMountState
    """当前关节状态"""

    arm_model: ArmKinematicProtocol
    """运动学协议实现"""

    base_transform: Transform = field(default_factory=Transform.Identity)
    """固定基座位姿"""

    base_transform_solver: Callable[[], Transform] | None = None
    """动态基座位姿求解器，优先级高于 `base_transform`"""

    joint_ui: tuple[JointUiSpec, ...] = ()
    """关节 UI 配置，长度应与 `arm_state.joint_positions` 一致"""

    color: Color = field(default_factory=lambda: Color.from_hex("#2a9d8f"))
    """绘图颜色"""

    link_point_solver: Callable[[tuple[float, ...]], tuple[Point, ...]] | None = None
    """可选渲染点求解器，返回 `Point` 序列"""


# endregion


# region 协议定义


class ArmSimulationModelProtocol(Protocol):
    """仿真模型协议"""

    def chain_names(self) -> tuple[str, ...]:
        """返回可交互链名称列表"""
        ...

    def get_binding(self, chain_name: str) -> ArmSimulationBinding:
        """按名称获取链绑定"""
        ...

    def snapshots(self) -> tuple[ChainSnapshot, ...]:
        """返回当前帧绘图快照"""
        ...


# endregion
JointAngularValue: TypeAlias = Degree | Radian
JointLinearValue: TypeAlias = float
JointUiValue: TypeAlias = JointAngularValue | JointLinearValue
