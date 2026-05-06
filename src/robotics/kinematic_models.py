from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from collections.abc import Callable
from src.utils.datas import AngleType, Radian, Transform

JointAngleValue = AngleType
JointAngleTuple = tuple[JointAngleValue, ...]


def as_radian(value: JointAngleValue | float | int) -> Radian:
    """将角度值归一化为 `Radian`。

    约定：
    - `Radian` 原样返回；
    - 其它数字输入按“弧度”解释，与 URDF 的角度单位保持一致。
    """

    if isinstance(value, Radian):
        return value
    return Radian.from_radians(float(value))

# region 数据结构


@dataclass(frozen=True, slots=True)
class MobileBaseState:
    """移动基座状态。

    该对象描述 AGV 小车在世界坐标系中的当前姿态，是整机串联关系中的根节点。

    设计思想：
    - 只保存“世界到基座”的刚体变换，不绑定定位模块或通信对象。
    - 使用不可变 dataclass，避免多线程读写时出现隐式覆盖。

    继承关系：
    - 不继承业务基类，保持数据承载职责单一。
    """

    world_to_base: Transform = field(default_factory=Transform.Identity)
    """世界坐标系到基座坐标系的变换。"""


@dataclass(frozen=True, slots=True)
class LiftState:
    """举升机构状态。

    该对象表达举升/俯仰机构相对基座末端的当前变换，用于把基座姿态传递到双肩安装面。

    设计思想：
    - 将举升机构视为一个已求解的等效刚体变换，避免在状态层内嵌运动学细节。
    - 与硬件控制解耦，状态层仅承载计算输入。
    """

    base_to_lift_end: Transform = field(default_factory=Transform.Identity)
    """基座坐标系到举升机构末端坐标系的变换。"""


@dataclass(frozen=True, slots=True)
class ArmMountState:
    """肩部安装位姿与关节状态。

    该对象用于表示某一只手臂（左/右）在整机上的安装关系与当前关节角输入。

    设计思想：
    - 拆分“安装位姿”和“关节值”，方便同一套 FK 模型复用到不同肩部安装点。
    - `joint_positions` 保持“按模型约定顺序”的纯数据，不做隐式单位转换。
    """

    lift_end_to_shoulder: Transform = field(default_factory=Transform.Identity)
    """举升末端到该肩部坐标系的安装变换。"""

    joint_positions: JointAngleTuple = ()
    """当前关节位置序列，单位和顺序由对应 `ArmKinematicModel` 约定。"""


@dataclass(frozen=True, slots=True)
class PalmMountState:
    """手掌安装与关节状态。

    该对象表达“手掌底座”与“手臂末端法兰”之间的安装关系，并保存手掌自身自由度。

    设计思想：
    - 支持三自由度、五自由度等不同手掌结构，用同一数据契约表达。
    - 手掌与手臂解耦，便于按末端工具快速切换不同正解模型。
    """

    arm_tcp_to_palm_base: Transform = field(default_factory=Transform.Identity)
    """手臂 TCP 到手掌基坐标系的安装变换。"""

    joint_positions: JointAngleTuple = ()
    """手掌关节位置序列，单位和顺序由对应 `PalmKinematicModel` 约定。"""


@dataclass(frozen=True, slots=True)
class ArmKinematicInput:
    """手臂正运动学输入。

    Parameters
    ----------
    joint_positions:
        手臂关节位置序列，形状语义为 `(N,)`，N 可为 6、7 或其它模型定义值。
    """

    joint_positions: JointAngleTuple
    """手臂关节位置序列。"""


@dataclass(frozen=True, slots=True)
class PalmKinematicInput:
    """手掌正运动学输入。

    Parameters
    ----------
    joint_positions:
        手掌关节位置序列，形状语义为 `(M,)`，M 由手掌结构定义，例如 3 或 5。
    """

    joint_positions: JointAngleTuple
    """手掌关节位置序列。"""


# region 协议定义


@runtime_checkable
class MobileBaseStateProtocol(Protocol):
    """移动基座状态协议。"""

    @property
    def world_to_base(self) -> Transform:
        """世界坐标系到基座坐标系的变换。"""
        ...


@runtime_checkable
class LiftStateProtocol(Protocol):
    """举升机构状态协议。"""

    @property
    def base_to_lift_end(self) -> Transform:
        """基座坐标系到举升机构末端坐标系的变换。"""
        ...


@runtime_checkable
class ArmMountStateProtocol(Protocol):
    """肩部安装与关节状态协议。"""

    @property
    def lift_end_to_shoulder(self) -> Transform:
        """举升末端到肩部坐标系的安装变换。"""
        ...

    @property
    def joint_positions(self) -> JointAngleTuple:
        """手臂关节位置序列。"""
        ...


@runtime_checkable
class PalmMountStateProtocol(Protocol):
    """手掌安装与关节状态协议。"""

    @property
    def arm_tcp_to_palm_base(self) -> Transform:
        """手臂 TCP 到手掌基坐标系的安装变换。"""
        ...

    @property
    def joint_positions(self) -> JointAngleTuple:
        """手掌关节位置序列。"""
        ...


@runtime_checkable
class ArmKinematicProtocol(Protocol):
    """手臂运动学协议。

    该协议约束了手臂的正向与逆向运动学入口。你完成选型后，可直接实现该协议并替换当前默认实现。
    """

    @property
    def name(self) -> str:
        """模型名称。"""
        ...

    def solve_tcp(self, joint_positions: JointAngleTuple) -> Transform:
        """正解：关节 -> TCP 位姿。"""
        ...

    def solve_joints(self, target_tcp_pose: Transform, reference_joints: JointAngleTuple) -> JointAngleTuple:
        """逆解：目标 TCP 位姿 + 参考关节 -> 关节解。"""
        ...


@runtime_checkable
class PalmKinematicProtocol(Protocol):
    """手掌运动学协议。

    该协议约束手掌/夹爪等末端工具的正逆运动学入口，便于三自由度和五自由度实现统一接入。
    """

    @property
    def name(self) -> str:
        """模型名称。"""
        ...

    def solve_grasp_pose(self, joint_positions: JointAngleTuple) -> Transform:
        """正解：手掌关节 -> 抓取位姿。"""
        ...

    def solve_joints(self, target_grasp_pose: Transform, reference_joints: JointAngleTuple) -> JointAngleTuple:
        """逆解：目标抓取位姿 + 参考关节 -> 手掌关节解。"""
        ...


# endregion


@dataclass(frozen=True, slots=True)
class ArmKinematicModel:
    """手臂正运动学模型。

    该对象封装某类手臂（如 6 轴或 7 轴）的 FK 入口。状态对象通过该模型计算肩部到 TCP 的变换。

    设计思想：
    - 使用显式 `fk_solver` 回调替代动态字符串分发，避免魔术调用。
    - `name` 仅用于日志与业务标识，不参与求解逻辑。
    """

    name: str
    """模型名称，例如 `arm_6dof`、`arm_7dof`。"""

    fk_solver: Callable[[ArmKinematicInput], Transform]
    """正解函数：输入关节值，返回肩部到 TCP 的变换。"""

    ik_solver: Callable[[Transform, JointAngleTuple], JointAngleTuple] | None = None
    """逆解函数：输入目标 TCP 位姿与参考关节，返回关节解；允许先不提供。"""

    def solve_tcp(self, joint_positions: JointAngleTuple) -> Transform:
        """计算肩部到手臂 TCP 的位姿。

        Parameters
        ----------
        joint_positions:
            关节位置序列，顺序与单位由该模型定义。

        Returns
        -------
        Transform
            肩部坐标系到 TCP 坐标系的变换。
        """

        return self.fk_solver(ArmKinematicInput(joint_positions=joint_positions))

    def solve_joints(self, target_tcp_pose: Transform, reference_joints: JointAngleTuple) -> JointAngleTuple:
        """计算目标 TCP 位姿对应的关节解。

        Parameters
        ----------
        target_tcp_pose:
            目标 TCP 位姿。
        reference_joints:
            参考关节序列，用于多解选择。

        Returns
        -------
        JointAngleTuple
            逆解得到的关节序列。
        """

        if self.ik_solver is None:
            raise NotImplementedError(f"模型 {self.name} 未配置 ik_solver")
        return self.ik_solver(target_tcp_pose, reference_joints)


@dataclass(frozen=True, slots=True)
class PalmKinematicModel:
    """手掌正运动学模型。

    该对象封装不同手掌（例如 3 自由度/5 自由度）的 FK 入口，用于计算手掌基座到抓取位姿。

    设计思想：
    - 手掌正解独立于手臂正解，允许工具快速更换并保持状态层稳定。
    - 抓取位姿定义为手掌坐标系下的业务抓取姿态，不与夹爪控制命令耦合。
    """

    name: str
    """模型名称，例如 `gripper_3dof`、`gripper_5dof`。"""

    fk_solver: Callable[[PalmKinematicInput], Transform]
    """正解函数：输入手掌关节值，返回手掌基座到抓取位姿的变换。"""

    ik_solver: Callable[[Transform, JointAngleTuple], JointAngleTuple] | None = None
    """逆解函数：输入目标抓取位姿与参考关节，返回手掌关节解；允许先不提供。"""

    def solve_grasp_pose(self, joint_positions: JointAngleTuple) -> Transform:
        """计算手掌基座到抓取位姿的变换。"""

        return self.fk_solver(PalmKinematicInput(joint_positions=joint_positions))

    def solve_joints(self, target_grasp_pose: Transform, reference_joints: JointAngleTuple) -> JointAngleTuple:
        """计算目标抓取位姿对应的手掌关节解。"""

        if self.ik_solver is None:
            raise NotImplementedError(f"模型 {self.name} 未配置 ik_solver")
        return self.ik_solver(target_grasp_pose, reference_joints)


# endregion
