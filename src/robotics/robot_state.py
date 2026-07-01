from __future__ import annotations

from dataclasses import dataclass, field

from src.robotics.kinematic_models import (
    ArmKinematicProtocol,
    ArmMountState,
    ArmMountStateProtocol,
    LiftState,
    LiftStateProtocol,
    MobileBaseState,
    MobileBaseStateProtocol,
    PalmKinematicProtocol,
    PalmMountState,
    PalmMountStateProtocol,
)
from src.utils.datas import Transform

# region 数据结构


@dataclass(slots=True)
class DualArmRobotState:
    """双臂复合机器人状态对象。

    该对象统一管理以下串联关系：
    `世界 -> 基座 (AGV) -> 举升机构 -> 左/右肩 -> 左/右手臂末端 (TCP) -> 左/右手掌抓取位姿`。

    职责边界：
    - 负责持有当前状态并提供便捷位姿查询方法。
    - 不负责网络通信、硬件驱动、轨迹规划和控制下发。

    设计思想：
    - 将机构状态（安装关系/关节值）与运动学模型（FK）解耦，避免不同手掌型号导致状态结构反复改动。
    - 使用显式左右臂字段，避免字符串键动态分发，提升可读性与可调试性。

    线程语义：
    - 当前类不加锁，默认由上层在单线程或外部互斥条件下更新。
    - 若未来跨线程更新，可在外层引入状态快照或读写锁。
    """

    base_state: MobileBaseStateProtocol = field(default_factory=MobileBaseState)
    """基座状态。"""

    lift_state: LiftStateProtocol = field(default_factory=LiftState)
    """举升机构状态。"""

    left_arm_state: ArmMountStateProtocol = field(default_factory=ArmMountState)
    """左臂肩部安装关系与关节状态。"""

    right_arm_state: ArmMountStateProtocol = field(default_factory=ArmMountState)
    """右臂肩部安装关系与关节状态。"""

    left_palm_state: PalmMountStateProtocol = field(default_factory=PalmMountState)
    """左手掌安装关系与关节状态。"""

    right_palm_state: PalmMountStateProtocol = field(default_factory=PalmMountState)
    """右手掌安装关系与关节状态。"""

    left_arm_model: ArmKinematicProtocol | None = None
    """左臂 FK 模型。"""

    right_arm_model: ArmKinematicProtocol | None = None
    """右臂 FK 模型。"""

    left_palm_model: PalmKinematicProtocol | None = None
    """左手掌 FK 模型。"""

    right_palm_model: PalmKinematicProtocol | None = None
    """右手掌 FK 模型。"""

    # endregion

    # region 位姿查询

    def get_base_pose(self) -> Transform:
        """获取世界到基座的位姿。"""

        return self.base_state.world_to_base

    def get_lift_end_pose(self) -> Transform:
        """获取世界到举升机构末端位姿。"""

        return self.get_base_pose() @ self.lift_state.base_to_lift_end

    def get_left_shoulder_pose(self) -> Transform:
        """获取世界到左肩位姿。"""

        return self.get_lift_end_pose() @ self.left_arm_state.lift_end_to_shoulder

    def get_right_shoulder_pose(self) -> Transform:
        """获取世界到右肩位姿。"""

        return self.get_lift_end_pose() @ self.right_arm_state.lift_end_to_shoulder

    def get_left_arm_tcp_pose(self) -> Transform:
        """获取世界到左臂 TCP 位姿。"""

        if self.left_arm_model is None:
            raise ValueError("未配置 left_arm_model，无法计算左臂 TCP 位姿")
        shoulder_to_tcp = self.left_arm_model.solve_tcp(self.left_arm_state.joint_positions)
        return self.get_left_shoulder_pose() @ shoulder_to_tcp

    def get_right_arm_tcp_pose(self) -> Transform:
        """获取世界到右臂 TCP 位姿。"""

        if self.right_arm_model is None:
            raise ValueError("未配置 right_arm_model，无法计算右臂 TCP 位姿")
        shoulder_to_tcp = self.right_arm_model.solve_tcp(self.right_arm_state.joint_positions)
        return self.get_right_shoulder_pose() @ shoulder_to_tcp

    def get_left_opening_detection(self) -> Transform:
        """获取世界到左手抓取位姿。"""

        if self.left_palm_model is None:
            raise ValueError("未配置 left_palm_model，无法计算左手抓取位姿")
        palm_base_to_grasp = self.left_palm_model.solve_opening_detection(self.left_palm_state.joint_positions)
        return self.get_left_arm_tcp_pose() @ self.left_palm_state.arm_tcp_to_palm_base @ palm_base_to_grasp

    def get_right_opening_detection(self) -> Transform:
        """获取世界到右手抓取位姿。"""

        if self.right_palm_model is None:
            raise ValueError("未配置 right_palm_model，无法计算右手抓取位姿")
        palm_base_to_grasp = self.right_palm_model.solve_opening_detection(self.right_palm_state.joint_positions)
        return self.get_right_arm_tcp_pose() @ self.right_palm_state.arm_tcp_to_palm_base @ palm_base_to_grasp

    # endregion
