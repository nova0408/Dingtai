from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.robotics.urdf_interface import (
    UrdfConverter,
    UrdfModel,
    validate_joint_count,
    validate_joint_limits_present,
)


# region 数据结构


@dataclass(frozen=True, slots=True)
class ArmUrdfStructure:
    """机械臂 URDF 封装结构。

    该结构作为 `src/arm` 模块中的领域对象入口，保留原始 `UrdfModel`，并补充结构化元数据。
    """

    model: UrdfModel
    """URDF 通用中间模型。"""

    axis_count: int
    """机械臂转动副数量。"""

    @classmethod
    def from_urdf_model(cls, model: UrdfModel) -> "ArmUrdfStructure":
        """从通用模型构造机械臂结构。"""

        axis_count = sum(1 for joint in model.joints if joint.joint_type == "revolute")
        return cls(model=model, axis_count=axis_count)


# endregion


# region 业务入口


def load_arm_structure_from_urdf(urdf_path: str | Path, expected_revolute_count: int) -> ArmUrdfStructure:
    """读取机械臂 URDF 并执行关节合法性检查。

    Parameters
    ----------
    urdf_path:
        URDF 文件路径。
    expected_revolute_count:
        期望转动副数量，例如六轴机械臂传入 `6`。

    Returns
    -------
    ArmUrdfStructure
        通过校验后的机械臂结构。

    Raises
    ------
    UrdfValidationError
        URDF 不满足关节数量约束时抛出。
    """

    converter = UrdfConverter()
    model = converter.from_file(urdf_path)
    validate_joint_count(model, joint_type="revolute", expected_count=expected_revolute_count)
    validate_joint_limits_present(model, joint_types=("revolute",))
    return converter.load_to_structure(model, ArmUrdfStructure.from_urdf_model)


def load_six_dof_arm_from_urdf(urdf_path: str | Path) -> ArmUrdfStructure:
    """读取六轴机械臂 URDF 并验证包含 6 个转动副。"""

    return load_arm_structure_from_urdf(urdf_path=urdf_path, expected_revolute_count=6)


# endregion
