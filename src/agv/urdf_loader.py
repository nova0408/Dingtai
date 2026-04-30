from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.robotics.urdf_interface import UrdfConverter, UrdfModel, validate_joint_count


# region 数据结构


@dataclass(frozen=True, slots=True)
class AgvUrdfStructure:
    """AGV URDF 封装结构。"""

    model: UrdfModel
    """URDF 通用中间模型。"""

    wheel_joint_count: int
    """轮系驱动关节数量（按 `continuous` 统计）。"""

    @classmethod
    def from_urdf_model(cls, model: UrdfModel) -> "AgvUrdfStructure":
        """从通用模型构造 AGV 结构。"""

        wheel_joint_count = sum(1 for joint in model.joints if joint.joint_type == "continuous")
        return cls(model=model, wheel_joint_count=wheel_joint_count)


# endregion


# region 业务入口


def load_agv_structure_from_urdf(urdf_path: str | Path, expected_continuous_count: int | None = None) -> AgvUrdfStructure:
    """读取 AGV URDF，并按需执行关节合法性检查。"""

    converter = UrdfConverter()
    model = converter.from_file(urdf_path)

    if expected_continuous_count is not None:
        validate_joint_count(model, joint_type="continuous", expected_count=expected_continuous_count)

    return converter.load_to_structure(model, AgvUrdfStructure.from_urdf_model)


# endregion
