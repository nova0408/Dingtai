from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar
import xml.etree.ElementTree as ET
from collections.abc import Callable

# region 数据结构


@dataclass(frozen=True, slots=True)
class UrdfPose:
    """URDF 位姿描述。

    Parameters
    ----------
    xyz:
        平移向量 `(x, y, z)`，单位通常为米。
    rpy:
        欧拉角 `(roll, pitch, yaw)`，单位通常为弧度。
    """

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """平移分量 `(x, y, z)`。"""

    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """姿态分量 `(roll, pitch, yaw)`。"""


@dataclass(frozen=True, slots=True)
class UrdfJointLimit:
    """URDF 关节运动限制。"""

    lower: float | None = None
    """关节下限。"""

    upper: float | None = None
    """关节上限。"""

    effort: float | None = None
    """力/力矩限制。"""

    velocity: float | None = None
    """速度限制。"""


@dataclass(frozen=True, slots=True)
class UrdfLink:
    """URDF 链节定义。

    该结构只保存 URDF `link` 的关键静态信息，作为通用中间层数据模型。

    设计思想：
    - 保持最小字段集合，先满足结构转换与合法性检查。
    - 不绑定具体机构领域对象，便于后续映射到任意业务结构。
    """

    name: str
    """链节名称。"""


@dataclass(frozen=True, slots=True)
class UrdfJoint:
    """URDF 关节定义。

    Parameters
    ----------
    name:
        关节名称。
    joint_type:
        关节类型字符串，例如 `revolute`、`continuous`、`prismatic`、`fixed`。
    parent_link:
        父链节名称。
    child_link:
        子链节名称。
    origin:
        父链节到该关节坐标系的固定变换。
    axis:
        关节运动轴向，形状语义为 `(3,)`。
    limit:
        关节运动限制；`fixed` 关节通常为空。
    """

    name: str
    """关节名称。"""

    joint_type: str
    """关节类型字符串。"""

    parent_link: str
    """父链节名称。"""

    child_link: str
    """子链节名称。"""

    origin: UrdfPose = field(default_factory=UrdfPose)
    """父链节到关节坐标系的固定变换。"""

    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """关节轴向 `(x, y, z)`。"""

    limit: UrdfJointLimit | None = None
    """关节运动限制。"""


@dataclass(frozen=True, slots=True)
class UrdfModel:
    """URDF 通用中间模型。

    该结构是 URDF XML 与业务对象之间的数据契约，可被任意机构模块复用。

    设计思想：
    - 解析阶段统一输出 `UrdfModel`，业务阶段再做结构映射和约束校验。
    - 不承载网格、惯量细节，降低模型耦合与解析复杂度。
    """

    robot_name: str
    """机器人名称。"""

    links: tuple[UrdfLink, ...] = ()
    """链节列表，形状语义 `(N,)`。"""

    joints: tuple[UrdfJoint, ...] = ()
    """关节列表，形状语义 `(M,)`。"""


# endregion


# region 协议定义


TStructure = TypeVar("TStructure", covariant=True)


class UrdfLoadable(Protocol[TStructure]):
    """URDF 目标装载协议。

    满足该协议的结构可直接由 `UrdfModel` 装载，作为各机构模块的统一入口。
    """

    @classmethod
    def from_urdf_model(cls, model: UrdfModel) -> TStructure:
        """从通用 URDF 模型构造目标结构。"""
        ...


# endregion


# region 转换与校验


class UrdfValidationError(ValueError):
    """URDF 合法性校验失败异常。"""


@dataclass(slots=True)
class UrdfConverter:
    """URDF 双向转换器。

    职责：
    - 读取 URDF XML 文档并转换为 `UrdfModel`。
    - 将 `UrdfModel` 序列化为 URDF XML 文本。
    - 支持在装载目标结构前执行通用/定制合法性检查。

    线程语义：
    - 该类为无状态工具对象，可在多线程并行读取不同文件。
    """

    def from_xml_text(self, xml_text: str) -> UrdfModel:
        """将 URDF XML 文本转换为通用模型。"""

        root = ET.fromstring(xml_text)
        if root.tag != "robot":
            raise UrdfValidationError(f"URDF 根节点必须为 robot，当前为 {root.tag}")

        robot_name = root.attrib.get("name", "")
        links = tuple(UrdfLink(name=elem.attrib["name"]) for elem in root.findall("link") if "name" in elem.attrib)

        joints: list[UrdfJoint] = []
        for elem in root.findall("joint"):
            name = elem.attrib.get("name", "")
            joint_type = elem.attrib.get("type", "")
            parent_elem = elem.find("parent")
            child_elem = elem.find("child")
            origin_elem = elem.find("origin")
            axis_elem = elem.find("axis")
            limit_elem = elem.find("limit")
            parent_link = "" if parent_elem is None else parent_elem.attrib.get("link", "")
            child_link = "" if child_elem is None else child_elem.attrib.get("link", "")
            origin = UrdfPose(
                xyz=self._parse_vec3("" if origin_elem is None else origin_elem.attrib.get("xyz", "")),
                rpy=self._parse_vec3("" if origin_elem is None else origin_elem.attrib.get("rpy", "")),
            )
            axis = self._parse_vec3(
                "" if axis_elem is None else axis_elem.attrib.get("xyz", ""), default=(1.0, 0.0, 0.0)
            )
            limit = self._parse_joint_limit(limit_elem)
            joints.append(
                UrdfJoint(
                    name=name,
                    joint_type=joint_type,
                    parent_link=parent_link,
                    child_link=child_link,
                    origin=origin,
                    axis=axis,
                    limit=limit,
                )
            )

        return UrdfModel(robot_name=robot_name, links=links, joints=tuple(joints))

    def from_file(self, urdf_path: str | Path) -> UrdfModel:
        """从 URDF 文件读取并转换为通用模型。"""

        xml_text = Path(urdf_path).read_text(encoding="utf-8")
        return self.from_xml_text(xml_text)

    def to_xml_text(self, model: UrdfModel) -> str:
        """将通用模型序列化为 URDF XML 文本。"""

        root = ET.Element("robot", {"name": model.robot_name})
        for link in model.links:
            ET.SubElement(root, "link", {"name": link.name})

        for joint in model.joints:
            joint_elem = ET.SubElement(
                root,
                "joint",
                {
                    "name": joint.name,
                    "type": joint.joint_type,
                },
            )
            ET.SubElement(joint_elem, "parent", {"link": joint.parent_link})
            ET.SubElement(joint_elem, "child", {"link": joint.child_link})
            ET.SubElement(
                joint_elem,
                "origin",
                {
                    "xyz": self._format_vec3(joint.origin.xyz),
                    "rpy": self._format_vec3(joint.origin.rpy),
                },
            )
            ET.SubElement(joint_elem, "axis", {"xyz": self._format_vec3(joint.axis)})
            if joint.limit is not None:
                limit_attrs: dict[str, str] = {}
                if joint.limit.lower is not None:
                    limit_attrs["lower"] = self._format_scalar(joint.limit.lower)
                if joint.limit.upper is not None:
                    limit_attrs["upper"] = self._format_scalar(joint.limit.upper)
                if joint.limit.effort is not None:
                    limit_attrs["effort"] = self._format_scalar(joint.limit.effort)
                if joint.limit.velocity is not None:
                    limit_attrs["velocity"] = self._format_scalar(joint.limit.velocity)
                ET.SubElement(joint_elem, "limit", limit_attrs)

        return ET.tostring(root, encoding="unicode")

    def to_file(self, model: UrdfModel, output_path: str | Path) -> None:
        """将通用模型写出为 URDF 文件。"""

        xml_text = self.to_xml_text(model)
        Path(output_path).write_text(xml_text, encoding="utf-8")

    def load_to_structure(
        self,
        model: UrdfModel,
        loader: Callable[[UrdfModel], TStructure],
        validators: tuple[Callable[[UrdfModel], None], ...] = (),
    ) -> TStructure:
        """将通用模型装载为目标结构，并在装载前做合法性校验。"""

        for validator in validators:
            validator(model)
        return loader(model)

    @staticmethod
    def _parse_vec3(raw: str, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
        """解析 URDF 三元向量字符串。"""

        if raw.strip() == "":
            return default
        tokens = raw.strip().split()
        if len(tokens) != 3:
            raise UrdfValidationError(f"三元向量格式非法：'{raw}'")
        return (float(tokens[0]), float(tokens[1]), float(tokens[2]))

    @staticmethod
    def _parse_joint_limit(limit_elem: ET.Element | None) -> UrdfJointLimit | None:
        """解析 URDF 关节限制元素。"""

        if limit_elem is None:
            return None
        return UrdfJointLimit(
            lower=UrdfConverter._parse_optional_float(limit_elem.attrib.get("lower")),
            upper=UrdfConverter._parse_optional_float(limit_elem.attrib.get("upper")),
            effort=UrdfConverter._parse_optional_float(limit_elem.attrib.get("effort")),
            velocity=UrdfConverter._parse_optional_float(limit_elem.attrib.get("velocity")),
        )

    @staticmethod
    def _parse_optional_float(raw: str | None) -> float | None:
        """解析可选浮点数字段。"""

        if raw is None or raw.strip() == "":
            return None
        return float(raw)

    @staticmethod
    def _format_vec3(vec: tuple[float, float, float]) -> str:
        """格式化三元向量。"""

        return f"{vec[0]:.9g} {vec[1]:.9g} {vec[2]:.9g}"

    @staticmethod
    def _format_scalar(value: float) -> str:
        """格式化标量值。"""

        return f"{value:.9g}"


def count_joints_by_type(model: UrdfModel, joint_type: str) -> int:
    """统计指定类型关节数量。"""

    return sum(1 for joint in model.joints if joint.joint_type == joint_type)


def validate_joint_count(model: UrdfModel, joint_type: str, expected_count: int) -> None:
    """校验某类型关节数量。

    Parameters
    ----------
    model:
        URDF 通用模型。
    joint_type:
        目标关节类型，例如 `revolute`。
    expected_count:
        期望数量。

    Raises
    ------
    UrdfValidationError
        数量不一致时抛出。
    """

    actual_count = count_joints_by_type(model, joint_type)
    if actual_count != expected_count:
        raise UrdfValidationError(
            f"关节数量校验失败：type={joint_type}, expected={expected_count}, actual={actual_count}"
        )


def validate_joint_limits_present(model: UrdfModel, joint_types: tuple[str, ...] = ("revolute", "prismatic")) -> None:
    """校验指定类型关节是否具备 `limit` 信息。

    Parameters
    ----------
    model:
        URDF 通用模型。
    joint_types:
        需要限制信息的关节类型集合。
    """

    missing_limit_names = [
        joint.name for joint in model.joints if joint.joint_type in joint_types and joint.limit is None
    ]
    if missing_limit_names:
        raise UrdfValidationError(f"以下关节缺少 limit 定义：{', '.join(missing_limit_names)}")


# endregion
