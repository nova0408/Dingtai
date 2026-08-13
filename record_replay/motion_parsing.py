"""回放动作文本到结构化目标的解析。"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

# region 数据结构


@dataclass(frozen=True, slots=True)
class ParsedArmPose:
    """CSV 笛卡尔目标的结构化表示。"""

    xyz_mm: tuple[float, float, float]
    "平移坐标，单位 mm。"
    rpy_deg: tuple[float, float, float]
    "SciPy 小写外禀 xyz 欧拉角，单位 deg。"
    has_elbow: bool
    "肘部约束开关。"
    elbow_deg: float
    "肘部角度，单位 deg。"
    conf_data: tuple[int, ...]
    "控制器构型数据。"


# endregion


# region 解析入口


def parse_joint_values(joints_text: str, expected_len: int = 7) -> list[float]:
    """解析 CSV joints 单元格。"""

    if joints_text.strip().lower() == "nan":
        raise ValueError("关节列为 NaN，不能解析为关节目标")
    parsed = ast.literal_eval(joints_text)
    if not isinstance(parsed, list) or len(parsed) != expected_len:
        raise ValueError(f"关节列长度无效：{joints_text}")
    return [_finite_float(value, "关节列") for value in parsed]


def parse_pose_values(pose_text: str) -> ParsedArmPose:
    """解析包含 elbow 与 confData 的完整 9 元 CSV pose。"""

    if pose_text.strip().lower() == "nan":
        raise ValueError("pose 列为 NaN，不能解析为笛卡尔目标")
    parsed = ast.literal_eval(pose_text.strip())
    if not isinstance(parsed, list) or len(parsed) != 9:
        raise ValueError("笛卡尔目标必须是包含 elbow/confData 的 9 元 list")
    if not isinstance(parsed[6], bool):
        raise ValueError("笛卡尔目标 has_elbow 必须是 bool")
    if not isinstance(parsed[8], list) or len(parsed[8]) != 8:
        raise ValueError("笛卡尔目标 confData 必须是 8 元 int list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in parsed[8]):
        raise ValueError("笛卡尔目标 confData 必须全部为 int")
    return ParsedArmPose(
        (
            _finite_float(parsed[0], "笛卡尔目标 xyz"),
            _finite_float(parsed[1], "笛卡尔目标 xyz"),
            _finite_float(parsed[2], "笛卡尔目标 xyz"),
        ),
        (
            _finite_float(parsed[3], "笛卡尔目标 rpy"),
            _finite_float(parsed[4], "笛卡尔目标 rpy"),
            _finite_float(parsed[5], "笛卡尔目标 rpy"),
        ),
        parsed[6],
        _finite_float(parsed[7], "笛卡尔目标 elbow"),
        tuple(parsed[8]),
    )


def _finite_float(value: object, label: str) -> float:
    """把 CSV 数值收紧为有限的非 bool 浮点数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必须是数字：{value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须是有限数：{value!r}")
    return result


# endregion
