"""回放动作文本到结构化目标的解析。"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# region 数据结构


@dataclass(frozen=True, slots=True)
class ParsedArmPose:
    """CSV 笛卡尔目标的结构化表示。"""

    xyz_mm: tuple[float, float, float]
    "平移坐标，单位 mm。"
    rpy_deg: tuple[float, float, float]
    "SciPy 小写外禀 xyz 欧拉角，单位 deg。"
    has_elbow: bool | None
    "肘部约束开关；None 表示复用当前上下文。"
    elbow_deg: float | None
    "肘部角度，单位 deg。"
    conf_data: tuple[int, ...] | None
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
    return [float(value) for value in parsed]


def parse_pose_values(pose_text: str) -> ParsedArmPose:
    """解析 CSV pose 单元格的 6 或 9 元格式。"""

    if pose_text.strip().lower() == "nan":
        raise ValueError("pose 列为 NaN，不能解析为笛卡尔目标")
    parsed = ast.literal_eval(pose_text)
    if not isinstance(parsed, list):
        raise ValueError("笛卡尔目标必须是 list 格式")
    if len(parsed) == 6:
        return ParsedArmPose(
            (float(parsed[0]), float(parsed[1]), float(parsed[2])),
            (float(parsed[3]), float(parsed[4]), float(parsed[5])),
            None,
            None,
            None,
        )
    if len(parsed) != 9 or not isinstance(parsed[8], list):
        raise ValueError("笛卡尔目标必须为 6 元或含 confData 的 9 元 list")
    return ParsedArmPose(
        (float(parsed[0]), float(parsed[1]), float(parsed[2])),
        (float(parsed[3]), float(parsed[4]), float(parsed[5])),
        bool(parsed[6]),
        float(parsed[7]),
        tuple(int(value) for value in parsed[8]),
    )


# endregion
