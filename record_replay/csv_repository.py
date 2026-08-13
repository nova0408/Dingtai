"""回放 CSV 的读取和文件名语义解析。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .contracts import ReplayRow
from .motion_parsing import parse_joint_values, parse_pose_values

REQUIRED_COLUMNS = frozenset({"timestamp", "type", "joints", "pose"})
"回放 CSV 必须在 start 冻结阶段提供的完整列集合。"

SUPPORTED_ACTION_TYPES = frozenset({"arm", "m6", "gripper", "lift"})
"执行器显式支持的 CSV 动作类型。"


def discover_csv_paths(record_dir: Path, max_files: int | None = None) -> list[Path]:
    """发现新命名 CSV；执行顺序由 action_sequence.json 决定。"""

    if not record_dir.is_dir():
        raise FileNotFoundError(f"CSV 目录不存在：{record_dir}")
    paths = sorted(
        (
            path
            for path in record_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        ),
        key=lambda path: path.name,
    )
    return paths if max_files is None else paths[:max_files]


def load_replay_rows(csv_path: Path) -> list[ReplayRow]:
    """读取一个 UTF-8 CSV 的动作记录。"""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(f"CSV 缺少必要列 {missing_columns}: file={csv_path}")
        rows: list[ReplayRow] = []
        for row_index, row in enumerate(reader, start=1):
            action_type = str(row.get("type", "")).strip().lower()
            if not action_type:
                raise ValueError(f"CSV 缺少 type: file={csv_path}, row={row_index}")
            if action_type not in SUPPORTED_ACTION_TYPES:
                raise ValueError(
                    f"CSV 包含不支持的 type={action_type!r}: file={csv_path}, row={row_index}"
                )
            joints_text = str(row.get("joints", "")).strip()
            pose_text = str(row.get("pose", "")).strip()
            joint_values: tuple[float, ...] | None = None
            arm_joint_rad: tuple[float, ...] | None = None
            arm_pose = None
            pose_value: float | None = None
            if action_type == "arm":
                joint_values = tuple(parse_joint_values(joints_text))
                arm_joint_rad = tuple(math.radians(value) for value in joint_values)
                if pose_text.lower() != "nan":
                    arm_pose = parse_pose_values(pose_text)
            elif action_type == "m6":
                joint_values = tuple(parse_joint_values(joints_text, expected_len=6))
            elif action_type in ("gripper", "lift"):
                pose_value = float(pose_text)
                if not math.isfinite(pose_value):
                    raise ValueError(
                        f"CSV {action_type} pose 必须是有限数: file={csv_path}, row={row_index}"
                    )
            rows.append(
                ReplayRow(
                    csv_path.name,
                    row_index,
                    action_type,
                    joints_text,
                    pose_text,
                    joint_values,
                    arm_joint_rad,
                    arm_pose,
                    pose_value,
                )
            )
    return rows
