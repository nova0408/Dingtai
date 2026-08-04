"""回放 CSV 的读取和文件名语义解析。"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from .contracts import ReplayRow
from .motion_parsing import parse_joint_values, parse_pose_values


def discover_csv_paths(record_dir: Path, max_files: int | None = None) -> list[Path]:
    """发现并按文件名排序 CSV 路径。"""

    if not record_dir.is_dir():
        raise FileNotFoundError(f"CSV 目录不存在：{record_dir}")
    numbered_paths = [
        (int(path.name.split("_", maxsplit=1)[0]), path.name, path)
        for path in record_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and path.name.split("_", maxsplit=1)[0].isdigit()
    ]
    numbered_paths.sort(key=lambda item: (item[0], item[1]))
    paths = [path for _, _, path in numbered_paths]
    return paths if max_files is None else paths[:max_files]


def load_replay_rows(csv_path: Path) -> list[ReplayRow]:
    """读取一个 UTF-8 CSV 的动作记录。"""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[ReplayRow] = []
        for row_index, row in enumerate(reader, start=1):
            action_type = str(row.get("type", "")).strip().lower()
            if not action_type:
                raise ValueError(f"CSV 缺少 type: file={csv_path}, row={row_index}")
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
            elif action_type == "m11":
                joint_values = tuple(parse_joint_values(joints_text, expected_len=11))
            elif action_type in ("gripper", "lift"):
                pose_value = float(pose_text)
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


def extract_csv_sequence(csv_name: str) -> int:
    """解析文件名前缀的阶段序号。"""

    return int(csv_name.split("_", maxsplit=1)[0])


def extract_sync_csv_sequence(csv_name: str) -> int | None:
    """解析第二段 `Sxx` 声明的右臂同步序号。"""

    parts = csv_name.split("_")
    match = re.fullmatch(r"S(\d+)", parts[1]) if len(parts) >= 2 else None
    return None if match is None else int(match.group(1))


def state_name_from_left_csv(csv_name: str, prefix: str) -> str:
    """生成服务对外发布的左臂 CSV 状态名。"""

    stem = Path(csv_name).stem
    return stem[len(prefix) :] if stem.startswith(prefix) else stem
