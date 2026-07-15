"""双臂拖动示教业务的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


# region 状态枚举


class ReplayServiceState(StrEnum):
    """常态化服务的可观测状态。"""

    WAITING = "waiting"
    NAVIGATING_TO_START = "navigating_to_start"
    REPLAYING = "replaying"
    NAVIGATING_TO_FINISH = "navigating_to_finish"
    FAILED = "failed"


# endregion


# region 数据结构


@dataclass(frozen=True, slots=True)
class ReplayRow:
    """一条 CSV 回放记录。"""

    csv_name: str
    "源 CSV 文件名。"
    row_index: int
    "CSV 数据行序号，从 1 开始。"
    action_type: str
    "动作类型，例如 arm、gripper、m11、lift。"
    joints_text: str
    "原始 joints 单元格文本。"
    pose_text: str
    "原始 pose 单元格文本。"


@dataclass(frozen=True, slots=True)
class CsvExecutionPlan:
    """一个左臂阶段及其关联右臂阶段。"""

    left_csv_path: Path
    "左臂主 CSV 路径。"
    right_start_csv_path: Path | None = None
    "启动时与左臂并行的右臂 CSV。"
    right_pre_stage_csv_paths: tuple[Path, ...] = ()
    "左臂前需要顺序执行的右臂 CSV。"
    right_sync_csv_path: Path | None = None
    "与左臂同步执行的右臂 CSV。"
    right_post_stage_csv_paths: tuple[Path, ...] = ()
    "最后一个左臂阶段后执行的右臂 CSV。"
    start_together: bool = False
    "是否在启动阶段并行执行左右 CSV。"


@dataclass(frozen=True, slots=True)
class ReplayStatusSnapshot:
    """可跨线程读取的当前执行状态快照。"""

    state: ReplayServiceState
    "服务阶段。"
    left_csv_state: str | None
    "当前左臂 CSV 去前缀后的状态名。"
    plan_index: int | None
    "当前执行计划索引，从 0 开始。"
    error_text: str | None
    "失败时的错误文本。"


# endregion
