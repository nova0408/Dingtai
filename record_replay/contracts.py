"""双臂拖动示教业务的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .motion_parsing import ParsedArmPose


# region 状态枚举


class ReplayServiceState(str, Enum):
    """常态化服务的可观测运行状态。"""

    IDLE = "idle"
    BUSY = "busy"

    def __str__(self) -> str:
        """保持 Python 3.11 ``StrEnum`` 的字符串表现。"""

        return self.value


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
    joint_values: tuple[float, ...] | None = None
    "启动阶段解析后的关节数值；单位为 CSV 原始角度或执行器归一化值。"
    arm_joint_rad: tuple[float, ...] | None = None
    "arm 行预解析后的关节弧度。"
    arm_pose: ParsedArmPose | None = None
    "arm 行预解析后的笛卡尔目标。"
    pose_value: float | None = None
    "gripper/lift 行预解析后的标量值。"


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
class ReplayCsvFileStatus:
    """一个已部署回放 CSV 的只读摘要。"""

    name: str
    "CSV 文件名。"
    row_count: int
    "CSV 可执行数据行数。"


@dataclass(frozen=True, slots=True)
class ReplayExecutionTaskStatus:
    """一个按实际执行顺序对齐的左右臂任务。"""

    sequence: int
    "执行序号，从 1 开始。"
    left_csv: str | None
    "该阶段执行的左臂 CSV；右臂单独运动时为空。"
    right_csv: str | None
    "该阶段执行的右臂 CSV；左臂单独运动时为空。"
    synchronized: bool
    "左右臂是否在该阶段同步启动。"


@dataclass(frozen=True, slots=True)
class ReplayStatusSnapshot:
    """可跨线程读取的当前执行状态快照。"""

    state: ReplayServiceState
    "服务阶段。"
    left_csv_state: str | None = None
    "当前左臂 CSV 去前缀后的状态名。"
    plan_index: int | None = None
    "当前执行计划索引，从 0 开始。"
    error_text: str | None = None
    "失败时的错误文本。"
    left_csv_files: tuple[ReplayCsvFileStatus, ...] = ()
    "左臂目录中已部署的 CSV 及其数据行数。"
    right_csv_files: tuple[ReplayCsvFileStatus, ...] = ()
    "右臂目录中已部署的 CSV 及其数据行数。"
    execution_tasks: tuple[ReplayExecutionTaskStatus, ...] = ()
    "按实际执行顺序排列、左右臂对齐的任务清单。"
    current_task_index: int | None = None
    "当前任务在 execution_tasks 中的下标，从 0 开始。"
    current_task_active: bool = False
    "current_task_index 对应任务是否仍在执行。"
    current_left_csv: str | None = None
    "左臂当前正在处理的 CSV 文件名。"
    current_right_csv: str | None = None
    "右臂当前正在处理的 CSV 文件名。"
    current_left_row: int | None = None
    "左臂当前处理到的 CSV 数据行，从 1 开始。"
    current_right_row: int | None = None
    "右臂当前处理到的 CSV 数据行，从 1 开始。"
    current_left_total_rows: int | None = None
    "左臂当前 CSV 的总数据行数。"
    current_right_total_rows: int | None = None
    "右臂当前 CSV 的总数据行数。"


# endregion
