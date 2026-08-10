"""双臂拖动示教业务的数据契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .motion_parsing import ParsedArmPose


# region 状态枚举


class ReplayServiceState(str, Enum):
    """常态化服务的可观测运行状态。"""

    IDLE = "idle"
    BUSY = "busy"
    RAPID_STOP = "rapid_stop"

    def __str__(self) -> str:
        """保持 Python 3.11 ``StrEnum`` 的字符串表现。"""

        return self.value


# endregion


# region 数据结构


ReplayErrorCode = Literal[
    "invalid_request",
    "invalid_index",
    "invalid_plan",
    "busy",
    "rapid_stop",
    "invalid_state",
    "stop_failed",
    "stop_requested",
    "execution_failed",
    "not_found",
    "internal_error",
]
"RecordReplay 对外返回的稳定错误码集合。"


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
class ReplayOffsetStatus:
    """一个 offset 来源在当前轮次的可观测状态。"""

    source: str
    "来源名称：head 或 three_ball。"
    available: bool
    "本轮是否已经得到该来源的有效 offset。"
    applied: bool
    "当前命名动作是否正在使用该来源；两个来源不能同时为 True。"


@dataclass(frozen=True, slots=True)
class ReplayStatusSnapshot:
    """可跨线程读取的当前执行状态快照。"""

    state: ReplayServiceState
    "服务阶段。"
    error_code: ReplayErrorCode | None = None
    "稳定错误码；正常状态为 null。"
    total_execution_count: int = 0
    "服务进程自启动以来已成功完成的回放轮次；HTTP 与 WebSocket 共用该值。"
    old_tray_current_index: int | None = None
    "本次执行使用的旧托盘当前位置 index。"
    old_tray_put_index: int | None = None
    "本次执行使用的旧托盘放置位置 index。"
    new_tray_current_index: int | None = None
    "本次执行使用的新托盘当前位置 index。"
    new_tray_put_index: int | None = None
    "本次执行使用的新托盘放置位置 index。"
    agv_navigation_enabled: bool | None = None
    "本次执行是否启用 AGV 导航。"
    agv_target: str | None = None
    "本次执行请求的 AGV 目标名称。"
    action_sequence_sha256: str | None = None
    "当前冻结动作顺序 JSON 的 SHA-256。"
    left_csv_state: str | None = None
    "当前左臂命名动作对应的 CSV 文件 stem。"
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
    current_left_action_name: str | None = None
    "左臂当前命名动作。"
    current_left_action_index: int | None = None
    "左臂当前多目标动作 index。"
    current_right_csv: str | None = None
    "右臂当前正在处理的 CSV 文件名。"
    current_right_action_name: str | None = None
    "右臂当前命名动作。"
    current_right_action_index: int | None = None
    "右臂当前多目标动作 index。"
    current_left_row: int | None = None
    "左臂当前处理到的 CSV 数据行，从 1 开始。"
    current_right_row: int | None = None
    "右臂当前处理到的 CSV 数据行，从 1 开始。"
    current_left_total_rows: int | None = None
    "左臂当前 CSV 的总数据行数。"
    current_right_total_rows: int | None = None
    "右臂当前 CSV 的总数据行数。"
    offset_statuses: tuple[ReplayOffsetStatus, ...] = (
        ReplayOffsetStatus("head", False, False),
        ReplayOffsetStatus("three_ball", False, False),
    )
    "头部 offset 与三球 offset 的列表状态。"


@dataclass(frozen=True, slots=True)
class ReplayExecutionCompletedEvent:
    """一次成功回放完成时发送给当前 WebSocket 订阅者的结束事件。"""

    snapshot: ReplayStatusSnapshot
    "完成后、计数已递增的最终状态快照。"


# endregion
