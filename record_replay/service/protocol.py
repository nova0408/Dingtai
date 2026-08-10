"""RecordReplay HTTP API 的显式响应协议。"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import (
    ReplayCsvFileStatus,
    ReplayExecutionTaskStatus,
    ReplayOffsetStatus,
    ReplayServiceState,
)
from .config_store import RuntimeParameters


@dataclass(frozen=True, slots=True)
class PriorUploadResponse:
    """先验 JSON 替换结果。"""

    accepted: bool
    "服务是否已完成先验文件替换。"

    file_name: str
    "被替换的先验文件名；目标路径来自统一动作 JSON。"

    backup_file: str | None
    "旧文件在服务端 `.archive` 下的相对路径。"


@dataclass(frozen=True, slots=True)
class ReplayPlanAction:
    """启动前展示的一项只读回放动作。"""

    sequence: int
    "当前机械臂侧在展开循环中的执行序号，从 1 开始。"

    loop: int
    "该动作所属的循环序号，从 1 开始。"

    csv: str
    "实际执行的 CSV 文件名，保留磁盘上的数字前缀。"

    action_name: str
    "统一动作 JSON 中的 function_name。"

    action_type: str
    "动作类型：capture、fast 或 precise。"

    speed: float
    "动作 speed，单位 mm/s。"

    zone: float
    "动作 zone，单位 mm。"

    index: int | None = None
    "多目标动作 index；普通动作为空。"

    final_speed: float | None = None
    "capture 动作末点速度，单位 mm/s。"

    settle_delay: float | None = None
    "capture 动作稳定等待时间，单位 s。"

    row_count: int = 0
    "对应 CSV 的可执行数据行数。"


@dataclass(frozen=True, slots=True)
class RecordReplayPlanResponse:
    """启动前读取的冻结动作计划摘要。"""

    state: ReplayServiceState
    "当前服务阶段。"

    accepted: bool = True
    "计划是否成功读取。"

    action_sequence_sha256: str | None = None
    "本次读取的 action_sequence.json SHA-256。"

    loop_count: int = 0
    "动作顺序配置中的循环次数。"

    left: tuple[ReplayPlanAction, ...] = ()
    "左臂按实际执行顺序展开后的动作。"

    right: tuple[ReplayPlanAction, ...] = ()
    "右臂按实际执行顺序展开后的动作。"

    error_text: str | None = None
    "读取或校验失败原因。"


@dataclass(frozen=True, slots=True)
class RecordReplayResponse:
    """HTTP API 返回的服务状态与可选持久化参数。"""

    state: ReplayServiceState
    accepted: bool = True
    action_sequence_sha256: str | None = None
    left_csv_state: str | None = None
    plan_index: int | None = None
    error_text: str | None = None
    left_csv_files: tuple[ReplayCsvFileStatus, ...] = ()
    right_csv_files: tuple[ReplayCsvFileStatus, ...] = ()
    execution_tasks: tuple[ReplayExecutionTaskStatus, ...] = ()
    current_task_sequence: int = 0
    current_task_active: bool = False
    total_execution_count: int = 0
    current_left_csv: str | None = None
    current_left_action_name: str | None = None
    current_left_action_index: int | None = None
    current_right_csv: str | None = None
    current_right_action_name: str | None = None
    current_right_action_index: int | None = None
    current_left_row: int | None = None
    current_right_row: int | None = None
    current_left_total_rows: int | None = None
    current_right_total_rows: int | None = None
    offset_statuses: tuple[ReplayOffsetStatus, ...] = ()
    parameters: RuntimeParameters | None = None

