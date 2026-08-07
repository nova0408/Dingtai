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

