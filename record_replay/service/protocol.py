"""RecordReplay HTTP API 的显式响应协议。"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import (
    ReplayCsvFileStatus,
    ReplayExecutionTaskStatus,
    ReplayOffsetStatus,
    ReplayErrorCode,
    ReplayExecutionPhase,
    ReplayServiceState,
)
from .config_store import RuntimeParameters


RECORD_REPLAY_API_VERSION = "1"
"RecordReplay HTTP API 主版本。"


@dataclass(frozen=True, slots=True)
class RecordReplayHealthResponse:
    """不访问现场设备的服务健康信息。"""

    service_version: str
    "RecordReplay 当前业务版本，供客户端做版本校验。"

    api_version: str
    "对外 HTTP API 主版本。"

    state: ReplayServiceState
    "服务当前回放状态；读取状态不会触发任何动作。"

    hardware_access: str = "lazy"
    "健康检查不访问硬件，仅在实际业务请求中按需连接。"


@dataclass(frozen=True, slots=True)
class RecordReplayErrorResponse:
    """HTTP 非 2xx 响应使用的稳定 JSON 错误对象。"""

    error_code: ReplayErrorCode
    """机器可判定的稳定错误码。"""

    error_text: str
    """面向客户端和现场人员的中文错误说明。"""


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
    "当前机械臂侧在本次单次执行中的序号，从 1 开始。"

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

    error_code: ReplayErrorCode | None = None
    "稳定错误码；正常计划为 null。"

    accepted: bool = True
    "计划是否成功读取。"

    action_sequence_sha256: str | None = None
    "本次读取的 action_sequence.json SHA-256。"

    old_tray_current_index: int | None = None
    "本次预览使用的旧托盘当前位置 index。"

    old_tray_put_index: int | None = None
    "本次预览使用的旧托盘放置位置 index。"

    new_tray_current_index: int | None = None
    "本次预览使用的新托盘当前位置 index。"

    new_tray_put_index: int | None = None
    "本次预览使用的新托盘放置位置 index。"

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
    execution_phase: ReplayExecutionPhase = ReplayExecutionPhase.IDLE
    error_code: ReplayErrorCode | None = None
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
    old_tray_current_index: int | None = None
    old_tray_put_index: int | None = None
    new_tray_current_index: int | None = None
    new_tray_put_index: int | None = None
    agv_navigation_enabled: bool | None = None
    agv_target: str | None = None
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

