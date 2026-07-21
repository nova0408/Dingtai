"""RecordReplay HTTP API 的显式响应协议。"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import ReplayServiceState
from .config_store import RuntimeParameters


@dataclass(frozen=True, slots=True)
class RecordReplayResponse:
    """HTTP API 返回的服务状态与可选持久化参数。"""

    state: ReplayServiceState
    accepted: bool = True
    left_csv_state: str | None = None
    plan_index: int | None = None
    error_text: str | None = None
    parameters: RuntimeParameters | None = None

