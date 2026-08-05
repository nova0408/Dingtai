"""双臂拖动示教自动执行业务包。"""

RECORD_REPLAY_VERSION = "1.12.0"
"RecordReplay 服务与人工 CLI 对齐的业务语义版本。"

from .context import ReplayContext
from .contracts import CsvExecutionPlan, ReplayRow, ReplayServiceState
from .settings import (
    OffsetConfig,
    ReplayArmSettings,
    ReplayCycleConfig,
    ReplayDeviceConnection,
    ReplayHandSettings,
    ReplayOffsetSettings,
    ReplayServiceSettings,
)

__all__ = [
    "CsvExecutionPlan",
    "OffsetConfig",
    "ReplayArmSettings",
    "ReplayContext",
    "ReplayCycleConfig",
    "ReplayDeviceConnection",
    "ReplayHandSettings",
    "ReplayOffsetSettings",
    "ReplayRow",
    "ReplayServiceSettings",
    "ReplayServiceState",
    "RECORD_REPLAY_VERSION",
]
