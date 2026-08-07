"""RecordReplay 进程重启后的最小运行状态持久化。"""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts import ReplayServiceState


class ReplayStateStore:
    """以 UTF-8 JSON 原子保存 idle/busy/rapid_stop 状态。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> ReplayServiceState:
        """读取上次状态；异常终止留下 busy 时强制进入 rapid_stop。"""

        if not self._path.is_file():
            return ReplayServiceState.IDLE
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            state = ReplayServiceState(str(payload["state"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ReplayServiceState.RAPID_STOP
        if state in {ReplayServiceState.BUSY, ReplayServiceState.RAPID_STOP}:
            return ReplayServiceState.RAPID_STOP
        return ReplayServiceState.IDLE

    def save(self, state: ReplayServiceState) -> None:
        """原子写入当前服务状态。"""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps({"state": state.value}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._path)
