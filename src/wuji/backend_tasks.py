from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


@dataclass(frozen=True, slots=True)
class _SdkRequest:
    action: str
    key: str
    axis_name: str | None = None
    device_name: str | None = None


class _SdkWorkerSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)


class _SdkWorker(QRunnable):
    def __init__(self, key: str, task: Any) -> None:
        super().__init__()
        self.key = key
        self.task = task
        self.signals = _SdkWorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.task()
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.key, f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(self.key, result)
