from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from gui.test.common import ActivatableTab


class AlgoPlaceholderTabWidget(QWidget, ActivatableTab):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel("预留算法测试页。\n当前版本只保留占位，不接入算法运行时。", self)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        layout.addStretch(1)

    def set_active(self, active: bool) -> None:
        _ = active

    def set_connection_ready(self, ready: bool) -> None:
        self.label.setText(
            "预留算法测试页。\n"
            + ("当前整机连接已就绪，可在后续版本接入算法链路。" if ready else "当前未连接整机。")
        )
