from __future__ import annotations

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.test.common import ActivatableTab, HoldRepeatController
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from qmlinker import GripperInfo
from src.wuji.dahuan_gripper_client import DahuanGripperClient


class GripperTabWidget(QWidget, ActivatableTab):
    HOLD_STEP = 20

    # region 初始化

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client: DahuanGripperClient | None = None
        self._active = False
        self._status_timer = QTimer(self)
        self._position_repeat = HoldRepeatController(self)
        self._current_position = 0

        self.enable_indicator: CasiaIndicatorLight
        self.online_label: QLabel
        self.calibrated_label: QLabel
        self.enable_label: QLabel
        self.position_label: QLabel
        self.state_label: QLabel
        self.position_spin: QSpinBox
        self.position_set_button: QPushButton
        self.position_minus_button: QPushButton
        self.position_plus_button: QPushButton
        self.calibrate_button: QPushButton

        self._setup_timer()
        self._setup_ui()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_timer(self) -> None:
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status)

    def _setup_ui(self) -> None:
        self.enable_indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        self.online_label = QLabel("-", self)
        self.calibrated_label = QLabel("-", self)
        self.enable_label = QLabel("-", self)
        self.position_label = QLabel("-", self)
        self.state_label = QLabel("-", self)
        self.position_spin = self._build_spin_box(0, 1000)
        self.position_set_button = QPushButton("set", self)
        self.position_minus_button = QPushButton("-", self)
        self.position_plus_button = QPushButton("+", self)
        self.calibrate_button = QPushButton("calibrate", self)

        layout = QHBoxLayout(self)
        layout.addWidget(self._build_status_group(), 1)
        layout.addWidget(self._build_control_group(), 1)

    def _build_spin_box(self, minimum: int, maximum: int) -> QSpinBox:
        spin_box = QSpinBox(self)
        spin_box.setRange(minimum, maximum)
        return spin_box

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("Status", self)
        layout = QGridLayout(group)
        layout.addWidget(self.enable_indicator, 0, 0, 1, 2)
        layout.addWidget(QLabel("online", self), 1, 0)
        layout.addWidget(self.online_label, 1, 1)
        layout.addWidget(QLabel("calibrated", self), 2, 0)
        layout.addWidget(self.calibrated_label, 2, 1)
        layout.addWidget(QLabel("enable", self), 3, 0)
        layout.addWidget(self.enable_label, 3, 1)
        layout.addWidget(QLabel("position", self), 4, 0)
        layout.addWidget(self.position_label, 4, 1)
        layout.addWidget(QLabel("state", self), 5, 0)
        layout.addWidget(self.state_label, 5, 1)
        return group

    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("Control", self)
        layout = QVBoxLayout(group)
        layout.addLayout(self._build_position_row())
        layout.addLayout(self._build_hold_row())
        layout.addWidget(self.calibrate_button)
        layout.addStretch(1)
        return group

    def _build_position_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(QLabel("pos", self))
        layout.addWidget(self.position_spin)
        layout.addWidget(self.position_set_button)
        return layout

    def _build_hold_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self.position_minus_button)
        layout.addWidget(self.position_plus_button)
        return layout

    def _connect_signals(self) -> None:
        self.enable_indicator.clicked.connect(self._on_enable_clicked)
        self.position_set_button.clicked.connect(self._on_position_set_clicked)
        self.position_minus_button.pressed.connect(self._on_position_minus_pressed)
        self.position_plus_button.pressed.connect(self._on_position_plus_pressed)
        self.position_minus_button.released.connect(self._position_repeat.stop)
        self.position_plus_button.released.connect(self._position_repeat.stop)
        self.calibrate_button.clicked.connect(self._calibrate)

    # endregion

    # region 生命周期

    def set_client(self, client: DahuanGripperClient | None) -> None:
        self._client = client
        self.set_connection_ready(client is not None)
        if client is None:
            self._status_timer.stop()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._status_timer.stop()
            return
        if self._client is None:
            self._set_status_unavailable()
            return
        self._try_enable()
        self._status_timer.start()
        self._refresh_status()

    def set_connection_ready(self, ready: bool) -> None:
        enabled = bool(ready)
        self.enable_indicator.setEnabled(enabled)
        for widget in (
            self.position_spin,
            self.position_set_button,
            self.position_minus_button,
            self.position_plus_button,
            self.calibrate_button,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self.enable_indicator.set_status(False)

    # endregion

    # region 使能

    def _try_enable(self) -> None:
        if self._client is None:
            return
        try:
            if not bool(self._client.get_status().enable):
                self._client.set_enable(True)
        except Exception:
            pass

    @Slot()
    def _on_enable_clicked(self) -> None:
        if self._client is None:
            return
        try:
            current_status = self._client.get_status()
            current_enabled = bool(current_status.enable)
            self._client.set_enable(not current_enabled)
            self._refresh_status()
        except Exception as exc:  # noqa: BLE001
            self._set_status_error(f"夹爪切换使能失败: {exc}")

    # endregion

    # region 刷新

    def _refresh_status(self) -> None:
        if self._client is None:
            return
        try:
            status = self._client.get_status()
            self._current_position = int(status.position or 0)
            self._apply_status(status)
        except Exception as exc:  # noqa: BLE001
            self._set_status_error(f"夹爪状态刷新失败: {exc}")

    # endregion

    # region 控制

    @Slot()
    def _on_position_set_clicked(self) -> None:
        self._set_position(int(self.position_spin.value()))

    @Slot()
    def _on_position_minus_pressed(self) -> None:
        self._position_repeat.start(lambda: self._nudge_position(-1))

    @Slot()
    def _on_position_plus_pressed(self) -> None:
        self._position_repeat.start(lambda: self._nudge_position(1))

    @Slot()
    def _calibrate(self) -> None:
        if self._client is None:
            return
        try:
            self._client.calibrate()
            self._refresh_status()
        except Exception as exc:  # noqa: BLE001
            self._set_status_error(f"校准失败: {exc}")

    def _set_position(self, position: int) -> None:
        if self._client is None:
            return
        try:
            self._client.set_pos(int(position))
            self._refresh_status()
        except Exception as exc:  # noqa: BLE001
            self._set_status_error(f"位置设置失败: {exc}")

    def _nudge_position(self, direction: int) -> None:
        target = min(max(self._current_position + direction * self.HOLD_STEP, 0), 1000)
        self._set_position(int(target))

    def _apply_status(self, status: GripperInfo) -> None:
        self.online_label.setText(str(bool(status.online)))
        self.calibrated_label.setText(str(bool(status.calibrated)))
        self.enable_label.setText(str(bool(status.enable)))
        self.position_label.setText(str(int(status.position or 0)))
        self.state_label.setText(str(int(status.state or 0)))
        self.enable_indicator.set_status(bool(status.enable))

    def _set_status_unavailable(self) -> None:
        self.online_label.setText("-")
        self.calibrated_label.setText("-")
        self.enable_label.setText("-")
        self.position_label.setText("-")
        self.state_label.setText("-")

    def _set_status_error(self, message: str) -> None:
        self.online_label.setText(message)
        self.calibrated_label.setText("-")
        self.enable_label.setText("-")
        self.position_label.setText("-")
        self.state_label.setText("-")

    # endregion
