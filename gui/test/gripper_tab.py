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

from gui.test.common import ActivatableTab, BackgroundCall, HoldRepeatController
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
        self._refresh_call = BackgroundCall(self)
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
        self._connect_background_signals()
        self.set_connection_ready(False)

    def _setup_timer(self) -> None:
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._request_refresh)

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

    def _connect_background_signals(self) -> None:
        self._refresh_call.succeeded.connect(self._on_refresh_succeeded)
        self._refresh_call.failed.connect(self._on_refresh_failed)

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
        self._request_refresh()

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
        self._refresh_call.start(self._ensure_enable)

    @Slot()
    def _on_enable_clicked(self) -> None:
        if self._client is None:
            return
        self._refresh_call.start(self._toggle_enable)

    # endregion

    # region 刷新

    def _refresh_status(self) -> None:
        if self._client is None:
            return
        self._request_refresh()

    def _request_refresh(self) -> None:
        if self._client is None:
            return
        self._refresh_call.start(self._read_status)

    def _read_status(self) -> GripperInfo:
        if self._client is None:
            raise RuntimeError("夹爪未连接")
        return self._client.get_status()

    @Slot(object)
    def _on_refresh_succeeded(self, payload: object) -> None:
        if not isinstance(payload, GripperInfo):
            return
        self._current_position = int(payload.position or 0)
        self._apply_status(payload)

    @Slot(str)
    def _on_refresh_failed(self, message: str) -> None:
        self._set_status_error(f"夹爪状态刷新失败: {message}")

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
        self._refresh_call.start(self._calibrate_async)

    def _set_position(self, position: int) -> None:
        if self._client is None:
            return
        self._refresh_call.start(lambda: self._set_position_async(int(position)))

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

    def _ensure_enable(self) -> GripperInfo:
        if self._client is None:
            raise RuntimeError("夹爪未连接")
        status = self._client.get_status()
        if not bool(status.enable):
            self._client.set_enable(True)
            status = self._client.get_status()
        return status

    def _toggle_enable(self) -> GripperInfo:
        if self._client is None:
            raise RuntimeError("夹爪未连接")
        status = self._client.get_status()
        self._client.set_enable(not bool(status.enable))
        return self._client.get_status()

    def _calibrate_async(self) -> GripperInfo:
        if self._client is None:
            raise RuntimeError("夹爪未连接")
        self._client.calibrate()
        return self._client.get_status()

    def _set_position_async(self, position: int) -> GripperInfo:
        if self._client is None:
            raise RuntimeError("夹爪未连接")
        self._client.set_pos(int(position))
        return self._client.get_status()

    # endregion
