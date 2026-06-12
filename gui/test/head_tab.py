from __future__ import annotations

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.test.common import ActivatableTab, AxisControlConfig, AxisControlRow, HoldRepeatController
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.arm.wuji_arm_protocol import WUJI_HEAD_AXIS_LIMITS
from src.wuji.head_client import WujiHeadClient


class HeadTabWidget(QWidget, ActivatableTab):
    HOLD_STEP_DEG = 2.0

    # region 初始化

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._client: WujiHeadClient |None= None
        self._active = False
        self._refresh_timer = QTimer(self)
        self._yaw_repeat = HoldRepeatController(self)
        self._pitch_repeat = HoldRepeatController(self)

        self.enable_indicator: CasiaIndicatorLight
        self.info_label: QLabel
        self.yaw_row: AxisControlRow
        self.pitch_row: AxisControlRow

        self._setup_timer()
        self._setup_ui()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_timer(self) -> None:
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._refresh_state)

    def _setup_ui(self) -> None:
        self.enable_indicator = self._build_enable_indicator()
        self.info_label = QLabel("head 未连接", self)
        self.yaw_row = self._build_yaw_row()
        self.pitch_row = self._build_pitch_row()

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(self._build_header_row())
        root_layout.addWidget(self._build_axis_group("Yaw", self.yaw_row))
        root_layout.addWidget(self._build_axis_group("Pitch", self.pitch_row))
        root_layout.addStretch(1)

    def _build_enable_indicator(self) -> CasiaIndicatorLight:
        return CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )

    def _build_header_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self.enable_indicator)
        layout.addWidget(self.info_label, 1)
        return layout

    def _build_yaw_row(self) -> AxisControlRow:
        return AxisControlRow(
            AxisControlConfig(
                axis_key="yaw",
                title="yaw",
                minimum=WUJI_HEAD_AXIS_LIMITS["head_yaw"].minimum,
                maximum=WUJI_HEAD_AXIS_LIMITS["head_yaw"].maximum,
                step=self.HOLD_STEP_DEG,
            ),
            repeat_controller=self._yaw_repeat,
            parent=self,
        )

    def _build_pitch_row(self) -> AxisControlRow:
        return AxisControlRow(
            AxisControlConfig(
                axis_key="pitch",
                title="pitch",
                minimum=-45.0,
                maximum=45.0,
                step=self.HOLD_STEP_DEG,
            ),
            repeat_controller=self._pitch_repeat,
            parent=self,
        )

    def _build_axis_group(self, title: str, row: AxisControlRow) -> QGroupBox:
        group = QGroupBox(title, self)
        layout = QVBoxLayout(group)
        layout.addWidget(row)
        return group

    def _connect_signals(self) -> None:
        self.enable_indicator.clicked.connect(self._on_enable_clicked)
        self.yaw_row.setRequested.connect(self._on_axis_target_requested)
        self.yaw_row.nudgeRequested.connect(self._on_axis_target_requested)
        self.pitch_row.setRequested.connect(self._on_axis_target_requested)
        self.pitch_row.nudgeRequested.connect(self._on_axis_target_requested)

    # endregion

    # region 生命周期

    def set_client(self, client: WujiHeadClient | None) -> None:
        self._client = client
        self.set_connection_ready(client is not None)
        if client is None:
            self._refresh_timer.stop()

    def set_active(self, active: bool) -> None:
        self._active = active
        if not self._active:
            self._refresh_timer.stop()
            return
        if self._client is None:
            self.info_label.setText("head 未连接")
            return
        self._try_enable_on_activate()
        self._refresh_timer.start()
        self._refresh_state()

    def set_connection_ready(self, ready: bool) -> None:
        self.enable_indicator.setEnabled(ready)
        self.yaw_row.set_row_enabled(ready)
        self.pitch_row.set_row_enabled(ready)
        if not ready:
            self.enable_indicator.set_status(False)
            self.yaw_row.set_current_value(None)
            self.pitch_row.set_current_value(None)

    # endregion

    # region 使能

    def _try_enable_on_activate(self) -> None:
        if not self._client:
            return
        try:
            if not self._client.get_enable():
                self._client.set_enable(True)
            self.enable_indicator.set_status(self._client.get_enable())
            self.info_label.setText(f"head enable={self._client.get_enable()}")
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"head enable 失败: {exc}")

    @Slot()
    def _on_enable_clicked(self) -> None:
        if not self._client:
            return
        try:
            current_enabled = self._client.get_enable()
            self._client.set_enable(not current_enabled)
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"head 使能切换失败: {exc}")

    # endregion

    # region 刷新

    def _refresh_state(self) -> None:
        if not self._client:
            return
        try:
            enabled = self._client.get_enable()
            yaw_value = float(self._client.get_head_yaw())
            pitch_value = float(self._client.get_head_pitch())
            self.enable_indicator.set_status(enabled)
            self.yaw_row.set_current_value(yaw_value, suffix="deg")
            self.pitch_row.set_current_value(pitch_value, suffix="deg")
            self.info_label.setText(f"head enable={enabled} yaw={yaw_value:.1f} pitch={pitch_value:.1f}")
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"head 刷新失败: {exc}")

    # endregion

    # region 控制

    @Slot(str, float)
    def _on_axis_target_requested(self, axis_key: str, value: float) -> None:
        if not self._client:
            return
        try:
            if axis_key == "yaw":
                self._client.set_head_yaw(float(value))
            else:
                self._client.set_head_pitch(float(value))
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"{axis_key} 设置失败: {exc}")

    # endregion

