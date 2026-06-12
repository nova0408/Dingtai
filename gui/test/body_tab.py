from __future__ import annotations

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.test.common import ActivatableTab, AxisControlConfig, AxisControlRow, HoldRepeatController
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.arm.wuji_arm_protocol import WUJI_BODY_AXIS_LIMITS
from src.wuji.body_client import WujiBodyClient


class _BodyAxisPanel(QGroupBox):
    """body 单轴面板。"""

    def __init__(
        self,
        title: str,
        axis_key: str,
        minimum: float,
        maximum: float,
        step: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.axis_key = axis_key
        self.enable_indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        self.info_label = QLabel("-", self)
        self.repeat_controller = HoldRepeatController(self)
        self.row = AxisControlRow(
            AxisControlConfig(
                axis_key=axis_key,
                title=axis_key,
                minimum=minimum,
                maximum=maximum,
                step=step,
            ),
            repeat_controller=self.repeat_controller,
            parent=self,
        )
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        top_row.addWidget(self.enable_indicator)
        top_row.addWidget(self.info_label, 1)
        layout.addLayout(top_row)
        layout.addWidget(self.row)


class BodyTabWidget(QWidget, ActivatableTab):
    LIFT_MAX_HEIGHT_MM = WUJI_BODY_AXIS_LIMITS["body_z"].maximum
    HOLD_STEP_MM = 10.0
    HOLD_STEP_DEG = 2.0

    # region 初始化

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client: WujiBodyClient | None = None
        self._active = False
        self._refresh_timer = QTimer(self)

        self.info_label: QLabel
        self.lift_panel: _BodyAxisPanel
        self.waist_panel: _BodyAxisPanel

        self._setup_timer()
        self._setup_ui()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_timer(self) -> None:
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._refresh_state)

    def _setup_ui(self) -> None:
        self.info_label = QLabel("body 未连接", self)
        self.lift_panel = self._build_lift_panel()
        self.waist_panel = self._build_waist_panel()

        layout = QVBoxLayout(self)
        layout.addWidget(self.info_label)
        layout.addWidget(self.lift_panel)
        layout.addWidget(self.waist_panel)
        layout.addStretch(1)

    def _build_lift_panel(self) -> _BodyAxisPanel:
        limit = WUJI_BODY_AXIS_LIMITS["body_z"]
        return _BodyAxisPanel(
            "Lift",
            "lift",
            limit.minimum,
            limit.maximum,
            self.HOLD_STEP_MM,
            self,
        )

    def _build_waist_panel(self) -> _BodyAxisPanel:
        limit = WUJI_BODY_AXIS_LIMITS["body_ry"]
        return _BodyAxisPanel(
            "Waist",
            "waist",
            limit.minimum,
            limit.maximum,
            self.HOLD_STEP_DEG,
            self,
        )

    def _connect_signals(self) -> None:
        self.lift_panel.enable_indicator.clicked.connect(self._on_lift_enable_clicked)
        self.waist_panel.enable_indicator.clicked.connect(self._on_waist_enable_clicked)
        self.lift_panel.row.setRequested.connect(self._on_axis_target_requested)
        self.lift_panel.row.nudgeRequested.connect(self._on_axis_target_requested)
        self.waist_panel.row.setRequested.connect(self._on_axis_target_requested)
        self.waist_panel.row.nudgeRequested.connect(self._on_axis_target_requested)

    # endregion

    # region 生命周期

    def set_client(self, client: WujiBodyClient | None) -> None:
        self._client = client
        self.set_connection_ready(client is not None)
        if client is None:
            self._refresh_timer.stop()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._refresh_timer.stop()
            return
        if self._client is None:
            self.info_label.setText("body 未连接")
            return
        self._try_enable_modules()
        self._refresh_timer.start()
        self._refresh_state()

    def set_connection_ready(self, ready: bool) -> None:
        enabled = bool(ready)
        self.lift_panel.enable_indicator.setEnabled(enabled)
        self.waist_panel.enable_indicator.setEnabled(enabled)
        self.lift_panel.row.set_row_enabled(enabled)
        self.waist_panel.row.set_row_enabled(enabled)
        if not enabled:
            self.lift_panel.row.set_current_value(None)
            self.waist_panel.row.set_current_value(None)
            self.lift_panel.enable_indicator.set_status(False)
            self.waist_panel.enable_indicator.set_status(False)

    # endregion

    # region 使能

    def _try_enable_modules(self) -> None:
        if self._client is None:
            return
        try:
            if not self._client.lift.get_enable():
                self._client.lift.set_enable(True)
        except Exception:
            pass
        try:
            if not self._client.waist.get_enable():
                self._client.waist.set_enable(True)
        except Exception:
            pass

    @Slot()
    def _on_lift_enable_clicked(self) -> None:
        self._toggle_lift_enable()

    @Slot()
    def _on_waist_enable_clicked(self) -> None:
        self._toggle_waist_enable()

    def _toggle_lift_enable(self) -> None:
        if self._client is None:
            return
        try:
            current_enabled = self._client.lift.get_enable()
            self._client.lift.set_enable(not current_enabled)
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"lift 使能切换失败: {exc}")

    def _toggle_waist_enable(self) -> None:
        if self._client is None:
            return
        try:
            current_enabled = self._client.waist.get_enable()
            self._client.waist.set_enable(not current_enabled)
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"waist 使能切换失败: {exc}")

    # endregion

    # region 刷新

    def _refresh_state(self) -> None:
        if self._client is None:
            return
        try:
            lift_enabled = self._client.lift.get_enable()
            waist_enabled = self._client.waist.get_enable()
            lift_value_mm = self._read_lift_height_mm()
            waist_value_deg = self._read_waist_pitch_deg()
            self.lift_panel.enable_indicator.set_status(lift_enabled)
            self.waist_panel.enable_indicator.set_status(waist_enabled)
            self.lift_panel.row.set_current_value(lift_value_mm, suffix="mm")
            self.waist_panel.row.set_current_value(waist_value_deg, suffix="deg")
            self.lift_panel.info_label.setText(f"height={lift_value_mm:.1f} mm")
            self.waist_panel.info_label.setText(f"pitch={waist_value_deg:.1f} deg")
            self.info_label.setText(f"body lift={lift_enabled} waist={waist_enabled}")
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"body 刷新失败: {exc}")

    def _read_lift_height_mm(self) -> float:
        if self._client is None:
            return 0.0
        result = self._client.lift.get_lift_height()
        if result is None:
            raise RuntimeError("lift height unavailable")
        height_scale, _timestamp = result
        return float(height_scale) * float(self.LIFT_MAX_HEIGHT_MM)

    def _read_waist_pitch_deg(self) -> float:
        if self._client is None:
            return 0.0
        pitch_value = self._client.waist.get_waist_pitch()
        if pitch_value is None:
            raise RuntimeError("waist pitch unavailable")
        return float(pitch_value)

    # endregion

    # region 控制

    @Slot(str, float)
    def _on_axis_target_requested(self, axis_key: str, value: float) -> None:
        if self._client is None:
            return
        try:
            if axis_key == "lift":
                target_scale = float(value) / float(self.LIFT_MAX_HEIGHT_MM)
                self._client.lift.set_lift_height_sync(target_scale)
            else:
                self._client.waist.set_waist_pitch(float(value))
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"{axis_key} 设置失败: {exc}")

    # endregion
