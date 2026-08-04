from __future__ import annotations

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.test.common import (
    ActivatableTab,
    AxisControlConfig,
    AxisControlRow,
    BackgroundCall,
    HoldRepeatController,
)
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.arm.wuji_arm_protocol import WUJI_HEAD_AXIS_LIMITS
from gui.test.robot_control_clients import RobotControlHeadClient


class HeadTabWidget(QWidget, ActivatableTab):
    """头部 Yaw/Pitch 双电机状态与控制页。"""

    HOLD_STEP_DEG = 2.0
    REFRESH_INTERVAL_MS = 250

    # region 初始化

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client: RobotControlHeadClient | None = None
        self._active = False
        self._refresh_busy = False
        self._action_busy = False
        self._refresh_timer = QTimer(self)
        self._refresh_call = BackgroundCall(self)
        self._action_call = BackgroundCall(self)
        self._yaw_repeat = HoldRepeatController(self)
        self._pitch_repeat = HoldRepeatController(self)
        self._setup_ui()
        self._setup_timer()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_ui(self) -> None:
        self.info_label = QLabel("头部未连接", self)
        self.yaw_enable_indicator = self._build_enable_indicator("Yaw 电机")
        self.pitch_enable_indicator = self._build_enable_indicator("Pitch 电机")
        self.yaw_row = self._build_yaw_row()
        self.pitch_row = self._build_pitch_row()

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.info_label)
        root_layout.addWidget(
            self._build_axis_group(
                "Yaw",
                self.yaw_enable_indicator,
                self.yaw_row,
            )
        )
        root_layout.addWidget(
            self._build_axis_group(
                "Pitch",
                self.pitch_enable_indicator,
                self.pitch_row,
            )
        )
        root_layout.addStretch(1)

    def _setup_timer(self) -> None:
        self._refresh_timer.setInterval(self.REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._request_refresh)

    def _connect_signals(self) -> None:
        self.yaw_enable_indicator.clicked.connect(self._on_yaw_enable_clicked)
        self.pitch_enable_indicator.clicked.connect(
            self._on_pitch_enable_clicked
        )
        self.yaw_row.setRequested.connect(self._on_axis_target_requested)
        self.yaw_row.nudgeRequested.connect(self._on_axis_target_requested)
        self.pitch_row.setRequested.connect(self._on_axis_target_requested)
        self.pitch_row.nudgeRequested.connect(self._on_axis_target_requested)
        self._refresh_call.succeeded.connect(self._on_refresh_succeeded)
        self._refresh_call.failed.connect(self._on_refresh_failed)
        self._refresh_call.finished.connect(self._on_refresh_finished)
        self._action_call.succeeded.connect(self._on_action_succeeded)
        self._action_call.failed.connect(self._on_action_failed)
        self._action_call.finished.connect(self._on_action_finished)

    def _build_enable_indicator(self, motor_name: str) -> CasiaIndicatorLight:
        indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        indicator.setToolTip(
            f"{motor_name}使能。当前 qmlinker 协议使用头部模块共同使能接口。"
        )
        return indicator

    def _build_yaw_row(self) -> AxisControlRow:
        return AxisControlRow(
            AxisControlConfig(
                axis_key="yaw",
                title="yaw",
                minimum=WUJI_HEAD_AXIS_LIMITS["head_yaw"].minimum,
                maximum=WUJI_HEAD_AXIS_LIMITS["head_yaw"].maximum,
                step=self.HOLD_STEP_DEG,
                unit="deg",
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
                unit="deg",
            ),
            repeat_controller=self._pitch_repeat,
            parent=self,
        )

    def _build_axis_group(
        self,
        title: str,
        enable_indicator: CasiaIndicatorLight,
        row: AxisControlRow,
    ) -> QGroupBox:
        group = QGroupBox(title, self)
        layout = QHBoxLayout(group)
        layout.addWidget(enable_indicator)
        layout.addWidget(row, 1)
        return group

    # endregion

    # region 生命周期

    def set_client(self, client: RobotControlHeadClient | None) -> None:
        self._client = client
        self._refresh_busy = False
        self._action_busy = False
        self.set_connection_ready(client is not None)
        if client is None:
            self._refresh_timer.stop()

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._refresh_timer.stop()
            return
        if self._client is None:
            self.info_label.setText("头部未连接")
            return
        self._refresh_timer.start()
        self._request_refresh()

    def set_connection_ready(self, ready: bool) -> None:
        self.yaw_enable_indicator.setEnabled(ready)
        self.pitch_enable_indicator.setEnabled(ready)
        self.yaw_row.set_row_enabled(ready)
        self.pitch_row.set_row_enabled(ready)
        if ready:
            return
        self.yaw_enable_indicator.set_status(False)
        self.pitch_enable_indicator.set_status(False)
        self.yaw_row.set_current_value(None)
        self.pitch_row.set_current_value(None)
        self.info_label.setText("头部未连接")

    # endregion

    # region 刷新

    def _request_refresh(self) -> None:
        if (
            self._client is None
            or self._refresh_busy
            or self._action_busy
        ):
            return
        self._refresh_busy = True
        self._refresh_call.start(self._read_state)

    def _read_state(self) -> tuple[bool, float, float]:
        client = self._client
        if client is None:
            raise RuntimeError("头部未连接")
        return (
            client.get_enable(),
            client.get_head_yaw(),
            client.get_head_pitch(),
        )

    @Slot(object)
    def _on_refresh_succeeded(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 3:
            return
        enabled, yaw_value, pitch_value = payload
        if (
            not isinstance(enabled, bool)
            or not isinstance(yaw_value, float)
            or not isinstance(pitch_value, float)
        ):
            return
        self.yaw_enable_indicator.set_status(enabled)
        self.pitch_enable_indicator.set_status(enabled)
        self.yaw_row.set_current_value(yaw_value, suffix="deg")
        self.pitch_row.set_current_value(pitch_value, suffix="deg")
        self.info_label.setText(
            f"头部状态已更新 | yaw={yaw_value:.1f}° pitch={pitch_value:.1f}°"
        )

    @Slot(str)
    def _on_refresh_failed(self, message: str) -> None:
        self.info_label.setText(f"头部刷新失败：{message}")

    @Slot()
    def _on_refresh_finished(self) -> None:
        self._refresh_busy = False

    # endregion

    # region 使能与控制

    @Slot()
    def _on_yaw_enable_clicked(self) -> None:
        self._request_enable_toggle("Yaw")

    @Slot()
    def _on_pitch_enable_clicked(self) -> None:
        self._request_enable_toggle("Pitch")

    def _request_enable_toggle(self, motor_name: str) -> None:
        if self._client is None or self._action_busy:
            return
        self._action_busy = True
        self.info_label.setText(f"{motor_name} 电机使能切换中…")
        self._action_call.start(
            lambda: self._toggle_module_enable(motor_name)
        )

    def _toggle_module_enable(self, motor_name: str) -> str:
        client = self._client
        if client is None:
            raise RuntimeError("头部未连接")
        target_enabled = not client.get_enable()
        client.set_enable(target_enabled)
        actual_enabled = client.get_enable()
        if actual_enabled != target_enabled:
            raise RuntimeError(
                f"服务端未进入目标使能状态：target={target_enabled}, "
                f"actual={actual_enabled}"
            )
        state_text = "使能" if actual_enabled else "禁用"
        return f"{motor_name} 电机已{state_text}"

    @Slot(str, float)
    def _on_axis_target_requested(self, axis_key: str, value: float) -> None:
        if self._client is None or self._action_busy:
            return
        self._action_busy = True
        self.info_label.setText(f"{axis_key} 目标下发中…")
        self._action_call.start(
            lambda: self._set_axis_target(axis_key, value)
        )

    def _set_axis_target(self, axis_key: str, value: float) -> str:
        client = self._client
        if client is None:
            raise RuntimeError("头部未连接")
        if axis_key == "yaw":
            client.set_head_yaw(value)
        elif axis_key == "pitch":
            client.set_head_pitch(value)
        else:
            raise ValueError(f"未知头部轴：{axis_key}")
        return f"{axis_key} 目标 {value:.1f}° 已确认"

    @Slot(object)
    def _on_action_succeeded(self, payload: object) -> None:
        self.info_label.setText(str(payload))
        QTimer.singleShot(100, self._request_refresh)

    @Slot(str)
    def _on_action_failed(self, message: str) -> None:
        self.info_label.setText(f"头部控制失败：{message}")
        QTimer.singleShot(100, self._request_refresh)

    @Slot()
    def _on_action_finished(self) -> None:
        self._action_busy = False

    # endregion
