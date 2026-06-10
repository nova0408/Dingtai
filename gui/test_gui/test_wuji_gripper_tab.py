from __future__ import annotations

from PySide6.QtCore import Signal
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

from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji.dahuan_gripper_client import DahuanGripperInfo


class WujiGripperTabWidget(QWidget):
    refreshRequested = Signal()
    enableRequested = Signal(bool)
    positionRequested = Signal(int)
    speedRequested = Signal(int)
    forceRequested = Signal(int)
    calibrateRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._latest_info: DahuanGripperInfo | None = None
        self._build_ui()
        self._set_info_unavailable()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)

        read_group = QGroupBox("寰宇夹爪读取状态", self)
        read_layout = QGridLayout(read_group)
        self.timestamp_label = QLabel("-", read_group)
        self.online_label = QLabel("-", read_group)
        self.calibrated_label = QLabel("-", read_group)
        self.enabled_label = QLabel("-", read_group)
        self.position_label = QLabel("-", read_group)
        self.ratio_label = QLabel("-", read_group)
        self.speed_label = QLabel("-", read_group)
        self.force_label = QLabel("-", read_group)
        self.state_label = QLabel("-", read_group)
        self.target_position_label = QLabel("-", read_group)

        read_layout.addWidget(QLabel("时间戳(ms)", read_group), 0, 0)
        read_layout.addWidget(self.timestamp_label, 0, 1)
        read_layout.addWidget(QLabel("在线", read_group), 1, 0)
        read_layout.addWidget(self.online_label, 1, 1)
        read_layout.addWidget(QLabel("已校准", read_group), 2, 0)
        read_layout.addWidget(self.calibrated_label, 2, 1)
        read_layout.addWidget(QLabel("已使能", read_group), 3, 0)
        read_layout.addWidget(self.enabled_label, 3, 1)
        read_layout.addWidget(QLabel("位置", read_group), 4, 0)
        read_layout.addWidget(self.position_label, 4, 1)
        read_layout.addWidget(QLabel("比例", read_group), 5, 0)
        read_layout.addWidget(self.ratio_label, 5, 1)
        read_layout.addWidget(QLabel("速度", read_group), 6, 0)
        read_layout.addWidget(self.speed_label, 6, 1)
        read_layout.addWidget(QLabel("力", read_group), 7, 0)
        read_layout.addWidget(self.force_label, 7, 1)
        read_layout.addWidget(QLabel("状态码", read_group), 8, 0)
        read_layout.addWidget(self.state_label, 8, 1)
        read_layout.addWidget(QLabel("目标位置", read_group), 9, 0)
        read_layout.addWidget(self.target_position_label, 9, 1)
        read_layout.setColumnStretch(1, 1)

        write_group = QGroupBox("寰宇夹爪设置控制", self)
        write_layout = QVBoxLayout(write_group)

        enable_group = QGroupBox("使能切换", write_group)
        enable_layout = QHBoxLayout(enable_group)
        self.enable_indicator = CasiaIndicatorLight(
            enable_group,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        enable_layout.addWidget(self.enable_indicator)
        enable_layout.addStretch(1)
        write_layout.addWidget(enable_group)

        position_group = QGroupBox("目标位置", write_group)
        position_layout = QHBoxLayout(position_group)
        self.open_button = QPushButton("开", position_group)
        self.close_button = QPushButton("关", position_group)
        self.open_button.setMinimumWidth(72)
        self.close_button.setMinimumWidth(72)
        position_layout.addWidget(self.open_button)
        position_layout.addWidget(self.close_button)
        write_layout.addWidget(position_group)

        speed_group = QGroupBox("速度设置", write_group)
        speed_layout = QHBoxLayout(speed_group)
        self.speed_spinbox = QSpinBox(speed_group)
        self.speed_spinbox.setRange(-1, 1000)
        self.speed_spinbox.setSpecialValueText("-")
        self.speed_apply_button = QPushButton("发送速度", speed_group)
        speed_layout.addWidget(self.speed_spinbox)
        speed_layout.addWidget(self.speed_apply_button)
        write_layout.addWidget(speed_group)

        force_group = QGroupBox("力设置", write_group)
        force_layout = QHBoxLayout(force_group)
        self.force_spinbox = QSpinBox(force_group)
        self.force_spinbox.setRange(-1, 1000)
        self.force_spinbox.setSpecialValueText("-")
        self.force_apply_button = QPushButton("发送力", force_group)
        force_layout.addWidget(self.force_spinbox)
        force_layout.addWidget(self.force_apply_button)
        write_layout.addWidget(force_group)

        action_group = QGroupBox("动作", write_group)
        action_layout = QHBoxLayout(action_group)
        self.calibrate_button = QPushButton("执行校准", action_group)
        self.refresh_button = QPushButton("刷新状态", action_group)
        action_layout.addWidget(self.calibrate_button)
        action_layout.addWidget(self.refresh_button)
        write_layout.addWidget(action_group)
        write_layout.addStretch(1)

        main_layout.addWidget(read_group, 1)
        main_layout.addWidget(write_group, 1)

        self.enable_indicator.clicked.connect(self._on_enable_indicator_clicked)
        self.open_button.clicked.connect(lambda _checked=False: self._emit_position_target(1000))
        self.close_button.clicked.connect(lambda _checked=False: self._emit_position_target(0))
        self.speed_apply_button.clicked.connect(lambda _checked=False: self._emit_speed_target())
        self.force_apply_button.clicked.connect(lambda _checked=False: self._emit_force_target())
        self.calibrate_button.clicked.connect(lambda _checked=False: self.calibrateRequested.emit())
        self.refresh_button.clicked.connect(lambda _checked=False: self.refreshRequested.emit())

    def update_info(self, info: DahuanGripperInfo) -> None:
        self._latest_info = info
        self.timestamp_label.setText(str(info.timestamp_ms))
        self.online_label.setText("是" if info.online else "否")
        self.calibrated_label.setText("是" if info.calibrated else "否")
        self.enabled_label.setText("未知" if info.enabled is None else ("是" if info.enabled else "否"))
        self.position_label.setText(str(info.position))
        self.ratio_label.setText(f"{float(info.position) / 1000.0:.3f}")
        self.speed_label.setText(str(info.speed))
        self.force_label.setText(str(info.force))
        self.state_label.setText(str(info.grip_state))
        self.target_position_label.setText(str(info.position))
        self.enable_indicator.setEnabled(bool(info.online))
        self.enable_indicator.set_status(bool(info.enabled) if info.enabled is not None else False)
        self._set_write_controls_enabled(bool(info.online))

    def _set_info_unavailable(self) -> None:
        self.timestamp_label.setText("-")
        self.online_label.setText("-")
        self.calibrated_label.setText("-")
        self.enabled_label.setText("-")
        self.position_label.setText("-")
        self.ratio_label.setText("-")
        self.speed_label.setText("-")
        self.force_label.setText("-")
        self.state_label.setText("-")
        self.target_position_label.setText("-")
        self.speed_spinbox.setValue(-1)
        self.force_spinbox.setValue(-1)
        self.enable_indicator.setEnabled(False)
        self._set_write_controls_enabled(False)

    def _set_write_controls_enabled(self, enabled: bool) -> None:
        self.open_button.setEnabled(enabled)
        self.close_button.setEnabled(enabled)
        self.speed_spinbox.setEnabled(enabled)
        self.speed_apply_button.setEnabled(enabled)
        self.force_spinbox.setEnabled(enabled)
        self.force_apply_button.setEnabled(enabled)
        self.calibrate_button.setEnabled(enabled)

    def _on_enable_indicator_clicked(self) -> None:
        if self._latest_info is None or not self._latest_info.online:
            return
        if self._latest_info.enabled is None:
            self.enableRequested.emit(True)
            return
        self.enableRequested.emit(not bool(self._latest_info.enabled))

    def _emit_position_target(self, position: int) -> None:
        if position < 0:
            return
        self.target_position_label.setText(str(int(position)))
        self.positionRequested.emit(int(position))

    def _emit_speed_target(self) -> None:
        if int(self.speed_spinbox.value()) < 0:
            return
        self.speedRequested.emit(int(self.speed_spinbox.value()))

    def _emit_force_target(self) -> None:
        if int(self.force_spinbox.value()) < 0:
            return
        self.forceRequested.emit(int(self.force_spinbox.value()))
