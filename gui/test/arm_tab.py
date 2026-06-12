from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.test.common import ActivatableTab, HoldRepeatController, gui_pose_to_matrix_m
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.arm.wuji_arm_protocol import ArmDeviceName, WUJI_ARM_JOINT_LIMITS_DEG
from src.wuji.arm_client import WujiArmClient


@dataclass(frozen=True, slots=True)
class ArmJointWidgets:
    current_label: QLabel
    minus_button: QPushButton
    plus_button: QPushButton
    spin_box: QDoubleSpinBox
    set_button: QPushButton
    repeat_controller: HoldRepeatController


class ArmTabWidget(QWidget, ActivatableTab):
    HOLD_STEP_DEG = 2.0
    STALE_THRESHOLD_MS = 2000

    # region 初始化

    def __init__(self, device_name: ArmDeviceName, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.device_name: ArmDeviceName = device_name
        self.title = title
        self._client: WujiArmClient | None = None
        self._active = False
        self._refresh_timer = QTimer(self)
        self._joint_widgets: list[ArmJointWidgets] = []
        self._joint_current_values: list[float | None] = [None] * 6
        self._pose_current_labels: dict[str, QLabel] = {}
        self._pose_input_boxes: dict[str, QDoubleSpinBox] = {}
        self._last_joint_state_received_monotonic_s: float | None = None
        self._last_joint_state_timestamp_ms: int | None = None
        self._last_joint_state_angles_deg: tuple[float, ...] | None = None

        self.enable_indicator: CasiaIndicatorLight
        self.info_label: QLabel
        self.apply_button: QPushButton

        self._setup_timer()
        self._setup_ui()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_timer(self) -> None:
        self._refresh_timer.setInterval(100)
        self._refresh_timer.timeout.connect(self._refresh_state)

    def _setup_ui(self) -> None:
        self.enable_indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        self.info_label = QLabel(f"{self.title} 未连接", self)

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(self._build_header_row())
        root_layout.addWidget(self._build_joint_group())
        root_layout.addWidget(self._build_pose_group())
        root_layout.addStretch(1)

    def _build_header_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self.enable_indicator)
        layout.addWidget(self.info_label, 1)
        return layout

    def _build_joint_group(self) -> QGroupBox:
        group = QGroupBox("Joint Control", self)
        layout = QGridLayout(group)
        layout.addWidget(QLabel("axis", group), 0, 0)
        layout.addWidget(QLabel("pos", group), 0, 1)
        for joint_index, limit in enumerate(WUJI_ARM_JOINT_LIMITS_DEG[self.device_name], start=1):
            widgets = self._build_joint_widgets(group, joint_index, limit.minimum_deg, limit.maximum_deg)
            self._joint_widgets.append(widgets)
            layout.addWidget(QLabel(f"J{joint_index}", group), joint_index, 0)
            layout.addWidget(widgets.current_label, joint_index, 1)
            layout.addWidget(widgets.minus_button, joint_index, 2)
            layout.addWidget(widgets.plus_button, joint_index, 3)
            layout.addWidget(widgets.spin_box, joint_index, 4)
            layout.addWidget(widgets.set_button, joint_index, 5)
        return group

    def _build_joint_widgets(
        self,
        parent: QWidget,
        joint_index: int,
        minimum_deg: float,
        maximum_deg: float,
    ) -> ArmJointWidgets:
        current_label = QLabel("-", parent)
        minus_button = QPushButton("-", parent)
        plus_button = QPushButton("+", parent)
        spin_box = QDoubleSpinBox(parent)
        spin_box.setDecimals(1)
        spin_box.setRange(minimum_deg, maximum_deg)
        spin_box.setSingleStep(self.HOLD_STEP_DEG)
        set_button = QPushButton("set", parent)
        repeat_controller = HoldRepeatController(self)
        set_button.clicked.connect(lambda _checked=False, index=joint_index - 1: self._set_joint_from_spin(index))
        minus_button.pressed.connect(
            lambda index=joint_index - 1, ctrl=repeat_controller: ctrl.start(lambda: self._nudge_joint(index, -1.0))
        )
        plus_button.pressed.connect(
            lambda index=joint_index - 1, ctrl=repeat_controller: ctrl.start(lambda: self._nudge_joint(index, 1.0))
        )
        minus_button.released.connect(repeat_controller.stop)
        plus_button.released.connect(repeat_controller.stop)
        return ArmJointWidgets(
            current_label=current_label,
            minus_button=minus_button,
            plus_button=plus_button,
            spin_box=spin_box,
            set_button=set_button,
            repeat_controller=repeat_controller,
        )

    def _build_pose_group(self) -> QGroupBox:
        group = QGroupBox("Pose", self)
        layout = QHBoxLayout(group)
        layout.addWidget(self._build_current_pose_group(), 1)
        layout.addWidget(self._build_input_pose_group(), 1)
        return group

    def _build_current_pose_group(self) -> QGroupBox:
        group = QGroupBox("Current", self)
        layout = QGridLayout(group)
        for row, field_name in enumerate(("x", "y", "z", "r", "p", "yaw")):
            layout.addWidget(QLabel(field_name, group), row, 0)
            value_label = QLabel("-", group)
            self._pose_current_labels[field_name] = value_label
            layout.addWidget(value_label, row, 1)
        return group

    def _build_input_pose_group(self) -> QGroupBox:
        group = QGroupBox("IK Input", self)
        layout = QGridLayout(group)
        units = {"x": "mm", "y": "mm", "z": "mm", "r": "deg", "p": "deg", "yaw": "deg"}
        for row, field_name in enumerate(("x", "y", "z", "r", "p", "yaw")):
            layout.addWidget(QLabel(field_name, group), row, 0)
            spin_box = QDoubleSpinBox(group)
            spin_box.setDecimals(1)
            spin_box.setRange(-9999.0, 9999.0)
            spin_box.setSingleStep(1.0)
            self._pose_input_boxes[field_name] = spin_box
            layout.addWidget(spin_box, row, 1)
            layout.addWidget(QLabel(units[field_name], group), row, 2)
        self.apply_button = QPushButton("apply", group)
        layout.addWidget(self.apply_button, 6, 0, 1, 3)
        return group

    def _connect_signals(self) -> None:
        self.enable_indicator.clicked.connect(self._on_enable_clicked)
        self.apply_button.clicked.connect(self._apply_pose_target)

    # endregion

    # region 生命周期

    def set_client(self, client: WujiArmClient | None) -> None:
        self._client = client
        self.set_connection_ready(client is not None)
        if client is None:
            self._refresh_timer.stop()
            self._last_joint_state_received_monotonic_s = None
            self._last_joint_state_timestamp_ms = None
            self._last_joint_state_angles_deg = None

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._refresh_timer.stop()
            return
        if self._client is None:
            self.info_label.setText(f"{self.title} 未连接")
            return
        self._try_enable()
        self._refresh_timer.start()
        self._refresh_state()

    def set_connection_ready(self, ready: bool) -> None:
        enabled = bool(ready)
        self.enable_indicator.setEnabled(enabled)
        for widgets in self._joint_widgets:
            widgets.minus_button.setEnabled(enabled)
            widgets.plus_button.setEnabled(enabled)
            widgets.spin_box.setEnabled(enabled)
            widgets.set_button.setEnabled(enabled)
        for spin_box in self._pose_input_boxes.values():
            spin_box.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        if not enabled:
            self.enable_indicator.set_status(False)
            self._show_stale_state("未连接")

    # endregion

    # region 使能

    def _try_enable(self) -> None:
        if self._client is None:
            return
        try:
            if not self._client.get_enable():
                self._client.set_enable(True)
            self.enable_indicator.set_status(bool(self._client.get_enable()))
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"{self.title} 使能失败: {exc}")

    @Slot()
    def _on_enable_clicked(self) -> None:
        if self._client is None:
            return
        try:
            current_enabled = bool(self._client.get_enable())
            self._client.set_enable(not current_enabled)
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"{self.title} 切换使能失败: {exc}")

    # endregion

    # region 刷新

    def _refresh_state(self) -> None:
        if self._client is None:
            return
        try:
            enabled = bool(self._client.get_enable())
            joint_states = tuple(self._client.get_joint_states(self.device_name))
            timestamp_ms = self._read_joint_timestamp_ms()
            self.enable_indicator.set_status(enabled)
            if len(joint_states) != 6:
                self._handle_missing_joint_state(joint_count=len(joint_states))
                return

            current_angles_deg = [float(joint.angle_deg) for joint in joint_states]
            if self._is_new_joint_state(tuple(current_angles_deg), timestamp_ms):
                self._last_joint_state_received_monotonic_s = time.monotonic()

            self._joint_current_values = list(current_angles_deg)
            for index, angle_deg in enumerate(current_angles_deg):
                self._joint_widgets[index].current_label.setText(f"{angle_deg:.1f} deg")

            pose_values = tuple(float(value) for value in self._client.current_fk_xyzrpy(self.device_name))
            self._update_pose_labels(pose_values)
            self.info_label.setText(f"{self.title} enable={enabled} ts={timestamp_ms}")
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"{self.title} 刷新失败: {exc}")

    def _show_stale_state(self, reason: str) -> None:
        self._joint_current_values = [None] * 6
        for widgets in self._joint_widgets:
            widgets.current_label.setText("-")
        for label in self._pose_current_labels.values():
            label.setText("-")
        self.info_label.setText(f"{self.title} 状态不可靠: {reason}")

    def _handle_missing_joint_state(self, joint_count: int) -> None:
        if self._last_joint_state_received_monotonic_s is None:
            self._show_stale_state(f"尚未收到有效状态 joint_count={joint_count}")
            return
        elapsed_ms = int((time.monotonic() - self._last_joint_state_received_monotonic_s) * 1000.0)
        if elapsed_ms > self.STALE_THRESHOLD_MS:
            self._show_stale_state(f"状态超时 {elapsed_ms} ms")
            return
        self.info_label.setText(f"{self.title} 等待新状态 joint_count={joint_count} last_ok={elapsed_ms} ms")

    def _update_pose_labels(self, pose_values: tuple[float, ...]) -> None:
        x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg = pose_values
        display_map = {
            "x": x_m * 1000.0,
            "y": y_m * 1000.0,
            "z": z_m * 1000.0,
            "r": roll_deg,
            "p": pitch_deg,
            "yaw": yaw_deg,
        }
        for key, value in display_map.items():
            unit = "mm" if key in {"x", "y", "z"} else "deg"
            self._pose_current_labels[key].setText(f"{value:.1f} {unit}")

    # endregion

    # region 控制

    def _set_joint_from_spin(self, joint_index: int) -> None:
        widgets = self._joint_widgets[joint_index]
        self._send_joint_target(joint_index, float(widgets.spin_box.value()))

    def _nudge_joint(self, joint_index: int, direction: float) -> None:
        widgets = self._joint_widgets[joint_index]
        current_value = self._joint_current_values[joint_index]
        if current_value is None:
            current_value = float(widgets.spin_box.value())
        target_value = current_value + direction * self.HOLD_STEP_DEG
        target_value = min(max(target_value, widgets.spin_box.minimum()), widgets.spin_box.maximum())
        self._send_joint_target(joint_index, float(target_value))

    def _send_joint_target(self, joint_index: int, target_angle_deg: float) -> None:
        if self._client is None:
            return
        try:
            self._client.set_joint(self.device_name, joint_index + 1, float(target_angle_deg))
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"J{joint_index + 1} 设置失败: {exc}")

    @Slot()
    def _apply_pose_target(self) -> None:
        if self._client is None:
            return
        try:
            reference_angles_deg = self._require_current_joint_values()
            pose_values = tuple(
                float(self._pose_input_boxes[name].value()) for name in ("x", "y", "z", "r", "p", "yaw")
            )
            pose_matrix = gui_pose_to_matrix_m(pose_values)
            ik_solution_rad = self._client.ik(
                pose_matrix,
                [float(np.deg2rad(value)) for value in reference_angles_deg],
            )
            raw_target_deg = [float(np.rad2deg(value)) for value in ik_solution_rad]
            aligned_target_deg = self._align_ik_solution(raw_target_deg, reference_angles_deg)
            self._client.set_joints(aligned_target_deg)
            self.info_label.setText(
                f"{self.title} IK 已发送: "
                + ", ".join(f"J{index + 1}={value:.1f}" for index, value in enumerate(aligned_target_deg))
            )
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"{self.title} IK 失败: {exc}")

    def _require_current_joint_values(self) -> list[float]:
        if any(value is None for value in self._joint_current_values):
            raise RuntimeError("当前 joint state 不可用，拒绝执行 IK")
        return [float(value) for value in self._joint_current_values if value is not None]

    def _align_ik_solution(self, raw_target_deg: list[float], current_angles_deg: list[float]) -> list[float]:
        aligned: list[float] = []
        limits = WUJI_ARM_JOINT_LIMITS_DEG[self.device_name]
        for target_deg, current_deg, limit in zip(raw_target_deg, current_angles_deg, limits, strict=True):
            candidates: list[float] = []
            for turns in range(-3, 4):
                candidate = float(target_deg) + 360.0 * float(turns)
                if limit.minimum_deg <= candidate <= limit.maximum_deg:
                    candidates.append(candidate)
            if not candidates:
                aligned.append(min(max(float(target_deg), limit.minimum_deg), limit.maximum_deg))
                continue
            aligned.append(min(candidates, key=lambda candidate: abs(candidate - current_deg)))
        return aligned

    # endregion

    # region 工具

    def _read_joint_timestamp_ms(self) -> int:
        if self._client is None:
            return 0
        try:
            timestamp_ms = int(self._client.thread_joint_states.joint_states_timestamp_ms)
            return timestamp_ms
        except Exception:
            return 0

    def _is_new_joint_state(self, joint_angles_deg: tuple[float, ...], timestamp_ms: int) -> bool:
        if self._last_joint_state_received_monotonic_s is None:
            self._last_joint_state_timestamp_ms = timestamp_ms if timestamp_ms > 0 else None
            self._last_joint_state_angles_deg = joint_angles_deg
            return True

        if timestamp_ms > 0 and timestamp_ms != self._last_joint_state_timestamp_ms:
            self._last_joint_state_timestamp_ms = timestamp_ms
            self._last_joint_state_angles_deg = joint_angles_deg
            return True

        if self._last_joint_state_angles_deg != joint_angles_deg:
            self._last_joint_state_angles_deg = joint_angles_deg
            if timestamp_ms > 0:
                self._last_joint_state_timestamp_ms = timestamp_ms
            return True

        return False

    # endregion
