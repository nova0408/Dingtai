from __future__ import annotations

from collections.abc import Callable
from queue import Queue
from threading import Thread

from loguru import logger
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.test.common import ActivatableTab, BackgroundCall
from gui.util_components.casia_indicator_light import (
    CasiaMultiStateIndicator,
    IndicatorState,
)
from gui.test.robot_control_clients import Ar5Snapshot, Ar5Side, RobotControlAr5Client


class _SerialActionWorker(QObject):
    """按用户触摸顺序串行执行 AR5 控制命令。"""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._queue: Queue[tuple[Callable[[], object], str]] = Queue()
        self._thread = Thread(target=self._run, name="ar5-gui-actions", daemon=True)
        self._thread.start()

    def submit(self, callback: Callable[[], object], success_message: str) -> None:
        self._queue.put((callback, success_message))

    def _run(self) -> None:
        while True:
            callback, success_message = self._queue.get()
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))
            else:
                self.succeeded.emit(success_message)
            finally:
                self._queue.task_done()


class ArmTabWidget(QWidget, ActivatableTab):
    """面向触摸屏的 AR5 状态与控制页。"""

    REFRESH_INTERVAL_MS = 250
    TOUCH_HEIGHT = 48
    WORK_MODE_STATES = (
        IndicatorState("automatic", "自动", "#1976d2"),
        IndicatorState("manual", "手动", "#00897b"),
        IndicatorState("unknown", "未知", "#607d8b"),
    )
    POWER_STATES = (
        IndicatorState("on", "上电", "#2e7d32"),
        IndicatorState("off", "下电", "#616161"),
        IndicatorState("estop", "急停", "#c62828"),
        IndicatorState("gstop", "安全门", "#ef6c00"),
        IndicatorState("safety_stop", "安全停止", "#ad1457"),
        IndicatorState("unknown", "未知", "#607d8b"),
    )
    OPERATION_STATES = (
        IndicatorState("idle", "空闲", "#546e7a"),
        IndicatorState("running", "运行", "#2e7d32"),
        IndicatorState("demo", "Demo", "#5e35b1"),
        IndicatorState("identify", "辨识", "#6d4c41"),
        IndicatorState("collaboration", "协作", "#00838f"),
        IndicatorState("error", "错误", "#c62828"),
        IndicatorState("debug", "调试", "#455a64"),
        IndicatorState("unknown", "未知", "#607d8b"),
    )
    # region 初始化

    def __init__(self, side: Ar5Side, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.side = side
        self.title = title
        self._client: RobotControlAr5Client | None = None
        self._snapshot: Ar5Snapshot | None = None
        self._active = False
        self._refresh_busy = False
        self._control_widgets: list[QWidget] = []
        self._joint_current_labels: list[QLabel] = []
        self._joint_target_boxes: list[QDoubleSpinBox] = []
        self._pose_current_labels: dict[str, QLabel] = {}
        self._pose_target_boxes: dict[str, QDoubleSpinBox] = {}
        self._state_labels: dict[str, QLabel] = {}
        self._refresh_timer = QTimer(self)
        self._refresh_call = BackgroundCall(self)
        self._action_worker = _SerialActionWorker(self)
        self._setup_ui()
        self._setup_timers()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        root_layout.addLayout(self._build_header())
        root_layout.addLayout(self._build_live_telemetry())

        self.control_tabs = QTabWidget(self)
        self.control_tabs.setDocumentMode(True)
        self.control_tabs.addTab(self._wrap_scroll(self._build_status_page()), "状态")
        self.control_tabs.addTab(self._wrap_scroll(self._build_move_page()), "Move")
        root_layout.addWidget(self.control_tabs, 1)

        self.setStyleSheet(
            """
            QGroupBox { font-size: 16px; font-weight: 600; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel, QComboBox, QDoubleSpinBox { font-size: 16px; }
            QLabel[themeRole="telemetry-value"] {
                background: #eef3f6;
                border: 1px solid #c3cdd3;
                border-radius: 6px;
                padding: 3px 2px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton { min-height: 48px; min-width: 72px; font-size: 17px; }
            QTabBar::tab { min-height: 48px; min-width: 120px; font-size: 17px; }
            """
        )

    def _build_header(self) -> QVBoxLayout:
        layout = QVBoxLayout()
        connection_row = QHBoxLayout()
        self.info_label = QLabel(f"{self.title} 未连接", self)
        self.stop_button = QPushButton("停止运动", self)
        self.stop_button.setStyleSheet(
            "QPushButton { background: #b3261e; color: white; font-weight: 700; }"
        )
        connection_row.addWidget(self.info_label, 1)
        connection_row.addWidget(self.stop_button)
        layout.addLayout(connection_row)

        state_row = QHBoxLayout()
        self.mode_indicator = CasiaMultiStateIndicator(
            "工作模式",
            self.WORK_MODE_STATES,
            self,
        )
        self.power_indicator = CasiaMultiStateIndicator(
            "使能状态",
            self.POWER_STATES,
            self,
        )
        self.operation_indicator = CasiaMultiStateIndicator(
            "机器人状态",
            self.OPERATION_STATES,
            self,
            interactive=False,
        )
        state_row.addWidget(self.mode_indicator)
        state_row.addWidget(self.power_indicator)
        state_row.addWidget(self.operation_indicator)
        state_row.addStretch(1)
        layout.addLayout(state_row)
        self._control_widgets.extend(
            (
                self.mode_indicator,
                self.power_indicator,
                self.stop_button,
            )
        )
        return layout

    def _build_live_telemetry(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)

        joint_group = QGroupBox("当前关节角（deg）", self)
        joint_layout = QHBoxLayout(joint_group)
        joint_layout.setContentsMargins(8, 12, 8, 6)
        joint_layout.setSpacing(4)
        for index in range(7):
            label = QLabel(f"J{index + 1}\n-", joint_group)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumWidth(66)
            label.setProperty("themeRole", "telemetry-value")
            self._joint_current_labels.append(label)
            joint_layout.addWidget(label, 1)
        layout.addWidget(joint_group, 7)

        pose_group = QGroupBox("当前 TCP / Elbow（mm / deg）", self)
        pose_layout = QHBoxLayout(pose_group)
        pose_layout.setContentsMargins(8, 12, 8, 6)
        pose_layout.setSpacing(4)
        for key in ("x", "y", "z", "rx", "ry", "rz", "elbow"):
            label = QLabel(f"{key.upper()}\n-", pose_group)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumWidth(62)
            label.setProperty("themeRole", "telemetry-value")
            self._pose_current_labels[key] = label
            pose_layout.addWidget(label, 1)
        layout.addWidget(pose_group, 7)
        return layout

    def _build_status_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        state_group = QGroupBox("设备信息", page)
        state_layout = QGridLayout(state_group)
        for row, (key, title) in enumerate(
            (
                ("type", "型号"),
                ("uid", "UID"),
            )
        ):
            state_layout.addWidget(QLabel(title, state_group), row, 0)
            label = QLabel("-", state_group)
            label.setTextInteractionFlags(label.textInteractionFlags())
            self._state_labels[key] = label
            state_layout.addWidget(label, row, 1)
        layout.addWidget(state_group)
        hint = QLabel(
            "工作模式、使能状态和机器人状态始终显示在页面顶部；"
            "点击工作模式或使能状态指示灯可执行允许的状态切换。",
            page,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_move_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        settings_group = QGroupBox("运动参数", page)
        settings_layout = QHBoxLayout(settings_group)
        self.speed_box = self._new_spin_box(1.0, 3000.0, 1000.0, 50.0, 0)
        self.zone_box = self._new_spin_box(0.0, 100.0, 10.0, 1.0, 1)
        settings_layout.addWidget(QLabel("末端线速度 mm/s", settings_group))
        settings_layout.addWidget(self.speed_box)
        settings_layout.addWidget(QLabel("转弯区 mm", settings_group))
        settings_layout.addWidget(self.zone_box)
        settings_layout.addStretch(1)
        layout.addWidget(settings_group)

        joint_group = QGroupBox("MoveAbsJ（deg）", page)
        joint_layout = QGridLayout(joint_group)
        for index in range(7):
            joint_layout.addWidget(QLabel(f"J{index + 1}", joint_group), index, 0)
            spin_box = self._new_spin_box(-540.0, 540.0, 0.0, 1.0, 2)
            self._joint_target_boxes.append(spin_box)
            joint_layout.addWidget(spin_box, index, 1)
            copy_button = QPushButton("取当前", joint_group)
            copy_button.clicked.connect(
                lambda _checked=False, joint_index=index: self._copy_current_joint(joint_index)
            )
            joint_layout.addWidget(copy_button, index, 2)
            self._control_widgets.append(copy_button)
        self.move_joints_button = QPushButton("执行全部关节目标", joint_group)
        joint_layout.addWidget(self.move_joints_button, 7, 0, 1, 3)
        layout.addWidget(joint_group)

        pose_group = QGroupBox("MoveL（TCP 使用 mm/deg，Elbow 使用 deg）", page)
        pose_layout = QGridLayout(pose_group)
        for row, (key, title, minimum, maximum) in enumerate(
            (
                ("x", "X", -2000.0, 2000.0),
                ("y", "Y", -2000.0, 2000.0),
                ("z", "Z", -2000.0, 2000.0),
                ("rx", "Rx", -360.0, 360.0),
                ("ry", "Ry", -360.0, 360.0),
                ("rz", "Rz", -360.0, 360.0),
                ("elbow", "Elbow", -360.0, 360.0),
            )
        ):
            pose_layout.addWidget(QLabel(title, pose_group), row, 0)
            spin_box = self._new_spin_box(minimum, maximum, 0.0, 1.0, 2)
            self._pose_target_boxes[key] = spin_box
            pose_layout.addWidget(spin_box, row, 1)
        self.copy_pose_button = QPushButton("复制当前 TCP 与 Elbow", pose_group)
        self.move_pose_button = QPushButton("执行 MoveL", pose_group)
        self.move_elbow_button = QPushButton("仅调整 Elbow（保持 TCP）", pose_group)
        pose_layout.addWidget(self.copy_pose_button, 0, 2, 2, 1)
        pose_layout.addWidget(self.move_pose_button, 2, 2, 2, 1)
        pose_layout.addWidget(self.move_elbow_button, 4, 2, 3, 1)
        layout.addWidget(pose_group)
        layout.addStretch(1)
        self._control_widgets.extend(
            (
                self.speed_box,
                self.zone_box,
                *self._joint_target_boxes,
                *self._pose_target_boxes.values(),
                self.move_joints_button,
                self.copy_pose_button,
                self.move_pose_button,
                self.move_elbow_button,
            )
        )
        return page

    @staticmethod
    def _wrap_scroll(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _new_spin_box(
        self,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        spin_box = QDoubleSpinBox(self)
        spin_box.setRange(minimum, maximum)
        spin_box.setValue(value)
        spin_box.setSingleStep(step)
        spin_box.setDecimals(decimals)
        spin_box.setMinimumHeight(self.TOUCH_HEIGHT)
        spin_box.setKeyboardTracking(False)
        return spin_box

    def _setup_timers(self) -> None:
        self._refresh_timer.setInterval(self.REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._request_refresh)

    def _connect_signals(self) -> None:
        self.mode_indicator.clicked.connect(self._toggle_operate_mode)
        self.power_indicator.clicked.connect(self._toggle_power)
        self.stop_button.clicked.connect(self._stop_motion)
        self.move_joints_button.clicked.connect(self._move_joints)
        self.copy_pose_button.clicked.connect(self._copy_current_pose)
        self.move_pose_button.clicked.connect(self._move_pose)
        self.move_elbow_button.clicked.connect(self._move_elbow)
        self._refresh_call.succeeded.connect(self._on_refresh_succeeded)
        self._refresh_call.failed.connect(self._on_refresh_failed)
        self._refresh_call.finished.connect(self._on_refresh_finished)
        self._action_worker.succeeded.connect(self._on_action_succeeded)
        self._action_worker.failed.connect(self._on_action_failed)

    # endregion

    # region 生命周期

    def set_client(self, client: RobotControlAr5Client | None) -> None:
        self._client = client
        self._snapshot = None
        self.set_connection_ready(client is not None)
        if client is None:
            self._refresh_timer.stop()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._refresh_timer.stop()
            return
        if self._client is None:
            self.info_label.setText(f"{self.title} 未连接")
            return
        self._refresh_timer.start()
        self._request_refresh()

    def set_connection_ready(self, ready: bool) -> None:
        for widget in self._control_widgets:
            widget.setEnabled(ready)
        if ready:
            return
        self.mode_indicator.set_state("unknown")
        self.power_indicator.set_state("unknown")
        self.operation_indicator.set_state("unknown")
        self.info_label.setText(f"{self.title} 未连接")
        self._clear_display()

    # endregion

    # region 刷新

    def _request_refresh(self) -> None:
        if self._client is None or self._refresh_busy:
            return
        self._refresh_busy = True
        self._refresh_call.start(self._client.read_snapshot)

    @Slot(object)
    def _on_refresh_succeeded(self, payload: object) -> None:
        if not isinstance(payload, Ar5Snapshot):
            return
        self._snapshot = payload
        self._update_snapshot(payload)

    @Slot(str)
    def _on_refresh_failed(self, message: str) -> None:
        self.info_label.setText(f"{self.title} 状态读取失败：{message}")

    @Slot()
    def _on_refresh_finished(self) -> None:
        self._refresh_busy = False

    def _update_snapshot(self, snapshot: Ar5Snapshot) -> None:
        self.mode_indicator.set_state(snapshot.operate_mode)
        self.power_indicator.set_state(self._power_indicator_state(snapshot.power_state))
        self.operation_indicator.set_state(
            self._operation_indicator_state(snapshot.operation_state)
        )
        self._state_labels["type"].setText(snapshot.robot_type)
        self._state_labels["uid"].setText(snapshot.robot_uid)
        for index, (label, value) in enumerate(
            zip(self._joint_current_labels, snapshot.joint_deg, strict=True),
            start=1,
        ):
            label.setText(f"J{index}\n{value:.2f}°")
        pose_values = (*snapshot.xyz_mm, *snapshot.rpy_deg)
        for key, value in zip(("x", "y", "z", "rx", "ry", "rz"), pose_values, strict=True):
            unit = "mm" if key in {"x", "y", "z"} else "°"
            self._pose_current_labels[key].setText(f"{key.upper()}\n{value:.2f}{unit}")
        self._pose_current_labels["elbow"].setText(
            f"ELBOW\n{snapshot.elbow_deg:.2f}°"
        )
        self.info_label.setText(f"{self.title} 已连接")

    def _clear_display(self) -> None:
        for label in self._state_labels.values():
            label.setText("-")
        for label in self._joint_current_labels:
            label.setText(f"{label.text().splitlines()[0]}\n-")
        for key, label in self._pose_current_labels.items():
            label.setText(f"{key.upper()}\n-")

    # endregion

    # region Move

    @Slot()
    def _toggle_operate_mode(self) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.operate_mode not in {"automatic", "manual"}:
            return
        automatic = snapshot.operate_mode != "automatic"
        self._run_action(
            lambda client: client.set_operate_mode(automatic),
            "已切换到自动模式" if automatic else "已切换到手动模式",
        )

    @Slot()
    def _toggle_power(self) -> None:
        snapshot = self._snapshot
        if snapshot is None or snapshot.power_state not in {"on", "off"}:
            if snapshot is not None:
                self.info_label.setText(
                    f"{self.title} 当前为安全状态，不能通过使能灯切换上下电"
                )
            return
        self._run_action(
            lambda client: client.set_power(snapshot.power_state != "on"),
            "电机状态已切换",
        )

    @Slot()
    def _stop_motion(self) -> None:
        self._run_action(lambda client: client.stop(), "已发送停止命令")

    @Slot()
    def _move_joints(self) -> None:
        if not self._confirm_motion_state("Move", "automatic"):
            return
        targets = tuple(float(box.value()) for box in self._joint_target_boxes)
        speed, zone = self._move_settings()
        self._run_action(
            lambda client: client.move_joints_deg(
                targets,
                speed_mm_s=speed,
                zone_mm=zone,
            ),
            "MoveAbsJ 已下发",
        )

    @Slot()
    def _move_pose(self) -> None:
        if not self._confirm_motion_state("Move", "automatic"):
            return
        xyz_mm = (
            float(self._pose_target_boxes["x"].value()),
            float(self._pose_target_boxes["y"].value()),
            float(self._pose_target_boxes["z"].value()),
        )
        rpy_deg = (
            float(self._pose_target_boxes["rx"].value()),
            float(self._pose_target_boxes["ry"].value()),
            float(self._pose_target_boxes["rz"].value()),
        )
        elbow_deg = float(self._pose_target_boxes["elbow"].value())
        speed, zone = self._move_settings()
        self._run_action(
            lambda client: client.move_cartesian(
                xyz_mm,
                rpy_deg,
                elbow_deg,
                speed_mm_s=speed,
                zone_mm=zone,
            ),
            "MoveL 已下发",
        )

    @Slot()
    def _move_elbow(self) -> None:
        if not self._confirm_motion_state("Move", "automatic"):
            return
        elbow_deg = float(self._pose_target_boxes["elbow"].value())
        speed, zone = self._move_settings()
        self._run_action(
            lambda client: client.move_elbow_deg(
                elbow_deg,
                speed_mm_s=speed,
                zone_mm=zone,
            ),
            "Elbow 调整已下发",
        )

    def _copy_current_joint(self, joint_index: int) -> None:
        if self._snapshot is None:
            return
        self._joint_target_boxes[joint_index].setValue(
            self._snapshot.joint_deg[joint_index]
        )

    @Slot()
    def _copy_current_pose(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        for key, value in zip(
            ("x", "y", "z", "rx", "ry", "rz"),
            (*snapshot.xyz_mm, *snapshot.rpy_deg),
            strict=True,
        ):
            self._pose_target_boxes[key].setValue(value)
        self._pose_target_boxes["elbow"].setValue(snapshot.elbow_deg)

    def _move_settings(self) -> tuple[float, float]:
        return float(self.speed_box.value()), float(self.zone_box.value())

    @staticmethod
    def _power_indicator_state(power_state: str) -> str:
        normalized = power_state.replace("-", "_").lower()
        if normalized in {"safetystop", "safety_stop"}:
            return "safety_stop"
        if normalized in {"on", "off", "estop", "gstop"}:
            return normalized
        return "unknown"

    @staticmethod
    def _operation_indicator_state(operation_state: str) -> str:
        if operation_state in {"moving", "rlProgram"}:
            return "running"
        if operation_state in {
            "dynamicIdentify",
            "frictionIdentify",
            "loadIdentify",
        }:
            return "identify"
        if operation_state in {"rtControlling", "collaboration", "collaborate"}:
            return "collaboration"
        if operation_state in {
            "idle",
            "demo",
            "error",
            "debug",
        }:
            return operation_state
        return "unknown"

    # endregion

    def _confirm_motion_state(
        self,
        motion_name: str,
        required_mode: str,
    ) -> bool:
        snapshot = self._snapshot
        mode_text = "自动" if required_mode == "automatic" else "手动"
        if (
            snapshot is not None
            and snapshot.operate_mode == required_mode
            and snapshot.power_state == "on"
        ):
            return True
        QMessageBox.warning(
            self,
            f"{motion_name} 状态不满足",
            f"请先自主完成以下设置：\n"
            f"1. 点击工作模式指示灯，切换为{mode_text}模式\n"
            "2. 点击使能状态指示灯，切换为上电\n\n"
            "GUI 不会在 Move 前自动改变机器人状态。",
        )
        return False

    # region 后台动作

    def _run_action(
        self,
        callback: Callable[[RobotControlAr5Client], object],
        success_message: str,
    ) -> None:
        client = self._client
        if client is None:
            return
        logger.info(
            "AR5 GUI action submitted: side={} title={} action={}",
            self.side,
            self.title,
            success_message,
        )
        self.info_label.setText(f"{self.title} 执行中…")
        self._action_worker.submit(lambda: callback(client), success_message)

    @Slot(object)
    def _on_action_succeeded(self, payload: object) -> None:
        logger.success(
            "AR5 GUI action succeeded: side={} title={} result={}",
            self.side,
            self.title,
            payload,
        )
        self.info_label.setText(f"{self.title} | {payload}")
        QTimer.singleShot(100, self._request_refresh)

    @Slot(str)
    def _on_action_failed(self, message: str) -> None:
        logger.error(
            "AR5 GUI action failed: side={} title={} error={}",
            self.side,
            self.title,
            message,
        )
        self.info_label.setText(f"{self.title} 控制失败：{message}")
        if self._snapshot is not None:
            self._update_snapshot(self._snapshot)

    # endregion
