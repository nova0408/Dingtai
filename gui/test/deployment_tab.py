from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.test.common import ActivatableTab, BackgroundCall
from record_replay.client import RecordReplayClient


@dataclass(frozen=True, slots=True)
class DeploymentTaskItem:
    """主页展示的一个左右臂对齐执行任务。"""

    sequence: int
    left_csv: str | None
    right_csv: str | None
    synchronized: bool


@dataclass(frozen=True, slots=True)
class DeploymentStatus:
    """从 RecordReplay ``GET /status`` 解析出的主页快照。"""

    state: str
    error_text: str | None
    tasks: tuple[DeploymentTaskItem, ...]
    current_task_sequence: int
    current_task_active: bool
    total_execution_count: int
    current_left_csv: str | None
    current_right_csv: str | None
    current_left_row: int | None
    current_right_row: int | None
    current_left_total_rows: int | None
    current_right_total_rows: int | None


class _ArmTaskWidget(QWidget):
    """单独展示一侧 AR5 任务，同时保留与另一侧一致的执行序号行。"""

    def __init__(self, side: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._side = side
        self._tasks: tuple[DeploymentTaskItem, ...] = ()
        self._items: dict[int, QTreeWidgetItem] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title, self)
        title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        layout.addWidget(title_label)
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(("次数", "CSV 任务", "行进度", "执行状态"))
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 90)
        self.tree.setColumnWidth(1, 330)
        self.tree.setColumnWidth(2, 130)
        self.tree.setColumnWidth(3, 150)
        layout.addWidget(self.tree, 1)

    def set_tasks(self, tasks: tuple[DeploymentTaskItem, ...]) -> None:
        """使用完整执行序号重建该侧任务，空白阶段也保留占位行。"""

        self._tasks = tasks
        self.tree.clear()
        self._items.clear()
        for task in tasks:
            csv_name = self._csv_name(task)
            item = QTreeWidgetItem(
                (
                    f"第 {task.sequence} 次",
                    csv_name or "—",
                    "等待" if csv_name else "—",
                    _pending_state(task, csv_name is not None),
                )
            )
            self.tree.addTopLevelItem(item)
            self._items[task.sequence] = item

    def update_status(
        self,
        current_sequence: int,
        current_active: bool,
        current_csv: str | None,
        current_row: int | None,
        total_rows: int | None,
    ) -> None:
        """刷新完成、当前和待执行状态，并保持两侧相同序号可对照。"""

        for task in self._tasks:
            item = self._items[task.sequence]
            csv_name = self._csv_name(task)
            self._reset_item_style(item)
            item.setText(2, "等待" if csv_name else "—")
            item.setText(3, _pending_state(task, csv_name is not None))
            if task.sequence < current_sequence or (
                task.sequence == current_sequence and not current_active
            ):
                item.setText(2, "完成" if csv_name else "—")
                item.setText(3, _finished_state(task, csv_name is not None))
            elif task.sequence == current_sequence and current_active:
                if csv_name is None:
                    item.setText(3, "本阶段无任务")
                    self._highlight_item(item, QColor("#e3f2fd"))
                else:
                    item.setText(
                        2,
                        _format_progress(current_row, total_rows)
                        if current_csv == csv_name
                        else "准备中",
                    )
                    item.setText(
                        3,
                        "执行中 · 同步" if task.synchronized else "执行中 · 单臂",
                    )
                    self._highlight_item(item, QColor("#c8e6c9"))
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)

    def _csv_name(self, task: DeploymentTaskItem) -> str | None:
        return task.left_csv if self._side == "left" else task.right_csv

    @staticmethod
    def _highlight_item(item: QTreeWidgetItem, color: QColor) -> None:
        brush = QBrush(color)
        for column in range(4):
            font = item.font(column)
            font.setBold(True)
            item.setBackground(column, brush)
            item.setFont(column, font)

    @staticmethod
    def _reset_item_style(item: QTreeWidgetItem) -> None:
        for column in range(4):
            font = item.font(column)
            font.setBold(False)
            item.setBackground(column, QBrush())
            item.setFont(column, font)


class DeploymentTabWidget(QWidget, ActivatableTab):
    """展示服务累计次数、当前条目和左右臂独立任务组件。"""

    REFRESH_INTERVAL_MS = 1000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service_host = ""
        self._active = False
        self._refresh_busy = False
        self._command_busy = False
        self._latest_state = "unknown"
        self._last_tasks: tuple[DeploymentTaskItem, ...] | None = None
        self._refresh_timer = QTimer(self)
        self._refresh_call = BackgroundCall(self)
        self._command_call = BackgroundCall(self)
        self._setup_ui()
        self._connect_signals()
        self._update_action_enablement()

    # region 初始化

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        header = QHBoxLayout()
        self.state_label = QLabel("RecordReplay 服务未配置", self)
        self.address_label = QLabel("-", self)
        self.address_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.refresh_button = QPushButton("立即刷新", self)
        header.addWidget(self.state_label, 2)
        header.addWidget(self.address_label, 1)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        summary_row = QHBoxLayout()
        count_frame = QFrame(self)
        count_frame.setObjectName("countFrame")
        count_layout = QHBoxLayout(count_frame)
        count_layout.addWidget(QLabel("服务累计执行次数", count_frame))
        self.total_execution_label = QLabel("0", count_frame)
        self.total_execution_label.setObjectName("executionCount")
        count_layout.addWidget(self.total_execution_label)
        count_layout.addStretch(1)
        summary_row.addWidget(count_frame, 1)

        self.agv_navigation_checkbox = QCheckBox("执行 AGV 导航", self)
        self.agv_navigation_checkbox.setChecked(True)
        self.start_button = QPushButton("启动执行", self)
        self.start_button.setObjectName("startButton")
        self.stop_button = QPushButton("停止执行（待实现）", self)
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("RecordReplay 停止接口暂未实现")
        summary_row.addWidget(self.agv_navigation_checkbox)
        summary_row.addWidget(self.start_button)
        summary_row.addWidget(self.stop_button)
        layout.addLayout(summary_row)

        current_frame = QFrame(self)
        current_frame.setObjectName("currentTaskFrame")
        current_layout = QVBoxLayout(current_frame)
        current_title = QLabel("当前正在执行", current_frame)
        current_title.setObjectName("currentTaskTitle")
        self.current_task_label = QLabel("当前没有正在执行的条目", current_frame)
        self.current_task_label.setObjectName("currentTaskLabel")
        self.current_task_label.setWordWrap(True)
        current_layout.addWidget(current_title)
        current_layout.addWidget(self.current_task_label)
        layout.addWidget(current_frame)

        task_layout = QHBoxLayout()
        self.left_task_widget = _ArmTaskWidget("left", "左 AR5 任务", self)
        self.right_task_widget = _ArmTaskWidget("right", "右 AR5 任务", self)
        task_layout.addWidget(self.left_task_widget, 1)
        task_layout.addWidget(self.right_task_widget, 1)
        layout.addLayout(task_layout, 1)

        self.setStyleSheet(
            """
            QLabel, QCheckBox { font-size: 17px; }
            QPushButton { min-height: 48px; min-width: 120px; font-size: 17px; }
            QCheckBox { min-height: 48px; spacing: 10px; }
            QCheckBox::indicator { width: 30px; height: 30px; }
            QFrame#countFrame {
                border: 1px solid #cfd8dc;
                border-radius: 10px;
                background: #ffffff;
            }
            QLabel#executionCount {
                color: #1565c0;
                font-size: 32px;
                font-weight: 700;
            }
            QPushButton#startButton {
                background: #2e7d32;
                color: white;
                font-weight: 700;
            }
            QPushButton#stopButton {
                background: #eeeeee;
                color: #757575;
                font-weight: 700;
            }
            QFrame#currentTaskFrame {
                border: 2px solid #90caf9;
                border-radius: 10px;
                background: #e3f2fd;
            }
            QLabel#currentTaskTitle { color: #455a64; font-size: 16px; }
            QLabel#currentTaskLabel {
                color: #0d47a1;
                font-size: 22px;
                font-weight: 700;
            }
            QTreeWidget { font-size: 17px; }
            QTreeWidget::item { min-height: 42px; }
            QHeaderView::section { min-height: 46px; font-size: 17px; }
            """
        )

    def _connect_signals(self) -> None:
        self.refresh_button.clicked.connect(self._request_refresh)
        self.start_button.clicked.connect(self._start_execution_requested)
        self._refresh_timer.setInterval(self.REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._request_refresh)
        self._refresh_call.succeeded.connect(self._on_refresh_succeeded)
        self._refresh_call.failed.connect(self._on_refresh_failed)
        self._refresh_call.finished.connect(self._on_refresh_finished)
        self._command_call.succeeded.connect(self._on_start_succeeded)
        self._command_call.failed.connect(self._on_start_failed)
        self._command_call.finished.connect(self._on_start_finished)

    # endregion

    # region 生命周期

    def set_service_host(self, host: str) -> None:
        """设置 RecordReplay 服务所在 Orin 地址。"""

        normalized_host = host.strip()
        if normalized_host == self._service_host:
            return
        self._service_host = normalized_host
        self.address_label.setText(
            f"http://{normalized_host}:6300" if normalized_host else "-"
        )
        self._latest_state = "unknown"
        self._last_tasks = None
        self.left_task_widget.set_tasks(())
        self.right_task_widget.set_tasks(())
        self.total_execution_label.setText("0")
        self.current_task_label.setText("当前没有正在执行的条目")
        self._update_action_enablement()
        if self._active:
            self._request_refresh()

    def set_active(self, active: bool) -> None:
        self._active = active
        if active and self._service_host:
            self._refresh_timer.start()
            self._request_refresh()
            return
        self._refresh_timer.stop()

    def set_connection_ready(self, ready: bool) -> None:
        """主页只依赖 RecordReplay HTTP，不跟随手动设备连接使能。"""

        del ready

    # endregion

    # region 状态刷新

    @Slot()
    def _request_refresh(self) -> None:
        if not self._service_host or self._refresh_busy:
            return
        self._refresh_busy = True
        self._refresh_call.start(
            lambda: self._new_client().get_status()
        )

    @Slot(object)
    def _on_refresh_succeeded(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        status = _parse_deployment_status(payload)
        self._latest_state = status.state
        self._update_overview(status)
        self._update_task_widgets(status)
        state_text = _translate_state(status.state)
        if status.error_text:
            self.state_label.setText(f"RecordReplay：{state_text} · {status.error_text}")
            self.state_label.setStyleSheet("color: #b3261e; font-weight: 700;")
        else:
            self.state_label.setText(f"RecordReplay：{state_text}")
            self.state_label.setStyleSheet("color: #137333; font-weight: 700;")
        self._update_action_enablement()

    @Slot(str)
    def _on_refresh_failed(self, message: str) -> None:
        self._latest_state = "unknown"
        self.state_label.setText(f"RecordReplay 状态不可用：{message}")
        self.state_label.setStyleSheet("color: #b3261e; font-weight: 700;")
        self._update_action_enablement()

    @Slot()
    def _on_refresh_finished(self) -> None:
        self._refresh_busy = False

    # endregion

    # region 执行控制

    @Slot()
    def _start_execution_requested(self) -> None:
        use_agv = self.agv_navigation_checkbox.isChecked()
        navigation_text = "包含 AGV 导航" if use_agv else "不执行 AGV 导航"
        answer = QMessageBox.question(
            self,
            "确认启动 RecordReplay",
            f"即将启动机械臂回放（{navigation_text}）。\n请确认设备运动区域安全。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._command_busy = True
        self._update_action_enablement()
        self.state_label.setText("正在发送启动指令…")
        self._command_call.start(lambda: self._new_client().start(use_agv))

    @Slot(object)
    def _on_start_succeeded(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self.state_label.setText("启动响应格式无效")
            return
        if payload.get("accepted") is True:
            self.state_label.setText("RecordReplay 已接受启动指令")
            return
        self.state_label.setText("RecordReplay 正在执行，未接受重复启动")

    @Slot(str)
    def _on_start_failed(self, message: str) -> None:
        self.state_label.setText(f"启动失败：{message}")
        self.state_label.setStyleSheet("color: #b3261e; font-weight: 700;")

    @Slot()
    def _on_start_finished(self) -> None:
        self._command_busy = False
        self._request_refresh()
        self._update_action_enablement()

    def _update_action_enablement(self) -> None:
        can_start = (
            bool(self._service_host)
            and not self._command_busy
            and self._latest_state in {"waiting", "failed"}
        )
        self.agv_navigation_checkbox.setEnabled(can_start)
        self.start_button.setEnabled(can_start)
        self.stop_button.setEnabled(False)

    # endregion

    # region 展示

    def _update_overview(self, status: DeploymentStatus) -> None:
        self.total_execution_label.setText(str(status.total_execution_count))
        if status.current_task_sequence <= 0 or not status.current_task_active:
            self.current_task_label.setText("当前没有正在执行的条目")
            return
        task = next(
            (
                item
                for item in status.tasks
                if item.sequence == status.current_task_sequence
            ),
            None,
        )
        if task is None:
            self.current_task_label.setText(
                f"任务第 {status.current_task_sequence} 项正在执行"
            )
            return
        mode_text = "同步运动" if task.synchronized else "单臂运动"
        self.current_task_label.setText(
            f"任务第 {task.sequence} 项　左臂：{task.left_csv or '—'}　｜　"
            f"右臂：{task.right_csv or '—'}　｜　{mode_text}"
        )

    def _update_task_widgets(self, status: DeploymentStatus) -> None:
        if status.tasks != self._last_tasks:
            self.left_task_widget.set_tasks(status.tasks)
            self.right_task_widget.set_tasks(status.tasks)
            self._last_tasks = status.tasks
        self.left_task_widget.update_status(
            status.current_task_sequence,
            status.current_task_active,
            status.current_left_csv,
            status.current_left_row,
            status.current_left_total_rows,
        )
        self.right_task_widget.update_status(
            status.current_task_sequence,
            status.current_task_active,
            status.current_right_csv,
            status.current_right_row,
            status.current_right_total_rows,
        )

    def _new_client(self) -> RecordReplayClient:
        return RecordReplayClient(
            base_url=f"http://{self._service_host}:6300",
            timeout_s=2.0,
        )

    # endregion


def _parse_deployment_status(payload: dict[str, object]) -> DeploymentStatus:
    """校验并解析 RecordReplay 状态响应。"""

    return DeploymentStatus(
        state=str(payload.get("state", "unknown")),
        error_text=_optional_text(payload.get("error_text")),
        tasks=_parse_execution_tasks(payload.get("execution_tasks")),
        current_task_sequence=_nonnegative_int(payload.get("current_task_sequence")),
        current_task_active=payload.get("current_task_active") is True,
        total_execution_count=_nonnegative_int(payload.get("total_execution_count")),
        current_left_csv=_optional_text(payload.get("current_left_csv")),
        current_right_csv=_optional_text(payload.get("current_right_csv")),
        current_left_row=_optional_int(payload.get("current_left_row")),
        current_right_row=_optional_int(payload.get("current_right_row")),
        current_left_total_rows=_optional_int(payload.get("current_left_total_rows")),
        current_right_total_rows=_optional_int(payload.get("current_right_total_rows")),
    )


def _parse_execution_tasks(value: object) -> tuple[DeploymentTaskItem, ...]:
    if not isinstance(value, list):
        return ()
    tasks: list[DeploymentTaskItem] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        sequence = item.get("sequence")
        synchronized = item.get("synchronized")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= 0
            or not isinstance(synchronized, bool)
        ):
            continue
        tasks.append(
            DeploymentTaskItem(
                sequence=sequence,
                left_csv=_optional_text(item.get("left_csv")),
                right_csv=_optional_text(item.get("right_csv")),
                synchronized=synchronized,
            )
        )
    return tuple(tasks)


def _pending_state(task: DeploymentTaskItem, has_csv: bool) -> str:
    if not has_csv:
        return "本阶段无任务"
    return "待执行 · 同步" if task.synchronized else "待执行 · 单臂"


def _finished_state(task: DeploymentTaskItem, has_csv: bool) -> str:
    if not has_csv:
        return "本阶段无任务"
    return "已完成 · 同步" if task.synchronized else "已完成 · 单臂"


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _nonnegative_int(value: object) -> int:
    parsed = _optional_int(value)
    return max(parsed, 0) if parsed is not None else 0


def _format_progress(current_row: int | None, total_rows: int | None) -> str:
    if current_row is None or total_rows is None:
        return "准备中"
    return f"{current_row} / {total_rows}"


def _translate_state(state: str) -> str:
    return {
        "waiting": "等待启动",
        "navigating_to_start": "前往起点",
        "replaying": "正在回放",
        "navigating_to_finish": "返回终点",
        "failed": "执行失败",
    }.get(state, state)
