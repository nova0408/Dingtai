from __future__ import annotations

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.test.common import ActivatableTab
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji.client_base import WujiQmlinkerBaseClient


class AgvTabWidget(QWidget, ActivatableTab):
    """AGV 调试页。"""

    _MOVE_SPEED_MPS = 0.3

    # region 初始化

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client: WujiQmlinkerBaseClient | None = None
        self._active = False
        self._refresh_timer = QTimer(self)

        self.enable_indicator: CasiaIndicatorLight
        self.info_label: QLabel
        self.navi_status_label: QLabel
        self.x_label: QLabel
        self.y_label: QLabel
        self.yaw_label: QLabel
        self.battery_label: QLabel
        self.target_combo: QComboBox
        self.navigate_button: QPushButton
        self.forward_button: QPushButton
        self.backward_button: QPushButton
        self.left_button: QPushButton
        self.right_button: QPushButton

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
        self.info_label = QLabel("AGV 未连接", self)
        self.navi_status_label = QLabel("-", self)
        self.x_label = QLabel("-", self)
        self.y_label = QLabel("-", self)
        self.yaw_label = QLabel("-", self)
        self.battery_label = QLabel("-", self)
        self.target_combo = QComboBox(self)
        self.target_combo.setEditable(True)
        self.target_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.target_combo.addItem("charge")
        self.navigate_button = QPushButton("导航到目标点", self)
        self.forward_button = QPushButton("前进", self)
        self.backward_button = QPushButton("后退", self)
        self.left_button = QPushButton("左移", self)
        self.right_button = QPushButton("右移", self)

        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self._build_status_group(), 1)
        root_layout.addWidget(self._build_control_group(), 1)

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox("AGV 状态", self)
        layout = QVBoxLayout(group)

        top_row = QHBoxLayout()
        top_row.addWidget(self.enable_indicator)
        top_row.addWidget(self.info_label, 1)
        layout.addLayout(top_row)

        grid = QGridLayout()
        grid.addWidget(QLabel("navi_status", group), 0, 0)
        grid.addWidget(self.navi_status_label, 0, 1)
        grid.addWidget(QLabel("agv_x", group), 1, 0)
        grid.addWidget(self.x_label, 1, 1)
        grid.addWidget(QLabel("agv_y", group), 2, 0)
        grid.addWidget(self.y_label, 2, 1)
        grid.addWidget(QLabel("agv_yaw", group), 3, 0)
        grid.addWidget(self.yaw_label, 3, 1)
        grid.addWidget(QLabel("agv_battery", group), 4, 0)
        grid.addWidget(self.battery_label, 4, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)
        return group

    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("AGV 控制", self)
        layout = QVBoxLayout(group)

        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("目标点", group))
        nav_row.addWidget(self.target_combo, 1)
        nav_row.addWidget(self.navigate_button)
        layout.addLayout(nav_row)

        move_grid = QGridLayout()
        move_grid.addWidget(self.forward_button, 0, 1)
        move_grid.addWidget(self.left_button, 1, 0)
        move_grid.addWidget(self.right_button, 1, 2)
        move_grid.addWidget(self.backward_button, 2, 1)
        layout.addLayout(move_grid)
        layout.addStretch(1)
        return group

    def _connect_signals(self) -> None:
        self.enable_indicator.clicked.connect(self._on_enable_clicked)
        self.navigate_button.clicked.connect(self._on_navigate_clicked)
        self.forward_button.clicked.connect(lambda: self._move_direction(0))
        self.backward_button.clicked.connect(lambda: self._move_direction(180))
        self.left_button.clicked.connect(lambda: self._move_direction(90))
        self.right_button.clicked.connect(lambda: self._move_direction(270))

    # endregion

    # region 生命周期

    def set_client(self, client: WujiQmlinkerBaseClient | None) -> None:
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
            self.info_label.setText("AGV 未连接")
            return
        self._try_enable_on_activate()
        self._refresh_timer.start()
        self._refresh_state()

    def set_connection_ready(self, ready: bool) -> None:
        enabled = bool(ready)
        self.enable_indicator.setEnabled(enabled)
        self.target_combo.setEnabled(enabled)
        self.navigate_button.setEnabled(enabled)
        self.forward_button.setEnabled(enabled)
        self.backward_button.setEnabled(enabled)
        self.left_button.setEnabled(enabled)
        self.right_button.setEnabled(enabled)
        if not enabled:
            self.enable_indicator.set_status(False)
            self.navi_status_label.setText("-")
            self.x_label.setText("-")
            self.y_label.setText("-")
            self.yaw_label.setText("-")
            self.battery_label.setText("-")

    # endregion

    # region 使能

    def _try_enable_on_activate(self) -> None:
        if self._client is None:
            return
        try:
            if not self._client.get_agv_enable():
                self._client.set_agv_enable(True)
        except Exception:
            pass

    @Slot()
    def _on_enable_clicked(self) -> None:
        if self._client is None:
            return
        try:
            current_enabled = self._client.get_agv_enable()
            self._client.set_agv_enable(not current_enabled)
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"AGV 使能切换失败: {exc}")

    # endregion

    # region 刷新

    def _refresh_state(self) -> None:
        if self._client is None:
            return
        try:
            enabled = self._client.get_agv_enable()
            runtime_info = self._client.get_agv_runtime_info()
            navi_status = str(runtime_info.get("agv_navi_status", ""))
            agv_x = self._as_float(runtime_info.get("agv_x"))
            agv_y = self._as_float(runtime_info.get("agv_y"))
            agv_yaw = self._as_float(runtime_info.get("agv_yaw"))
            agv_battery = self._as_float(runtime_info.get("agv_battery"))

            self.enable_indicator.set_status(enabled)
            self.navi_status_label.setText(navi_status or "-")
            self.x_label.setText(f"{agv_x:.3f} m")
            self.y_label.setText(f"{agv_y:.3f} m")
            self.yaw_label.setText(f"{agv_yaw:.1f} deg")
            self.battery_label.setText(f"{agv_battery:.0f}%")
            self.info_label.setText(f"AGV enable={enabled}")
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"AGV 刷新失败: {exc}")

    # endregion

    # region 控制

    @Slot()
    def _on_navigate_clicked(self) -> None:
        if self._client is None:
            return
        target_name = self.target_combo.currentText().strip()
        if not target_name:
            self.info_label.setText("AGV 导航目标不能为空")
            return
        try:
            self._client.agv_navigate_to(target_name)
            self.info_label.setText(f"AGV 导航请求已发送: {target_name}")
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"AGV 导航失败: {exc}")

    def _move_direction(self, direction_deg: int) -> None:
        if self._client is None:
            return
        try:
            self._client.agv_real_time_translate(self._MOVE_SPEED_MPS, direction_deg)
            self._refresh_state()
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"AGV 平移失败: {exc}")

    # endregion

    # region 工具

    @staticmethod
    def _as_float(value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    # endregion
