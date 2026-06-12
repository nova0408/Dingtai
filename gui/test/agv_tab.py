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

from gui.test.common import ActivatableTab, BackgroundCall
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji.agv_client import WujiAgvClient


class AgvTabWidget(QWidget, ActivatableTab):
    """AGV 调试页。"""

    _MOVE_SPEED_MPS = 0.3

    # region 初始化

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client: WujiAgvClient | None = None
        self._active = False
        self._refresh_in_flight = False
        self._refresh_timer = QTimer(self)
        self._refresh_call = BackgroundCall(self)

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
        self._connect_background_signals()
        self.set_connection_ready(False)

    def _setup_timer(self) -> None:
        self._refresh_timer.setInterval(300)
        self._refresh_timer.timeout.connect(self._request_refresh)

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
        self.target_combo.setPlaceholderText("未读取到目标点")
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

    def _connect_background_signals(self) -> None:
        self._refresh_call.succeeded.connect(self._on_refresh_succeeded)
        self._refresh_call.failed.connect(self._on_refresh_failed)
        self._refresh_call.finished.connect(self._on_refresh_finished)

    # endregion

    # region 生命周期

    def set_client(self, client: WujiAgvClient | None) -> None:
        self._client = client
        self._refresh_in_flight = False
        self._reload_navigation_targets()
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
        self._refresh_timer.start()
        self._request_refresh()

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

    # region 刷新

    def _request_refresh(self) -> None:
        if self._client is None or self._refresh_in_flight:
            return
        self._refresh_in_flight = True
        self._refresh_call.start(self._read_runtime_state)

    def _read_runtime_state(self) -> dict[str, object]:
        if self._client is None:
            raise RuntimeError("AGV 未连接")
        runtime_info = dict(self._client.get_runtime_info())
        runtime_info["agv_enabled"] = self._client.try_get_enable()
        return runtime_info

    @Slot(object)
    def _on_refresh_succeeded(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        runtime_info = payload
        navi_status = str(runtime_info.get("agv_navi_status", ""))
        agv_x = self._as_float(runtime_info.get("agv_x"))
        agv_y = self._as_float(runtime_info.get("agv_y"))
        agv_yaw = self._as_float(runtime_info.get("agv_yaw"))
        agv_battery = self._as_float(runtime_info.get("agv_battery"))
        enabled_state = runtime_info.get("agv_enabled")

        if isinstance(enabled_state, bool):
            self.enable_indicator.set_status(enabled_state)
        self.navi_status_label.setText(navi_status or "-")
        self.x_label.setText(f"{agv_x:.3f} m")
        self.y_label.setText(f"{agv_y:.3f} m")
        self.yaw_label.setText(f"{agv_yaw:.1f} deg")
        self.battery_label.setText(f"{agv_battery:.0f}%")
        self.info_label.setText("AGV 状态刷新成功")

    @Slot(str)
    def _on_refresh_failed(self, message: str) -> None:
        self.info_label.setText(f"AGV 刷新失败: {message}")

    @Slot()
    def _on_refresh_finished(self) -> None:
        self._refresh_in_flight = False

    # endregion

    # region 控制

    @Slot()
    def _on_enable_clicked(self) -> None:
        if self._client is None:
            return
        self._refresh_call.start(self._toggle_enable)

    def _toggle_enable(self) -> None:
        if self._client is None:
            raise RuntimeError("AGV 未连接")
        current_enabled = self._client.try_get_enable()
        if current_enabled is None:
            raise RuntimeError("AGV 使能状态不可读取")
        changed = bool(self._client.set_enable(not current_enabled))
        if not changed:
            raise RuntimeError("AGV 使能切换失败")
        self.enable_indicator.set_status(not current_enabled)
        return None

    @Slot()
    def _on_navigate_clicked(self) -> None:
        if self._client is None:
            return
        target_name = self.target_combo.currentText().strip()
        if not target_name:
            self.info_label.setText("AGV 导航目标不能为空")
            return
        client = self._client
        self._refresh_call.start(lambda: client.navigate_to(target_name))
        self.info_label.setText(f"AGV 导航请求已发送: {target_name}")

    def _move_direction(self, direction_deg: int) -> None:
        if self._client is None:
            return
        client = self._client
        self._refresh_call.start(
            lambda: client.real_time_translate(self._MOVE_SPEED_MPS, direction_deg)
        )
        self._request_refresh()

    # endregion

    # region 工具

    @staticmethod
    def _as_float(value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    def _reload_navigation_targets(self) -> None:
        self.target_combo.clear()
        client = self._client
        if client is None:
            self.target_combo.setPlaceholderText("未读取到目标点")
            return
        target_names = client.get_navigation_targets()
        if target_names:
            self.target_combo.addItems(target_names)
            self.target_combo.setCurrentIndex(0)
            return
        self.target_combo.setPlaceholderText("未读取到目标点")

    # endregion
