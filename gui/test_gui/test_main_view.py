from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QSpinBox, QWidget

from gui.test_gui.TestWujiCasiaArm_ui import Ui_MainWindow
from gui.test_gui.test_wuji_casia_arm import TestWujiCasiaArmWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight

SshConnectionState = Literal["disconnected", "connecting", "connected", "disconnecting"]


@dataclass(slots=True)
class WujiCasiaDebugContext:
    """整机调试页面传递 SSH 连接信息的上下文。"""

    host_alias: str = "orin"
    host: str = "192.168.100.70"
    port: int = 22
    username: str = "wuji-brain"
    password: str = "wuji-brain"
    connected: bool = False
    connection_state: SshConnectionState = "disconnected"


class TestMainView(QMainWindow):
    """整机 GUI 调试主窗口。"""

    sshConnectRequested = Signal(object)
    sshDisconnectRequested = Signal(object)
    sshStateRefreshRequested = Signal(object)
    dofTargetRequested = Signal(object, str, float)
    dofValueRefreshRequested = Signal(object, str)
    enableToggleRequested = Signal(object, str, bool)
    enableStateRefreshRequested = Signal(object, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.context = WujiCasiaDebugContext()
        self._state_refresh_timer = QTimer(self)
        self._state_refresh_timer.setInterval(500)
        self._ssh_status_timer = QTimer(self)
        self._ssh_status_timer.setInterval(1000)

        self._setup_ssh_context_editor()
        self._setup_robot_tab()
        self._connect_signals()
        self._refresh_ssh_status_ui()
        self._ssh_status_timer.start()

    def _setup_ssh_context_editor(self) -> None:
        placeholder = self.ui.widget
        layout = QHBoxLayout(placeholder)
        layout.setContentsMargins(0, 0, 0, 0)

        self.host_edit = QLineEdit(self.context.host, placeholder)
        self.host_edit.setPlaceholderText("host")
        self.port_spinbox = QSpinBox(placeholder)
        self.port_spinbox.setRange(1, 65535)
        self.port_spinbox.setValue(self.context.port)
        self.user_edit = QLineEdit(self.context.username, placeholder)
        self.user_edit.setPlaceholderText("user")
        self.password_edit = QLineEdit(self.context.password, placeholder)
        self.password_edit.setPlaceholderText("password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.ssh_status_indicator = CasiaIndicatorLight(
            placeholder,
            text=("已连接", "未连接"),
            font_size=10,
            default_status=False,
        )
        self.ssh_status_label = QLabel(placeholder)

        layout.addWidget(QLabel("host", placeholder))
        layout.addWidget(self.host_edit)
        layout.addWidget(QLabel("port", placeholder))
        layout.addWidget(self.port_spinbox)
        layout.addWidget(QLabel("user", placeholder))
        layout.addWidget(self.user_edit)
        layout.addWidget(QLabel("password", placeholder))
        layout.addWidget(self.password_edit)
        layout.addWidget(QLabel("state", placeholder))
        layout.addWidget(self.ssh_status_indicator)
        layout.addWidget(self.ssh_status_label)

    def _setup_robot_tab(self) -> None:
        self.robot_widget = TestWujiCasiaArmWidget(self.ui.robot_tab)
        layout = QHBoxLayout(self.ui.robot_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.robot_widget)
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.robot_tab), "Wuji CASIA Arm")

    def _connect_signals(self) -> None:
        self.ui.pushButton.clicked.connect(self._on_connect_button_clicked)
        self.ui.pushButton_2.clicked.connect(self._on_disconnect_button_clicked)
        self.robot_widget.dofTargetRequested.connect(self._on_dof_target_requested)
        self.robot_widget.dofValueRefreshRequested.connect(self._on_dof_value_refresh_requested)
        self.robot_widget.enableToggleRequested.connect(self._on_enable_toggle_requested)
        self.robot_widget.enableStateRefreshRequested.connect(self._on_enable_state_refresh_requested)
        self._state_refresh_timer.timeout.connect(self._on_state_refresh_timer_timeout)
        self._ssh_status_timer.timeout.connect(self._on_ssh_status_timer_timeout)

    def _sync_context_from_ui(self) -> None:
        self.context.host = self.host_edit.text().strip()
        self.context.port = int(self.port_spinbox.value())
        self.context.username = self.user_edit.text().strip()
        self.context.password = self.password_edit.text()

    @Slot()
    def _on_connect_button_clicked(self) -> None:
        self._sync_context_from_ui()
        self._set_ssh_connection_state("connecting")
        self.sshConnectRequested.emit(self.context)
        self.statusBar().showMessage(
            f"SSH connect requested: {self.context.host_alias} {self.context.username}@{self.context.host}:{self.context.port}"
        )

    @Slot()
    def _on_disconnect_button_clicked(self) -> None:
        self._state_refresh_timer.stop()
        self._set_ssh_connection_state("disconnecting")
        self.sshDisconnectRequested.emit(self.context)
        self.statusBar().showMessage("SSH disconnect requested")

    @Slot(str, float)
    def _on_dof_target_requested(self, axis_name: str, value: float) -> None:
        self.dofTargetRequested.emit(self.context, axis_name, value)
        self.statusBar().showMessage(f"DoF target requested: {axis_name} -> {value:.3f}")

    @Slot(str)
    def _on_dof_value_refresh_requested(self, axis_name: str) -> None:
        self.dofValueRefreshRequested.emit(self.context, axis_name)

    @Slot(str, bool)
    def _on_enable_toggle_requested(self, device_name: str, requested_enabled: bool) -> None:
        self.enableToggleRequested.emit(self.context, device_name, requested_enabled)
        self.statusBar().showMessage(f"Enable toggle requested: {device_name} -> {requested_enabled}")

    @Slot(str)
    def _on_enable_state_refresh_requested(self, device_name: str) -> None:
        self.enableStateRefreshRequested.emit(self.context, device_name)
        self.statusBar().showMessage(f"Enable state refresh requested: {device_name}")

    def update_enable_state(self, device_name: str, enabled: bool) -> None:
        self.robot_widget.update_enable_state(device_name, enabled)

    def update_enable_states(self, states: dict[str, bool]) -> None:
        self.robot_widget.update_enable_states(states)

    def update_dof_value(self, axis_name: str, value: float) -> None:
        self.robot_widget.update_dof_value(axis_name, value)

    def update_dof_values(self, values: dict[str, float]) -> None:
        self.robot_widget.update_dof_values(values)

    def update_ssh_connection_state(self, connected: bool, message: str = "") -> None:
        """用真实 SSH 检测结果刷新连接状态。"""

        self._set_ssh_connection_state("connected" if connected else "disconnected")
        if message:
            self.statusBar().showMessage(message)

    @Slot()
    def _on_state_refresh_timer_timeout(self) -> None:
        if not self.context.connected:
            return
        self.robot_widget.request_all_dof_values_refresh()

    @Slot()
    def _on_ssh_status_timer_timeout(self) -> None:
        self._sync_context_from_ui()
        self.sshStateRefreshRequested.emit(self.context)

    def _set_ssh_connection_state(self, state: SshConnectionState) -> None:
        previous_connected = self.context.connected
        self.context.connection_state = state
        self.context.connected = state == "connected"
        self._refresh_ssh_status_ui()

        if self.context.connected and not previous_connected:
            self.robot_widget.request_all_enable_states_refresh()
            self.robot_widget.request_all_dof_values_refresh()
            self._state_refresh_timer.start()
        elif not self.context.connected:
            self._state_refresh_timer.stop()

    def _refresh_ssh_status_ui(self) -> None:
        state_text_map = {
            "disconnected": "未连接",
            "connecting": "连接中",
            "connected": "已连接",
            "disconnecting": "断开中",
        }
        self.ssh_status_indicator.set_status(self.context.connected)
        self.ssh_status_label.setText(state_text_map[self.context.connection_state])
        self.ui.pushButton.setEnabled(self.context.connection_state not in {"connecting", "connected"})
        self.ui.pushButton_2.setEnabled(self.context.connection_state not in {"disconnected", "disconnecting"})


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = TestMainView()
    window.show()
    sys.exit(app.exec())
