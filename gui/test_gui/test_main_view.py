from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loguru import logger
from PySide6.QtCore import QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSpinBox,
    QWidget,
)

from gui.test_gui.TestWujiCasiaArm_ui import Ui_MainWindow
from gui.test_gui.test_wuji_camera_tab import WujiCameraTabWidget
from gui.test_gui.test_wuji_casia_arm import TestWujiCasiaArmWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji import WujiQmlinkerBackend, load_wuji_robot_network_config

ServiceConnectionState = Literal["disconnected", "connecting", "connected", "disconnecting"]


@dataclass(slots=True)
class WujiCasiaDebugContext:
    """整机调试页面传递 qmlinker 连接信息的上下文。"""

    host_alias: str = "base_control"
    host: str = "192.168.100.60"
    port: int = 50062
    username: str = "wuji-brain"
    password: str = "wuji-brain"
    connected: bool = False
    connection_state: ServiceConnectionState = "disconnected"


class TestMainView(QMainWindow):
    """整机 GUI 调试主窗口。"""

    serviceConnectRequested = Signal(object)
    serviceDisconnectRequested = Signal(object)
    serviceStateRefreshRequested = Signal(object)
    dofTargetRequested = Signal(object, str, float)
    enableToggleRequested = Signal(object, str, bool)
    enableStateRefreshRequested = Signal(object, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        network_config = load_wuji_robot_network_config()
        self.context = WujiCasiaDebugContext(
            host=network_config.qmlinker.host,
            port=network_config.qmlinker.port,
        )
        self._arm_backend = WujiQmlinkerBackend(
            self,
            service_host_alias=self.context.host_alias,
        )
        self._state_refresh_timer = QTimer(self)
        self._state_refresh_timer.setInterval(50)
        self._service_status_timer = QTimer(self)
        self._service_status_timer.setInterval(1000)

        self._setup_service_context_editor()
        self._setup_robot_tab()
        self._setup_camera_tab()
        self._connect_signals()
        self._refresh_service_status_ui()
        self._service_status_timer.start()

    def _setup_service_context_editor(self) -> None:
        placeholder = self.ui.widget
        layout = QHBoxLayout(placeholder)
        layout.setContentsMargins(0, 0, 0, 0)

        self.host_edit = QLineEdit(self.context.host, placeholder)
        self.host_edit.setPlaceholderText("host")
        self.port_spinbox = QSpinBox(placeholder)
        self.port_spinbox.setRange(1, 65535)
        self.port_spinbox.setValue(self.context.port)
        self.user_edit = QLineEdit(self.context.username, placeholder)
        self.user_edit.setPlaceholderText("unused user")
        self.password_edit = QLineEdit(self.context.password, placeholder)
        self.password_edit.setPlaceholderText("unused password")
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.service_status_indicator = CasiaIndicatorLight(
            placeholder,
            text=("已连接", "未连接"),
            font_size=10,
            default_status=False,
        )
        self.service_status_label = QLabel(placeholder)

        layout.addWidget(QLabel("control ip", placeholder))
        layout.addWidget(self.host_edit)
        layout.addWidget(QLabel("port", placeholder))
        layout.addWidget(self.port_spinbox)
        layout.addWidget(QLabel("user", placeholder))
        layout.addWidget(self.user_edit)
        layout.addWidget(QLabel("password", placeholder))
        layout.addWidget(self.password_edit)
        layout.addWidget(QLabel("service", placeholder))
        layout.addWidget(self.service_status_indicator)
        layout.addWidget(self.service_status_label)

    def _setup_robot_tab(self) -> None:
        self.robot_widget = TestWujiCasiaArmWidget(self.ui.robot_tab)
        layout = QHBoxLayout(self.ui.robot_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.robot_widget)
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.robot_tab), "Wuji CASIA Arm")

    def _setup_camera_tab(self) -> None:
        self.camera_widget = WujiCameraTabWidget(self.ui.image_tab)
        layout = QHBoxLayout(self.ui.image_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.camera_widget)
        self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.image_tab), "Camera")

    def _connect_signals(self) -> None:
        self.ui.pushButton.clicked.connect(self._on_connect_button_clicked)
        self.ui.pushButton_2.clicked.connect(self._on_disconnect_button_clicked)
        self.robot_widget.dofTargetRequested.connect(self._on_dof_target_requested)
        self.robot_widget.enableToggleRequested.connect(self._on_enable_toggle_requested)
        self.robot_widget.enableStateRefreshRequested.connect(
            self._on_enable_state_refresh_requested
        )
        self.camera_widget.cameraSelected.connect(self._on_camera_selected)
        self.camera_widget.cameraEnableToggleRequested.connect(self._on_camera_enable_toggle_requested)
        self.camera_widget.rgbStreamRequested.connect(self._on_camera_rgb_stream_requested)
        self.camera_widget.rgbdStreamRequested.connect(self._on_camera_rgbd_stream_requested)
        self.camera_widget.streamStopRequested.connect(self._arm_backend.stop_camera_stream)
        self.ui.tabWidget.currentChanged.connect(self._on_main_tab_current_changed)
        self._state_refresh_timer.timeout.connect(self._on_state_refresh_timer_timeout)
        self._service_status_timer.timeout.connect(self._on_service_status_timer_timeout)
        self.serviceConnectRequested.connect(lambda _context: self._arm_backend.connect_service())
        self.serviceDisconnectRequested.connect(
            lambda _context: self._arm_backend.disconnect_service()
        )
        self.serviceStateRefreshRequested.connect(
            lambda _context: self._arm_backend.refresh_service_state()
        )
        self.dofTargetRequested.connect(
            lambda _context, axis, value: self._arm_backend.set_dof_target(axis, value)
        )
        self.enableToggleRequested.connect(
            lambda _context, device_name, enabled: self._arm_backend.set_enable_state(
                device_name,
                enabled,
            )
        )
        self.enableStateRefreshRequested.connect(
            lambda _context, device_name: self._arm_backend.refresh_enable_state(device_name)
        )
        self._arm_backend.serviceStateChanged.connect(self.update_service_connection_state)
        self._arm_backend.enableStateReceived.connect(self.update_enable_state)
        self._arm_backend.dofValuesReceived.connect(self.update_dof_values)
        self._arm_backend.cameraInventoryReceived.connect(self.camera_widget.update_camera_inventory)
        self._arm_backend.cameraEnableStateReceived.connect(self.camera_widget.update_camera_enable_state)
        self._arm_backend.cameraIntrinsicsReceived.connect(self.camera_widget.update_intrinsics)
        self._arm_backend.cameraFrameReceived.connect(self.camera_widget.update_frame)
        self._arm_backend.requestFailed.connect(self._on_backend_request_failed)

    def _sync_context_from_ui(self) -> None:
        self.context.host = self.host_edit.text().strip()
        self.context.port = int(self.port_spinbox.value())
        self.context.username = self.user_edit.text().strip()
        self.context.password = self.password_edit.text()
        self._arm_backend.configure_endpoint(self.context.host, self.context.port)

    @Slot()
    def _on_connect_button_clicked(self) -> None:
        self._sync_context_from_ui()
        logger.info(
            "TestMainView qmlinker connect clicked: alias={} target={}:{}",
            self.context.host_alias,
            self.context.host,
            self.context.port,
        )
        self._set_service_connection_state("connecting")
        self.serviceConnectRequested.emit(self.context)
        self.statusBar().showMessage(
            "qmlinker connect requested: " f"{self.context.host}:{self.context.port}"
        )

    @Slot()
    def _on_disconnect_button_clicked(self) -> None:
        logger.info("TestMainView qmlinker disconnect clicked: alias={}", self.context.host_alias)
        self._state_refresh_timer.stop()
        self._arm_backend.stop_camera_stream()
        self.camera_widget.clear_images()
        self._set_service_connection_state("disconnecting")
        self.serviceDisconnectRequested.emit(self.context)
        self.statusBar().showMessage("qmlinker disconnect requested")

    @Slot(str, float)
    def _on_dof_target_requested(self, axis_name: str, value: float) -> None:
        self.dofTargetRequested.emit(self.context, axis_name, value)
        self.statusBar().showMessage(f"DoF target requested: {axis_name} -> {value:.3f}")

    @Slot(str, bool)
    def _on_enable_toggle_requested(self, device_name: str, requested_enabled: bool) -> None:
        self.enableToggleRequested.emit(self.context, device_name, requested_enabled)
        self.statusBar().showMessage(
            f"Enable toggle requested: {device_name} -> {requested_enabled}"
        )

    @Slot(str)
    def _on_enable_state_refresh_requested(self, device_name: str) -> None:
        self.enableStateRefreshRequested.emit(self.context, device_name)
        self.statusBar().showMessage(f"Enable state refresh requested: {device_name}")

    @Slot(int)
    def _on_main_tab_current_changed(self, index: int) -> None:
        if self.ui.tabWidget.widget(index) is self.ui.image_tab:
            self._arm_backend.refresh_camera_inventory()
            self.camera_widget.activate_default_camera()

    @Slot(str)
    def _on_camera_selected(self, camera_name: str) -> None:
        self._arm_backend.refresh_camera_intrinsics(camera_name)
        self._arm_backend.refresh_camera_enable_state(camera_name)
        self.statusBar().showMessage(f"Camera selected: {camera_name}")

    @Slot(str, bool)
    def _on_camera_enable_toggle_requested(self, camera_name: str, requested_enabled: bool) -> None:
        self._arm_backend.set_camera_enable_state(camera_name, requested_enabled)
        self.statusBar().showMessage(f"Camera enable requested: {camera_name} -> {requested_enabled}")

    @Slot(str)
    def _on_camera_rgb_stream_requested(self, camera_name: str) -> None:
        self._arm_backend.start_camera_rgb_stream(camera_name)
        self.statusBar().showMessage(f"Camera RGB stream requested: {camera_name}")

    @Slot(str)
    def _on_camera_rgbd_stream_requested(self, camera_name: str) -> None:
        self._arm_backend.start_camera_rgbd_stream(camera_name)
        self.statusBar().showMessage(f"Camera RGBD stream requested: {camera_name}")

    def update_enable_state(self, device_name: str, enabled: bool) -> None:
        self.robot_widget.update_enable_state(device_name, enabled)

    def update_enable_states(self, states: dict[str, bool]) -> None:
        self.robot_widget.update_enable_states(states)

    def update_dof_value(self, axis_name: str, value: float) -> None:
        self.robot_widget.update_dof_value(axis_name, value)

    def update_dof_values(self, values: dict[str, float]) -> None:
        self.robot_widget.update_dof_values(values)

    def update_service_connection_state(self, connected: bool, message: str = "") -> None:
        """用 qmlinker 检测结果刷新连接状态。"""

        previous_state = self.context.connection_state
        self._set_service_connection_state("connected" if connected else "disconnected")
        if message:
            self.statusBar().showMessage(message)
        if not connected and previous_state in {"connecting", "connected"}:
            logger.error("qmlinker connection refresh failed: {}", message or "qmlinker 连接失败")

    @Slot(str)
    def _on_backend_request_failed(self, message: str) -> None:
        logger.error("TestMainView backend request failed: {}", message)
        self.statusBar().showMessage(message)

    @Slot()
    def _on_state_refresh_timer_timeout(self) -> None:
        if not self.context.connected:
            return
        self.robot_widget.update_dof_values(self._arm_backend.snapshot_state_values())
        self.robot_widget.update_enable_states(self._arm_backend.snapshot_enable_states())

    @Slot()
    def _on_service_status_timer_timeout(self) -> None:
        self._sync_context_from_ui()
        if not self.context.connected:
            return
        self.serviceStateRefreshRequested.emit(self.context)

    def _set_service_connection_state(self, state: ServiceConnectionState) -> None:
        previous_connected = self.context.connected
        if self.context.connection_state != state:
            logger.info(
                "TestMainView qmlinker UI state: {} -> {}",
                self.context.connection_state,
                state,
            )
        self.context.connection_state = state
        self.context.connected = state == "connected"
        self._refresh_service_status_ui()

        if self.context.connected and not previous_connected:
            self._state_refresh_timer.start()
        elif not self.context.connected:
            self._state_refresh_timer.stop()

    def _refresh_service_status_ui(self) -> None:
        state_text_map = {
            "disconnected": "未连接",
            "connecting": "连接中",
            "connected": "已连接",
            "disconnecting": "断开中",
        }
        self.service_status_indicator.set_status(self.context.connected)
        self.service_status_label.setText(state_text_map[self.context.connection_state])
        self.ui.pushButton.setEnabled(
            self.context.connection_state not in {"connecting", "connected"}
        )
        self.ui.pushButton_2.setEnabled(
            self.context.connection_state not in {"disconnected", "disconnecting"}
        )


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = TestMainView()
    window.show()
    sys.exit(app.exec())

