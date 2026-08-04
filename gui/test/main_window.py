from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger
from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from camera_pipeline.service.http_client import CameraPipelineHttpClient
from gui.test.agv_tab import AgvTabWidget
from gui.test.algo_tab import AlgoPlaceholderTabWidget
from gui.test.arm_tab import ArmTabWidget
from gui.test.body_tab import BodyTabWidget
from gui.test.camera_bridge import CameraBridge
from gui.test.camera_tab import WujiCameraTabWidget
from gui.test.common import ActivatableTab, BackgroundCall
from gui.test.deployment_tab import DeploymentTabWidget
from gui.test.gripper_tab import GripperTabWidget
from gui.test.hand_tab import M6HandTabWidget
from gui.test.head_tab import HeadTabWidget
from gui.test.prior_calibration_tab import PriorCalibrationTabWidget
from gui.test.robot_control_clients import (
    RobotControlAgvClient,
    RobotControlAr5Client,
    RobotControlBodyClient,
    RobotControlGripperClient,
    RobotControlHeadClient,
    RobotControlRightHandClient,
    RobotControlStatusStream,
)
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from robot_control.service.client import RobotControlClient
from src.wuji.camera_protocol import WujiCameraFrame
from src.wuji.prior_calibration import PriorCalibrationRecorder

DEFAULT_GATEWAY_HOST = "wujibrain-desktop"
LEGACY_GATEWAY_IPS = frozenset({"192.168.1.128", "192.168.100.50"})
DEFAULT_ROBOT_CONTROL_API_PREFIX = "/api/v1/robot-control"
DEFAULT_CAMERA_API_PREFIX = "/api/v1/camera"
DEFAULT_CAMERA_WEBSOCKET_PREFIX = "/api/v1/camera-ws"
SETTINGS_ORGANIZATION = "DingTai"
SETTINGS_APPLICATION = "WujiTouchGui"
SETTINGS_LAST_HOST_KEY = "connection/last_host"
SETTINGS_LEGACY_IP_KEY = "connection/last_ip"


@dataclass(slots=True)
class ConnectionBundle:
    """GUI 全部设备连接及其统一释放顺序。"""

    robot_control: RobotControlClient
    status_stream: RobotControlStatusStream
    left_arm: RobotControlAr5Client
    right_arm: RobotControlAr5Client
    agv: RobotControlAgvClient
    head: RobotControlHeadClient
    body: RobotControlBodyClient
    gripper: RobotControlGripperClient
    right_hand: RobotControlRightHandClient
    camera: CameraPipelineHttpClient

    def close(self) -> None:
        """停止共享状态流并关闭相机 HTTP 客户端。"""

        self.status_stream.close()
        try:
            self.camera.close()
        except Exception:
            pass


class _ConnectionWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._call = BackgroundCall(self)
        self._call.succeeded.connect(self.succeeded)
        self._call.failed.connect(self.failed)

    def start(self, callback) -> None:  # noqa: ANN001
        self._call.start(callback)


class TestGuiMainWindow(QMainWindow):
    """触摸屏优先的整机调试主窗口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bundle: ConnectionBundle | None = None
        self._tabs: list[ActivatableTab] = []
        self._requested_host = DEFAULT_GATEWAY_HOST
        self._camera_bridge = CameraBridge(self)
        self._connection_worker = _ConnectionWorker(self)
        self._settings = QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        self._build_ui()
        self._connect_signals()
        self._load_last_network_config()
        self._apply_connection_state(False, "未连接")
        self.setWindowState(self.windowState() | Qt.WindowState.WindowMaximized)
        self._on_current_tab_changed(self.tab_widget.currentIndex())

    # region 初始化

    def _build_ui(self) -> None:
        self.setWindowTitle("DingTai 触摸控制台")
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        connect_row = QHBoxLayout()
        self.host_edit = QLineEdit(central)
        self.host_edit.setPlaceholderText("Gateway HTTPS 主机名（例如 wujibrain-desktop）")
        self.host_edit.setMinimumHeight(52)
        self.connect_button = QPushButton("连接", central)
        self.disconnect_button = QPushButton("断开", central)
        self.connect_button.setMinimumHeight(52)
        self.disconnect_button.setMinimumHeight(52)
        self.connection_indicator = CasiaIndicatorLight(
            central,
            text=("已连接", "未连接"),
            font_size=14,
            default_status=False,
        )
        self.connection_indicator.setMinimumHeight(52)
        self.connection_label = QLabel("未连接", central)
        self.connection_label.setWordWrap(True)
        connect_row.addWidget(QLabel("接入 IP", central))
        connect_row.addWidget(self.host_edit, 2)
        connect_row.addWidget(self.connect_button)
        connect_row.addWidget(self.disconnect_button)
        connect_row.addWidget(self.connection_indicator)
        connect_row.addWidget(self.connection_label, 2)
        root_layout.addLayout(connect_row)

        self.tab_widget = QTabWidget(central)
        self.tab_widget.setDocumentMode(True)
        self.deployment_tab = DeploymentTabWidget(self.tab_widget)
        self.agv_tab = AgvTabWidget(self.tab_widget)
        self.head_tab = HeadTabWidget(self.tab_widget)
        self.body_tab = BodyTabWidget(self.tab_widget)
        self.left_arm_tab = ArmTabWidget("left", "左 AR5", self.tab_widget)
        self.right_arm_tab = ArmTabWidget("right", "右 AR5", self.tab_widget)
        self.gripper_tab = GripperTabWidget(self.tab_widget)
        self.hand_tab = M6HandTabWidget(self.tab_widget)
        self.camera_tab = WujiCameraTabWidget(self.tab_widget)
        self.prior_calibration_tab = PriorCalibrationTabWidget(self.tab_widget)
        self.algo_tab = AlgoPlaceholderTabWidget(self.tab_widget)
        for title, widget in (
            ("项目主页", self.deployment_tab),
            ("AGV", self.agv_tab),
            ("头部", self.head_tab),
            ("升降", self.body_tab),
            ("左 AR5", self.left_arm_tab),
            ("右 AR5", self.right_arm_tab),
            ("夹爪", self.gripper_tab),
            ("右手", self.hand_tab),
            ("相机", self.camera_tab),
            ("先验标定", self.prior_calibration_tab),
            ("算法", self.algo_tab),
        ):
            self.tab_widget.addTab(widget, title)
        root_layout.addWidget(self.tab_widget, 1)

        self._tabs = [
            self.deployment_tab,
            self.agv_tab,
            self.head_tab,
            self.body_tab,
            self.left_arm_tab,
            self.right_arm_tab,
            self.gripper_tab,
            self.hand_tab,
            self.prior_calibration_tab,
            self.algo_tab,
        ]
        self.setStyleSheet("""
            QMainWindow, QWidget { font-size: 16px; }
            QLineEdit { padding: 8px; font-size: 18px; }
            QPushButton { min-height: 46px; min-width: 82px; font-size: 17px; }
            QTabBar::tab { min-height: 52px; min-width: 108px; font-size: 16px; }
            QStatusBar {
                background: #e8eef2;
                border-top: 2px solid #78909c;
                color: #263238;
                min-height: 32px;
                padding: 3px 10px;
                font-size: 14px;
                font-weight: 600;
            }
            """)
        self.statusBar().setSizeGripEnabled(False)

    def _connect_signals(self) -> None:
        self.connect_button.clicked.connect(self._connect_requested)
        self.disconnect_button.clicked.connect(self._disconnect_requested)
        self.tab_widget.currentChanged.connect(self._on_current_tab_changed)
        self.camera_tab.cameraSelected.connect(self._camera_bridge.refresh_camera)
        self.camera_tab.rgbdStreamRequested.connect(self._camera_bridge.start_rgbd_stream)
        self.camera_tab.streamStopRequested.connect(self._camera_bridge.stop_stream)
        self.prior_calibration_tab.cameraStreamRequested.connect(self._camera_bridge.start_rgb_stream)
        self.prior_calibration_tab.streamStopRequested.connect(self._camera_bridge.stop_stream)
        self._camera_bridge.inventoryReady.connect(self.camera_tab.update_camera_inventory)
        self._camera_bridge.connectionStateReady.connect(self.camera_tab.update_camera_connection_state)
        self._camera_bridge.intrinsicsReady.connect(self.camera_tab.update_intrinsics)
        self._camera_bridge.frameReady.connect(self._on_camera_frame_ready)
        self._camera_bridge.errorRaised.connect(self._on_camera_error)
        self._connection_worker.succeeded.connect(self._on_bundle_ready)
        self._connection_worker.failed.connect(self._on_bundle_failed)
        self.host_edit.editingFinished.connect(self._update_deployment_host)

    def _load_last_network_config(self) -> None:
        saved_host = str(
            self._settings.value(
                SETTINGS_LAST_HOST_KEY,
                self._settings.value(SETTINGS_LEGACY_IP_KEY, DEFAULT_GATEWAY_HOST),
            )
        ).strip()
        if saved_host in LEGACY_GATEWAY_IPS:
            saved_host = DEFAULT_GATEWAY_HOST
        self.host_edit.setText(saved_host or DEFAULT_GATEWAY_HOST)
        self._update_deployment_host()

    # endregion

    # region 连接生命周期

    @Slot()
    def _connect_requested(self) -> None:
        requested_host = self.host_edit.text().strip()
        if not requested_host:
            self._apply_connection_state(False, "连接失败：Gateway 地址不能为空")
            return
        self._requested_host = requested_host
        self._settings.setValue(SETTINGS_LAST_HOST_KEY, requested_host)
        self.deployment_tab.set_service_host(requested_host)
        self._disconnect_requested()
        self._apply_connection_state(False, "正在连接 Gateway HTTP…")
        self.connect_button.setEnabled(False)
        self._connection_worker.start(self._create_connection_bundle)

    @Slot(object)
    def _on_bundle_ready(self, payload: object) -> None:
        if not isinstance(payload, ConnectionBundle):
            return
        self._bundle = payload
        self.agv_tab.set_client(payload.agv)
        self.head_tab.set_client(payload.head)
        self.body_tab.set_client(payload.body)
        self.left_arm_tab.set_client(payload.left_arm)
        self.right_arm_tab.set_client(payload.right_arm)
        self.gripper_tab.set_client(payload.gripper)
        self.hand_tab.set_client(payload.right_hand)
        self.prior_calibration_tab.set_clients(
            payload.left_arm,
            payload.head,
            PriorCalibrationRecorder(payload.camera),
        )
        self._camera_bridge.set_client(payload.camera)
        mode_text = "Gateway HTTPS"
        self._apply_connection_state(True, f"已连接 · {mode_text}")
        self._on_current_tab_changed(self.tab_widget.currentIndex())
        logger.info("GUI connected: gateway_host={} mode={}", self._requested_host, mode_text)

    @Slot(str)
    def _on_bundle_failed(self, message: str) -> None:
        logger.error("GUI connect failed: {}", message)
        self._disconnect_requested()
        self._apply_connection_state(False, f"连接失败：{message}")

    @Slot()
    def _disconnect_requested(self) -> None:
        for tab in self._tabs:
            tab.set_active(False)
        self._camera_bridge.stop_stream()
        self._camera_bridge.set_client(None)
        self.agv_tab.set_client(None)
        self.head_tab.set_client(None)
        self.body_tab.set_client(None)
        self.left_arm_tab.set_client(None)
        self.right_arm_tab.set_client(None)
        self.gripper_tab.set_client(None)
        self.hand_tab.set_client(None)
        self.prior_calibration_tab.set_clients(None, None, None)
        bundle = self._bundle
        self._bundle = None
        if bundle is not None:
            try:
                bundle.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("bundle close failed: {}", exc)
        self._apply_connection_state(False, "已断开")
        self._on_current_tab_changed(self.tab_widget.currentIndex())

    def closeEvent(self, event) -> None:  # noqa: ANN001, N802
        """窗口关闭前释放共享状态流与 HTTP 客户端。"""

        self._disconnect_requested()
        super().closeEvent(event)

    # endregion
    # region 连接构造

    def _create_connection_bundle(self) -> ConnectionBundle:
        started_at = time.monotonic()
        access_host = self._requested_host
        stage = "RobotControl Gateway HTTPS"
        logger.info("GUI connection begin: gateway_host={}", access_host)
        robot_control: RobotControlClient | None = None
        status_stream: RobotControlStatusStream | None = None
        camera_client: CameraPipelineHttpClient | None = None
        try:
            gateway_base_url = f"https://{access_host}"
            robot_control = RobotControlClient(
                base_url=gateway_base_url,
                api_prefix=DEFAULT_ROBOT_CONTROL_API_PREFIX,
                timeout_s=10.0,
            )
            health = robot_control.get_health()
            logger.info("RobotControl health: {}", health)
            status_stream = RobotControlStatusStream(robot_control)
            status_stream.start()
            status_stream.wait_ready(timeout_s=15.0)

            stage = "CameraPipeline Gateway HTTPS/WebSocket"
            logger.info("GUI connection stage begin: {}", stage)
            camera_client = CameraPipelineHttpClient(
                base_url=gateway_base_url,
                websocket_url=f"wss://{access_host}",
                api_prefix=DEFAULT_CAMERA_API_PREFIX,
                websocket_prefix=DEFAULT_CAMERA_WEBSOCKET_PREFIX,
                timeout_s=60.0,
                stream_timeout_s=30.0,
            )
            logger.info(
                "GUI connection stage ready: {} base_url={}",
                stage,
                gateway_base_url,
            )

            stage = "RobotControl HTTP 设备适配器"
            if robot_control is None or status_stream is None:
                raise RuntimeError("RobotControl HTTP 客户端未初始化")
            bundle = ConnectionBundle(
                robot_control=robot_control,
                status_stream=status_stream,
                left_arm=RobotControlAr5Client("left", robot_control, status_stream),
                right_arm=RobotControlAr5Client("right", robot_control, status_stream),
                agv=RobotControlAgvClient(robot_control, status_stream),
                head=RobotControlHeadClient(robot_control, status_stream),
                body=RobotControlBodyClient(robot_control, status_stream),
                gripper=RobotControlGripperClient(robot_control, status_stream),
                right_hand=RobotControlRightHandClient(robot_control, status_stream),
                camera=camera_client,
            )
            logger.info(
                "GUI connection completed: mode=gateway-https elapsed_s={:.3f}",
                time.monotonic() - started_at,
            )
            return bundle
        except Exception as exc:
            logger.error(
                "GUI connection stage failed: stage={} elapsed_s={:.3f}",
                stage,
                time.monotonic() - started_at,
            )
            if camera_client is not None:
                camera_client.close()
            if status_stream is not None:
                status_stream.close()
            raise RuntimeError(f"{stage}失败：{type(exc).__name__}: {exc}") from exc

    # endregion

    # region 状态分发

    def _apply_connection_state(self, connected: bool, message: str) -> None:
        self.connection_indicator.set_status(connected)
        self.connection_label.setText(message)
        self.connect_button.setEnabled(not connected)
        self.disconnect_button.setEnabled(connected)
        self.host_edit.setEnabled(not connected)
        for tab in self._tabs:
            tab.set_connection_ready(connected)
        self.algo_tab.set_connection_ready(connected)
        self.statusBar().showMessage(message)

    @Slot(int)
    def _on_current_tab_changed(self, index: int) -> None:
        current_widget = self.tab_widget.widget(index)
        for tab in self._tabs:
            tab.set_active(tab is current_widget)
        if current_widget is self.camera_tab:
            self._camera_bridge.activate()
            self.camera_tab.activate_default_camera()
        elif current_widget is self.prior_calibration_tab:
            pass
        else:
            self._camera_bridge.stop_stream()

    @Slot(object, int)
    def _on_camera_frame_ready(self, frame: object, run_id: int) -> None:
        _ = run_id
        if isinstance(frame, WujiCameraFrame):
            try:
                current_widget = self.tab_widget.currentWidget()
                if current_widget is self.camera_tab:
                    self.camera_tab.update_frame(frame)
                elif current_widget is self.prior_calibration_tab:
                    self.prior_calibration_tab.update_frame(frame)
            except Exception as exc:
                self._on_camera_error(f"相机画面刷新失败：{type(exc).__name__}: {exc}")

    @Slot(str)
    def _on_camera_error(self, message: str) -> None:
        """将相机错误限制在当前相机相关页，避免污染主窗口连接状态。"""

        logger.error("GUI camera error: {}", message)
        current_widget = self.tab_widget.currentWidget()
        if current_widget is self.camera_tab:
            self.camera_tab.show_camera_error(message)
        elif current_widget is self.prior_calibration_tab:
            self.prior_calibration_tab.show_camera_error(message)
        else:
            logger.warning("Camera error received while another tab is active: {}", message)

    @Slot(str)
    def _show_status_message(self, message: str) -> None:
        logger.error("GUI error: {}", message)
        self.connection_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot()
    def _update_deployment_host(self) -> None:
        host = self.host_edit.text().strip()
        if host:
            self.deployment_tab.set_service_host(host)

    # endregion
