from __future__ import annotations

import ipaddress
import platform
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

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
from qmlinker import create_channel

from camera_pipeline.client import CameraPipelineClient
from gui.test.agv_tab import AgvTabWidget
from gui.test.algo_tab import AlgoPlaceholderTabWidget
from gui.test.arm_tab import ArmTabWidget
from gui.test.body_tab import BodyTabWidget
from gui.test.camera_bridge import CameraBridge
from gui.test.camera_tab import WujiCameraTabWidget
from gui.test.common import ActivatableTab, BackgroundCall
from gui.test.deployment_tab import DeploymentTabWidget
from gui.test.gripper_tab import GripperTabWidget
from gui.test.head_tab import HeadTabWidget
from gui.test.prior_calibration_tab import PriorCalibrationTabWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji.agv_client import WujiAgvClient
from src.wuji.ar5_client import (
    AR5_DIRECT_IPS,
    AR5_SSH_FORWARD_PORTS,
    AR5_TUNNEL_IPS,
    Ar5Client,
    Ar5ConnectionConfig,
)
from src.wuji.body_client import WujiBodyClient
from src.wuji.camera_protocol import WujiCameraFrame
from src.wuji.dahuan_gripper_client import DahuanGripperClient
from src.wuji.head_client import WujiHeadClient
from src.wuji.prior_calibration import PriorCalibrationRecorder
from src.wuji.qmlinker_session import WujiQmlinkerSession, WujiSshForward
from src.wuji.zmq_camera_catalog import (
    SUPPORTED_WUJI_ZMQ_CAMERAS,
    SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
)
from src.wuji.zmq_camera_client import WujiZmqCameraClient

DEFAULT_ORIN_IP = "192.168.100.50"
DEFAULT_CONTROL_IP = "192.168.100.60"
DEFAULT_AGV_IP = "192.168.100.70"
DEFAULT_QMLINKER_PORT = 50062
DEFAULT_CAMERA_CONTROL_PORT = 5570
DEFAULT_CAMERA_PIPELINE_PORT = 6200
DEFAULT_GRIPPER_PORT = 50066
DEFAULT_CAMERA_STREAM_TIMEOUT_MS = 2000
SETTINGS_ORGANIZATION = "DingTai"
SETTINGS_APPLICATION = "WujiTouchGui"
SETTINGS_LAST_IP_KEY = "connection/last_ip"
SETTINGS_LEFT_ARM_IP_KEY = "connection/left_arm_ip"
SETTINGS_RIGHT_ARM_IP_KEY = "connection/right_arm_ip"


@dataclass(slots=True)
class ConnectionBundle:
    """GUI 全部设备连接及其统一释放顺序。"""

    session: WujiQmlinkerSession
    direct: bool
    left_arm: Ar5Client
    right_arm: Ar5Client
    agv: WujiAgvClient
    head: WujiHeadClient
    body: WujiBodyClient
    gripper: DahuanGripperClient
    camera: WujiZmqCameraClient
    camera_pipeline: CameraPipelineClient

    def close(self) -> None:
        """停止机械臂、关闭相机并释放 SSH 转发。"""

        for arm in (self.left_arm, self.right_arm):
            try:
                arm.close()
            except Exception:
                pass
        try:
            self.camera.close()
        except Exception:
            pass
        try:
            self.camera_pipeline.close()
        except Exception:
            pass
        self.session.close()


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
        self._requested_host = DEFAULT_ORIN_IP
        self._requested_left_arm_ip = AR5_DIRECT_IPS["left"]
        self._requested_right_arm_ip = AR5_DIRECT_IPS["right"]
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
        self.host_edit.setPlaceholderText("Orin 接入 IP（本机可输入 DHCP 地址）")
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
        self.camera_tab = WujiCameraTabWidget(self.tab_widget)
        self.prior_calibration_tab = PriorCalibrationTabWidget(self.tab_widget)
        self.algo_tab = AlgoPlaceholderTabWidget(self.tab_widget)
        for title, widget in (
            ("项目主页", self.deployment_tab),
            ("AGV", self.agv_tab),
            ("头部", self.head_tab),
            ("升降/腰部", self.body_tab),
            ("左 AR5", self.left_arm_tab),
            ("右 AR5", self.right_arm_tab),
            ("夹爪", self.gripper_tab),
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
            self.prior_calibration_tab,
            self.algo_tab,
        ]
        self.setStyleSheet(
            """
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
            """
        )
        self.statusBar().setSizeGripEnabled(False)

    def _connect_signals(self) -> None:
        self.connect_button.clicked.connect(self._connect_requested)
        self.disconnect_button.clicked.connect(self._disconnect_requested)
        self.tab_widget.currentChanged.connect(self._on_current_tab_changed)
        self.camera_tab.cameraSelected.connect(self._camera_bridge.refresh_camera)
        self.camera_tab.rgbdStreamRequested.connect(self._camera_bridge.start_rgbd_stream)
        self.camera_tab.streamStopRequested.connect(self._camera_bridge.stop_stream)
        self.prior_calibration_tab.cameraStreamRequested.connect(
            self._camera_bridge.start_rgb_stream
        )
        self.prior_calibration_tab.streamStopRequested.connect(
            self._camera_bridge.stop_stream
        )
        self._camera_bridge.inventoryReady.connect(self.camera_tab.update_camera_inventory)
        self._camera_bridge.enableStateReady.connect(self.camera_tab.update_camera_enable_state)
        self._camera_bridge.intrinsicsReady.connect(self.camera_tab.update_intrinsics)
        self._camera_bridge.frameReady.connect(self._on_camera_frame_ready)
        self._camera_bridge.errorRaised.connect(self._show_status_message)
        self._connection_worker.succeeded.connect(self._on_bundle_ready)
        self._connection_worker.failed.connect(self._on_bundle_failed)
        self.host_edit.editingFinished.connect(self._update_deployment_host)

    def _load_last_network_config(self) -> None:
        saved_host = str(self._settings.value(SETTINGS_LAST_IP_KEY, DEFAULT_ORIN_IP)).strip()
        saved_left_arm_ip = str(
            self._settings.value(SETTINGS_LEFT_ARM_IP_KEY, AR5_DIRECT_IPS["left"])
        ).strip()
        saved_right_arm_ip = str(
            self._settings.value(SETTINGS_RIGHT_ARM_IP_KEY, AR5_DIRECT_IPS["right"])
        ).strip()
        self.host_edit.setText(saved_host or DEFAULT_ORIN_IP)
        self.left_arm_tab.set_connection_ip(
            saved_left_arm_ip or AR5_DIRECT_IPS["left"]
        )
        self.right_arm_tab.set_connection_ip(
            saved_right_arm_ip or AR5_DIRECT_IPS["right"]
        )
        self._update_deployment_host()

    # endregion

    # region 连接生命周期

    @Slot()
    def _connect_requested(self) -> None:
        requested_host = self.host_edit.text().strip()
        requested_left_arm_ip = self.left_arm_tab.connection_ip()
        requested_right_arm_ip = self.right_arm_tab.connection_ip()
        invalid_label = _first_invalid_ip(
            (
                ("Orin 接入 IP", requested_host),
                ("左 AR5 IP", requested_left_arm_ip),
                ("右 AR5 IP", requested_right_arm_ip),
            )
        )
        if invalid_label is not None:
            self._apply_connection_state(False, f"连接失败：{invalid_label} 无效")
            return
        self._requested_host = requested_host
        self._requested_left_arm_ip = requested_left_arm_ip
        self._requested_right_arm_ip = requested_right_arm_ip
        self._settings.setValue(SETTINGS_LAST_IP_KEY, requested_host)
        self._settings.setValue(SETTINGS_LEFT_ARM_IP_KEY, requested_left_arm_ip)
        self._settings.setValue(SETTINGS_RIGHT_ARM_IP_KEY, requested_right_arm_ip)
        self.deployment_tab.set_service_host(requested_host)
        self._disconnect_requested()
        self.left_arm_tab.set_connection_ip_enabled(False)
        self.right_arm_tab.set_connection_ip_enabled(False)
        self._apply_connection_state(False, "正在检测网络并连接…")
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
        self.prior_calibration_tab.set_clients(
            payload.left_arm,
            payload.head,
            PriorCalibrationRecorder(payload.camera_pipeline),
        )
        self._camera_bridge.set_client(payload.camera)
        mode_text = "平板直连" if payload.direct else "本机 SSH 转发"
        self._apply_connection_state(True, f"已连接 · {mode_text}")
        self._on_current_tab_changed(self.tab_widget.currentIndex())
        logger.info("GUI connected: access_ip={} mode={}", self._requested_host, mode_text)

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
        """窗口关闭前释放 SDK、相机和 SSH 资源。"""

        self._disconnect_requested()
        super().closeEvent(event)

    # endregion

    # region 连接构造

    def _create_connection_bundle(self) -> ConnectionBundle:
        started_at = time.monotonic()
        access_host = self._requested_host
        stage = "接入 IP 探测"
        logger.info(
            "GUI connection begin: access_ip={} left_ar5_ip={} right_ar5_ip={}",
            access_host,
            self._requested_left_arm_ip,
            self._requested_right_arm_ip,
        )
        if not _ping_host(access_host):
            logger.error("GUI connection access ping failed: access_ip={}", access_host)
            raise RuntimeError(f"接入 IP 不可达：{access_host}")
        logger.info("GUI connection access ping passed: access_ip={}", access_host)

        direct = _ping_host(DEFAULT_CONTROL_IP)
        mode_text = "direct" if direct else "ssh"
        logger.info(
            "GUI connection mode selected: mode={} control_ip={} control_ping={}",
            mode_text,
            DEFAULT_CONTROL_IP,
            direct,
        )
        session: WujiQmlinkerSession | None = None
        left_arm: Ar5Client | None = None
        right_arm: Ar5Client | None = None
        camera_client: WujiZmqCameraClient | None = None
        camera_pipeline_client: CameraPipelineClient | None = None
        try:
            stage = "SSH 共享转发" if not direct else "直连会话"
            logger.info("GUI connection stage begin: {}", stage)
            session = WujiQmlinkerSession(
                host=DEFAULT_CONTROL_IP,
                port=DEFAULT_QMLINKER_PORT,
                direct=direct,
                ssh_host=None if direct else access_host,
                tunnel_forwards=() if direct else self._build_shared_tunnel_forwards(),
            )
            logger.info(
                "GUI connection stage ready: {} summary={}",
                stage,
                session.debug_connection_summary(),
            )

            stage = "qmlinker 通道就绪"
            logger.info("GUI connection stage begin: {}", stage)
            session.check_ready()
            logger.info("GUI connection stage ready: {}", stage)

            stage = "左 AR5 SDK 连接"
            left_sdk_ip = (
                self._requested_left_arm_ip if direct else AR5_TUNNEL_IPS["left"]
            )
            logger.info(
                "GUI connection stage begin: {} sdk_ip={} controller_ip={}",
                stage,
                left_sdk_ip,
                self._requested_left_arm_ip,
            )
            left_arm = Ar5Client(
                Ar5ConnectionConfig(
                    side="left",
                    robot_ip=left_sdk_ip,
                )
            )
            logger.info("GUI connection stage ready: {}", stage)

            stage = "右 AR5 SDK 连接"
            right_sdk_ip = (
                self._requested_right_arm_ip
                if direct
                else AR5_TUNNEL_IPS["right"]
            )
            logger.info(
                "GUI connection stage begin: {} sdk_ip={} controller_ip={}",
                stage,
                right_sdk_ip,
                self._requested_right_arm_ip,
            )
            right_arm = Ar5Client(
                Ar5ConnectionConfig(
                    side="right",
                    robot_ip=right_sdk_ip,
                )
            )
            logger.info("GUI connection stage ready: {}", stage)

            stage = "相机客户端创建"
            logger.info("GUI connection stage begin: {}", stage)
            camera_endpoints = (
                SUPPORTED_WUJI_ZMQ_CAMERAS
                if direct
                else SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL
            )
            camera_target = session.resolve_target(
                DEFAULT_CONTROL_IP,
                DEFAULT_CAMERA_CONTROL_PORT,
            )
            camera_host, camera_port = _parse_target(camera_target)
            camera_client = WujiZmqCameraClient(
                host=camera_host,
                control_port=camera_port,
                request_timeout_ms=max(500, int(session.request_timeout_s * 1000.0)),
                stream_timeout_ms=DEFAULT_CAMERA_STREAM_TIMEOUT_MS,
                camera_endpoints=camera_endpoints,
            )
            logger.info(
                "GUI connection stage ready: {} target={}:{}",
                stage,
                camera_host,
                camera_port,
            )

            stage = "CameraPipeline 客户端创建"
            logger.info("GUI connection stage begin: {}", stage)
            pipeline_target = session.resolve_target(
                DEFAULT_CONTROL_IP,
                DEFAULT_CAMERA_PIPELINE_PORT,
            )
            camera_pipeline_client = CameraPipelineClient(
                service_addr=f"tcp://{pipeline_target}",
                timeout_ms=60_000,
            )
            logger.info(
                "GUI connection stage ready: {} target={}",
                stage,
                pipeline_target,
            )

            stage = "AGV/头部/身体/夹爪客户端创建"
            logger.info("GUI connection stage begin: {}", stage)
            gripper_target = session.resolve_target(
                DEFAULT_CONTROL_IP,
                DEFAULT_GRIPPER_PORT,
            )
            bundle = ConnectionBundle(
                session=session,
                direct=direct,
                left_arm=left_arm,
                right_arm=right_arm,
                agv=WujiAgvClient(
                    create_channel(session.resolve_target(DEFAULT_AGV_IP, DEFAULT_QMLINKER_PORT)),
                    request_timeout_s=session.request_timeout_s,
                ),
                head=WujiHeadClient(session.channel),
                body=WujiBodyClient(session.channel),
                gripper=DahuanGripperClient(create_channel(gripper_target)),
                camera=camera_client,
                camera_pipeline=camera_pipeline_client,
            )
            logger.info(
                "GUI connection completed: mode={} elapsed_s={:.3f}",
                mode_text,
                time.monotonic() - started_at,
            )
            return bundle
        except Exception as exc:
            logger.exception(
                "GUI connection stage failed: stage={} mode={} elapsed_s={:.3f}",
                stage,
                mode_text,
                time.monotonic() - started_at,
            )
            if left_arm is not None:
                left_arm.close()
            if right_arm is not None:
                right_arm.close()
            if camera_client is not None:
                camera_client.close()
            if camera_pipeline_client is not None:
                camera_pipeline_client.close()
            if session is not None:
                session.close()
            raise RuntimeError(
                f"{stage}失败：{type(exc).__name__}: {exc}"
            ) from exc

    def _build_shared_tunnel_forwards(self) -> tuple[WujiSshForward, ...]:
        forwards = [
            WujiSshForward(
                local_host="127.0.0.1",
                local_port=DEFAULT_QMLINKER_PORT - 1,
                remote_host=DEFAULT_CONTROL_IP,
                remote_port=DEFAULT_QMLINKER_PORT,
            ),
            WujiSshForward(
                local_host="127.0.0.1",
                local_port=DEFAULT_QMLINKER_PORT + 1,
                remote_host=DEFAULT_AGV_IP,
                remote_port=DEFAULT_QMLINKER_PORT,
            ),
            WujiSshForward(
                local_host="127.0.0.1",
                local_port=DEFAULT_GRIPPER_PORT - 1,
                remote_host=DEFAULT_CONTROL_IP,
                remote_port=DEFAULT_GRIPPER_PORT,
            ),
            WujiSshForward(
                local_host="127.0.0.1",
                local_port=DEFAULT_CAMERA_CONTROL_PORT - 1,
                remote_host=DEFAULT_CONTROL_IP,
                remote_port=DEFAULT_CAMERA_CONTROL_PORT,
            ),
            WujiSshForward(
                local_host="127.0.0.1",
                local_port=DEFAULT_CAMERA_PIPELINE_PORT - 1,
                remote_host=DEFAULT_CONTROL_IP,
                remote_port=DEFAULT_CAMERA_PIPELINE_PORT,
            ),
        ]
        arm_remote_ips = {
            "left": self._requested_left_arm_ip,
            "right": self._requested_right_arm_ip,
        }
        for side in ("left", "right"):
            for port in AR5_SSH_FORWARD_PORTS:
                forwards.append(
                    WujiSshForward(
                        local_host=AR5_TUNNEL_IPS[side],
                        local_port=port,
                        remote_host=arm_remote_ips[side],
                        remote_port=port,
                    )
                )
        for endpoint in SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL:
            forwards.append(
                WujiSshForward(
                    local_host="127.0.0.1",
                    local_port=endpoint.stream_port,
                    remote_host=DEFAULT_CONTROL_IP,
                    remote_port=endpoint.stream_port + 1,
                )
            )
        return tuple(forwards)

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
            current_widget = self.tab_widget.currentWidget()
            if current_widget is self.camera_tab:
                self.camera_tab.update_frame(frame)
            elif current_widget is self.prior_calibration_tab:
                self.prior_calibration_tab.update_frame(frame)

    @Slot(str)
    def _show_status_message(self, message: str) -> None:
        self.connection_label.setText(message)
        self.statusBar().showMessage(message)

    @Slot()
    def _update_deployment_host(self) -> None:
        host = self.host_edit.text().strip()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            return
        self.deployment_tab.set_service_host(host)

    # endregion


def _ping_host(host: str, timeout_ms: int = 1200) -> bool:
    """跨 Windows/Linux 探测单个 IP 是否可达。"""

    if platform.system() == "Windows":
        command = ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:
        timeout_s = max(1, int(round(timeout_ms / 1000.0)))
        command = ["ping", "-c", "1", "-W", str(timeout_s), host]
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=max(2.0, timeout_ms / 1000.0 + 1.0),
    )
    return completed.returncode == 0


def _parse_target(target: str) -> tuple[str, int]:
    """解析 ``host:port`` 连接目标。"""

    parsed = urlsplit(f"tcp://{target}")
    if parsed.hostname is None or parsed.port is None:
        raise RuntimeError(f"无效连接目标：{target}")
    return parsed.hostname, parsed.port


def _first_invalid_ip(items: tuple[tuple[str, str], ...]) -> str | None:
    for label, value in items:
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return label
    return None
