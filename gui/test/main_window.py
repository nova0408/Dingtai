from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.test.algo_tab import AlgoPlaceholderTabWidget
from gui.test.agv_tab import AgvTabWidget
from gui.test.arm_tab import ArmTabWidget
from gui.test.body_tab import BodyTabWidget
from gui.test.camera_bridge import CameraBridge
from gui.test.common import ActivatableTab
from gui.test.gripper_tab import GripperTabWidget
from gui.test.hand_tab import M11HandTabWidget
from gui.test.head_tab import HeadTabWidget
from gui.test.camera_tab import WujiCameraTabWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji.camera_protocol import WujiCameraFrame
from src.wuji.arm_client import WujiArmClient
from src.wuji.body_client import WujiBodyClient
from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.dahuan_gripper_client import DahuanGripperClient
from src.wuji.head_client import WujiHeadClient
from src.wuji.protocol import WujiQmlinkerConfig, load_wuji_robot_network_config
from src.wuji.right_hand_client import WujiRightHandClient
from src.wuji.zmq_camera_client import WujiZmqCameraClient, WujiZmqCameraConfig


@dataclass(slots=True)
class ConnectionBundle:
    base: WujiQmlinkerBaseClient
    head: WujiHeadClient
    body: WujiBodyClient
    left_arm: WujiArmClient
    right_arm: WujiArmClient
    gripper: DahuanGripperClient
    hand: WujiRightHandClient
    camera: WujiZmqCameraClient

    def close(self) -> None:
        try:
            self.left_arm.stop()
        except Exception:
            pass
        try:
            self.right_arm.stop()
        except Exception:
            pass
        try:
            self.camera.close()
        except Exception:
            pass
        self.base.close()


class TestGuiMainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bundle: ConnectionBundle | None = None
        self._tabs: list[ActivatableTab] = []
        self._camera_bridge = CameraBridge(self)
        self._build_ui()
        self._connect_signals()
        self._load_default_network_config()
        self._apply_connection_state(False, "未连接")

    def _build_ui(self) -> None:
        self.setWindowTitle("Wuji Test GUI")
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        connect_row = QHBoxLayout()
        self.host_edit = QLineEdit(central)
        self.port_spin = QSpinBox(central)
        self.port_spin.setRange(1, 65535)
        self.connect_button = QPushButton("Connect", central)
        self.disconnect_button = QPushButton("Disconnect", central)
        self.connection_indicator = CasiaIndicatorLight(
            central,
            text=("已连接", "未连接"),
            font_size=12,
            default_status=False,
        )
        self.connection_label = QLabel("未连接", central)
        connect_row.addWidget(QLabel("host", central))
        connect_row.addWidget(self.host_edit, 1)
        connect_row.addWidget(QLabel("port", central))
        connect_row.addWidget(self.port_spin)
        connect_row.addWidget(self.connect_button)
        connect_row.addWidget(self.disconnect_button)
        connect_row.addWidget(self.connection_indicator)
        connect_row.addWidget(self.connection_label, 1)
        root_layout.addLayout(connect_row)

        self.tab_widget = QTabWidget(central)
        self.agv_tab = AgvTabWidget(self.tab_widget)
        self.head_tab = HeadTabWidget(self.tab_widget)
        self.body_tab = BodyTabWidget(self.tab_widget)
        self.left_arm_tab = ArmTabWidget("left_arm", "Left Arm", self.tab_widget)
        self.right_arm_tab = ArmTabWidget("right_arm", "Right Arm", self.tab_widget)
        self.gripper_tab = GripperTabWidget(self.tab_widget)
        self.hand_tab = M11HandTabWidget(self.tab_widget)
        self.camera_tab = WujiCameraTabWidget(self.tab_widget)
        self.algo_tab = AlgoPlaceholderTabWidget(self.tab_widget)
        for title, widget in (
            ("agv", self.agv_tab),
            ("head", self.head_tab),
            ("body", self.body_tab),
            ("left arm", self.left_arm_tab),
            ("right arm", self.right_arm_tab),
            ("gripper", self.gripper_tab),
            ("m11 hand", self.hand_tab),
            ("camera", self.camera_tab),
            ("algo", self.algo_tab),
        ):
            self.tab_widget.addTab(widget, title)
        root_layout.addWidget(self.tab_widget, 1)

        self._tabs = [
            self.agv_tab,
            self.head_tab,
            self.body_tab,
            self.left_arm_tab,
            self.right_arm_tab,
            self.gripper_tab,
            self.hand_tab,
            self.algo_tab,
        ]

    def _connect_signals(self) -> None:
        self.connect_button.clicked.connect(self._connect_requested)
        self.disconnect_button.clicked.connect(self._disconnect_requested)
        self.tab_widget.currentChanged.connect(self._on_current_tab_changed)
        self.camera_tab.cameraSelected.connect(self._camera_bridge.refresh_camera)
        self.camera_tab.cameraEnableToggleRequested.connect(self._camera_bridge.set_enable)
        self.camera_tab.rgbStreamRequested.connect(self._camera_bridge.start_rgb_stream)
        self.camera_tab.rgbdStreamRequested.connect(self._camera_bridge.start_rgbd_stream)
        self.camera_tab.streamStopRequested.connect(self._camera_bridge.stop_stream)
        self._camera_bridge.inventoryReady.connect(self.camera_tab.update_camera_inventory)
        self._camera_bridge.enableStateReady.connect(self.camera_tab.update_camera_enable_state)
        self._camera_bridge.intrinsicsReady.connect(self.camera_tab.update_intrinsics)
        self._camera_bridge.frameReady.connect(self._on_camera_frame_ready)
        self._camera_bridge.errorRaised.connect(self._show_status_message)

    def _load_default_network_config(self) -> None:
        config = load_wuji_robot_network_config().qmlinker
        self.host_edit.setText(config.host)
        self.port_spin.setValue(int(config.port))

    def _connect_requested(self) -> None:
        self._disconnect_requested()
        try:
            bundle = self._create_connection_bundle()
            self._bundle = bundle
            self.agv_tab.set_client(bundle.base)
            self.head_tab.set_client(bundle.head)
            self.body_tab.set_client(bundle.body)
            self.left_arm_tab.set_client(bundle.left_arm)
            self.right_arm_tab.set_client(bundle.right_arm)
            self.gripper_tab.set_client(bundle.gripper)
            self.hand_tab.set_client(bundle.hand)
            self._camera_bridge.set_client(bundle.camera)
            self._apply_connection_state(True, "qmlinker connected")
            self._on_current_tab_changed(self.tab_widget.currentIndex())
            logger.info("new test gui connected: host={} port={}", self.host_edit.text().strip(), self.port_spin.value())
        except Exception as exc:  # noqa: BLE001
            logger.error("new test gui connect failed: {}", exc)
            self._disconnect_requested()
            self._apply_connection_state(False, f"连接失败: {exc}")

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
        bundle = self._bundle
        self._bundle = None
        if bundle is not None:
            try:
                bundle.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("bundle close failed: {}", exc)
        self._apply_connection_state(False, "已断开")

    def _create_connection_bundle(self) -> ConnectionBundle:
        runtime_config = load_wuji_robot_network_config()
        qmlinker_config = WujiQmlinkerConfig(
            host=self.host_edit.text().strip(),
            port=int(self.port_spin.value()),
            default_speed_ratio=runtime_config.qmlinker.default_speed_ratio,
            request_timeout_s=runtime_config.qmlinker.request_timeout_s,
            stream_first_timeout_s=runtime_config.qmlinker.stream_first_timeout_s,
        )
        base_client = WujiQmlinkerBaseClient(qmlinker_config)
        base_client.check_ready()
        camera_client = WujiZmqCameraClient(
            WujiZmqCameraConfig(
                host=qmlinker_config.host,
                request_timeout_ms=max(500, int(qmlinker_config.request_timeout_s * 1000.0)),
                stream_timeout_ms=max(500, int(qmlinker_config.stream_first_timeout_s * 1000.0)),
            )
        )
        return ConnectionBundle(
            base=base_client,
            head=WujiHeadClient(base_client),
            body=WujiBodyClient(base_client),
            left_arm=WujiArmClient(base_client, "left_arm"),
            right_arm=WujiArmClient(base_client, "right_arm"),
            gripper=DahuanGripperClient(base_client),
            hand=WujiRightHandClient(base_client),
            camera=camera_client,
        )

    def _apply_connection_state(self, connected: bool, message: str) -> None:
        self.connection_indicator.set_status(bool(connected))
        self.connection_label.setText(message)
        for tab in self._tabs:
            tab.set_connection_ready(bool(connected))
        self.algo_tab.set_connection_ready(bool(connected))
        self.statusBar().showMessage(message)

    def _on_current_tab_changed(self, index: int) -> None:
        current_widget = self.tab_widget.widget(index)
        for tab in self._tabs:
            tab.set_active(tab is current_widget)
        if current_widget is self.camera_tab:
            self._camera_bridge.activate()
            self.camera_tab.activate_default_camera()
        else:
            self._camera_bridge.stop_stream()

    def _on_camera_frame_ready(self, frame: object, run_id: int) -> None:
        _ = run_id
        if isinstance(frame, WujiCameraFrame):
            self.camera_tab.update_frame(frame)

    def _show_status_message(self, message: str) -> None:
        self.connection_label.setText(message)
        self.statusBar().showMessage(message)
