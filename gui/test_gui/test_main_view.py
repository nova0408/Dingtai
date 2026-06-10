from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Any, Literal, cast

import numpy as np
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
from gui.test_gui.test_wuji_gripper_tab import WujiGripperTabWidget
from gui.test_gui.test_wuji_pose_tab import LEFT_CAMERA_NAME, WujiPoseTabWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.arm.wuji_arm_protocol import ArmDeviceName
from src.wuji.dahuan_gripper_backend import DahuanGripperBackend
from src.wuji.dahuan_gripper_client import DahuanGripperInfo
from src.wuji.protocol import load_wuji_robot_network_config
from src.wuji.backend import WujiQmlinkerBackend

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
        self._gripper_backend = DahuanGripperBackend(self)
        self._state_refresh_timer = QTimer(self)
        self._state_refresh_timer.setInterval(50)
        self._service_status_timer = QTimer(self)
        self._service_status_timer.setInterval(1000)
        self._gripper_refresh_timer = QTimer(self)
        self._gripper_refresh_timer.setInterval(1000)
        self.gripper_widget: WujiGripperTabWidget | None = None

        self._setup_service_context_editor()
        self._setup_robot_tab()
        self._setup_camera_tab()
        self._setup_pose_tab()
        self._connect_signals()
        self._refresh_service_status_ui()
        self._service_status_timer.start()
        self._gripper_backend.probe_gripper()

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

    def _setup_pose_tab(self) -> None:
        self.pose_tab = QWidget(self.ui.tabWidget)
        orin_service_addr = "tcp://{0}:6220".format(load_wuji_robot_network_config().orin_ip)
        self.pose_widget = WujiPoseTabWidget(self.pose_tab, service_addr=orin_service_addr)
        layout = QHBoxLayout(self.pose_tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pose_widget)
        self.ui.tabWidget.addTab(self.pose_tab, "pose")

    def _connect_signals(self) -> None:
        self.ui.pushButton.clicked.connect(self._on_connect_button_clicked)
        self.ui.pushButton_2.clicked.connect(self._on_disconnect_button_clicked)
        self.robot_widget.dofTargetRequested.connect(self._on_dof_target_requested)
        self.robot_widget.enableToggleRequested.connect(self._on_enable_toggle_requested)
        self.robot_widget.enableStateRefreshRequested.connect(
            self._on_enable_state_refresh_requested
        )
        self.robot_widget.armIkRequested.connect(self._on_arm_ik_requested)
        self.robot_widget.armStopRequested.connect(self._on_arm_stop_requested)
        self.robot_widget.rightHandEnableRequested.connect(
            lambda enabled: self._arm_backend.set_right_hand_enable_state(enabled)
        )
        self.robot_widget.rightHandPoseRequested.connect(lambda: self._arm_backend.set_right_hand_demo_pose())
        self.robot_widget.rightHandAxisTargetRequested.connect(
            lambda axis_name, value: self._arm_backend.set_dof_target(axis_name, value)
        )
        self.robot_widget.agvEnableRequested.connect(self._arm_backend.set_agv_enable_state)
        self.robot_widget.agvMoveRequested.connect(self._on_agv_move_requested)
        self.robot_widget.agvNavigateRequested.connect(self._on_agv_navigate_requested)
        self.robot_widget.agvChargeRequested.connect(self._on_agv_charge_requested)
        self.robot_widget.agvStopRequested.connect(self._arm_backend.agv_stop)
        self.camera_widget.cameraSelected.connect(self._on_camera_selected)
        self.camera_widget.cameraEnableToggleRequested.connect(self._on_camera_enable_toggle_requested)
        self.camera_widget.rgbStreamRequested.connect(self._on_camera_rgb_stream_requested)
        self.camera_widget.rgbdStreamRequested.connect(self._on_camera_rgbd_stream_requested)
        self.camera_widget.streamStopRequested.connect(self._arm_backend.stop_camera_stream)
        self.ui.tabWidget.currentChanged.connect(self._on_main_tab_current_changed)
        self._state_refresh_timer.timeout.connect(self._on_state_refresh_timer_timeout)
        self._service_status_timer.timeout.connect(self._on_service_status_timer_timeout)
        self._gripper_refresh_timer.timeout.connect(self._on_gripper_refresh_timer_timeout)
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
        self._gripper_backend.availabilityResolved.connect(self._on_gripper_availability_resolved)
        self._gripper_backend.gripperInfoReceived.connect(self._on_gripper_info_received)
        self._gripper_backend.requestFailed.connect(self._on_backend_request_failed)

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

    @Slot(str)
    def _on_agv_move_requested(self, direction: str) -> None:
        self._arm_backend.agv_move_direction(direction)
        self.statusBar().showMessage(f"AGV move requested: direction={direction}")

    @Slot(str)
    def _on_agv_navigate_requested(self, target_name: str) -> None:
        self._arm_backend.agv_navigate_to(target_name)
        self.statusBar().showMessage(f"AGV navigate requested: {target_name}")

    @Slot()
    def _on_agv_charge_requested(self) -> None:
        self._arm_backend.agv_navigate_to_charge()
        self.statusBar().showMessage("AGV charge requested")

    @Slot(str, tuple, tuple)
    def _on_arm_ik_requested(self, device_name: str, target_pose: tuple[float, ...], reference_joints: tuple[float, ...]) -> None:
        typed_device = self._as_arm_device_name(device_name)
        refreshed_joints = self._refresh_arm_reference_joints(typed_device)
        if len(refreshed_joints) != 6:
            refreshed_joints = [float(value) for value in reference_joints]
        if len(refreshed_joints) != 6:
            self.statusBar().showMessage(f"Arm IK requested failed: invalid reference joints for {device_name}")
            return

        try:
            target_pose_matrix = _gui_pose_to_matrix_m(target_pose)
        except Exception as exc:  # noqa: BLE001
            logger.error("Arm IK pose conversion failed for {}: {}", device_name, exc)
            self.statusBar().showMessage(f"Arm IK requested failed: invalid pose for {device_name}")
            return

        ik_result = self._arm_backend.ik(
            typed_device,
            target_pose_matrix,
            np.deg2rad(np.asarray(refreshed_joints, dtype=np.float64)),
        )
        try:
            ik_result_array = np.asarray(ik_result, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Arm IK result conversion failed for {}: {}", device_name, exc)
            self.robot_widget.update_arm_ik_result(typed_device, None)
        else:
            if ik_result_array.size == 6:
                ik_angles_deg = tuple(float(value) for value in np.rad2deg(ik_result_array.reshape(-1)))
                self.robot_widget.update_arm_ik_result(typed_device, ik_angles_deg)
                self._arm_backend.set_joints(typed_device, ik_angles_deg)
            else:
                self.robot_widget.update_arm_ik_result(typed_device, None)
        self.statusBar().showMessage(f"Arm IK requested: {device_name}")

    @Slot(str)
    def _on_arm_stop_requested(self, device_name: str) -> None:
        typed_device = self._as_arm_device_name(device_name)
        joints = self._arm_backend.get_joint_states(typed_device)
        joint_list = list(cast(Any, joints))
        joint_angles = [float(getattr(joint, "angle_deg", 0.0)) for joint in joint_list]
        self._arm_backend.set_joints(typed_device, joint_angles)
        self.statusBar().showMessage(f"Arm stop requested: {device_name}")

    @Slot(int)
    def _on_main_tab_current_changed(self, index: int) -> None:
        self.pose_widget.set_active(self.ui.tabWidget.widget(index) is self.pose_tab)
        if self.ui.tabWidget.widget(index) is self.ui.image_tab:
            self._arm_backend.refresh_camera_inventory()
            self.camera_widget.activate_default_camera()
        if self.ui.tabWidget.widget(index) is self.pose_tab:
            self.statusBar().showMessage(f"Pose tab activated: {LEFT_CAMERA_NAME} via pose context")

    @Slot(bool, str)
    def _on_gripper_availability_resolved(self, available: bool, message: str) -> None:
        if not available:
            logger.warning("left gripper unavailable: {}", message)
            self.statusBar().showMessage(message)
            return
        if self.gripper_widget is None:
            self.gripper_widget = WujiGripperTabWidget(self.robot_widget)
            self.gripper_widget.refreshRequested.connect(self._gripper_backend.refresh_gripper_info)
            self.gripper_widget.enableRequested.connect(self._gripper_backend.set_gripper_enable)
            self.gripper_widget.positionRequested.connect(self._gripper_backend.set_gripper_position)
            self.gripper_widget.speedRequested.connect(self._gripper_backend.set_gripper_speed)
            self.gripper_widget.forceRequested.connect(self._gripper_backend.set_gripper_force)
            self.gripper_widget.calibrateRequested.connect(self._gripper_backend.calibrate_gripper)
            self.robot_widget.ensure_extra_tab("gripper", "Left Gripper", self.gripper_widget)
        self._gripper_refresh_timer.start()
        self.statusBar().showMessage(message)

    @Slot(object)
    def _on_gripper_info_received(self, payload: object) -> None:
        if self.gripper_widget is None:
            return
        if not isinstance(payload, DahuanGripperInfo):
            logger.error("TestMainView gripper payload type error: {}", type(payload).__name__)
            return
        self.gripper_widget.update_info(payload)

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
        for device_name in ("left_arm", "right_arm"):
            typed_device = self._as_arm_device_name(device_name)
            try:
                pose = self._arm_backend.current_fk_fast(typed_device)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Arm FK refresh failed for {}: {}", device_name, exc)
            else:
                self.robot_widget.update_arm_fk_pose(typed_device, pose)

    @Slot()
    def _on_service_status_timer_timeout(self) -> None:
        self._sync_context_from_ui()
        if not self.context.connected:
            return
        self.serviceStateRefreshRequested.emit(self.context)

    @Slot()
    def _on_gripper_refresh_timer_timeout(self) -> None:
        if self.gripper_widget is None:
            self._gripper_refresh_timer.stop()
            return
        self._gripper_backend.refresh_gripper_info()

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

    def closeEvent(self, event) -> None:  # noqa: N802
        """关闭窗口时释放 gripper 后端持有的基础客户端。"""

        self._gripper_backend.close()
        super().closeEvent(event)

    def _refresh_arm_reference_joints(self, device_name: ArmDeviceName) -> list[float]:
        """读取机械臂当前关节角，作为 IK 的实时参考值。"""

        joints = self._arm_backend.get_joint_states(device_name)
        joint_list = list(cast(Any, joints))
        current_angles = [float(getattr(joint, "angle_deg", 0.0)) for joint in joint_list]
        if len(current_angles) != 6:
            logger.warning("Arm IK reference refresh failed: {} joint_count={}", device_name, len(current_angles))
            return []
        return current_angles

    @staticmethod
    def _as_arm_device_name(device_name: str) -> ArmDeviceName:
        if device_name == "left_arm":
            return "left_arm"
        if device_name == "right_arm":
            return "right_arm"
        raise ValueError(f"unsupported arm device: {device_name}")


def _gui_pose_to_matrix_m(pose_values: tuple[float, ...]) -> np.ndarray:
    """将 GUI 输入的 mm + deg 位姿转换为接口要求的 m + rad 矩阵。"""

    if len(pose_values) != 6:
        raise ValueError(f"expected 6 pose values, got {len(pose_values)}")

    x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = (float(value) for value in pose_values)
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(roll_rad), -sin(roll_rad)],
            [0.0, sin(roll_rad), cos(roll_rad)],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cos(pitch_rad), 0.0, sin(pitch_rad)],
            [0.0, 1.0, 0.0],
            [-sin(pitch_rad), 0.0, cos(pitch_rad)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [cos(yaw_rad), -sin(yaw_rad), 0.0],
            [sin(yaw_rad), cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    pose_matrix = np.eye(4, dtype=np.float64)
    pose_matrix[:3, :3] = rz @ ry @ rx
    pose_matrix[0, 3] = x_mm / 1000.0
    pose_matrix[1, 3] = y_mm / 1000.0
    pose_matrix[2, 3] = z_mm / 1000.0
    return pose_matrix


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = TestMainView()
    window.show()
    sys.exit(app.exec())

