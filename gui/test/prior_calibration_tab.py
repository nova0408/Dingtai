from __future__ import annotations

import numpy as np
from loguru import logger
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.test.arm_tab import ArmTabWidget
from gui.test.camera_tab import ImagePreviewLabel
from gui.test.common import ActivatableTab, BackgroundCall
from gui.util_components.casia_indicator_light import (
    CasiaMultiStateIndicator,
    IndicatorState,
)
from src.wuji.ar5_client import Ar5Client, Ar5Snapshot
from src.wuji.camera_protocol import WujiCameraFrame
from src.wuji.head_client import WujiHeadClient
from src.wuji.prior_calibration import (
    BALL_CAMERA_NAME,
    DEFAULT_BALL_COLORS,
    HEAD_CAMERA_NAME,
    PriorBallSampleProgress,
    PriorCalibrationRecorder,
    PriorCalibrationResult,
)


class PriorCalibrationTabWidget(QWidget, ActivatableTab):
    """左臂三球与头部 ChArUco 先验标定页。"""

    REFRESH_INTERVAL_MS = 250
    HEAD_YAW_DEG = 60.0
    HEAD_PITCH_DEG = 45.0
    HEAD_POSITION_TOLERANCE_DEG = 1.0
    DRAG_STATES = (
        IndicatorState("off", "关闭", "#607d8b"),
        IndicatorState("drag", "拖动", "#00897b"),
        IndicatorState("unknown", "未知", "#607d8b"),
    )

    ballSampleReady = Signal(object)
    cameraStreamRequested = Signal(str)
    streamStopRequested = Signal()

    # region 初始化

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._arm: Ar5Client | None = None
        self._head: WujiHeadClient | None = None
        self._recorder: PriorCalibrationRecorder | None = None
        self._snapshot: Ar5Snapshot | None = None
        self._active = False
        self._refresh_busy = False
        self._action_busy = False
        self._prior_busy = False
        self._camera_name = BALL_CAMERA_NAME
        self._last_frame_bgr: np.ndarray | None = None
        self._ball_overlay_bgr: np.ndarray | None = None
        self._head_overlay_bgr: np.ndarray | None = None
        self._joint_targets_initialized = False
        self._ball_colors: list[str] = list(DEFAULT_BALL_COLORS)
        self._ball_color_buttons: list[QPushButton] = []
        self._refresh_timer = QTimer(self)
        self._refresh_call = BackgroundCall(self)
        self._action_call = BackgroundCall(self)
        self._prior_call = BackgroundCall(self)
        self._joint_boxes: list[QDoubleSpinBox] = []
        self._joint_current_labels: list[QLabel] = []
        self._control_widgets: list[QWidget] = []
        self._setup_ui()
        self._setup_timer()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(10)

        preview_group = QGroupBox("标定相机预览", self)
        preview_layout = QVBoxLayout(preview_group)
        self.camera_title_label = QLabel("左臂相机 · left_hand_camera", preview_group)
        self.camera_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview = ImagePreviewLabel("等待左臂相机图像", preview_group)
        preview_layout.addWidget(self.camera_title_label)
        preview_layout.addWidget(self.preview, 1)
        preview_group.setMinimumWidth(340)
        root_layout.addWidget(preview_group, 1)

        controls = QWidget(self)
        controls.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self._build_arm_group())
        controls_layout.addWidget(self._build_prior_group())
        controls_layout.addWidget(self._build_head_group())
        self.status_label = QLabel("等待连接", controls)
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setMinimumWidth(390)
        scroll.setWidget(controls)
        root_layout.addWidget(scroll, 1)

    def _setup_timer(self) -> None:
        self._refresh_timer.setInterval(self.REFRESH_INTERVAL_MS)
        self._refresh_timer.timeout.connect(self._request_refresh)

    def _connect_signals(self) -> None:
        self.mode_indicator.clicked.connect(self._toggle_mode)
        self.power_indicator.clicked.connect(self._toggle_power)
        self.drag_indicator.clicked.connect(self._toggle_drag)
        self.move_button.clicked.connect(self._move_joints)
        self.record_ball_button.clicked.connect(self._record_ball_prior)
        self.ball_overlay_checkbox.toggled.connect(self._refresh_preview)
        self.head_camera_button.clicked.connect(self._toggle_head_camera)
        self.head_yaw_button.clicked.connect(self._move_head_yaw)
        self.head_pitch_button.clicked.connect(self._move_head_pitch)
        self.record_head_button.clicked.connect(self._record_head_prior)
        self.head_overlay_checkbox.toggled.connect(self._refresh_preview)
        self.ballSampleReady.connect(self._on_ball_sample_ready)
        self._refresh_call.succeeded.connect(self._on_refresh_succeeded)
        self._refresh_call.failed.connect(self._on_refresh_failed)
        self._refresh_call.finished.connect(self._on_refresh_finished)
        self._action_call.succeeded.connect(self._on_action_succeeded)
        self._action_call.failed.connect(self._on_action_failed)
        self._action_call.finished.connect(self._on_action_finished)
        self._prior_call.succeeded.connect(self._on_prior_succeeded)
        self._prior_call.failed.connect(self._on_prior_failed)
        self._prior_call.finished.connect(self._on_prior_finished)

    def _build_arm_group(self) -> QGroupBox:
        group = QGroupBox("左 AR5 姿态准备", self)
        layout = QVBoxLayout(group)
        status_layout = QGridLayout()
        self.mode_indicator = CasiaMultiStateIndicator(
            "工作模式",
            ArmTabWidget.WORK_MODE_STATES,
            group,
        )
        self.power_indicator = CasiaMultiStateIndicator(
            "使能状态",
            ArmTabWidget.POWER_STATES,
            group,
        )
        self.operation_indicator = CasiaMultiStateIndicator(
            "机器人状态",
            ArmTabWidget.OPERATION_STATES,
            group,
            interactive=False,
        )
        self.drag_indicator = CasiaMultiStateIndicator(
            "拖动状态",
            self.DRAG_STATES,
            group,
        )
        status_layout.addWidget(QLabel("模式", group), 0, 0)
        status_layout.addWidget(self.mode_indicator, 0, 1)
        status_layout.addWidget(QLabel("使能", group), 0, 2)
        status_layout.addWidget(self.power_indicator, 0, 3)
        status_layout.addWidget(QLabel("运行", group), 1, 0)
        status_layout.addWidget(self.operation_indicator, 1, 1)
        status_layout.addWidget(QLabel("拖动", group), 1, 2)
        status_layout.addWidget(self.drag_indicator, 1, 3)
        status_layout.setColumnStretch(1, 1)
        status_layout.setColumnStretch(3, 1)
        layout.addLayout(status_layout)

        joint_layout = QGridLayout()
        joint_layout.addWidget(QLabel("关节", group), 0, 0)
        joint_layout.addWidget(QLabel("目标角度", group), 0, 1)
        joint_layout.addWidget(QLabel("当前角度", group), 0, 2)
        for index in range(7):
            box = QDoubleSpinBox(group)
            box.setRange(-540.0, 540.0)
            box.setDecimals(2)
            box.setSingleStep(1.0)
            box.setMinimumHeight(42)
            box.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            current_label = QLabel("--", group)
            current_label.setAlignment(
                Qt.AlignmentFlag.AlignRight
                | Qt.AlignmentFlag.AlignVCenter
            )
            current_label.setMinimumWidth(84)
            self._joint_boxes.append(box)
            self._joint_current_labels.append(current_label)
            row = index + 1
            joint_layout.addWidget(QLabel(f"J{index + 1}", group), row, 0)
            joint_layout.addWidget(box, row, 1)
            joint_layout.addWidget(current_label, row, 2)
        joint_layout.setColumnStretch(1, 1)
        layout.addLayout(joint_layout)
        self.move_button = QPushButton("运动到", group)
        layout.addWidget(self.move_button)
        self._control_widgets.extend(
            (
                self.mode_indicator,
                self.power_indicator,
                self.drag_indicator,
                *self._joint_boxes,
                self.move_button,
            )
        )
        return group

    def _build_prior_group(self) -> QGroupBox:
        group = QGroupBox("左臂三球先验", self)
        layout = QVBoxLayout(group)
        color_layout = QGridLayout()
        color_specs = (
            ("球 1 · X 轴", "默认黄色"),
            ("球 2 · 原点", "默认红色"),
            ("球 3 · XOY 平面", "默认蓝色"),
        )
        for index, (caption, tooltip) in enumerate(color_specs):
            button = QPushButton(group)
            button.setToolTip(f"{tooltip}；点击打开 Qt 颜色选择器")
            button.clicked.connect(
                lambda _checked=False, color_index=index: (
                    self._choose_ball_color(color_index)
                )
            )
            self._ball_color_buttons.append(button)
            color_layout.addWidget(QLabel(caption, group), index, 0)
            color_layout.addWidget(button, index, 1)
        color_layout.setColumnStretch(1, 1)
        layout.addLayout(color_layout)
        self.ball_color_order_label = QLabel(group)
        self.ball_color_order_label.setWordWrap(True)
        layout.addWidget(self.ball_color_order_label)
        self.ball_overlay_checkbox = QCheckBox("显示三球检测 Overlay", group)
        self.ball_progress_bar = QProgressBar(group)
        self.ball_progress_bar.setRange(0, 1)
        self.ball_progress_bar.setValue(0)
        self.ball_progress_bar.setFormat("尚未采集")
        self.record_ball_button = QPushButton("记录左臂先验", group)
        layout.addWidget(self.ball_overlay_checkbox)
        layout.addWidget(self.ball_progress_bar)
        layout.addWidget(self.record_ball_button)
        self._refresh_ball_color_controls()
        self._control_widgets.append(self.record_ball_button)
        return group

    def _build_head_group(self) -> QGroupBox:
        group = QGroupBox("头部 ChArUco 先验", self)
        layout = QVBoxLayout(group)
        self.head_yaw_button = QPushButton(
            f"Yaw 运动到 {self.HEAD_YAW_DEG:.0f}°",
            group,
        )
        self.head_pitch_button = QPushButton(
            f"Pitch 运动到 {self.HEAD_PITCH_DEG:.0f}°",
            group,
        )
        self.head_camera_button = QPushButton("切换到头部相机", group)
        self.head_overlay_checkbox = QCheckBox("显示头部检测 Overlay", group)
        self.record_head_button = QPushButton("获取头部先验", group)
        layout.addWidget(self.head_yaw_button)
        layout.addWidget(self.head_pitch_button)
        layout.addWidget(self.head_camera_button)
        layout.addWidget(self.head_overlay_checkbox)
        layout.addWidget(self.record_head_button)
        self._control_widgets.extend(
            (
                self.head_yaw_button,
                self.head_pitch_button,
                self.head_camera_button,
                self.head_overlay_checkbox,
                self.record_head_button,
            )
        )
        return group

    # endregion

    # region 生命周期

    def set_clients(
        self,
        arm: Ar5Client | None,
        head: WujiHeadClient | None,
        recorder: PriorCalibrationRecorder | None,
    ) -> None:
        self._arm = arm
        self._head = head
        self._recorder = recorder
        self._snapshot = None
        self._joint_targets_initialized = False
        self.set_connection_ready(
            arm is not None and head is not None and recorder is not None
        )

    def set_connection_ready(self, ready: bool) -> None:
        for widget in self._control_widgets:
            widget.setEnabled(ready)
        if ready:
            return
        self.mode_indicator.set_state("unknown")
        self.power_indicator.set_state("unknown")
        self.operation_indicator.set_state("unknown")
        self.drag_indicator.set_state("unknown")
        for label in self._joint_current_labels:
            label.setText("--")
        self.status_label.setText("等待连接")
        self.preview.clear_preview("等待相机连接")

    def set_active(self, active: bool) -> None:
        self._active = active
        if not active:
            self._refresh_timer.stop()
            self.streamStopRequested.emit()
            return
        if self._arm is None:
            self.status_label.setText("先验标定设备未连接")
            return
        self._refresh_timer.start()
        self._select_camera(BALL_CAMERA_NAME)
        self._request_refresh()

    # endregion

    # region 刷新与预览

    def _request_refresh(self) -> None:
        if (
            self._arm is None
            or self._refresh_busy
            or self._action_busy
            or self._prior_busy
        ):
            return
        self._refresh_busy = True
        self._refresh_call.start(self._arm.read_snapshot)

    @Slot(object)
    def _on_refresh_succeeded(self, payload: object) -> None:
        if not isinstance(payload, Ar5Snapshot):
            return
        self._snapshot = payload
        self.mode_indicator.set_state(payload.operate_mode)
        self.power_indicator.set_state(
            ArmTabWidget._power_indicator_state(payload.power_state)
        )
        self.operation_indicator.set_state(
            ArmTabWidget._operation_indicator_state(payload.operation_state)
        )
        self.drag_indicator.set_state(
            "drag" if payload.operation_state == "drag" else "off"
        )
        for label, value in zip(
            self._joint_current_labels,
            payload.joint_deg,
            strict=True,
        ):
            label.setText(f"{value:.2f}°")
        if not self._joint_targets_initialized:
            for box, value in zip(
                self._joint_boxes,
                payload.joint_deg,
                strict=True,
            ):
                box.setValue(value)
            self._joint_targets_initialized = True

    @Slot(str)
    def _on_refresh_failed(self, message: str) -> None:
        self.status_label.setText(f"左 AR5 状态读取失败：{message}")

    @Slot()
    def _on_refresh_finished(self) -> None:
        self._refresh_busy = False

    def update_frame(self, frame: WujiCameraFrame) -> None:
        if frame.camera_name != self._camera_name:
            return
        self._last_frame_bgr = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
        self._refresh_preview()

    @Slot()
    def _refresh_preview(self) -> None:
        image = self._last_frame_bgr
        if (
            self._camera_name == BALL_CAMERA_NAME
            and self.ball_overlay_checkbox.isChecked()
            and self._ball_overlay_bgr is not None
        ):
            image = self._ball_overlay_bgr
        elif (
            self._camera_name == HEAD_CAMERA_NAME
            and self.head_overlay_checkbox.isChecked()
            and self._head_overlay_bgr is not None
        ):
            image = self._head_overlay_bgr
        if image is not None:
            self.preview.set_preview_pixmap(_bgr_to_pixmap(image))

    def _select_camera(self, camera_name: str) -> None:
        self._camera_name = camera_name
        self._last_frame_bgr = None
        is_head = camera_name == HEAD_CAMERA_NAME
        self.camera_title_label.setText(
            "头部相机 · head_camera"
            if is_head
            else "左臂相机 · left_hand_camera"
        )
        self.head_camera_button.setText(
            "切换到左臂相机" if is_head else "切换到头部相机"
        )
        self.preview.clear_preview(
            "等待头部相机图像" if is_head else "等待左臂相机图像"
        )
        self.cameraStreamRequested.emit(camera_name)
        self._refresh_preview()

    @Slot()
    def _toggle_head_camera(self) -> None:
        target = (
            BALL_CAMERA_NAME
            if self._camera_name == HEAD_CAMERA_NAME
            else HEAD_CAMERA_NAME
        )
        self._select_camera(target)

    # endregion

    # region 控制

    def _choose_ball_color(self, index: int) -> None:
        """使用 Qt 颜色对话框修改指定球的参考颜色。"""

        current = QColor(self._ball_colors[index])
        selected = QColorDialog.getColor(
            current,
            self,
            f"选择球 {index + 1} 的颜色",
        )
        if not selected.isValid():
            return
        color_hex = selected.name(QColor.NameFormat.HexRgb).lower()
        if color_hex in (
            color
            for color_index, color in enumerate(self._ball_colors)
            if color_index != index
        ):
            QMessageBox.warning(
                self,
                "球颜色重复",
                "三个球必须使用不同颜色，否则无法建立稳定的颜色身份。",
            )
            return
        self._ball_colors[index] = color_hex
        self._refresh_ball_color_controls()

    def _refresh_ball_color_controls(self) -> None:
        """刷新三个颜色按钮及当前顺序说明。"""

        for button, color_hex in zip(
            self._ball_color_buttons,
            self._ball_colors,
            strict=True,
        ):
            color = QColor(color_hex)
            text_color = "#000000" if color.lightness() > 128 else "#ffffff"
            button.setText(color_hex.upper())
            button.setStyleSheet(
                "QPushButton {"
                f"background-color: {color_hex}; color: {text_color};"
                "border: 1px solid #606060; border-radius: 4px;"
                "}"
            )
        self.ball_color_order_label.setText(
            "当前顺序："
            f"{self._ball_colors[0].upper()}（X 轴） → "
            f"{self._ball_colors[1].upper()}（原点） → "
            f"{self._ball_colors[2].upper()}（XOY 平面）"
        )

    @Slot()
    def _toggle_mode(self) -> None:
        automatic = (
            self._snapshot is None
            or self._snapshot.operate_mode != "automatic"
        )
        self._run_action(
            lambda: self._require_arm().set_operate_mode(automatic),
            "左 AR5 已切换为自动模式" if automatic else "左 AR5 已切换为手动模式",
        )

    @Slot()
    def _toggle_power(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        if snapshot.power_state not in {"on", "off"}:
            QMessageBox.warning(
                self,
                "使能状态不允许切换",
                "当前为急停、安全门或安全停止状态，请先完成对应恢复操作。",
            )
            return
        enabled = snapshot.power_state != "on"
        self._run_action(
            lambda: self._require_arm().set_power(enabled),
            "左 AR5 已上电" if enabled else "左 AR5 已下电",
        )

    @Slot()
    def _toggle_drag(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        enabled = snapshot.operation_state != "drag"
        if enabled and (
            snapshot.operate_mode != "manual"
        ):
            QMessageBox.warning(
                self,
                "拖动状态不满足",
                "请先点击模式指示灯，将左 AR5 切换为手动模式。",
            )
            return
        self._run_action(
            lambda: self._require_arm().set_drag_enabled(enabled),
            "左 AR5 拖动已开启" if enabled else "左 AR5 拖动已关闭",
        )

    @Slot()
    def _move_joints(self) -> None:
        snapshot = self._snapshot
        if (
            snapshot is None
            or snapshot.operate_mode != "automatic"
            or snapshot.power_state != "on"
        ):
            QMessageBox.warning(
                self,
                "MoveJoint 状态不满足",
                "请先自主切换为自动模式并上电，页面不会自动改变控制状态。",
            )
            return
        targets = tuple(box.value() for box in self._joint_boxes)
        self._run_action(
            lambda: self._require_arm().move_joints_deg(targets),
            "左 AR5 MoveJoint 已下发",
        )

    @Slot()
    def _move_head_yaw(self) -> None:
        self._run_action(
            lambda: self._require_head().set_head_yaw(self.HEAD_YAW_DEG),
            f"头部 Yaw 已运动到 {self.HEAD_YAW_DEG:.0f}°",
        )

    @Slot()
    def _move_head_pitch(self) -> None:
        self._run_action(
            lambda: self._require_head().set_head_pitch(self.HEAD_PITCH_DEG),
            f"头部 Pitch 已运动到 {self.HEAD_PITCH_DEG:.0f}°",
        )

    def _run_action(self, callback, message: str) -> None:  # noqa: ANN001
        if self._action_busy:
            return
        self._action_busy = True
        self.status_label.setText("控制命令执行中…")
        self._action_call.start(lambda: (callback(), message)[1])

    @Slot(object)
    def _on_action_succeeded(self, payload: object) -> None:
        self.status_label.setText(str(payload))
        QTimer.singleShot(100, self._request_refresh)

    @Slot(str)
    def _on_action_failed(self, message: str) -> None:
        self.status_label.setText(f"控制失败：{message}")
        if self._snapshot is not None:
            self._on_refresh_succeeded(self._snapshot)

    @Slot()
    def _on_action_finished(self) -> None:
        self._action_busy = False

    # endregion

    # region 先验记录

    @Slot()
    def _record_ball_prior(self) -> None:
        if self._prior_busy or self._snapshot is None:
            return
        snapshot = self._snapshot
        if snapshot.operation_state not in {"idle", "drag"}:
            QMessageBox.warning(
                self,
                "左臂状态不稳定",
                "记录三球先验前请停止 Move/Jog，使左臂保持空闲或拖动状态。",
            )
            return
        self._prior_busy = True
        self.record_ball_button.setEnabled(False)
        target_count = self._require_recorder().ball_sample_count
        self.ball_progress_bar.setRange(0, target_count)
        self.ball_progress_bar.setValue(0)
        self.ball_progress_bar.setFormat(f"0/{target_count}")
        if self._camera_name != BALL_CAMERA_NAME:
            self._select_camera(BALL_CAMERA_NAME)
        self.ball_overlay_checkbox.setChecked(True)
        self.status_label.setText("左臂三球先验开始采集…")
        self._prior_call.start(
            lambda: self._require_recorder().record_ball_prior(
                snapshot,
                ball_colors=(
                    self._ball_colors[0],
                    self._ball_colors[1],
                    self._ball_colors[2],
                ),
                progress=lambda sample: self.ballSampleReady.emit(sample),
            )
        )

    @Slot(object)
    def _on_ball_sample_ready(self, payload: object) -> None:
        """显示每一帧有效三球检测的采集进度和真实 overlay。"""

        if not isinstance(payload, PriorBallSampleProgress):
            logger.error("三球采集进度类型异常：{}", type(payload).__name__)
            return
        self.ball_progress_bar.setRange(0, payload.total)
        self.ball_progress_bar.setValue(payload.current)
        self.ball_progress_bar.setFormat(f"{payload.current}/{payload.total}")
        self.status_label.setText(
            f"左臂三球先验采集：{payload.current}/{payload.total}"
        )
        self._ball_overlay_bgr = payload.overlay_bgr
        self._refresh_preview()

    @Slot()
    def _record_head_prior(self) -> None:
        if self._prior_busy:
            return
        self._prior_busy = True
        self.record_head_button.setEnabled(False)
        if self._camera_name != HEAD_CAMERA_NAME:
            self._select_camera(HEAD_CAMERA_NAME)
        self.head_overlay_checkbox.setChecked(True)
        self.status_label.setText("头部 ChArUco 先验检测中…")
        self._prior_call.start(self._capture_head_prior)

    def _capture_head_prior(self) -> PriorCalibrationResult:
        head = self._require_head()
        yaw = head.get_head_yaw()
        pitch = head.get_head_pitch()
        if (
            abs(yaw - self.HEAD_YAW_DEG) > self.HEAD_POSITION_TOLERANCE_DEG
            or abs(pitch - self.HEAD_PITCH_DEG)
            > self.HEAD_POSITION_TOLERANCE_DEG
        ):
            raise RuntimeError(
                "头部未到标定位置："
                f"yaw={yaw:.2f}° pitch={pitch:.2f}°"
            )
        return self._require_recorder().record_head_prior()

    @Slot(object)
    def _on_prior_succeeded(self, payload: object) -> None:
        if not isinstance(payload, PriorCalibrationResult):
            logger.error(
                "先验获取返回了非预期结果类型：{}",
                type(payload).__name__,
            )
            return
        self.status_label.setText(f"{payload.message}\n{payload.result_path}")
        if payload.overlay_bgr is not None:
            if payload.calibration_kind == "ball":
                self._ball_overlay_bgr = payload.overlay_bgr
                if self._camera_name != BALL_CAMERA_NAME:
                    self._select_camera(BALL_CAMERA_NAME)
                self.ball_overlay_checkbox.setChecked(True)
            else:
                self._head_overlay_bgr = payload.overlay_bgr
                if self._camera_name != HEAD_CAMERA_NAME:
                    self._select_camera(HEAD_CAMERA_NAME)
                self.head_overlay_checkbox.setChecked(True)
            self._refresh_preview()
        logger.success(
            "先验获取成功：result_path={} overlay={}",
            payload.result_path,
            payload.overlay_bgr is not None,
        )

    @Slot(str)
    def _on_prior_failed(self, message: str) -> None:
        self.status_label.setText(f"先验获取失败：{message}")
        logger.error("先验获取失败：{}", message)

    @Slot()
    def _on_prior_finished(self) -> None:
        self._prior_busy = False
        ready = self._arm is not None and self._recorder is not None
        self.record_ball_button.setEnabled(ready)
        self.record_head_button.setEnabled(ready)

    # endregion

    # region 工具

    def _require_arm(self) -> Ar5Client:
        if self._arm is None:
            raise RuntimeError("左 AR5 未连接")
        return self._arm

    def _require_head(self) -> WujiHeadClient:
        if self._head is None:
            raise RuntimeError("头部未连接")
        return self._head

    def _require_recorder(self) -> PriorCalibrationRecorder:
        if self._recorder is None:
            raise RuntimeError("CameraPipeline 未连接")
        return self._recorder

    # endregion


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    """把 `(H, W, 3)` uint8 BGR 图像转换为 Qt Pixmap。"""

    bgr = np.ascontiguousarray(image_bgr)
    height, width = bgr.shape[:2]
    image = QImage(
        bgr.data,
        width,
        height,
        int(bgr.strides[0]),
        QImage.Format.Format_BGR888,
    ).copy()
    return QPixmap.fromImage(image)
