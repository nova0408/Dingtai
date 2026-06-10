from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, sqrt
import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.test_gui.uitl_dof_widget_model import DoFWidgetModel
from gui.test_gui.uitl_dof_widget_view import UtilDoFWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.servers.common import JointLimit
from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.protocol import WujiRobotRuntimeStructure, WujiRuntimeModuleSpec


@dataclass(frozen=True, slots=True)
class DebugAxisSpec:
    """单个可动轴在调试界面中的显示与控制规格。"""

    axis_name: str
    limit: JointLimit
    step: float = 1.0
    hold_step: float = 1.0
    control_supported: bool = True
    refresh_supported: bool = True


@dataclass(frozen=True, slots=True)
class DebugModuleSpec:
    """单个可动模块在调试界面中的分组规格。"""

    tab_name: str
    title: str
    device_name: str
    axes: tuple[DebugAxisSpec, ...] = ()
    enable_supported: bool = True
    refresh_supported: bool = True


class DebugModulePanel(QWidget):
    dofTargetRequested = Signal(str, float)
    enableToggleRequested = Signal(str, bool)

    def __init__(self, specs: tuple[DebugModuleSpec, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dof_widgets_by_axis: dict[str, UtilDoFWidget] = {}
        self.refresh_axis_names: list[str] = []
        self.enable_indicators: dict[str, CasiaIndicatorLight] = {}
        self.refresh_device_names: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for spec in specs:
            group = QGroupBox(spec.title, self)
            group_layout = QVBoxLayout(group)
            group_layout.addLayout(self._create_enable_row(group, spec))
            self._add_axis_widgets(group_layout, group, spec.axes)
            layout.addWidget(group)
        layout.addStretch(1)

    def _create_enable_row(self, parent: QWidget, spec: DebugModuleSpec) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(QLabel("使能:", parent))
        indicator = CasiaIndicatorLight(parent, text=("使能", "禁用"), font_size=12, default_status=False)
        indicator.setObjectName(f"{spec.device_name}_enable")
        indicator.setEnabled(spec.enable_supported)
        if spec.enable_supported:
            indicator.clicked.connect(
                lambda name=spec.device_name, light=indicator: self._request_enable_toggle(
                    name,
                    light,
                )
            )
            self.enable_indicators[spec.device_name] = indicator
            if spec.refresh_supported:
                self.refresh_device_names.append(spec.device_name)
        layout.addWidget(indicator)
        layout.addStretch(1)
        return layout

    def _add_axis_widgets(self, layout: QVBoxLayout, parent: QWidget, axes: tuple[DebugAxisSpec, ...]) -> None:
        if not axes:
            label = QLabel("未配置可动轴", parent)
            label.setEnabled(False)
            layout.addWidget(label)
            return
        for axis in axes:
            model = DoFWidgetModel(
                axis.axis_name,
                axis.limit.minimum,
                axis.limit.maximum,
                axis.limit.unit,
                step=axis.step,
                hold_step=axis.hold_step,
            )
            widget = UtilDoFWidget(model=model, parent=parent)
            widget.setObjectName(f"dof_{axis.axis_name}")
            widget.setEnabled(axis.control_supported)
            if axis.control_supported:
                widget.targetRequested.connect(self.dofTargetRequested)
            if axis.refresh_supported:
                self.dof_widgets_by_axis[axis.axis_name] = widget
                self.refresh_axis_names.append(axis.axis_name)
            layout.addWidget(widget)

    def update_dof_value(self, axis_name: str, value: float) -> None:
        widget = self.dof_widgets_by_axis.get(axis_name)
        if widget is not None:
            widget.update_feedback_value(value)

    def update_enable_state(self, device_name: str, enabled: bool) -> None:
        indicator = self.enable_indicators.get(device_name)
        if indicator is not None:
            indicator.set_status(enabled)

    def _request_enable_toggle(self, device_name: str, indicator: CasiaIndicatorLight) -> None:
        self.enableToggleRequested.emit(device_name, not bool(indicator.property("status")))


@dataclass(frozen=True, slots=True)
class ArmAxisSpec:
    """机械臂单轴显示与控制规格。

    职责边界：
    - 只描述 GUI 需要的关节名、限位与步进，不承载任何业务逻辑。
    - 不负责读取实时关节，也不负责下发控制命令。

    设计思想：
    - 将机械臂六轴的显示规则和控制步长集中成静态结构，避免散落在 widget 内部。
    - 使用不可变 dataclass，便于在界面初始化阶段一次性构造并复用。

    生命周期：
    - 纯 UI 配置数据，不持有硬件句柄或线程资源。

    继承关系：
    - 不继承业务基类。
    """

    axis_name: str
    "关节名，例如 `left_j1`。"

    minimum: float
    "关节最小角度，单位 deg。"

    maximum: float
    "关节最大角度，单位 deg。"

    step: float = 1.0
    "单次点击 `-` / `+` 的角度步进，单位 deg。"


@dataclass(frozen=True, slots=True)
class PoseFieldSpec:
    """正逆解输入输出字段规格。"""

    field_name: str
    "字段名，例如 `x` 或 `yaw`。"

    label_text: str
    "界面显示文本。"

    unit_text: str
    "单位文本，例如 `mm`、`deg` 或空字符串。"


POSE_POSITION_SCALE_MM = 1000.0
# GUI 侧位姿显示的长度换算比例。接口返回与下发保持米，界面显示统一换算为 mm。


class ArmModulePanel(QWidget):
    """单个机械臂的专用操作面板。

    职责边界：
    - 只负责单个机械臂的六轴读数、单轴控制，以及 FK / IK 输入输出界面。
    - 不直接访问 backend，不保存连接状态，不做网络请求。

    设计思想：
    - 用固定三栏布局收口机械臂界面：左侧读数、中间控制、右侧正逆解。
    - 读数和控制分离，避免把状态展示与控制交互耦合到同一组旧 DoF 控件里。

    生命周期：
    - 纯 Qt 控件，不持有硬件资源；由父窗口创建并随窗口销毁。

    继承关系：
    - 不继承业务基类，作为 GUI 专用 widget。
    """

    axisTargetRequested = Signal(str, float)
    ikRequested = Signal(str, tuple, tuple)
    stopRequested = Signal(str)
    enableToggleRequested = Signal(str, bool)

    def __init__(self, device_name: str, title: str, axes: tuple[ArmAxisSpec, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._device_name = device_name
        self._axes = axes
        self._enabled = False
        self._axis_current_values: dict[str, float] = {}
        self.enable_indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        self.axis_readout_labels: dict[str, QLabel] = {}
        self.axis_spin_boxes: dict[str, QDoubleSpinBox] = {}
        self.fk_value_labels: dict[str, QLabel] = {}
        self.ik_input_boxes: dict[str, QDoubleSpinBox] = {}
        self.ik_result_label: QLabel | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox(title, self)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(8, 8, 8, 8)

        enable_row = QHBoxLayout()
        enable_row.addWidget(self.enable_indicator)
        enable_row.addStretch(1)
        group_layout.addLayout(enable_row)

        body_row = QHBoxLayout()
        body_row.addWidget(self._build_axis_readout_column(group))
        body_row.addWidget(self._build_axis_control_column(group))
        body_row.addWidget(self._build_fk_ik_column(group))
        group_layout.addLayout(body_row)
        root_layout.addWidget(group)
        root_layout.addStretch(1)

    def _build_axis_readout_column(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for axis in self._axes:
            row = QHBoxLayout()
            row.addWidget(QLabel(axis.axis_name, container))
            value_label = QLabel("-/-/-", container)
            value_label.setMinimumWidth(150)
            self.axis_readout_labels[axis.axis_name] = value_label
            row.addWidget(value_label, 1)
            layout.addLayout(row)
        layout.addStretch(1)
        return container

    def update_enable_state(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self.enable_indicator.set_status(self._enabled)
        for widget in self.findChildren(QDoubleSpinBox):
            widget.setEnabled(self._enabled)

    def _build_axis_control_column(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for axis in self._axes:
            row = QHBoxLayout()
            spin = QDoubleSpinBox(container)
            spin.setDecimals(1)
            spin.setRange(axis.minimum, axis.maximum)
            spin.setSingleStep(axis.step)
            spin.setValue(axis.minimum)
            self.axis_spin_boxes[axis.axis_name] = spin
            set_button = QPushButton("set", container)
            minus_button = QPushButton("-", container)
            plus_button = QPushButton("+", container)
            set_button.clicked.connect(lambda _checked=False, axis_name=axis.axis_name, box=spin: self.axisTargetRequested.emit(axis_name, float(box.value())))
            minus_button.clicked.connect(lambda _checked=False, axis_name=axis.axis_name, box=spin: self._nudge_axis(axis_name, box, -1.0))
            plus_button.clicked.connect(lambda _checked=False, axis_name=axis.axis_name, box=spin: self._nudge_axis(axis_name, box, 1.0))
            row.addWidget(spin)
            row.addWidget(set_button)
            row.addWidget(minus_button)
            row.addWidget(plus_button)
            layout.addLayout(row)
        layout.addStretch(1)
        return container

    def _build_fk_ik_column(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        fk_group = QGroupBox("FK", container)
        fk_layout = QVBoxLayout(fk_group)
        fk_layout.addLayout(self._build_pose_row(fk_group, (("x", "x"), ("y_pos", "y"), ("z", "z"))))
        fk_layout.addLayout(self._build_pose_row(fk_group, (("r", "r"), ("p", "p"), ("yaw", "yaw"))))
        layout.addWidget(fk_group)

        ik_group = QGroupBox("IK", container)
        ik_layout = QVBoxLayout(ik_group)
        ik_layout.addLayout(self._build_ik_input_row(ik_group, (("x", "x", "mm"), ("y_pos", "y", "mm"), ("z", "z", "mm"))))
        ik_layout.addLayout(self._build_ik_input_row(ik_group, (("r", "r", "deg"), ("p", "p", "deg"), ("yaw", "yaw", "deg"))))
        button_row = QHBoxLayout()
        apply_button = QPushButton("apply", ik_group)
        stop_button = QPushButton("stop", ik_group)
        apply_button.clicked.connect(lambda _checked=False: self.ikRequested.emit(self._device_name, self._collect_ik_pose(), self._collect_reference_joints()))
        stop_button.clicked.connect(lambda _checked=False: self.stopRequested.emit(self._device_name))
        button_row.addWidget(apply_button)
        button_row.addWidget(stop_button)
        ik_layout.addLayout(button_row)
        self.ik_result_label = QLabel("IK result: -", ik_group)
        self.ik_result_label.setWordWrap(True)
        ik_layout.addWidget(self.ik_result_label)
        layout.addWidget(ik_group)
        layout.addStretch(1)
        return container

    def _build_pose_row(self, parent: QWidget, fields: tuple[tuple[str, str], ...]) -> QHBoxLayout:
        row = QHBoxLayout()
        for field_name, label_text in fields:
            label = QLabel(f"{label_text}: -.-", parent)
            self.fk_value_labels[field_name] = label
            row.addWidget(label)
        return row

    def _build_ik_input_row(self, parent: QWidget, fields: tuple[tuple[str, str, str], ...]) -> QHBoxLayout:
        row = QHBoxLayout()
        for field_name, label_text, unit_text in fields:
            box = QDoubleSpinBox(parent)
            box.setDecimals(1)
            box.setRange(-9999.0, 9999.0)
            box.setSingleStep(1.0)
            self.ik_input_boxes[field_name] = box
            row.addWidget(QLabel(label_text, parent))
            row.addWidget(box)
            if unit_text:
                row.addWidget(QLabel(unit_text, parent))
        return row

    def _nudge_axis(self, axis_name: str, box: QDoubleSpinBox, direction: float) -> None:
        current_value = self._axis_current_values.get(axis_name, float(box.value()))
        target_value = current_value + direction * box.singleStep()
        self.axisTargetRequested.emit(axis_name, float(target_value))

    def _collect_ik_pose(self) -> tuple[float, ...]:
        return tuple(float(self.ik_input_boxes[field].value()) for field in ("x", "y_pos", "z", "r", "p", "yaw"))

    def _collect_reference_joints(self) -> tuple[float, ...]:
        return tuple(float(self.axis_spin_boxes[axis.axis_name].value()) for axis in self._axes)

    def update_axis_readouts(self, values: dict[str, float]) -> None:
        for axis in self._axes:
            label = self.axis_readout_labels.get(axis.axis_name)
            if label is None:
                continue
            current_value = values.get(axis.axis_name)
            if current_value is None:
                label.setText("-/-/-")
                continue
            self._axis_current_values[axis.axis_name] = float(current_value)
            label.setText(f"{axis.minimum:.1f}/{current_value:.1f}/{axis.maximum:.1f}")

    def update_fk_pose(self, pose: object) -> None:
        values = self._extract_pose_values(pose)
        if values is None:
            return
        x, y_pos, z, roll, pitch, yaw = values
        self.fk_value_labels["x"].setText(f"x: {x:.1f} mm")
        self.fk_value_labels["y_pos"].setText(f"y: {y_pos:.1f} mm")
        self.fk_value_labels["z"].setText(f"z: {z:.1f} mm")
        self.fk_value_labels["r"].setText(f"r: {roll:.1f} deg")
        self.fk_value_labels["p"].setText(f"p: {pitch:.1f} deg")
        self.fk_value_labels["yaw"].setText(f"yaw: {yaw:.1f} deg")

    def update_ik_inputs(self, values: tuple[float, ...]) -> None:
        if len(values) != 6:
            return
        for field_name, value in zip(("x", "y_pos", "z", "r", "p", "yaw"), values, strict=True):
            self.ik_input_boxes[field_name].setValue(float(value))

    def update_ik_result(self, values: tuple[float, ...] | list[float] | None) -> None:
        if self.ik_result_label is None:
            return
        if values is None:
            self.ik_result_label.setText("IK result: -")
            return
        result_values = tuple(float(value) for value in values)
        if len(result_values) != 6:
            self.ik_result_label.setText(f"IK result: invalid result length={len(result_values)}")
            return
        self.ik_result_label.setText(
            "IK result: "
            + ", ".join(f"J{index + 1}={value:.1f} deg" for index, value in enumerate(result_values))
        )

    def _extract_pose_values(self, pose: object) -> tuple[float, float, float, float, float, float] | None:
        if pose is None:
            return None
        matrix = np.asarray(pose, dtype=np.float64)
        if matrix.shape == (4, 4):
            return self._matrix_to_gui_pose_values(matrix)
        if hasattr(pose, "translation") and hasattr(pose, "rotation"):
            translation = getattr(pose, "translation")
            rotation = getattr(pose, "rotation")
            x = float(getattr(translation, "x", 0.0)) * POSE_POSITION_SCALE_MM
            y = float(getattr(translation, "y", 0.0)) * POSE_POSITION_SCALE_MM
            z = float(getattr(translation, "z", 0.0)) * POSE_POSITION_SCALE_MM
            if hasattr(rotation, "as_zyx"):
                yaw, pitch, roll = rotation.as_zyx(degrees=True)
                return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)
            if all(hasattr(rotation, attr) for attr in ("w", "x", "y", "z")):
                try:
                    from src.utils.datas.kinematics.quaternion import Quaternion

                    quat = Quaternion(float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z))
                    yaw, pitch, roll = quat.as_zyx(degrees=True)
                    return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)
                except Exception:  # noqa: BLE001
                    return float(x), float(y), float(z), 0.0, 0.0, 0.0
        if matrix.size == 16:
            reshaped = matrix.reshape(4, 4)
            return self._matrix_to_gui_pose_values(reshaped)
        if isinstance(pose, (tuple, list)) and len(pose) >= 6:
            return tuple(float(item) for item in pose[:6])  # type: ignore[return-value]
        return None

    def _matrix_to_gui_pose_values(self, matrix: np.ndarray) -> tuple[float, float, float, float, float, float]:
        """将 4x4 位姿矩阵转换为 GUI 展示使用的 `mm + deg` 组合。"""

        x = float(matrix[0, 3]) * POSE_POSITION_SCALE_MM
        y = float(matrix[1, 3]) * POSE_POSITION_SCALE_MM
        z = float(matrix[2, 3]) * POSE_POSITION_SCALE_MM

        r00 = float(matrix[0, 0])
        r10 = float(matrix[1, 0])
        r20 = float(matrix[2, 0])
        r21 = float(matrix[2, 1])
        r22 = float(matrix[2, 2])

        cy = sqrt(r00 * r00 + r10 * r10)
        if cy > 1e-9:
            roll_rad = atan2(r21, r22)
            pitch_rad = atan2(-r20, cy)
            yaw_rad = atan2(r10, r00)
        else:
            roll_rad = 0.0
            pitch_rad = atan2(-r20, cy)
            yaw_rad = atan2(-float(matrix[0, 1]), float(matrix[1, 1]))

        return (
            x,
            y,
            z,
            float(np.rad2deg(roll_rad)),
            float(np.rad2deg(pitch_rad)),
            float(np.rad2deg(yaw_rad)),
        )


class StatusAxisPanel(QWidget):
    """用于右手和 AGV 这类状态展示与控制复用的专用面板。"""

    enableRequested = Signal(bool)

    def __init__(self, title: str, axis_names: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.value_labels: dict[str, QLabel] = {}
        self.enable_indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox(title, self)
        group_layout = QVBoxLayout(group)
        enable_row = QHBoxLayout()
        enable_row.addWidget(self.enable_indicator)
        enable_row.addStretch(1)
        group_layout.addLayout(enable_row)
        values_grid = QGridLayout()
        for row, axis_name in enumerate(axis_names):
            values_grid.addWidget(QLabel(axis_name, group), row, 0)
            value_label = QLabel("-", group)
            self.value_labels[axis_name] = value_label
            values_grid.addWidget(value_label, row, 1)
        values_grid.setColumnStretch(1, 1)
        group_layout.addLayout(values_grid)
        layout.addWidget(group)
        layout.addStretch(1)
        self.enable_indicator.clicked.connect(self._on_enable_indicator_clicked)

    def update_values(self, values: dict[str, float], formatter: Callable[[str, float], str]) -> None:
        for axis_name, label in self.value_labels.items():
            if axis_name in values:
                label.setText(formatter(axis_name, float(values[axis_name])))

    def update_enable_state(self, enabled: bool) -> None:
        self.enable_indicator.set_status(bool(enabled))

    def _on_enable_indicator_clicked(self) -> None:
        self.enableRequested.emit(not bool(self.enable_indicator.property("status")))


class RightHandControlPanel(StatusAxisPanel):
    """右手控制面板，提供状态展示与右手专用控制入口。"""

    enableRequested = Signal(bool)
    setPoseRequested = Signal()
    axisTargetRequested = Signal(str, float)

    def __init__(self, axis_names: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__("右手灵巧手", axis_names, parent)
        self.axis_spin_boxes: dict[str, QDoubleSpinBox] = {}
        self.axis_current_labels: dict[str, QLabel] = {}
        self._axis_current_values: dict[str, float] = {}

        controls_group = QGroupBox("Control", self)
        controls_layout = QVBoxLayout(controls_group)
        self.enable_button = QPushButton("Toggle Enable", controls_group)
        self.pose_button = QPushButton("Set Demo Pose", controls_group)
        controls_layout.addWidget(self.enable_button)
        controls_layout.addWidget(self.pose_button)
        for axis_name in axis_names:
            row = QHBoxLayout()
            row.addWidget(QLabel(axis_name, controls_group))
            current_label = QLabel("-/-/-", controls_group)
            self.axis_current_labels[axis_name] = current_label
            row.addWidget(current_label)
            spin = QDoubleSpinBox(controls_group)
            spin.setDecimals(3)
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.01)
            spin.setValue(0.0)
            self.axis_spin_boxes[axis_name] = spin
            set_button = QPushButton("set", controls_group)
            minus_button = QPushButton("-", controls_group)
            plus_button = QPushButton("+", controls_group)
            set_button.clicked.connect(lambda _checked=False, name=axis_name, box=spin: self.axisTargetRequested.emit(name, float(box.value())))
            minus_button.clicked.connect(lambda _checked=False, name=axis_name, box=spin: self._nudge_axis(name, box, -1.0))
            plus_button.clicked.connect(lambda _checked=False, name=axis_name, box=spin: self._nudge_axis(name, box, 1.0))
            row.addWidget(spin)
            row.addWidget(set_button)
            row.addWidget(minus_button)
            row.addWidget(plus_button)
            controls_layout.addLayout(row)
        controls_layout.addStretch(1)
        layout = self.layout()
        assert isinstance(layout, QVBoxLayout)
        layout.insertWidget(1, controls_group)
        self.enable_button.clicked.connect(lambda _checked=False: self.enableRequested.emit(True))
        self.pose_button.clicked.connect(lambda _checked=False: self.setPoseRequested.emit())

    def update_values(self, values: dict[str, float], formatter: Callable[[str, float], str]) -> None:
        super().update_values(values, formatter)
        for axis_name, label in self.axis_current_labels.items():
            if axis_name in values:
                self._axis_current_values[axis_name] = float(values[axis_name])
                label.setText(formatter(axis_name, float(values[axis_name])))

    def _nudge_axis(self, axis_name: str, box: QDoubleSpinBox, direction: float) -> None:
        current_value = self._axis_current_values.get(axis_name, float(box.value()))
        target_value = max(0.0, min(1.0, current_value + direction * box.singleStep()))
        self.axisTargetRequested.emit(axis_name, float(target_value))


class AgvControlPanel(StatusAxisPanel):
    """AGV 控制面板，保留状态展示并提供底盘动作入口。"""

    enableRequested = Signal(bool)
    moveRequested = Signal(str)
    navigateRequested = Signal(str)
    chargeRequested = Signal()
    stopRequested = Signal()

    def __init__(self, axis_names: tuple[str, ...], parent: QWidget | None = None) -> None:
        super().__init__("AGV", axis_names, parent)
        controls_group = QGroupBox("Control", self)
        controls_layout = QVBoxLayout(controls_group)
        self.enable_button = QPushButton("Toggle Enable", controls_group)
        move_row_1 = QHBoxLayout()
        move_row_2 = QHBoxLayout()
        self.forward_button = QPushButton("Forward", controls_group)
        self.backward_button = QPushButton("Backward", controls_group)
        self.left_button = QPushButton("Left", controls_group)
        self.right_button = QPushButton("Right", controls_group)
        self.stop_button = QPushButton("Stop", controls_group)
        self.nav_target_edit = QLineEdit(controls_group)
        self.nav_target_edit.setPlaceholderText("point name")
        self.nav_target_button = QPushButton("Go To Point", controls_group)
        self.charge_button = QPushButton("Go Charge", controls_group)
        move_row_1.addWidget(self.forward_button)
        move_row_1.addWidget(self.backward_button)
        move_row_2.addWidget(self.left_button)
        move_row_2.addWidget(self.right_button)
        controls_layout.addWidget(self.enable_button)
        controls_layout.addLayout(move_row_1)
        controls_layout.addLayout(move_row_2)
        controls_layout.addWidget(self.stop_button)
        nav_row = QHBoxLayout()
        nav_row.addWidget(QLabel("Target", controls_group))
        nav_row.addWidget(self.nav_target_edit, 1)
        nav_row.addWidget(self.nav_target_button)
        controls_layout.addLayout(nav_row)
        charge_row = QHBoxLayout()
        charge_row.addWidget(self.charge_button)
        controls_layout.addLayout(charge_row)
        controls_layout.addStretch(1)
        layout = self.layout()
        assert isinstance(layout, QVBoxLayout)
        layout.insertWidget(1, controls_group)
        self.enable_button.clicked.connect(lambda _checked=False: self.enableRequested.emit(True))
        self.forward_button.clicked.connect(lambda _checked=False: self.moveRequested.emit("forward"))
        self.backward_button.clicked.connect(lambda _checked=False: self.moveRequested.emit("backward"))
        self.left_button.clicked.connect(lambda _checked=False: self.moveRequested.emit("left"))
        self.right_button.clicked.connect(lambda _checked=False: self.moveRequested.emit("right"))
        self.stop_button.clicked.connect(lambda _checked=False: self.stopRequested.emit())
        self.nav_target_button.clicked.connect(lambda _checked=False: self.navigateRequested.emit(self.nav_target_edit.text().strip()))
        self.charge_button.clicked.connect(lambda _checked=False: self.chargeRequested.emit())

    def update_navigation_target(self, target_name: str) -> None:
        self.nav_target_edit.setText(target_name)


class TestWujiCasiaArmWidget(QWidget):
    dofTargetRequested = Signal(str, float)
    dofValueRefreshRequested = Signal(str)
    enableToggleRequested = Signal(str, bool)
    enableStateRefreshRequested = Signal(str)
    armIkRequested = Signal(str, tuple, tuple)
    armStopRequested = Signal(str)
    rightHandEnableRequested = Signal(bool)
    rightHandPoseRequested = Signal()
    rightHandAxisTargetRequested = Signal(str, float)
    agvEnableRequested = Signal(bool)
    agvMoveRequested = Signal(str)
    agvNavigateRequested = Signal(str)
    agvChargeRequested = Signal()
    agvStopRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dof_widgets_by_axis: dict[str, UtilDoFWidget] = {}
        self.arm_panels_by_device: dict[str, ArmModulePanel] = {}
        self.enable_indicators: dict[str, CasiaIndicatorLight] = {}
        self._tab_panels_by_index: list[DebugModulePanel] = []
        self._extra_tabs_by_name: dict[str, QWidget] = {}
        self._read_only_panels_by_tab: dict[str, StatusAxisPanel] = {}
        self._tab_widget = QTabWidget(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

        self._build_tabs()

    def _make_discovery_client(self) -> WujiQmlinkerBaseClient:
        return WujiQmlinkerBaseClient()

    def _discover_runtime_structure(self) -> WujiRobotRuntimeStructure:
        client = self._make_discovery_client()
        try:
            return client.describe_robot_runtime_structure()
        finally:
            client.close()

    def _convert_module(self, module: WujiRuntimeModuleSpec) -> DebugModuleSpec:
        return DebugModuleSpec(
            tab_name=module.tab_name,
            title=module.title,
            device_name=module.device_name,
            axes=tuple(
                DebugAxisSpec(
                    axis_name=axis.axis_name,
                    limit=JointLimit(axis.axis_name, axis.minimum, axis.maximum, axis.unit),
                    step=self._resolve_axis_step(axis.axis_name, axis.unit),
                    hold_step=self._resolve_axis_hold_step(axis.axis_name, axis.unit),
                    control_supported=axis.control_supported,
                    refresh_supported=axis.refresh_supported,
                )
                for axis in module.axes
            ),
            enable_supported=module.enable_supported,
            refresh_supported=module.refresh_supported,
        )

    def _resolve_axis_step(self, axis_name: str, unit: str) -> float:
        """返回调试界面单轴步进值。"""

        if axis_name == "body_z":
            return 5.0
        if unit == "deg":
            return 1.0
        return 1.0

    def _resolve_axis_hold_step(self, axis_name: str, unit: str) -> float:
        """返回调试界面长按循环增量。"""

        if axis_name == "body_z":
            return 5.0
        if unit == "deg":
            return 1.0
        return self._resolve_axis_step(axis_name, unit)

    def _build_tabs(self) -> None:
        specs_by_tab: dict[str, list[DebugModuleSpec]] = {"body": [], "arm": []}
        right_hand_axis_names: list[str] = []
        agv_axis_names: list[str] = []
        arm_specs_by_device: dict[str, list[ArmAxisSpec]] = {}
        for module in self._discover_runtime_structure().modules:
            spec = self._convert_module(module)
            if spec.tab_name in {"body", "arm"}:
                specs_by_tab.setdefault(spec.tab_name, []).append(spec)
                if spec.tab_name == "arm":
                    arm_specs_by_device[spec.device_name] = [
                        ArmAxisSpec(axis.axis_name, axis.limit.minimum, axis.limit.maximum, axis.step)
                        for axis in spec.axes
                    ]
                continue
            if spec.tab_name == "hand":
                right_hand_axis_names.extend(axis.axis_name for axis in spec.axes)
                continue
            if spec.tab_name == "agv":
                agv_axis_names.extend(axis.axis_name for axis in spec.axes)
                continue

        panel = DebugModulePanel(tuple(specs_by_tab["body"]), self._tab_widget)
        panel.dofTargetRequested.connect(self._on_dof_target_requested)
        panel.enableToggleRequested.connect(self._request_enable_toggle)
        self.dof_widgets_by_axis.update(panel.dof_widgets_by_axis)
        self.enable_indicators.update(panel.enable_indicators)
        self._tab_widget.addTab(self._wrap_scroll_area(panel), "Body")
        self._tab_panels_by_index.append(panel)

        for device_name, title in (("left_arm", "Left Arm"), ("right_arm", "Right Arm")):
            arm_panel = ArmModulePanel(device_name, title, tuple(arm_specs_by_device.get(device_name, ())), self._tab_widget)
            arm_panel.axisTargetRequested.connect(self._on_dof_target_requested)
            arm_panel.ikRequested.connect(self._on_arm_ik_requested)
            arm_panel.stopRequested.connect(self._on_arm_stop_requested)
            arm_panel.enableToggleRequested.connect(
                lambda _arm_name, enabled, name=device_name: self.enableToggleRequested.emit(name, enabled)
            )
            self.arm_panels_by_device[device_name] = arm_panel
            self._tab_widget.addTab(self._wrap_scroll_area(arm_panel), title)

        right_hand_panel = RightHandControlPanel(tuple(right_hand_axis_names), self._tab_widget)
        right_hand_panel.enableRequested.connect(lambda enabled: self.rightHandEnableRequested.emit(enabled))
        right_hand_panel.setPoseRequested.connect(lambda: self.rightHandPoseRequested.emit())
        right_hand_panel.axisTargetRequested.connect(lambda axis_name, value: self.rightHandAxisTargetRequested.emit(axis_name, value))
        self._read_only_panels_by_tab["right_hand"] = right_hand_panel
        self._tab_widget.addTab(self._wrap_scroll_area(right_hand_panel), "Right Hand")

        agv_panel = AgvControlPanel(tuple(agv_axis_names), self._tab_widget)
        agv_panel.enableRequested.connect(lambda enabled: self.agvEnableRequested.emit(enabled))
        agv_panel.moveRequested.connect(lambda direction: self.agvMoveRequested.emit(direction))
        agv_panel.navigateRequested.connect(lambda target_name: self.agvNavigateRequested.emit(target_name))
        agv_panel.chargeRequested.connect(lambda: self.agvChargeRequested.emit())
        agv_panel.stopRequested.connect(lambda: self.agvStopRequested.emit())
        self._read_only_panels_by_tab["agv"] = agv_panel
        self._tab_widget.addTab(self._wrap_scroll_area(agv_panel), "AGV")

    def ensure_extra_tab(self, tab_name: str, title: str, widget: QWidget) -> None:
        if tab_name in self._extra_tabs_by_name:
            return
        self._extra_tabs_by_name[tab_name] = widget
        self._tab_widget.addTab(self._wrap_scroll_area(widget), title)

    def _wrap_scroll_area(self, widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea(self._tab_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area

    @Slot(str, float)
    def _on_dof_target_requested(self, axis_name: str, value: float) -> None:
        self.dofTargetRequested.emit(axis_name, value)

    def update_dof_value(self, axis_name: str, value: float) -> None:
        widget = self.dof_widgets_by_axis.get(axis_name)
        if widget is not None:
            widget.update_feedback_value(value)

    def update_dof_values(self, values: dict[str, float]) -> None:
        for axis_name, value in values.items():
            self.update_dof_value(axis_name, value)
        for arm_panel in self.arm_panels_by_device.values():
            arm_panel.update_axis_readouts(values)
        self._update_read_only_values(values)

    def request_dof_value_refresh(self, axis_name: str) -> None:
        if axis_name in self.dof_widgets_by_axis:
            self.dofValueRefreshRequested.emit(axis_name)

    def request_all_dof_values_refresh(self) -> None:
        for axis_name in self.visible_refresh_axis_names():
            self.dofValueRefreshRequested.emit(axis_name)

    def update_enable_state(self, device_name: str, enabled: bool) -> None:
        indicator = self.enable_indicators.get(device_name)
        if indicator is not None:
            indicator.set_status(enabled)
        arm_panel = self.arm_panels_by_device.get(device_name)
        if arm_panel is not None:
            arm_panel.update_enable_state(enabled)
        read_only_panel = self._read_only_panels_by_tab.get(device_name)
        if read_only_panel is not None:
            read_only_panel.update_enable_state(enabled)

    def update_enable_states(self, states: dict[str, bool]) -> None:
        for device_name, enabled in states.items():
            self.update_enable_state(device_name, enabled)

    def update_arm_fk_pose(self, device_name: str, pose: object) -> None:
        arm_panel = self.arm_panels_by_device.get(device_name)
        if arm_panel is not None:
            arm_panel.update_fk_pose(pose)

    def update_arm_ik_inputs(self, device_name: str, values: tuple[float, ...]) -> None:
        arm_panel = self.arm_panels_by_device.get(device_name)
        if arm_panel is not None:
            arm_panel.update_ik_inputs(values)

    def update_arm_ik_result(self, device_name: str, values: tuple[float, ...] | list[float] | None) -> None:
        arm_panel = self.arm_panels_by_device.get(device_name)
        if arm_panel is not None:
            arm_panel.update_ik_result(values)

    def _update_read_only_values(self, values: dict[str, float]) -> None:
        right_hand_panel = self._read_only_panels_by_tab.get("right_hand")
        if right_hand_panel is not None:
            right_hand_panel.update_values(values, lambda _axis, value: f"{value:.3f}")
        agv_panel = self._read_only_panels_by_tab.get("agv")
        if agv_panel is not None:
            def _format_agv(axis_name: str, value: float) -> str:
                if axis_name == "agv_battery":
                    return f"{value:.0f}%"
                if axis_name == "agv_yaw":
                    return f"{value:.1f} deg"
                return f"{value:.3f}"

            agv_panel.update_values(values, _format_agv)

    @Slot(str, tuple, tuple)
    def _on_arm_ik_requested(self, device_name: str, target_pose: tuple[float, ...], reference_joints: tuple[float, ...]) -> None:
        self.armIkRequested.emit(device_name, target_pose, reference_joints)

    @Slot(str)
    def _on_arm_stop_requested(self, device_name: str) -> None:
        self.armStopRequested.emit(device_name)

    def request_enable_state_refresh(self, device_name: str) -> None:
        if device_name in self.enable_indicators:
            self.enableStateRefreshRequested.emit(device_name)

    def request_all_enable_states_refresh(self) -> None:
        for device_name in self.visible_refresh_device_names():
            self.enableStateRefreshRequested.emit(device_name)

    def visible_refresh_axis_names(self) -> tuple[str, ...]:
        panel = self._current_panel()
        return tuple(panel.refresh_axis_names) if panel is not None else ()

    def visible_refresh_device_names(self) -> tuple[str, ...]:
        panel = self._current_panel()
        return tuple(panel.refresh_device_names) if panel is not None else ()

    def _current_panel(self) -> DebugModulePanel | None:
        index = self._tab_widget.currentIndex()
        if index < 0 or index >= len(self._tab_panels_by_index):
            return None
        return self._tab_panels_by_index[index]

    def _request_enable_toggle(self, device_name: str, enabled: bool) -> None:
        self.enableToggleRequested.emit(device_name, enabled)
