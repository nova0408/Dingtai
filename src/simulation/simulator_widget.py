from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.util_components.casia_value_slider import CasiaValueSlider
from src.simulation.arm_kinematics_adapter import ArmSimulationModel
from src.simulation.qt_matplotlib_widget import MatplotKinematicsWidget
from src.robotics.kinematic_models import ArmMountState
from src.utils.datas import Degree, Radian

# region 数据结构


@dataclass(frozen=True, slots=True)
class JointSliderBinding:
    """关节滑条与元数据绑定"""

    joint_name: str
    """关节名称"""

    slider: CasiaValueSlider
    """关节滑条"""

    scale: float
    """滑条整数值到实际关节值的缩放系数"""


# endregion


# region Qt 组件


class KinematicsSimulationWidget(QWidget):
    """基于 `kinematic_models` 的三维仿真组件"""

    def __init__(self, model: ArmSimulationModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model

        self._plot_widget = MatplotKinematicsWidget(self)
        self._chain_combo = QComboBox(self)
        self._status_label = QLabel("Ready", self)

        self._target_x_slider = CasiaValueSlider(Qt.Orientation.Horizontal)
        self._target_y_slider = CasiaValueSlider(Qt.Orientation.Horizontal)
        self._target_z_slider = CasiaValueSlider(Qt.Orientation.Horizontal)
        self._solve_ik_button = QPushButton("Solve IK", self)

        self._joint_group = QGroupBox("Joints", self)
        self._joint_layout = QGridLayout(self._joint_group)
        self._joint_bindings: list[JointSliderBinding] = []

        self._target_xyz: tuple[float, float, float] | None = None

        self._setup_ui()
        self._connect_signals()
        self._reload_chain_options()
        self._refresh_from_model()

    def _setup_ui(self) -> None:
        """构建布局与控件"""

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        root.addWidget(self._plot_widget, stretch=3)

        right_panel = QVBoxLayout()
        chain_group = QGroupBox("Chain Selection", self)
        chain_layout = QVBoxLayout(chain_group)
        chain_layout.addWidget(self._chain_combo)
        right_panel.addWidget(chain_group)
        right_panel.addWidget(self._joint_group)

        target_group = QGroupBox("IK Target (m)", self)
        target_layout = QGridLayout(target_group)
        for slider, min_raw, max_raw, default_raw in (
            (self._target_x_slider, -1500, 1500, 300),
            (self._target_y_slider, -1500, 1500, 0),
            (self._target_z_slider, 0, 2000, 900),
        ):
            slider.setRange(min_raw, max_raw)
            slider.setValue(default_raw)
            slider.set_value_converter(lambda v: f"{v / 1000.0:.3f}")

        target_layout.addWidget(QLabel("Target X"), 0, 0)
        target_layout.addWidget(self._target_x_slider, 0, 1)
        target_layout.addWidget(QLabel("Target Y"), 1, 0)
        target_layout.addWidget(self._target_y_slider, 1, 1)
        target_layout.addWidget(QLabel("Target Z"), 2, 0)
        target_layout.addWidget(self._target_z_slider, 2, 1)
        target_layout.addWidget(self._solve_ik_button, 3, 0, 1, 2)

        right_panel.addWidget(target_group)
        right_panel.addWidget(self._status_label)
        right_panel.addStretch(1)

        right_holder = QWidget(self)
        right_holder.setLayout(right_panel)
        right_holder.setMinimumWidth(460)
        root.addWidget(right_holder, stretch=2)

    def _connect_signals(self) -> None:
        """连接 UI 信号槽"""

        self._chain_combo.currentTextChanged.connect(self._on_chain_changed)
        self._solve_ik_button.clicked.connect(self._on_solve_ik)
        self._target_x_slider.valueChanged.connect(self._on_target_changed)
        self._target_y_slider.valueChanged.connect(self._on_target_changed)
        self._target_z_slider.valueChanged.connect(self._on_target_changed)
        self._target_x_slider.sliderPressed.connect(self._on_slider_press)
        self._target_y_slider.sliderPressed.connect(self._on_slider_press)
        self._target_z_slider.sliderPressed.connect(self._on_slider_press)
        self._target_x_slider.sliderReleased.connect(self._on_slider_release)
        self._target_y_slider.sliderReleased.connect(self._on_slider_release)
        self._target_z_slider.sliderReleased.connect(self._on_slider_release)

    def _reload_chain_options(self) -> None:
        """刷新链下拉框内容"""

        self._chain_combo.clear()
        self._chain_combo.addItems(list(self._model.chain_names()))

    def _active_binding(self):
        """返回当前选中链绑定"""

        return self._model.get_binding(self._chain_combo.currentText())

    def _clear_joint_layout(self) -> None:
        """清空关节滑条区域"""

        while self._joint_layout.count() > 0:
            item = self._joint_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._joint_bindings.clear()

    def _build_joint_sliders(self) -> None:
        """根据当前链重建关节滑条"""

        self._clear_joint_layout()
        binding = self._active_binding()
        for row, (ui_spec, value) in enumerate(zip(binding.joint_ui, binding.arm_state.joint_positions, strict=True)):
            label = QLabel(ui_spec.name)
            slider = CasiaValueSlider(Qt.Orientation.Horizontal)
            min_value = ui_spec.min_value.value if isinstance(ui_spec.min_value, Degree) else ui_spec.min_value
            max_value = ui_spec.max_value.value if isinstance(ui_spec.max_value, Degree) else ui_spec.max_value
            slider.setRange(int(min_value * 10.0), int(max_value * 10.0))
            slider.setValue(int(value * 10.0))
            slider.set_value_converter(lambda raw: f"{raw / 10.0:.1f}")
            slider.valueChanged.connect(self._on_joint_slider_changed)
            slider.sliderPressed.connect(self._on_slider_press)
            slider.sliderReleased.connect(self._on_slider_release)

            self._joint_layout.addWidget(label, row, 0)
            self._joint_layout.addWidget(slider, row, 1)
            self._joint_bindings.append(JointSliderBinding(ui_spec.name, slider, 10.0))

    def _collect_slider_values(self) -> tuple[float, ...]:
        """读取滑条当前值并转换为关节值序列"""

        return tuple(binding.slider.value() / binding.scale for binding in self._joint_bindings)

    @staticmethod
    def _to_display_float(value: Degree | Radian | float) -> float:
        """将关节值转换为 UI 可显示的浮点值"""

        if isinstance(value, Degree):
            return value.value
        if isinstance(value, Radian):
            return Degree.from_radians(value.value).value
        return float(value)

    def _on_chain_changed(self, _: str) -> None:
        """切换链时重建关节面板"""

        self._build_joint_sliders()
        self._refresh_from_model()

    def _on_joint_slider_changed(self) -> None:
        """滑条改变后更新模型并重绘"""

        chain_name = self._chain_combo.currentText()
        self._model.set_joint_positions(chain_name, self._collect_slider_values())
        self._status_label.setText(f"Updated joints for {chain_name}")
        self._refresh_plot()

    def _on_target_changed(self) -> None:
        """更新 IK 目标点"""

        self._target_xyz = (
            self._target_x_slider.value() / 1000.0,
            self._target_y_slider.value() / 1000.0,
            self._target_z_slider.value() / 1000.0,
        )
        self._refresh_plot()

    def _on_slider_press(self) -> None:
        """滑条拖动开始：临时屏蔽画布交互，避免误旋转。"""

        self._plot_widget.set_interaction_enabled(False)

    def _on_slider_release(self) -> None:
        """滑条拖动结束：恢复画布交互。"""

        self._plot_widget.set_interaction_enabled(True)

    def _on_solve_ik(self) -> None:
        """触发当前链 IK 求解并同步 UI"""

        chain_name = self._chain_combo.currentText()
        target_xyz = (
            self._target_x_slider.value() / 1000.0,
            self._target_y_slider.value() / 1000.0,
            self._target_z_slider.value() / 1000.0,
        )
        result = self._model.solve_chain_ik(chain_name, target_xyz)
        for slider_binding, value in zip(self._joint_bindings, result, strict=True):
            slider_binding.slider.blockSignals(True)
            slider_binding.slider.setValue(int(self._to_display_float(value) * slider_binding.scale))
            slider_binding.slider.blockSignals(False)

        self._status_label.setText(f"IK solved for {chain_name}")
        self._target_xyz = target_xyz
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        """刷新三维绘图"""

        self._plot_widget.render_snapshots(
            self._model.snapshots(),
            target_xyz=self._target_xyz,
            active_chain=self._chain_combo.currentText(),
        )

    def _refresh_from_model(self) -> None:
        """从模型状态刷新 UI 与图像"""

        if not self._joint_bindings:
            self._build_joint_sliders()
        self._status_label.setText(f"Active chain: {self._chain_combo.currentText()}")
        self._refresh_plot()

    def set_status_text(self, text: str) -> None:
        """设置状态栏文本。"""

        self._status_label.setText(text)

    def replace_model(self, model: ArmSimulationModel, preserve_joint_values: bool = True) -> None:
        """替换仿真模型，并按关节名称尽量保留当前关节值。"""

        selected_chain = self._chain_combo.currentText()
        old_model = self._model

        if preserve_joint_values:
            for chain_name in old_model.chain_names():
                if chain_name not in model.bindings:
                    continue
                old_binding = old_model.get_binding(chain_name)
                new_binding = model.get_binding(chain_name)
                old_values = {ui.name: value for ui, value in zip(old_binding.joint_ui, old_binding.arm_state.joint_positions, strict=True)}
                merged_positions = tuple(old_values.get(ui.name, value) for ui, value in zip(new_binding.joint_ui, new_binding.arm_state.joint_positions, strict=True))
                new_binding.arm_state = ArmMountState(
                    lift_end_to_shoulder=new_binding.arm_state.lift_end_to_shoulder,
                    joint_positions=merged_positions,
                )

        self._model = model
        self._reload_chain_options()
        if selected_chain and selected_chain in self._model.chain_names():
            self._chain_combo.setCurrentText(selected_chain)
        self._build_joint_sliders()
        self._refresh_from_model()


# endregion
