from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QApplication, QWidget

from gui.test_gui.TestRobotTab_ui import Ui_Form
from gui.test_gui.uitl_dof_widget_model import DoFWidgetModel
from gui.test_gui.uitl_dof_widget_view import UtilDoFWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.servers.common import JointLimit
from src.servers.wuji_ind_casia_arm import WujiIndCasiaArmServer


@dataclass(frozen=True, slots=True)
class DebugAxisBinding:
    """UI 占位控件与协议轴名的绑定关系。"""

    placeholder_name: str
    axis_name: str
    limit: JointLimit
    step: float = 1.0


class TestWujiCasiaArmWidget(QWidget):
    """无际 CASIA 整机调试页。"""

    dofTargetRequested = Signal(str, float)
    dofValueRefreshRequested = Signal(str)
    enableToggleRequested = Signal(str, bool)
    enableStateRefreshRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self._robot = WujiIndCasiaArmServer()
        self.dof_widgets: dict[str, UtilDoFWidget] = {}
        self.dof_widgets_by_axis: dict[str, UtilDoFWidget] = {}
        self.enable_indicators: dict[str, CasiaIndicatorLight] = {}

        self._replace_dof_placeholders()
        self._replace_enable_placeholders()

    def _replace_dof_placeholders(self) -> None:
        for binding in self._build_axis_bindings():
            placeholder = getattr(self.ui, binding.placeholder_name)
            model = DoFWidgetModel(
                name=binding.axis_name,
                minimum=binding.limit.minimum,
                maximum=binding.limit.maximum,
                unit=binding.limit.unit,
                step=binding.step,
            )
            widget = UtilDoFWidget.replace_placeholder(placeholder, model)
            widget.targetRequested.connect(self._on_dof_target_requested)
            self.dof_widgets[binding.placeholder_name] = widget
            self.dof_widgets_by_axis[binding.axis_name] = widget

    def _replace_enable_placeholders(self) -> None:
        enable_names = {
            "body_enable_2": "AGV",
            "body_enable": "body",
            "left_enable": "left_arm",
            "right_enable": "right_arm",
        }
        for placeholder_name, device_name in enable_names.items():
            placeholder = getattr(self.ui, placeholder_name)
            indicator = CasiaIndicatorLight.replace_placeholder(
                placeholder,
                text=("使能", "禁用"),
                font_size=12,
                default_status=False,
            )
            indicator.setObjectName(placeholder_name)
            indicator.setEnabled(True)
            indicator.clicked.connect(lambda name=device_name, light=indicator: self._request_enable_toggle(name, light))
            self.enable_indicators[device_name] = indicator

    def _build_axis_bindings(self) -> tuple[DebugAxisBinding, ...]:
        body_limits = {item.name: item for item in self._robot.body.get_movable_range()}
        left_limits = self._robot.left_arm.get_movable_range()
        right_limits = self._robot.right_arm.get_movable_range()
        agv_limits = (
            JointLimit("agv_x", -1.0, 1.0, "m"),
            JointLimit("agv_y", -1.0, 1.0, "m"),
            JointLimit("agv_yaw", -180.0, 180.0, "deg"),
        )

        return (
            DebugAxisBinding("dof_agv_x", "agv_x", agv_limits[0], 0.01),
            DebugAxisBinding("dof_agv_y", "agv_y", agv_limits[1], 0.01),
            DebugAxisBinding("dof_agv_yaw", "agv_yaw", agv_limits[2], 1.0),
            DebugAxisBinding("dof_roll", "head_yaw", body_limits["head_yaw"], 1.0),
            DebugAxisBinding("dof_pitch", "head_pitch", body_limits["head_pitch"], 1.0),
            *(
                DebugAxisBinding(f"dof_l_{idx}", f"left_{limit.name}", limit, 1.0)
                for idx, limit in enumerate(left_limits, start=1)
            ),
            *(
                DebugAxisBinding(f"dof_r_{idx}", f"right_{limit.name}", limit, 1.0)
                for idx, limit in enumerate(right_limits, start=1)
            ),
        )

    @Slot(str, float)
    def _on_dof_target_requested(self, axis_name: str, value: float) -> None:
        self.dofTargetRequested.emit(axis_name, value)

    def update_dof_value(self, axis_name: str, value: float) -> None:
        """用真实硬件反馈值刷新 DoF 显示。"""

        widget = self.dof_widgets_by_axis.get(axis_name)
        if widget is None:
            raise KeyError(f"未知 DoF 轴：{axis_name}")
        widget.update_feedback_value(value)

    def update_dof_values(self, values: dict[str, float]) -> None:
        """批量刷新真实硬件反馈值。"""

        for axis_name, value in values.items():
            self.update_dof_value(axis_name, value)

    def request_dof_value_refresh(self, axis_name: str) -> None:
        """请求外层读取单个 DoF 真实反馈值。"""

        if axis_name not in self.dof_widgets_by_axis:
            raise KeyError(f"未知 DoF 轴：{axis_name}")
        self.dofValueRefreshRequested.emit(axis_name)

    def request_all_dof_values_refresh(self) -> None:
        """请求外层读取全部 DoF 真实反馈值。"""

        for axis_name in self.dof_widgets_by_axis:
            self.dofValueRefreshRequested.emit(axis_name)

    def update_enable_state(self, device_name: str, enabled: bool) -> None:
        """用真实硬件使能状态刷新指示灯。"""

        indicator = self.enable_indicators.get(device_name)
        if indicator is None:
            raise KeyError(f"未知使能设备：{device_name}")
        indicator.set_status(enabled)

    def update_enable_states(self, states: dict[str, bool]) -> None:
        """批量刷新真实硬件使能状态。"""

        for device_name, enabled in states.items():
            self.update_enable_state(device_name, enabled)

    def request_enable_state_refresh(self, device_name: str) -> None:
        """请求外层读取真实硬件使能状态。"""

        if device_name not in self.enable_indicators:
            raise KeyError(f"未知使能设备：{device_name}")
        self.enableStateRefreshRequested.emit(device_name)

    def request_all_enable_states_refresh(self) -> None:
        """请求外层读取全部硬件使能状态。"""

        for device_name in self.enable_indicators:
            self.enableStateRefreshRequested.emit(device_name)

    def _request_enable_toggle(self, device_name: str, indicator: CasiaIndicatorLight) -> None:
        requested_status = not bool(indicator.property("status"))
        self.enableToggleRequested.emit(device_name, requested_status)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = TestWujiCasiaArmWidget()
    window.show()
    sys.exit(app.exec())
