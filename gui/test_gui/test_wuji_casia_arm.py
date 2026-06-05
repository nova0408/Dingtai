from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QApplication, QGroupBox, QHBoxLayout, QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from gui.test_gui.uitl_dof_widget_model import DoFWidgetModel
from gui.test_gui.uitl_dof_widget_view import UtilDoFWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.agv import WUJI_AGV_STATUS_AXES
from src.arm import WUJI_ARM_JOINT_LIMITS_DEG
from src.hand import WUJI_HAND_SPECS, load_wuji_hand_instances
from src.servers.common import JointLimit
from loguru import logger

@dataclass(frozen=True, slots=True)
class DebugAxisSpec:
    """单个可动轴在调试界面中的显示与控制规格。"""

    axis_name: str
    limit: JointLimit
    step: float = 1.0
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
    """根据模块规格动态生成使能指示灯与 DoF 控件。"""

    dofTargetRequested = Signal(str, float)
    enableToggleRequested = Signal(str, bool)

    def __init__(self, specs: tuple[DebugModuleSpec, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dof_widgets_by_axis: dict[str, UtilDoFWidget] = {}
        self.refresh_axis_names: list[str] = []
        self.enable_indicators: dict[str, CasiaIndicatorLight] = {}
        self.refresh_device_names: list[str] = []

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._build_modules(specs)
        self._main_layout.addStretch(1)

    def _build_modules(self, specs: tuple[DebugModuleSpec, ...]) -> None:
        for spec in specs:
            group = QGroupBox(spec.title, self)
            group_layout = QVBoxLayout(group)
            group_layout.addLayout(self._create_enable_row(group, spec))
            self._add_axis_widgets(group_layout, group, spec.axes)
            self._main_layout.addWidget(group)

    def _create_enable_row(self, parent: QWidget, spec: DebugModuleSpec) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(QLabel("使能:", parent))

        indicator = CasiaIndicatorLight(
            parent,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        indicator.setObjectName(f"{spec.device_name}_enable")
        indicator.setEnabled(spec.enable_supported)
        if spec.enable_supported:
            indicator.clicked.connect(
                lambda name=spec.device_name, light=indicator: self._request_enable_toggle(name, light)
            )
        layout.addWidget(indicator)
        layout.addStretch(1)

        if spec.enable_supported:
            self.enable_indicators[spec.device_name] = indicator
            if spec.refresh_supported:
                self.refresh_device_names.append(spec.device_name)
        return layout

    def _add_axis_widgets(
        self,
        layout: QVBoxLayout,
        parent: QWidget,
        axes: tuple[DebugAxisSpec, ...],
    ) -> None:
        if not axes:
            empty_label = QLabel("未配置可动轴", parent)
            empty_label.setEnabled(False)
            layout.addWidget(empty_label)
            return

        for axis in axes:
            model = DoFWidgetModel(
                name=axis.axis_name,
                minimum=axis.limit.minimum,
                maximum=axis.limit.maximum,
                unit=axis.limit.unit,
                step=axis.step,
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
        if widget is None:
            raise KeyError(f"未知 DoF 轴：{axis_name}")
        widget.update_feedback_value(value)

    def update_enable_state(self, device_name: str, enabled: bool) -> None:
        indicator = self.enable_indicators.get(device_name)
        if indicator is None:
            raise KeyError(f"未知使能设备：{device_name}")
        indicator.set_status(enabled)

    def _request_enable_toggle(self, device_name: str, indicator: CasiaIndicatorLight) -> None:
        requested_status = not bool(indicator.property("status"))
        self.enableToggleRequested.emit(device_name, requested_status)


class TestWujiCasiaArmWidget(QWidget):
    """无际 CASIA 整机调试页。"""

    dofTargetRequested = Signal(str, float)
    dofValueRefreshRequested = Signal(str)
    enableToggleRequested = Signal(str, bool)
    enableStateRefreshRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dof_widgets: dict[str, UtilDoFWidget] = {}
        self.dof_widgets_by_axis: dict[str, UtilDoFWidget] = {}
        self.enable_indicators: dict[str, CasiaIndicatorLight] = {}
        self._tab_panels: dict[str, DebugModulePanel] = {}
        self._tab_panels_by_index: list[DebugModulePanel] = []

        self._tab_widget = QTabWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

        self._build_tabs()

    def _build_module_specs(self) -> tuple[DebugModuleSpec, ...]:
        left_limits = tuple(
            JointLimit(item.name, item.minimum_deg, item.maximum_deg, item.unit)
            for item in WUJI_ARM_JOINT_LIMITS_DEG["left_arm"]
        )
        right_limits = tuple(
            JointLimit(item.name, item.minimum_deg, item.maximum_deg, item.unit)
            for item in WUJI_ARM_JOINT_LIMITS_DEG["right_arm"]
        )
        body_limits = (
            JointLimit("body_z", 0.0, 850.0, "mm"),
            JointLimit("body_ry", -30.0, 30.0, "deg"),
        )
        head_limits = (
            JointLimit("head_yaw", -90.0, 90.0, "deg"),
        )

        return (
            DebugModuleSpec(
                tab_name="body",
                title="body",
                device_name="body",
                axes=(
                    DebugAxisSpec("body_z", body_limits[0], 10.0),
                    DebugAxisSpec("body_ry", body_limits[1], 1.0),
                ),
            ),
            DebugModuleSpec(
                tab_name="body",
                title="head",
                device_name="head",
                axes=(DebugAxisSpec("head_yaw", head_limits[0], 1.0),),
            ),
            DebugModuleSpec(
                tab_name="arm",
                title="left arm",
                device_name="left_arm",
                axes=tuple(
                    DebugAxisSpec(f"left_{limit.name}", limit, 1.0) for limit in left_limits
                ),
            ),
            DebugModuleSpec(
                tab_name="arm",
                title="right arm",
                device_name="right_arm",
                axes=tuple(
                    DebugAxisSpec(f"right_{limit.name}", limit, 1.0) for limit in right_limits
                ),
            ),
            DebugModuleSpec(
                tab_name="agv",
                title="AGV",
                device_name="agv",
                axes=tuple(
                    DebugAxisSpec(
                        axis.axis_name,
                        JointLimit(axis.axis_name, axis.minimum, axis.maximum, axis.unit),
                        1.0,
                        control_supported=False,
                    )
                    for axis in WUJI_AGV_STATUS_AXES
                ),
                enable_supported=False,
            ),
            *self._build_hand_module_specs(),
        )

    def _build_hand_module_specs(self) -> tuple[DebugModuleSpec, ...]:
        return tuple(
            DebugModuleSpec(
                tab_name="hand",
                title=f"{instance.title} ({instance.spec_name})",
                device_name=instance.device_name,
                axes=tuple(
                    DebugAxisSpec(
                        axis_name=f"{instance.device_name}_{limit.name}",
                        limit=JointLimit(limit.name, limit.minimum, limit.maximum, limit.unit),
                        step=0.05,
                        control_supported=False,
                    )
                    for limit in WUJI_HAND_SPECS[instance.spec_name]
                ),
                enable_supported=False,
            )
            for instance in load_wuji_hand_instances()
        )

    def _build_tabs(self) -> None:
        specs_by_tab: dict[str, list[DebugModuleSpec]] = {
            "body": [],
            "arm": [],
            "hand": [],
            "agv": [],
        }
        for spec in self._build_module_specs():
            specs_by_tab.setdefault(spec.tab_name, []).append(spec)

        for tab_name, title in (("body", "Body"), ("arm", "Arm"), ("hand", "Hand"), ("agv", "AGV")):
            panel = DebugModulePanel(tuple(specs_by_tab[tab_name]), self._tab_widget)
            panel.dofTargetRequested.connect(self._on_dof_target_requested)
            panel.enableToggleRequested.connect(self._request_enable_toggle)
            self.dof_widgets_by_axis.update(panel.dof_widgets_by_axis)
            self.enable_indicators.update(panel.enable_indicators)
            self._tab_panels[tab_name] = panel
            self._tab_widget.addTab(self._wrap_scroll_area(panel), title)
            self._tab_panels_by_index.append(panel)

    def _wrap_scroll_area(self, widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea(self._tab_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)
        return scroll_area

    @Slot(str, float)
    def _on_dof_target_requested(self, axis_name: str, value: float) -> None:
        self.dofTargetRequested.emit(axis_name, value)

    def update_dof_value(self, axis_name: str, value: float) -> None:
        """用真实硬件反馈值刷新 DoF 显示。"""

        widget = self.dof_widgets_by_axis.get(axis_name)
        if widget is None:
            # raise KeyError(f"未知 DoF 轴：{axis_name}")
            logger.warning(f"未知 DoF 轴：{axis_name}")
            return
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

        for axis_name in self.visible_refresh_axis_names():
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

        for device_name in self.visible_refresh_device_names():
            self.enableStateRefreshRequested.emit(device_name)

    def visible_refresh_axis_names(self) -> tuple[str, ...]:
        """返回当前显示 tab 中需要刷新的轴名。"""

        panel = self._current_panel()
        if panel is None:
            return ()
        return tuple(panel.refresh_axis_names)

    def visible_refresh_device_names(self) -> tuple[str, ...]:
        """返回当前显示 tab 中需要刷新的使能设备名。"""

        panel = self._current_panel()
        if panel is None:
            return ()
        return tuple(panel.refresh_device_names)

    def _current_panel(self) -> DebugModulePanel | None:
        index = self._tab_widget.currentIndex()
        if index < 0 or index >= len(self._tab_panels_by_index):
            return None
        return self._tab_panels_by_index[index]

    @Slot(str, bool)
    def _request_enable_toggle(self, device_name: str, requested_status: bool) -> None:
        self.enableToggleRequested.emit(device_name, requested_status)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = TestWujiCasiaArmWidget()
    window.show()
    sys.exit(app.exec())
