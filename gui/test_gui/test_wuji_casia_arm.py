from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from gui.test_gui.uitl_dof_widget_model import DoFWidgetModel
from gui.test_gui.uitl_dof_widget_view import UtilDoFWidget
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.servers.common import JointLimit
from src.wuji.qmlinker_client import WujiQmlinkerClient
from src.wuji.qmlinker_protocol import WujiRobotRuntimeStructure, WujiRuntimeModuleSpec


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
            indicator.clicked.connect(lambda name=spec.device_name, light=indicator: self._request_enable_toggle(name, light))
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


class TestWujiCasiaArmWidget(QWidget):
    dofTargetRequested = Signal(str, float)
    dofValueRefreshRequested = Signal(str)
    enableToggleRequested = Signal(str, bool)
    enableStateRefreshRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dof_widgets_by_axis: dict[str, UtilDoFWidget] = {}
        self.enable_indicators: dict[str, CasiaIndicatorLight] = {}
        self._tab_panels_by_index: list[DebugModulePanel] = []
        self._tab_widget = QTabWidget(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

        self._build_tabs()

    def _make_discovery_client(self) -> WujiQmlinkerClient:
        return WujiQmlinkerClient()

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
        specs_by_tab: dict[str, list[DebugModuleSpec]] = {"body": [], "arm": [], "hand": [], "agv": []}
        for module in self._discover_runtime_structure().modules:
            spec = self._convert_module(module)
            specs_by_tab.setdefault(spec.tab_name, []).append(spec)

        for tab_name, title in (("body", "Body"), ("arm", "Arm"), ("hand", "Hand"), ("agv", "AGV")):
            panel = DebugModulePanel(tuple(specs_by_tab[tab_name]), self._tab_widget)
            panel.dofTargetRequested.connect(self._on_dof_target_requested)
            panel.enableToggleRequested.connect(self._request_enable_toggle)
            self.dof_widgets_by_axis.update(panel.dof_widgets_by_axis)
            self.enable_indicators.update(panel.enable_indicators)
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
        widget = self.dof_widgets_by_axis.get(axis_name)
        if widget is not None:
            widget.update_feedback_value(value)

    def update_dof_values(self, values: dict[str, float]) -> None:
        for axis_name, value in values.items():
            self.update_dof_value(axis_name, value)

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

    def update_enable_states(self, states: dict[str, bool]) -> None:
        for device_name, enabled in states.items():
            self.update_enable_state(device_name, enabled)

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
