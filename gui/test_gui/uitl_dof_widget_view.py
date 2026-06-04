from __future__ import annotations

from typing import cast

from PySide6.QtCore import QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication, QWidget

from gui.test_gui.uitl_dof_widget_controller import DoFWidgetController
from gui.test_gui.uitl_dof_widget_model import DoFWidgetModel
from gui.test_gui.UtilDoFWidget_ui import Ui_Form
from gui.util_components.casia_value_converter import CasiaValueConverter
from gui.util_components.casia_value_slider import CasiaValueSlider


class DoFSliderValueConverter(CasiaValueConverter):
    """DoF 滑块整数值与真实物理值的转换器。"""

    def __init__(self, scale: int, unit: str) -> None:
        self._scale = scale
        self._unit = unit

    def convert(self, value: int) -> str:
        physical_value = value / self._scale
        suffix = f" {self._unit}" if self._unit else ""
        return f"{physical_value:.2f}{suffix}"

    def convert_edit(self, value: int) -> str:
        physical_value = value / self._scale
        return f"{physical_value:.2f}"

    def convert_back(self, text: str) -> int:
        value_text = text.strip()
        if self._unit and value_text.endswith(self._unit):
            value_text = value_text[: -len(self._unit)].strip()
        return int(round(float(value_text) * self._scale))


class UtilDoFWidget(QWidget):
    """单个自由度调试控件视图。"""

    targetRequested = Signal(str, float)
    valueChanged = Signal(str, float)

    _SLIDER_SCALE = 100
    _HOLD_SEND_INTERVAL_MS = 250

    @classmethod
    def replace_placeholder(
        cls,
        placeholder: QWidget,
        model: DoFWidgetModel,
    ) -> "UtilDoFWidget":
        if placeholder is None:
            raise ValueError("必须提供需要替换的 DoF 占位 widget")

        parent = placeholder.parentWidget()
        if parent is None:
            raise ValueError("DoF 占位 widget 缺少父控件")

        parent_layout = parent.layout()
        if parent_layout is None:
            raise ValueError("DoF 占位 widget 的父控件必须有 layout")

        widget = cls(model=model, parent=parent)
        widget.setObjectName(placeholder.objectName())
        parent_layout.replaceWidget(placeholder, widget)
        placeholder.deleteLater()
        return widget

    def __init__(self, model: DoFWidgetModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self._replace_slider()
        self._controller = DoFWidgetController(model, self)
        self._syncing_slider = False
        self._hold_direction = 0
        self._hold_timer = QTimer(self)
        self._hold_timer.setInterval(self._HOLD_SEND_INTERVAL_MS)

        self._setup_static_text()
        self._setup_slider_range()
        self._connect_signals()
        self.update_feedback_value(model.feedback_value)

    @property
    def model(self) -> DoFWidgetModel:
        return self._controller.model

    @property
    def value_slider(self) -> CasiaValueSlider:
        return cast(CasiaValueSlider, self.ui.value_HSlider)

    def _replace_slider(self) -> None:
        old_slider = self.ui.value_HSlider
        parent = old_slider.parentWidget()
        if parent is None:
            raise ValueError("DoF 滑块占位控件必须有父控件和 layout")
        parent_layout = parent.layout()
        if parent_layout is None:
            raise ValueError("DoF 滑块占位控件必须有父控件和 layout")

        slider = CasiaValueSlider(old_slider.orientation(), parent)
        slider.setObjectName(old_slider.objectName())
        slider.set_drag_edit_enabled(False)
        slider.set_click_edit_enabled(True)
        slider.set_slider_width_chars(8)

        parent_layout.replaceWidget(old_slider, slider)
        old_slider.deleteLater()
        self.ui.value_HSlider = slider

    def _setup_static_text(self) -> None:
        model = self.model
        self.ui.dof_label.setText(model.name)
        self.ui.min_label.setText(self._format_value(model.minimum))
        self.ui.max_label.setText(self._format_value(model.maximum))

    def _setup_slider_range(self) -> None:
        self.value_slider.setRange(self._to_slider_value(self.model.minimum), self._to_slider_value(self.model.maximum))
        self.value_slider.set_value_converter(DoFSliderValueConverter(self._SLIDER_SCALE, self.model.unit))

    def _connect_signals(self) -> None:
        self.value_slider.valueChanged.connect(self._on_slider_value_changed)
        self.ui.forward_button.pressed.connect(lambda: self._start_hold(-1))
        self.ui.forward_button.released.connect(self._stop_hold)
        self.ui.backward_button.pressed.connect(lambda: self._start_hold(1))
        self.ui.backward_button.released.connect(self._stop_hold)
        self._hold_timer.timeout.connect(self._send_hold_step)
        self._controller.feedbackValueChanged.connect(self._refresh_feedback_value)
        self._controller.targetRequested.connect(self.targetRequested)

    def _start_hold(self, direction: int) -> None:
        """启动单个方向的连续关节目标发送。"""

        self._hold_direction = direction
        self._send_hold_step(first_step=True)
        self._hold_timer.start()

    def _stop_hold(self) -> None:
        """停止长按连续发送，避免释放后继续产生控制指令。"""

        self._hold_timer.stop()
        self._hold_direction = 0

    def _send_hold_step(self, first_step: bool = False) -> None:
        """按固定节流周期发送一步目标值。"""

        if self._hold_direction < 0:
            if first_step:
                self._controller.move_forward()
            else:
                self._controller.continue_forward()
            return
        if self._hold_direction > 0:
            if first_step:
                self._controller.move_backward()
            else:
                self._controller.continue_backward()

    @Slot(int)
    def _on_slider_value_changed(self, raw_value: int) -> None:
        if self._syncing_slider:
            return
        value = raw_value / self._SLIDER_SCALE
        self._controller.request_target_value(value)
        self._sync_slider_to_feedback()

    @Slot(float)
    def update_feedback_value(self, value: float) -> None:
        self._controller.update_feedback_value(value)

    @Slot(float)
    def _refresh_feedback_value(self, value: float) -> None:
        self.ui.dof_label.setText(f"{self.model.name}: {self._format_value(value)}")
        self._sync_slider_to_feedback()
        self.valueChanged.emit(self.model.name, value)

    def _sync_slider_to_feedback(self) -> None:
        self._syncing_slider = True
        self.value_slider.setValue(self._to_slider_value(self.model.feedback_value))
        self._syncing_slider = False

    def _to_slider_value(self, value: float) -> int:
        return int(round(value * self._SLIDER_SCALE))

    def _format_value(self, value: float) -> str:
        suffix = f" {self.model.unit}" if self.model.unit else ""
        return f"{value:.0f}{suffix}"


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = UtilDoFWidget(DoFWidgetModel(name="j1", minimum=-180.0, maximum=180.0, unit="deg"))
    window.show()
    sys.exit(app.exec())
