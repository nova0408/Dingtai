from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from gui.test_gui.uitl_dof_widget_model import DoFWidgetModel


class DoFWidgetController(QObject):
    """单个自由度调试控件控制器。"""

    feedbackValueChanged = Signal(float)
    targetRequested = Signal(str, float)

    def __init__(self, model: DoFWidgetModel, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = model

    @property
    def model(self) -> DoFWidgetModel:
        return self._model

    @Slot(float)
    def update_feedback_value(self, value: float) -> None:
        valid_value = self._model.set_feedback_value(value)
        self.feedbackValueChanged.emit(valid_value)

    @Slot(float)
    def request_target_value(self, value: float) -> None:
        valid_value = self._model.set_command_value(value)
        self.targetRequested.emit(self._model.name, valid_value)

    @Slot()
    def move_forward(self) -> None:
        valid_value = self._model.offset_command_from_feedback(-self._model.step)
        self.targetRequested.emit(self._model.name, valid_value)

    @Slot()
    def move_backward(self) -> None:
        valid_value = self._model.offset_command_from_feedback(self._model.step)
        self.targetRequested.emit(self._model.name, valid_value)

    @Slot()
    def continue_forward(self) -> None:
        valid_value = self._model.offset_command(-self._model.step)
        self.targetRequested.emit(self._model.name, valid_value)

    @Slot()
    def continue_backward(self) -> None:
        valid_value = self._model.offset_command(self._model.step)
        self.targetRequested.emit(self._model.name, valid_value)
