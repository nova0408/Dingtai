from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import cos, sin
from threading import Event, Thread
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QWidget


class ActivatableTab:
    """供主窗口统一分发生命周期的轻量接口。"""

    def set_active(self, active: bool) -> None:
        raise NotImplementedError

    def set_connection_ready(self, ready: bool) -> None:
        raise NotImplementedError


class HoldRepeatController(QObject):
    """统一处理“点击一步，长按连发，松开立停”。"""

    def __init__(
        self,
        parent: QObject | None = None,
        interval_ms: int = 20,
        hold_delay_ms: int = 20,
    ) -> None:
        super().__init__(parent)
        self._hold_delay_timer = QTimer(self)
        self._hold_delay_timer.setSingleShot(True)
        self._hold_delay_timer.setInterval(hold_delay_ms)
        self._hold_delay_timer.timeout.connect(self._start_repeat)
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._on_timeout)
        self._callback: Callable[[], None] | None = None
        self._pressed = False

    def start(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._pressed = True
        self._on_timeout()
        self._hold_delay_timer.start()

    def stop(self) -> None:
        self._pressed = False
        self._hold_delay_timer.stop()
        self._timer.stop()
        self._callback = None

    def is_active(self) -> bool:
        return self._pressed

    def _start_repeat(self) -> None:
        if self._pressed and self._callback is not None:
            self._timer.start()

    def _on_timeout(self) -> None:
        if self._pressed and self._callback is not None:
            self._callback()


class StreamReaderWorker(QObject):
    """在后台线程消费阻塞式迭代器。"""

    valueReceived = Signal(object, int)
    errorRaised = Signal(str, int)
    finished = Signal(int)

    def __init__(
        self,
        stream_factory: Callable[[], Any],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._stream_factory = stream_factory
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._run_id = 0

    def start(self) -> int:
        self.stop()
        self._stop_event = Event()
        self._run_id += 1
        run_id = self._run_id
        self._thread = Thread(
            target=self._run,
            args=(run_id,),
            name=f"{self.__class__.__name__}-{run_id}",
            daemon=True,
        )
        self._thread.start()
        return run_id

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self, run_id: int) -> None:
        try:
            iterator = self._stream_factory()
            for value in iterator:
                if self._stop_event.is_set() or run_id != self._run_id:
                    break
                self.valueReceived.emit(value, run_id)
            self.finished.emit(run_id)
        except Exception as exc:  # noqa: BLE001
            if not self._stop_event.is_set() and run_id == self._run_id:
                self.errorRaised.emit(str(exc), run_id)
                self.finished.emit(run_id)


@dataclass(frozen=True, slots=True)
class AxisControlConfig:
    axis_key: str
    title: str
    minimum: float
    maximum: float
    step: float
    decimals: int = 1
    unit: str = ""


class AxisControlRow(QWidget):
    """通用单轴控制行。"""

    setRequested = Signal(str, float)
    nudgeRequested = Signal(str, float)

    def __init__(
        self,
        config: AxisControlConfig,
        repeat_controller: HoldRepeatController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self._repeat_controller = repeat_controller
        self._current_value: float | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(config.title, self)
        self.value_label = QLabel("-", self)
        self.value_label.setMinimumWidth(90)
        self.minus_button = QPushButton("-", self)
        self.plus_button = QPushButton("+", self)
        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setDecimals(config.decimals)
        self.spin_box.setRange(config.minimum, config.maximum)
        self.spin_box.setSingleStep(config.step)
        self.set_button = QPushButton("set", self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.minus_button)
        layout.addWidget(self.plus_button)
        layout.addWidget(self.spin_box)
        if config.unit:
            layout.addWidget(QLabel(config.unit, self))
        layout.addWidget(self.set_button)
        layout.addStretch(1)

        self.set_button.clicked.connect(self._on_set_clicked)
        self.minus_button.pressed.connect(lambda: self._repeat_controller.start(lambda: self._emit_nudge(-1.0)))
        self.plus_button.pressed.connect(lambda: self._repeat_controller.start(lambda: self._emit_nudge(1.0)))
        self.minus_button.released.connect(self._repeat_controller.stop)
        self.plus_button.released.connect(self._repeat_controller.stop)

    def set_current_value(self, value: float | None, *, suffix: str = "") -> None:
        self._current_value = value
        if value is None:
            self.value_label.setText("-")
            return
        display = f"{value:.{self.config.decimals}f}"
        if suffix:
            display = f"{display} {suffix}"
        self.value_label.setText(display)

    def set_row_enabled(self, enabled: bool) -> None:
        self.minus_button.setEnabled(enabled)
        self.plus_button.setEnabled(enabled)
        self.spin_box.setEnabled(enabled)
        self.set_button.setEnabled(enabled)

    def _on_set_clicked(self) -> None:
        self.setRequested.emit(self.config.axis_key, float(self.spin_box.value()))

    def _emit_nudge(self, direction: float) -> None:
        current_value = self._current_value
        if current_value is None:
            current_value = float(self.spin_box.value())
        target_value = current_value + direction * self.config.step
        target_value = min(max(target_value, self.config.minimum), self.config.maximum)
        self.nudgeRequested.emit(self.config.axis_key, float(target_value))


def gui_pose_to_matrix_m(pose_values: Sequence[float]) -> np.ndarray:
    """将 GUI 输入的 mm + deg 位姿转换为 m 制 4x4 变换矩阵。"""

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

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rz @ ry @ rx
    matrix[0, 3] = x_mm / 1000.0
    matrix[1, 3] = y_mm / 1000.0
    matrix[2, 3] = z_mm / 1000.0
    return matrix


def now_timestamp_ms() -> int:
    return int(time.time() * 1000.0)
