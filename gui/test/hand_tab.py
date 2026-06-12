from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gui.test.common import ActivatableTab, AxisControlConfig, AxisControlRow, HoldRepeatController, StreamReaderWorker
from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji.right_hand_client import WujiRightHandClient
from src.wuji.right_hand_specs import RIGHT_HAND_ACTUATOR_SPECS

FINGER_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("thumb", (0, 1, 2)),
    ("index", (3, 4)),
    ("middle", (5, 6)),
    ("ring", (7, 8)),
    ("little", (9, 10)),
)




class M11HandTabWidget(QWidget, ActivatableTab):
    STEP = 0.1

    # region 初始化

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._client: WujiRightHandClient | None = None
        self._active = False
        self._stream_run_id = 0
        self._current_values: dict[int, float] = {}
        self._axis_rows: dict[int, AxisControlRow] = {}
        self._stream_worker = StreamReaderWorker(self._stream_values, self)

        self.enable_indicator: CasiaIndicatorLight
        self.info_label: QLabel

        self._setup_ui()
        self._connect_signals()
        self.set_connection_ready(False)

    def _setup_ui(self) -> None:
        self.enable_indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        self.info_label = QLabel("m11 hand 未连接", self)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self._build_info_group())
        root_layout.addLayout(self._build_finger_layout())
        root_layout.addStretch(1)

    def _build_info_group(self) -> QGroupBox:
        group = QGroupBox("Hand Info", self)
        layout = QHBoxLayout(group)
        layout.addWidget(self.enable_indicator)
        layout.addWidget(self.info_label, 1)
        return group

    def _build_finger_layout(self) -> QHBoxLayout:
        finger_layout = QHBoxLayout()
        for finger_name, actuator_ids in FINGER_GROUPS:
            finger_layout.addWidget(self._build_finger_group(finger_name, actuator_ids), 1)
        return finger_layout

    def _build_finger_group(self, finger_name: str, actuator_ids: tuple[int, ...]) -> QGroupBox:
        group = QGroupBox(finger_name, self)
        layout = QVBoxLayout(group)
        for actuator_id in actuator_ids:
            layout.addWidget(self._build_axis_row(actuator_id))
        return group

    def _build_axis_row(self, actuator_id: int) -> AxisControlRow:
        spec = RIGHT_HAND_ACTUATOR_SPECS[actuator_id]
        row = AxisControlRow(
            AxisControlConfig(
                axis_key=str(actuator_id),
                title=spec.label,
                minimum=spec.minimum,
                maximum=spec.maximum,
                step=self.STEP,
                decimals=2,
            ),
            repeat_controller=HoldRepeatController(self),
            parent=self,
        )
        self._axis_rows[actuator_id] = row
        return row

    def _connect_signals(self) -> None:
        self.enable_indicator.clicked.connect(self._on_enable_clicked)
        for row in self._axis_rows.values():
            row.setRequested.connect(self._on_axis_target_requested)
            row.nudgeRequested.connect(self._on_axis_target_requested)
        self._stream_worker.valueReceived.connect(self._on_stream_values_received)
        self._stream_worker.errorRaised.connect(self._on_stream_error)

    # endregion

    # region 生命周期

    def set_client(self, client: WujiRightHandClient | None) -> None:
        self._client = client
        self.set_connection_ready(client is not None)
        if client is None:
            self._stream_worker.stop()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            self._stream_worker.stop()
            return
        if self._client is None:
            self.info_label.setText("m11 hand 未连接")
            return
        self._try_enable()
        self._stream_run_id = self._stream_worker.start()

    def set_connection_ready(self, ready: bool) -> None:
        enabled = bool(ready)
        self.enable_indicator.setEnabled(enabled)
        for row in self._axis_rows.values():
            row.set_row_enabled(enabled)
        if not enabled:
            self.enable_indicator.set_status(False)
            for row in self._axis_rows.values():
                row.set_current_value(None)

    # endregion

    # region 使能

    def _try_enable(self) -> None:
        if self._client is None:
            return
        try:
            if not self._client.get_enable():
                self._client.set_enable(True)
        except Exception:
            pass

    @Slot()
    def _on_enable_clicked(self) -> None:
        if self._client is None:
            return
        try:
            current_enabled = bool(self._client.get_enable())
            self._client.set_enable(not current_enabled)
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"hand 使能切换失败: {exc}")

    # endregion

    # region 刷新

    def _stream_values(self) -> Iterable[dict[str, float]]:
        if self._client is None:
            return ()
        return self._client.stream_right_hand_values()

    @Slot(object, int)
    def _on_stream_values_received(self, payload: object, run_id: int) -> None:
        if run_id != self._stream_run_id or not isinstance(payload, dict):
            return
        mapped: dict[int, float] = {}
        for axis_name, value in payload.items():
            if not isinstance(axis_name, str) or not isinstance(value, int | float):
                continue
            if not axis_name.startswith("right_hand_a"):
                continue
            actuator_text = axis_name.removeprefix("right_hand_a")
            if actuator_text.isdigit():
                mapped[int(actuator_text)] = float(value)
        if mapped:
            self._current_values.update(mapped)
        self._update_axis_rows()
        self._update_info_label()

    @Slot(str, int)
    def _on_stream_error(self, message: str, run_id: int) -> None:
        if run_id != self._stream_run_id:
            return
        self.info_label.setText(f"hand 流读取失败: {message}")

    def _update_axis_rows(self) -> None:
        for actuator_id, row in self._axis_rows.items():
            row.set_current_value(self._current_values.get(actuator_id))

    def _update_info_label(self) -> None:
        if self._client is None:
            return
        try:
            enabled = bool(self._client.get_enable())
            self.enable_indicator.set_status(enabled)
            self.info_label.setText(
                f"hand enable={enabled} values="
                + ", ".join(
                    f"a{actuator_id}={self._current_values.get(actuator_id, 0.0):.2f}"
                    for actuator_id in sorted(self._axis_rows)
                )
            )
        except Exception:
            self.info_label.setText("hand 状态读取异常")

    # endregion

    # region 控制

    @Slot(str, float)
    def _on_axis_target_requested(self, axis_key: str, value: float) -> None:
        if self._client is None:
            return
        actuator_id = int(axis_key)
        try:
            self._client.set_right_hand_axis(actuator_id, float(value))
            self._current_values[actuator_id] = float(value)
            self._axis_rows[actuator_id].set_current_value(float(value))
        except Exception as exc:  # noqa: BLE001
            self.info_label.setText(f"a{actuator_id} 设置失败: {exc}")

    # endregion
