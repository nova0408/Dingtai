from __future__ import annotations

import ast
import csv
import re
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSignalBlocker, Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# pyright: reportMissingImports=false


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"
DEFAULT_RECORD_DIR = PROJECT_ROOT / "record_left"  # 左臂轨迹 CSV 所在目录。
DEFAULT_LEFT_ARM_IP = "192.168.1.161"
DEFAULT_LOCAL_IP = "192.168.1.116"
DEFAULT_TOOL_NAME = "g_tool_0"  # 与手眼标定脚本一致的全局工具名称。
DEFAULT_WOBJ_NAME = "g_wobj_0"  # 与手眼标定脚本一致的全局工件名称。
JOINT_COUNT = 7
SLIDER_SCALE = 10  # 关节滑条精度，10 表示 0.1 deg。
JOINT_LIMITS_DEG: tuple[tuple[float, float], ...] = (
    (-178.0, 178.0),
    (-120.0, 120.0),
    (-178.0, 178.0),
    (-60.0, 145.0),
    (-178.0, 178.0),
    (-60.0, 60.0),
    (-60.0, 60.0),
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk.xcoresdk import xCoreSDK_python

# region 数据结构


@dataclass(slots=True)
class TrajectoryPoint:
    """一个可编辑的左臂轨迹点。"""

    file_index: int
    point_index: int
    source_row_index: int
    timestamp: str
    joints_deg: tuple[float, ...]
    # GUI / CSV 统一展示为 xyz(mm) + rpy XYZ(deg)；不要混入 SDK 的 m/rad 计算单位。
    tcp_values: tuple[Any, ...]
    raw_row: dict[str, str]

    @property
    def key(self) -> tuple[int, int]:
        return self.file_index, self.point_index


@dataclass(slots=True)
class TrajectoryGroup:
    """一个 CSV 文件对应的一组轨迹。"""

    path: Path
    fieldnames: list[str]
    raw_rows: list[dict[str, str]]
    points: list[TrajectoryPoint]
    visible: bool = True
    dirty: bool = False


@dataclass(frozen=True, slots=True)
class ContinuityMetric:
    """当前点相对相邻点的关节连续性指标，单位均为 deg。"""

    previous_l2_deg: float | None
    previous_max_deg: float | None
    next_l2_deg: float | None
    next_max_deg: float | None


# endregion


# region CSV 与计算


def _parse_list(raw_text: str) -> tuple[Any, ...]:
    parsed = ast.literal_eval(raw_text)
    if not isinstance(parsed, list):
        raise ValueError(f"字段不是列表：{raw_text!r}")
    return tuple(parsed)


def _file_sort_key(path: Path) -> tuple[int, int, str]:
    """优先按文件名前缀编号排序，无编号文件排在其后。"""

    match = re.match(r"^(\d+)", path.stem)
    if match is None:
        return 1, 0, path.name.casefold()
    return 0, int(match.group(1)), path.name.casefold()


def _load_trajectory_groups(record_dir: Path) -> list[TrajectoryGroup]:
    if not record_dir.exists():
        raise FileNotFoundError(f"轨迹目录不存在：{record_dir}")
    csv_paths = sorted(record_dir.glob("*.csv"), key=_file_sort_key)
    if not csv_paths:
        raise FileNotFoundError(f"轨迹目录中没有 CSV: {record_dir}")

    groups: list[TrajectoryGroup] = []
    for file_index, csv_path in enumerate(csv_paths):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV 缺少表头：{csv_path}")
            fieldnames = list(reader.fieldnames)
            raw_rows = [dict(row) for row in reader if row]

        points: list[TrajectoryPoint] = []
        for source_row_index, row in enumerate(raw_rows):
            if row.get("type", "").strip().lower() != "arm":
                continue
            joints_text = row.get("joints", "").strip()
            pose_text = row.get("pose", "").strip()
            if not joints_text.startswith("[") or not pose_text.startswith("["):
                continue
            joints = tuple(float(value) for value in _parse_list(joints_text))
            if len(joints) != JOINT_COUNT:
                raise ValueError(
                    f"{csv_path.name} 第 {source_row_index + 2} 行关节数为 {len(joints)}，应为 {JOINT_COUNT}"
                )
            tcp_values = _parse_list(pose_text)
            if len(tcp_values) < 6:
                raise ValueError(
                    f"{csv_path.name} 第 {source_row_index + 2} 行 pose 少于 6 项"
                )
            points.append(
                TrajectoryPoint(
                    file_index=file_index,
                    point_index=len(points),
                    source_row_index=source_row_index,
                    timestamp=row.get("timestamp", ""),
                    joints_deg=joints,
                    tcp_values=tcp_values,
                    raw_row=row,
                )
            )
        groups.append(TrajectoryGroup(csv_path, fieldnames, raw_rows, points))
    return groups


def _flatten_points(groups: list[TrajectoryGroup]) -> list[TrajectoryPoint]:
    return [point for group in groups for point in group.points]


def _joint_delta(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> tuple[float, float]:
    delta = np.asarray(rhs, dtype=np.float64) - np.asarray(lhs, dtype=np.float64)
    return float(np.linalg.norm(delta)), float(np.max(np.abs(delta)))


def _continuity_metrics(points: list[TrajectoryPoint]) -> list[ContinuityMetric]:
    metrics: list[ContinuityMetric] = []
    for index, point in enumerate(points):
        previous = (
            _joint_delta(points[index - 1].joints_deg, point.joints_deg)
            if index > 0
            else None
        )
        following = (
            _joint_delta(point.joints_deg, points[index + 1].joints_deg)
            if index + 1 < len(points)
            else None
        )
        metrics.append(
            ContinuityMetric(
                previous_l2_deg=None if previous is None else previous[0],
                previous_max_deg=None if previous is None else previous[1],
                next_l2_deg=None if following is None else following[0],
                next_max_deg=None if following is None else following[1],
            )
        )
    return metrics


def _format_values(values: tuple[Any, ...], digits: int = 2) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def _format_continuity(metric: ContinuityMetric) -> str:
    previous = (
        "起点"
        if metric.previous_l2_deg is None
        else f"前 L2 {metric.previous_l2_deg:.2f} / max {metric.previous_max_deg:.2f}"
    )
    following = (
        "终点"
        if metric.next_l2_deg is None
        else f"后 L2 {metric.next_l2_deg:.2f} / max {metric.next_max_deg:.2f}"
    )
    return f"{previous}; {following} deg"


def _fk_pose_to_tcp(fk_pose: Any, original_tcp: tuple[Any, ...]) -> tuple[Any, ...]:
    # SDK FK 返回 trans(m) + rpy(rad)，这里只在显示/落盘边界转换成 mm/deg。
    prefix = (
        float(fk_pose.trans[0]) * 1000.0,
        float(fk_pose.trans[1]) * 1000.0,
        float(fk_pose.trans[2]) * 1000.0,
        float(np.degrees(float(fk_pose.rpy[0]))),
        float(np.degrees(float(fk_pose.rpy[1]))),
        float(np.degrees(float(fk_pose.rpy[2]))),
    )
    suffix = original_tcp[6:]
    if len(suffix) >= 3:
        suffix = (
            bool(fk_pose.hasElbow),
            float(np.degrees(float(fk_pose.elbow))),
            list(fk_pose.confData),
            *suffix[3:],
        )
    return prefix + suffix


def _serialize_list(values: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, bool):
            parts.append("True" if value else "False")
        elif isinstance(value, (int, float, np.integer, np.floating)):
            parts.append(f"{float(value):.6f}")
        else:
            parts.append(repr(value))
    return "[" + ", ".join(parts) + "]"


def _snapshot_csv(csv_path: Path) -> Path:
    """保存 CSV 前在项目 `.archive` 中创建保留相对结构的快照。"""

    try:
        relative_path = csv_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"只允许保存项目目录内的 CSV: {csv_path}") from exc
    archive_path = PROJECT_ROOT / ".archive" / relative_path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    snapshot_path = archive_path.with_name(f"{archive_path.name}.{timestamp}.bak")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, snapshot_path)
    return snapshot_path


# endregion


# region Qt 界面


class TrajectoryOptimizerWindow(QMainWindow):
    """左臂多文件轨迹连续性查看与关节微调窗口。"""

    def __init__(self, record_dir: Path) -> None:
        super().__init__()
        self._record_dir = record_dir
        self._groups: list[TrajectoryGroup] = []
        self._points: list[TrajectoryPoint] = []
        self._metrics: list[ContinuityMetric] = []
        self._selected_index: int | None = None
        self._robot: Any | None = None
        self._robot_model: Any | None = None
        self._toolset: Any | None = None

        self._figure = Figure(figsize=(8.0, 7.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        # Matplotlib 当前类型声明不会把 projection="3d" 收窄为 Axes3D。
        self._axes: Any = self._figure.add_subplot(111, projection="3d")
        self._visibility_layout = QHBoxLayout()
        self._visibility_checks: list[QCheckBox] = []
        self._tabs = QTabWidget(self)
        self._joint_figure = Figure(figsize=(7.0, 5.0))
        self._joint_canvas = FigureCanvasQTAgg(self._joint_figure)
        self._joint_axes = self._joint_figure.add_subplot(111)
        self._table = QTableWidget(self)
        self._detail_label = QLabel("请选择轨迹点", self)
        self._point_combo = QComboBox(self)
        self._previous_button = QPushButton("向前", self)
        self._next_button = QPushButton("向后", self)
        self._tcp_label = QLabel("TCP: -", self)
        self._continuity_label = QLabel("连续性：-", self)
        self._joint_sliders: list[QSlider] = []
        self._joint_spins: list[QDoubleSpinBox] = []
        self._save_button = QPushButton("保存全部改动", self)
        self._reload_button = QPushButton("重新加载", self)
        self._status_label = QLabel("未加载", self)

        self._setup_ui()
        self._connect_signals()
        self._load_data()

    def _setup_ui(self) -> None:
        self.setWindowTitle("左臂行动轨迹优化")
        self.resize(1780, 980)
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel(f"轨迹目录：{self._record_dir}", self), 1)
        toolbar.addWidget(self._reload_button)
        toolbar.addWidget(self._save_button)
        root_layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        visibility_scroll = QScrollArea(self)
        visibility_scroll.setWidgetResizable(True)
        visibility_scroll.setFixedHeight(64)
        visibility_widget = QWidget(visibility_scroll)
        visibility_widget.setLayout(self._visibility_layout)
        visibility_scroll.setWidget(visibility_widget)
        left_layout.addWidget(visibility_scroll)
        left_layout.addWidget(self._canvas, 1)
        left_layout.addWidget(self._status_label)
        splitter.addWidget(left_panel)

        self._setup_list_tab()
        self._setup_adjust_tab()
        self._setup_joint_curve_tab()
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)

    def _setup_list_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["轨迹组", "点", "关节角 (deg)", "TCP (mm/deg)", "关节连续性 (deg)"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._detail_label.setMinimumHeight(150)
        layout.addWidget(self._detail_label)
        self._tabs.addTab(page, "轨迹列表")

    def _setup_adjust_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        selection_row = QHBoxLayout()
        selection_row.addWidget(self._point_combo, 1)
        selection_row.addWidget(self._previous_button)
        selection_row.addWidget(self._next_button)
        layout.addLayout(selection_row)
        info_row = QHBoxLayout()
        self._tcp_label.setWordWrap(True)
        self._continuity_label.setWordWrap(True)
        info_row.addWidget(self._tcp_label, 1)
        info_row.addWidget(self._continuity_label, 1)
        layout.addLayout(info_row)

        sliders_widget = QWidget(self)
        sliders_layout = QGridLayout(sliders_widget)
        for joint_index in range(JOINT_COUNT):
            title = QLabel(f"J{joint_index + 1}", self)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            slider = QSlider(Qt.Orientation.Vertical, self)
            slider.setMinimumHeight(520)
            spin = QDoubleSpinBox(self)
            spin.setDecimals(1)
            spin.setSingleStep(0.1)
            spin.setSuffix("°")
            sliders_layout.addWidget(
                title, 0, joint_index, alignment=Qt.AlignmentFlag.AlignCenter
            )
            sliders_layout.addWidget(
                slider, 1, joint_index, alignment=Qt.AlignmentFlag.AlignCenter
            )
            sliders_layout.addWidget(spin, 2, joint_index)
            self._joint_sliders.append(slider)
            self._joint_spins.append(spin)
        layout.addWidget(sliders_widget, 1)
        self._tabs.addTab(page, "关节微调")

    def _setup_joint_curve_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(self._joint_canvas, 1)
        self._tabs.addTab(page, "关节曲线")

    def _connect_signals(self) -> None:
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._point_combo.currentIndexChanged.connect(self._on_combo_index_changed)
        self._previous_button.clicked.connect(self._select_previous_point)
        self._next_button.clicked.connect(self._select_next_point)
        self._reload_button.clicked.connect(self._load_data)
        self._save_button.clicked.connect(self._save_all)
        for joint_index, slider in enumerate(self._joint_sliders):
            slider.valueChanged.connect(
                lambda value, index=joint_index: self._on_slider_changed(index, value)
            )
        for joint_index, spin in enumerate(self._joint_spins):
            spin.valueChanged.connect(
                lambda value, index=joint_index: self._on_spin_changed(index, value)
            )

    @Slot()
    def _load_data(self) -> None:
        try:
            groups = _load_trajectory_groups(self._record_dir)
        except Exception as exc:
            self._status_label.setText(str(exc))
            QMessageBox.warning(self, "轨迹加载失败", str(exc))
            return
        self._groups = groups
        self._points = _flatten_points(groups)
        self._metrics = _continuity_metrics(self._points)
        self._selected_index = 0 if self._points else None
        self._rebuild_visibility_checks()
        self._configure_joint_ranges()
        self._refresh_point_combo()
        self._refresh_table()
        self._sync_selection_to_ui()
        self._refresh_plot()
        self._refresh_joint_curves()
        self._status_label.setText(
            f"已按编号顺序加载 {len(groups)} 组、{len(self._points)} 个轨迹点"
        )

    def _rebuild_visibility_checks(self) -> None:
        while self._visibility_layout.count():
            item = self._visibility_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._visibility_checks.clear()
        for group_index, group in enumerate(self._groups):
            checkbox = QCheckBox(f"{group.path.name} ({len(group.points)})", self)
            checkbox.setChecked(group.visible)
            checkbox.toggled.connect(
                lambda checked, index=group_index: self._on_group_visibility_changed(
                    index, checked
                )
            )
            self._visibility_layout.addWidget(checkbox)
            self._visibility_checks.append(checkbox)
        self._visibility_layout.addStretch(1)

    def _configure_joint_ranges(self) -> None:
        if not self._points:
            return
        for (minimum, maximum), slider, spin in zip(
            JOINT_LIMITS_DEG,
            self._joint_sliders,
            self._joint_spins,
            strict=True,
        ):
            slider.setRange(
                round(minimum * SLIDER_SCALE), round(maximum * SLIDER_SCALE)
            )
            spin.setRange(minimum, maximum)

    def _refresh_point_combo(self) -> None:
        with QSignalBlocker(self._point_combo):
            self._point_combo.clear()
            for point in self._points:
                group_name = self._groups[point.file_index].path.name
                self._point_combo.addItem(f"{group_name} / 点 {point.point_index + 1}")

    def _refresh_table(self) -> None:
        self._metrics = _continuity_metrics(self._points)
        with QSignalBlocker(self._table):
            self._table.setRowCount(len(self._points))
            for row_index, point in enumerate(self._points):
                group = self._groups[point.file_index]
                values = (
                    group.path.name,
                    str(point.point_index + 1),
                    _format_values(point.joints_deg),
                    _format_values(point.tcp_values[:6]),
                    _format_continuity(self._metrics[row_index]),
                )
                for column, value in enumerate(values):
                    self._table.setItem(row_index, column, QTableWidgetItem(value))

    def _sync_selection_to_ui(self) -> None:
        if self._selected_index is None or not self._points:
            return
        point = self._points[self._selected_index]
        metric = self._metrics[self._selected_index]
        with QSignalBlocker(self._table), QSignalBlocker(self._point_combo):
            self._table.selectRow(self._selected_index)
            self._point_combo.setCurrentIndex(self._selected_index)
        for joint_index, value in enumerate(point.joints_deg):
            with QSignalBlocker(self._joint_sliders[joint_index]), QSignalBlocker(
                self._joint_spins[joint_index]
            ):
                self._joint_sliders[joint_index].setValue(round(value * SLIDER_SCALE))
                self._joint_spins[joint_index].setValue(value)
        group = self._groups[point.file_index]
        detail_lines = [
            f"文件：{group.path}",
            f"文件内点序号：{point.point_index + 1}",
            f"CSV 原始行号：{point.source_row_index + 2}",
            f"时间戳：{point.timestamp}",
            f"关节角 (deg): {_format_values(point.joints_deg, 6)}",
            f"TCP xyz(mm) + rpy XYZ(deg): {_format_values(point.tcp_values[:6], 6)}",
            f"连续性：{_format_continuity(metric)}",
        ]
        self._detail_label.setText("\n".join(detail_lines))
        self._tcp_label.setText(
            f"TCP xyz(mm) + rpy XYZ(deg)\n{_format_values(point.tcp_values[:6], 3)}"
        )
        self._continuity_label.setText(f"关节连续性\n{_format_continuity(metric)}")
        self._previous_button.setEnabled(self._selected_index > 0)
        self._next_button.setEnabled(self._selected_index + 1 < len(self._points))

    @Slot()
    def _on_table_selection_changed(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if indexes:
            self._select_point(int(indexes[0].row()))

    @Slot(int)
    def _on_combo_index_changed(self, index: int) -> None:
        self._select_point(index)

    def _select_point(self, index: int) -> None:
        if index < 0 or index >= len(self._points):
            return
        self._selected_index = index
        self._sync_selection_to_ui()
        self._refresh_plot()
        self._refresh_joint_curves()

    @Slot()
    def _select_previous_point(self) -> None:
        if self._selected_index is not None:
            self._select_point(self._selected_index - 1)

    @Slot()
    def _select_next_point(self) -> None:
        if self._selected_index is not None:
            self._select_point(self._selected_index + 1)

    @Slot(int, bool)
    def _on_group_visibility_changed(self, group_index: int, checked: bool) -> None:
        self._groups[group_index].visible = checked
        self._refresh_plot()
        self._refresh_joint_curves()

    @Slot(int, int)
    def _on_slider_changed(self, joint_index: int, slider_value: int) -> None:
        value_deg = slider_value / SLIDER_SCALE
        with QSignalBlocker(self._joint_spins[joint_index]):
            self._joint_spins[joint_index].setValue(value_deg)
        self._apply_joint_value(joint_index, value_deg)

    @Slot(int, float)
    def _on_spin_changed(self, joint_index: int, value_deg: float) -> None:
        with QSignalBlocker(self._joint_sliders[joint_index]):
            self._joint_sliders[joint_index].setValue(round(value_deg * SLIDER_SCALE))
        self._apply_joint_value(joint_index, value_deg)

    def _apply_joint_value(self, joint_index: int, value_deg: float) -> None:
        if self._selected_index is None:
            return
        point = self._points[self._selected_index]
        joints = list(point.joints_deg)
        joints[joint_index] = value_deg
        new_joints = tuple(joints)
        try:
            tcp_values = self._calculate_fk(new_joints, point.tcp_values)
        except Exception as exc:
            self._sync_selection_to_ui()
            self._status_label.setText(f"FK 失败，J{joint_index + 1} 改动未应用：{exc}")
            return
        updated_point = replace(point, joints_deg=new_joints, tcp_values=tcp_values)
        self._points[self._selected_index] = updated_point
        group = self._groups[point.file_index]
        group.points[point.point_index] = updated_point
        group.dirty = True
        self._refresh_table()
        self._sync_selection_to_ui()
        self._refresh_plot()
        self._refresh_joint_curves()
        self._status_label.setText(
            f"已调整 {group.path.name} 点 {point.point_index + 1} 的 J{joint_index + 1}"
        )

    def _ensure_fk_context(self) -> None:
        if self._robot_model is not None and self._toolset is not None:
            return
        ec: dict[str, object] = {}
        if DEFAULT_LOCAL_IP:
            robot = xCoreSDK_python.xMateErProRobot(
                DEFAULT_LEFT_ARM_IP, DEFAULT_LOCAL_IP
            )
        else:
            robot = xCoreSDK_python.xMateErProRobot(DEFAULT_LEFT_ARM_IP)
        toolset = robot.setToolset(DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME, ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(
                f"setToolset 失败：ec={ec.get('ec', 0)}, message={ec.get('message', '')}"
            )
        self._robot = robot
        self._robot_model = robot.model()
        self._toolset = toolset
        logger.success(
            "FK 已使用 hand-eye 同款 toolset: tool={} wobj={}",
            DEFAULT_TOOL_NAME,
            DEFAULT_WOBJ_NAME,
        )

    def _calculate_fk(
        self, joints_deg: tuple[float, ...], original_tcp: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        self._ensure_fk_context()
        if self._robot_model is None or self._toolset is None:
            raise RuntimeError("FK 上下文未初始化")
        ec: dict[str, object] = {}
        # calcFk 输入必须保持 SDK 原始角度单位 rad，不使用界面展示的 deg。
        joints_rad = [float(np.radians(value)) for value in joints_deg]
        fk_pose = self._robot_model.calcFk(joints_rad, self._toolset, ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(
                f"calcFk 失败：ec={ec.get('ec', 0)}, message={ec.get('message', '')}"
            )
        return _fk_pose_to_tcp(fk_pose, original_tcp)

    @Slot()
    def _save_all(self) -> None:
        dirty_groups = [group for group in self._groups if group.dirty]
        if not dirty_groups:
            self._status_label.setText("没有待保存的改动")
            return
        try:
            for group in dirty_groups:
                snapshot_path = _snapshot_csv(group.path)
                logger.info("保存前快照：{}", snapshot_path)
                for point in group.points:
                    row = dict(group.raw_rows[point.source_row_index])
                    row["joints"] = _serialize_list(point.joints_deg)
                    row["pose"] = _serialize_list(point.tcp_values)
                    group.raw_rows[point.source_row_index] = row
                with group.path.open("w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=group.fieldnames)
                    writer.writeheader()
                    writer.writerows(group.raw_rows)
                group.dirty = False
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self._status_label.setText(f"已保存 {len(dirty_groups)} 个 CSV 文件")
        logger.success("轨迹改动保存成功，文件数 {}", len(dirty_groups))

    def _refresh_plot(self) -> None:
        self._axes.clear()
        self._axes.set_title("TCP 轨迹（按 CSV 文件分组）")
        self._axes.set_xlabel("X (mm)")
        self._axes.set_ylabel("Y (mm)")
        self._axes.set_zlabel("Z (mm)")
        color_map = colormaps["tab20"]
        for group_index, group in enumerate(self._groups):
            if not group.visible or not group.points:
                continue
            xyz = np.asarray(
                [
                    [float(value) for value in point.tcp_values[:3]]
                    for point in group.points
                ],
                dtype=np.float64,
            )
            color = color_map(group_index % 20)
            self._axes.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                "-o",
                color=color,
                linewidth=1.8,
                markersize=3.5,
                label=group.path.stem,
            )
        if self._selected_index is not None and self._selected_index < len(
            self._points
        ):
            point = self._points[self._selected_index]
            if self._groups[point.file_index].visible:
                xyz = [float(value) for value in point.tcp_values[:3]]
                self._axes.scatter(
                    xs=[xyz[0]],
                    ys=[xyz[1]],
                    zs=[xyz[2]],
                    c="#ff006e",
                    s=90,
                    marker="*",
                    depthshade=False,
                    label="当前点",
                )
        if any(group.visible and group.points for group in self._groups):
            self._axes.legend(loc="best", fontsize=8)
        self._canvas.draw_idle()

    def _refresh_joint_curves(self) -> None:
        self._joint_axes.clear()
        self._joint_axes.set_title("关节角曲线")
        self._joint_axes.set_xlabel("轨迹点编号")
        self._joint_axes.set_ylabel("关节角 (deg)")
        self._joint_axes.grid(True, alpha=0.25)
        joint_colors = colormaps["tab10"]
        point_offset = 0
        has_visible_points = False
        for group in self._groups:
            group_size = len(group.points)
            if not group.visible or not group.points:
                point_offset += group_size
                continue
            show_labels = not has_visible_points
            has_visible_points = True
            joints = np.asarray(
                [point.joints_deg for point in group.points], dtype=np.float64
            )
            point_numbers = np.arange(
                point_offset + 1, point_offset + group_size + 1, dtype=np.int64
            )
            for joint_index in range(JOINT_COUNT):
                self._joint_axes.plot(
                    point_numbers,
                    joints[:, joint_index],
                    color=joint_colors(joint_index),
                    linewidth=1.5,
                    label=f"J{joint_index + 1}" if show_labels else "_nolegend_",
                )
            point_offset += group_size
        if has_visible_points:
            self._joint_axes.legend(loc="best", ncols=2)
        if self._selected_index is not None:
            self._joint_axes.axvline(
                self._selected_index + 1,
                color="#ff006e",
                linestyle="--",
                linewidth=1.2,
                label="当前点",
            )
        self._joint_figure.tight_layout()
        self._joint_canvas.draw_idle()


# endregion


# region 主入口


def main(record_dir: Path = DEFAULT_RECORD_DIR) -> int:
    """启动左臂轨迹优化 GUI，支持 IDE 直接运行。"""

    app = QApplication.instance() or QApplication(sys.argv)
    window = TrajectoryOptimizerWindow(record_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


# endregion
