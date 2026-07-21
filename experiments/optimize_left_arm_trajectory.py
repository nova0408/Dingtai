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
from matplotlib import colormaps, rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation
from PySide6.QtCore import QSignalBlocker, Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# pyright: reportMissingImports=false


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"
DEFAULT_RECORD_DIR = PROJECT_ROOT / "record_replay" / "records" / "left"  # 左臂轨迹 CSV 所在目录。
DEFAULT_LEFT_ARM_IP = "192.168.1.161"
DEFAULT_LOCAL_IP = "192.168.1.116"
DEFAULT_TOOL_NAME = "g_tool_0"  # 与手眼标定脚本一致的全局工具名称。
DEFAULT_WOBJ_NAME = "g_wobj_0"  # 与手眼标定脚本一致的全局工件名称。
JOINT_COUNT = 7
VISIBILITY_KIND_ROLE = int(Qt.ItemDataRole.UserRole)
VISIBILITY_INDEX_ROLE = VISIBILITY_KIND_ROLE + 1
TCP_AXIS_LENGTH_MM = 35.0

# Matplotlib 不会自动继承 Qt 的中文字体，显式给出 Windows / Linux 字体回退顺序。
rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False

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
    visible: bool = True
    dirty: bool = False

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


def _elbow_deg(tcp_values: tuple[Any, ...]) -> float:
    """读取 CSV pose 中的臂角展示值，旧的六项 pose 默认显示为 0 deg。"""

    return float(tcp_values[7]) if len(tcp_values) >= 8 else 0.0


def _updated_tcp_values(
    original_tcp: tuple[Any, ...],
    tcp_prefix: tuple[float, ...],
    elbow_deg: float,
) -> tuple[Any, ...]:
    """更新 TCP 和 elbow，同时保留 CSV 中已有的构型及扩展字段。"""

    suffix = original_tcp[6:]
    if len(suffix) >= 2:
        return tcp_prefix + (True, elbow_deg, *suffix[2:])
    return tcp_prefix + (True, elbow_deg)


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
    """左臂多文件轨迹查看与 TCP / elbow 逆解编辑窗口。"""

    def __init__(self, record_dir: Path) -> None:
        super().__init__()
        self._record_dir = record_dir
        self._groups: list[TrajectoryGroup] = []
        self._points: list[TrajectoryPoint] = []
        self._table_point_indices: list[int] = []
        self._metrics: list[ContinuityMetric] = []
        self._selected_index: int | None = None
        self._robot: Any | None = None
        self._robot_model: Any | None = None
        self._toolset: Any | None = None

        self._figure = Figure(figsize=(8.0, 7.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        # Matplotlib 当前类型声明不会把 projection="3d" 收窄为 Axes3D。
        self._axes: Any = self._figure.add_subplot(111, projection="3d")
        self._visibility_tree = QTreeWidget(self)
        self._tabs = QTabWidget(self)
        self._joint_figure = Figure(figsize=(7.0, 5.0))
        self._joint_canvas = FigureCanvasQTAgg(self._joint_figure)
        self._joint_axes = self._joint_figure.add_subplot(111)
        self._table = QTableWidget(self)
        self._detail_label = QLabel("请选择轨迹点", self)
        self._tcp_spins: list[QDoubleSpinBox] = []
        self._continuity_label = QLabel("连续性：-", self)
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
        self._visibility_tree.setHeaderLabel("轨迹组 / 点（勾选控制图中可见性）")
        self._visibility_tree.setAlternatingRowColors(True)
        self._visibility_tree.setMinimumHeight(180)
        left_layout.addWidget(self._visibility_tree)
        left_layout.addWidget(self._canvas, 1)
        left_layout.addWidget(self._status_label)
        splitter.addWidget(left_panel)

        self._setup_list_tab()
        self._setup_joint_curve_tab()
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)

    def _setup_list_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            [
                "轨迹组",
                "点",
                "关节角 (deg)",
                "TCP (mm/deg)",
                "Elbow (deg)",
                "关节连续性 (deg)",
            ]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)

        tcp_editor = QWidget(self)
        tcp_layout = QGridLayout(tcp_editor)
        tcp_layout.addWidget(QLabel("当前点 TCP（修改后立即逆解）", self), 0, 0, 1, 6)
        tcp_names = ("X (mm)", "Y (mm)", "Z (mm)", "Rx (deg)", "Ry (deg)", "Rz (deg)")
        for component_index, name in enumerate(tcp_names):
            spin = QDoubleSpinBox(self)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            if component_index < 3:
                spin.setRange(-5000.0, 5000.0)
            else:
                spin.setRange(-360.0, 360.0)
            tcp_layout.addWidget(QLabel(name, self), 1, component_index)
            tcp_layout.addWidget(spin, 2, component_index)
            self._tcp_spins.append(spin)
        layout.addWidget(tcp_editor)

        self._continuity_label.setWordWrap(True)
        layout.addWidget(self._continuity_label)
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._detail_label.setMinimumHeight(150)
        layout.addWidget(self._detail_label)
        self._tabs.addTab(page, "轨迹列表")

    def _setup_joint_curve_tab(self) -> None:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(self._joint_canvas, 1)
        self._tabs.addTab(page, "关节曲线")

    def _connect_signals(self) -> None:
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._visibility_tree.itemChanged.connect(self._on_visibility_item_changed)
        self._visibility_tree.itemClicked.connect(self._on_visibility_item_clicked)
        self._reload_button.clicked.connect(self._load_data)
        self._save_button.clicked.connect(self._save_all)
        for component_index, spin in enumerate(self._tcp_spins):
            spin.valueChanged.connect(
                lambda value, index=component_index: self._on_tcp_value_changed(
                    index, value
                )
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
        self._rebuild_visibility_tree()
        self._refresh_table()
        self._sync_selection_to_ui()
        self._refresh_plot()
        self._refresh_joint_curves()
        self._status_label.setText(
            f"已按编号顺序加载 {len(groups)} 组、{len(self._points)} 个轨迹点"
        )

    def _rebuild_visibility_tree(self) -> None:
        with QSignalBlocker(self._visibility_tree):
            self._visibility_tree.clear()
            point_offset = 0
            for group_index, group in enumerate(self._groups):
                group_item = QTreeWidgetItem(
                    self._visibility_tree,
                    [f"{group.path.name} ({len(group.points)})"],
                )
                group_item.setData(0, VISIBILITY_KIND_ROLE, "group")
                group_item.setData(0, VISIBILITY_INDEX_ROLE, group_index)
                group_item.setFlags(group_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                group_item.setCheckState(
                    0,
                    Qt.CheckState.Checked
                    if group.visible
                    else Qt.CheckState.Unchecked,
                )
                for local_index, point in enumerate(group.points):
                    point_item = QTreeWidgetItem(
                        group_item,
                        [f"点 {local_index + 1}  {point.timestamp}"],
                    )
                    point_item.setData(0, VISIBILITY_KIND_ROLE, "point")
                    point_item.setData(
                        0, VISIBILITY_INDEX_ROLE, point_offset + local_index
                    )
                    point_item.setFlags(
                        point_item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    point_item.setCheckState(
                        0,
                        Qt.CheckState.Checked
                        if point.visible
                        else Qt.CheckState.Unchecked,
                    )
                point_offset += len(group.points)

    def _refresh_table(self) -> None:
        self._metrics = _continuity_metrics(self._points)
        self._table_point_indices = [
            point_index
            for point_index, point in enumerate(self._points)
            if self._groups[point.file_index].visible and point.visible
        ]
        with QSignalBlocker(self._table):
            self._table.setRowCount(len(self._table_point_indices))
            for row_index, point_index in enumerate(self._table_point_indices):
                point = self._points[point_index]
                group = self._groups[point.file_index]
                values = (
                    group.path.name,
                    str(point.point_index + 1),
                    _format_values(point.joints_deg),
                    _format_values(point.tcp_values[:6]),
                    f"{_elbow_deg(point.tcp_values):.3f}",
                    _format_continuity(self._metrics[row_index]),
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column != 4:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self._table.setItem(row_index, column, item)

    def _sync_selection_to_ui(self) -> None:
        if self._selected_index is None or not self._points:
            with QSignalBlocker(self._table):
                self._table.clearSelection()
            for spin in self._tcp_spins:
                spin.setEnabled(False)
            self._continuity_label.setText("关节连续性：-")
            self._detail_label.setText("当前没有可见轨迹点")
            return
        point = self._points[self._selected_index]
        metric = self._metrics[self._selected_index]
        with QSignalBlocker(self._table):
            if self._selected_index in self._table_point_indices:
                self._table.selectRow(
                    self._table_point_indices.index(self._selected_index)
                )
            else:
                self._table.clearSelection()
        for spin, value in zip(self._tcp_spins, point.tcp_values[:6], strict=True):
            with QSignalBlocker(spin):
                spin.setEnabled(True)
                spin.setValue(float(value))
        group = self._groups[point.file_index]
        detail_lines = [
            f"文件：{group.path}",
            f"文件内点序号：{point.point_index + 1}",
            f"CSV 原始行号：{point.source_row_index + 2}",
            f"时间戳：{point.timestamp}",
            f"关节角 (deg): {_format_values(point.joints_deg, 6)}",
            f"TCP xyz(mm) + rpy xyz(deg): {_format_values(point.tcp_values[:6], 6)}",
            f"Elbow (deg): {_elbow_deg(point.tcp_values):.6f}",
            f"连续性：{_format_continuity(metric)}",
        ]
        self._detail_label.setText("\n".join(detail_lines))
        self._continuity_label.setText(f"关节连续性：{_format_continuity(metric)}")

    @Slot()
    def _on_table_selection_changed(self) -> None:
        indexes = self._table.selectionModel().selectedRows()
        if indexes:
            table_row = int(indexes[0].row())
            self._select_point(self._table_point_indices[table_row])

    def _select_point(self, index: int) -> None:
        if index < 0 or index >= len(self._points):
            return
        self._selected_index = index
        self._sync_selection_to_ui()
        self._refresh_plot()
        self._refresh_joint_curves()

    def _on_visibility_item_changed(
        self, item: QTreeWidgetItem, _column: int
    ) -> None:
        checked = item.checkState(0) == Qt.CheckState.Checked
        item_kind = item.data(0, VISIBILITY_KIND_ROLE)
        item_index = int(item.data(0, VISIBILITY_INDEX_ROLE))
        if item_kind == "group":
            self._groups[item_index].visible = checked
        elif item_kind == "point":
            self._points[item_index].visible = checked
        self._refresh_table()
        if self._selected_index not in self._table_point_indices:
            self._selected_index = (
                self._table_point_indices[0] if self._table_point_indices else None
            )
        self._sync_selection_to_ui()
        self._refresh_plot()
        self._refresh_joint_curves()

    def _on_visibility_item_clicked(
        self, item: QTreeWidgetItem, _column: int
    ) -> None:
        if item.data(0, VISIBILITY_KIND_ROLE) != "point":
            return
        point_index = int(item.data(0, VISIBILITY_INDEX_ROLE))
        point = self._points[point_index]
        if point.visible and self._groups[point.file_index].visible:
            self._select_point(point_index)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 4:
            return
        try:
            elbow_deg = float(item.text())
        except ValueError:
            self._status_label.setText("Elbow 必须是有效数字")
            self._refresh_table()
            self._sync_selection_to_ui()
            return
        point_index = self._table_point_indices[item.row()]
        self._select_point(point_index)
        point = self._points[point_index]
        self._apply_ik_edit(
            tuple(float(value) for value in point.tcp_values[:6]),
            elbow_deg,
            "Elbow",
        )

    @Slot(int, float)
    def _on_tcp_value_changed(self, component_index: int, value: float) -> None:
        if self._selected_index is None:
            return
        point = self._points[self._selected_index]
        tcp_prefix = [float(component) for component in point.tcp_values[:6]]
        tcp_prefix[component_index] = value
        self._apply_ik_edit(
            tuple(tcp_prefix),
            _elbow_deg(point.tcp_values),
            f"TCP {component_index + 1}",
        )

    def _apply_ik_edit(
        self,
        tcp_prefix: tuple[float, ...],
        elbow_deg: float,
        edited_field: str,
    ) -> None:
        if self._selected_index is None:
            return
        point = self._points[self._selected_index]
        tcp_values = _updated_tcp_values(point.tcp_values, tcp_prefix, elbow_deg)
        try:
            joints_deg = self._calculate_ik(tcp_values, elbow_deg)
        except Exception as exc:
            self._sync_selection_to_ui()
            self._status_label.setText(f"IK 失败，{edited_field} 改动未应用：{exc}")
            return
        updated_point = replace(
            point,
            joints_deg=joints_deg,
            tcp_values=tcp_values,
            dirty=True,
        )
        self._points[self._selected_index] = updated_point
        group = self._groups[point.file_index]
        group.points[point.point_index] = updated_point
        group.dirty = True
        self._refresh_table()
        self._sync_selection_to_ui()
        self._refresh_plot()
        self._refresh_joint_curves()
        self._status_label.setText(
            f"已更新 {group.path.name} 点 {point.point_index + 1} 的 {edited_field} 并重算 joints"
        )

    def _ensure_ik_context(self) -> None:
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
            "IK 已使用 hand-eye 同款 toolset: tool={} wobj={}",
            DEFAULT_TOOL_NAME,
            DEFAULT_WOBJ_NAME,
        )

    def _calculate_ik(
        self, tcp_values: tuple[Any, ...], elbow_deg: float
    ) -> tuple[float, ...]:
        self._ensure_ik_context()
        if self._robot_model is None or self._toolset is None:
            raise RuntimeError("IK 上下文未初始化")
        ec: dict[str, object] = {}
        # v0.7.1 示例要求 CartesianPosition 使用 m/rad，并在 calcIk 前写入 elbow(rad)。
        target_pose = xCoreSDK_python.CartesianPosition(
            [
                *(float(value) / 1000.0 for value in tcp_values[:3]),
                *(float(np.radians(float(value))) for value in tcp_values[3:6]),
            ]
        )
        target_pose.elbow = float(np.radians(elbow_deg))
        if len(tcp_values) >= 9 and isinstance(tcp_values[8], list):
            conf_data = [int(value) for value in tcp_values[8]]
            if conf_data:
                target_pose.confData = conf_data
        joints_rad = self._robot_model.calcIk(target_pose, self._toolset, ec)
        if ec.get("ec", 0) != 0:
            raise RuntimeError(
                f"calcIk 失败：ec={ec.get('ec', 0)}, message={ec.get('message', '')}"
            )
        if len(joints_rad) != JOINT_COUNT:
            raise RuntimeError(f"calcIk 返回关节数异常：{len(joints_rad)}")
        return tuple(float(np.degrees(float(value))) for value in joints_rad)

    @Slot()
    def _save_all(self) -> None:
        writable_groups = [
            (
                group,
                [point for point in group.points if point.visible and point.dirty],
            )
            for group in self._groups
            if group.visible
        ]
        writable_groups = [
            (group, points) for group, points in writable_groups if points
        ]
        hidden_dirty_count = sum(
            1
            for group in self._groups
            for point in group.points
            if point.dirty and (not group.visible or not point.visible)
        )
        if not writable_groups:
            if hidden_dirty_count:
                self._status_label.setText(
                    f"没有可写回的可见改动；{hidden_dirty_count} 个隐藏点保持原 CSV 不变"
                )
            else:
                self._status_label.setText("没有待保存的改动")
            return
        try:
            for group, dirty_points in writable_groups:
                snapshot_path = _snapshot_csv(group.path)
                logger.info("保存前快照：{}", snapshot_path)
                for point in dirty_points:
                    row = dict(group.raw_rows[point.source_row_index])
                    row["joints"] = _serialize_list(point.joints_deg)
                    row["pose"] = _serialize_list(point.tcp_values)
                    group.raw_rows[point.source_row_index] = row
                with group.path.open("w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=group.fieldnames)
                    writer.writeheader()
                    writer.writerows(group.raw_rows)
                for point in dirty_points:
                    point.dirty = False
                group.dirty = any(point.dirty for point in group.points)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        saved_count = len(writable_groups)
        status = f"已保存 {saved_count} 个 CSV 文件"
        if hidden_dirty_count:
            status += f"；{hidden_dirty_count} 个隐藏点保持原 CSV 不变"
        self._status_label.setText(status)
        logger.success("轨迹改动保存成功，文件数 {}", saved_count)

    def _refresh_plot(self) -> None:
        self._axes.clear()
        self._axes.set_title("TCP 轨迹（按 CSV 文件分组，x 红 / z 蓝）")
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
            visible_mask = np.asarray(
                [point.visible for point in group.points], dtype=np.bool_
            )
            if not np.any(visible_mask):
                continue
            visible_xyz = xyz[visible_mask]
            plot_xyz = xyz.copy()
            plot_xyz[~visible_mask] = np.nan
            color = color_map(group_index % 20)
            self._axes.plot(
                plot_xyz[:, 0],
                plot_xyz[:, 1],
                plot_xyz[:, 2],
                "-o",
                color=color,
                linewidth=1.8,
                markersize=3.5,
                label=group.path.stem,
            )
            visible_rpy_deg = np.asarray(
                [
                    [float(value) for value in point.tcp_values[3:6]]
                    for point in group.points
                    if point.visible
                ],
                dtype=np.float64,
            )
            rotations = Rotation.from_euler(
                "xyz", visible_rpy_deg, degrees=True
            )
            x_directions = rotations.apply(
                np.tile(np.asarray([1.0, 0.0, 0.0]), (len(visible_xyz), 1))
            )
            z_directions = rotations.apply(
                np.tile(np.asarray([0.0, 0.0, 1.0]), (len(visible_xyz), 1))
            )
            self._axes.quiver(
                visible_xyz[:, 0],
                visible_xyz[:, 1],
                visible_xyz[:, 2],
                x_directions[:, 0],
                x_directions[:, 1],
                x_directions[:, 2],
                length=TCP_AXIS_LENGTH_MM,
                normalize=True,
                color="#e63946",
                linewidth=1.0,
            )
            self._axes.quiver(
                visible_xyz[:, 0],
                visible_xyz[:, 1],
                visible_xyz[:, 2],
                z_directions[:, 0],
                z_directions[:, 1],
                z_directions[:, 2],
                length=TCP_AXIS_LENGTH_MM,
                normalize=True,
                color="#277da1",
                linewidth=1.0,
            )
        if self._selected_index is not None and self._selected_index < len(
            self._points
        ):
            point = self._points[self._selected_index]
            if self._groups[point.file_index].visible and point.visible:
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
        if any(
            group.visible and any(point.visible for point in group.points)
            for group in self._groups
        ):
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
            visible_mask = np.asarray(
                [point.visible for point in group.points], dtype=np.bool_
            )
            if not np.any(visible_mask):
                point_offset += group_size
                continue
            show_labels = not has_visible_points
            has_visible_points = True
            joints = np.asarray(
                [point.joints_deg for point in group.points], dtype=np.float64
            )
            joints[~visible_mask] = np.nan
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
        if (
            self._selected_index is not None
            and self._points[self._selected_index].visible
            and self._groups[self._points[self._selected_index].file_index].visible
        ):
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
