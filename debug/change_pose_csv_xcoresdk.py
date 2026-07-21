#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"
DEFAULT_RECORD_DIR = PROJECT_ROOT / "record_replay" / "records" / "left"
DEFAULT_TOOL_NAME = "g_tool_0"
DEFAULT_WOBJ_NAME = "g_wobj_0"
MM_PER_M = 1000.0
DEFAULT_LOCAL_IP = os.environ.get("DINGTAI_XCORESDK_LOCAL_IP", "192.168.1.116").strip()
DEFAULT_ARM_ROBOT_IPS = {
    "left": os.environ.get("DINGTAI_XCORESDK_LEFT_IP", "192.168.1.161").strip(),
    "right": os.environ.get("DINGTAI_XCORESDK_RIGHT_IP", "192.168.1.160").strip(),
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from sdk.xcoresdk import xCoreSDK_python

# region 数据结构


@dataclass(frozen=True, slots=True)
class PoseCsvRow:
    """CSV 中单行记录。"""

    csv_row_index: int
    "CSV 数据行索引，不含表头。"

    timestamp: str
    "时间戳。"

    record_type: str
    "记录类型。"

    joints_text: str
    "原始 joints 文本。"

    pose_text: str
    "原始 pose 文本。"

    joints_deg: tuple[float, ...] | None
    "解析后的关节角，单位 deg。"

    pose_values: tuple[Any, ...] | None
    "解析后的 pose 值。"

    raw_row: dict[str, str]
    "原始字段字典。"


@dataclass(frozen=True, slots=True)
class RecomputedPoseRow:
    """基于 joints 重算后的 pose 结果。"""

    pose_values: tuple[Any, ...] | None
    "重算后的完整 pose。"

    status: str
    "重算结果说明。"


# endregion


# region 基础方法


def _parse_list_field(raw_text: str) -> list[Any]:
    """解析 CSV 列表字段。"""

    parsed = ast.literal_eval(raw_text)
    if not isinstance(parsed, list):
        raise ValueError(f"字段不是列表：{raw_text!r}")
    return parsed


def _infer_arm_side_from_csv_path(csv_path: Path) -> str:
    """根据 CSV 路径推断左右臂。"""

    lowered = str(csv_path).replace("\\", "/").lower()
    if "records/left" in lowered or "_left_" in lowered:
        return "left"
    if "records/right" in lowered or "_right_" in lowered:
        return "right"
    raise ValueError(f"无法从路径判断左右臂：{csv_path}")


def _read_pose_csv(csv_path: Path) -> tuple[list[PoseCsvRow], list[str]]:
    """读取 CSV 全部行，保留非 arm 行以便原样回写。"""

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: list[PoseCsvRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV 缺少表头")
        fieldnames = list(reader.fieldnames)
        for csv_row_index, row in enumerate(reader):
            raw_row = {key: str(value) if value is not None else "" for key, value in row.items()}
            joints_text = raw_row.get("joints", "").strip()
            pose_text = raw_row.get("pose", "").strip()
            joints_deg: tuple[float, ...] | None = None
            pose_values: tuple[Any, ...] | None = None
            if joints_text.startswith("["):
                joints_deg = tuple(float(value) for value in _parse_list_field(joints_text))
            if pose_text.startswith("["):
                pose_values = tuple(_parse_list_field(pose_text))
            rows.append(
                PoseCsvRow(
                    csv_row_index=csv_row_index,
                    timestamp=raw_row.get("timestamp", ""),
                    record_type=raw_row.get("type", ""),
                    joints_text=joints_text,
                    pose_text=pose_text,
                    joints_deg=joints_deg,
                    pose_values=pose_values,
                    raw_row=raw_row,
                )
            )
    return rows, fieldnames


def _format_pose6(pose_values: tuple[Any, ...] | None) -> str:
    """格式化 pose 前 6 个值用于界面显示。"""

    if pose_values is None:
        return "NaN"
    if len(pose_values) < 6:
        return str(list(pose_values))
    return "[" + ", ".join(f"{float(value):.2f}" for value in pose_values[:6]) + "]"


def _format_joints(joints_deg: tuple[float, ...] | None) -> str:
    """格式化关节角显示文本。"""

    if joints_deg is None:
        return "NaN"
    return "[" + ", ".join(f"{float(value):.2f}" for value in joints_deg) + "]"


def _format_pose_for_csv(pose_values: tuple[Any, ...]) -> str:
    """按 CSV 约定格式化完整 pose。"""

    parts: list[str] = []
    for value in pose_values:
        if isinstance(value, bool):
            parts.append("True" if value else "False")
        elif isinstance(value, (int, float, np.integer, np.floating)):
            parts.append(f"{float(value):.2f}")
        elif isinstance(value, (list, tuple)):
            nested = ", ".join(str(int(item)) if isinstance(item, (int, np.integer)) else str(item) for item in value)
            parts.append(f"[{nested}]")
        else:
            parts.append(str(value))
    return "[" + ", ".join(parts) + "]"


def _error_code(ec: dict[str, object]) -> int:
    """把 SDK ec 字典中的错误码安全转成 int。"""

    raw_value = ec.get("ec", 0)
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    if isinstance(raw_value, str):
        return int(raw_value)
    return 0


def _joint_values_to_pose_values(
    fk_pose: xCoreSDK_python.CartesianPosition,
    original_pose: tuple[Any, ...] | None,
) -> tuple[Any, ...]:
    """把 FK 结果转换回 CSV pose，并尽量保留附加字段。"""

    pose_prefix = (
        float(fk_pose.trans[0]) * MM_PER_M,
        float(fk_pose.trans[1]) * MM_PER_M,
        float(fk_pose.trans[2]) * MM_PER_M,
        float(np.rad2deg(float(fk_pose.rpy[0]))),
        float(np.rad2deg(float(fk_pose.rpy[1]))),
        float(np.rad2deg(float(fk_pose.rpy[2]))),
    )
    if original_pose is None:
        return pose_prefix

    suffix: list[Any] = []
    if len(original_pose) >= 7:
        suffix.append(bool(getattr(fk_pose, "hasElbow", original_pose[6])))
    if len(original_pose) >= 8:
        elbow_deg = float(np.rad2deg(float(getattr(fk_pose, "elbow", 0.0))))
        suffix.append(elbow_deg)
    if len(original_pose) >= 9:
        original_conf = original_pose[8]
        fk_conf = list(getattr(fk_pose, "confData", []))
        if fk_conf:
            suffix.append([int(value) for value in fk_conf])
        else:
            suffix.append(list(original_conf) if isinstance(original_conf, (list, tuple)) else original_conf)
    if len(original_pose) > 9:
        suffix.extend(original_pose[9:])
    return pose_prefix + tuple(suffix)


# endregion


# region SDK


def _create_robot_and_toolset(csv_path: Path) -> tuple[Any, Any]:
    """连接机器人并获取指定工具工件组。"""

    arm_side = _infer_arm_side_from_csv_path(csv_path)
    robot_ip = DEFAULT_ARM_ROBOT_IPS.get(arm_side, "").strip()
    if not robot_ip:
        raise RuntimeError(f"未配置 {arm_side} arm IP")

    if DEFAULT_LOCAL_IP:
        robot = xCoreSDK_python.xMateErProRobot(robot_ip, DEFAULT_LOCAL_IP)
    else:
        robot = xCoreSDK_python.xMateErProRobot(robot_ip)

    ec: dict[str, object] = {}
    toolset = robot.setToolset(DEFAULT_TOOL_NAME, DEFAULT_WOBJ_NAME, ec)
    if _error_code(ec) != 0:
        raise RuntimeError(
            f"setToolset 失败：tool={DEFAULT_TOOL_NAME} wobj={DEFAULT_WOBJ_NAME} "
            f"ec={ec.get('ec', 0)} message={ec.get('message', '')}"
        )
    return robot.model(), toolset


def _recompute_pose_rows(rows: list[PoseCsvRow], robot_model: Any, toolset: Any) -> list[RecomputedPoseRow]:
    """基于 joints 对全部 arm 行做 FK 重算。"""

    results: list[RecomputedPoseRow] = []
    for row in rows:
        if row.record_type.strip().lower() != "arm":
            results.append(RecomputedPoseRow(None, "非 arm 行"))
            continue
        if row.joints_deg is None:
            results.append(RecomputedPoseRow(None, "joints 无法解析"))
            continue
        if row.pose_text.strip().upper() == "NAN":
            results.append(RecomputedPoseRow(None, "原始 pose 为 NaN"))
            continue
        ec: dict[str, object] = {}
        try:
            joints_rad = [float(np.deg2rad(value)) for value in row.joints_deg]
            fk_pose = robot_model.calcFk(joints_rad, toolset, ec)
            if _error_code(ec) != 0:
                raise RuntimeError(f"calcFk 失败：ec={ec.get('ec', 0)} message={ec.get('message', '')}")
            pose_values = _joint_values_to_pose_values(fk_pose, row.pose_values)
            results.append(RecomputedPoseRow(pose_values=pose_values, status="ok"))
        except Exception as exc:
            results.append(RecomputedPoseRow(pose_values=None, status=str(exc)))
    return results


# endregion


# region Qt 界面


class PoseCsvViewer(QMainWindow):
    """CSV 原始 pose 与重算 pose 核对窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self._record_dir = DEFAULT_RECORD_DIR
        self._csv_files: list[Path] = []
        self._csv_path: Path | None = None
        self._csv_fieldnames: list[str] = []
        self._rows: list[PoseCsvRow] = []
        self._recomputed_rows: list[RecomputedPoseRow] = []

        self._file_combo = QComboBox(self)
        self._refresh_files_button = QPushButton("刷新文件列表", self)
        self._save_button = QPushButton("保存覆盖原文件", self)
        self._status_label = QLabel("未加载", self)
        self._original_table = QTableWidget(self)
        self._recomputed_table = QTableWidget(self)

        self._setup_ui()
        self._connect_signals()
        self._refresh_file_list()

    def _setup_ui(self) -> None:
        """初始化界面。"""

        self.setWindowTitle("Pose CSV 核对与重算")
        self.resize(1500, 900)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("CSV 文件：", self))
        top_layout.addWidget(self._file_combo, stretch=1)
        top_layout.addWidget(self._refresh_files_button)
        root_layout.addLayout(top_layout)

        root_layout.addWidget(QLabel("原始值", self))
        self._setup_table(self._original_table)
        root_layout.addWidget(self._original_table, stretch=1)

        root_layout.addWidget(QLabel("重新计算后的值", self))
        self._setup_table(self._recomputed_table)
        root_layout.addWidget(self._recomputed_table, stretch=1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self._status_label, stretch=1)
        bottom_layout.addWidget(self._save_button)
        root_layout.addLayout(bottom_layout)

    def _setup_table(self, table: QTableWidget) -> None:
        """统一表格样式。"""

        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["序号", "timestamp", "type", "joints", "pose"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

    def _connect_signals(self) -> None:
        """连接信号。"""

        self._refresh_files_button.clicked.connect(self._refresh_file_list)
        self._file_combo.currentIndexChanged.connect(self._on_file_changed)
        self._save_button.clicked.connect(self._save_current_file)
        self._original_table.itemSelectionChanged.connect(self._sync_selection_from_original)
        self._recomputed_table.itemSelectionChanged.connect(self._sync_selection_from_recomputed)

    def _refresh_file_list(self) -> None:
        """读取 record_replay/records/left 下全部 CSV 到下拉框。"""

        self._csv_files = sorted(self._record_dir.glob("*.csv"))
        with QSignalBlocker(self._file_combo):
            self._file_combo.clear()
            for csv_path in self._csv_files:
                self._file_combo.addItem(csv_path.name, csv_path)
        if not self._csv_files:
            self._csv_path = None
            self._rows = []
            self._recomputed_rows = []
            self._refresh_tables()
            self._status_label.setText(f"目录下没有 CSV: {self._record_dir}")
            return
        self._file_combo.setCurrentIndex(0)
        self._load_selected_file()

    @Slot()
    def _on_file_changed(self) -> None:
        self._load_selected_file()

    def _load_selected_file(self) -> None:
        """加载当前下拉框选中的 CSV。"""

        current_path = self._file_combo.currentData()
        if not isinstance(current_path, Path):
            self._csv_path = None
            return
        self._csv_path = current_path
        try:
            self._rows, self._csv_fieldnames = _read_pose_csv(current_path)
            self._recompute_current_file(show_error_dialog=False)
        except Exception as exc:
            self._rows = []
            self._recomputed_rows = []
            self._refresh_tables()
            self._status_label.setText(f"加载失败：{exc}")
            QMessageBox.warning(self, "加载失败", str(exc))

    def _recompute_current_file(self, show_error_dialog: bool = True) -> None:
        """按当前工具工件重新计算全部 pose。"""

        if self._csv_path is None:
            if show_error_dialog:
                QMessageBox.warning(self, "无法计算", "当前没有选中的 CSV")
            return
        try:
            robot_model, toolset = _create_robot_and_toolset(self._csv_path)
            self._recomputed_rows = _recompute_pose_rows(self._rows, robot_model, toolset)
            self._refresh_tables()
            ok_count = sum(1 for item in self._recomputed_rows if item.status == "ok")
            self._status_label.setText(f"已使用 {DEFAULT_TOOL_NAME} / {DEFAULT_WOBJ_NAME} 重算 {ok_count} 行")
        except Exception as exc:
            self._recomputed_rows = []
            self._refresh_tables()
            self._status_label.setText(f"重算失败：{exc}")
            if show_error_dialog:
                QMessageBox.warning(self, "重算失败", str(exc))

    def _refresh_tables(self) -> None:
        """刷新原始表和重算表。"""

        self._fill_original_table()
        self._fill_recomputed_table()

    def _fill_original_table(self) -> None:
        """填充原始表。"""

        self._original_table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            self._set_table_row(
                self._original_table,
                row_index,
                [
                    str(row.csv_row_index + 1),
                    row.timestamp,
                    row.record_type,
                    _format_joints(row.joints_deg) if row.record_type.strip().lower() == "arm" else row.joints_text,
                    _format_pose6(row.pose_values) if row.record_type.strip().lower() == "arm" else row.pose_text,
                ],
            )
        self._original_table.resizeColumnsToContents()

    def _fill_recomputed_table(self) -> None:
        """填充重算表。"""

        self._recomputed_table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            recomputed = self._recomputed_rows[row_index] if row_index < len(self._recomputed_rows) else None
            if recomputed is None:
                pose_text = "尚未计算"
            elif row.pose_text.strip().upper() == "NAN":
                pose_text = "NaN"
            elif recomputed.pose_values is None:
                pose_text = recomputed.status
            else:
                pose_text = _format_pose6(recomputed.pose_values)
            self._set_table_row(
                self._recomputed_table,
                row_index,
                [
                    str(row.csv_row_index + 1),
                    row.timestamp,
                    row.record_type,
                    _format_joints(row.joints_deg) if row.record_type.strip().lower() == "arm" else row.joints_text,
                    pose_text,
                ],
            )
        self._recomputed_table.resizeColumnsToContents()

    def _set_table_row(self, table: QTableWidget, row_index: int, values: list[str]) -> None:
        """写入一整行单元格。"""

        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, column_index, item)

    @Slot()
    def _sync_selection_from_original(self) -> None:
        self._sync_table_selection(self._original_table, self._recomputed_table)

    @Slot()
    def _sync_selection_from_recomputed(self) -> None:
        self._sync_table_selection(self._recomputed_table, self._original_table)

    def _sync_table_selection(self, source_table: QTableWidget, target_table: QTableWidget) -> None:
        """同步上下表选中行。"""

        if source_table.selectionModel() is None:
            return
        selected_rows = source_table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row_index = int(selected_rows[0].row())
        with QSignalBlocker(target_table):
            target_table.clearSelection()
            target_table.selectRow(row_index)

    @Slot()
    def _save_current_file(self) -> None:
        """把重算后的 pose 覆盖回原 CSV，仅修改 pose。"""

        if self._csv_path is None:
            QMessageBox.warning(self, "无法保存", "当前没有选中的 CSV")
            return
        if not self._recomputed_rows:
            QMessageBox.warning(self, "无法保存", "请先点击“重新计算”")
            return
        if not self._csv_fieldnames:
            QMessageBox.warning(self, "无法保存", "CSV 表头为空")
            return

        updated_rows: list[dict[str, str]] = []
        updated_count = 0
        for row_index, row in enumerate(self._rows):
            row_dict = dict(row.raw_row)
            recomputed = self._recomputed_rows[row_index]
            if row.pose_text.strip().upper() == "NAN":
                row_dict["pose"] = row.pose_text
            elif row.record_type.strip().lower() == "arm" and recomputed.pose_values is not None:
                row_dict["pose"] = _format_pose_for_csv(recomputed.pose_values)
                updated_count += 1
            updated_rows.append(row_dict)

        try:
            with self._csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=self._csv_fieldnames)
                writer.writeheader()
                writer.writerows(updated_rows)
            self._rows, self._csv_fieldnames = _read_pose_csv(self._csv_path)
            self._refresh_tables()
            self._status_label.setText(f"已覆盖保存 {self._csv_path.name}，更新 pose {updated_count} 行，joints 未改动")
        except Exception as exc:
            self._status_label.setText(f"保存失败：{exc}")
            QMessageBox.warning(self, "保存失败", str(exc))


# endregion


# region 主入口


def main() -> int:
    """启动 CSV pose 核对窗口。"""

    app = QApplication.instance() or QApplication(sys.argv)
    window = PoseCsvViewer()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
