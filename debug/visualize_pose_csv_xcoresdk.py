#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from loguru import logger
from PySide6.QtCore import QFileSystemWatcher, QSignalBlocker, QTimer, Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "sdk"
DEFAULT_CSV_PATH = PROJECT_ROOT / "record_left" / "close_door_left_20260629_143547.csv"
CSV_REFRESH_INTERVAL_MS = 500
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
    """CSV 中的一行位姿记录。

    该结构保留 CSV 原始行文本与解析结果，便于动态编辑后重写整行。
    它不持有 Qt 对象和 SDK 对象，只负责当前行的数据契约。

    Attributes
    ----------
    timestamp:
        记录时间戳，字符串原样保留。
    record_type:
        记录类型，通常为 `arm`。
    joints_deg:
        关节角数组，单位 deg。
    pose_values:
        原始 pose 解析后的序列，前 3 项通常为 xyz_mm，后 3 项通常为 rpy_deg。
    raw_row:
        该行所有字段的原始字典，用于严格按当前 CSV 字段顺序回写。
    source_row_text:
        当前行对应的整行 CSV 文本，不带表头。
    """

    timestamp: str
    "记录时间戳。"

    record_type: str
    "记录类型。"

    joints_deg: tuple[float, ...]
    "关节角数组，单位 deg。"

    pose_values: tuple[Any, ...]
    "解析后的 pose 数值序列。"

    raw_row: dict[str, str]
    "当前行原始字段字典。"

    source_row_text: str
    "当前行整行 CSV 文本。"


@dataclass(frozen=True, slots=True)
class IkSolveResult:
    """CSV 行对应的逆解结果。

    该结构记录一次 IK 求解的状态和结果，不负责重试或刷新。
    它用于表格展示、编辑后即时反馈和复制前校验。

    Attributes
    ----------
    solved_joints_deg:
        求解得到的关节角数组，单位 deg。
    distance_deg:
        与输入关节角的欧氏距离，单位 deg。
    success:
        是否成功得到可用解。
    message:
        结果说明。
    """

    solved_joints_deg: tuple[float, ...]
    "求解得到的关节角数组，单位 deg。"

    distance_deg: float
    "与输入关节角的距离，单位 deg。"

    success: bool
    "是否成功。"

    message: str
    "结果说明。"


@dataclass(frozen=True, slots=True)
class PoseMatrix:
    """位姿矩阵与坐标轴信息。"""

    matrix: np.ndarray
    "4x4 齐次矩阵，单位 m 和弧度语义。"
    origin_mm: tuple[float, float, float]
    "位姿原点，单位 mm。"
    axes_mm: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    "x/y/z 轴方向向量，长度已按显示比例换算到 mm。"


# endregion


# region 基础解析


def _parse_list_field(raw_text: str) -> list[Any]:
    """解析 CSV 中的 Python 列表字符串。"""

    parsed = ast.literal_eval(raw_text)
    if not isinstance(parsed, list):
        raise ValueError(f"字段不是列表: {raw_text!r}")
    return parsed


def _format_float_sequence(values: list[float] | tuple[float, ...], decimals: int = 2) -> str:
    """将浮点序列格式化为便于显示的字符串。"""

    return ", ".join(f"{float(value):.{decimals}f}" for value in values)


def _row_to_csv_text(row: dict[str, str], fieldnames: list[str]) -> str:
    """把单行字典重新编码成一行 CSV 文本。"""

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writerow(row)
    return buffer.getvalue().strip("\r\n")


def _serialize_row(row: dict[str, str], fieldnames: list[str]) -> str:
    """按当前字段顺序输出整行 CSV，不含表头。"""

    return _row_to_csv_text(row, fieldnames)


def _deg_distance(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> float:
    """计算两个关节角序列的欧氏距离。"""

    if len(lhs) != len(rhs):
        raise ValueError(f"关节长度不一致: {len(lhs)} vs {len(rhs)}")
    diff = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    return float(np.linalg.norm(diff))


def _rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> tuple[float, float, float]:
    """把旋转矩阵转换为 xyz 顺序的 rpy 角，单位 deg。"""

    sy = float(np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
    singular = sy < 1e-9
    if not singular:
        rx = np.arctan2(rotation[2, 1], rotation[2, 2])
        ry = np.arctan2(-rotation[2, 0], sy)
        rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        rx = np.arctan2(-rotation[1, 2], rotation[1, 1])
        ry = np.arctan2(-rotation[2, 0], sy)
        rz = 0.0
    return float(np.rad2deg(rx)), float(np.rad2deg(ry)), float(np.rad2deg(rz))


def _pose_to_matrix_and_axes_m(pose_values: tuple[Any, ...], axis_scale_mm: float = 60.0) -> PoseMatrix:
    """将 CSV pose 转成 4x4 矩阵，并计算绘图用 xyz 轴。

    Parameters
    ----------
    pose_values:
        pose 字段解析结果，前 3 项为 xyz_mm，后 3 项为 rpy_deg。
    axis_scale_mm:
        绘图时坐标轴向量长度，单位 mm。

    Returns
    -------
    PoseMatrix
        包含齐次矩阵、原点和三个轴向量的结构。
    """

    if len(pose_values) < 6:
        raise ValueError(f"pose 长度不足，至少需要 6 项，实际 {len(pose_values)}")

    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = (float(value) for value in pose_values[:6])
    rx_rad = np.deg2rad(rx_deg)
    ry_rad = np.deg2rad(ry_deg)
    rz_rad = np.deg2rad(rz_deg)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(rx_rad), -np.sin(rx_rad)], [0.0, np.sin(rx_rad), np.cos(rx_rad)]], dtype=np.float64)
    ry = np.array([[np.cos(ry_rad), 0.0, np.sin(ry_rad)], [0.0, 1.0, 0.0], [-np.sin(ry_rad), 0.0, np.cos(ry_rad)]], dtype=np.float64)
    rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0.0], [np.sin(rz_rad), np.cos(rz_rad), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    rotation = rz @ ry @ rx
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[0, 3] = x_mm / MM_PER_M
    matrix[1, 3] = y_mm / MM_PER_M
    matrix[2, 3] = z_mm / MM_PER_M

    axes = tuple(tuple(float(v) for v in rotation[:, index] * axis_scale_mm) for index in range(3))
    return PoseMatrix(matrix=matrix, origin_mm=(x_mm, y_mm, z_mm), axes_mm=axes)  # type: ignore[arg-type]


def _pose_values_to_cartesian_position(pose_values: tuple[Any, ...]) -> xCoreSDK_python.CartesianPosition:
    """按 xCoreSDK 示例把 CSV pose 转成 SDK `CartesianPosition`。

    CSV 中前 3 项是 mm，后 3 项是 deg；SDK `CartesianPosition([...])`
    需要的是 m 和 rad。若 CSV 中带有 hasElbow / elbow / confData，也一并带入。
    """

    if len(pose_values) < 6:
        raise ValueError(f"pose 长度不足，至少需要 6 项，实际 {len(pose_values)}")

    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = (float(value) for value in pose_values[:6])
    cart_pose = xCoreSDK_python.CartesianPosition(
        [
            x_mm / MM_PER_M,
            y_mm / MM_PER_M,
            z_mm / MM_PER_M,
            float(np.deg2rad(rx_deg)),
            float(np.deg2rad(ry_deg)),
            float(np.deg2rad(rz_deg)),
        ]
    )
    if len(pose_values) >= 7:
        cart_pose.hasElbow = bool(pose_values[6])
    if len(pose_values) >= 8:
        cart_pose.elbow = float(np.deg2rad(float(pose_values[7])))
    if len(pose_values) >= 9 and isinstance(pose_values[8], (list, tuple)):
        cart_pose.confData = [int(value) for value in pose_values[8]]
    logger.debug(
        "构造 CartesianPosition: xyz_mm={} rpy_deg={} hasElbow={} elbow_deg={} confData={}",
        [float(value) for value in pose_values[:3]],
        [float(value) for value in pose_values[3:6]],
        bool(getattr(cart_pose, "hasElbow", False)),
        float(np.rad2deg(float(getattr(cart_pose, "elbow", 0.0)))),
        list(getattr(cart_pose, "confData", [])),
    )
    return cart_pose


def _format_pose_for_display(pose_values: tuple[Any, ...]) -> str:
    """把 pose 的前 6 个值格式化为便于查看的文本。"""

    return "[" + ", ".join(f"{float(value):.2f}" for value in pose_values[:6]) + "]"


def _parse_pose_text(raw_text: str) -> tuple[float, float, float, float, float, float]:
    """把用户编辑的 pose 文本解析为 xyzrpy 六元组。"""

    normalized = raw_text.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        parsed = _parse_list_field(normalized)
        if len(parsed) != 6:
            raise ValueError(f"pose 必须正好包含 6 个值，实际 {len(parsed)}")
        return tuple(float(value) for value in parsed)  # type: ignore[return-value]
    normalized = normalized.replace("，", ",")
    tokens = [token.strip() for token in normalized.split(",") if token.strip()]
    if len(tokens) != 6:
        raise ValueError(f"pose 必须正好包含 6 个逗号分隔值，实际 {len(tokens)}")
    return tuple(float(token) for token in tokens)  # type: ignore[return-value]


def _format_pose_text_for_editor(pose_values: tuple[Any, ...]) -> str:
    """把 pose 的前 6 个值转成可编辑文本。"""

    return ", ".join(f"{float(value):.2f}" for value in pose_values[:6])


def _parse_joints_text(raw_text: str, expected_len: int | None = None) -> tuple[float, ...]:
    """把用户编辑的 joints 文本解析为关节角序列，单位 deg。"""

    normalized = raw_text.strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        parsed = _parse_list_field(normalized)
        joints = tuple(float(value) for value in parsed)
        if expected_len is not None and len(joints) != expected_len:
            raise ValueError(f"joints 数量应为 {expected_len}，实际 {len(joints)}")
        return joints
    normalized = normalized.replace("，", ",")
    tokens = [token.strip() for token in normalized.split(",") if token.strip()]
    if not tokens:
        raise ValueError("joints 不能为空")
    joints = tuple(float(token) for token in tokens)
    if expected_len is not None and len(joints) != expected_len:
        raise ValueError(f"joints 数量应为 {expected_len}，实际 {len(joints)}")
    return joints


def _replace_pose_prefix(original_pose: tuple[Any, ...], new_pose6: tuple[float, float, float, float, float, float]) -> tuple[Any, ...]:
    """只替换 pose 的前 6 个值，后续上下文字段保持不变。"""

    suffix = tuple(original_pose[6:])
    return tuple(float(value) for value in new_pose6) + suffix


def _format_full_pose_for_csv(pose_values: tuple[Any, ...]) -> str:
    """把完整 pose 序列按 CSV 原格式要求序列化。"""

    parts: list[str] = []
    for value in pose_values:
        if isinstance(value, bool):
            parts.append("True" if value else "False")
        elif isinstance(value, (int, float, np.integer, np.floating)):
            parts.append(f"{float(value):.2f}")
        else:
            parts.append(str(value))
    return "[" + ", ".join(parts) + "]"


def _joint_values_to_cartesian_pose_values(fk_pose: xCoreSDK_python.CartesianPosition, original_pose: tuple[Any, ...]) -> tuple[Any, ...]:
    """把 FK 结果转回 CSV pose 语义，并尽量保留原上下文字段。"""

    pose_prefix = (
        float(fk_pose.trans[0]) * MM_PER_M,
        float(fk_pose.trans[1]) * MM_PER_M,
        float(fk_pose.trans[2]) * MM_PER_M,
        float(np.rad2deg(float(fk_pose.rpy[0]))),
        float(np.rad2deg(float(fk_pose.rpy[1]))),
        float(np.rad2deg(float(fk_pose.rpy[2]))),
    )
    has_elbow = bool(getattr(fk_pose, "hasElbow", False))
    elbow_deg = float(np.rad2deg(float(getattr(fk_pose, "elbow", 0.0))))
    conf_data = list(getattr(fk_pose, "confData", list(original_pose[8]) if len(original_pose) >= 9 and isinstance(original_pose[8], (list, tuple)) else []))
    return pose_prefix + (has_elbow, elbow_deg, conf_data)


def _infer_arm_side_from_csv_path(csv_path: Path) -> str | None:
    """根据 CSV 路径推断当前机械臂侧别。"""

    lowered = str(csv_path).replace("\\", "/").lower()
    if "record_left" in lowered or "_left_" in lowered or lowered.endswith("_left.csv"):
        return "left"
    if "record_right" in lowered or "_right_" in lowered or lowered.endswith("_right.csv"):
        return "right"
    return None


# endregion


# region CSV 读取


def _read_pose_csv(csv_path: Path) -> tuple[list[PoseCsvRow], list[str]]:
    """读取 CSV 并解析位姿和关节序列。

    当前调试页只处理 `type=arm` 的记录。像 `gripper,NaN,200.00`
    这类中间手掌控制行不参与绘图、IK/FK 或编辑，直接跳过。
    """

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    rows: list[PoseCsvRow] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("CSV 缺少表头")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            if not row:
                continue
            record_type = str(row.get("type", "")).strip().lower()
            if record_type != "arm":
                continue
            joints_raw = row.get("joints", "")
            pose_raw = row.get("pose", "")
            if not joints_raw or not pose_raw:
                continue
            joints_text = str(joints_raw).strip()
            pose_text = str(pose_raw).strip()
            if not joints_text.startswith("[") or not pose_text.startswith("["):
                continue
            joints_values = tuple(float(value) for value in _parse_list_field(joints_text))
            pose_values = tuple(_parse_list_field(pose_text))
            source_row_text = _serialize_row(row, fieldnames)
            rows.append(
                PoseCsvRow(
                    timestamp=row.get("timestamp", ""),
                    record_type=row.get("type", ""),
                    joints_deg=joints_values,
                    pose_values=pose_values,
                    raw_row=dict(row),
                    source_row_text=source_row_text,
                )
            )
    return rows, fieldnames


# endregion


# region IK 求解


def _normalize_ik_output(raw_result: Any) -> list[tuple[float, ...]]:
    """把 SDK IK 返回值整理成候选关节解列表。

    xCoreSDK `calcIk()` 返回的关节值单位为 rad，这里统一转换为 deg，
    以便与 CSV 中记录的关节角和界面显示保持一致。
    """

    if raw_result is None:
        return []
    if isinstance(raw_result, (list, tuple)) and raw_result and isinstance(raw_result[0], (list, tuple, np.ndarray)):
        return [tuple(float(np.rad2deg(value)) for value in candidate) for candidate in raw_result]
    if isinstance(raw_result, (list, tuple, np.ndarray)):
        return [tuple(float(np.rad2deg(value)) for value in raw_result)]
    return []


def _solve_nearest_ik(
    robot_model: Any,
    target_pose: xCoreSDK_python.CartesianPosition,
    reference_joints_deg: tuple[float, ...],
    ec: dict[str, object],
) -> IkSolveResult:
    """基于 SDK IK 结果选取与输入关节最接近的解。"""

    toolset = xCoreSDK_python.Toolset()
    logger.debug(
        "开始 calcIk: reference_joints_deg={} trans_m={} rpy_rad={} hasElbow={} elbow_rad={} confData={}",
        [float(value) for value in reference_joints_deg],
        list(getattr(target_pose, "trans", [])),
        list(getattr(target_pose, "rpy", [])),
        bool(getattr(target_pose, "hasElbow", False)),
        float(getattr(target_pose, "elbow", 0.0)),
        list(getattr(target_pose, "confData", [])),
    )
    raw_result = robot_model.calcIk(target_pose, toolset, ec)
    logger.debug("calcIk 返回: ec={} message={} raw_result={}", ec.get("ec", 0), ec.get("message", ""), raw_result)
    candidates = _normalize_ik_output(raw_result)
    logger.debug("calcIk 候选解(deg): {}", candidates)
    if not candidates:
        return IkSolveResult((), float("inf"), False, f"calcIk 失败: ec={ec.get('ec', 0)}, message={ec.get('message', '')}")

    best_candidate = min(candidates, key=lambda candidate: _deg_distance(candidate, reference_joints_deg))
    if best_candidate and all(abs(float(value)) < 1e-9 for value in best_candidate):
        logger.debug("calcIk 最优解为全 0，需重点检查输入 pose、elbow/confData 与模型上下文是否匹配")
    return IkSolveResult(
        solved_joints_deg=best_candidate,
        distance_deg=_deg_distance(best_candidate, reference_joints_deg),
        success=True,
        message="ok",
    )


# endregion


# region Qt 界面


class PoseCsvViewer(QMainWindow):
    """用于查看 CSV 位姿并调用 xCoreSDK 逆解的调试窗口。"""

    def __init__(self, csv_path: Path) -> None:
        super().__init__()
        self._csv_path = csv_path
        self._csv_fieldnames: list[str] = []
        self._rows: list[PoseCsvRow] = []
        self._ik_results: list[IkSolveResult] = []
        self._last_mtime_ns: int | None = None
        self._robot_model: Any | None = None

        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(CSV_REFRESH_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._reload_if_needed)

        self._figure = Figure(figsize=(7.0, 6.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")
        self._table = QTableWidget(self)
        self._detail = QPlainTextEdit(self)
        self._detail.setReadOnly(True)
        self._path_edit = QLineEdit(str(self._csv_path), self)
        self._browse_button = QPushButton("选择 CSV", self)
        self._reload_button = QPushButton("重新加载", self)
        self._copy_button = QPushButton("复制整行", self)
        self._save_button = QPushButton("保存回 CSV", self)
        self._pose_editor = QLineEdit(self)
        self._joints_editor = QLineEdit(self)
        self._status_label = QLabel("未加载", self)

        self._selected_row_index: int | None = None

        self._setup_ui()
        self._connect_signals()
        self._load_csv(initial=True)
        self._poll_timer.start()

    def _setup_ui(self) -> None:
        """构建界面。"""

        self.setWindowTitle("CSV Pose Viewer + xCoreSDK IK")
        self.resize(1520, 940)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("CSV 文件", self))
        top_bar.addWidget(self._path_edit, 1)
        top_bar.addWidget(self._browse_button)
        top_bar.addWidget(self._reload_button)
        top_bar.addWidget(self._copy_button)
        root_layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(self._canvas, 4)
        left_layout.addWidget(self._status_label)
        left_layout.addWidget(self._detail, 1)
        splitter.addWidget(left_panel)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["joints", "pose"])
        right_layout.addWidget(self._table, 5)

        edit_box = QWidget(self)
        edit_layout = QFormLayout(edit_box)
        self._pose_editor.setPlaceholderText("例如 327.64, 141.52, -45.84, -66.29, 32.88, -94.60")
        edit_layout.addRow("编辑 pose", self._pose_editor)
        edit_layout.addRow("编辑 joints", self._joints_editor)
        right_layout.addWidget(edit_box)

        button_bar = QHBoxLayout()
        button_bar.addWidget(self._save_button)
        button_bar.addStretch(1)
        right_layout.addLayout(button_bar)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter)

        copy_action = QAction("复制整行", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        self.addAction(copy_action)
        copy_action.triggered.connect(self._copy_current_row_csv)

    def _connect_signals(self) -> None:
        """连接 Qt 信号。"""

        self._table.itemSelectionChanged.connect(self._on_row_selection_changed)
        self._copy_button.clicked.connect(self._copy_current_row_csv)
        self._save_button.clicked.connect(self._save_current_row)
        self._reload_button.clicked.connect(self._load_csv_from_path_edit)
        self._browse_button.clicked.connect(self._browse_csv)
        self._pose_editor.editingFinished.connect(self._apply_pose_editor)
        self._joints_editor.editingFinished.connect(self._apply_joints_editor)
        self._path_edit.editingFinished.connect(self._load_csv_from_path_edit)

    @Slot()
    def _browse_csv(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(self, "选择 CSV", str(self._csv_path), "CSV Files (*.csv);;All Files (*)")
        if not path_text:
            return
        self._path_edit.setText(path_text)
        self._load_csv_from_path_edit()

    @Slot()
    def _load_csv_from_path_edit(self) -> None:
        self._csv_path = Path(self._path_edit.text().strip() or str(DEFAULT_CSV_PATH))
        self._robot_model = None
        self._load_csv(initial=False)

    @Slot()
    def _reload_if_needed(self) -> None:
        if not self._csv_path.exists():
            self._status_label.setText("CSV 不存在")
            return
        current_mtime_ns = self._csv_path.stat().st_mtime_ns
        if self._last_mtime_ns != current_mtime_ns:
            self._load_csv(initial=False)

    @Slot()
    def _on_file_changed(self) -> None:
        self._reload_if_needed()
        if self._csv_path.exists():
            self._watcher.addPath(str(self._csv_path))

    def _ensure_robot_model(self) -> bool:
        """在有机器人地址时构造 SDK 机器人模型。"""

        if self._robot_model is not None:
            return True
        arm_side = _infer_arm_side_from_csv_path(self._csv_path)
        if arm_side is None:
            self._status_label.setText("无法从 CSV 路径判断左右臂，IK 未启用")
            return False
        robot_ip = DEFAULT_ARM_ROBOT_IPS.get(arm_side, "").strip()
        if not robot_ip:
            self._status_label.setText(f"未配置 {arm_side} arm IP，IK 未启用")
            return False
        try:
            if DEFAULT_LOCAL_IP:
                sdk_robot = xCoreSDK_python.xMateErProRobot(robot_ip, DEFAULT_LOCAL_IP)
            else:
                sdk_robot = xCoreSDK_python.xMateErProRobot(robot_ip)
            self._robot_model = sdk_robot.model()
            self._status_label.setText(f"IK 已连接 {arm_side} arm: {robot_ip}")
            return True
        except Exception as exc:
            self._status_label.setText(f"IK 初始化失败: {exc}")
            self._robot_model = None
            return False

    def _load_csv(self, initial: bool) -> None:
        old_block = self._table.blockSignals(True)
        try:
            rows, fieldnames = _read_pose_csv(self._csv_path)
        except Exception as exc:
            self._table.blockSignals(old_block)
            self._status_label.setText(f"加载失败: {exc}")
            if initial:
                QMessageBox.critical(self, "CSV 加载失败", str(exc))
            return

        self._rows = rows
        self._csv_fieldnames = fieldnames
        self._last_mtime_ns = self._csv_path.stat().st_mtime_ns
        if self._csv_path.exists():
            self._watcher.addPath(str(self._csv_path))

        self._solve_all_rows()
        self._refresh_table()
        self._refresh_plot()
        self._restore_selection()
        self._status_label.setText(f"已加载 {len(self._rows)} 行")
        self._table.blockSignals(old_block)

    def _solve_all_rows(self) -> None:
        """对所有 CSV 行做 IK 求解。"""

        self._ik_results = []
        if not self._ensure_robot_model():
            self._ik_results = [IkSolveResult((), float("inf"), False, "IK 未启用") for _ in self._rows]
            return

        for row in self._rows:
            try:
                logger.debug(
                    "批量 IK 行: timestamp={} joints_deg={} pose_values={}",
                    row.timestamp,
                    list(row.joints_deg),
                    list(row.pose_values),
                )
                target_pose = _pose_values_to_cartesian_position(row.pose_values)
                ec: dict[str, object] = {}
                result = _solve_nearest_ik(self._robot_model, target_pose, row.joints_deg, ec)
                self._ik_results.append(result)
            except Exception as exc:
                logger.debug("批量 IK 失败: timestamp={} exc={}", row.timestamp, repr(exc))
                self._ik_results.append(IkSolveResult((), float("inf"), False, str(exc)))

    def _refresh_table(self) -> None:
        """刷新右侧表格。"""

        self._table.setRowCount(len(self._rows))
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["joints", "pose"])
        for row_index, row in enumerate(self._rows):
            joints_text = _format_float_sequence(row.joints_deg)
            pose_text = _format_pose_for_display(row.pose_values)
            self._table.setItem(row_index, 0, QTableWidgetItem(joints_text))
            self._table.setItem(row_index, 1, QTableWidgetItem(pose_text))
        self._table.resizeColumnsToContents()

    def _refresh_plot(self) -> None:
        """按顺序绘制 CSV pose、连线并画出 xyz 轴。"""

        self._axes.clear()
        self._axes.set_title("Pose path")
        self._axes.set_xlabel("X (mm)")
        self._axes.set_ylabel("Y (mm)")
        self._axes.set_zlabel("Z (mm)")
        if not self._rows:
            self._canvas.draw_idle()
            return

        pose_infos = [_pose_to_matrix_and_axes_m(row.pose_values) for row in self._rows]
        points = np.array([info.origin_mm for info in pose_infos], dtype=np.float64)
        self._axes.plot(points[:, 0], points[:, 1], points[:, 2], "-o", linewidth=2.0, markersize=4.0)
        self._axes.scatter(points[0, 0], points[0, 1], points[0, 2], c="green", s=80, label="start")
        self._axes.scatter(points[-1, 0], points[-1, 1], points[-1, 2], c="red", s=80, label="end")

        for index, info in enumerate(pose_infos):
            origin = np.array(info.origin_mm, dtype=np.float64)
            x_axis = origin + np.array(info.axes_mm[0], dtype=np.float64)
            y_axis = origin + np.array(info.axes_mm[1], dtype=np.float64)
            z_axis = origin + np.array(info.axes_mm[2], dtype=np.float64)
            alpha = 0.35 if index != self._selected_row_index else 0.95
            self._axes.quiver(*origin, *(x_axis - origin), color="#d62828", alpha=alpha, length=1.0, normalize=False)
            self._axes.quiver(*origin, *(y_axis - origin), color="#2a9d8f", alpha=alpha, length=1.0, normalize=False)
            self._axes.quiver(*origin, *(z_axis - origin), color="#264653", alpha=alpha, length=1.0, normalize=False)

        self._axes.legend(loc="best")
        self._canvas.draw_idle()

    def _restore_selection(self) -> None:
        """恢复选中行并同步编辑区。"""

        if not self._rows:
            self._selected_row_index = None
            self._pose_editor.clear()
            self._joints_editor.clear()
            self._detail.setPlainText("请选择一行")
            return

        if self._selected_row_index is None or self._selected_row_index >= len(self._rows):
            self._selected_row_index = 0
        self._table.selectRow(self._selected_row_index)
        self._sync_editor_from_row()

    def _current_row_index(self) -> int | None:
        indexes = self._table.selectionModel().selectedRows() if self._table.selectionModel() is not None else []
        if indexes:
            return int(indexes[0].row())
        return self._selected_row_index

    def _current_row(self) -> PoseCsvRow | None:
        row_index = self._current_row_index()
        if row_index is None or row_index < 0 or row_index >= len(self._rows):
            return None
        return self._rows[row_index]

    def _sync_editor_from_row(self) -> None:
        """把当前行内容同步到编辑区。"""

        row = self._current_row()
        if row is None:
            self._pose_editor.clear()
            self._joints_editor.clear()
            return
        with QSignalBlocker(self._pose_editor), QSignalBlocker(self._joints_editor):
            self._pose_editor.setText(_format_pose_text_for_editor(row.pose_values))
            self._joints_editor.setText(_format_float_sequence(row.joints_deg))
            self._joints_editor.setPlaceholderText(
                f"请输入 {len(row.joints_deg)} 个关节值，例如 {_format_float_sequence(row.joints_deg)}"
            )
        self._update_detail_from_selection()

    @Slot()
    def _on_row_selection_changed(self) -> None:
        row_index = self._current_row_index()
        if row_index is None:
            return
        self._selected_row_index = row_index
        self._sync_editor_from_row()
        self._refresh_plot()

    @Slot()
    def _apply_pose_editor(self) -> None:
        row_index = self._current_row_index()
        if row_index is None:
            return
        row = self._rows[row_index]
        try:
            pose_values = _parse_pose_text(self._pose_editor.text())
            ec: dict[str, object] = {}
            updated_pose = _replace_pose_prefix(row.pose_values, pose_values)
            logger.debug(
                "手动编辑 pose: row={} original_pose={} edited_pose6={} updated_pose={}",
                row_index,
                list(row.pose_values),
                list(pose_values),
                list(updated_pose),
            )
            target_pose = _pose_values_to_cartesian_position(updated_pose)
            if self._ensure_robot_model() and self._robot_model is not None:
                result = _solve_nearest_ik(self._robot_model, target_pose, row.joints_deg, ec)
            else:
                result = IkSolveResult((), float("inf"), False, "IK 未启用")
            updated_row = PoseCsvRow(
                timestamp=row.timestamp,
                record_type=row.record_type,
                joints_deg=result.solved_joints_deg if result.success and result.solved_joints_deg else row.joints_deg,
                pose_values=updated_pose,
                raw_row={**row.raw_row, "joints": self._format_joints_for_csv(result, row), "pose": _format_pose_text_for_editor(updated_pose)},
                source_row_text=row.source_row_text,
            )
            self._rows[row_index] = updated_row
            self._ik_results[row_index] = result
            self._refresh_table()
            self._refresh_plot()
            self._sync_editor_from_row()
            if result.success and result.solved_joints_deg:
                self._status_label.setText(
                    f"第 {row_index + 1} 行 IK 已更新: {_format_float_sequence(result.solved_joints_deg)}"
                )
            else:
                self._status_label.setText(f"第 {row_index + 1} 行 IK 未更新: {result.message}")
        except Exception as exc:
            self._status_label.setText(f"第 {row_index + 1} 行 pose 解析失败: {exc}")
            QMessageBox.warning(self, "pose 解析失败", str(exc))

    @Slot()
    def _apply_joints_editor(self) -> None:
        row_index = self._current_row_index()
        if row_index is None:
            return
        row = self._rows[row_index]
        try:
            joints_deg = _parse_joints_text(self._joints_editor.text(), expected_len=len(row.joints_deg))
            if not self._ensure_robot_model() or self._robot_model is None:
                raise RuntimeError("IK/FK 未启用")
            ec: dict[str, object] = {}
            joint_values_rad = [float(np.deg2rad(value)) for value in joints_deg]
            logger.debug(
                "手动编辑 joints: row={} joints_deg={} joints_rad={}",
                row_index,
                list(joints_deg),
                joint_values_rad,
            )
            fk_pose = self._robot_model.calcFk(joint_values_rad, ec)
            logger.debug(
                "calcFk 返回: ec={} message={} trans_m={} rpy_rad={} hasElbow={} elbow_rad={} confData={}",
                ec.get("ec", 0),
                ec.get("message", ""),
                list(getattr(fk_pose, "trans", [])),
                list(getattr(fk_pose, "rpy", [])),
                bool(getattr(fk_pose, "hasElbow", False)),
                float(getattr(fk_pose, "elbow", 0.0)),
                list(getattr(fk_pose, "confData", [])),
            )
            if ec.get("ec", 0) != 0:
                raise RuntimeError(f"calcFk 失败: ec={ec.get('ec', 0)}, message={ec.get('message', '')}")
            updated_pose = _joint_values_to_cartesian_pose_values(fk_pose, row.pose_values)
            logger.debug("calcFk 转回 pose_values={}", list(updated_pose))
            updated_row = PoseCsvRow(
                timestamp=row.timestamp,
                record_type=row.record_type,
                joints_deg=joints_deg,
                pose_values=updated_pose,
                raw_row={
                    **row.raw_row,
                    "joints": "[" + ", ".join(f"{float(value):.2f}" for value in joints_deg) + "]",
                    "pose": _format_full_pose_for_csv(updated_pose),
                },
                source_row_text=row.source_row_text,
            )
            self._rows[row_index] = updated_row
            self._ik_results[row_index] = IkSolveResult(joints_deg, 0.0, True, "fk-updated")
            self._refresh_table()
            self._refresh_plot()
            self._sync_editor_from_row()
            self._status_label.setText(f"第 {row_index + 1} 行 FK 已更新: {_format_pose_text_for_editor(updated_pose)}")
        except Exception as exc:
            self._status_label.setText(f"第 {row_index + 1} 行 joints 解析失败: {exc}")
            QMessageBox.warning(self, "joints 解析失败", str(exc))

    def _format_joints_for_csv(self, result: IkSolveResult, row: PoseCsvRow) -> str:
        """把当前关节值按 CSV 里的列表格式序列化。"""

        return "[" + ", ".join(f"{float(value):.2f}" for value in row.joints_deg) + "]"

    def _serialize_current_row(self) -> str:
        """按当前表头顺序输出选中行整行 CSV。"""

        row_index = self._current_row_index()
        if row_index is None:
            raise RuntimeError("没有可复制的行")
        row = self._rows[row_index]
        output_row = dict(row.raw_row)
        output_row["joints"] = self._format_joints_for_csv(self._ik_results[row_index], row)
        output_row["pose"] = _format_full_pose_for_csv(row.pose_values)
        return _serialize_row(output_row, self._csv_fieldnames)

    @Slot()
    def _copy_current_row_csv(self) -> None:
        try:
            csv_text = self._serialize_current_row()
        except Exception as exc:
            QMessageBox.warning(self, "复制失败", str(exc))
            return
        QApplication.clipboard().setText(csv_text)
        self._status_label.setText("已复制当前行 CSV")

    @Slot()
    def _save_current_row(self) -> None:
        row_index = self._current_row_index()
        if row_index is None:
            return
        try:
            pose_values = _parse_pose_text(self._pose_editor.text())
            ec: dict[str, object] = {}
            updated_pose = _replace_pose_prefix(self._rows[row_index].pose_values, pose_values)
            target_pose = _pose_values_to_cartesian_position(updated_pose)
            if self._ensure_robot_model() and self._robot_model is not None:
                result = _solve_nearest_ik(self._robot_model, target_pose, self._rows[row_index].joints_deg, ec)
            else:
                result = IkSolveResult((), float("inf"), False, "IK 未启用")
            row = self._rows[row_index]
            new_row = PoseCsvRow(
                timestamp=row.timestamp,
                record_type=row.record_type,
                joints_deg=result.solved_joints_deg if result.success and result.solved_joints_deg else row.joints_deg,
                pose_values=updated_pose,
                raw_row={**row.raw_row, "joints": self._format_joints_for_csv(result, row), "pose": _format_pose_text_for_editor(updated_pose)},
                source_row_text=row.source_row_text,
            )
            self._rows[row_index] = new_row
            self._ik_results[row_index] = result
            self._write_csv_back()
            self._refresh_table()
            self._refresh_plot()
            self._sync_editor_from_row()
            if result.success and result.solved_joints_deg:
                self._status_label.setText(f"已保存第 {row_index + 1} 行，IK joints={_format_float_sequence(result.solved_joints_deg)}")
            else:
                self._status_label.setText(f"已保存第 {row_index + 1} 行，但 IK 未更新: {result.message}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", str(exc))

    def _write_csv_back(self) -> None:
        """把当前 rows 写回 CSV 文件。"""

        if not self._csv_fieldnames:
            raise RuntimeError("CSV 表头为空")
        with self._csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self._csv_fieldnames)
            writer.writeheader()
            for row_index, row in enumerate(self._rows):
                row_dict = dict(row.raw_row)
                ik_result = self._ik_results[row_index] if row_index < len(self._ik_results) else IkSolveResult((), float("inf"), False, "missing")
                row_dict["joints"] = self._format_joints_for_csv(ik_result, row)
                row_dict["pose"] = _format_full_pose_for_csv(row.pose_values)
                writer.writerow(row_dict)

    def _update_detail_from_selection(self) -> None:
        """显示当前选中行的详细信息。"""

        row = self._current_row()
        if row is None:
            self._detail.setPlainText("请选择一行")
            return
        row_index = self._current_row_index() or 0
        ik_result = self._ik_results[row_index] if row_index < len(self._ik_results) else IkSolveResult((), float("inf"), False, "missing")
        matrix_info = _pose_to_matrix_and_axes_m(row.pose_values)
        detail = [
            f"joints_deg: {_format_float_sequence(row.joints_deg)}",
            f"pose_raw: {_format_pose_text_for_editor(row.pose_values)}",
            f"origin_mm: {_format_float_sequence(matrix_info.origin_mm)}",
            f"x_axis_mm: {_format_float_sequence(matrix_info.axes_mm[0])}",
            f"y_axis_mm: {_format_float_sequence(matrix_info.axes_mm[1])}",
            f"z_axis_mm: {_format_float_sequence(matrix_info.axes_mm[2])}",
            f"ik_success: {ik_result.success}",
            f"ik_joints_deg: {_format_float_sequence(ik_result.solved_joints_deg) if ik_result.success else ik_result.message}",
        ]
        self._detail.setPlainText("\n".join(detail))


# endregion


# region 主入口


def main() -> int:
    """启动位姿 CSV 调试窗口。"""

    app = QApplication.instance() or QApplication(sys.argv)
    window = PoseCsvViewer(DEFAULT_CSV_PATH)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
