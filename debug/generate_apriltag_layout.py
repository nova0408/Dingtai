from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import sys

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "Downloads"
DEFAULT_DICTIONARY_NAME = "DICT_APRILTAG_16H5"
DEFAULT_TAG_IDS_TEXT = "3,4,5"
DEFAULT_ROWS = 1
DEFAULT_ROW_COUNTS_TEXT = "3"
DEFAULT_TAG_SIZE_PX = 240
DEFAULT_TAG_GAP_PX = 60
DEFAULT_ROTATIONS_TEXT = "0,0,0"
DEFAULT_BACKGROUND_COLOR = (245, 245, 245)

DICTIONARY_NAME_TO_ID: dict[str, int] = {
    "DICT_APRILTAG_16H5": int(cv2.aruco.DICT_APRILTAG_16h5),
    "DICT_APRILTAG_25H9": int(cv2.aruco.DICT_APRILTAG_25h9),
    "DICT_APRILTAG_36H10": int(cv2.aruco.DICT_APRILTAG_36h10),
    "DICT_APRILTAG_36H11": int(cv2.aruco.DICT_APRILTAG_36h11),
}


@dataclass(frozen=True)
class LayoutTagSpec:
    tag_id: int
    row_index: int
    col_index: int
    rotation_deg: float
    x_px: int
    y_px: int


@dataclass(frozen=True)
class LayoutConfig:
    dictionary_name: str
    tag_ids: list[int]
    row_counts: list[int]
    tag_size_px: int
    gap_px: int
    rotations_deg: list[float]
    output_root: Path


class ApriltagLayoutBuilderWindow(QMainWindow):
    # region 初始化

    def __init__(self) -> None:
        super().__init__()
        self._output_root = DEFAULT_OUTPUT_ROOT
        self._preview_image: np.ndarray | None = None

        self._dictionary_combo: QLineEdit
        self._tag_ids_input: QLineEdit
        self._rows_spin: QSpinBox
        self._row_counts_input: QLineEdit
        self._tag_size_input: QLineEdit
        self._gap_input: QLineEdit
        self._rotations_input: QLineEdit
        self._output_root_label: QLabel
        self._browse_output_button: QPushButton
        self._preview_label: QLabel
        self._status_label: QLabel
        self._save_button: QPushButton
        self._json_preview: QTextEdit

        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._refresh_preview()

    def _setup_window(self) -> None:
        self.setWindowTitle("AprilTag Multi-Layout Builder")
        self.resize(1100, 840)

    def _setup_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.addWidget(self._build_config_group())
        root_layout.addWidget(self._build_preview_group(), 1)
        root_layout.addWidget(self._build_status_group())
        self.setCentralWidget(central)

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("布局参数", self)
        layout = QFormLayout(group)

        self._dictionary_combo = QLineEdit(DEFAULT_DICTIONARY_NAME, group)
        self._tag_ids_input = QLineEdit(DEFAULT_TAG_IDS_TEXT, group)

        self._rows_spin = QSpinBox(group)
        self._rows_spin.setRange(1, 32)
        self._rows_spin.setValue(DEFAULT_ROWS)

        self._row_counts_input = QLineEdit(DEFAULT_ROW_COUNTS_TEXT, group)
        self._row_counts_input.setPlaceholderText("例如 3,2,4，表示每行依次放几个 tag")

        self._tag_size_input = QLineEdit(str(DEFAULT_TAG_SIZE_PX), group)
        self._tag_size_input.setValidator(QIntValidator(8, 10000, self._tag_size_input))

        self._gap_input = QLineEdit(str(DEFAULT_TAG_GAP_PX), group)
        self._gap_input.setValidator(QIntValidator(0, 10000, self._gap_input))

        self._rotations_input = QLineEdit(DEFAULT_ROTATIONS_TEXT, group)
        self._rotations_input.setPlaceholderText("每个 tag 的旋转角度，单位 deg，例如 0,15,-30")

        self._output_root_label = QLabel(str(self._output_root), group)
        self._output_root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._browse_output_button = QPushButton("选择输出目录", group)
        self._save_button = QPushButton("保存布局", group)

        layout.addRow("字典", self._dictionary_combo)
        layout.addRow("tag_id 列表", self._tag_ids_input)
        layout.addRow("行数", self._rows_spin)
        layout.addRow("每行 tag 数", self._row_counts_input)
        layout.addRow("单 tag 尺寸 (px)", self._tag_size_input)
        layout.addRow("行/列间距 (px)", self._gap_input)
        layout.addRow("每个 tag 旋转 (deg)", self._rotations_input)
        layout.addRow("输出目录", self._build_output_selector_row(group))
        layout.addRow("", self._save_button)
        return group

    def _build_output_selector_row(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._output_root_label, 1)
        row.addWidget(self._browse_output_button)
        return container

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("实时预览", self)
        layout = QVBoxLayout(group)

        scroll = QScrollArea(group)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._preview_label = QLabel(scroll)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet("background: #f5f5f5; border: 1px solid #b8b8b8;")
        self._preview_label.setMinimumSize(800, 520)
        scroll.setWidget(self._preview_label)

        self._json_preview = QTextEdit(group)
        self._json_preview.setReadOnly(True)
        self._json_preview.setMinimumHeight(180)
        self._json_preview.setPlaceholderText("布局 JSON 预览")

        layout.addWidget(scroll, 2)
        layout.addWidget(self._json_preview, 1)
        return group

    def _build_status_group(self) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self._status_label = QLabel("等待编辑布局参数。", container)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._status_label, 1)
        return container

    def _connect_signals(self) -> None:
        self._dictionary_combo.textChanged.connect(self._on_inputs_changed)
        self._tag_ids_input.textChanged.connect(self._on_inputs_changed)
        self._rows_spin.valueChanged.connect(self._on_inputs_changed)
        self._row_counts_input.textChanged.connect(self._on_inputs_changed)
        self._tag_size_input.textChanged.connect(self._on_inputs_changed)
        self._gap_input.textChanged.connect(self._on_inputs_changed)
        self._rotations_input.textChanged.connect(self._on_inputs_changed)
        self._browse_output_button.clicked.connect(self._on_browse_output_clicked)
        self._save_button.clicked.connect(self._on_save_clicked)

    # endregion

    # region 交互

    @Slot()
    def _on_inputs_changed(self) -> None:
        self._refresh_preview()

    @Slot()
    def _on_browse_output_clicked(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            str(self._output_root),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected_dir:
            return
        self._output_root = Path(selected_dir)
        self._output_root_label.setText(str(self._output_root))
        self._refresh_preview()

    @Slot()
    def _on_save_clicked(self) -> None:
        try:
            config = self._read_config()
            payload = self._build_layout_payload(config)
            output_dir = self._create_output_dir(config.output_root)
            image_path = output_dir / "apriltag_layout.png"
            json_path = output_dir / "apriltag_layout.json"
            if self._preview_image is None:
                raise RuntimeError("预览图为空，无法保存。")
            if not cv2.imwrite(str(image_path), cv2.cvtColor(self._preview_image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"保存图片失败：{image_path}")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))
            self._status_label.setText(f"保存失败: {exc}")
            return

        self._status_label.setText(f"已保存布局到：{output_dir}")
        QMessageBox.information(
            self,
            "保存完成",
            f"布局图片与 JSON 已保存。\n目录：\n{output_dir}",
        )

    # endregion

    # region 预览与构造

    def _refresh_preview(self) -> None:
        try:
            config = self._read_config()
            payload = self._build_layout_payload(config)
            preview_image = self._render_layout(config)
        except Exception as exc:  # noqa: BLE001
            self._preview_image = None
            self._preview_label.setText(f"预览不可用:\n{exc}")
            self._json_preview.setPlainText(str(exc))
            self._status_label.setText(f"参数错误: {exc}")
            return

        self._preview_image = preview_image
        pixmap = self._to_pixmap(preview_image)
        scaled = pixmap.scaled(
            self._preview_label.size().expandedTo(pixmap.size()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)
        self._json_preview.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self._status_label.setText(
            f"已生成预览：{len(config.tag_ids)} 个 tag，{len(config.row_counts)} 行。"
        )

    def _read_config(self) -> LayoutConfig:
        dictionary_name = self._dictionary_combo.text().strip()
        if dictionary_name not in DICTIONARY_NAME_TO_ID:
            raise ValueError(f"不支持的字典名：{dictionary_name}")

        tag_ids = self._parse_int_list(self._tag_ids_input.text(), "tag_id 列表")
        if not tag_ids:
            raise ValueError("tag_id 列表不能为空。")

        rows = int(self._rows_spin.value())
        row_counts = self._parse_int_list(self._row_counts_input.text(), "每行 tag 数")
        if len(row_counts) != rows:
            raise ValueError(f"行数是 {rows}，但每行 tag 数提供了 {len(row_counts)} 个值。")
        if any(count <= 0 for count in row_counts):
            raise ValueError("每行 tag 数必须都大于 0。")
        if sum(row_counts) > len(tag_ids):
            raise ValueError("tag_id 数量不足以填满所有行。")

        tag_size_px = int(self._tag_size_input.text().strip() or "0")
        if tag_size_px <= 0:
            raise ValueError("单 tag 尺寸必须大于 0。")

        gap_px = int(self._gap_input.text().strip() or "0")
        if gap_px < 0:
            raise ValueError("间距不能为负数。")

        rotations_deg = self._parse_float_list(self._rotations_input.text(), "旋转角度")
        if len(rotations_deg) != len(tag_ids):
            raise ValueError(f"旋转角度数量 {len(rotations_deg)} 必须与 tag_id 数量 {len(tag_ids)} 一致。")

        return LayoutConfig(
            dictionary_name=dictionary_name,
            tag_ids=tag_ids,
            row_counts=row_counts,
            tag_size_px=tag_size_px,
            gap_px=gap_px,
            rotations_deg=rotations_deg,
            output_root=self._output_root,
        )

    def _build_layout_payload(self, config: LayoutConfig) -> dict[str, object]:
        specs = self._build_tag_specs(config)
        edge_px = max(0, config.gap_px)
        row_heights = [config.tag_size_px for _ in config.row_counts]
        row_widths = [
            count * config.tag_size_px + max(0, count - 1) * config.gap_px
            for count in config.row_counts
        ]
        canvas_width = max(row_widths) + 2 * edge_px
        canvas_height = sum(row_heights) + max(0, len(row_heights) - 1) * config.gap_px + 2 * edge_px
        return {
            "dictionary_name": config.dictionary_name,
            "tag_size_px": config.tag_size_px,
            "gap_px": config.gap_px,
            "edge_px": edge_px,
            "canvas_size_px": [canvas_width, canvas_height],
            "rows": len(config.row_counts),
            "row_counts": config.row_counts,
            "tags": [
                {
                    "tag_id": spec.tag_id,
                    "row_index": spec.row_index,
                    "col_index": spec.col_index,
                    "rotation_deg": spec.rotation_deg,
                    "position_px": [spec.x_px, spec.y_px],
                }
                for spec in specs
            ],
        }

    def _build_tag_specs(self, config: LayoutConfig) -> list[LayoutTagSpec]:
        specs: list[LayoutTagSpec] = []
        tag_index = 0
        edge_px = max(0, config.gap_px)
        y_cursor = edge_px
        for row_index, row_count in enumerate(config.row_counts):
            row_width = row_count * config.tag_size_px + max(0, row_count - 1) * config.gap_px
            x_cursor = edge_px
            for col_index in range(row_count):
                specs.append(
                    LayoutTagSpec(
                        tag_id=config.tag_ids[tag_index],
                        row_index=row_index,
                        col_index=col_index,
                        rotation_deg=config.rotations_deg[tag_index],
                        x_px=x_cursor,
                        y_px=y_cursor,
                    )
                )
                tag_index += 1
                x_cursor += config.tag_size_px + config.gap_px
            y_cursor += config.tag_size_px + config.gap_px
        return specs

    def _render_layout(self, config: LayoutConfig) -> np.ndarray:
        specs = self._build_tag_specs(config)
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_NAME_TO_ID[config.dictionary_name])
        tag_lookup = {spec.tag_id: spec for spec in specs}

        row_heights = [config.tag_size_px for _ in config.row_counts]
        row_widths = [
            count * config.tag_size_px + max(0, count - 1) * config.gap_px
            for count in config.row_counts
        ]
        edge_px = max(0, config.gap_px)
        canvas_width = max(row_widths) + 2 * edge_px
        canvas_height = sum(row_heights) + max(0, len(row_heights) - 1) * config.gap_px + 2 * edge_px

        canvas = np.full((canvas_height, canvas_width, 3), DEFAULT_BACKGROUND_COLOR, dtype=np.uint8)
        y_cursor = edge_px
        tag_index = 0
        for row_index, row_count in enumerate(config.row_counts):
            x_cursor = edge_px
            for col_index in range(row_count):
                tag_id = config.tag_ids[tag_index]
                rotation_deg = tag_lookup[tag_id].rotation_deg
                marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), config.tag_size_px, borderBits=1)
                marker_rgb = self._marker_to_rgb(marker)
                rotated = self._rotate_marker(marker_rgb, rotation_deg, DEFAULT_BACKGROUND_COLOR)
                self._paste_marker(canvas, rotated, x_cursor, y_cursor)
                self._draw_label(canvas, f"id={tag_id} {rotation_deg:.1f}deg", x_cursor, y_cursor + config.tag_size_px + 28)
                x_cursor += config.tag_size_px + config.gap_px
                tag_index += 1
            y_cursor += config.tag_size_px + config.gap_px

        header = f"{config.dictionary_name} | tags={config.tag_ids}"
        self._draw_header(canvas, header)
        return canvas

    @staticmethod
    def _marker_to_rgb(marker_binary: np.ndarray) -> np.ndarray:
        rgb = np.empty((marker_binary.shape[0], marker_binary.shape[1], 3), dtype=np.uint8)
        rgb[:] = (245, 245, 245)
        rgb[marker_binary == 0] = (0, 0, 0)
        return rgb

    @staticmethod
    def _rotate_marker(marker_rgb: np.ndarray, rotation_deg: float, background_color: tuple[int, int, int]) -> np.ndarray:
        if abs(rotation_deg) < 1e-6:
            return marker_rgb
        height, width = marker_rgb.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0)
        return cv2.warpAffine(
            marker_rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background_color,
        )

    @staticmethod
    def _paste_marker(canvas: np.ndarray, marker_rgb: np.ndarray, x: int, y: int) -> None:
        h, w = marker_rgb.shape[:2]
        canvas[y : y + h, x : x + w] = marker_rgb

    @staticmethod
    def _draw_label(canvas: np.ndarray, text: str, x: int, y: int) -> None:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_header(canvas: np.ndarray, text: str) -> None:
        cv2.putText(canvas, text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)

    # endregion

    # region 工具

    @staticmethod
    def _parse_int_list(text: str, field_name: str) -> list[int]:
        values = [item.strip() for item in text.split(",") if item.strip()]
        if not values:
            raise ValueError(f"{field_name} 不能为空。")
        try:
            return [int(item) for item in values]
        except ValueError as exc:
            raise ValueError(f"{field_name} 必须是逗号分隔整数。") from exc

    @staticmethod
    def _parse_float_list(text: str, field_name: str) -> list[float]:
        values = [item.strip() for item in text.split(",") if item.strip()]
        if not values:
            raise ValueError(f"{field_name} 不能为空。")
        try:
            return [float(item) for item in values]
        except ValueError as exc:
            raise ValueError(f"{field_name} 必须是逗号分隔数字。") from exc

    @staticmethod
    def _create_output_dir(output_root: Path) -> Path:
        output_dir = output_root / datetime.now().strftime("%m%d-%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        contiguous = np.ascontiguousarray(image)
        height, width, _ = contiguous.shape
        qimage = QImage(
            contiguous.data,
            width,
            height,
            int(contiguous.strides[0]),
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    # endregion


def main() -> int:
    app = QApplication(sys.argv)
    window = ApriltagLayoutBuilderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
