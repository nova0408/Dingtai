from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QImage, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ColorValue = tuple[int, int, int]

DICTIONARY_OPTIONS: list[str] = [
    "DICT_4X4",
    "DICT_5X5",
    "DICT_6X6",
    "DICT_7X7",
]

DICTIONARY_FAMILY_TO_PREDEFINED_NAME: dict[str, str] = {
    "DICT_4X4": "DICT_4X4_1000",
    "DICT_5X5": "DICT_5X5_1000",
    "DICT_6X6": "DICT_6X6_1000",
    "DICT_7X7": "DICT_7X7_1000",
}


@dataclass(frozen=True)
class MarkerGenerationConfig:
    """ArUco 标记生成参数。"""

    dictionary_name: str
    size_px: int
    start_id: int
    count: int
    foreground_color: ColorValue
    background_color: ColorValue
    add_outer_outline: bool
    output_root: Path


class ArucoMarkerGeneratorWindow(QMainWindow):
    """单个 ArUco 标记块批量生成窗口。"""

    # region 初始化

    def __init__(self) -> None:
        super().__init__()
        self._download_root = self._resolve_download_root()

        self._dictionary_name_combo: QComboBox
        self._quantity_spin: QSpinBox
        self._size_input: QLineEdit
        self._start_id_spin: QSpinBox
        self._foreground_color_preview: QLabel
        self._foreground_color_button: QPushButton
        self._background_color_preview: QLabel
        self._background_color_button: QPushButton
        self._outline_checkbox: QCheckBox
        self._output_root_label: QLabel
        self._browse_output_button: QPushButton
        self._output_dir_label: QLabel
        self._preview_label: QLabel
        self._status_label: QLabel
        self._generate_button: QPushButton
        self._foreground_color: ColorValue = (0, 0, 0)
        self._background_color: ColorValue = (255, 255, 255)
        self._output_root = self._download_root

        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._apply_dictionary_limits()
        self._refresh_preview()

    def _setup_window(self) -> None:
        self.setWindowTitle("ArUco Marker Generator")
        self.resize(820, 700)

    def _setup_ui(self) -> None:
        central_widget = QWidget(self)
        root_layout = QVBoxLayout(central_widget)
        root_layout.addWidget(self._build_config_group())
        root_layout.addWidget(self._build_preview_group(), 1)
        root_layout.addWidget(self._build_status_group())
        self.setCentralWidget(central_widget)

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("生成参数", self)
        layout = QFormLayout(group)

        self._dictionary_name_combo = QComboBox(group)
        for dictionary_name in DICTIONARY_OPTIONS:
            self._dictionary_name_combo.addItem(dictionary_name, dictionary_name)
        self._dictionary_name_combo.setCurrentText("DICT_4X4")

        self._quantity_spin = QSpinBox(group)
        self._quantity_spin.setRange(1, 1000)
        self._quantity_spin.setValue(1)

        self._size_input = QLineEdit("256", group)
        self._size_input.setValidator(QIntValidator(32, 8192, self._size_input))
        self._size_input.setPlaceholderText("输入单个标记边长，单位 px")

        self._start_id_spin = QSpinBox(group)
        self._start_id_spin.setValue(0)

        self._foreground_color_preview = self._build_color_preview_label(group)
        self._foreground_color_button = QPushButton("选择前景色", group)

        self._background_color_preview = self._build_color_preview_label(group)
        self._background_color_button = QPushButton("选择背景色", group)

        self._outline_checkbox = QCheckBox("额外添加一层外围白色描边", group)
        self._outline_checkbox.setChecked(False)

        self._output_root_label = QLabel(str(self._output_root), group)
        self._output_root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._browse_output_button = QPushButton("选择输出目录", group)

        self._output_dir_label = QLabel(self._build_timestamp_dir_hint(), group)
        self._output_dir_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._generate_button = QPushButton("生成标记", group)

        layout.addRow("字典", self._dictionary_name_combo)
        layout.addRow("数量", self._quantity_spin)
        layout.addRow("尺寸 (px)", self._size_input)
        layout.addRow("起始 ID", self._start_id_spin)
        layout.addRow("前景色", self._build_color_selector_row(group, self._foreground_color_preview, self._foreground_color_button))
        layout.addRow("背景色", self._build_color_selector_row(group, self._background_color_preview, self._background_color_button))
        layout.addRow("外围描边", self._outline_checkbox)
        layout.addRow("默认输出目录", self._build_output_selector_row(group))
        layout.addRow("本次目录预览", self._output_dir_label)
        layout.addRow("", self._generate_button)
        self._update_color_preview(self._foreground_color_preview, self._foreground_color)
        self._update_color_preview(self._background_color_preview, self._background_color)
        return group

    @staticmethod
    def _build_color_preview_label(parent: QWidget) -> QLabel:
        label = QLabel(parent)
        label.setFixedSize(56, 24)
        label.setStyleSheet("border: 1px solid #666666;")
        return label

    @staticmethod
    def _build_color_selector_row(parent: QWidget, preview_label: QLabel, button: QPushButton) -> QWidget:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(preview_label)
        layout.addWidget(button)
        layout.addStretch(1)
        return container

    def _build_output_selector_row(self, parent: QWidget) -> QWidget:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._output_root_label, 1)
        layout.addWidget(self._browse_output_button)
        return container

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("预览", self)
        layout = QVBoxLayout(group)

        self._preview_label = QLabel(group)
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(360)
        self._preview_label.setStyleSheet("background-color: #f3f3f3; border: 1px solid #b8b8b8;")

        hint_label = QLabel(
            "预览显示当前参数下的第一个标记。生成时会按起始 ID 连续输出多个单标记 PNG。",
            group,
        )
        hint_label.setWordWrap(True)

        layout.addWidget(self._preview_label, 1)
        layout.addWidget(hint_label)
        return group

    def _build_status_group(self) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._status_label = QLabel("等待生成。", container)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._status_label, 1)
        return container

    def _connect_signals(self) -> None:
        self._dictionary_name_combo.currentIndexChanged.connect(self._on_dictionary_changed)
        self._quantity_spin.valueChanged.connect(self._on_inputs_changed)
        self._size_input.textChanged.connect(self._on_inputs_changed)
        self._start_id_spin.valueChanged.connect(self._on_inputs_changed)
        self._foreground_color_button.clicked.connect(self._on_select_foreground_color_clicked)
        self._background_color_button.clicked.connect(self._on_select_background_color_clicked)
        self._browse_output_button.clicked.connect(self._on_select_output_dir_clicked)
        self._outline_checkbox.toggled.connect(self._on_inputs_changed)
        self._generate_button.clicked.connect(self._on_generate_button_clicked)

    # endregion

    # region 交互

    @Slot()
    def _on_dictionary_changed(self) -> None:
        self._apply_dictionary_limits()
        self._on_inputs_changed()

    @Slot()
    def _on_inputs_changed(self) -> None:
        self._output_dir_label.setText(self._build_timestamp_dir_hint())
        self._refresh_preview()

    @Slot()
    def _on_select_foreground_color_clicked(self) -> None:
        selected_color = self._select_color(self._foreground_color, "选择前景色")
        if selected_color is None:
            return
        self._foreground_color = selected_color
        self._update_color_preview(self._foreground_color_preview, selected_color)
        self._on_inputs_changed()

    @Slot()
    def _on_select_background_color_clicked(self) -> None:
        selected_color = self._select_color(self._background_color, "选择背景色")
        if selected_color is None:
            return
        self._background_color = selected_color
        self._update_color_preview(self._background_color_preview, selected_color)
        self._on_inputs_changed()

    @Slot()
    def _on_select_output_dir_clicked(self) -> None:
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
        self._on_inputs_changed()

    @Slot()
    def _on_generate_button_clicked(self) -> None:
        try:
            config = self._read_config()
            output_dir = self._create_output_dir(config.output_root)
            written_files = self._generate_markers(config, output_dir)
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            self._status_label.setText(f"参数错误: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "生成失败", str(exc))
            self._status_label.setText(f"生成失败: {exc}")
            return

        self._status_label.setText(f"已生成 {len(written_files)} 个标记，输出目录：{output_dir}")
        QMessageBox.information(
            self,
            "生成完成",
            f"已生成 {len(written_files)} 个标记。\n输出目录：\n{output_dir}",
        )

    # endregion

    # region 预览与配置

    def _refresh_preview(self) -> None:
        try:
            config = self._read_config()
            marker_image = self._build_marker_image(
                dictionary_name=config.dictionary_name,
                marker_id=config.start_id,
                size_px=config.size_px,
                foreground_color=config.foreground_color,
                background_color=config.background_color,
                add_outer_outline=config.add_outer_outline,
            )
        except Exception as exc:  # noqa: BLE001
            self._preview_label.clear()
            self._preview_label.setText(f"预览不可用:\n{exc}")
            return

        pixmap = self._to_pixmap(marker_image)
        scaled_pixmap = pixmap.scaled(
            360,
            360,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled_pixmap)

    def _read_config(self) -> MarkerGenerationConfig:
        size_text = self._size_input.text().strip()
        if not size_text:
            raise ValueError("尺寸不能为空。")

        size_px = int(size_text)
        if size_px <= 0:
            raise ValueError("尺寸必须大于 0。")

        dictionary_name = self._current_dictionary_name()
        dictionary_capacity = self._dictionary_capacity(dictionary_name)
        count = int(self._quantity_spin.value())
        start_id = int(self._start_id_spin.value())
        if start_id + count > dictionary_capacity:
            raise ValueError(
                f"起始 ID {start_id} 加数量 {count} 超出 {dictionary_name} 的容量 {dictionary_capacity}。"
            )

        foreground_color = self._foreground_color
        background_color = self._background_color
        if foreground_color == background_color:
            raise ValueError("前景色和背景色不能相同。")

        return MarkerGenerationConfig(
            dictionary_name=dictionary_name,
            size_px=size_px,
            start_id=start_id,
            count=count,
            foreground_color=foreground_color,
            background_color=background_color,
            add_outer_outline=self._outline_checkbox.isChecked(),
            output_root=self._output_root,
        )

    def _apply_dictionary_limits(self) -> None:
        dictionary_capacity = self._dictionary_capacity(self._current_dictionary_name())
        self._quantity_spin.setRange(1, dictionary_capacity)
        self._start_id_spin.setRange(0, dictionary_capacity - 1)

        current_count = int(self._quantity_spin.value())
        current_start_id = self._start_id_spin.value()
        if current_start_id + current_count > dictionary_capacity:
            max_start_id = max(0, dictionary_capacity - current_count)
            self._start_id_spin.setValue(max_start_id)

    def _build_timestamp_dir_hint(self) -> str:
        return str(self._output_root / datetime.now().strftime("%m%d-%H%M%S"))

    @staticmethod
    def _resolve_download_root() -> Path:
        candidate = Path.home() / "Downloads"
        return candidate if candidate.exists() else Path.home()

    def _current_dictionary_name(self) -> str:
        return str(self._dictionary_name_combo.currentData())

    @staticmethod
    def _dictionary_id(dictionary_name: str) -> int:
        predefined_name = DICTIONARY_FAMILY_TO_PREDEFINED_NAME[dictionary_name]
        return int(getattr(cv2.aruco, predefined_name))

    def _dictionary_capacity(self, dictionary_name: str) -> int:
        dictionary = cv2.aruco.getPredefinedDictionary(self._dictionary_id(dictionary_name))
        return int(dictionary.bytesList.shape[0])

    # endregion

    # region 生成

    def _generate_markers(self, config: MarkerGenerationConfig, output_dir: Path) -> list[Path]:
        written_files: list[Path] = []
        for offset in range(config.count):
            marker_id = config.start_id + offset
            marker_image = self._build_marker_image(
                dictionary_name=config.dictionary_name,
                marker_id=marker_id,
                size_px=config.size_px,
                foreground_color=config.foreground_color,
                background_color=config.background_color,
                add_outer_outline=config.add_outer_outline,
            )
            output_path = output_dir / self._build_output_filename(
                dictionary_name=config.dictionary_name,
                marker_id=marker_id,
                size_px=config.size_px,
                add_outer_outline=config.add_outer_outline,
            )
            marker_bgr = cv2.cvtColor(marker_image, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(output_path), marker_bgr):
                raise RuntimeError(f"写入文件失败: {output_path}")
            written_files.append(output_path)
        return written_files

    def _build_marker_image(
        self,
        dictionary_name: str,
        marker_id: int,
        size_px: int,
        foreground_color: ColorValue,
        background_color: ColorValue,
        add_outer_outline: bool,
    ) -> np.ndarray:
        dictionary = cv2.aruco.getPredefinedDictionary(self._dictionary_id(dictionary_name))
        marker_binary = cv2.aruco.generateImageMarker(
            dictionary,
            marker_id,
            size_px,
            borderBits=1,
        )
        colorized_marker = self._colorize_marker(marker_binary, foreground_color, background_color)
        if add_outer_outline:
            colorized_marker = self._add_outer_outline(
                colorized_marker,
                size_px=size_px,
                marker_size=int(dictionary.markerSize),
                outline_color=background_color,
            )
        return colorized_marker

    @staticmethod
    def _colorize_marker(
        marker_binary: np.ndarray,
        foreground_color: ColorValue,
        background_color: ColorValue,
    ) -> np.ndarray:
        height, width = marker_binary.shape
        marker_rgb = np.empty((height, width, 3), dtype=np.uint8)
        marker_mask = marker_binary == 0
        marker_rgb[:, :] = background_color
        marker_rgb[marker_mask] = foreground_color
        return marker_rgb

    @staticmethod
    def _add_outer_outline(
        marker_image: np.ndarray,
        size_px: int,
        marker_size: int,
        outline_color: ColorValue,
    ) -> np.ndarray:
        total_cells = marker_size + 2
        outline_thickness = max(1, round(size_px / total_cells))
        return cv2.copyMakeBorder(
            marker_image,
            outline_thickness,
            outline_thickness,
            outline_thickness,
            outline_thickness,
            borderType=cv2.BORDER_CONSTANT,
            value=outline_color,
        )

    @staticmethod
    def _build_output_filename(
        dictionary_name: str,
        marker_id: int,
        size_px: int,
        add_outer_outline: bool,
    ) -> str:
        outline_suffix = "_outlined" if add_outer_outline else ""
        dictionary_slug = dictionary_name.lower().replace("dict_", "")
        return f"aruco_{dictionary_slug}_id{marker_id:02d}_{size_px}px{outline_suffix}.png"

    @staticmethod
    def _create_output_dir(output_root: Path) -> Path:
        output_dir = output_root / datetime.now().strftime("%m%d-%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir

    # endregion

    # region 工具

    @staticmethod
    def _select_color(initial_color: ColorValue, title: str) -> ColorValue | None:
        selected_color = QColorDialog.getColor(QColor(*initial_color), None, title)
        if not selected_color.isValid():
            return None
        return selected_color.red(), selected_color.green(), selected_color.blue()

    @staticmethod
    def _update_color_preview(preview_label: QLabel, color: ColorValue) -> None:
        red_value, green_value, blue_value = color
        preview_label.setStyleSheet(
            f"background-color: rgb({red_value}, {green_value}, {blue_value}); border: 1px solid #666666;"
        )
        preview_label.setToolTip(f"RGB({red_value}, {green_value}, {blue_value})")

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        contiguous_image = np.ascontiguousarray(image)
        height, width, _ = contiguous_image.shape
        qimage = QImage(
            contiguous_image.data,
            width,
            height,
            int(contiguous_image.strides[0]),
            QImage.Format.Format_RGB888,
        )
        return QPixmap.fromImage(qimage.copy())

    # endregion


def main() -> int:
    app = QApplication(sys.argv)
    window = ArucoMarkerGeneratorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
