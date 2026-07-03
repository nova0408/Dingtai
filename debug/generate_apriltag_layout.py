from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import sys

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QDoubleValidator, QIntValidator, QPixmap
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_OUTPUT_ROOT = Path.home() / "Downloads"
DEFAULT_DICTIONARY_NAME = "DICT_APRILTAG_16H5"
DEFAULT_SQUARES_X = 5
DEFAULT_SQUARES_Y = 7
DEFAULT_SQUARE_LENGTH_PX = 240
DEFAULT_MARKER_LENGTH_RATIO = 0.7
DEFAULT_BACKGROUND_COLOR = (245, 245, 245)

DICTIONARY_NAME_TO_ID: dict[str, int] = {
    "DICT_APRILTAG_16H5": int(cv2.aruco.DICT_APRILTAG_16h5),
    "DICT_APRILTAG_25H9": int(cv2.aruco.DICT_APRILTAG_25h9),
    "DICT_APRILTAG_36H10": int(cv2.aruco.DICT_APRILTAG_36h10),
    "DICT_APRILTAG_36H11": int(cv2.aruco.DICT_APRILTAG_36h11),
}


@dataclass(frozen=True)
class CharucoBoardConfig:
    dictionary_name: str
    squares_x: int
    squares_y: int
    square_length_px: int
    marker_length_ratio: float
    output_root: Path


class CharucoBoardBuilderWindow(QMainWindow):
    # region 初始化

    def __init__(self) -> None:
        super().__init__()
        self._output_root = DEFAULT_OUTPUT_ROOT
        self._preview_image: np.ndarray | None = None

        self._dictionary_input: QLineEdit
        self._squares_x_input: QLineEdit
        self._squares_y_input: QLineEdit
        self._square_length_input: QLineEdit
        self._marker_ratio_input: QLineEdit
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
        self.setWindowTitle("ChArUco Board Builder")
        self.resize(1100, 840)

    def _setup_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.addWidget(self._build_config_group())
        root_layout.addWidget(self._build_preview_group(), 1)
        root_layout.addWidget(self._build_status_group())
        self.setCentralWidget(central)

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("ChArUco 参数", self)
        layout = QFormLayout(group)

        self._dictionary_input = QLineEdit(DEFAULT_DICTIONARY_NAME, group)
        self._squares_x_input = QLineEdit(str(DEFAULT_SQUARES_X), group)
        self._squares_x_input.setValidator(QIntValidator(2, 200, self._squares_x_input))
        self._squares_y_input = QLineEdit(str(DEFAULT_SQUARES_Y), group)
        self._squares_y_input.setValidator(QIntValidator(2, 200, self._squares_y_input))

        self._square_length_input = QLineEdit(str(DEFAULT_SQUARE_LENGTH_PX), group)
        self._square_length_input.setValidator(QIntValidator(10, 10000, self._square_length_input))

        self._marker_ratio_input = QLineEdit(str(DEFAULT_MARKER_LENGTH_RATIO), group)
        self._marker_ratio_input.setValidator(QDoubleValidator(0.05, 0.95, 3, self._marker_ratio_input))

        self._output_root_label = QLabel(str(self._output_root), group)
        self._output_root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._browse_output_button = QPushButton("选择输出目录", group)
        self._save_button = QPushButton("保存板图", group)

        layout.addRow("字典", self._dictionary_input)
        layout.addRow("棋盘横向方块数", self._squares_x_input)
        layout.addRow("棋盘纵向方块数", self._squares_y_input)
        layout.addRow("单方块边长 (px)", self._square_length_input)
        layout.addRow("marker 边长比例", self._marker_ratio_input)
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
        self._json_preview.setPlaceholderText("板定义 JSON 预览")

        layout.addWidget(scroll, 2)
        layout.addWidget(self._json_preview, 1)
        return group

    def _build_status_group(self) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        self._status_label = QLabel("等待编辑 ChArUco 参数。", container)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._status_label, 1)
        return container

    def _connect_signals(self) -> None:
        self._dictionary_input.textChanged.connect(self._on_inputs_changed)
        self._squares_x_input.textChanged.connect(self._on_inputs_changed)
        self._squares_y_input.textChanged.connect(self._on_inputs_changed)
        self._square_length_input.textChanged.connect(self._on_inputs_changed)
        self._marker_ratio_input.textChanged.connect(self._on_inputs_changed)
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
            payload = self._build_board_payload(config)
            output_dir = self._create_output_dir(config.output_root)
            image_path = output_dir / "charuco_board.png"
            json_path = output_dir / "charuco_board.json"
            pnp_json_path = output_dir / "charuco_board_pnp.json"
            if self._preview_image is None:
                raise RuntimeError("预览图为空，无法保存。")
            if not cv2.imwrite(str(image_path), cv2.cvtColor(self._preview_image, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"保存图片失败：{image_path}")
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            pnp_json_path.write_text(
                json.dumps(self._build_pnp_payload(config), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))
            self._status_label.setText(f"保存失败: {exc}")
            return

        self._status_label.setText(f"已保存板图到：{output_dir}")
        QMessageBox.information(
            self,
            "保存完成",
            f"ChArUco 板图片与 JSON 已保存。\n目录：\n{output_dir}",
        )

    # endregion

    # region 预览与构造

    def _refresh_preview(self) -> None:
        try:
            config = self._read_config()
            payload = self._build_board_payload(config)
            preview_image = self._render_board(config)
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
            f"已生成预览：{config.squares_x} x {config.squares_y} 方块。"
        )

    def _read_config(self) -> CharucoBoardConfig:
        dictionary_name = self._dictionary_input.text().strip()
        if dictionary_name not in DICTIONARY_NAME_TO_ID:
            raise ValueError(f"不支持的字典名：{dictionary_name}")

        squares_x = int(self._squares_x_input.text().strip() or "0")
        squares_y = int(self._squares_y_input.text().strip() or "0")
        if squares_x < 2 or squares_y < 2:
            raise ValueError("棋盘方块数必须都大于等于 2。")

        square_length_px = int(self._square_length_input.text().strip() or "0")
        if square_length_px <= 0:
            raise ValueError("单方块边长必须大于 0。")

        marker_ratio = float(self._marker_ratio_input.text().strip() or "0")
        if not 0.05 < marker_ratio < 0.95:
            raise ValueError("marker 边长比例必须在 0.05 到 0.95 之间。")

        return CharucoBoardConfig(
            dictionary_name=dictionary_name,
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_px=square_length_px,
            marker_length_ratio=marker_ratio,
            output_root=self._output_root,
        )

    def _build_board_payload(self, config: CharucoBoardConfig) -> dict[str, object]:
        marker_length_px = config.square_length_px * config.marker_length_ratio
        board = self._build_board(config)
        board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
        return {
            "kind": "charuco_board",
            "dictionary_name": config.dictionary_name,
            "squares_x": config.squares_x,
            "squares_y": config.squares_y,
            "square_length_px": config.square_length_px,
            "marker_length_px": marker_length_px,
            "marker_length_ratio": config.marker_length_ratio,
            "canvas_size_px": [
                config.squares_x * config.square_length_px,
                config.squares_y * config.square_length_px,
            ],
            "chessboard_corners": board_corners.tolist(),
        }

    def _build_pnp_payload(self, config: CharucoBoardConfig) -> dict[str, object]:
        board = self._build_board(config)
        board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)
        charuco_ids = np.arange(board_corners.shape[0], dtype=np.int32).reshape(-1, 1)
        return {
            "dictionary_name": config.dictionary_name,
            "squares_x": config.squares_x,
            "squares_y": config.squares_y,
            "square_length_px": config.square_length_px,
            "marker_length_px": config.square_length_px * config.marker_length_ratio,
            "charuco_ids": charuco_ids.reshape(-1).tolist(),
            "object_points_mm": board_corners.tolist(),
            "object_points_by_id": {
                str(int(charuco_id)): point.tolist()
                for charuco_id, point in zip(charuco_ids.reshape(-1), board_corners, strict=True)
            },
        }

    def _render_board(self, config: CharucoBoardConfig) -> np.ndarray:
        board = self._build_board(config)
        width = config.squares_x * config.square_length_px
        height = config.squares_y * config.square_length_px
        image = board.generateImage((width, height))
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._apply_preview_background(image)

    @staticmethod
    def _build_board(config: CharucoBoardConfig) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_NAME_TO_ID[config.dictionary_name])
        return cv2.aruco.CharucoBoard(
            (config.squares_x, config.squares_y),
            float(config.square_length_px),
            float(config.square_length_px * config.marker_length_ratio),
            dictionary,
        )

    @staticmethod
    def _apply_preview_background(image_rgb: np.ndarray) -> np.ndarray:
        canvas = np.full((*image_rgb.shape[:2], 3), DEFAULT_BACKGROUND_COLOR, dtype=np.uint8)
        mask = np.any(image_rgb < 250, axis=2)
        canvas[mask] = image_rgb[mask]
        return canvas

    # endregion

    # region 工具

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
    window = CharucoBoardBuilderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
