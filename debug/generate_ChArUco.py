from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt, Slot
from PySide6.QtGui import (
    QImage,
    QIntValidator,
    QMouseEvent,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
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
DEFAULT_SQUARE_LENGTH = 240.0
DEFAULT_MARKER_LENGTH_RATIO = 0.7
DEFAULT_PREVIEW_LONG_SIDE_PX = 600
DEFAULT_MARGIN_SIZE = 240.0
PREVIEW_BACKGROUND_COLOR = (195, 220, 255)
EXPORT_DPI = 600
MIN_CHARUCO_SQUARES_X = 3
MIN_CHARUCO_SQUARES_Y = 3

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
    square_length: float
    margin_size: float
    output_root: Path


class ZoomablePreviewLabel(QLabel):
    """支持滚轮缩放和鼠标拖动平移的预览画布。"""

    def __init__(self, scroll_area: QScrollArea, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scroll_area = scroll_area
        self._base_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._dragging = False
        self._last_drag_pos = QPoint()

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background: rgb({PREVIEW_BACKGROUND_COLOR[0]}, {PREVIEW_BACKGROUND_COLOR[1]}, {PREVIEW_BACKGROUND_COLOR[2]});"
            " border: 1px solid #b8b8b8;"
        )
        self.setMinimumSize(800, 520)
        self.setMouseTracking(True)

    def set_base_pixmap(self, pixmap: QPixmap) -> None:
        self._base_pixmap = pixmap
        self._zoom_factor = self._fit_zoom_factor()
        self._update_pixmap()

    def reset_view(self) -> None:
        self._zoom_factor = self._fit_zoom_factor()
        self._update_pixmap()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._base_pixmap.isNull():
            return super().wheelEvent(event)

        delta = event.angleDelta().y()
        if delta == 0:
            return

        scale_step = 1.15 if delta > 0 else 1 / 1.15
        old_factor = self._zoom_factor
        self._zoom_factor = max(0.1, min(10.0, self._zoom_factor * scale_step))
        if self._zoom_factor != old_factor:
            self._update_pixmap()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and not self.pixmap().isNull():
            self._dragging = True
            self._last_drag_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._dragging:
            delta = event.position().toPoint() - self._last_drag_pos
            self._last_drag_pos = event.position().toPoint()
            self._scroll_area.horizontalScrollBar().setValue(
                self._scroll_area.horizontalScrollBar().value() - delta.x()
            )
            self._scroll_area.verticalScrollBar().setValue(self._scroll_area.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _fit_zoom_factor(self) -> float:
        if self._base_pixmap.isNull():
            return 1.0
        viewport = self._scroll_area.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return 1.0
        width_ratio = viewport.width() / self._base_pixmap.width()
        height_ratio = viewport.height() / self._base_pixmap.height()
        return max(0.1, min(width_ratio, height_ratio))

    def _update_pixmap(self) -> None:
        if self._base_pixmap.isNull():
            self.clear()
            return
        scaled = self._base_pixmap.scaled(
            max(1, int(self._base_pixmap.width() * self._zoom_factor)),
            max(1, int(self._base_pixmap.height() * self._zoom_factor)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(scaled)
        self.resize(scaled.size())


class CharucoBoardBuilderWindow(QMainWindow):
    # region 初始化

    def __init__(self) -> None:
        super().__init__()
        self._output_root = DEFAULT_OUTPUT_ROOT
        self._preview_image: np.ndarray | None = None
        self._preview_error_message: str | None = None

        self._dictionary_input: QLineEdit
        self._squares_x_input: QLineEdit
        self._squares_y_input: QLineEdit
        self._square_length_input: QLineEdit
        self._margin_size_input: QLineEdit
        self._output_root_label: QLabel
        self._browse_output_button: QPushButton
        self._preview_label: ZoomablePreviewLabel
        self._status_label: QLabel
        self._save_button: QPushButton
        self._json_preview: QTextEdit

        self._setup_window()
        self._setup_ui()
        self._connect_signals()
        self._refresh_preview()

    def _setup_window(self) -> None:
        self.setWindowTitle("generate_ChArUco")
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

        self._square_length_input = QLineEdit(str(DEFAULT_SQUARE_LENGTH), group)
        self._square_length_input.setValidator(QIntValidator(10, 10000, self._square_length_input))
        self._margin_size_input = QLineEdit(str(DEFAULT_MARGIN_SIZE), group)
        self._margin_size_input.setValidator(QIntValidator(0, 10000, self._margin_size_input))

        self._output_root_label = QLabel(str(self._output_root), group)
        self._output_root_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._browse_output_button = QPushButton("选择输出目录", group)
        self._save_button = QPushButton("保存板图", group)

        layout.addRow("字典", self._dictionary_input)
        layout.addRow("棋盘横向方块数", self._squares_x_input)
        layout.addRow("棋盘纵向方块数", self._squares_y_input)
        layout.addRow("单方块边长", self._square_length_input)
        layout.addRow("marginSize", self._margin_size_input)
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

        self._preview_label = ZoomablePreviewLabel(scroll, scroll)
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
        self._margin_size_input.textChanged.connect(self._on_inputs_changed)
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
            self._validate_board_compatibility(config)
            payload = self._build_board_payload(config)
            output_dir = self._create_output_dir(config.output_root)
            image_path = output_dir / "charuco_board.png"
            json_path = output_dir / "charuco_board.json"
            export_image = self._render_export_board(config)
            self._save_png_with_dpi(image_path, export_image)
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except ValueError as exc:
            self._status_label.setText(f'<span style="color:#d00000;">保存失败：{exc}</span>')
            return
        except Exception as exc:  # noqa: BLE001
            self._status_label.setText(f'<span style="color:#d00000;">保存失败：{exc}</span>')
            return

        self._status_label.setText(f"已保存板图到：{output_dir}")

    # endregion

    # region 预览与构造

    def _refresh_preview(self) -> None:
        try:
            config = self._read_config()
            self._validate_board_compatibility(config)
            payload = self._build_board_payload(config)
            preview_image = self._render_preview_board(config)
        except Exception as exc:  # noqa: BLE001
            self._preview_image = None
            self._preview_error_message = str(exc)
            self._preview_label.clear()
            self._json_preview.setPlainText(str(exc))
            self._status_label.setText(f'<span style="color:#d00000;">预览不可用：{exc}</span>')
            return

        self._preview_image = preview_image
        self._preview_error_message = None
        pixmap = self._to_pixmap(preview_image)
        self._preview_label.set_base_pixmap(pixmap)
        self._json_preview.setPlainText(json.dumps(payload, ensure_ascii=False, indent=2))
        self._status_label.setText(f"已生成预览：{config.squares_x} x {config.squares_y} 方块。")

    def _read_config(self) -> CharucoBoardConfig:
        dictionary_name = self._dictionary_input.text().strip()
        if dictionary_name not in DICTIONARY_NAME_TO_ID:
            raise ValueError(f"不支持的字典名：{dictionary_name}")

        squares_x = int(self._squares_x_input.text().strip() or "0")
        squares_y = int(self._squares_y_input.text().strip() or "0")
        if squares_x < 2 or squares_y < 2:
            raise ValueError("棋盘方块数必须都大于等于 2。")

        square_length = float(self._square_length_input.text().strip() or "0")
        if square_length <= 0:
            raise ValueError("单方块边长必须大于 0。")

        margin_size = float(self._margin_size_input.text().strip() or "0")
        if margin_size < 0:
            raise ValueError("marginSize 不能小于 0。")

        return CharucoBoardConfig(
            dictionary_name=dictionary_name,
            squares_x=squares_x,
            squares_y=squares_y,
            square_length=square_length,
            margin_size=margin_size,
            output_root=self._output_root,
        )

    def _validate_board_compatibility(self, config: CharucoBoardConfig) -> None:
        """校验板参数是否与 `cv2.aruco.CharucoBoard` 的可用范围一致。"""
        if config.squares_x < MIN_CHARUCO_SQUARES_X or config.squares_y < MIN_CHARUCO_SQUARES_Y:
            raise ValueError("ChArUco 板至少需要 3x3 方块；2x2 这类板型不支持创建或预览。")
        if config.dictionary_name not in DICTIONARY_NAME_TO_ID:
            raise ValueError(f"不支持的字典名：{config.dictionary_name}")

    def _build_board_payload(self, config: CharucoBoardConfig) -> dict[str, object]:
        return {
            "dictionary_name": config.dictionary_name,
            "squares_x": config.squares_x,
            "squares_y": config.squares_y,
            "square_length": config.square_length,
            "margin_size": config.margin_size,
            "marker_length": config.square_length * DEFAULT_MARKER_LENGTH_RATIO,
        }

    def _render_preview_board(self, config: CharucoBoardConfig) -> np.ndarray:
        board_gray = self._render_export_board(config)
        height, width = board_gray.shape
        preview = np.zeros((height, width, 4), dtype=np.uint8)
        preview[:, :, :3] = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2RGB)
        preview[:, :, 3] = 255
        return preview

    def _render_export_board(self, config: CharucoBoardConfig) -> np.ndarray:
        board = self._build_board(config)
        square_px = self._preview_square_pixels(config)
        margin_px = max(0, int(round(config.margin_size)))
        image_width = (config.squares_x + 2) * square_px
        image_height = (config.squares_y + 2) * square_px
        board_image = board.generateImage((image_width, image_height), marginSize=margin_px, borderBits=1)
        if board_image.ndim == 3:
            board_image = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
        return np.where(board_image < 128, 0, 255).astype(np.uint8)

    def _preview_square_pixels(self, config: CharucoBoardConfig) -> int:
        board_long_side = max(config.squares_x, config.squares_y)
        return max(1, int(round(DEFAULT_PREVIEW_LONG_SIDE_PX / board_long_side)))

    @staticmethod
    def _build_board(config: CharucoBoardConfig) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_NAME_TO_ID[config.dictionary_name])
        return cv2.aruco.CharucoBoard(
            (config.squares_x, config.squares_y),
            float(config.square_length),
            float(config.square_length * DEFAULT_MARKER_LENGTH_RATIO),
            dictionary,
        )

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
        height, width, channels = contiguous.shape
        if channels != 4:
            raise ValueError(f"预览图应为 RGBA 四通道，实际为 {contiguous.shape}")
        qimage = QImage(
            contiguous.data,
            width,
            height,
            int(contiguous.strides[0]),
            QImage.Format.Format_RGBA8888,
        )
        return QPixmap.fromImage(qimage.copy())

    @staticmethod
    def _save_png_with_dpi(image_path: Path, image_gray: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(image_gray)
        if contiguous.ndim != 2:
            raise ValueError(f"导出图像应为灰度二值图，实际为 {contiguous.shape}")
        pil_image = Image.fromarray(contiguous, mode="L")
        pil_image.save(image_path, dpi=(EXPORT_DPI, EXPORT_DPI))

    # endregion


def main() -> int:
    app = QApplication(sys.argv)
    window = CharucoBoardBuilderWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
