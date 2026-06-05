from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.util_components.casia_indicator_light import CasiaIndicatorLight
from src.wuji import (
    DEFAULT_WUJI_CAMERA,
    SUPPORTED_WUJI_CAMERAS,
    WujiCameraFrame,
    WujiCameraIntrinsicsInfo,
)


class ImagePreviewLabel(QLabel):
    """保持比例显示相机图像的 QLabel。"""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setText(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("border: 1px solid rgb(85, 85, 85); background: rgb(30, 30, 30);")

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._refresh_scaled_pixmap()

    def clear_preview(self, text: str) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def _refresh_scaled_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class WujiCameraTabWidget(QWidget):
    """无际 qmlinker 相机测试页。"""

    cameraSelected = Signal(str)
    cameraEnableToggleRequested = Signal(str, bool)
    rgbStreamRequested = Signal(str)
    rgbdStreamRequested = Signal(str)
    streamStopRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_camera_name = DEFAULT_WUJI_CAMERA
        self._activated_once = False

        self.camera_combo = QComboBox(self)
        for spec in SUPPORTED_WUJI_CAMERAS:
            self.camera_combo.addItem(spec.title, spec.name)

        self.enable_indicator = CasiaIndicatorLight(
            self,
            text=("使能", "禁用"),
            font_size=12,
            default_status=False,
        )
        self.rgb_preview = ImagePreviewLabel("RGB 图像", self)
        self.depth_preview = ImagePreviewLabel("Depth 图像", self)
        self.rgb_button = QPushButton("获取 RGB 流", self)
        self.rgbd_button = QPushButton("获取 RGBD 流", self)
        self.intrinsics_label = QLabel("内参: -", self)
        self.resolution_label = QLabel("分辨率: -", self)

        self._build_layout()
        self._connect_signals()
        self.set_current_camera(DEFAULT_WUJI_CAMERA)

    def activate_default_camera(self) -> None:
        """首次打开 tab 时切换到默认头部相机并请求刷新。"""

        if not self._activated_once:
            self._activated_once = True
            self.set_current_camera(DEFAULT_WUJI_CAMERA)
        self.cameraSelected.emit(self._current_camera_name)

    def set_current_camera(self, camera_name: str) -> None:
        """更新下拉框当前相机。"""

        for index in range(self.camera_combo.count()):
            if self.camera_combo.itemData(index) == camera_name:
                self.camera_combo.setCurrentIndex(index)
                self._current_camera_name = camera_name
                return

    def update_camera_enable_state(self, camera_name: str, enabled: bool) -> None:
        """刷新当前相机使能状态指示灯。"""

        if camera_name != self._current_camera_name:
            return
        self.enable_indicator.set_status(enabled)

    def update_intrinsics(self, intrinsics: WujiCameraIntrinsicsInfo) -> None:
        """刷新当前相机内参与分辨率文本。"""

        if intrinsics.camera_name != self._current_camera_name:
            return
        self.intrinsics_label.setText(
            "内参: "
            f"fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
            f"cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}, "
            f"dist={self._format_distortion(intrinsics.distortion)}"
        )
        self.resolution_label.setText(f"分辨率: {intrinsics.width} x {intrinsics.height}")

    def update_frame(self, frame: WujiCameraFrame) -> None:
        """刷新 RGB 与可选深度图预览。"""

        if frame.camera_name != self._current_camera_name:
            return
        self.rgb_preview.set_preview_pixmap(_bgr_to_pixmap(frame.color_bgr))
        if frame.depth is not None:
            self.depth_preview.set_preview_pixmap(_depth_to_pixmap(frame.depth))

    def clear_images(self) -> None:
        """清空当前图像预览。"""

        self.rgb_preview.clear_preview("RGB 图像")
        self.depth_preview.clear_preview("Depth 图像")

    def _build_layout(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("相机:", self))
        top_layout.addWidget(self.camera_combo)
        top_layout.addWidget(QLabel("状态:", self))
        top_layout.addWidget(self.enable_indicator)
        top_layout.addStretch(1)
        main_layout.addLayout(top_layout)

        preview_layout = QGridLayout()
        preview_layout.addWidget(self._build_preview_group("RGB", self.rgb_preview, self.rgb_button), 0, 0)
        preview_layout.addWidget(self._build_preview_group("RGBD", self.depth_preview, self.rgbd_button), 0, 1)
        main_layout.addLayout(preview_layout, 1)

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.intrinsics_label)
        bottom_layout.addWidget(self.resolution_label)
        main_layout.addLayout(bottom_layout)

    def _build_preview_group(self, title: str, preview: ImagePreviewLabel, button: QPushButton) -> QGroupBox:
        group = QGroupBox(title, self)
        layout = QVBoxLayout(group)
        layout.addWidget(preview, 1)
        layout.addWidget(button)
        return group

    def _connect_signals(self) -> None:
        self.camera_combo.currentIndexChanged.connect(self._on_camera_combo_current_index_changed)
        self.enable_indicator.clicked.connect(self._on_enable_indicator_clicked)
        self.rgb_button.clicked.connect(self._on_rgb_button_clicked)
        self.rgbd_button.clicked.connect(self._on_rgbd_button_clicked)

    @Slot(int)
    def _on_camera_combo_current_index_changed(self, index: int) -> None:
        camera_name = str(self.camera_combo.itemData(index))
        if not camera_name:
            return
        self._current_camera_name = camera_name
        self.clear_images()
        self.streamStopRequested.emit()
        self.cameraSelected.emit(camera_name)

    @Slot()
    def _on_enable_indicator_clicked(self) -> None:
        requested_status = not bool(self.enable_indicator.property("status"))
        self.cameraEnableToggleRequested.emit(self._current_camera_name, requested_status)

    @Slot()
    def _on_rgb_button_clicked(self) -> None:
        self.depth_preview.clear_preview("Depth 图像")
        self.rgbStreamRequested.emit(self._current_camera_name)

    @Slot()
    def _on_rgbd_button_clicked(self) -> None:
        self.rgbdStreamRequested.emit(self._current_camera_name)

    def _format_distortion(self, values: tuple[float, ...]) -> str:
        if not values:
            return "[]"
        return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    """将 BGR 图像转换为 QPixmap。"""

    bgr = np.ascontiguousarray(image_bgr)
    height, width = bgr.shape[:2]
    qimage = QImage(
        bgr.data,
        width,
        height,
        int(bgr.strides[0]),
        QImage.Format.Format_BGR888,
    ).copy()
    return QPixmap.fromImage(qimage)


def _depth_to_pixmap(depth: np.ndarray) -> QPixmap:
    """将深度矩阵归一化为灰度 QPixmap。"""

    depth_array = np.asarray(depth)
    valid = np.isfinite(depth_array) & (depth_array > 0)
    gray = np.zeros(depth_array.shape[:2], dtype=np.uint8)
    if np.any(valid):
        valid_values = depth_array[valid].astype(np.float32, copy=False)
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        if max_value > min_value:
            gray[valid] = np.clip(
                (valid_values - min_value) * 255.0 / (max_value - min_value),
                0.0,
                255.0,
            ).astype(np.uint8)
    gray = np.ascontiguousarray(gray)
    height, width = gray.shape[:2]
    qimage = QImage(
        gray.data,
        width,
        height,
        int(gray.strides[0]),
        QImage.Format.Format_Grayscale8,
    ).copy()
    return QPixmap.fromImage(qimage)
