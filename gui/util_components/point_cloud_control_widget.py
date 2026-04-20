from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

import numpy as np
from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.utils.Datas import Transform

from .casia_tree import PointCloudNode
from .point_cloud_info_widget import PointCloudInfoWidget


class PointCloudColorizeMode(Enum):
    """点云着色模式。"""

    SOLID = auto()
    RANDOM = auto()
    CYCLE = auto()
    HEIGHT = auto()


@dataclass(slots=True)
class PointCloudDisplayStyle:
    """点云显示样式。

    Parameters
    ----------
    colorize_mode : PointCloudColorizeMode
        着色模式。
    solid_color : tuple[float, float, float]
        单色模式使用的 RGB 颜色，范围 [0, 1]。
    """

    colorize_mode: PointCloudColorizeMode = PointCloudColorizeMode.SOLID
    solid_color: tuple[float, float, float] = (0.95, 0.35, 0.35)

    def normalized(self) -> PointCloudDisplayStyle:
        """返回归一化后的样式副本。

        Returns
        -------
        PointCloudDisplayStyle
            颜色范围被裁剪后的样式对象。
        """
        r, g, b = self.solid_color
        color = (
            min(max(float(r), 0.0), 1.0),
            min(max(float(g), 0.0), 1.0),
            min(max(float(b), 0.0), 1.0),
        )
        return PointCloudDisplayStyle(
            colorize_mode=self.colorize_mode,
            solid_color=color,
        )


class PointCloudControlWidget(QWidget):
    """点云控制组合控件。

    Notes
    -----
    该控件封装两部分能力：

    1. 位姿编辑
       通过 `PointCloudInfoWidget` 编辑 Transform。
    2. 显示样式编辑
       支持单色 / 随机 / 周期 / 高度四种着色模式。

    对外可直接作为“点云控制面板”即插即用，主要接口包括：

    - `set_node`
    - `set_transform`
    - `current_transform`
    - `set_display_style`
    - `current_display_style`

    并发出以下信号：

    - `transform_changed`
    - `display_style_changed`
    - `value_changed`
    """

    transform_changed = Signal(object)
    display_style_changed = Signal(object)
    value_changed = Signal(object, object)

    _MODE_TEXT_SOLID: Final[str] = "单色"
    _MODE_TEXT_RANDOM: Final[str] = "随机"
    _MODE_TEXT_CYCLE: Final[str] = "周期"
    _MODE_TEXT_HEIGHT: Final[str] = "高度"

    def __init__(
        self,
        display_name: str = "",
        transform: Transform | np.ndarray | None = None,
        display_style: PointCloudDisplayStyle | None = None,
        node: PointCloudNode | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """初始化控件。

        Parameters
        ----------
        display_name : str, optional
            显示名称。
        transform : Transform | np.ndarray | None, optional
            初始变换。
        display_style : PointCloudDisplayStyle | None, optional
            初始显示样式。
        node : PointCloudNode | None, optional
            初始节点。若提供，则优先从 node 加载。
        parent : QWidget | None, optional
            父控件。
        """
        super().__init__(parent)

        self._display_name = display_name
        self._style = (display_style or PointCloudDisplayStyle()).normalized()
        self._node: PointCloudNode | None = None

        self.info_widget = PointCloudInfoWidget(self)

        self.mode_label = QLabel("着色方式", self)
        self.mode_combo = QComboBox(self)
        self.color_button = QPushButton("选择颜色", self)
        self.color_preview = QFrame(self)

        self._init_ui()
        self._init_mode_combo()
        self._connect_signals()

        if node is not None:
            self.set_node(node)
        else:
            if display_name:
                self.set_display_name(display_name)
            if transform is not None:
                self.set_transform(transform)

        self.set_display_style(self._style)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        """初始化 UI。"""
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.color_preview.setFixedSize(36, 20)
        self.color_preview.setFrameShape(QFrame.Shape.StyledPanel)
        self.color_preview.setFrameShadow(QFrame.Shadow.Sunken)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)
        main_layout.addWidget(self.info_widget)

        style_row = QHBoxLayout()
        style_row.setContentsMargins(0, 0, 0, 0)
        style_row.setSpacing(6)
        style_row.addWidget(self.mode_label)
        style_row.addWidget(self.mode_combo, 1)
        style_row.addWidget(self.color_button)
        style_row.addWidget(self.color_preview)
        main_layout.addLayout(style_row)

    def _init_mode_combo(self) -> None:
        """初始化模式下拉框。"""
        self.mode_combo.addItem(self._MODE_TEXT_SOLID, PointCloudColorizeMode.SOLID)
        self.mode_combo.addItem(self._MODE_TEXT_RANDOM, PointCloudColorizeMode.RANDOM)
        self.mode_combo.addItem(self._MODE_TEXT_CYCLE, PointCloudColorizeMode.CYCLE)
        self.mode_combo.addItem(self._MODE_TEXT_HEIGHT, PointCloudColorizeMode.HEIGHT)

    def _connect_signals(self) -> None:
        """连接信号。"""
        self.info_widget.pc_data_changed.connect(self._on_info_transform_changed)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.color_button.clicked.connect(self._on_pick_color)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def set_display_name(self, display_name: str) -> None:
        """设置显示名称。

        Parameters
        ----------
        display_name : str
            显示名称。
        """
        self._display_name = display_name
        if hasattr(self.info_widget, "pc_index_label"):
            self.info_widget.pc_index_label.setText(display_name)

    def set_node(self, node: PointCloudNode) -> None:
        """从节点加载控件状态。

        Parameters
        ----------
        node : PointCloudNode
            节点对象。
        """
        self._node = node
        self._display_name = node.display_name
        self.info_widget.show_node_info(node)

    def current_transform(self) -> Transform:
        """返回当前 Transform。

        Returns
        -------
        Transform
            当前控件中的 Transform。
        """
        if hasattr(self.info_widget, "current_transform"):
            method = getattr(self.info_widget, "current_transform")
            if callable(method):
                result = method()
                if isinstance(result, Transform):
                    return result

        x = float(self.info_widget.pc_x_spin.value())
        y = float(self.info_widget.pc_y_spin.value())
        z = float(self.info_widget.pc_z_spin.value())
        rz = float(self.info_widget.pc_rz_spin.value())
        ry = float(self.info_widget.pc_ry_spin.value())
        rx = float(self.info_widget.pc_rx_spin.value())
        return Transform.from_list([x, y, z, rz, ry, rx])

    def set_transform(self, transform: Transform | np.ndarray) -> None:
        """设置当前 Transform。

        Parameters
        ----------
        transform : Transform | np.ndarray
            目标变换。
        """
        t = self._coerce_transform(transform)
        matrix = t.as_SE3()

        temp_node = PointCloudNode(
            display_name=self._display_name or "",
            index=-1,
            transform=matrix,
            is_branch=False,
        )
        self.info_widget.show_node_info(temp_node)

    def current_display_style(self) -> PointCloudDisplayStyle:
        """返回当前显示样式。

        Returns
        -------
        PointCloudDisplayStyle
            当前显示样式。
        """
        return self._style.normalized()

    def set_display_style(self, style: PointCloudDisplayStyle) -> None:
        """设置显示样式。

        Parameters
        ----------
        style : PointCloudDisplayStyle
            目标样式。
        """
        self._style = style.normalized()
        self._sync_style_widgets()

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------
    def _on_info_transform_changed(self, transform: object) -> None:
        """响应 Transform 变化。

        Parameters
        ----------
        transform : object
            内部信号发出的 Transform。
        """
        self.transform_changed.emit(transform)
        self.value_changed.emit(transform, self.current_display_style())

    def _on_mode_changed(self) -> None:
        """响应着色模式变化。"""
        data = self.mode_combo.currentData()
        if not isinstance(data, PointCloudColorizeMode):
            raise TypeError(
                f"QComboBox 当前数据类型错误，预期 PointCloudColorizeMode，实际为 {type(data).__name__}",
            )

        self._style = PointCloudDisplayStyle(
            colorize_mode=data,
            solid_color=self._style.solid_color,
        ).normalized()

        self._sync_style_widgets(keep_combo=True)
        self.display_style_changed.emit(self.current_display_style())
        self.value_changed.emit(self.current_transform(), self.current_display_style())

    def _on_pick_color(self) -> None:
        """选择单色模式颜色。"""
        r, g, b = self._style.solid_color
        initial = QColor.fromRgbF(r, g, b)

        qcolor = QColorDialog.getColor(initial, self, "选择点云颜色")
        if not qcolor.isValid():
            return

        self._style = PointCloudDisplayStyle(
            colorize_mode=self._style.colorize_mode,
            solid_color=(qcolor.redF(), qcolor.greenF(), qcolor.blueF()),
        ).normalized()

        self._sync_style_widgets(keep_combo=True)
        self.display_style_changed.emit(self.current_display_style())
        self.value_changed.emit(self.current_transform(), self.current_display_style())

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _sync_style_widgets(self, keep_combo: bool = False) -> None:
        """同步样式到 UI。

        Parameters
        ----------
        keep_combo : bool, optional
            是否保留下拉框当前状态。
        """
        style = self._style.normalized()

        if not keep_combo:
            blocker = QSignalBlocker(self.mode_combo)
            self.mode_combo.setCurrentIndex(self._mode_to_index(style.colorize_mode))
            del blocker

        is_solid = style.colorize_mode is PointCloudColorizeMode.SOLID
        self.color_button.setEnabled(is_solid)
        self.color_preview.setEnabled(is_solid)
        self._update_color_preview(style.solid_color)

    def _mode_to_index(self, mode: PointCloudColorizeMode) -> int:
        """将枚举模式映射为下拉索引。

        Parameters
        ----------
        mode : PointCloudColorizeMode
            着色模式。

        Returns
        -------
        int
            下拉框索引。
        """
        if mode is PointCloudColorizeMode.SOLID:
            return 0
        if mode is PointCloudColorizeMode.RANDOM:
            return 1
        if mode is PointCloudColorizeMode.CYCLE:
            return 2
        if mode is PointCloudColorizeMode.HEIGHT:
            return 3
        raise ValueError(f"未知着色模式：{mode}")

    def _update_color_preview(self, color: tuple[float, float, float]) -> None:
        """更新颜色预览。

        Parameters
        ----------
        color : tuple[float, float, float]
            RGB 颜色，范围 [0, 1]。
        """
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        self.color_preview.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid rgb(90, 90, 90);")

    def _coerce_transform(self, value: Transform | np.ndarray) -> Transform:
        """将输入转换为 Transform。

        Parameters
        ----------
        value : Transform | np.ndarray
            输入值。

        Returns
        -------
        Transform
            Transform 对象。
        """
        if isinstance(value, Transform):
            return value

        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"变换矩阵必须为 4x4，当前 shape={matrix.shape}")
        return Transform.from_SE3(matrix)
