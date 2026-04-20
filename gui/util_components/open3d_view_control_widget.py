from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class O3DViewControlWidget(QWidget):
    """通用画布视图控制控件。

    该控件是“即开即用”的：初始化时传入 canvas，
    内部会自动完成所有信号绑定。

    Parameters
    ----------
    canvas : object | None, optional
        被控制的画布对象。要求至少提供以下接口：

        - show_origin_axis: bool 属性
        - set_standard_view(view_name: str, zoom: float | None = None)
        - set_point_size(size: float)

    parent : QWidget | None, optional
        父级控件。

    Notes
    -----
    这里不强依赖具体的 O3DViewerWidget 类型，
    只依赖其公开接口，便于后续复用。
    """

    def __init__(self, canvas: object | None = None, parent: QWidget | None = None):
        super().__init__(parent)

        self._canvas: object | None = None
        self._point_size: float = 3.0
        self._point_size_step: float = 1.0
        self._point_size_min: float = 1.0
        self._point_size_max: float = 20.0

        self._init_ui()
        self.set_canvas(canvas)

    def _init_ui(self) -> None:
        """初始化界面。"""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 1. 原点坐标轴
        self.origin_axis_checkbox = QCheckBox("显示原点坐标轴")
        main_layout.addWidget(self.origin_axis_checkbox)

        # 2. 标准视图
        view_label = QLabel("标准视图")
        main_layout.addWidget(view_label)

        view_layout = QGridLayout()
        view_layout.setHorizontalSpacing(6)
        view_layout.setVerticalSpacing(6)

        self.front_button = QPushButton("正视")
        self.top_button = QPushButton("俯视")
        self.side_button = QPushButton("侧视")
        self.reset_button = QPushButton("重置")

        view_layout.addWidget(self.front_button, 0, 0)
        view_layout.addWidget(self.top_button, 0, 1)
        view_layout.addWidget(self.side_button, 1, 0)
        view_layout.addWidget(self.reset_button, 1, 1)

        main_layout.addLayout(view_layout)

        # 3. 点大小
        point_label = QLabel("点大小")
        main_layout.addWidget(point_label)

        point_layout = QHBoxLayout()
        point_layout.setSpacing(6)

        self.point_size_minus_button = QPushButton("-")
        self.point_size_plus_button = QPushButton("+")
        self.point_size_value_label = QLabel()

        self.point_size_minus_button.setFixedWidth(32)
        self.point_size_plus_button.setFixedWidth(32)
        self.point_size_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        point_layout.addWidget(self.point_size_minus_button)
        point_layout.addWidget(self.point_size_value_label, 1)
        point_layout.addWidget(self.point_size_plus_button)

        main_layout.addLayout(point_layout)
        main_layout.addStretch(1)

        # 内部自动绑定
        self.origin_axis_checkbox.toggled.connect(self._on_origin_axis_toggled)
        self.front_button.clicked.connect(self._on_front_view_clicked)
        self.top_button.clicked.connect(self._on_top_view_clicked)
        self.side_button.clicked.connect(self._on_side_view_clicked)
        self.reset_button.clicked.connect(self._on_reset_view_clicked)
        self.point_size_minus_button.clicked.connect(self._on_point_size_minus_clicked)
        self.point_size_plus_button.clicked.connect(self._on_point_size_plus_clicked)

        self._update_point_size_label()
        self._set_enabled_by_canvas(False)

    def set_canvas(self, canvas: object | None) -> None:
        """设置或更换被控制的画布。

        Parameters
        ----------
        canvas : object | None
            被控制的画布对象。
        """
        self._canvas = canvas
        self._sync_from_canvas()
        self._set_enabled_by_canvas(canvas is not None)

    def _set_enabled_by_canvas(self, enabled: bool) -> None:
        """根据是否有画布决定控件可用性。"""
        self.origin_axis_checkbox.setEnabled(enabled)
        self.front_button.setEnabled(enabled)
        self.top_button.setEnabled(enabled)
        self.side_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.point_size_minus_button.setEnabled(enabled)
        self.point_size_plus_button.setEnabled(enabled)

    def _sync_from_canvas(self) -> None:
        """从画布同步当前状态到 UI。"""
        if self._canvas is None:
            self.origin_axis_checkbox.setChecked(False)
            return

        show_axis = getattr(self._canvas, "show_origin_axis", False)
        self.origin_axis_checkbox.blockSignals(True)
        self.origin_axis_checkbox.setChecked(bool(show_axis))
        self.origin_axis_checkbox.blockSignals(False)

        # 如果外部已经给画布设置过点大小，也可以在这里尝试读取。
        # 当前 O3DViewerWidget 未暴露 get_point_size()，因此这里维持本地值。
        self._apply_point_size()

    def _update_point_size_label(self) -> None:
        """刷新点大小数值显示。"""
        self.point_size_value_label.setText(f"{self._point_size:.1f}")

    def _apply_point_size(self) -> None:
        """将当前点大小应用到画布。"""
        if self._canvas is None:
            return

        set_point_size = getattr(self._canvas, "set_point_size", None)
        if callable(set_point_size):
            set_point_size(self._point_size)

        self._update_point_size_label()

    @Slot(bool)
    def _on_origin_axis_toggled(self, checked: bool) -> None:
        """切换原点坐标轴显示。"""
        if self._canvas is None:
            return

        if hasattr(self._canvas, "show_origin_axis"):
            setattr(self._canvas, "show_origin_axis", checked)

    @Slot()
    def _on_front_view_clicked(self) -> None:
        """切换为正视图。"""
        if self._canvas is None:
            return

        set_standard_view = getattr(self._canvas, "set_standard_view", None)
        if callable(set_standard_view):
            set_standard_view("front")

    @Slot()
    def _on_top_view_clicked(self) -> None:
        """切换为俯视图。"""
        if self._canvas is None:
            return

        set_standard_view = getattr(self._canvas, "set_standard_view", None)
        if callable(set_standard_view):
            set_standard_view("top")

    @Slot()
    def _on_side_view_clicked(self) -> None:
        """切换为侧视图。

        Notes
        -----
        这里默认使用 right。
        如果你的业务语义更希望从另一侧看，可以改成 left。
        """
        if self._canvas is None:
            return

        set_standard_view = getattr(self._canvas, "set_standard_view", None)
        if callable(set_standard_view):
            set_standard_view("right")

    @Slot()
    def _on_reset_view_clicked(self) -> None:
        """重置视图。"""
        if self._canvas is None:
            return

        reset_view = getattr(self._canvas, "reset_view", None)
        if callable(reset_view):
            reset_view()

    @Slot()
    def _on_point_size_minus_clicked(self) -> None:
        """减小点大小。"""
        self._point_size = max(self._point_size_min, self._point_size - self._point_size_step)
        self._apply_point_size()

    @Slot()
    def _on_point_size_plus_clicked(self) -> None:
        """增大点大小。"""
        self._point_size = min(self._point_size_max, self._point_size + self._point_size_step)
        self._apply_point_size()

    def set_point_size_range(self, minimum: float, maximum: float) -> None:
        """设置点大小范围。

        Parameters
        ----------
        minimum : float
            最小点大小。
        maximum : float
            最大点大小。
        """
        if minimum <= 0:
            raise ValueError("minimum 必须大于 0")
        if maximum < minimum:
            raise ValueError("maximum 必须大于等于 minimum")

        self._point_size_min = minimum
        self._point_size_max = maximum
        self._point_size = min(max(self._point_size, minimum), maximum)
        self._apply_point_size()

    def set_point_size_step(self, step: float) -> None:
        """设置点大小调节步长。

        Parameters
        ----------
        step : float
            步长。
        """
        if step <= 0:
            raise ValueError("step 必须大于 0")
        self._point_size_step = step

    def set_point_size_value(self, value: float) -> None:
        """直接设置当前点大小。

        Parameters
        ----------
        value : float
            点大小。
        """
        if value <= 0:
            raise ValueError("value 必须大于 0")

        self._point_size = min(max(value, self._point_size_min), self._point_size_max)
        self._apply_point_size()
