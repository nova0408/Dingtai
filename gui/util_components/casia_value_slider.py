from collections.abc import Callable

from PySide6.QtCore import Property, QRect, Qt, Slot
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeyEvent, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QSlider, QStyle, QStyleOptionSlider

from .casia_value_converter import CallableValueConverter, CasiaValueConverter, IntValueConverter


class CasiaValueSlider(QSlider):
    """带数值显示和点击编辑能力的滑块控件。

    Notes
    -----
    该控件在标准 ``QSlider`` 基础上增加以下能力：

    1. 在滑块中心绘制当前值文本。
    2. 支持 ``CasiaValueConverter``，用于控制显示文本和输入反解析。
    3. 支持单击滑块中心进入编辑模式，按回车提交。
    4. 支持通过 QSS 的 ``qproperty-*`` 修改主要视觉样式。

    Examples
    --------
    QSS 示例::

        CasiaValueSlider {
            qproperty-handleStartColor: #3f6ea8;
            qproperty-handleEndColor: #4d86c8;
            qproperty-grooveStartColor: #2f2f2f;
            qproperty-grooveEndColor: #484848;
            qproperty-sliderTextColor: white;
            qproperty-editorTextColor: white;
            qproperty-editorBackgroundColor: #2b2b2b;
            qproperty-editorBorderColor: #6fa8dc;
            qproperty-grooveThickness: 8;
        }

    Parameters
    ----------
    orientation : Qt.Orientation
        滑块方向。
    parent : QWidget | None
        父控件。
    """

    # region 初始化与缩放

    @staticmethod
    def _get_touch_scale() -> float:
        """读取应用级触屏缩放比例。

        Returns
        -------
        float
            不小于 1.0 的缩放比例。
        """
        app = QApplication.instance()
        if app is None:
            return 1.0
        try:
            scale = float(app.property("touchScale") or 1.0)
        except (TypeError, ValueError):
            return 1.0
        return max(1.0, scale)

    def _scaled(self, value: int) -> int:
        """按触屏比例缩放整数尺寸。

        Parameters
        ----------
        value : int
            原始像素尺寸。

        Returns
        -------
        int
            缩放后的像素尺寸。
        """
        return int(round(value * self._touch_scale))

    def __init__(self, orientation: Qt.Orientation = Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._touch_scale = self._get_touch_scale()
        self._interaction_enabled: bool = True
        self._click_edit_enabled: bool = True
        self._value_converter: CasiaValueConverter = IntValueConverter()
        self._font_size: int = self._scaled(16)  # 默认字体大小
        self._slider_width_chars: int = 6  # 默认滑块宽度（字符数）
        self._min_font_size: int = self._scaled(10)  # 最小字体大小
        self._slider_height: int = self._scaled(40)  # 滑块高度
        self._groove_thickness: int = self._scaled(8)
        self._handle_start_color = QColor("#5c5c5c")
        self._handle_end_color = QColor("#787878")
        self._groove_start_color = QColor("#b1b1b1")
        self._groove_end_color = QColor("#c4c4c4")
        self._slider_text_color = QColor(Qt.GlobalColor.white)
        self._editor_text_color = QColor(Qt.GlobalColor.white)
        self._editor_background_color = QColor("#2b2b2b")
        self._editor_border_color = QColor("#6fa8dc")
        self._press_pos = None
        self._pressed_on_handle = False
        self._drag_threshold = QApplication.startDragDistance()
        self._value_editor = QLineEdit(self)
        self._value_editor.hide()
        self._value_editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_editor.returnPressed.connect(self._on_value_editor_return_pressed)

        # 设置一些默认样式，确保滑块足够大以显示文本
        self.setMinimumHeight(self._scaled(60))
        self._update_style()

    # endregion

    # region Qt 属性与公开配置

    def get_interaction_enabled(self) -> bool:
        """获取用户交互开关。

        Returns
        -------
        bool
            ``True`` 表示允许拖动、滚轮和键盘调节。
        """
        return self._interaction_enabled

    def set_interaction_enabled(self, enabled: bool) -> None:
        """设置用户交互开关。

        Parameters
        ----------
        enabled : bool
            是否允许用户交互。
        """
        self._interaction_enabled = bool(enabled)

    interactionEnabled = Property(bool, get_interaction_enabled, set_interaction_enabled)

    def get_click_edit_enabled(self) -> bool:
        """获取点击编辑开关。

        Returns
        -------
        bool
            ``True`` 表示允许单击滑块中心进入文本编辑模式。
        """
        return self._click_edit_enabled

    def set_click_edit_enabled(self, enabled: bool) -> None:
        """设置点击编辑开关。

        Parameters
        ----------
        enabled : bool
            是否允许点击编辑。
        """
        self._click_edit_enabled = bool(enabled)
        if not self._click_edit_enabled:
            self._cancel_value_edit()

    clickEditEnabled = Property(bool, get_click_edit_enabled, set_click_edit_enabled)

    def get_handle_start_color(self) -> QColor:
        """获取滑块手柄渐变起始色。"""
        return QColor(self._handle_start_color)

    def set_handle_start_color(self, color: QColor) -> None:
        """设置滑块手柄渐变起始色。"""
        self._handle_start_color = QColor(color)
        self._update_style()

    handleStartColor = Property(QColor, get_handle_start_color, set_handle_start_color)

    def get_handle_end_color(self) -> QColor:
        """获取滑块手柄渐变结束色。"""
        return QColor(self._handle_end_color)

    def set_handle_end_color(self, color: QColor) -> None:
        """设置滑块手柄渐变结束色。"""
        self._handle_end_color = QColor(color)
        self._update_style()

    handleEndColor = Property(QColor, get_handle_end_color, set_handle_end_color)

    def get_groove_start_color(self) -> QColor:
        """获取滑槽渐变起始色。"""
        return QColor(self._groove_start_color)

    def set_groove_start_color(self, color: QColor) -> None:
        """设置滑槽渐变起始色。"""
        self._groove_start_color = QColor(color)
        self._update_style()

    grooveStartColor = Property(QColor, get_groove_start_color, set_groove_start_color)

    def get_groove_end_color(self) -> QColor:
        """获取滑槽渐变结束色。"""
        return QColor(self._groove_end_color)

    def set_groove_end_color(self, color: QColor) -> None:
        """设置滑槽渐变结束色。"""
        self._groove_end_color = QColor(color)
        self._update_style()

    grooveEndColor = Property(QColor, get_groove_end_color, set_groove_end_color)

    def get_slider_text_color(self) -> QColor:
        """获取滑块中心文本颜色。"""
        return QColor(self._slider_text_color)

    def set_slider_text_color(self, color: QColor) -> None:
        """设置滑块中心文本颜色。"""
        self._slider_text_color = QColor(color)
        self.update()

    sliderTextColor = Property(QColor, get_slider_text_color, set_slider_text_color)

    def get_editor_text_color(self) -> QColor:
        """获取点击编辑输入框文本颜色。"""
        return QColor(self._editor_text_color)

    def set_editor_text_color(self, color: QColor) -> None:
        """设置点击编辑输入框文本颜色。"""
        self._editor_text_color = QColor(color)
        self._update_editor_style()

    editorTextColor = Property(QColor, get_editor_text_color, set_editor_text_color)

    def get_editor_background_color(self) -> QColor:
        """获取点击编辑输入框背景色。"""
        return QColor(self._editor_background_color)

    def set_editor_background_color(self, color: QColor) -> None:
        """设置点击编辑输入框背景色。"""
        self._editor_background_color = QColor(color)
        self._update_editor_style()

    editorBackgroundColor = Property(QColor, get_editor_background_color, set_editor_background_color)

    def get_editor_border_color(self) -> QColor:
        """获取点击编辑输入框边框色。"""
        return QColor(self._editor_border_color)

    def set_editor_border_color(self, color: QColor) -> None:
        """设置点击编辑输入框边框色。"""
        self._editor_border_color = QColor(color)
        self._update_editor_style()

    editorBorderColor = Property(QColor, get_editor_border_color, set_editor_border_color)

    def get_groove_thickness(self) -> int:
        """获取滑槽厚度。"""
        return self._groove_thickness

    def set_groove_thickness(self, thickness: int) -> None:
        """设置滑槽厚度。"""
        if thickness <= 0:
            return
        self._groove_thickness = int(thickness)
        self._update_style()

    grooveThickness = Property(int, get_groove_thickness, set_groove_thickness)

    def set_value_converter(self, converter: CasiaValueConverter | Callable[[int], str] | None):
        """设置数值转换器。

        Parameters
        ----------
        converter : CasiaValueConverter | Callable[[int], str] | None
            转换器对象，或兼容旧接口的 ``Callable[[int], str]``。
        """
        if converter is None:
            self._value_converter = IntValueConverter()
        elif isinstance(converter, CasiaValueConverter):
            self._value_converter = converter
        else:
            self._value_converter = CallableValueConverter(converter)
        self.update()

    def set_font_size(self, size: int):
        """设置滑块上显示的字体大小。

        Parameters
        ----------
        size : int
            字体大小，不得小于最小字体大小。
        """
        if size >= self._min_font_size:
            self._font_size = size
            self._update_style()
            self.update()
        else:
            print(f"Warning: Font size {size} is less than minimum {self._min_font_size}")

    def set_slider_width_chars(self, width_chars: int):
        """设置滑块宽度。

        Parameters
        ----------
        width_chars : int
            滑块宽度，单位为字符数。
        """
        if width_chars > 0:
            self._slider_width_chars = width_chars
            self._update_style()
            self.update()

    def set_slider_height(self, height: int):
        """设置滑块高度。

        Parameters
        ----------
        height : int
            滑块手柄高度。
        """
        if height > 0:
            self._slider_height = height
            self._update_style()
            self.update()

    # endregion

    # region 样式计算

    def _update_style(self):
        """根据当前尺寸和 QSS 属性刷新滑块样式。"""
        font = QFont("Arial", self._font_size)
        font_metrics = QFontMetrics(font)
        char_width = font_metrics.horizontalAdvance("M")
        slider_width = char_width * self._slider_width_chars
        border_radius = min(slider_width, self._slider_height) // 4

        if self.orientation() == Qt.Orientation.Horizontal:
            style_sheet = f"""
            QSlider::handle:horizontal {{
                width: {slider_width}px;
                height: {self._slider_height}px;
                margin: -{self._slider_height//2 - 5}px 0;
                border-radius: {border_radius}px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self._handle_start_color.name()}, stop:1 {self._handle_end_color.name()});
            }}
            QSlider::groove:horizontal {{
                height: {self._groove_thickness}px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {self._groove_start_color.name()}, stop:1 {self._groove_end_color.name()});
                border-radius: {self._groove_thickness // 2}px;
            }}
            """
        else:
            style_sheet = f"""
            QSlider::handle:vertical {{
                width: {self._slider_height}px;
                height: {slider_width}px;
                margin: 0 -{self._slider_height//2 - 5}px;
                border-radius: {border_radius}px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {self._handle_start_color.name()}, stop:1 {self._handle_end_color.name()});
            }}
            QSlider::groove:vertical {{
                width: {self._groove_thickness}px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {self._groove_start_color.name()}, stop:1 {self._groove_end_color.name()});
                border-radius: {self._groove_thickness // 2}px;
            }}
            """

        self.setStyleSheet(style_sheet)
        self._update_editor_style()

    def _update_editor_style(self) -> None:
        """刷新点击编辑输入框样式。"""
        self._value_editor.setStyleSheet(
            f"""
            QLineEdit {{
                color: {self._editor_text_color.name()};
                background-color: {self._editor_background_color.name()};
                border: 1px solid {self._editor_border_color.name()};
                border-radius: 3px;
                padding: 0px;
            }}
            """
        )

    # endregion

    # region 显示文本与绘制

    def get_display_text(self) -> str:
        """获取滑块中心显示文本。

        Returns
        -------
        str
            经转换器处理后的显示文本。
        """
        current_value = self.value()

        try:
            return self._value_converter.convert(current_value)
        except (TypeError, ValueError):
            return str(current_value)

    def paintEvent(self, event):
        """绘制滑块和手柄中心文本。"""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Arial", self._font_size)
        painter.setFont(font)
        painter.setPen(self._slider_text_color)

        slider_rect = self._handle_rect()
        if slider_rect.isValid():
            display_text = self.get_display_text()

            font_metrics = QFontMetrics(font)
            text_width = font_metrics.horizontalAdvance(display_text)
            current_font_size = self._font_size

            # 文本过长时仅缩小绘制字体，不改变控件配置，避免布局抖动。
            while text_width > slider_rect.width() * 0.9 and current_font_size > self._min_font_size:
                current_font_size -= 1
                smaller_font = QFont(font)
                smaller_font.setPointSize(current_font_size)
                font_metrics = QFontMetrics(smaller_font)
                text_width = font_metrics.horizontalAdvance(display_text)

            if current_font_size != self._font_size:
                adjusted_font = QFont(font)
                adjusted_font.setPointSize(current_font_size)
                painter.setFont(adjusted_font)

            painter.drawText(slider_rect, Qt.AlignmentFlag.AlignCenter, display_text)

        painter.end()

    # endregion

    # region Qt 事件

    def mousePressEvent(self, event: QMouseEvent):
        """记录鼠标按下位置，用于区分点击编辑和拖动。"""
        if not self._interaction_enabled:
            event.ignore()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.pos()
            self._pressed_on_handle = self._handle_rect().contains(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动超过拖动阈值后取消本次点击编辑判定。"""
        if not self._interaction_enabled:
            event.ignore()
            return
        if self._press_pos is not None and (event.pos() - self._press_pos).manhattanLength() > self._drag_threshold:
            self._pressed_on_handle = False
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """释放鼠标时根据按下/释放位置决定是否进入编辑模式。"""
        if not self._interaction_enabled:
            event.ignore()
            return
        should_edit = self._is_click_edit_release(event)
        super().mouseReleaseEvent(event)
        self._press_pos = None
        self._pressed_on_handle = False
        if should_edit:
            self._start_value_edit()

    def wheelEvent(self, event: QWheelEvent):
        """根据交互开关拦截滚轮调节。"""
        if not self._interaction_enabled:
            event.ignore()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        """根据交互状态处理键盘调节。"""
        if self._value_editor.isVisible():
            event.ignore()
            return
        if not self._interaction_enabled:
            event.ignore()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event):
        """控件尺寸变化时同步编辑框位置。"""
        super().resizeEvent(event)
        if self._value_editor.isVisible():
            self._position_value_editor()

    # endregion

    # region 尺寸配置读取

    def get_font_size(self) -> int:
        """获取当前字体大小。

        Returns
        -------
        int
            当前滑块文本字体大小。
        """
        return self._font_size

    def get_slider_width_chars(self) -> int:
        """获取当前滑块宽度字符数。

        Returns
        -------
        int
            手柄宽度对应的字符数量。
        """
        return self._slider_width_chars

    def get_min_font_size(self) -> int:
        """获取最小字体大小。

        Returns
        -------
        int
            文本自动缩小时允许使用的最小字号。
        """
        return self._min_font_size

    def set_min_font_size(self, size: int):
        """设置最小字体大小。

        Parameters
        ----------
        size : int
            文本自动缩小时允许使用的最小字号。
        """
        if size > 0:
            self._min_font_size = size
            if self._font_size < size:
                self._font_size = size
            self._update_style()
            self.update()

    def get_slider_height(self) -> int:
        """获取滑块高度。

        Returns
        -------
        int
            手柄高度。
        """
        return self._slider_height

    # endregion

    # region 点击编辑内部逻辑

    def _handle_rect(self) -> QRect:
        """获取当前手柄矩形。

        Returns
        -------
        QRect
            当前样式计算出的手柄矩形。
        """
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        return self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )

    def _is_click_edit_release(self, event: QMouseEvent) -> bool:
        """判断一次鼠标释放是否应触发点击编辑。

        Parameters
        ----------
        event : QMouseEvent
            鼠标释放事件。

        Returns
        -------
        bool
            ``True`` 表示本次释放可进入编辑模式。
        """
        if not self._click_edit_enabled:
            return False
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self._press_pos is None or not self._pressed_on_handle:
            return False
        if (event.pos() - self._press_pos).manhattanLength() > self._drag_threshold:
            return False
        return self._handle_rect().contains(event.pos())

    def _start_value_edit(self) -> None:
        """显示覆盖在手柄上的输入框。"""
        self._value_editor.setText(self.get_display_text())
        self._position_value_editor()
        self._value_editor.selectAll()
        self._value_editor.show()
        self._value_editor.setFocus(Qt.FocusReason.MouseFocusReason)

    def _position_value_editor(self) -> None:
        """将输入框同步到当前手柄矩形内。"""
        handle_rect = self._handle_rect()
        if not handle_rect.isValid():
            return
        self._value_editor.setGeometry(handle_rect.adjusted(2, 2, -2, -2))

    def _cancel_value_edit(self) -> None:
        """关闭输入框并恢复滑块绘制。"""
        self._value_editor.hide()
        self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.update()

    @Slot()
    def _on_value_editor_return_pressed(self) -> None:
        """处理输入框回车提交。"""
        try:
            value = self._value_converter.convert_back(self._value_editor.text())
        except (TypeError, ValueError):
            self._cancel_value_edit()
            return
        if not self.minimum() <= value <= self.maximum():
            self._cancel_value_edit()
            return
        old_value = self.value()
        self.setValue(value)
        if old_value == value:
            self.valueChanged.emit(value)
        self._cancel_value_edit()

    # endregion


# 使用示例
if __name__ == "__main__":
    import os
    import sys

    python_env_path = sys.prefix
    pyside6_plugin_path = os.path.join(python_env_path, "Lib", "site-packages", "PySide6", "plugins", "platforms")
    # 设置 QT_QPA_PLATFORM_PLUGIN_PATH 环境变量
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyside6_plugin_path

    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    class DemoWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("ValueSlider Demo")
            self.setGeometry(100, 100, 500, 500)

            layout = QVBoxLayout()

            # 示例 1: 默认设置
            self.slider1 = CasiaValueSlider(Qt.Orientation.Horizontal)
            self.slider1.setRange(0, 100)
            self.slider1.setValue(50)
            layout.addWidget(QLabel("默认设置 (字体 14，宽度 6 字符):"))
            layout.addWidget(self.slider1)

            # 示例 2: 自定义字体大小和宽度
            self.slider2 = CasiaValueSlider(Qt.Orientation.Horizontal)
            self.slider2.setRange(0, 100)
            self.slider2.setValue(75)
            self.slider2.set_font_size(16)
            self.slider2.set_slider_width_chars(8)
            self.slider2.set_value_converter(lambda x: f"{x}%")
            layout.addWidget(QLabel("大字体宽滑块 (16px, 8 字符):"))
            layout.addWidget(self.slider2)

            # 示例 3: 窄滑块
            self.slider3 = CasiaValueSlider(Qt.Orientation.Horizontal)
            self.slider3.setRange(0, 100)
            self.slider3.setValue(25)
            self.slider3.set_slider_width_chars(4)
            self.slider3.set_value_converter(lambda x: f"{x}")
            layout.addWidget(QLabel("窄滑块 (4 字符):"))
            layout.addWidget(self.slider3)

            # 示例 4: 垂直滑块
            self.slider4 = CasiaValueSlider(Qt.Orientation.Vertical)
            self.slider4.setRange(0, 100)
            self.slider4.setValue(60)
            self.slider4.set_slider_width_chars(6)
            self.slider4.set_value_converter(lambda x: f"{x}dB")
            layout.addWidget(QLabel("垂直滑块："))

            # 水平布局放置垂直滑块
            h_layout = QHBoxLayout()
            h_layout.addWidget(self.slider4)
            h_layout.addStretch()
            layout.addLayout(h_layout)

            # 控制按钮
            control_layout = QVBoxLayout()

            btn1 = QPushButton("设置滑块 1 宽度为 10 字符")
            btn1.clicked.connect(lambda: self.slider1.set_slider_width_chars(10))
            control_layout.addWidget(btn1)

            btn2 = QPushButton("设置滑块 2 字体为 12")
            btn2.clicked.connect(lambda: self.slider2.set_font_size(12))
            control_layout.addWidget(btn2)

            btn3 = QPushButton("设置滑块 3 高度为 40px")
            btn3.clicked.connect(lambda: self.slider3.set_slider_height(40))
            control_layout.addWidget(btn3)

            layout.addLayout(control_layout)
            self.setLayout(layout)

    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())
