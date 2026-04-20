from collections.abc import Callable

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QApplication, QSlider, QStyle, QStyleFactory, QStyleOptionSlider


class CasiaValueSlider(QSlider):
    @staticmethod
    def _get_touch_scale() -> float:
        app = QApplication.instance()
        if app is None:
            return 1.0
        try:
            scale = float(app.property("touchScale") or 1.0)
        except (TypeError, ValueError):
            return 1.0
        return max(1.0, scale)

    def _scaled(self, value: int) -> int:
        return int(round(value * self._touch_scale))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._touch_scale = self._get_touch_scale()
        self._value_converter: Callable[[int], str] | None = None
        self._font_size: int = self._scaled(16)  # 默认字体大小
        self._slider_width_chars: int = 6  # 默认滑块宽度（字符数）
        self._min_font_size: int = self._scaled(10)  # 最小字体大小
        self._slider_height: int = self._scaled(40)  # 滑块高度

        # 设置一些默认样式，确保滑块足够大以显示文本
        self.setMinimumHeight(self._scaled(60))
        self._update_style()

    def set_value_converter(self, converter: Callable[[int], str]):
        """
        设置数值转换器

        Args:
            converter: 可调用对象，接收当前 value 值，返回要显示的字符串
        """
        self._value_converter = converter
        self.update()

    def set_font_size(self, size: int):
        """
        设置滑块上显示的字体大小

        Args:
            size: 字体大小，不得小于最小字体大小
        """
        if size >= self._min_font_size:
            self._font_size = size
            self._update_style()
            self.update()
        else:
            print(f"Warning: Font size {size} is less than minimum {self._min_font_size}")

    def set_slider_width_chars(self, width_chars: int):
        """
        设置滑块宽度（以字符数为单位）

        Args:
            width_chars: 滑块宽度（字符数）
        """
        if width_chars > 0:
            self._slider_width_chars = width_chars
            self._update_style()
            self.update()

    def set_slider_height(self, height: int):
        """
        设置滑块高度

        Args:
            height: 滑块高度
        """
        if height > 0:
            self._slider_height = height
            self._update_style()
            self.update()

    def _update_style(self):
        """更新滑块样式以调整尺寸"""
        # 计算滑块宽度（基于字符数和字体大小）
        font = QFont("Arial", self._font_size)
        font_metrics = QFontMetrics(font)
        char_width = font_metrics.horizontalAdvance("M")
        slider_width = char_width * self._slider_width_chars

        # 设置样式表来调整滑块手柄大小
        if self.orientation() == Qt.Orientation.Horizontal:
            style_sheet = f"""
            QSlider::handle:horizontal {{
                width: {slider_width}px;
                height: {self._slider_height}px;
                margin: -{self._slider_height//2 - 5}px 0;
                border-radius: {min(slider_width, self._slider_height)//4}px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5c5c5c, stop:1 #787878);
            }}
            QSlider::groove:horizontal {{
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #b1b1b1, stop:1 #c4c4c4);
                border-radius: 4px;
            }}
            """
        else:
            style_sheet = f"""
            QSlider::handle:vertical {{
                width: {self._slider_height}px;
                height: {slider_width}px;
                margin: 0 -{self._slider_height//2 - 5}px;
                border-radius: {min(slider_width, self._slider_height)//4}px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5c5c5c, stop:1 #787878);
            }}
            QSlider::groove:vertical {{
                width: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #b1b1b1, stop:1 #c4c4c4);
                border-radius: 4px;
            }}
            """

        self.setStyleSheet(style_sheet)

    def get_display_text(self) -> str:
        """
        获取要在滑块上显示的文本
        """
        current_value = self.value()

        if self._value_converter is not None:
            try:
                return self._value_converter(current_value)
            except Exception as e:
                # 如果转换器出错，回退到默认显示
                print(f"Value converter error: {e}")
                return str(current_value)
        else:
            # 默认显示数值
            return str(current_value)

    def paintEvent(self, event):
        # 先调用父类的绘制方法，绘制基本的滑块
        super().paintEvent(event)

        # 创建绘制器
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 设置初始字体
        font = QFont("Arial", self._font_size)
        painter.setFont(font)

        # 设置文本颜色
        painter.setPen(Qt.GlobalColor.white)

        # 获取滑块的位置和大小
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        # 获取滑块手柄的矩形区域
        slider_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self
        )

        # 如果滑块矩形有效，则在其中绘制文本
        if slider_rect.isValid():
            display_text = self.get_display_text()

            # 确保文本适合滑块大小
            font_metrics = QFontMetrics(font)
            text_width = font_metrics.horizontalAdvance(display_text)
            current_font_size = self._font_size

            # 如果文本太宽，逐步减小字体大小直到合适或达到最小值
            while text_width > slider_rect.width() * 0.9 and current_font_size > self._min_font_size:
                current_font_size -= 1
                smaller_font = QFont(font)
                smaller_font.setPointSize(current_font_size)
                font_metrics = QFontMetrics(smaller_font)
                text_width = font_metrics.horizontalAdvance(display_text)

            # 应用调整后的字体
            if current_font_size != self._font_size:
                adjusted_font = QFont(font)
                adjusted_font.setPointSize(current_font_size)
                painter.setFont(adjusted_font)

            # 在滑块中心绘制文本
            painter.drawText(slider_rect, Qt.AlignmentFlag.AlignCenter, display_text)

        painter.end()

    def get_font_size(self) -> int:
        """获取当前字体大小"""
        return self._font_size

    def get_slider_width_chars(self) -> int:
        """获取当前滑块宽度（字符数）"""
        return self._slider_width_chars

    def get_min_font_size(self) -> int:
        """获取最小字体大小"""
        return self._min_font_size

    def set_min_font_size(self, size: int):
        """设置最小字体大小"""
        if size > 0:
            self._min_font_size = size
            if self._font_size < size:
                self._font_size = size
            self._update_style()
            self.update()

    def get_slider_height(self) -> int:
        """获取滑块高度"""
        return self._slider_height


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
