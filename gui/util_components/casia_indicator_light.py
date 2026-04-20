import math
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Property, QEvent, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# --- 数据类保持不变 ---
@dataclass
class IndicatorStatus:
    TRUE: str
    FALSE: str


def _asIndicatorStatus(text: IndicatorStatus | tuple | list) -> IndicatorStatus:
    true_value, false_value = "真", "假"
    if isinstance(text, IndicatorStatus):
        return text
    elif isinstance(text, (tuple, list)):
        if len(text) >= 2:
            true_value, false_value = str(text[0]), str(text[1])
    return IndicatorStatus(true_value, false_value)


class CasiaIndicatorLight(QWidget):
    # 增加点击信号
    statusChanged = Signal(bool)
    clicked = Signal()

    @classmethod
    def replace_placeholder(
        cls, placeholder, text=("Connected", "Disconnected"), font_size=14, status_colors=None, default_status=False
    ):
        if placeholder is None:
            raise ValueError("必须提供需要替换的 widget")

        parent = placeholder.parent()
        if parent is None:
            raise ValueError("缺少需要替换的 widget 的父 widget")

        parent_layout = parent.layout()
        if parent_layout is None:
            raise ValueError("需要替换的 widget 的父 widget 必须有 layout")

        indicator = cls(parent=parent, text=text, font_size=font_size, default_status=default_status)
        if status_colors:
            indicator.setStatusColors(*status_colors)

        parent_layout.replaceWidget(placeholder, indicator)
        placeholder.deleteLater()
        return indicator

    def __init__(
        self,
        parent=None,
        text: IndicatorStatus | tuple | list = ("运行", "停止"),
        status_color: tuple[QColor | Any, QColor | Any] = (Qt.GlobalColor.green, Qt.GlobalColor.red),
        text_color: tuple[QColor | Any, QColor | Any] = (Qt.GlobalColor.black, Qt.GlobalColor.white),
        default_status: bool = False,
        font_size: int = 14,
    ):
        super().__init__(parent=parent)

        self._status = default_status
        self._text_obj = _asIndicatorStatus(text)
        self._status_colors = list(status_color)  # 转为 list 以便修改
        self._text_colors = list(text_color)
        self._font_size = font_size

        # 交互状态
        self._is_pressed = False
        # self._is_hovered = False

        self._formatted_true = ""
        self._formatted_false = ""

        # 初始化
        self._update_formatted_text_and_size()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # 开启鼠标追踪以便做 Hover 效果（可选，如果不需要 hover 变色可去掉）
        self.setMouseTracking(True)

    # --- 核心逻辑区 (保持之前的智能算法) ---

    def _is_mostly_english(self, text: str) -> bool:
        if not text:
            return False
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return ascii_count / len(text) > 0.8

    def _smart_wrap_text(self, text: str) -> str:
        """智能分行算法"""
        if not text:
            return ""
        limit = 15 if self._is_mostly_english(text) else 16
        if len(text) > limit:
            text = text[:limit]
        length = len(text)

        if self._is_mostly_english(text):
            width = max(4, int(math.sqrt(length * 2.5)))
            return "\n".join(textwrap.wrap(text, width=width))
        else:
            if length <= 3:
                return text
            elif length == 4:
                return f"{text[:2]}\n{text[2:]}"
            else:
                cols = max(2, math.ceil(math.sqrt(length)))
                lines = [text[i : i + cols] for i in range(0, length, cols)]
                return "\n".join(lines)

    def _update_formatted_text_and_size(self):
        """核心尺寸计算"""
        self._formatted_true = self._smart_wrap_text(self._text_obj.TRUE)
        self._formatted_false = self._smart_wrap_text(self._text_obj.FALSE)

        font = QFont()
        font.setPointSize(self._font_size)
        font.setBold(True)
        fm = QFontMetrics(font)

        max_diagonal = 0
        for txt in [self._formatted_true, self._formatted_false]:
            if not txt:
                continue
            rect = fm.boundingRect(QRect(0, 0, 1000, 1000), 0, txt)
            diag = math.sqrt(rect.width() ** 2 + rect.height() ** 2)
            if diag > max_diagonal:
                max_diagonal = diag

        diameter = int(max_diagonal / 0.85)
        min_d = self._font_size * 2.5
        diameter = max(int(diameter), int(min_d))

        self.setFixedSize(diameter, diameter)
        self.update()

    # --- 增强：覆盖标准 setFont 方法 ---
    # 这样如果父控件调用了 setFont，我们也能响应
    def setFont(self, font: QFont):
        super().setFont(font)
        # 提取字号并更新
        if font.pointSize() > 0:
            self._font_size = font.pointSize()
        elif font.pixelSize() > 0:
            # 简单粗暴的转换，实际可能需要 DPI 计算
            self._font_size = int(font.pixelSize() * 0.75)
        self._update_formatted_text_and_size()

    # --- 增强：鼠标交互事件 ---

    def mousePressEvent(self, event: QMouseEvent):
        # 1. 如果被禁用，直接忽略，不传递事件
        if not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressed = True
            self.update()  # 触发重绘以显示按压效果
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        # 1. 如果被禁用，直接忽略
        if not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_pressed = False
            self.update()
            # 只有在控件内释放才算点击
            if self.rect().contains(event.pos()):
                self.clicked.emit()
        super().mouseReleaseEvent(event)

    # --- 新增：状态改变事件 ---
    def changeEvent(self, event: QEvent):
        # 当 Enabled 属性改变时，需要强制重绘以显示灰色状态
        # 同时也应该重置按压状态，防止状态锁死
        if event.type() == QEvent.Type.EnabledChange:
            self._is_pressed = False
            self.update()
        super().changeEvent(event)

    # --- 属性接口 ---

    @Property(bool)
    def status(self):
        return self._status

    @status.setter
    def status(self, value: bool):
        if self._status != value:
            self._status = value
            self.statusChanged.emit(value)
            self.update()

    def set_status(self, value: bool):
        if self._status != value:
            self._status = value
            self.statusChanged.emit(value)
            self.update()

    @Property(int)
    def fontSize(self):
        return self._font_size

    @fontSize.setter
    def fontSize(self, size: int):
        self._font_size = size
        self._update_formatted_text_and_size()

    # 为了方便外部修改颜色，提供专门的方法
    def setStatusColors(self, true_color: QColor, false_color: QColor):
        self._status_colors = [true_color, false_color]
        self.update()

    def setTextColors(self, true_color: QColor, false_color: QColor):
        self._text_colors = [true_color, false_color]
        self.update()

    # --- 绘图 ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height())
        rect = QRect(0, 0, side, side)
        rect.moveCenter(self.rect().center())

        idx = 0 if self._status else 1

        # 1. 获取颜色
        bg_color = QColor(self._status_colors[idx])

        # 2. 交互反馈：如果按下，颜色变深 (Darker)
        if self._is_pressed:
            bg_color = bg_color.darker(115)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg_color)
        painter.drawEllipse(rect)

        # 3. 绘制文字
        painter.setPen(self._text_colors[idx])
        font = QFont()
        font.setPointSize(self._font_size)
        font.setBold(True)
        painter.setFont(font)

        display_text = self._formatted_true if self._status else self._formatted_false

        scale_factor = 0.85
        target_w = int(side * scale_factor)
        target_h = int(side * scale_factor)
        draw_rect = QRect(0, 0, target_w, target_h)
        draw_rect.moveCenter(rect.center())

        painter.drawText(draw_rect, Qt.AlignmentFlag.AlignCenter, display_text)


# ==========================================
#              综合测试控制台
# ==========================================
class DemoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("指示灯控件综合测试")
        self.resize(900, 600)

        main_layout = QHBoxLayout(self)

        # --- 左侧：展示区 ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.display_area = QWidget()
        self.display_layout = QVBoxLayout(self.display_area)
        self.display_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setWidget(self.display_area)

        # 创建几个不同类型的指示灯
        self.indicators = []

        # 1. 交互式开关 (点击切换)
        self.ind_interactive = CasiaIndicatorLight(
            text=("ON", "OFF"), status_color=(Qt.GlobalColor.green, Qt.GlobalColor.darkGray)
        )
        self.ind_interactive.clicked.connect(lambda: self.log("指示灯 1 被点击"))
        # 模拟按钮行为：点击反转状态
        self.ind_interactive.clicked.connect(lambda: self.ind_interactive.setStatus(not self.ind_interactive.status))
        self.add_indicator_row("1. 交互式开关 (点击我):", self.ind_interactive)

        # 2. 4 字排版测试
        self.ind_text_layout = CasiaIndicatorLight(text=("系统正常", "断开连接"), font_size=16)
        self.add_indicator_row("2. 4 字排版 (2x2):", self.ind_text_layout)

        # 3. 长文本测试
        self.ind_long_text = CasiaIndicatorLight(text=("这是一个非常长的文本测试会自动换行", "短"), font_size=16)
        self.add_indicator_row("3. 长文本 (均衡):", self.ind_long_text)

        main_layout.addWidget(scroll, 6)  # 左侧占 60%

        # --- 右侧：控制面板 ---
        control_panel = QGroupBox("控制面板")
        ctrl_layout = QVBoxLayout(control_panel)
        ctrl_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1. 字体大小控制
        ctrl_layout.addWidget(QLabel("<b>全局字体大小:</b>"))
        self.slider_font = QSlider(Qt.Orientation.Horizontal)
        self.slider_font.setRange(10, 40)
        self.slider_font.setValue(14)
        self.label_font_val = QLabel("14 px")
        self.slider_font.valueChanged.connect(self.update_all_fonts)

        font_layout = QHBoxLayout()
        font_layout.addWidget(self.slider_font)
        font_layout.addWidget(self.label_font_val)
        ctrl_layout.addLayout(font_layout)

        ctrl_layout.addWidget(QLabel("<small>* 拖动滑块，所有控件尺寸会实时重算</small>"))
        ctrl_layout.addSpacing(20)

        # 2. 颜色控制
        ctrl_layout.addWidget(QLabel("<b>长文本控件颜色 (状态：真):</b>"))
        btn_color_true = QPushButton("修改 True 背景色")
        btn_color_true.clicked.connect(lambda: self.pick_color("true_bg"))
        btn_txt_true = QPushButton("修改 True 文字色")
        btn_txt_true.clicked.connect(lambda: self.pick_color("true_txt"))

        ctrl_layout.addWidget(btn_color_true)
        ctrl_layout.addWidget(btn_txt_true)
        ctrl_layout.addSpacing(20)

        # 3. 状态切换控制
        btn_toggle_all = QPushButton("切换所有灯状态")
        btn_toggle_all.clicked.connect(self.toggle_all)
        ctrl_layout.addWidget(btn_toggle_all)

        ctrl_layout.addStretch()

        # 日志区
        self.log_label = QLabel("日志：准备就绪")
        self.log_label.setWordWrap(True)
        self.log_label.setProperty("themeRole", "log-label")
        ctrl_layout.addWidget(self.log_label)

        main_layout.addWidget(control_panel, 4)  # 右侧占 40%

    def add_indicator_row(self, label_text, widget):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(150)
        row.addWidget(lbl)
        row.addWidget(widget)
        row.addStretch()  # 让灯靠左
        self.display_layout.addLayout(row)
        self.display_layout.addSpacing(20)
        self.indicators.append(widget)

    def log(self, text):
        self.log_label.setText(f"日志：{text}")

    def update_all_fonts(self, val):
        self.label_font_val.setText(f"{val} px")
        for ind in self.indicators:
            ind.fontSize = val

    def toggle_all(self):
        for ind in self.indicators:
            ind.status = not ind.status
        self.log("所有状态已切换")

    def pick_color(self, target):
        color = QColorDialog.getColor()
        if color.isValid():
            if target == "true_bg":
                # 修改 ind_long_text 的 True 状态背景色
                current_colors = self.ind_long_text._status_colors
                self.ind_long_text.setStatusColors(color, current_colors[1])
            elif target == "true_txt":
                # 修改 ind_long_text 的 True 状态文字色
                current_colors = self.ind_long_text._text_colors
                self.ind_long_text.setTextColors(color, current_colors[1])
            self.ind_long_text.status = True  # 强制切到 True 看效果
            self.log(f"颜色已修改：{target}")


if __name__ == "__main__":
    import os
    import sys

    # 设置 QT_QPA_PLATFORM_PLUGIN_PATH 环境变量
    pyside6_dir = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6")
    os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside6_dir, "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(pyside6_dir, "plugins", "platforms")
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())
