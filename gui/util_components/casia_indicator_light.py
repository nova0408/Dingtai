import math
import sys
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Property, QEvent, QRect, QSize, Qt, Signal
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


@dataclass(frozen=True, slots=True)
class IndicatorState:
    """多模式指示灯的单个可视状态。"""

    key: str
    text: str
    color: str
    text_color: str = "#ffffff"


def _asIndicatorStatus(text: IndicatorStatus | tuple | list) -> IndicatorStatus:
    true_value, false_value = "真", "假"
    if isinstance(text, IndicatorStatus):
        return text
    elif isinstance(text, (tuple, list)):
        if len(text) >= 2:
            true_value, false_value = str(text[0]), str(text[1])
    return IndicatorStatus(true_value, false_value)


class CasiaIndicatorLight(QWidget):
    _TEXT_FILL_RATIO = 0.8

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
        minimum_font_size: int = 8,
    ):
        super().__init__(parent=parent)

        self._status = default_status
        self._text_obj = _asIndicatorStatus(text)
        self._status_colors = list(status_color)  # 转为 list 以便修改
        self._text_colors = list(text_color)
        self._minimum_font_size = max(1, minimum_font_size)
        self._font_size = max(self._minimum_font_size, font_size)
        self._margin = 4

        # 交互状态
        self._is_pressed = False
        # self._is_hovered = False

        self._formatted_true = ""
        self._formatted_false = ""

        # 初始化
        self.setContentsMargins(self._margin, self._margin, self._margin, self._margin)
        self._update_formatted_text_and_size()
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

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
        length = len(text)

        if self._is_mostly_english(text):
            width = min(limit, max(4, int(math.sqrt(length * 2.5))))
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

    def _font_with_point_size(self, point_size: int) -> QFont:
        font = QFont(self.font())
        font.setPointSize(point_size)
        font.setBold(True)
        return font

    def _required_content_side(self, point_size: int) -> int:
        font_metrics = QFontMetrics(self._font_with_point_size(point_size))
        required_side = 0
        for text in (self._formatted_true, self._formatted_false):
            if not text:
                continue
            text_rect = font_metrics.boundingRect(
                QRect(0, 0, 10000, 10000),
                Qt.AlignmentFlag.AlignCenter,
                text,
            )
            required_side = max(required_side, text_rect.width(), text_rect.height())
        return math.ceil(required_side / self._TEXT_FILL_RATIO)

    def _outer_size_for_font(self, point_size: int) -> QSize:
        content_side = self._required_content_side(point_size)
        outer_side = content_side + self._margin * 2
        return QSize(outer_side, outer_side)

    def _fitted_font_size(self, content_side: int) -> int:
        low = self._minimum_font_size
        high = self._font_size
        while low < high:
            candidate = (low + high + 1) // 2
            if self._required_content_side(candidate) <= content_side:
                low = candidate
            else:
                high = candidate - 1
        return low

    def _update_formatted_text_and_size(self):
        """更新文字排版，以及预设字号和最小字号对应的尺寸约束。"""
        self._formatted_true = self._smart_wrap_text(self._text_obj.TRUE)
        self._formatted_false = self._smart_wrap_text(self._text_obj.FALSE)
        self.setMinimumSize(self.minimumSizeHint())
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        return self._outer_size_for_font(self._font_size)

    def minimumSizeHint(self) -> QSize:
        return self._outer_size_for_font(self._minimum_font_size)

    # --- 增强：覆盖标准 setFont 方法 ---
    # 这样如果父控件调用了 setFont，我们也能响应
    def setFont(self, font: QFont | str | Sequence[str]):
        if isinstance(font, QFont) and font.pointSize() <= 0 and font.pixelSize() <= 0:
            normalized_font = QFont(font)
            normalized_font.setPointSize(self._minimum_font_size)
            font = normalized_font
        super().setFont(font)
        effective_font = self.font()
        # 提取字号并更新
        if effective_font.pointSize() > 0:
            self._font_size = max(self._minimum_font_size, effective_font.pointSize())
        elif effective_font.pixelSize() > 0:
            # 简单粗暴的转换，实际可能需要 DPI 计算
            self._font_size = max(self._minimum_font_size, int(effective_font.pixelSize() * 0.75))
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

    def _get_status(self) -> bool:
        return self._status

    def _set_status(self, value: bool) -> None:
        if self._status != value:
            self._status = value
            self.statusChanged.emit(value)
            self.update()

    status = Property(bool, _get_status, _set_status)

    def set_status(self, value: bool) -> None:
        if self._status != value:
            self._status = value
            self.statusChanged.emit(value)
            self.update()

    def _get_font_size(self) -> int:
        return self._font_size

    def _set_font_size(self, size: int) -> None:
        self._font_size = max(self._minimum_font_size, size)
        self._update_formatted_text_and_size()

    fontSize = Property(int, _get_font_size, _set_font_size)

    def _get_minimum_font_size(self) -> int:
        return self._minimum_font_size

    def _set_minimum_font_size(self, size: int) -> None:
        self._minimum_font_size = max(1, size)
        self._font_size = max(self._minimum_font_size, self._font_size)
        self._update_formatted_text_and_size()

    minimumFontSize = Property(int, _get_minimum_font_size, _set_minimum_font_size)

    # 支持 QSS: qproperty-margin: 8;
    def _get_margin(self) -> int:
        return self._margin

    def _set_margin(self, value: int) -> None:
        new_margin = max(0, int(value))
        if self._margin == new_margin:
            return
        self._margin = new_margin
        self.setContentsMargins(self._margin, self._margin, self._margin, self._margin)
        self._update_formatted_text_and_size()

    margin = Property(int, _get_margin, _set_margin)

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

        content_rect = self.contentsRect()
        side = min(content_rect.width(), content_rect.height())
        rect = QRect(0, 0, side, side)
        rect.moveCenter(content_rect.center())

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
        font = self._font_with_point_size(self._fitted_font_size(side))
        painter.setFont(font)

        display_text = self._formatted_true if self._status else self._formatted_false

        target_w = int(side * self._TEXT_FILL_RATIO)
        target_h = int(side * self._TEXT_FILL_RATIO)
        draw_rect = QRect(0, 0, target_w, target_h)
        draw_rect.moveCenter(rect.center())

        painter.drawText(draw_rect, Qt.AlignmentFlag.AlignCenter, display_text)


class CasiaMultiStateIndicator(QWidget):
    """可点击的多模式状态指示灯。

    控件只负责按状态键显示文字和颜色，不在内部推断业务状态，也不会自行切换状态。
    点击后由页面决定是否允许切换并调用硬件接口，因此同一控件既可用于可交互的模式、
    电机状态，也可用于只读的机器人运行状态。
    """

    clicked = Signal()
    stateChanged = Signal(str)

    def __init__(
        self,
        caption: str,
        states: Sequence[IndicatorState],
        parent: QWidget | None = None,
        default_state: str = "unknown",
        interactive: bool = True,
    ) -> None:
        super().__init__(parent)
        if not states:
            raise ValueError("多模式指示灯至少需要一个状态")
        self._caption = caption
        self._states = {state.key: state for state in states}
        self._unknown_state = IndicatorState("unknown", "未知", "#607d8b")
        self._state_key = default_state
        self._interactive = interactive
        self._is_pressed = False
        self.setFixedSize(72, 72)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if interactive
            else Qt.CursorShape.ArrowCursor
        )
        self._update_tooltip()

    @property
    def state(self) -> str:
        """返回当前原始状态键。"""

        return self._state_key

    def set_state(self, state_key: str) -> None:
        """按原始状态键刷新显示；未注册状态统一显示为未知。"""

        if self._state_key == state_key:
            return
        self._state_key = state_key
        self._update_tooltip()
        self.stateChanged.emit(state_key)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(72, 72)

    def _update_tooltip(self) -> None:
        state = self._states.get(self._state_key, self._unknown_state)
        self.setToolTip(f"{self._caption}：{state.text}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            self._interactive
            and self.isEnabled()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._is_pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self._interactive or not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            was_pressed = self._is_pressed
            self._is_pressed = False
            self.update()
            if was_pressed and self.rect().contains(event.pos()):
                self.clicked.emit()
        super().mouseReleaseEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.EnabledChange:
            self._is_pressed = False
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if self._interactive and self.isEnabled()
                else Qt.CursorShape.ArrowCursor
            )
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event: QEvent) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        state = self._states.get(self._state_key, self._unknown_state)
        background = QColor(state.color)
        if not self.isEnabled():
            background = QColor("#9e9e9e")
        elif self._is_pressed:
            background = background.darker(115)

        body = self.rect().adjusted(2, 2, -2, -2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawEllipse(body)

        painter.setPen(QColor(state.text_color))
        font = QFont(self.font())
        point_size = font.pointSize()
        font.setPointSize(max(9, point_size if point_size > 0 else 11))
        font.setBold(True)
        painter.setFont(font)
        display_text = state.text
        if len(display_text) == 4:
            display_text = f"{display_text[:2]}\n{display_text[2:]}"
        painter.drawText(
            body.adjusted(6, 6, -6, -6),
            Qt.AlignmentFlag.AlignCenter,
            display_text,
        )


class CasiaToggleSwitch(QWidget):
    """适合触摸操作的开关控件。"""

    toggled = Signal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        checked: bool = False,
        tooltip_prefix: str = "",
    ) -> None:
        super().__init__(parent)
        self._checked = checked
        self._pressed = False
        self._tooltip_prefix = tooltip_prefix
        self.setFixedSize(76, 42)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_tooltip()

    @property
    def checked(self) -> bool:
        return self._checked

    def set_checked(self, checked: bool) -> None:
        if self._checked == checked:
            return
        self._checked = checked
        self._update_tooltip()
        self.update()

    def _update_tooltip(self) -> None:
        state_text = "开启" if self._checked else "关闭"
        prefix = f"{self._tooltip_prefix}：" if self._tooltip_prefix else ""
        self.setToolTip(f"{prefix}{state_text}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if not self.isEnabled():
            return
        if event.button() == Qt.MouseButton.LeftButton:
            was_pressed = self._pressed
            self._pressed = False
            if was_pressed and self.rect().contains(event.pos()):
                self._checked = not self._checked
                self._update_tooltip()
                self.toggled.emit(self._checked)
            self.update()
        super().mouseReleaseEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.EnabledChange:
            self._pressed = False
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if self.isEnabled()
                else Qt.CursorShape.ArrowCursor
            )
            self.update()
        super().changeEvent(event)

    def paintEvent(self, event: QEvent) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self.rect().adjusted(2, 5, -2, -5)
        if not self.isEnabled():
            track_color = QColor("#bdbdbd")
        elif self._checked:
            track_color = QColor("#2e7d32")
        else:
            track_color = QColor("#78909c")
        if self._pressed:
            track_color = track_color.darker(115)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        radius = track.height() / 2
        painter.drawRoundedRect(track, radius, radius)

        knob_size = track.height() - 6
        knob_left = (
            track.right() - knob_size - 3
            if self._checked
            else track.left() + 3
        )
        knob = QRect(knob_left, track.top() + 3, knob_size, knob_size)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(knob)


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
        self.ind_interactive.clicked.connect(lambda: self.ind_interactive.set_status(not self.ind_interactive.status))
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
            self.ind_long_text.set_status(True)  # 强制切到 True 看效果
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
