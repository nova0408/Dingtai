from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from PySide6.QtCore import (
    QAbstractAnimation,
    QByteArray,
    QEasingCurve,
    QEvent,
    QMimeData,
    QObject,
    QPoint,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QDrag, QIcon, QMouseEvent, QPainter, QPaintEvent, QPalette, QPen, QPolygon
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractScrollArea,
    QApplication,
    QFrame,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

QWIDGETSIZE_MAX = 16777215
ANIMATION_DURATION_MS = 140

DEFAULT_FIXED_MIN_HEIGHT = 72
DEFAULT_FIXED_PREFERRED_HEIGHT = 132
DEFAULT_SCROLL_MIN_HEIGHT = 96
DEFAULT_SCROLL_PREFERRED_HEIGHT = 150
DEFAULT_EXPANDING_MIN_HEIGHT = 200
DEFAULT_EXPANDING_PREFERRED_HEIGHT = 280
MAX_AUTO_FIXED_PREFERRED_HEIGHT = 260
MAX_AUTO_SCROLL_PREFERRED_HEIGHT = 190
MAX_AUTO_EXPANDING_PREFERRED_HEIGHT = 420


def clamp(value: int, min_value: int, max_value: int) -> int:
    """限制整数范围。"""
    return max(min_value, min(value, max_value))


class SectionContentMode(Enum):
    """Section 内容高度策略。"""

    AUTO = auto()
    FIXED_PANEL = auto()
    SCROLL_CONTENT = auto()
    EXPANDING_PANEL = auto()


@dataclass
class SectionState:
    """单个分组项的运行状态。"""

    expanded: bool = True
    visible: bool = True
    preferred_height: int = DEFAULT_FIXED_PREFERRED_HEIGHT
    min_height: int = DEFAULT_FIXED_MIN_HEIGHT
    animated_height: int = DEFAULT_FIXED_PREFERRED_HEIGHT
    user_resized: bool = False


class ToolBoxHeader(QAbstractButton):
    """工具箱标题栏。"""

    toggle_requested = Signal()
    drag_requested = Signal(QPoint)
    context_menu_requested = Signal(QPoint)

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

    def __init__(self, title: str, icon: QIcon | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('toolBoxHeader')
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._touch_scale = self._get_touch_scale()
        self.setMinimumHeight(self._scaled(44))

        self._expanded = True
        self._active = False
        self._drag_start_pos = QPoint()
        self._drag_started = False

        self.setText(title)
        self.setIcon(icon if icon is not None else QIcon())
        self.customContextMenuRequested.connect(lambda pos: self.context_menu_requested.emit(self.mapToGlobal(pos)))

    def set_expanded(self, expanded: bool) -> None:
        if self._expanded == expanded:
            return
        self._expanded = expanded
        self.update()

    def set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self.update()

    def is_active(self) -> bool:
        return self._active

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text())
        icon_w = 16 if not self.icon().isNull() else 0
        width = 12 + 12 + 8 + icon_w + (6 if icon_w else 0) + text_w + 12
        height = max(self._scaled(44), fm.height() + self._scaled(18))
        return QSize(width, height)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        pal = self.palette()
        base_color = pal.color(QPalette.ColorRole.Button)
        text_color = pal.color(QPalette.ColorRole.ButtonText)
        border_color = pal.color(QPalette.ColorRole.Mid)
        light_color = pal.color(QPalette.ColorRole.Light)
        highlight_color = pal.color(QPalette.ColorRole.Highlight)

        fill_color = base_color
        if self.underMouse():
            fill_color = fill_color.lighter(103)
        if self.isDown():
            fill_color = fill_color.darker(104)

        painter.fillRect(rect, fill_color)
        if self._active:
            painter.fillRect(QRect(rect.left(), rect.top(), 3, rect.height() + 1), highlight_color)

        painter.setPen(border_color)
        painter.drawLine(rect.topLeft(), rect.topRight())
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        painter.drawLine(rect.topRight(), rect.bottomRight())
        painter.setPen(light_color if self._expanded else border_color)
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        left = 8
        arrow_rect = QRect(left, rect.center().y() - 5, 10, 10)
        self._draw_arrow(painter, arrow_rect, self._expanded, text_color)
        left = arrow_rect.right() + 8

        if not self.icon().isNull():
            icon_rect = QRect(left, rect.center().y() - 8, 16, 16)
            self.icon().paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter, QIcon.Mode.Normal, QIcon.State.Off)
            left = icon_rect.right() + 6

        painter.setPen(text_color)
        text_rect = QRect(left, rect.top(), rect.width() - left - 8, rect.height())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.text())

    def _draw_arrow(self, painter: QPainter, rect: QRect, expanded: bool, color) -> None:
        cx = rect.center().x()
        cy = rect.center().y()
        if expanded:
            points = QPolygon([QPoint(cx - 4, cy - 2), QPoint(cx + 4, cy - 2), QPoint(cx, cy + 3)])
        else:
            points = QPolygon([QPoint(cx - 2, cy - 4), QPoint(cx - 2, cy + 4), QPoint(cx + 3, cy)])
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(points)
        painter.restore()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.globalPosition().toPoint()
            self._drag_started = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            dist = (event.globalPosition().toPoint() - self._drag_start_pos).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._drag_started = True
                self.setDown(False)
                self.drag_requested.emit(self._drag_start_pos)
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_started:
            self._drag_started = False
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.toggle_requested.emit()
        super().mouseReleaseEvent(event)


class ToolBoxContentFrame(QWidget):
    """内容区容器。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('toolBoxContentFrame')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(0)

    def set_content_widget(self, widget: QWidget) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            old_widget = item.widget()
            if old_widget is not None:
                old_widget.setParent(None)
        self._layout.addWidget(widget)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Base))


class ToolBoxResizeHandle(QWidget):
    """内容高度拖拽条。"""

    drag_started = Signal()
    drag_delta = Signal(int)
    drag_finished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('toolBoxResizeHandle')
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setFixedHeight(6)
        self._pressed = False
        self._last_global_y = 0

    def sizeHint(self) -> QSize:
        return QSize(100, 6)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        rect = self.rect()
        mid = self.palette().color(QPalette.ColorRole.Mid)
        light = self.palette().color(QPalette.ColorRole.Light)
        dark = self.palette().color(QPalette.ColorRole.Dark)
        y1 = rect.center().y() - 1
        y2 = rect.center().y()
        painter.setPen(light)
        painter.drawLine(8, y1, rect.width() - 8, y1)
        painter.setPen(dark if self._pressed else mid)
        painter.drawLine(8, y2, rect.width() - 8, y2)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self._last_global_y = event.globalPosition().toPoint().y()
            self.drag_started.emit()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._pressed:
            curr_y = event.globalPosition().toPoint().y()
            delta = curr_y - self._last_global_y
            self._last_global_y = curr_y
            self.drag_delta.emit(delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._pressed and event.button() == Qt.MouseButton.LeftButton:
            self._pressed = False
            self.drag_finished.emit()
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ToolBoxSection(QWidget):
    """单个分组 Section。"""

    toggle_requested = Signal(QObject)
    context_menu_requested = Signal(QObject, QPoint)
    drag_requested = Signal(QObject, QPoint)
    activated = Signal(QObject)
    layout_state_changed = Signal()

    def __init__(
        self,
        title: str,
        widget: QWidget,
        icon: QIcon | None = None,
        parent: QWidget | None = None,
        *,
        content_mode: SectionContentMode = SectionContentMode.AUTO,
        preferred_height: int | None = None,
        min_height: int | None = None,
        resizable: bool = True,
        grow_priority: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setObjectName('toolBoxSection')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._content_widget = widget
        self._content_mode = content_mode
        self._preferred_height_override = preferred_height
        self._min_height_override = min_height
        self._resizable = resizable
        self._grow_priority = max(0, int(grow_priority))
        self._sync_pending = False
        self.state = SectionState()

        self.header = ToolBoxHeader(title, icon, self)
        self.content_frame = ToolBoxContentFrame(self)
        self.resize_handle = ToolBoxResizeHandle(self)
        self.content_frame.set_content_widget(widget)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(1, 1, 1, 1)
        self._layout.setSpacing(0)
        self._layout.addWidget(self.header)
        self._layout.addWidget(self.content_frame)
        self._layout.addWidget(self.resize_handle)

        self._animation = QPropertyAnimation(self, b'dummy', self)
        self._animation.setDuration(ANIMATION_DURATION_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.valueChanged.connect(self._on_animation_value_changed)
        self._animation.finished.connect(self._on_animation_finished)

        self._install_content_observers()
        self._init_height_state()
        self._connect_signals()
        self._sync_visual_state()

    def _resolve_auto_mode(self) -> SectionContentMode:
        if self._content_mode is not SectionContentMode.AUTO:
            return self._content_mode
        if isinstance(self._content_widget, QAbstractScrollArea):
            return SectionContentMode.SCROLL_CONTENT
        size_policy = self._content_widget.sizePolicy().verticalPolicy()
        if size_policy in {QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Ignored}:
            return SectionContentMode.EXPANDING_PANEL
        return SectionContentMode.FIXED_PANEL


    def content_mode(self) -> SectionContentMode:
        return self._resolve_auto_mode()

    def grow_priority(self) -> int:
        return self._grow_priority

    def _install_content_observers(self) -> None:
        self._content_widget.installEventFilter(self)
        layout = self._content_widget.layout()
        if layout is not None:
            layout.installEventFilter(self)

    def _connect_signals(self) -> None:
        self.header.toggle_requested.connect(lambda: self.toggle_requested.emit(self))
        self.header.context_menu_requested.connect(lambda pos: self.context_menu_requested.emit(self, pos))
        self.header.drag_requested.connect(lambda pos: self.drag_requested.emit(self, pos))
        self.header.pressed.connect(lambda: self.activated.emit(self))
        self.resize_handle.drag_started.connect(lambda: self.activated.emit(self))
        self.resize_handle.drag_delta.connect(self._on_resize_delta)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        event_type = event.type()
        layout = self._content_widget.layout()
        if watched in {self._content_widget, layout} and event_type in {
            QEvent.Type.LayoutRequest,
            QEvent.Type.Show,
            QEvent.Type.ShowToParent,
            QEvent.Type.Resize,
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
            QEvent.Type.PolishRequest,
        }:
            self._schedule_sync_height_profile()
        return super().eventFilter(watched, event)

    def _schedule_sync_height_profile(self) -> None:
        if self._sync_pending:
            return
        self._sync_pending = True
        QTimer.singleShot(0, self._flush_sync_height_profile)

    def _flush_sync_height_profile(self) -> None:
        self._sync_pending = False
        self._sync_height_profile_from_content()

    def _safe_min_hint(self) -> int:
        widget = self._content_widget
        value = max(widget.minimumSizeHint().height(), widget.minimumHeight())
        return max(0, value)

    def _safe_size_hint(self) -> int:
        widget = self._content_widget
        value = max(widget.sizeHint().height(), widget.minimumSizeHint().height(), widget.minimumHeight())
        return max(0, value)

    def _compute_content_height_profile(self) -> tuple[int, int]:
        mode = self._resolve_auto_mode()
        min_hint = self._safe_min_hint()
        size_hint = self._safe_size_hint()

        if mode is SectionContentMode.FIXED_PANEL:
            min_h = max(DEFAULT_FIXED_MIN_HEIGHT, min_hint)
            preferred_h = min(max(DEFAULT_FIXED_PREFERRED_HEIGHT, size_hint, min_h), MAX_AUTO_FIXED_PREFERRED_HEIGHT)
        elif mode is SectionContentMode.SCROLL_CONTENT:
            min_h = max(DEFAULT_SCROLL_MIN_HEIGHT, min(min_hint if min_hint > 0 else DEFAULT_SCROLL_MIN_HEIGHT, 140))
            preferred_h = min(max(DEFAULT_SCROLL_PREFERRED_HEIGHT, min_h), MAX_AUTO_SCROLL_PREFERRED_HEIGHT)
        else:
            min_h = max(DEFAULT_EXPANDING_MIN_HEIGHT, min_hint)
            preferred_h = min(max(DEFAULT_EXPANDING_PREFERRED_HEIGHT, size_hint, min_h), MAX_AUTO_EXPANDING_PREFERRED_HEIGHT)

        if self._min_height_override is not None:
            min_h = max(0, self._min_height_override)
        if self._preferred_height_override is not None:
            preferred_h = max(min_h, self._preferred_height_override)
        else:
            preferred_h = max(min_h, preferred_h)
        return min_h, preferred_h

    def _init_height_state(self) -> None:
        min_h, preferred_h = self._compute_content_height_profile()
        self.state.min_height = min_h
        self.state.preferred_height = preferred_h
        self.state.animated_height = preferred_h
        self._apply_content_height(preferred_h if self.state.expanded and self.state.visible else 0)

    def _sync_height_profile_from_content(self) -> None:
        min_h, auto_preferred_h = self._compute_content_height_profile()
        self.state.min_height = min_h
        if self.state.user_resized:
            self.state.preferred_height = max(self.state.preferred_height, min_h)
        else:
            self.state.preferred_height = auto_preferred_h
        if self.state.expanded and self.state.visible and self._animation.state() != QAbstractAnimation.State.Running:
            stable_h = self._stable_content_height()
            self.state.animated_height = stable_h
            self._apply_stable_constraints()
        self.updateGeometry()
        self.layout_state_changed.emit()

    def _stable_content_height(self) -> int:
        return max(self.state.preferred_height, self.state.min_height)

    def _apply_fixed_content_height(self, height: int) -> None:
        height = max(0, int(height))
        self.content_frame.setMinimumHeight(height)
        self.content_frame.setMaximumHeight(height)

    def _apply_stable_constraints(self) -> None:
        mode = self._resolve_auto_mode()
        stable_h = self._stable_content_height()
        if mode is SectionContentMode.FIXED_PANEL:
            self.content_frame.setMinimumHeight(stable_h)
            self.content_frame.setMaximumHeight(QWIDGETSIZE_MAX)
        elif mode is SectionContentMode.SCROLL_CONTENT:
            self.content_frame.setMinimumHeight(self.state.min_height)
            self.content_frame.setMaximumHeight(stable_h)
        else:
            self.content_frame.setMinimumHeight(self.state.min_height)
            self.content_frame.setMaximumHeight(QWIDGETSIZE_MAX)

    def _apply_content_height(self, height: int) -> None:
        if self._animation.state() == QAbstractAnimation.State.Running:
            self._apply_fixed_content_height(height)
            return
        if not self.state.visible or not self.state.expanded:
            self._apply_fixed_content_height(0)
            return
        self._apply_stable_constraints()

    def _current_visible_content_height(self) -> int:
        if not self.state.visible or not self.state.expanded:
            return 0
        if self._animation.state() == QAbstractAnimation.State.Running:
            return max(0, self.state.animated_height)
        return max(self.state.min_height, self.content_frame.height(), min(self._stable_content_height(), self.content_frame.maximumHeight()))

    def title(self) -> str:
        return self.header.text()

    def set_title(self, title: str) -> None:
        self.header.setText(title)
        self.header.updateGeometry()
        self.header.update()
        self.updateGeometry()

    def icon(self) -> QIcon:
        return self.header.icon()

    def set_icon(self, icon: QIcon) -> None:
        self.header.setIcon(icon)
        self.header.update()

    def content_widget(self) -> QWidget:
        return self._content_widget

    def is_expanded(self) -> bool:
        return self.state.expanded

    def is_visible_section(self) -> bool:
        return self.state.visible

    def set_active(self, active: bool) -> None:
        self.header.set_active(active)

    def is_active(self) -> bool:
        return self.header.is_active()

    def _sync_visual_state(self) -> None:
        self.header.set_expanded(self.state.expanded)
        content_visible = self.state.visible and (self.state.expanded or self._animation.state() == QAbstractAnimation.State.Running)
        self.content_frame.setVisible(content_visible)
        self.resize_handle.setVisible(self._resizable and self.state.visible and self.state.expanded)

    def _on_animation_value_changed(self, value) -> None:
        height = int(value)
        self.state.animated_height = max(0, height)
        self._apply_fixed_content_height(self.state.animated_height)
        self.updateGeometry()
        self.layout_state_changed.emit()

    def _on_animation_finished(self) -> None:
        if self.state.expanded and self.state.visible:
            stable_h = self._stable_content_height()
            self.state.animated_height = stable_h
            self._apply_stable_constraints()
            self.content_frame.show()
        else:
            self.state.animated_height = 0
            self._apply_fixed_content_height(0)
            self.content_frame.hide()
        self._sync_visual_state()
        self.updateGeometry()
        self.layout_state_changed.emit()

    def expand(self, animated: bool = True) -> None:
        if self.state.expanded:
            return
        self.state.expanded = True
        self.header.set_expanded(True)
        self.content_frame.show()
        self.resize_handle.show()
        target_h = self._stable_content_height()
        self._animation.stop()
        if animated:
            self.state.animated_height = 0
            self._apply_fixed_content_height(0)
            self._animation.setStartValue(0)
            self._animation.setEndValue(target_h)
            self._animation.start()
        else:
            self.state.animated_height = target_h
            self._apply_stable_constraints()
            self._sync_visual_state()
            self.updateGeometry()
            self.layout_state_changed.emit()

    def collapse(self, animated: bool = True) -> None:
        if not self.state.expanded:
            return
        current_h = self._current_visible_content_height()
        self.state.expanded = False
        self.header.set_expanded(False)
        self.resize_handle.hide()
        self._animation.stop()
        if animated:
            self.state.animated_height = current_h
            self._apply_fixed_content_height(current_h)
            self._animation.setStartValue(current_h)
            self._animation.setEndValue(0)
            self._animation.start()
        else:
            self.state.animated_height = 0
            self._apply_fixed_content_height(0)
            self.content_frame.hide()
            self._sync_visual_state()
            self.updateGeometry()
            self.layout_state_changed.emit()

    def toggle(self, animated: bool = True) -> None:
        if self.state.expanded:
            self.collapse(animated=animated)
        else:
            self.expand(animated=animated)

    def set_section_visible(self, visible: bool) -> None:
        self.state.visible = visible
        self.setVisible(visible)
        if visible and self.state.expanded:
            stable_h = self._stable_content_height()
            self.state.animated_height = stable_h
            self._apply_content_height(stable_h)
        else:
            self.state.animated_height = 0
            self._apply_fixed_content_height(0)
        self._sync_visual_state()
        self.layout_state_changed.emit()

    def _on_resize_delta(self, delta: int) -> None:
        if not self.state.expanded or not self._resizable:
            return
        self._animation.stop()
        base_h = self._current_visible_content_height()
        new_h = max(self.state.min_height, base_h + delta)
        self.state.user_resized = True
        self.state.preferred_height = new_h
        self.state.animated_height = new_h
        self._apply_stable_constraints()
        self.updateGeometry()
        self.layout_state_changed.emit()

    def reset_preferred_height(self) -> None:
        self.state.user_resized = False
        min_h, preferred_h = self._compute_content_height_profile()
        self.state.min_height = min_h
        self.state.preferred_height = preferred_h
        self.state.animated_height = preferred_h if self.state.expanded and self.state.visible else 0
        if self.state.expanded and self.state.visible:
            self._apply_stable_constraints()
        self.updateGeometry()
        self.layout_state_changed.emit()

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        border = self.palette().color(QPalette.ColorRole.Mid)
        background = self.palette().color(QPalette.ColorRole.Window)
        painter.fillRect(rect, background)
        painter.setPen(border)
        painter.drawRect(rect)

    def sizeHint(self) -> QSize:
        header_h = self.header.sizeHint().height()
        handle_h = self.resize_handle.sizeHint().height() if self._resizable and self.state.visible and self.state.expanded else 0
        content_h = self._current_visible_content_height()
        margins = self._layout.contentsMargins()
        return QSize(260, header_h + handle_h + content_h + margins.top() + margins.bottom())

    def minimumSizeHint(self) -> QSize:
        header_h = self.header.sizeHint().height()
        handle_h = self.resize_handle.sizeHint().height() if self._resizable and self.state.visible and self.state.expanded else 0
        content_h = self.state.min_height if self.state.visible and self.state.expanded else 0
        margins = self._layout.contentsMargins()
        return QSize(180, header_h + handle_h + content_h + margins.top() + margins.bottom())


class ToolBoxDropIndicator(QWidget):
    """拖拽排序插入线。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('toolBoxDropIndicator')
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

    def sizeHint(self) -> QSize:
        return QSize(100, 2)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        pen = QPen(self.palette().color(QPalette.ColorRole.Highlight))
        pen.setWidth(2)
        painter.setPen(pen)
        y = self.height() // 2
        painter.drawLine(0, y, self.width(), y)


class AdvancedToolBox(QWidget):
    """高级工具箱组件。"""

    current_section_changed = Signal(int)
    order_changed = Signal()
    visibility_changed = Signal(int, bool)
    toggled = Signal(int, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('advancedToolBox')
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._sections: list[ToolBoxSection] = []
        self._current_index = -1
        self._sync_pending = False
        self._drag_source_index = -1
        self._drop_target_index = -1
        self._drop_insert_after = False

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(0, 0, 0, 0)
        self._root_layout.setSpacing(0)

        self._scroll_area = QScrollArea(self)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._root_layout.addWidget(self._scroll_area)

        self._container = QWidget()
        self._container.setObjectName('advancedToolBoxContainer')
        self._container.setAcceptDrops(True)
        self._container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(6)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._bottom_spacer = QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        self._layout.addSpacerItem(self._bottom_spacer)
        self._scroll_area.setWidget(self._container)
        self._drop_indicator = ToolBoxDropIndicator(self._container)
        self._container.installEventFilter(self)
        self._scroll_area.viewport().installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched in {self._container, self._scroll_area.viewport()} and event.type() == QEvent.Type.Resize:
            self._schedule_layout_sync()
        return super().eventFilter(watched, event)

    def _schedule_layout_sync(self) -> None:
        if self._sync_pending:
            return
        self._sync_pending = True
        QTimer.singleShot(0, self._flush_layout_sync)

    def _flush_layout_sync(self) -> None:
        self._sync_pending = False
        self._update_section_stretches()
        self._container.adjustSize()
        self.updateGeometry()
        self.update()

    def _update_section_stretches(self) -> None:
        for i, section in enumerate(self._sections):
            stretch = 0
            if section.isVisible() and section.is_expanded() and section.content_mode() is SectionContentMode.EXPANDING_PANEL:
                stretch = max(1, section.grow_priority())
            self._layout.setStretch(i, stretch)

    def count(self) -> int:
        return len(self._sections)

    def current_index(self) -> int:
        return self._current_index

    def section_at(self, index: int) -> ToolBoxSection:
        return self._sections[index]

    def widget(self, index: int) -> QWidget:
        return self._sections[index].content_widget()

    def title(self, index: int) -> str:
        return self._sections[index].title()

    def set_title(self, index: int, title: str) -> None:
        self._sections[index].set_title(title)
        self._schedule_layout_sync()

    def set_icon(self, index: int, icon: QIcon) -> None:
        self._sections[index].set_icon(icon)

    def _section_insert_layout_index(self, logical_index: int) -> int:
        return clamp(logical_index, 0, len(self._sections))

    def add_widget(
        self,
        widget: QWidget,
        title: str,
        icon: QIcon | None = None,
        expanded: bool = True,
        *,
        content_mode: SectionContentMode = SectionContentMode.AUTO,
        preferred_height: int | None = None,
        min_height: int | None = None,
        resizable: bool | None = None,
        grow_priority: int = 0,
    ) -> int:
        if resizable is None:
            resizable = content_mode in {SectionContentMode.SCROLL_CONTENT, SectionContentMode.EXPANDING_PANEL}
        section = ToolBoxSection(
            title,
            widget,
            icon,
            self._container,
            content_mode=content_mode,
            preferred_height=preferred_height,
            min_height=min_height,
            resizable=resizable,
            grow_priority=grow_priority,
        )
        self._connect_section_signals(section)
        insert_index = self._section_insert_layout_index(len(self._sections))
        self._layout.insertWidget(insert_index, section)
        self._sections.append(section)
        if not expanded:
            section.collapse(animated=False)
        if self._current_index < 0:
            self.set_current_index(0)
        self._schedule_layout_sync()
        return len(self._sections) - 1

    def remove_widget(self, index: int) -> None:
        if not (0 <= index < len(self._sections)):
            return
        section = self._sections.pop(index)
        self._layout.removeWidget(section)
        section.deleteLater()
        if self._current_index >= len(self._sections):
            self._current_index = len(self._sections) - 1
        self._refresh_active_states()
        self._schedule_layout_sync()

    def clear(self) -> None:
        while self._sections:
            self.remove_widget(len(self._sections) - 1)

    def set_current_index(self, index: int) -> None:
        if not (0 <= index < len(self._sections)):
            self._current_index = -1
            self._refresh_active_states()
            return
        if self._current_index != index:
            self._current_index = index
            self._refresh_active_states()
            self.current_section_changed.emit(index)
        else:
            self._refresh_active_states()

    def _refresh_active_states(self) -> None:
        for i, section in enumerate(self._sections):
            section.set_active(i == self._current_index)

    def set_section_visible(self, index: int, visible: bool) -> None:
        if not (0 <= index < len(self._sections)):
            return
        self._sections[index].set_section_visible(visible)
        self.visibility_changed.emit(index, visible)
        if index == self._current_index and not visible:
            self._select_first_visible_section()
        self._schedule_layout_sync()

    def is_section_visible(self, index: int) -> bool:
        if not (0 <= index < len(self._sections)):
            return False
        return self._sections[index].is_visible_section()

    def _select_first_visible_section(self) -> None:
        for i, section in enumerate(self._sections):
            if section.isVisible():
                self.set_current_index(i)
                return
        self.set_current_index(-1)

    def expand(self, index: int, animated: bool = True) -> None:
        if not (0 <= index < len(self._sections)):
            return
        self._sections[index].expand(animated=animated)
        self.toggled.emit(index, True)
        self._schedule_layout_sync()

    def collapse(self, index: int, animated: bool = True) -> None:
        if not (0 <= index < len(self._sections)):
            return
        self._sections[index].collapse(animated=animated)
        self.toggled.emit(index, False)
        self._schedule_layout_sync()

    def toggle(self, index: int, animated: bool = True) -> None:
        if not (0 <= index < len(self._sections)):
            return
        section = self._sections[index]
        section.toggle(animated=animated)
        self.toggled.emit(index, section.is_expanded())
        self._schedule_layout_sync()

    def expand_all(self, animated: bool = False) -> None:
        for i in range(len(self._sections)):
            self.expand(i, animated=animated)

    def collapse_all(self, animated: bool = False) -> None:
        for i in range(len(self._sections)):
            self.collapse(i, animated=animated)

    def _connect_section_signals(self, section: ToolBoxSection) -> None:
        section.toggle_requested.connect(self._on_section_toggle_requested)
        section.context_menu_requested.connect(self._on_section_context_menu_requested)
        section.drag_requested.connect(self._on_section_drag_requested)
        section.activated.connect(self._on_section_activated)
        section.layout_state_changed.connect(self._on_section_layout_state_changed)

    def _on_section_layout_state_changed(self) -> None:
        self._schedule_layout_sync()

    def _on_section_toggle_requested(self, section_obj: QObject) -> None:
        section = self._as_section(section_obj)
        index = self._index_of_section(section)
        if index < 0:
            return
        self.set_current_index(index)
        self.toggle(index, animated=True)

    def _on_section_context_menu_requested(self, section_obj: QObject, global_pos: QPoint) -> None:
        section = self._as_section(section_obj)
        index = self._index_of_section(section)
        if index >= 0:
            self.set_current_index(index)
        self._show_context_menu(global_pos)

    def _on_section_drag_requested(self, section_obj: QObject, global_pos: QPoint) -> None:
        section = self._as_section(section_obj)
        index = self._index_of_section(section)
        if index < 0:
            return
        self.set_current_index(index)
        self._start_drag(index, global_pos)

    def _on_section_activated(self, section_obj: QObject) -> None:
        section = self._as_section(section_obj)
        index = self._index_of_section(section)
        if index >= 0:
            self.set_current_index(index)

    def _as_section(self, obj: QObject) -> ToolBoxSection:
        assert isinstance(obj, ToolBoxSection)
        return obj

    def _index_of_section(self, section: ToolBoxSection) -> int:
        try:
            return self._sections.index(section)
        except ValueError:
            return -1

    def _visible_sections(self) -> list[ToolBoxSection]:
        return [section for section in self._sections if section.isVisible()]

    def _show_context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        act_expand_all = QAction('全部展开', menu)
        act_collapse_all = QAction('全部收起', menu)
        act_expand_all.triggered.connect(lambda: self.expand_all(animated=False))
        act_collapse_all.triggered.connect(lambda: self.collapse_all(animated=False))
        menu.addAction(act_expand_all)
        menu.addAction(act_collapse_all)
        menu.addSeparator()
        for i, section in enumerate(self._sections):
            action = QAction(section.title(), menu)
            action.setCheckable(True)
            action.setChecked(section.is_section_visible())
            action.triggered.connect(lambda checked, idx=i: self.set_section_visible(idx, checked))
            menu.addAction(action)
        menu.exec(global_pos)

    def _start_drag(self, src_index: int, global_pos: QPoint) -> None:
        self._drag_source_index = src_index
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData('application/x-advanced-toolbox-index', QByteArray(str(src_index).encode('utf-8')))
        drag.setMimeData(mime)
        pixmap = self._sections[src_index].header.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(self._sections[src_index].header.mapFromGlobal(global_pos))
        drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat('application/x-advanced-toolbox-index'):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if not event.mimeData().hasFormat('application/x-advanced-toolbox-index'):
            super().dragMoveEvent(event)
            return
        pos = self._container.mapFrom(self, event.position().toPoint())
        result = self._calc_drop_position(pos)
        if result is None:
            self._drop_indicator.hide()
            super().dragMoveEvent(event)
            return
        target_index, insert_after, y = result
        self._drop_target_index = target_index
        self._drop_insert_after = insert_after
        self._drop_indicator.setGeometry(4, y - 1, max(20, self._container.width() - 8), 2)
        self._drop_indicator.show()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event: QEvent) -> None:
        self._drop_indicator.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        self._drop_indicator.hide()
        if not event.mimeData().hasFormat('application/x-advanced-toolbox-index'):
            super().dropEvent(event)
            return
        try:
            src_index = int(bytes(event.mimeData().data('application/x-advanced-toolbox-index')).decode('utf-8'))
        except ValueError:
            super().dropEvent(event)
            return
        if self._drop_target_index < 0 or not (0 <= src_index < len(self._sections)):
            super().dropEvent(event)
            return
        dest_index = self._drop_target_index + (1 if self._drop_insert_after else 0)
        if dest_index > src_index:
            dest_index -= 1
        if dest_index == src_index:
            event.acceptProposedAction()
            return
        self._reorder_sections(src_index, dest_index)
        event.acceptProposedAction()

    def _calc_drop_position(self, pos: QPoint) -> tuple[int, bool, int] | None:
        for i, section in enumerate(self._sections):
            if not section.isVisible():
                continue
            section_rect = section.geometry()
            header_rect = section.header.geometry().translated(section.pos())
            if header_rect.contains(pos):
                mid_y = header_rect.center().y()
                insert_after = pos.y() > mid_y
                line_y = header_rect.bottom() + 1 if insert_after else header_rect.top()
                return i, insert_after, line_y
            if section_rect.contains(pos):
                mid_y = section_rect.center().y()
                insert_after = pos.y() > mid_y
                line_y = section_rect.bottom() + 1 if insert_after else section_rect.top()
                return i, insert_after, line_y
        visible_sections = [(i, s) for i, s in enumerate(self._sections) if s.isVisible()]
        if visible_sections:
            last_index, last_section = visible_sections[-1]
            if pos.y() > last_section.geometry().bottom():
                return last_index, True, last_section.geometry().bottom() + 1
        return None

    def _reorder_sections(self, src_index: int, dest_index: int) -> None:
        if not (0 <= src_index < len(self._sections)):
            return
        dest_index = clamp(dest_index, 0, len(self._sections) - 1)
        section = self._sections.pop(src_index)
        self._layout.removeWidget(section)
        self._sections.insert(dest_index, section)
        self._layout.insertWidget(self._section_insert_layout_index(dest_index), section)
        if self._current_index == src_index:
            self._current_index = dest_index
        else:
            if src_index < self._current_index <= dest_index:
                self._current_index -= 1
            elif dest_index <= self._current_index < src_index:
                self._current_index += 1
        self._refresh_active_states()
        self.order_changed.emit()
        self._schedule_layout_sync()

    def sizeHint(self) -> QSize:
        margins = self._layout.contentsMargins()
        total_h = margins.top() + margins.bottom()
        spacing = self._layout.spacing()
        visible_sections = self._visible_sections()
        for i, section in enumerate(visible_sections):
            total_h += section.sizeHint().height()
            if i > 0:
                total_h += spacing
        scroll_hint = self._scroll_area.sizeHint()
        return QSize(max(280, scroll_hint.width()), max(200, min(total_h, 640)))

    def minimumSizeHint(self) -> QSize:
        margins = self._layout.contentsMargins()
        total_h = margins.top() + margins.bottom()
        spacing = self._layout.spacing()
        visible_sections = self._visible_sections()
        for i, section in enumerate(visible_sections):
            total_h += section.minimumSizeHint().height()
            if i > 0:
                total_h += spacing
        return QSize(180, min(max(120, total_h), 320))
