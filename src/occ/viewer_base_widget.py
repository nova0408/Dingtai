from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from OCC.Core.AIS import AIS_InteractiveContext
from OCC.Core.V3d import V3d_View

from .viewer_core import OCCViewer


# region Qt 宿主控件
class qtBaseViewerWidget(QWidget):
    """承载 OCC 原生窗口句柄的基础 Qt 控件。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.viewer3d = OCCViewer()
        self.context: AIS_InteractiveContext = self.viewer3d.Context
        self.view: V3d_View = self.viewer3d.View

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAutoFillBackground(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewer3d.View.MustBeResized()

    def paintEngine(self):
        return None

    def _to_view_xy(self, x: float, y: float) -> tuple[int, int]:
        # Qt 事件坐标是逻辑像素，这里统一转换为 OCC 期望的视口像素。
        dpr = float(self.devicePixelRatioF())
        return int(round(x * dpr)), int(round(y * dpr))


# endregion
