from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtGui
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtWidgets import QApplication, QRubberBand, QStyleFactory

from OCC.Core.AIS import AIS_ViewCube
from OCC.Core.Aspect import Aspect_GradientFillMethod, Aspect_TypeOfTriedronPosition
from OCC.Core.Graphic3d import (
    Graphic3d_AspectLine3d,
    Graphic3d_GraduatedTrihedron,
    Graphic3d_TransformPers,
    Graphic3d_TransModeFlags,
    Graphic3d_Vec2i,
)
from OCC.Core.Prs3d import (
    Prs3d_DatumAspect,
    Prs3d_DP_XAxis,
    Prs3d_DP_YAxis,
    Prs3d_DP_ZAxis,
    Prs3d_LineAspect,
)
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.V3d import V3d_Xneg, V3d_Xpos, V3d_Yneg, V3d_Ypos, V3d_Zneg, V3d_Zpos

from .tools import (
    X_AXIS_COLOR,
    Y_AXIS_COLOR,
    Z_AXIS_COLOR,
    RGB_to_Quantity_Color,
    createTrihedron,
)
from .viewer_base_widget import qtBaseViewerWidget


# region 高层 3D 交互控件
class qtViewer3dWidget(qtBaseViewerWidget):
    signal_AISs_selected = Signal(list)

    def __init__(
        self,
        parent=None,
        view_trihedron: bool = False,
        origin_trihedron: bool = False,
        view_cube: bool = True,
        bg_color_aspect=(
            (40, 40, 40),
            (150, 150, 150),
            Aspect_GradientFillMethod.Aspect_GradientFillMethod_Vertical,
        ),
        selection_color: tuple[int, int, int] = (13, 141, 255),
        enable_multiply_select: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("qt_viewer_3d")

        self._rubber_band: QRubberBand | None = None
        self.enable_multiply_select = enable_multiply_select
        self._view_trihedron = view_trihedron
        self._origin_trihedron = origin_trihedron
        self._view_cube = view_cube
        self._bg_gradient_color = bg_color_aspect
        self._selection_color = selection_color

        self._zoom_area = False
        self._select_area = False
        self._inited = False

        self._drag_start_logical_x = 0
        self._drag_start_logical_y = 0
        self._drag_start_view_x = 0
        self._drag_start_view_y = 0
        self._draw_box_logical: list[int] = []

        self._qApp = QApplication.instance()
        self._key_map: dict[int, Callable] = {}
        self._current_cursor = "arrow"
        self._available_cursors: dict[str, QtGui.QCursor] = {}

    @property
    def qApp(self):
        return self._qApp

    # region 初始化与渲染装配
    def InitDriver(self) -> None:
        self.viewer3d.Create(
            window_handle=int(self.winId()),
            parent=self,
            display_glinfo=False,
            draw_face_boundaries=False,
        )
        self.viewer3d.SetModeShaded()
        self.viewer3d.EnableAntiAliasing()
        self._inited = True

        self._key_map = {
            ord("F"): self.viewer3d.FitAll,
            ord("G"): self.viewer3d.change_selection_mode,
            ord("H"): self.viewer3d.SetModeHLR,
            ord("S"): self.viewer3d.SetModeShaded,
            ord("W"): self.viewer3d.SetModeWireFrame,
        }

        self.create_cursors()
        self.viewer3d.set_selection_color(1, RGB_to_Quantity_Color(self._selection_color))

        if self._view_trihedron:
            self.display_view_trihedron()
        if self._origin_trihedron:
            self.display_origin_trihedron()
        if self._view_cube:
            self.display_view_cube()

        self.viewer3d.View.SetBgGradientColors(
            RGB_to_Quantity_Color(self._bg_gradient_color[0]),
            RGB_to_Quantity_Color(self._bg_gradient_color[1]),
            self._bg_gradient_color[2],
            True,
        )
        self.viewer3d.SetRasterizationMode()

    # endregion

    # region Qt 事件
    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        code = event.key()
        if event.modifiers() == Qt.KeyboardModifier.NoModifier and code in self._key_map:
            self._key_map[code]()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self._inited:
            self.viewer3d.Repaint()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self._inited:
            self.viewer3d.Repaint()

    def paintEvent(self, event):
        if not self._inited:
            self.InitDriver()
        self.viewer3d.Context.UpdateCurrentViewer()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            zoom_factor = 1.1 if event.modifiers() == Qt.KeyboardModifier.ShiftModifier else 2.0
        else:
            zoom_factor = 0.9 if event.modifiers() == Qt.KeyboardModifier.ShiftModifier else 0.5
        self.viewer3d.set_zoom_factor(zoom_factor)

    def mousePressEvent(self, event):
        self.setFocus()
        logical_pt = event.position().toPoint()
        view_x, view_y = self._to_view_xy(logical_pt.x(), logical_pt.y())

        self._drag_start_logical_x = logical_pt.x()
        self._drag_start_logical_y = logical_pt.y()
        self._drag_start_view_x = view_x
        self._drag_start_view_y = view_y

        self.viewer3d.start_rotation(view_x, view_y)

    def mouseMoveEvent(self, event):
        logical_pt = event.position().toPoint()
        view_x, view_y = self._to_view_xy(logical_pt.x(), logical_pt.y())
        buttons = event.buttons()
        modifiers = event.modifiers()

        if buttons == Qt.MouseButton.MiddleButton and modifiers == Qt.KeyboardModifier.ControlModifier:
            dx = view_x - self._drag_start_view_x
            dy = view_y - self._drag_start_view_y
            self._drag_start_view_x = view_x
            self._drag_start_view_y = view_y
            self.cursor = "pan"
            self.viewer3d.pan(dx, -dy)
            self._draw_box_logical = []
            return

        if buttons == Qt.MouseButton.MiddleButton and modifiers == Qt.KeyboardModifier.NoModifier:
            self.cursor = "rotate"
            self.viewer3d.rotation(view_x, view_y)
            self._draw_box_logical = []
            return

        if buttons == Qt.MouseButton.LeftButton and modifiers == Qt.KeyboardModifier.NoModifier:
            self._select_area = True
            self._calculate_draw_box(event)
            return

        if buttons == Qt.MouseButton.RightButton and modifiers == Qt.KeyboardModifier.ShiftModifier:
            self._zoom_area = True
            self.cursor = "zoom-area"
            self._calculate_draw_box(event)
            return

        self._draw_box_logical = []
        self.viewer3d.MoveTo(view_x, view_y)
        self.cursor = "arrow"

    def mouseReleaseEvent(self, event):
        if self._rubber_band:
            self._rubber_band.hide()

        logical_pt = event.position().toPoint()
        view_x, view_y = self._to_view_xy(logical_pt.x(), logical_pt.y())

        if event.button() == Qt.MouseButton.LeftButton:
            if self._select_area and self._draw_box_logical and self.enable_multiply_select:
                sx, sy, dx, dy = self._draw_box_logical
                svx, svy = self._to_view_xy(sx, sy)
                evx, evy = self._to_view_xy(sx + dx, sy + dy)
                self.viewer3d.select_area(svx, svy, evx, evy)
                self._select_area = False
                self.update()
                if self.viewer3d.selected_AISs:
                    self.signal_AISs_selected.emit(self.viewer3d.selected_AISs)
            else:
                if event.modifiers() == Qt.KeyboardModifier.ControlModifier and self.enable_multiply_select:
                    self.viewer3d.shift_select(view_x, view_y)
                else:
                    self.viewer3d.select(view_x, view_y)
                if self.viewer3d.selected_AISs:
                    self.signal_AISs_selected.emit(self.viewer3d.selected_AISs)

        elif event.button() == Qt.MouseButton.RightButton:
            if self._zoom_area and self._draw_box_logical:
                sx, sy, dx, dy = self._draw_box_logical
                svx, svy = self._to_view_xy(sx, sy)
                evx, evy = self._to_view_xy(sx + dx, sy + dy)
                self.viewer3d.zoom_area_to(svx, svy, evx, evy)
                self._zoom_area = False
                self.update()

        self._draw_box_logical = []
        self.cursor = "arrow"

    # endregion

    # region 光标与框选
    @property
    def cursor(self) -> str:
        return self._current_cursor

    @cursor.setter
    def cursor(self, value: str) -> None:
        if self._current_cursor != value:
            self._current_cursor = value
            cursor = self._available_cursors.get(value)
            if cursor:
                self.qApp.setOverrideCursor(cursor)
            else:
                self.qApp.restoreOverrideCursor()

    def _calculate_draw_box(self, event: QtGui.QMouseEvent, tolerance: int = 2):
        point = event.position().toPoint()
        dx = point.x() - self._drag_start_logical_x
        dy = point.y() - self._drag_start_logical_y
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            return None

        self._draw_box_logical = [
            self._drag_start_logical_x,
            self._drag_start_logical_y,
            dx,
            dy,
        ]
        self.drawRubberBand(
            self._drag_start_logical_x,
            self._drag_start_logical_y,
            point.x(),
            point.y(),
        )
        return self._draw_box_logical

    def drawRubberBand(self, min_x: int, min_y: int, max_x: int, max_y: int) -> None:
        rect = QRect()
        rect.setX(min(min_x, max_x))
        rect.setY(min(min_y, max_y))
        rect.setWidth(abs(max_x - min_x))
        rect.setHeight(abs(max_y - min_y))

        if not self._rubber_band:
            self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
            self._rubber_band.setStyle(QStyleFactory.create("windows"))
        self._rubber_band.setGeometry(rect)
        self._rubber_band.show()

    def create_cursors(self) -> None:
        self._available_cursors = {
            "arrow": QtGui.QCursor(Qt.CursorShape.ArrowCursor),
            "pan": QtGui.QCursor(Qt.CursorShape.SizeAllCursor),
            "rotate": QtGui.QCursor(Qt.CursorShape.CrossCursor),
            "zoom": QtGui.QCursor(Qt.CursorShape.SizeVerCursor),
            "zoom-area": QtGui.QCursor(Qt.CursorShape.SizeVerCursor),
        }
        self._current_cursor = "arrow"

    # endregion

    # region 场景辅助显示
    def display_view_trihedron(self) -> None:
        trihedron = createTrihedron(arrow_length=50.0)
        trihedron.SetTransformPersistence(
            Graphic3d_TransformPers(
                Graphic3d_TransModeFlags.Graphic3d_TMF_TriedronPers,
                Aspect_TypeOfTriedronPosition.Aspect_TOTP_RIGHT_LOWER,
                Graphic3d_Vec2i(80, 50),
            )
        )
        self.viewer3d.Context.Display(trihedron, False)

    def display_view_cube(self) -> None:
        view_cube = AIS_ViewCube()

        axis_size = 5.0
        view_cube.SetAxesRadius(axis_size)
        view_cube.SetAxesConeRadius(axis_size * 1.5)
        view_cube.SetAxesSphereRadius(axis_size * 1.5)

        drawer = view_cube.Attributes()
        drawer.SetDatumAspect(Prs3d_DatumAspect())
        datum_aspect = drawer.DatumAspect()
        datum_aspect.TextAspect(Prs3d_DP_XAxis).SetColor(X_AXIS_COLOR)
        datum_aspect.TextAspect(Prs3d_DP_YAxis).SetColor(Y_AXIS_COLOR)
        datum_aspect.TextAspect(Prs3d_DP_ZAxis).SetColor(Z_AXIS_COLOR)

        datum_aspect.ShadingAspect(Prs3d_DP_XAxis).SetColor(X_AXIS_COLOR)
        datum_aspect.ShadingAspect(Prs3d_DP_YAxis).SetColor(Y_AXIS_COLOR)
        datum_aspect.ShadingAspect(Prs3d_DP_ZAxis).SetColor(Z_AXIS_COLOR)

        drawer.SetFaceBoundaryDraw(True)
        drawer.SetFaceBoundaryAspect(Prs3d_LineAspect(Graphic3d_AspectLine3d()))
        drawer.FaceBoundaryAspect().SetColor(Quantity_Color(228 / 255, 144 / 255, 255 / 255, Quantity_TOC_RGB))

        view_cube.SetBoxSideLabel(V3d_Xneg, "Left")
        view_cube.SetBoxSideLabel(V3d_Xpos, "Right")
        view_cube.SetBoxSideLabel(V3d_Yneg, "Front")
        view_cube.SetBoxSideLabel(V3d_Ypos, "Rear")
        view_cube.SetBoxSideLabel(V3d_Zpos, "Top")
        view_cube.SetBoxSideLabel(V3d_Zneg, "Bottom")

        view_cube.SetFont("Microsoft YaHei")
        view_cube.SetFontHeight(30)
        view_cube.SetBoxColor(Quantity_Color(228 / 255, 144 / 255, 255 / 255, Quantity_TOC_RGB))
        view_cube.SetTransparency(0.9)
        view_cube.SetBoxFacetExtension(14)
        view_cube.SetAxesLabels("X", "Y", "Z")
        view_cube.SetRoundRadius(0.1)
        view_cube.SetBoxEdgeGap(1)

        self.viewer3d.Context.Display(view_cube, False)

    def display_origin_trihedron(self, arrow_length: float = 1000.0) -> None:
        trihedron = createTrihedron(arrow_length=arrow_length)
        self.viewer3d.Context.Display(trihedron, False)

    def display_graduated_trihedron(self) -> None:
        trihedron_data = Graphic3d_GraduatedTrihedron()
        self.viewer3d.View.GraduatedTrihedronDisplay(trihedron_data)

    # endregion

    # region 对外便捷接口
    def erase_all(self) -> None:
        self.viewer3d.EraseAll()
        if self._view_trihedron:
            self.display_view_trihedron()
        if self._origin_trihedron:
            self.display_origin_trihedron()
        if self._view_cube:
            self.display_view_cube()
        self.context.UpdateCurrentViewer()

    # endregion


# endregion
