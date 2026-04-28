from __future__ import annotations

import itertools
from collections.abc import Callable

from OCC.Core.AIS import AIS_InteractiveContext, AIS_Shaded, AIS_Shape, AIS_WireFrame
from OCC.Core.Graphic3d import (
    Graphic3d_Camera,
    Graphic3d_RenderingParams,
    Graphic3d_RM_RASTERIZATION,
    Graphic3d_RM_RAYTRACING,
    Graphic3d_StereoMode_QuadBuffer,
)
from OCC.Core.Quantity import Quantity_Color
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_ShapeEnum,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
)
from OCC.Core.V3d import V3d_AmbientLight, V3d_View, V3d_Viewer
from OCC.Core.Visualization import Display3d

from .tools import ensure_occ_casroot

ensure_occ_casroot()


# region OCC Viewer 核心能力
class OCCViewer(Display3d):
    """OCC 原生视图封装：管理显示模式、选择、相机与渲染参数。"""

    def __init__(self) -> None:
        super().__init__()
        self._parent = None
        self._inited = False
        self.Context: AIS_InteractiveContext = self.GetContext()
        self.Viewer: V3d_Viewer = self.GetViewer()
        self.View: V3d_View = self.GetView()
        self.default_drawer = None
        self.camera: Graphic3d_Camera | None = None
        self.selected_AISs: list[object] = []
        self._select_callbacks: list[Callable] = []
        self.selection_modes = itertools.cycle([TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID])
        self._current_selection_mode: TopAbs_ShapeEnum = TopAbs_SOLID

    # region 回调与基础动作
    def register_select_callback(self, callback: Callable) -> None:
        if not callable(callback):
            raise AssertionError("callback must be callable")
        self._select_callbacks.append(callback)

    def unregister_callback(self, callback: Callable) -> None:
        if callback not in self._select_callbacks:
            raise AssertionError("callback is not registered")
        self._select_callbacks.remove(callback)

    def MoveTo(self, x: int, y: int) -> None:
        self.Context.MoveTo(x, y, self.View, True)

    def FitAll(self) -> None:
        self.View.ZFitAll()
        self.View.FitAll()
        self.Context.UpdateCurrentViewer()

    def Repaint(self) -> None:
        self.Viewer.Redraw()

    def EraseAll(self) -> None:
        self.Context.EraseAll(True)

    # endregion

    # region 初始化与渲染配置
    def Create(
        self,
        window_handle=None,
        parent=None,
        create_default_lights: bool = True,
        draw_face_boundaries: bool = False,
        phong_shading: bool = True,
        display_glinfo: bool = False,
    ) -> None:
        self._parent = parent
        if window_handle:
            self.Init(int(window_handle))
        else:
            self.InitOffscreen(640, 480)

        if display_glinfo:
            self.GlInfo()

        if create_default_lights:
            self.Viewer.SetDefaultLights()
            self.Viewer.AddLight(V3d_AmbientLight())
            self.Viewer.SetLightOn()

        self.camera = self.View.Camera()
        self.default_drawer = self.Context.DefaultDrawer()
        self.default_drawer.SetFaceBoundaryDraw(bool(draw_face_boundaries))

        chord_dev = self.default_drawer.MaximalChordialDeviation() / 10.0
        self.default_drawer.SetMaximalChordialDeviation(chord_dev)

        if phong_shading:
            from OCC.Core.Graphic3d import Graphic3d_TOSM_FRAGMENT

            self.View.SetShadingModel(Graphic3d_TOSM_FRAGMENT)

        self._inited = True

    def SetModeWireFrame(self) -> None:
        self.View.SetComputedMode(False)
        self.Context.SetDisplayMode(AIS_WireFrame, True)

    def SetModeShaded(self) -> None:
        self.View.SetComputedMode(False)
        self.Context.SetDisplayMode(AIS_Shaded, True)

    def SetModeHLR(self) -> None:
        self.View.SetComputedMode(True)

    def SetRenderingParams(
        self,
        Method=Graphic3d_RM_RASTERIZATION,
        RaytracingDepth=3,
        IsShadowEnabled=True,
        IsReflectionEnabled=False,
        IsAntialiasingEnabled=False,
        IsTransparentShadowEnabled=False,
        StereoMode=Graphic3d_StereoMode_QuadBuffer,
        AnaglyphFilter=Graphic3d_RenderingParams.Anaglyph_RedCyan_Optimized,
        ToReverseStereo=False,
    ) -> None:
        self.ChangeRenderingParams(
            Method,
            RaytracingDepth,
            IsShadowEnabled,
            IsReflectionEnabled,
            IsAntialiasingEnabled,
            IsTransparentShadowEnabled,
            StereoMode,
            AnaglyphFilter,
            ToReverseStereo,
        )

    def SetRasterizationMode(self) -> None:
        self.SetRenderingParams()

    def SetRaytracingMode(self, depth: int = 3) -> None:
        self.SetRenderingParams(
            Method=Graphic3d_RM_RAYTRACING,
            RaytracingDepth=depth,
            IsAntialiasingEnabled=True,
            IsShadowEnabled=True,
            IsReflectionEnabled=True,
            IsTransparentShadowEnabled=True,
        )

    def set_selection_color(self, display_mode: int, color: Quantity_Color) -> None:
        selection_style = self.Context.SelectionStyle()
        selection_style.SetDisplayMode(display_mode)
        selection_style.SetColor(color)

    def EnableAntiAliasing(self) -> None:
        self.SetNbMsaaSample(4)

    def DisableAntiAliasing(self) -> None:
        self.SetNbMsaaSample(0)

    # endregion

    # region 视角与交互动作
    def pan(self, dx: int, dy: int) -> None:
        self.View.Pan(dx, dy)

    def rotation(self, x: int, y: int) -> None:
        self.View.Rotation(x, y)

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self.View.SetZoom(zoom_factor)

    def zoom_area_to(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.View.WindowFit(x1, y1, x2, y2)

    def start_rotation(self, x: int, y: int) -> None:
        self.View.StartRotation(x, y)

    # endregion

    # region 选择模式与选择结果
    def change_selection_mode(self) -> None:
        self._current_selection_mode = next(self.selection_modes)
        self.set_selection_mode()

    def set_selection_mode(self, mode: TopAbs_ShapeEnum | None = None) -> None:
        self.Context.Deactivate()
        if mode is not None:
            self.Context.Activate(AIS_Shape.SelectionMode(mode), True)
        else:
            self.Context.Activate(AIS_Shape.SelectionMode(self._current_selection_mode), True)
        self.Context.UpdateSelected(True)

    def select_area(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        self.Context.Select(xmin, ymin, xmax, ymax, self.View, True)
        self.Context.InitSelected()
        self.selected_AISs = []
        while self.Context.MoreSelected():
            self.selected_AISs.append(self.Context.SelectedInteractive())
            self.Context.NextSelected()
        for callback in self._select_callbacks:
            callback(self.selected_AISs, xmin, ymin, xmax, ymax)

    def select(self, x: int, y: int) -> None:
        self.Context.Select(True)
        self.Context.InitSelected()
        self.selected_AISs = []
        if self.Context.MoreSelected():
            self.selected_AISs.append(self.Context.SelectedInteractive())
        for callback in self._select_callbacks:
            callback(self.selected_AISs, x, y)

    def shift_select(self, x: int, y: int) -> None:
        self.Context.ShiftSelect(True)
        self.Context.InitSelected()
        self.selected_AISs = []
        while self.Context.MoreSelected():
            self.selected_AISs.append(self.Context.SelectedInteractive())
            self.Context.NextSelected()
        self.Context.UpdateSelected(True)
        for callback in self._select_callbacks:
            callback(self.selected_AISs, x, y)

    # endregion


# endregion
