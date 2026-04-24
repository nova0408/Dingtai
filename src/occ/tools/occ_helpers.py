from __future__ import annotations

import numpy as np

from OCC.Core.AIS import AIS_Shape, AIS_Trihedron
from OCC.Core.Geom import Geom_Axis2Placement
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Core.Prs3d import Prs3d_DP_XAxis, Prs3d_DP_YAxis, Prs3d_DP_ZAxis
from OCC.Core.Quantity import (
    Quantity_Color,
    Quantity_NOC_BLUE,
    Quantity_NOC_GREEN,
    Quantity_NOC_RED,
    Quantity_TOC_RGB,
)
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TopLoc import TopLoc_Location

# region 坐标轴配色
X_AXIS_COLOR = Quantity_Color(Quantity_NOC_RED)
Y_AXIS_COLOR = Quantity_Color(Quantity_NOC_GREEN)
Z_AXIS_COLOR = Quantity_Color(Quantity_NOC_BLUE)
# endregion


# region 通用 OCC 工具函数
def createTrihedron(
    input: AIS_Shape | TopLoc_Location | gp_Trsf | None = None,
    arrow_length: float = 0.2,
) -> AIS_Trihedron:
    """根据输入变换创建三轴坐标系。"""
    match input:
        case AIS_Shape():
            trsf = input.Shape().Location().Transformation()
        case TopLoc_Location():
            trsf = input.Transformation()
        case gp_Trsf():
            trsf = input
        case _:
            trsf = gp_Trsf()

    # 直接应用局部变换，保证与 AIS_Shape 的姿态语义一致。
    base_ax2 = gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0), gp_Dir(1.0, 0.0, 0.0))
    trihedron = AIS_Trihedron(Geom_Axis2Placement(base_ax2))
    trihedron.SetLocalTransformation(trsf)
    trihedron.SetDrawArrows(True)
    trihedron.SetSize(float(arrow_length))
    trihedron.SetDatumPartColor(Prs3d_DP_XAxis, X_AXIS_COLOR)
    trihedron.SetDatumPartColor(Prs3d_DP_YAxis, Y_AXIS_COLOR)
    trihedron.SetDatumPartColor(Prs3d_DP_ZAxis, Z_AXIS_COLOR)
    trihedron.SetArrowColor(Prs3d_DP_XAxis, X_AXIS_COLOR)
    trihedron.SetArrowColor(Prs3d_DP_YAxis, Y_AXIS_COLOR)
    trihedron.SetArrowColor(Prs3d_DP_ZAxis, Z_AXIS_COLOR)
    trihedron.SetTextColor(Prs3d_DP_XAxis, X_AXIS_COLOR)
    trihedron.SetTextColor(Prs3d_DP_YAxis, Y_AXIS_COLOR)
    trihedron.SetTextColor(Prs3d_DP_ZAxis, Z_AXIS_COLOR)
    return trihedron


def RGB_to_Quantity_Color(r: int | tuple[int, int, int] | list[int], g: int = 0, b: int = 0) -> Quantity_Color:
    """将 0-255 的 RGB 值转换为 OCC 颜色对象。"""
    if isinstance(r, (tuple, list)):
        r, g, b = r
    return Quantity_Color(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, Quantity_TOC_RGB)


def occ_to_string(value: str) -> TCollection_ExtendedString:
    """统一 OCC 字符串类型构造，便于后续替换/扩展。"""
    return TCollection_ExtendedString(value)


def Trans2trsf(mat: np.ndarray) -> gp_Trsf:
    """4x4 齐次矩阵转换为 OCC `gp_Trsf`。"""
    trsf = gp_Trsf()
    trsf.SetValues(
        mat[0, 0],
        mat[0, 1],
        mat[0, 2],
        mat[0, 3],
        mat[1, 0],
        mat[1, 1],
        mat[1, 2],
        mat[1, 3],
        mat[2, 0],
        mat[2, 1],
        mat[2, 2],
        mat[2, 3],
    )
    return trsf


# endregion
