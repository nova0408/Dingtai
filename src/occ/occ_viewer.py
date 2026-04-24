from __future__ import annotations

from .tools import (
    RGB_to_Quantity_Color,
    Trans2trsf,
    createTrihedron,
    ensure_occ_casroot,
    occ_to_string,
)
from .viewer_base_widget import qtBaseViewerWidget
from .viewer_core import OCCViewer
from .viewer_widget import qtViewer3dWidget

# region 兼容导出入口
# 说明：
# 1. 本文件保留历史导入路径 `src.occ.occ_viewer`，避免下游调用中断。
# 2. 具体实现已拆分到不同代码页：viewer_core / viewer_base_widget / viewer_widget。
# 3. 工具函数和辅助能力收敛到 `src.occ.tools` 子模块。


__all__ = [
    "ensure_occ_casroot",
    "createTrihedron",
    "RGB_to_Quantity_Color",
    "occ_to_string",
    "Trans2trsf",
    "OCCViewer",
    "qtBaseViewerWidget",
    "qtViewer3dWidget",
]
# endregion
