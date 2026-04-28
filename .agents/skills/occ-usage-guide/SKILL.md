---
name: occ-usage-guide
description: 规范 Dingtai 项目中 pythonOCC 的导入、选面、拓扑映射与三角化调用。用于编写/修复 OCC 脚本时避免版本猜测和动态魔术分发，保证静态检查友好。
---

# OCC Usage Guide

## 目标

1. 在 `DingTai` 环境下稳定使用 pythonOCC。
2. 统一 OCC 导入与调用路径，减少版本差异导致的导入错误。
3. 避免 `hasattr/getattr/setattr` 式魔术分发，保持代码可静态检查。

## 适用范围

1. `src/occ/` 下 OCC 相关模块。
2. `experiments/`、`test/` 下 OCC 几何处理、选面与点云导出脚本。

## 强制规则

1. Python 解释器优先使用：`C:\Users\ICO\anaconda3\envs\DingTai\python.exe`。
2. `TopExp` 模块导入必须写成：`from OCC.Core.TopExp import topexp`。
3. 不允许写：`from OCC.Core.TopExp import TopExp`（DingTai 环境无该符号）。
4. 面映射统一使用：`topexp.MapShapes(shape, TopAbs_FACE, face_map)`。
5. `TopTools_IndexedMapOfShape` 数量统一使用：`face_map.Size()`。
6. 面三角化统一使用：`BRep_Tool.Triangulation(face, loc)`。
7. 不在业务脚本中引入 `hasattr/getattr/setattr` 做 OCC 版本分发；需要跨版本适配时，集中到独立适配层。

## 推荐实现片段

```python
from OCC.Core.TopExp import topexp
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

face_map = TopTools_IndexedMapOfShape()
topexp.MapShapes(shape, TopAbs_FACE, face_map)
face_count = int(face_map.Size())

loc = TopLoc_Location()
tri = BRep_Tool.Triangulation(face, loc)
```

## 最小验证

1. 语法检查：`python -m py_compile <changed_files>`。
2. 入口检查：`python <script>.py --help`。
3. 说明验证边界：仅静态验证 / GUI 与硬件未实测。

## 输出要求

1. 明确声明采用的 OCC 导入路径（如 `topexp`、`BRep_Tool.Triangulation`）。
2. 明确声明是否使用 DingTai 环境完成验证。
3. 若出现接口差异，先给出 `.pyi`/实际导出证据，再改代码。
