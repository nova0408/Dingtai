---
name: open3d-usage-guide
description: 基于本项目 Open3D 实践提供点云处理与可视化工作流。用于需要在项目中编写或修复 Open3D 代码的场景，尤其是点云读取、基础预处理、使用 O3DVisualizer 可视化、材质与背景设置、相机重置、窗口生命周期管理等任务。
---

# Open3D Usage Guide

## 目标

复用 `test/pointcloud/test_coarse_registration_methods_visual.py` 的已验证用法，快速完成：

1. 点云读取与基础校验
2. 使用 `O3DVisualizer` 做结果可视化
3. 可视化窗口和渲染参数设置

## 最小读取模板

```python
from pathlib import Path
import open3d as o3d

pcd_path = Path("experiments/pcd1.pcd")
if not pcd_path.exists():
    raise FileNotFoundError(f"点云文件不存在：{pcd_path}")

pcd = o3d.io.read_point_cloud(pcd_path)
if len(pcd.points) == 0:
    raise RuntimeError("点云为空")
```

## 最小可视化模板（O3DVisualizer）
```python
import numpy as np
import open3d as o3d

app = o3d.visualization.gui.Application.instance
app.initialize()

vis = o3d.visualization.O3DVisualizer("Open3D Viewer", 1440, 900)
vis.show_settings = True
vis.show_skybox(False)
vis.set_background(np.array([0, 0, 0, 0]), None)

mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 1.5

vis.add_geometry("pcd", pcd, mat)
vis.reset_camera_to_default()

app.add_window(vis)
app.run()
```

## 对齐可视化模板（原始 + 配准后）

将 `raw_*` 与 `reg_*` 同时 `add_geometry` 到同一窗口，用于快速观察配准前后效果。

## 常用设置约定

1. 对点云优先使用 `defaultUnlit`，减少光照影响，便于看颜色和几何重叠。
2. 统一设置 `point_size`，避免不同窗口视觉尺度不一致。
3. 关闭 skybox 并使用深色透明背景，突出点云。
4. 添加坐标轴 `create_coordinate_frame`，减少视角误判。
5. 显示前调用 `reset_camera_to_default()`。

## 窗口生命周期注意事项

1. 单次展示流程使用 `Application.instance -> initialize -> add_window -> run`。
2. 在循环中连续打开多个窗口时，避免混乱复用几何对象；每次显示前拷贝 `PointCloud`。
3. 如果任务是自动化测试，不要默认阻塞调用 `app.run()`；优先拆分为可选可视化开关。

## 与本项目保持一致的实现要点

1. 点云为空时显式抛错，不静默跳过。
2. 材质、背景、相机设置集中在一个函数，避免在业务逻辑散落重复代码。
