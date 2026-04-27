---
name: orbbec-realtime-preview-conventions
description: 统一 Orbbec 实时测试脚本的 Open3D 视角、cv2 窗口命名与 2D 预览尺寸规则，确保不同脚本预览一致且可用。
---

# Orbbec Realtime Preview Conventions

## 目标

1. Orbbec 相关测试脚本在 3D/2D 预览行为上保持一致。
2. 预览窗口默认可读、可对比、可复现实验结果。
3. 避免 cv2 中文窗口标题兼容问题。

## 适用范围

1. `test/pointcloud/` 下 Orbbec 实时预览脚本。
2. `experiments/` 下用于 Orbbec 点云/识别验证的脚本。

## 强制规则

1. Open3D 视角必须与 `experiments/test_colored_pointcloud_registration.py` 一致：
   - `lookat=[0.0, 0.0, 0.0]`
   - `front=[0.0, 0.0, -1.0]`
   - `up=[0.0, -1.0, 0.0]`
2. cv2 窗口标题必须使用 ASCII 英文，不允许中文标题。
3. 2D 预览窗口尺寸必须保证最小长边 `>= 800` 像素，并保持原始宽高比。
4. 默认参数放在文件顶部常量区，并给出中文注释（用途 + 单位）。
5. 测试脚本需支持 IDE 直跑与 CLI 覆盖双模式（遵循 `test-script-dual-run-mode`）。
6. 验证优先用项目环境 `C:\Users\ICO\anaconda3\envs\DingTai\python.exe`；若未连硬件，仅声明静态验证结论。

## 推荐实现片段

```python
view = vis.get_view_control()
view.set_lookat([0.0, 0.0, 0.0])
view.set_front([0.0, 0.0, -1.0])
view.set_up([0.0, -1.0, 0.0])

cv2.namedWindow("Orbbec preview", cv2.WINDOW_NORMAL)
win_w, win_h = _compute_preview_window_size(src_w=img_w, src_h=img_h, min_long_side=800)
cv2.resizeWindow("Orbbec preview", win_w, win_h)
```

## 输出要求

1. 说明 3D 视角是否对齐参考脚本。
2. 说明 2D 窗口标题是否为 ASCII 英文。
3. 说明 2D 窗口是否满足最小长边与比例约束。
4. 说明验证边界（静态检查/硬件实测）。
