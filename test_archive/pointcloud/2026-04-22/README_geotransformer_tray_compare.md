# GeoTransformer 料盘定位对比测试说明

本文档说明 `test_orbbec_tray_pose_geotransformer_compare.py` 的完整流程、参数、ROI 约束、可视化确认机制和自动建模板模式。

## 1. 目标

该脚本用于把你当前的「相机 + 零样本分割」流程和「GeoTransformer 点云配准」串起来，做人工观察式对比测试。

## 2. 当前流程

1. 从 Orbbec 相机采集 RGBD 点云。
2. 用 zero-shot 分割得到料盘 2D mask。
3. 把 mask 回投到 3D 点云，得到料盘 ROI。
4. 模板准备：

- 若有模板文件，直接读取。
- 若无模板且启用自动建模板：先进入预览窗口，确认识别正常后，再采集多帧 ROI 融合模板。

1. 进入源帧预览窗口，再次确认当前帧 ROI 后锁定 `src`。
2. 保存 `src.npy/ref.npy/gt.npy`，调用 GeoTransformer 官方 `demo.py`。

说明：`gt.npy` 当前写单位阵，仅用于兼容 demo 输入；主要观察配准可视化和估计变换稳定性。

## 3. ROI 约束

1. ROI 尽量只包含料盘，不要混入桌面和背景。
2. ROI 点数足够（默认 `--min-tray-points=600`）。
3. 视角不要极端遮挡，保证几何结构可见。
4. 体素尺度要合理（默认 `--voxel-size-m=0.01`）。

## 4. 为什么必须要模板

GeoTransformer 在本脚本中做的是“两点云配准”：

1. `src`：当前相机帧料盘点云。
2. `ref`：模板点云（几何参考）。

没有 `ref` 就没有配准目标，因此必须有模板。

## 5. 可视化确认机制

默认启用交互预览（你提到的问题已按此修正）：

1. 模板构建前先预览，确认识别到物体再开始。
2. 源帧抓取前再预览，确认后才锁定当前帧。

默认按键：

- `S`：确认当前帧
- `Q`：退出

## 6. 自动建模板模式

当模板缺失（或你强制重建）时：

1. 先经过人工确认；
2. 采集多帧 ROI；
3. 融合 + 体素下采样；
4. 写入模板（默认：`C:\Project Documents\鼎泰项目\GeoTransformer\template\tray_template.npy`）。

## 7. 关键参数

- `--auto-build-template/--no-auto-build-template`
- `--force-rebuild-template/--no-force-rebuild-template`
- `--template-output`
- `--template-build-frames`
- `--template-build-attempts`
- `--template-build-min-accepted`
- `--interactive-preview/--no-interactive-preview`
- `--require-confirm-before-template/--no-require-confirm-before-template`
- `--require-confirm-before-source/--no-require-confirm-before-source`
- `--confirm-key` / `--quit-key`

## 8. 运行示例

### 8.1 推荐：自动建模板 + 人工确认 + 对比测试

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe C:\Projects\Dingtai\test\pointcloud\test_orbbec_tray_pose_geotransformer_compare.py \
  --geot-python "C:\\Path\\To\\GeoTransformerEnv\\python.exe"
```

### 8.2 强制重建模板

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe C:\Projects\Dingtai\test\pointcloud\test_orbbec_tray_pose_geotransformer_compare.py \
  --force-rebuild-template \
  --template-build-frames 10 \
  --template-build-attempts 120 \
  --geot-python "C:\\Path\\To\\GeoTransformerEnv\\python.exe"
```

### 8.3 非交互自动模式（不推荐）

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe C:\Projects\Dingtai\test\pointcloud\test_orbbec_tray_pose_geotransformer_compare.py \
  --no-interactive-preview \
  --geot-python "C:\\Path\\To\\GeoTransformerEnv\\python.exe"
```

## 9. 输出

每次运行在 `--save-root` 下生成时间戳目录，包含：

1. `src.npy`
2. `ref.npy`
3. `gt.npy`
4. `seg_overlay.png`

## 10. 验证边界

当前完成的是代码级实现与语法验证。真实效果仍需你在现场连接相机进行人工观察验证。
