# 托盘检测 HuggingFace 模型说明

## 1. 作用范围

本目录的 HuggingFace 模型仅用于托盘检测链路：

1. `GroundingDINO`：在 RGB 图像中做 zero-shot 托盘候选框检测。
2. `SAM`：可选，用于把候选框细化为像素级 mask。

对应代码入口：

1. `detector.py` 中 `TrayPointExcluder.detect()`：输出 2D 托盘检测结果。
2. `detector.py` 中 `TrayPointExcluder.exclude_points()`：将 2D mask 映射为点云排除掩码。
3. `pipeline.py` 中 `TrayDetectionPipeline`：实时场景下做快速掩码 + 异步高置信刷新。

不在本模块职责内：

1. 相机采集、帧同步。
2. 三平面拟合与抓取位姿估计。
3. GUI 可视化窗口管理。

## 2. 模型与默认配置

默认模型定义在 `types.py -> TrayDetectionConfig`：

1. `gd_model_id="IDEA-Research/grounding-dino-base"`
2. `sam_model_id="facebook/sam-vit-base"`

默认策略：

1. `hf_local_files_only=True`，默认仅本地加载，不主动联网。
2. `use_sam=True` 时会加载 SAM；`False` 时只用检测框生成矩形 mask。

## 3. 安装与依赖

建议固定在 `DingTai` 环境执行：

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe -m pip install -U transformers torch pillow
```

若需要 GPU 推理，请确认：

1. `torch.cuda.is_available()` 为 `True`。
2. `TrayDetectionConfig.device` 设置为 `cuda:0`（或其他可用 CUDA 设备）。

## 4. 缓存机制

缓存逻辑在 `model_cache.py`：

1. `prepare_hf_cache_dir()`：创建项目缓存目录并设置环境变量：
   - `HF_HOME`
   - `TRANSFORMERS_CACHE`
   - `HUGGINGFACE_HUB_CACHE`
2. `load_pretrained_with_project_cache()`：优先从项目缓存加载；未命中时按配置尝试 HuggingFace 缓存或下载。
3. 成功加载后，若对象支持 `save_pretrained`，会写回项目缓存副本（`project_store`）。

默认缓存目录：

1. `C:\Projects\Dingtai\.cache\huggingface`

目录结构示意：

```text
.cache/huggingface/
  transformers/
  hub/
  project_store/
    grounding_dino_processor/
    grounding_dino_model/
    sam_processor/
    sam_model/
```

## 5. 首次下载与离线运行

首次下载（允许联网）：

1. 将 `TrayDetectionConfig.hf_local_files_only=False`。
2. 如有代理，设置 `proxy_url`（默认支持 `http://127.0.0.1:4444`）。
3. 运行一次检测流程，模型会被缓存。

离线运行（推荐日常）：

1. 将 `hf_local_files_only=True`（默认）。
2. 确保模型已存在于项目缓存或全局 HuggingFace 缓存。

## 6. 最小使用示例

```python
from src.pointcloud.tray_detection import (
    TrayDetectionConfig,
    TrayPointExcluder,
)

cfg = TrayDetectionConfig(
    hf_local_files_only=True,
    use_sam=False,
    device="cuda:0",
)
detector = TrayPointExcluder(cfg)
detections = detector.detect(frame_bgr)
```

实时流程示例：

```python
from src.pointcloud.tray_detection import (
    TrayDetectionPipeline,
    TrayRuntimeState,
)

pipeline = TrayDetectionPipeline(TrayDetectionPipeline.build_default_detector())
state = TrayRuntimeState()
mask, from_detector = pipeline.segment_tray(frame_bgr, state)
```

## 7. 常见问题

1. 报错“本地缓存不可用”：
   - 原因：`hf_local_files_only=True` 且缓存未命中。
   - 处理：临时改为 `False` 完成首轮下载。
2. 报错“请求使用 CUDA，但未检测到可用 GPU”：
   - 原因：设备配置与环境不一致。
   - 处理：改为 `device="cpu"` 或安装正确 CUDA 版本 PyTorch。
3. 检测速度慢：
   - 处理：`use_sam=False`、减小 `detect_max_side`、提高 `detect_every_n`。

## 8. 变更边界

本说明仅针对 `src/pointcloud/tray_detection/` 子模块。若改动 `grasp_pose` 或 `three_plane` 流程，需分别在对应模块文档补充说明。
