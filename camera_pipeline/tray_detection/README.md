# Tray Detection

> 暂时移除：本模块已从 CameraPipeline 运行时、RPC 协议和公共 client API 摘除，
> `__init__.py` 已删除。目录内源码仅作后续恢复参考，当前 Python 3.12 部署不会加载本模块。

## 单一职责

`tray_detection` 只负责在输入彩色帧中检测、分割并排序托盘，输出托盘几何摘要和可选 mask/debug。它不连接相机、不创建 RPC、不等待稳定帧，也不调用 opening 算法。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `types.py` | 检测配置、内部检测结果和运行状态 |
| `detector.py` | Grounding DINO 候选框与可选 SAM 分割 |
| `pipeline.py` | 完整检测与缓存结果快速更新 |
| `motion_shift.py` | 相邻图像全局平移估计 |
| `engine.py` | 将算法结果组装为协议响应 |
| `service.py` | 纯计算执行器的薄入口 |
| `protocol.py` | 请求、响应和 debug 协议 |
| `model_cache.py` | 模型进程内缓存 |
| `tune_target_bbox.py` | 离线候选框参数调优工具 |

## 算法理论

第一阶段使用 Grounding DINO，根据文本 prompt 在图像中产生开放词汇目标框和置信度。候选通过目标关键词、置信度、数量和重叠抑制筛选。

第二阶段根据配置选择：

- 直接从目标区域生成快速 mask；或
- 使用 SAM 对候选框细化分割。

为减少大模型逐帧推理成本，`TrayDetectionPipeline` 周期性执行完整检测，其余帧根据灰度图相位/运动估计平移缓存 mask，并结合阈值和形态学操作更新区域。最终结果按图像 X 方向从左到右排序。

## 输入协议

`OrinTrayDetectionRequest` 提供请求号、相机名、可选 `frame_id` 和 `enable_debug`。真实帧由业务层传给 `OrinTrayDetectionService.compute()`。

## 输出协议

`OrinTrayDetectionResponse` 包含：

- 实际处理的 `frame_id`、时间戳和来源；
- `tray_count`；
- 每个托盘的 bbox、center、面积和置信度摘要；
- `debug_artifacts` 调试产物元组和逐托盘 mask；
- 错误字段。

关闭 debug 时不生成叠加图和大体积调试载荷，`debug_artifacts` 返回空元组。

## 成功与失败语义

- 成功检测：返回按图像 X 方向排序的 `tray_results`。
- 未检测到托盘：返回成功空结果 `tray_count=0`、`tray_results=()`。
- 输入非法、模型缓存缺失、CUDA 不可用或推理失败：抛出异常，由服务层转换为统一错误。
- debug 关闭：`debug_artifacts=()`，不生成图像和 mask 预览。

## 调参建议

建议使用真实现场数据按以下顺序调节：

1. `prompt`、`target_keywords`：先确保语义目标正确。
2. `box_threshold`、`text_threshold`、`min_confidence`：平衡漏检和误检。
3. `max_targets`、`topk_objects`：根据真实托盘数量限制候选。
4. `mask_iou_suppress`：控制重叠托盘去重。
5. `min_mask_pixels`：排除远处或噪声小区域。
6. `detect_max_side`：在速度和小目标精度之间取舍。
7. `use_sam` 与 SAM 相关阈值：仅在框 mask 无法满足边界精度时启用。
8. `detect_every_n`、`motion_max_shift_px`：根据相机运动速度调整缓存复用周期。

调参必须同时记录推理耗时、显存和不同光照下的误检率，不能只看单帧效果。

## Hugging Face 模型缓存

默认缓存目录由 `TrayDetectionConfig.hf_cache_dir` 提供：

```text
<项目根目录>/.cache/huggingface
```

运行时会同时配置：

- `HF_HOME=<cache_dir>`；
- `TRANSFORMERS_CACHE=<cache_dir>/transformers`；
- `HUGGINGFACE_HUB_CACHE=<cache_dir>/hub`；
- 项目整理后的模型副本位于 `<cache_dir>/project_store/<role>/<model-id>`。

正式 Orin 默认使用 `hf_local_files_only=True`。缓存不存在时不会静默联网，模型加载会失败。应在能够访问 Hugging Face 的 DingTai 环境中提前下载：

```powershell
$env:HTTP_PROXY='http://127.0.0.1:4444'
$env:HTTPS_PROXY='http://127.0.0.1:4444'
$env:ALL_PROXY='http://127.0.0.1:4444'
@'
from pathlib import Path
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

cache_dir = Path.cwd() / ".cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base",
    cache_dir=cache_dir,
    local_files_only=False,
)
AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base",
    cache_dir=cache_dir,
    local_files_only=False,
)

# 仅在 TrayDetectionConfig.use_sam=True 时需要：
SamProcessor.from_pretrained(
    "facebook/sam-vit-base",
    cache_dir=cache_dir,
    local_files_only=False,
)
SamModel.from_pretrained(
    "facebook/sam-vit-base",
    cache_dir=cache_dir,
    local_files_only=False,
)
'@ | C:\Users\ICO\anaconda3\envs\DingTai\python.exe -
```

下载后应恢复离线配置，并先验证：

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe -c "from camera_pipeline.tray_detection.types import TrayDetectionConfig; print(TrayDetectionConfig().hf_cache_dir)"
```

部署到 Orin 时需要把整个 `.cache/huggingface` 目录按相同项目相对路径同步过去，或者显式设置 `hf_cache_dir` 指向 Orin 上已有缓存。当前代码固定按 `transformers==5.6.0` 接口调用 `threshold` 和 `text_threshold`，不保留多版本参数探测分支。

## 局限性

- Grounding DINO 受 prompt、模型缓存和场景域差异影响。
- 快速帧更新主要处理二维平移，明显旋转、尺度变化或遮挡时应重新检测。
- 黑色托盘在低照度、反光或黑色背景中对比度不足时容易失败。
- SAM 会增加显存和延迟。
- 输出 mask 是二维观测，不直接证明三维托盘平面有效。
- 模型和 CUDA 只能在 Orin 真实环境验证。
