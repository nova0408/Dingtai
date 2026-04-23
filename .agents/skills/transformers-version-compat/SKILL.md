---
name: transformers-version-compat
description: 在本仓库固定使用 transformers 5.6.0，统一接口写法、代理下载和项目 conda 环境约束，避免后续反复做版本兼容分支。
---

# Transformers Version Compat

## 目标

1. 在 `DingTai` 项目中稳定使用 `transformers` 推理能力。
2. 固定 `transformers` 版本为 `5.6.0`，后续开发按该版本接口编写。
3. 统一模型下载代理和环境检查流程。

## 强制规则

1. 所有 `transformers` 检查与脚本验证，默认使用项目 conda 环境，不使用 `base`。
2. 版本锁定：`transformers==5.6.0`。除非用户明确要求，不要主动升级或降级。
3. GroundingDINO 后处理接口按 `5.6.0` 写法：使用 `threshold` 和 `text_threshold`，不要写 `box_threshold`。
4. 除非用户明确要求，不要新增多版本兼容分支或签名探测逻辑。
5. 涉及 `from_pretrained` 下载时，优先支持代理注入（`HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`）。
6. 变更后必须做最小静态验证（例如 `py_compile`），并明确是否已做硬件实测。

## 固定版本与检查命令

```powershell
C:\Users\ICO\anaconda3\envs\DingTai\python.exe -c "import transformers,inspect; print(transformers.__version__)"

C:\Users\ICO\anaconda3\envs\DingTai\python.exe -c "import inspect; from transformers.models.grounding_dino.processing_grounding_dino import GroundingDinoProcessor; print(inspect.signature(GroundingDinoProcessor.post_process_grounded_object_detection))"
```

要求：第一条命令输出必须是 `5.6.0`。如果不是，先与用户确认是否允许调整环境。

## GroundingDINO 接口约定（transformers 5.6.0）

`post_process_grounded_object_detection` 使用：

1. `threshold=<float>`
2. `text_threshold=<float>`
3. `target_sizes=<list[tuple[int,int]]>`

## 推荐代理约定

默认代理可设为：`http://127.0.0.1:4444`

```powershell
$env:HTTP_PROXY='http://127.0.0.1:4444'
$env:HTTPS_PROXY='http://127.0.0.1:4444'
$env:ALL_PROXY='http://127.0.0.1:4444'
```

## 输出要求

1. 明确报告：当前 `transformers` 版本、关键 API 签名、是否启用代理。
2. 明确报告：代码是否严格按 `transformers 5.6.0` 编写。
3. 若版本不一致，先提示版本偏差和建议动作，不默认引入兼容实现。
