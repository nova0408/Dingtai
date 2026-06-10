---
name: dingtai-static-check-workflow
description: 统一 Dingtai 仓库静态检查流程。用于在固定 DingTai conda 环境下执行 ruff、pyright 与相关验证，并通过本 skill 的 `scripts/check` 提供通用 CLI、VSCode 与 hook 入口，保证 IDE 与命令行结果一致。
---

# Dingtai Static Check Workflow

## 目标

1. 统一静态检查入口，避免每个脚本各自维护检查命令。
2. 固定使用 `DingTai` 环境，避免 `base` 或其他环境导致检查漂移。
3. 对齐 VSCode（Pylance）与 CLI 的类型检查体验。
4. 所有由本 skill 触发的测试、验证与静态检查，都必须在 `DingTai` 环境中执行，不允许切换到其他环境“临时通过”。

## 适用范围

1. 本仓库全部 Python 代码：`src/`、`gui/`、`test/`、`experiments/`、`debug/`。
2. 本仓库所有新增或改动的 Python 文件。

## 强制规则

1. 必须通过 `.agents/skills/dingtai-static-check-workflow/scripts/check` 下的脚本执行检查，不直接依赖当前 PATH。
2. 必须固定使用 DingTai 环境中的可执行文件：
   - 按 Conda 环境名 `DingTai` 解析 `ruff.exe`
   - 按 Conda 环境名 `DingTai` 解析 `pyright.exe`
3. 由本 skill 触发的测试、验证和静态检查，必须共享同一个 `DingTai` 环境，不得混用 `base`、系统 Python 或其他 Conda 环境。
4. 检查流程顺序固定为：先 `ruff`，后 `pyright`。
5. 运行全量检查时，默认命令为：
   - `powershell -ExecutionPolicy Bypass -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_all_checks.ps1 -Target .`
6. 运行单项检查时，必须使用：
   - `run_ruff.ps1`
   - `run_pyright.ps1`
7. 结论中必须明确区分：
   - 已验证：静态检查（ruff/pyright）结果。
   - 未验证：硬件行为、实时相机链路、GUI 交互行为。
8. 不允许为了通过静态检查而随意更改业务默认行为或控制流程。
9. 所有 `*_ui.py` 文件默认豁免静态检查与人工修改；这类文件视为 Qt 插件自动生成产物，只允许通过生成流程更新，不允许手工修补。

## 标准命令

全仓检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_all_checks.ps1 -Target .
```

按目录检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_ruff.ps1 -Target .\src
powershell -ExecutionPolicy Bypass -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_pyright.ps1 -Target .\test
```

## Hook 对接

1. 项目级生命周期 hooks 位于 `.codex/hooks.json` 与 `.codex/hooks/`。
2. 文件编辑后的 `PostToolUse` hook 会对被编辑的 Python 文件调用本 skill 的检查脚本。
3. hook 中允许 `ruff --fix` 做确定性自动修复，不允许用模型猜测修复 pyright 类型错误。
4. hook 检查失败时应阻断继续处理，并把失败原因返回给 Codex 继续显式处理。
5. hook 与命令行脚本都必须自动跳过 `*_ui.py` 文件，避免对 Qt 自动生成代码做检查或改写。

## VSCode 对齐要求

1. `python.defaultInterpreterPath` 必须指向：
   - 当前机器上名为 `DingTai` 的 Conda 环境解释器
2. 使用仓库根目录 `pyrightconfig.json` 与 `ruff.toml` 作为统一配置。
3. VSCode 任务入口应调用 `.agents/skills/dingtai-static-check-workflow/scripts/check` 下脚本，不得回退到旧路径。

## 失败处理

1. 若脚本路径失效，先修复 `.vscode/tasks.json`、`.codex/hooks.json` 与本 skill `scripts/check` 路径一致性。
2. 若环境缺少工具，先在 DingTai 环境安装后再执行检查，不允许切到其他环境“临时通过”。
3. 若 pyright 报第三方库缺失类型，先确认解释器与包安装状态，再决定是否做最小范围类型忽略配置。
4. 若测试或验证需要运行额外脚本，也必须先确认其解释器指向 `DingTai` 环境，再继续执行。
