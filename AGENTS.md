# AGENTS.md

## 目的

本文件定义我在 Dingtai 项目中使用 Codex 时的通用偏好。
如果项目子目录内存在更具体的 `AGENTS.md`，则优先遵循子目录内说明。
当前项目处于预研阶段，优先尝试新的内容，不要保持旧代码的兼容性。

---

## 项目结构

- `src/`：长期维护的业务、算法、硬件适配与公共数据结构。
- GUI 不属于本仓库的产品实现；正式 GUI 由独立的 `wuji_gui` 项目维护。
- `test/`：可运行验证脚本，默认支持 CLI 与 IDE 直跑双模式。
- `experiments/`：预研脚本，可以快速试验，但沉淀到 `src/` 前必须整理职责与数据结构。
- `debug/`：临时调试入口，不作为长期公共接口。
- `.agents/skills/`：仓库级 Codex skills。每个 skill 自己维护 `scripts/`、`references/`、`assets/`。
- `.codex/`：项目级 Codex 配置与 hooks。hooks 负责可确定执行的前后置检查。

---

## 推荐的做法

- 数据优先，数据驱动
- 代码必须具备 Windows 与 Linux 的兼容性
- 修改代码前先定位最小归属模块，再在正确层级做最小范围修改。
- 读取和写入文本文件必须显式使用 UTF-8。
- 修改前先用 `git status` 和 `git diff` 确认工作区边界；文件恢复与变更审查统一由 Git 管理。
- 文件编辑后优先运行最小静态检查；Python 文件默认走 `ruff` 后 `pyright`。
- 涉及硬件、GUI 或实时相机链路时，明确说明哪些只是语法/静态验证，哪些没有实际连接硬件验证。
- `src/` 下 Python 代码必须遵循 `.agents/skills/dingtai-src-python-style/SKILL.md`。
- 涉及几何、姿态、角度、颜色时，优先使用 `src.utils.datas` 已有类型。
- 禁止使用 `getattr` 这类魔术式调用。
- 已经具备明确类型的属性和变量禁止再套类型转换，尽可能不使用类型转换；只有在外部输入、反序列化或类型无法静态确定时才允许做最小必要转换。
- 静态检查统一使用 `.agents/skills/dingtai-static-check-workflow/scripts/check/` 下脚本。
- 静态检查默认使用 DingTai Conda 环境。

---

## PowerShell 执行环境

- Windows 下所有 PowerShell 命令统一使用 PowerShell 7 的 `pwsh`，默认添加 `-NoProfile`，避免用户配置影响自动化结果。
- 禁止调用 `powershell`、`powershell.exe` 或任何 Windows PowerShell 5.1 进程；不得在 `pwsh` 不可用时静默回退到 Windows PowerShell 5.1。
- 执行脚本统一使用 `pwsh -NoProfile -File <script.ps1>`；执行内联命令统一使用 `pwsh -NoProfile -Command <command>`。
- 如果 `pwsh` 不可用，应停止相关 PowerShell 操作并明确报告环境缺失，而不是使用 Windows PowerShell 5.1 继续执行。

---

## 标准命令

全仓静态检查：

```powershell
pwsh -NoProfile -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_all_checks.ps1 -Target .
```

按目录或文件检查：

```powershell
pwsh -NoProfile -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_ruff.ps1 -Target .\src
pwsh -NoProfile -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_pyright.ps1 -Target .\test
```

Git 变更检查：

```powershell
git status --short
git diff -- .\src\example.py
```

---

## Codex Hooks

- `.codex/hooks.json` 是项目级 hooks 入口。
- hooks 不再生成编辑前快照；文件恢复与变更审查统一由 Git 管理。
- `PostToolUse` 在 `apply_patch` 后扫描 UTF-8、字面量 ``
 ``、字面量 `
`、替换字符和 NUL。
- `PostToolUse` 对 Python 文件运行 `ruff --fix` 和 `pyright`。
- hooks 只能做确定性检查和修复；不能用模型猜测修复静态检查错误。

---

## 不推荐的做法

- 保持旧代码的结构和行为
- 继续使用旧的 `.agents/tools` 或 `.agent/tools` 路径。
- 为了通过检查静默改变默认行为、时序、重试策略或控制流程。
- 在 UI 层直接新增 IO、硬件控制或算法实现来绕过既有分层。
- 用无结构 `dict` 长距离透传参数；参数过多时优先提取 dataclass。
- 使用不必要的动态分发或魔术式调用。
- 不要盲目增加不必要的复杂性；如果现有接口只需修补一个方法，就不要顺手引入缓存层、额外包装层或新的兼容分支。

## 旋转约定

项目中使用的欧拉旋转均使用 `scipy.spatial.transform.Rotation.as_euler("xyz", degrees=True)` 表示，即 SciPy 小写外禀 xyz 约定。
