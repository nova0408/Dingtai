# AGENTS.md

## 目的

本文件定义我在 Dingtai 项目中使用 Codex 时的通用偏好。
如果项目子目录内存在更具体的 `AGENTS.md`，则优先遵循子目录内说明。
当前项目处于预研阶段，优先尝试新的内容，不要保持旧代码的兼容性。

---

## 项目结构

- `src/`：长期维护的业务、算法、硬件适配与公共数据结构。
- `gui/`：Qt / PySide6 界面层，不在 UI 层重复实现存储、硬件或算法细节。
- `test/`：可运行验证脚本，默认支持 CLI 与 IDE 直跑双模式。
- `experiments/`：预研脚本，可以快速试验，但沉淀到 `src/` 前必须整理职责与数据结构。
- `debug/`：临时调试入口，不作为长期公共接口。
- `.agents/skills/`：仓库级 Codex skills。每个 skill 自己维护 `scripts/`、`references/`、`assets/`。
- `.codex/`：项目级 Codex 配置与 hooks。hooks 负责可确定执行的前后置检查。
- `.archive/`：修改前快照目录，快照必须保留从项目根目录开始的相对路径结构。

---

## 推荐的做法

- 数据优先，数据驱动
- 修改代码前先定位最小归属模块，再在正确层级做最小范围修改。
- 读取和写入文本文件必须显式使用 UTF-8。
- 修改文件前必须在 `.archive/` 下生成快照。
- 文件编辑后优先运行最小静态检查；Python 文件默认走 `ruff` 后 `pyright`。
- 涉及硬件、GUI 或实时相机链路时，明确说明哪些只是语法/静态验证，哪些没有实际连接硬件验证。
- `src/` 下 Python 代码必须遵循 `.agents/skills/dingtai-src-python-style/SKILL.md`。
- 涉及几何、姿态、角度、颜色时，优先使用 `src.utils.datas` 已有类型。
- Windows PowerShell 文本编辑优先使用 `.agents/skills/windows-powershell-utf8-safe-edit/scripts/` 下脚本。
- 静态检查统一使用 `.agents/skills/dingtai-static-check-workflow/scripts/check/` 下脚本。

---

## 标准命令

全仓静态检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_all_checks.ps1 -Target .
```

按目录或文件检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_ruff.ps1 -Target .\src
powershell -ExecutionPolicy Bypass -File .\.agents\skills\dingtai-static-check-workflow\scripts\check\run_pyright.ps1 -Target .\test
```

UTF-8 快照与替换：

```powershell
powershell -ExecutionPolicy Bypass -File .\.agents\skills\windows-powershell-utf8-safe-edit\scripts\utf8_snapshot.ps1 -Path .\src\example.py
powershell -ExecutionPolicy Bypass -File .\.agents\skills\windows-powershell-utf8-safe-edit\scripts\utf8_replace.ps1 -Path .\src\example.py -Pattern "old" -Replacement "new"
```

---

## Codex Hooks

- `.codex/hooks.json` 是项目级 hooks 入口。
- `PreToolUse` 在 `apply_patch` 前解析补丁路径，并为已有文件生成 `.archive` 快照。
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
