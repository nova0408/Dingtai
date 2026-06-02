---
name: windows-powershell-utf8-safe-edit
description: Windows 文本编辑入口技能。用于在不确定 shell 版本时统一执行 UTF-8 安全编辑流程，并路由到 PowerShell 5.1 或 PowerShell 7 专用技能；包含快照规范、hooks、网络代理下载规范与经验回写要求。
---

# Windows PowerShell UTF-8 安全编辑入口

## 目标
- 统一入口：先判断 shell，再路由到专用技能执行。
- 与 `.codex/hooks` 配合，在文件编辑前后执行快照、编码扫描和静态检查。

## 使用规则
1. 先使用 `$windows-file-edit-shell-router` 判定执行环境。
2. 判定为 PowerShell 7 时，使用 `$windows-powershell-7-utf8-safe-edit`。
3. 判定为 PowerShell 5.1 时，使用 `$windows-powershell-5-utf8-safe-edit`。
4. 若无法判定，按“`pwsh` 优先，`powershell` 回退”的策略执行。
5. 每次遇到可复用错误，新增到对应专用技能，并把重复条目压缩整合。
6. 面向 Windows 5.1 的可执行脚本避免硬编码中文路径默认值；优先 `$PSScriptRoot + 相对目录`，并保存为 `UTF-8 with BOM`，防止运行时路径乱码。
7. 对特殊字符路径统一执行：UTF-8 控制台编码 + Unicode Form C 归一化 + `-LiteralPath` 文件操作。
8. 仓库级可执行脚本位于本 skill 的 `scripts/`，禁止回退到旧的 `.agents/tools` 平铺目录。
9. `.codex/hooks/pre_tool_use_safe_edit.ps1` 在 `apply_patch` 前为已有文件快照。
10. `.codex/hooks/post_tool_use_static_check.ps1` 在 `apply_patch` 后执行 UTF-8 扫描、确定性文本修复、`ruff --fix` 和 `pyright`。

## 最小入口流程
```powershell
where.exe pwsh
where.exe powershell
```

路由策略：
- `pwsh` 可用：默认走 PS7 专用技能。
- `pwsh` 不可用：走 PS5.1 专用技能。

## 相关专用技能
- `.agents/skills/windows-powershell-5-utf8-safe-edit/SKILL.md`
- `.agents/skills/windows-powershell-7-utf8-safe-edit/SKILL.md`
- `.agents/skills/windows-file-edit-shell-router/SKILL.md`
