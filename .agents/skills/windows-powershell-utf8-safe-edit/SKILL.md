---
name: windows-powershell-utf8-safe-edit
description: Windows 文本编辑兼容入口技能。用于在不确定 shell 版本时统一执行 UTF-8 安全编辑流程，并路由到 PowerShell 5.1 或 PowerShell 7 专用技能；包含快照规范、网络代理下载规范与经验回写要求。
---

# Windows PowerShell UTF-8 安全编辑（兼容入口）

## 目标
- 统一入口：先判断 shell，再路由到专用技能执行。
- 保留兼容：历史任务仍可直接使用本技能，不中断旧链路。

## 使用规则
1. 先使用 `$windows-file-edit-shell-router` 判定执行环境。
2. 判定为 PowerShell 7 时，使用 `$windows-powershell-7-utf8-safe-edit`。
3. 判定为 PowerShell 5.1 时，使用 `$windows-powershell-5-utf8-safe-edit`。
4. 若无法判定，按“`pwsh` 优先，`powershell` 回退”的策略执行。
5. 每次遇到可复用错误，新增到对应专用技能，并把重复条目压缩整合。

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
