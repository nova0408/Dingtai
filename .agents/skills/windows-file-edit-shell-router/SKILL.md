---
name: windows-file-edit-shell-router
description: Windows 文件编辑路由技能。用于在编辑文本前自动判定应使用 PowerShell 5.1 还是 PowerShell 7，并转交到对应 UTF-8 安全编辑技能执行。
---

# Windows 文件编辑 Shell 路由

## 目标
- 在执行任何批量文本编辑前，先判定 shell。
- 统一转交到专用 skill，避免混用规则。

## 路由流程
1. 探测 `pwsh`：
```powershell
where.exe pwsh
```
2. 探测 `powershell`：
```powershell
where.exe powershell
```
3. 决策：
- `pwsh` 可用：使用 `$windows-powershell-7-utf8-safe-edit`。
- 仅 `powershell` 可用：使用 `$windows-powershell-5-utf8-safe-edit`。
- 两者都不可用：先修复环境，再执行编辑。

## 环境修复建议
1. 安装 `pwsh` 失败且 GitHub 直连不稳定时，必须使用代理 `127.0.0.1:4444`。
2. 安装后应验证：
```powershell
pwsh -NoLogo -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'
```

## 输出约束
1. 明确当前选择的 shell 与原因。
2. 明确调用了哪个专用 skill。
3. 若发生错误，记录可复用原因并要求回写到对应 skill。
