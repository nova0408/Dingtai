---
name: windows-powershell-utf8-safe-edit
description: 在 Windows PowerShell 5.1 中做文本替换/批量编辑时，强制 UTF-8 安全读写；修改前必须做快照，快照统一放在仓库根目录 `.archive/` 并保留原始目录结构。
---

# Windows PowerShell UTF-8 安全编辑

## 适用场景
- 用 PowerShell 修改 `.py/.md/.json/.toml/.yaml/.ps1` 等文本文件
- 需要 `Regex.Replace` 批量替换或整段重写
- 文件包含中文或其他非 ASCII 字符
- 需要在 Windows PowerShell 5.1 下稳定执行（默认编码与 PowerShell 7 不同）

## 强制规则
1. 小范围修改优先 `apply_patch`。
2. 使用 PowerShell 替换时，读取和写回都必须显式 UTF-8。
3. 修改前必须做快照，且快照必须位于 `<repo>/.archive/`（保留原目录结构）。
4. 快照必须保留原始相对目录结构，禁止把目录打平到文件名里。
5. 替换后必须做最小验证（语法/编译/关键文本检查）。
6. 任何正则替换必须检查“命中数量”，零命中默认视为失败。

## 快照路径规范（强制）
- 快照根：`<repo>/.archive/`
- 结构：`<repo>/.archive/<relative_dir>/<file>.<yyyymmdd-HHmmss>.bak`
- 示例：`src/cli/a.py` -> `.archive/src/cli/a.py.20260410-120501.bak`

## `websocket.py` 实测结果（用于技能回归）
以下问题已在 `websocket.py` 复制件上复现：

1. `Get-Content`（不加 `-Raw`）+ `Set-Content` 会改写文件内容  
`websocket.py` 复制件长度从 `5595` 变为 `5584`，即使你“看起来没改内容”。
2. 正则零命中是静默失败  
`Regex.Replace` 找不到模式时，常见流程会直接写回原文，容易被误判为替换成功。
3. 单引号中的 `` `r`n `` 会被当字面量写入  
`'line1`r`nline2'` 会写入反引号字符，不会产生真实换行。

## 推荐命令行工具（本仓库）
统一放在 `.agents/tools/`：

- `utf8_snapshot.ps1`：生成 `.archive` 快照
- `utf8_replace.ps1`：安全替换（快照 + UTF-8 + 命中校验 + 可选 `py_compile`）

示例：

```powershell
pwsh -File .agents/tools/utf8_snapshot.ps1 -Path websocket.py
pwsh -File .agents/tools/utf8_replace.ps1 -Path websocket.py -Pattern 'ws test' -Replacement 'ws test v2' -ValidatePython
```

## 换行转义防踩坑
1. 不要在单引号字符串里写 `` `r`n `` 期待它变成换行。
2. 需要换行时，优先使用真实换行或 `[Environment]::NewLine`。
3. 替换后做字面量污染检查：

```powershell
Select-String -Path <target_paths> -SimpleMatch '`r`n'
```

## 禁止写法
- `Get-Content`（不加 `-Raw`）读取后直接写回文件
- `Get-Content -Raw` 不带显式编码后直接写回
- `Set-Content` / `Out-File` 不带编码参数写文本（PowerShell 5.1 风险更高）
- 未校验替换结果就批量覆盖多文件
- 用单引号字符串拼接 `` `r`n `` 作为换行
- 快照不放在 `.archive/`
- 把目录结构打平到文件名中

## 输出要求
- 说明是否显式使用 UTF-8 读取与写回
- 说明快照路径（应位于 `.archive/` 且保留目录结构）
- 说明做了哪些最小验证
- 明确未验证项
