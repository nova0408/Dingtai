---
name: windows-powershell-5-utf8-safe-edit
description: 在 Windows PowerShell 5.1 下执行 UTF-8 安全文本编辑。用于 `powershell` 环境或 `pwsh` 不可用时，要求编辑前快照、显式 UTF-8 读写、替换命中校验与最小验证。
---

# Windows PowerShell 5.1 UTF-8 安全编辑

## 强制规则
1. 小范围修改优先 `apply_patch`。
2. 修改前必须生成 `.archive` 快照并保留目录结构。
3. 读取与写回必须显式 UTF-8。
4. 正则替换必须检查命中数量，零命中视为失败。
5. 替换后至少执行一项最小验证（如 `py_compile`/关键文本检查）。

## 执行命令模板
```powershell
powershell -File .agents/tools/utf8_snapshot.ps1 -Path <relative-path>
powershell -File .agents/tools/utf8_replace.ps1 -Path <relative-path> -Pattern '<pattern>' -Replacement '<replacement>'
```

## 编码与换行注意事项（5.1 重点）
1. 禁止 `Get-Content`（不加 `-Raw`）读后直接写回。
2. 禁止在单引号字符串中拼 `` `r`n `` 期待真实换行。
3. 替换后检查是否引入字面量 `` `r`n ``：
```powershell
Select-String -Path <target-path> -SimpleMatch '`r`n'
```
4. 对会被 `powershell -File` 直接执行的脚本，优先保存为 `UTF-8 with BOM`，避免 5.1 把 UTF-8 无 BOM 按 ANSI 解释。
5. 在脚本参数默认值中避免硬编码中文路径字面量；优先用 `$PSScriptRoot` 组合相对路径（如 `Join-Path $PSScriptRoot 'GeoTransformer'`），从根源规避路径乱码目录被创建。
6. 涉及特殊字符路径（中文、重音、组合字符）时，写路径前先做 Unicode 归一化（Form C），并优先使用 `-LiteralPath` 参数，避免被通配符或编码差异误解析。
7. 下载/解压脚本可显式设置：
`[Console]::InputEncoding/OutputEncoding` 与 `$OutputEncoding` 为 UTF-8，降低日志与参数传递乱码风险。

## 常见故障
- `pwsh` 不存在：继续使用 `powershell` 执行，不阻塞编辑。
- GitHub 下载失败：使用代理 `127.0.0.1:4444` 下载后再执行后续流程。
