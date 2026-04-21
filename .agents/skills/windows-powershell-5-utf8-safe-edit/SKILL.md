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

## 常见故障
- `pwsh` 不存在：继续使用 `powershell` 执行，不阻塞编辑。
- GitHub 下载失败：使用代理 `127.0.0.1:4444` 下载后再执行后续流程。
