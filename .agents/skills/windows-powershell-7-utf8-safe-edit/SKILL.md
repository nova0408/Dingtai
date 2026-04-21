---
name: windows-powershell-7-utf8-safe-edit
description: 在 PowerShell 7 (`pwsh`) 下执行 UTF-8 安全文本编辑。用于已安装 `pwsh` 的环境，要求编辑前快照、显式 UTF-8 读写、替换命中校验、最小验证，并支持代理下载回退。
---

# PowerShell 7 UTF-8 安全编辑

## 强制规则
1. 小范围修改优先 `apply_patch`。
2. 修改前必须生成 `.archive` 快照并保留目录结构。
3. 文本读写必须显式 UTF-8，避免跨工具链编码漂移。
4. 正则替换必须校验命中数量，零命中视为失败。
5. 替换后执行最小验证（如 `py_compile`、关键断言、目标片段检查）。

## 执行命令模板
```powershell
pwsh -File .agents/tools/utf8_snapshot.ps1 -Path <relative-path>
pwsh -File .agents/tools/utf8_replace.ps1 -Path <relative-path> -Pattern '<pattern>' -Replacement '<replacement>'
```

## 回退策略
1. 若 `pwsh` 不可用，自动回退到 `powershell` 并切换 `windows-powershell-5-utf8-safe-edit`。
2. 若下载或安装依赖失败，优先走代理：
```powershell
curl.exe -x http://127.0.0.1:4444 -L "<github-url>" -o "<local-file>"
```

## 故障经验维护
1. 每次出现可复用错误，都要新增到本技能或对应技能。
2. 更新时合并同类条目，避免技能膨胀和重复描述。
