# 通用静态检查入口

本目录提供仓库级通用检查脚本，固定使用 `DingTai` 环境，避免不同终端环境导致结果漂移。

## 脚本

- `run_ruff.ps1`：运行 `ruff check`，可选 `-Fix` 自动修复。
- `run_pyright.ps1`：运行 `pyright` 类型检查。
- `run_all_checks.ps1`：先 `ruff` 后 `pyright`，任一失败即返回非 0。

## 用法

```powershell
powershell -ExecutionPolicy Bypass -File .\.agent\tools\check\run_all_checks.ps1 -Target .
```

仅检查某个目录或文件：

```powershell
powershell -ExecutionPolicy Bypass -File .\.agent\tools\check\run_ruff.ps1 -Target .\src
powershell -ExecutionPolicy Bypass -File .\.agent\tools\check\run_pyright.ps1 -Target .\test\pointcloud
```
