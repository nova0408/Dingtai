---
name: pyinstaller-packaging
description: Windows 下 Python 项目使用 PyInstaller 打包的通用技能，覆盖 spec 布局、Qt 插件部署、动态库收集、资源打包与最小验收流程。
---

# PyInstaller 打包通用技能（Windows）

## 目标
给出可复现、可交付的 Windows 打包流程，避免“本机能跑、打包后崩溃”。

## 基线流程
1. 在目标 conda 环境安装 `pyinstaller`。
2. 以项目根目录 `*.spec` 作为唯一打包入口。
3. 执行：
```powershell
conda run -n <env_name> pyinstaller --noconfirm --clean <project>.spec
```
4. 产物默认在：`dist/<app_name>`，中间文件在：`build/<app_name>`。

## spec 设计要点
- `datas`：配置、模板、QSS、模型、脚本等非 Python 资源必须显式收集。
- `binaries`：`ctypes.LoadLibrary` 依赖的 DLL 必须显式收集。
- `runtime_hooks`：用于修正运行时环境变量（特别是 Qt）。
- 对需要源码反射/JIT 的包（如部分 TorchScript 场景）使用 `module_collection_mode={'pkg':'py'}`。

## Qt 项目额外约束
1. 平台插件目录：`<exe_dir>/platforms/qwindows.dll`。
2. `qt.conf` 放在 exe 同级目录，并使用相对路径。
3. 启动前清理外部 Qt 环境变量污染（IDE、系统环境）。
4. 必要时设置：
- `QT_PLUGIN_PATH=<exe_dir>`
- `QT_QPA_PLATFORM_PLUGIN_PATH=<exe_dir>/platforms`

## 路径策略
- 统一从运行根目录推导资源路径（`root_dir` 语义一致）。
- 不要混用多个根目录语义，避免源码运行/打包运行行为分叉。

## 最小验收清单
1. 文件存在性检查：
- exe 文件
- 关键 DLL
- 关键资源文件（配置、QSS、模板）
2. 启动冒烟：
- 应用可启动到主窗口或主流程入口
3. 错误回归：
- 对历史报错（DLL 缺失、Qt 插件、资源路径）逐条复测

## 交付记录建议
每次打包输出以下信息：
- 使用的环境名与 Python 版本
- spec 文件路径
- 打包命令
- 产物路径
- 已验证项与未验证项（硬件相关需单独声明）
