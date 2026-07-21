---
name: record-replay-service-maintainer
description: 安全维护、迁移、部署和评审 Dingtai 根目录 `record_replay/` 服务。用于修改回放业务、Orin API 服务、设备网关、systemd、部署同步、本机 SSH 测试入口或 Orin 直连测试入口；强制本机与 Orin 代码完全一致，并禁止 Codex、CI、hook 或无人值守流程直接运行任何 record_replay 测试或 start 指令，避免机械臂及其他设备意外运动。
---

# Record Replay Service Maintainer

## 安全红线

1. 任何 `record_replay` 运行入口都视为可能触发机械臂、AGV、夹爪、M11 或升降机构运动。
2. 禁止 Codex、CI、hook、静态检查脚本或无人值守任务执行以下操作：
   - 运行文件名、模块名或测试名包含 `record_replay` 的测试、CLI、冒烟脚本。
   - 执行 `python -m record_replay.service` 做启动冒烟。
   - 向服务发送 `{"operation":"start"}`。
   - 调用 `RecordReplayCycleService.run_once()`、`run_forever()` 或等价硬件入口。
3. 即使测试使用 fake，也不得自动执行，避免未来替身失效后静默触发真实设备。
4. 只允许执行不会导入并运行业务入口的静态验证：UTF-8 扫描、ruff、pyright、`py_compile`、`compileall` 和 skill 结构校验。
5. 需要真实运行时必须停止自动操作，明确说明运动风险，由现场人员确认设备区域安全后手动执行。

## 代码边界

1. `record_replay/` 与 `camera_pipeline/` 同级，是 Orin 部署的独立服务，不属于 `src.business`。
2. 本机仓库 `C:\Projects\Dingtai\record_replay` 是唯一源代码。
3. 禁止只修改 `/home/wuji-brain/workspace/record_replay` 中的远端文件。
4. 业务、设备、API 职责保持分离：
   - `context.py` 管理运行资源和状态交换。
   - gateway 负责设备连接与协议收窄。
   - action/executor/cycle 模块负责业务动作与编排。
   - `service/` 只负责 API、进程生命周期和业务桥接，不复制运动逻辑。
5. `record_replay` 可依赖明确的 `src.wuji` 硬件适配和 `camera_pipeline` 协议，但不得依赖 `test` 中的实现。
6. 业务行为以 `test/wuji/record_replay_cli.py` 的最新人工验证语义为基准；迁移实现与 CLI 不一致时必须显式列出差异，禁止声称完全等价。
7. 服务数据目录固定，不提供路径参数或环境变量覆盖：
   - `record_replay/prior_data/ball_pose_prior.json` 与 `charuco_board_prior.json` 保存 `test/wuji/prior_record.py` 的人工采集结果。
   - `record_replay/records/left/` 保存左臂预录 CSV。
   - `record_replay/records/right/` 保存右臂预录 CSV。
8. 先验 JSON 与预录 CSV 属于部署源文件，必须和 Python 代码一起参与本机与 Orin 文件清单及 SHA-256 校验。
9. 三球与 Board 检测只调用 Orin 本机部署的 `CameraPipelineClient` 业务方法；`record_replay/` 禁止导入、配置或描述底层 ZMQ。
10. 左右臂 IP 和现场设备地址由 Orin 服务入口固定，不允许本机测试或 API 覆盖。

## 本机与 Orin 同步

按以下顺序部署：

1. 先修改本机源代码并生成 `.archive` 快照。
2. 只把本机 `record_replay/` 的最终内容同步到 `/home/wuji-brain/workspace/record_replay/`。
3. 同步时排除 `__pycache__`、`.pyc`、日志和运行产物。
4. 不做双向合并，不以 Orin 文件覆盖本机源代码。
5. 同步后分别生成相对路径文件清单和 SHA-256；只有两端清单及每个文件哈希完全一致时才报告部署一致。
6. 哈希不一致时停止服务启动与硬件测试，先重新同步并定位差异。

推荐对两端同一集合进行校验：

```powershell
Get-ChildItem .\record_replay -Recurse -File |
  Where-Object { $_.Extension -ne '.pyc' -and $_.FullName -notmatch '__pycache__' } |
  Sort-Object FullName |
  Get-FileHash -Algorithm SHA256
```

```bash
find /home/wuji-brain/workspace/record_replay -type f \
  ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 \
  | sort -z | xargs -0 sha256sum
```

Windows 与 Linux 输出路径不同，比较前只保留相对于 `record_replay/` 的路径和哈希。

## 测试环境必须分离

### 本机人工测试

1. 只使用 `test/record_replay/local/` 下明确标注为 local 的入口。
2. 本机只建立 Orin 上 RecordReplay HTTP API 的 SSH 转发，不直接转发或连接机械臂、qmlinker、AGV、相机服务。
3. 本机只允许读取状态、读取或修改持久化运行参数，以及在人工安全确认后发送 start。
4. 运行必须由现场人员手动发起，并在触发动作前再次确认安全。

### Orin 人工测试

1. 只使用 `test/record_replay/orin/` 下明确标注为 Orin 的入口或已部署 API 客户端。
2. Orin 直接访问机械臂、qmlinker、AGV 和 camera_pipeline，禁止创建 SSH 转发。
3. 不复用本机测试入口，不通过平台判断在一个文件内切换两套连接方式。
4. 运行必须由现场人员在 Orin 上手动发起；Codex 只能给出命令，不能执行命令。

## 修改与验证流程

1. 先检查 `record_replay/README.md`、API 协议、业务入口和对应人工测试入口。
2. 修改前为已有文件生成 `.archive` 快照。
3. 做最小范围修改，不添加旧 `src.business.record_replay` 兼容层。
4. 更新 README，明确本机/Orin 连接差异、API 行为和硬件风险。
5. 只运行 ruff、pyright、编码扫描、compile-only 验证。
6. 最终明确报告：
   - 本机静态验证结果。
   - 是否完成 Orin 文件哈希一致性校验。
   - 未运行任何 record_replay 测试或 start API。
   - 硬件行为未经本回合验证。
