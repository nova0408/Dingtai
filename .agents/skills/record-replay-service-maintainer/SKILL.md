---
name: record-replay-service-maintainer
description: 安全维护、迁移、部署、版本管理和评审 Dingtai 根目录 `record_replay/` 服务。用于修改回放业务、Orin API、设备网关、功能、systemd、部署同步或本机/Orin 人工测试入口；强制维护 CHANGELOG 语义版本、区分两端连接与测试目录，并禁止无人值守运行任何 record_replay 测试或 start。
---

# Record Replay Service Maintainer

## 安全红线

1. 任何 `record_replay` 运行入口都视为可能触发机械臂、AGV、夹爪、M11 或升降机构运动。
2. 禁止 Codex、CI、hook、静态检查脚本或无人值守任务执行以下操作：
   - 运行文件名、模块名或测试名包含 `record_replay` 的测试、CLI、冒烟脚本。
   - 执行 `python -m record_replay.service` 做启动冒烟。
   - 向服务发送任何 `POST /start` 请求。
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
5. `record_replay` 可依赖同级 `camera_pipeline` 公共协议与 Orin 已安装第三方包，运行时禁止导入仓库 `src` 或 `test`。
   先验拍摄和手眼计算由同级 `calibration_service` 提供；该服务只读取 RobotControl 状态、
   请求 CameraPipeline 拍摄和执行计算，不提供任何设备控制接口，默认使用 Orin 本机 6600 端口。
   Calibration Service 有独立版本和变更日志；仅新增该独立服务不升级 RecordReplay 版本。
6. 业务行为以 `test/wuji/record_replay_cli.py` 的最新人工验证语义为基准；迁移实现与 CLI 不一致时必须显式列出差异，禁止声称完全等价。
7. 服务数据目录固定，不提供路径参数或环境变量覆盖：
   - `record_replay/prior_data/ball_pose_prior.json` 与 `charuco_board_prior.json` 保存 `test/wuji/prior_record.py` 的人工采集结果。
   - `hand_eye_result.txt`、`charuco_offset_history.csv`、`left_head_base_camera.npy` 和
     `right_head_base_camera.npy` 是 `/start` 前全量检查的部署先验。
   - `record_replay/records/left/` 保存左臂预录 CSV。
   - `record_replay/records/right/` 保存右臂预录 CSV。
8. 服务启动只建立 HTTP 监听，不加载或校验先验；人工 `POST /start` 创建回放业务前必须
   全量检查上述先验，并在错误响应中逐项报告缺失或无效文件，不得因为缺少先验阻止 HTTP 服务启动。
9. 仅允许通过 `POST /prior/ball-pose` 和 `POST /prior/charuco` 替换两个 JSON 先验；请求内容
   必须先校验，成功后原子替换，旧文件备份到服务端 `record_replay/.archive/prior_data/`。
   `.archive` 是服务端运行备份，必须从部署归档和清单中排除。
10. 先验 JSON 与预录 CSV 属于部署源文件，必须和 Python 代码一起参与本机与 Orin 文件清单及 SHA-256 校验。
11. 三球与 Board 检测只调用 Orin 本机部署的 `CameraPipelineClient` 业务方法；`record_replay/` 禁止导入、配置或描述底层 ZMQ。
12. 左右臂 IP 和现场设备地址由 Orin 服务入口固定，不允许本机测试或 API 覆盖。

## CameraPipeline 依赖部署

1. CameraPipeline 公共 client、API 或线协议变化时，把 `camera_pipeline/` 与
   `record_replay/` 作为同一批次同步；禁止只替换 CameraPipeline 后继续运行加载
   旧客户端的 RecordReplay 进程。
2. 部署顺序固定为：确认 RecordReplay 处于 `waiting`、停止 RecordReplay、替换两端
   文件、启动并验证 CameraPipeline、启动并验证 RecordReplay。
3. 停止 RecordReplay 只能是部署中的临时状态；部署成功必须恢复 HTTP 业务就绪。
4. 服务恢复只允许 systemd 启动和只读 `/status` 检查，仍禁止发送 `/start`、运行
   RecordReplay 测试或调用任何硬件执行入口。
5. 从 Windows 执行时优先使用
   `scripts/sync_and_restart_services.ps1`，不要手写 sudo 或拆分部署步骤。
6. 总控脚本允许向 RecordReplay 重启脚本传递 `--non-interactive`，但该模式仅允许
   systemd 重启和只读 `/status` 就绪检查；不得扩展为 `/start` 或任何执行授权。
7. 五服务完整部署的启动顺序为 CameraPipeline、RecordReplay、RobotControl、
   Calibration Service、API Gateway；Calibration Service 只做 6600 端口状态检查，
   不触发拍摄、标定或设备控制。

## 本机与 Orin 同步

按以下顺序部署：

1. 先修改本机源代码并生成 `.archive` 快照。
2. 只把本机 `record_replay/` 的最终内容同步到 `/home/wuji-brain/workspace/record_replay/`。
3. 同步时排除 `__pycache__`、`.pyc`、日志和运行产物。
4. 不做双向合并，不以 Orin 文件覆盖本机源代码。
5. 同步后分别生成相对路径文件清单和 SHA-256；只有两端清单及每个文件哈希完全一致时才报告部署一致。
6. 哈希不一致时停止服务启动与硬件测试，先重新同步并定位差异。

### 测试脚本是非镜像部署

不得根据本机目录层级推断 Orin 目标目录。仅按下列显式映射部署：

| 用途 | 本机唯一源文件 | Orin 执行文件 |
| --- | --- | --- |
| 只读状态 | `test/record_replay/orin/record_replay_static_status.py` | `/home/wuji-brain/workspace/test/record_replay_static_status.py` |
| 人工 start | `test/record_replay/orin/record_replay_start.py` | `/home/wuji-brain/workspace/test/record_replay_start.py` |

1. 禁止创建或同步 `/home/wuji-brain/workspace/test/record_replay/orin/`。
2. 禁止把 `test/record_replay/local/` 中任何文件部署到 Orin。
3. 部署后对上表每一对文件单独比较 SHA-256；不对两端 `test/` 目录做同层级清单比较。
4. 如果 Orin 上存在误创建的镜像目录，先为其文件在本机 `.archive/` 保留快照，再删除精确路径。

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

## 版本与变更日志

1. `record_replay/CHANGELOG.md` 是服务功能版本号的唯一权威来源，当前基线版本为 `2.2.0`。
2. 版本号必须使用 `a.b.c`：
   - `a`：重大重构或重大更新。
   - `b`：功能调整，包括新增、删除或改变 HTTP API、回放流程及设备行为。
   - `c`：缺陷修复和不改变功能边界的优化。
3. 修改公共 client、HTTP API、请求响应字段、回放功能、设备行为、配置语义或部署行为时，必须在同一批改动中升级版本号，并在 CHANGELOG 顶部追加带日期的记录。
4. 同一批改动只升级一次；同时包含多类改动时按影响最大的版本位升级，不得为每个文件分别升级。
5. CHANGELOG 属于 RecordReplay 部署源文件，必须同步到 Orin，并参与相对路径清单和 SHA-256 一致性校验。API 或功能已改变但版本号或日志未更新时，不得报告完成。
6. `RECORD_REPLAY_VERSION` 发生变化后，在用户授权远端部署且本机静态/契约检查通过时，必须尝试执行
   `scripts/sync_and_restart_services.ps1 -RecordReplayOnly`；必须记录本机期望版本、远端实际版本、
   同步文件清单、SHA-256、远端备份、只读 `/status` 和服务状态。远端不可达或同步失败时不得报告
   RecordReplay 版本已更新；整个流程仍不得发送 `/start` 或运行回放测试。

## 测试环境必须分离

### 本机人工测试

1. 只使用 `test/record_replay/local/` 下明确标注为 local 的入口。
2. 本机 RecordReplay HTTP API 固定使用 `http://192.168.1.128:6300`。
3. 禁止把本机 API 改为 `127.0.0.1:6300`，禁止为 RecordReplay API 建立 SSH 端口转发。
4. 本机不直接转发或连接机械臂、qmlinker、AGV 或相机服务。
5. 本机只允许读取状态、读取或修改持久化运行参数，以及在人工安全确认后发送 start。
6. 运行必须由现场人员手动发起，并在触发动作前再次确认安全。

### Orin 人工测试

1. 本机维护源文件时只修改 `test/record_replay/orin/` 中上表指定的两个入口。
2. Orin 上只从 `/home/wuji-brain/workspace/test/record_replay_static_status.py` 或 `record_replay_start.py` 执行。
3. Orin 脚本 RecordReplay HTTP API 固定使用 `http://127.0.0.1:6300`，禁止使用 `192.168.1.128:6300`。
4. Orin 直接访问已部署 API，禁止创建 SSH 转发。
5. 不复用本机测试入口，不通过平台判断在一个文件内切换两套连接方式。
6. 运行必须由现场人员在 Orin 上手动发起；Codex 只能给出命令，不能执行命令。

## 修改与验证流程

1. 先检查 `record_replay/README.md`、API 协议、业务入口和对应人工测试入口。
2. 修改前为已有文件生成 `.archive` 快照。
3. 做最小范围修改，不添加旧 `src.business.record_replay` 兼容层。
4. 更新 README、`API Reference.md` 和 `openapi.yaml`，明确本机/Orin 连接差异、API 行为、
   先验全量检查、JSON 替换及 `.archive` 备份语义和硬件风险。
5. 按改动影响升级 `record_replay/CHANGELOG.md` 的版本号并追加版本记录。
6. 只运行 ruff、pyright、编码扫描、compile-only 验证。
7. 每次涉及测试脚本时，在报告完成前显式检查：
   - local 源文件仅出现 `192.168.1.128:6300`；
   - Orin 源文件仅出现 `127.0.0.1:6300`；
   - Orin 远端两个文件位于 `/home/wuji-brain/workspace/test/` 根层。
8. 最终明确报告：
   - 本机静态验证结果。
   - 是否完成 Orin 文件哈希一致性校验。
   - 未运行任何 record_replay 测试或 start API。
   - 硬件行为未经本回合验证。
