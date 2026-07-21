# 双臂记录回放服务

本服务从 `test/wuji/record_replay_cli.py` 拆分而来，位于仓库根目录，和
`camera_pipeline` 同级。业务代码不导入 `test`，在 Orin 上直接连接机械臂、
qmlinker、AGV 和 Orin 本机部署的 camera_pipeline，并由 HTTP API 触发一轮执行。
三球和 Board 检测全部通过 `CameraPipelineClient` 的业务接口完成；本服务不导入、配置或
描述 camera_pipeline 内部使用的传输实现，也不自行订阅相机帧或实现检测算法。

## 固定数据目录

服务的数据路径固定在包内，不允许通过启动参数覆盖：

- `record_replay/prior_data/`：`test/wuji/prior_record.py` 记录的先验结果。
  服务固定读取 `ball_pose_prior.json`；同目录同时保存 `charuco_board_prior.json`。
- `record_replay/records/left/`：提前录制的左臂 CSV。
- `record_replay/records/right/`：提前录制的右臂 CSV。

部署时必须把代码、先验 JSON 和两侧 CSV 作为同一个 `record_replay/` 目录同步到 Orin，
并参与文件清单和 SHA-256 一致性校验。

## 循环状态

```text
waiting
  -> navigating_to_start (AGV navigate_to("3") + raw_status 从 busy 变为 idel)
  -> replaying          (按左/右 CSV 执行计划回放)
  -> navigating_to_finish (AGV navigate_to("1") + get_runtime_info 到位确认)
  -> waiting
```

任一阶段失败时，状态进入 `failed`，错误文本写入 `ReplayContext.snapshot()`。
下一轮必须重新发出触发指令。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `contracts.py` | 不可变 CSV、计划和服务状态数据契约。 |
| `settings.py` | 唯一配置页：现场连接、机械臂、手部、offset、AGV 与重试策略 dataclass。 |
| `context.py` | 运行资源、停止事件和可查询状态快照的唯一跨模块边界。 |
| `csv_repository.py` / `execution_plan.py` | CSV 文件发现、解析和左右臂阶段计划构建。 |
| `arm_gateway.py` / `hand_gateway.py` | SDK/qmlinker 设备连接、准备和释放。 |
| `arm_actions.py` / `hand_actions.py` | 单一设备动作语义与旧回放时序。 |
| `dual_arm_executor.py` | 双臂阶段、并行、flush、offset 触发边界。 |
| `offset_*.py` | 三球检测、手眼变换及全局纠偏数据流。 |
| `agv_navigation.py` / `cycle_service.py` | AGV 到位确认与常态化循环。 |

## 状态查询

调用方通过 `ReplayContext.snapshot()` 获得 `ReplayStatusSnapshot`：

- `state`：当前服务阶段；
- `left_csv_state`：当前左臂 CSV 去除 `state_prefix` 后的文件名；
- `plan_index`：当前左臂计划下标；
- `error_text`：失败原因。

## 测试安全红线

`record_replay` 会直接控制机械臂、AGV、夹爪、M11 和升降机构。禁止 Codex、CI、
hook 或其他无人值守流程运行任何 record_replay 测试、启动 service 冒烟，或发送
`{"operation":"start"}`。即使测试当前使用 fake，也只能做静态检查，不能自动执行。

本机与 Orin 测试完全分离：

- `test/record_replay/local/`：本机人工测试。硬件入口必须先建立并验证 SSH 转发。
- `test/record_replay/orin/`：Orin 人工测试。直接连接现场设备或已部署 API，禁止建立 SSH 转发。

任何真实运行都必须由现场人员确认设备运动区域安全后手动触发。

## 本机交互启动

本机入口是 `test/record_replay/local/record_replay_local_manual.py`。它只建立 Orin 上
RecordReplay HTTP API 的 SSH 转发，不连接或转发机械臂、qmlinker、AGV 与相机服务。
入口支持读取状态、读取配置、修改运行参数，以及在完整人工安全确认后发送 start。

```powershell
python test/record_replay/local/record_replay_local_manual.py
```

配置修改由 Orin 服务立即持久化到 `record_replay/config.json`，本机入口同时把服务返回的
完整配置写回本机同名文件，使修改后的数值成为新默认值并保持部署源一致。

### 设备连接参数

`ReplayDeviceConnection` 是唯一进入业务包的现场设备连接数据：

- `left_arm_ip`、`right_arm_ip`；
- `qmlinker_host`、`qmlinker_port`；
- `gripper_port`。

左右臂 IP、qmlinker、gripper 和 AGV 地址固定在 Orin 服务入口，不提供 CLI、配置文件或
本机 API 覆盖。服务直接访问现场设备。本机只转发 `127.0.0.1:6300` 的 HTTP 管理 API。

### 运行策略参数

所有会影响执行时序、速度、重试、容差、三球采样和 AGV 轮询的参数只定义在
`settings.py`：

- `ReplayArmSettings`：NRT、tool/wobj、MoveAbsJ、reset 与机械臂型号；
- `ReplayHandSettings`：夹爪/M11/升降动作与容差；
- `ReplayOffsetSettings`：offset 触发、采样、速度和三球鲁棒聚合；
- `OffsetConfig`：相机名、先验捕获与手眼结果路径；
- `ReplayServiceSettings`：AGV、触发文件及非运动调用重试。

业务模块通过 `ReplayContext.config.settings` 或 `ReplayRuntime.settings` 读取这些数据，
禁止重新声明模块级调试常量。若需现场调参，请在本机入口构造
`ReplayServiceSettings` 的定制实例，再传入 `ReplayCycleConfig`。

## Orin 服务 API

服务入口为 `python -m record_replay.service`，默认监听 `http://0.0.0.0:6300`。
进程启动只建立 API 监听，不会自动执行机械臂动作。调用 `start` 后，业务线程执行
一轮既有的 AGV、CSV、双臂和 offset 流程；重复 `start` 会返回 `accepted=false`，
不会并发控制同一组设备。`status` 可查询当前阶段、CSV 状态、计划下标和错误文本。

HTTP API：`GET /status`、`GET /config`、`POST /config`、`POST /start`。配置更新 body
是字段到数值的 JSON object；服务执行期间拒绝修改配置。

systemd 模板位于 `record_replay/service/record-replay.service`。默认值按 Orin 的
`/home/wuji-brain/workspace` 与现场网段解析。手眼标定结果可通过命令行覆盖；设备地址、
CSV 与先验路径固定；服务不创建 SSH 隧道。

## 本机与 Orin 文件一致性

本机 `record_replay/` 是唯一源代码。禁止只修改 Orin 上的副本。每次部署后必须对
本机与 `/home/wuji-brain/workspace/record_replay/` 生成相对文件清单和 SHA-256，排除
`__pycache__`、`.pyc`、日志和运行产物；只有文件清单与每个哈希都一致时才能报告同步完成。
