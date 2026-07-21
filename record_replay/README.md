# 双臂记录回放服务

本服务从 `test/wuji/record_replay_cli.py` 拆分而来，位于仓库根目录，和
`camera_pipeline` 同级。业务代码不导入 `test`，在 Orin 上直接连接机械臂、
qmlinker、AGV 和 Orin 本机部署的 camera_pipeline，并由 HTTP API 触发一轮执行。
三球和 Board 检测全部通过 `CameraPipelineClient` 的业务接口完成；本服务不导入、配置或
描述 camera_pipeline 内部使用的传输实现，也不自行订阅相机帧或实现检测算法。

## 固定数据目录

服务的数据路径固定在包内，不允许通过启动参数覆盖：

- `record_replay/prior_data/`：`test/wuji/prior_record.py` 记录的先验结果。
  服务固定读取 `ball_pose_prior.json` 和 `hand_eye_result.txt`；同目录
  同时保存 `charuco_board_prior.json`。
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
| `arm_gateway.py` / `hand_gateway.py` | SDK/qmlinker 设备连接、准备和释放；直接持有 qmlinker 原生对象。 |
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

- `test/record_replay/local/`：本机人工测试，固定访问 `http://192.168.1.128:6300`。
- `test/record_replay/orin/`：Orin 脚本的本机源目录；部署时两个指定脚本平铺到
  `/home/wuji-brain/workspace/test/`，不镜像该目录层级。

任何真实运行都必须由现场人员确认设备运动区域安全后手动触发。

## 本机交互启动

本机入口是 `test/record_replay/local/record_replay_local_manual.py`。它直接访问 Orin 管理网地址
`http://192.168.1.128:6300`，不连接或转发机械臂、qmlinker、AGV 与相机服务。
入口支持读取状态、读取配置、修改运行参数，以及在完整人工安全确认后发送 start。

HTTP 客户端的公共入口是 `record_replay.client.RecordReplayClient`。默认连接
`http://192.168.1.128:6300`：

```python
from record_replay.client import RecordReplayClient

client = RecordReplayClient()
print(client.get_status())
```

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
本机 API 覆盖。服务直接访问现场设备；本机只访问 `192.168.1.128:6300` HTTP API。

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

HTTP API：`GET /status`、`GET /config`、`GET /device-status`、`POST /config`、`POST /start`。配置更新 body
是字段到数值的 JSON object；服务执行期间拒绝修改配置。`POST /start` 必须提供
`{"enable_agv_navigation": true|false}`；为 `false` 时同时跳过去起点与返回终点导航，
双臂 CSV 回放仍会执行。

Orin 人工测试的本机源文件与远端执行文件是非镜像映射：

- `test/record_replay/orin/record_replay_static_status.py` →
  `/home/wuji-brain/workspace/test/record_replay_static_status.py`；
- `test/record_replay/orin/record_replay_start.py` →
  `/home/wuji-brain/workspace/test/record_replay_start.py`。

两个 Orin 脚本都固定访问 `http://127.0.0.1:6300`。状态脚本只读；start 脚本需完整
安全确认，默认禁用 AGV，支持 `--agv` 或 `--no-agv`。禁止部署到
`/home/wuji-brain/workspace/test/record_replay/orin/`。

### 设备诊断接口

`GET /device-status` 只读取设备，不调用任何上电、使能、标定或运动指令。回放任务
运行期间拒绝诊断，避免同一设备被并发访问。响应包含：

- `all_connected`：双臂、夹爪、Head 和 Lift 是否全部完成有效状态读取；
- `left_arm` / `right_arm`：`connected`、`error`、IP、预期/实际机型、UID、
  `operate_mode`、`operation_state`、`power_state` 和 `powered_on`；
- `gripper`：`connected`、`online`、`calibrated`、`enabled`、`position` 和 `state`；
- `head`：`connected`、`enabled`、`yaw_deg` 和 `pitch_deg`；
- `lift`：`connected`、`enabled` 和 `height_mm`。

每个设备独立返回 `connected/error`，单项断线不会遮蔽其他设备的诊断结果。双臂会对
IP 上控制器实际上报机型与左右臂预期机型进行一致性校验。本机可直接调用：

```python
from record_replay.client import RecordReplayClient

print(RecordReplayClient().get_device_status())  # 设备串行读取，该方法默认超时 30 s
```

systemd 模板位于 `record_replay/service/record-replay.service`，并声明依赖
`camera-pipeline.service`。默认值按 Orin 的
`/home/wuji-brain/workspace` 与现场网段解析。设备地址、CSV、先验与手眼标定路径均固定；
服务不创建 SSH 隧道。

`record_replay/` 是独立部署包，运行时代码禁止导入仓库 `src` 或 `test`。设备 gateway
直接创建并持有 qmlinker 原生对象，不再添加二次客户端封装；姿态换算依赖 Orin `wuji`
环境中已安装的 NumPy 与 SciPy。

### Orin 人工重启

现场人员确认设备运动区域安全后，可在 Orin 上交互执行：

```bash
bash scripts/restart_record_replay_service.sh
```

脚本显示 `我已确认现场安全并同意重启RecordReplay服务 [Y/n]`，输入 `Y`、`y` 或
直接回车继续，输入 `n` 取消；非交互环境直接拒绝。它会先检查三球先验、左右臂 CSV，
并在已有服务状态不是 `waiting` 或状态不可查询时拒绝停止进程。停止阶段只发送 SIGTERM；
若服务不能干净退出则停止操作，不使用 SIGKILL。脚本只重启 HTTP API 服务，不发送
`POST /start`，因此不会主动开始一轮回放。

## 本机与 Orin 文件一致性

本机 `record_replay/` 是唯一源代码。禁止只修改 Orin 上的副本。每次部署后必须对
本机与 `/home/wuji-brain/workspace/record_replay/` 生成相对文件清单和 SHA-256，排除
`__pycache__`、`.pyc`、日志和运行产物；只有文件清单与每个哈希都一致时才能报告同步完成。
