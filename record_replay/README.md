# 双臂记录回放服务

当前 RecordReplay 服务业务语义版本：`3.9.0`；对应人工验证入口
`test/wuji/record_replay_cli.py` 的版本为 `1.10.1`。两者均使用 M6 右手动作语义。

面向 GUI 和其它项目的完整 HTTP 契约见 [API Reference](API%20Reference.md)，机器可读描述见
[OpenAPI](openapi.yaml)。

本服务从 `test/wuji/record_replay_cli.py` 拆分而来，位于仓库根目录，和
`camera_pipeline` 同级。业务代码不导入 `test`；机械臂通过 Orin 本机 RobotControl 服务访问，
手部、AGV 与 CameraPipeline 仍由各自网关管理，并由 HTTP API 触发一次执行。
三球和 Board 检测全部通过本服务内的 `camera_client.py` HTTP 协议适配完成；本服务不导入
CameraPipeline Python 包、不引用仓库 `src/`，也不自行订阅相机帧或实现检测算法。

## 固定数据目录

服务的数据路径固定在包内，不允许通过启动参数覆盖：

- `record_replay/prior_data/`：GUI 先验标定页或 `test/wuji/prior_record.py` 记录的先验结果。
  服务固定读取 `ball_pose_prior.json` 和 `hand_eye_result.txt`；同目录
  同时保存 `charuco_board_prior.json`。三球尺寸字段统一为 `diameter_mm`，单位 mm；
  不再接受旧的 `radius_mm` 先验文件。重新记录的三球条目还包含 `hsv_ranges` 和
  `observed_color_hex`；颜色和坐标语义从先验文件的 `local_coordinate_frame`
  读取，服务端不维护固定球色。服务启动只建立 HTTP 监听，不加载或校验先验。人工调用
  `POST /start` 时才会完整检查全部先验；无效或缺失项会在响应中逐项列出，不会回退到占位
  球心。三球运行时只依赖 `ball_pose_prior.json`，`ball_debug_overlay.jpg` 属于本地调试
  证据，不是远端运行依赖。
  `ball_pose_prior.json` 中的 `tcp_pose_matrix` 由 AR5 SDK 原始 `trans(m)+rpy(rad)`
  按小写外禀 `xyz` 直接构造，矩阵平移保持 m；`tcp_translation_mm` 和
  `tcp_rpy_degrees` 仅用于人工查看。GUI 根据 30 帧颜色波动生成窄 HSV 范围；
  Hue 使用周期为 180 的环形统计并保持 6–8 的半宽，红色跨越 179/0 时保存为两段，
  S/V 下限不高于 140/120，以免只分割出球体高饱和高亮局部。
- `record_replay/records/left/`：提前录制的左臂 CSV。
- `record_replay/records/right/`：提前录制的右臂 CSV。
- `record_replay/action_sequence.json`：统一承载动作顺序、动作类别、每项 speed、zone、offset
  策略和先验文件入口。四个托盘位置 index 不写入该文件，只由每次 `GET /plan` 或
  `POST /start` 请求传入。

CSV 文件名使用 `<action>_<arm>[_<timestamp>].csv`；多目标动作使用
`<action>_<index>_<arm>[_<timestamp>].csv`，例如 `get_tray_1_left_20260630_154830.csv`。
录制时也兼容在最前面增加纯数字前缀，例如 `01_go_out_left.csv`、`10_get_new_tray_left.csv`、
`11_before_put_new_tray_left.csv`。该前缀只用于动作名匹配，不改写实际文件名；执行、状态展示和
CSV 行记录仍按磁盘上的完整文件名处理。`Sxx` 仍不参与动作名解析。

部署脚本只同步 `record_replay/` 中的服务代码和静态配置；`prior_data/` 与 `records/` 属于远端
现场数据目录，不进入上传包、部署清单或 SHA-256 校验。替换服务代码时会原样保留 Orin 上已有的
两个目录，本机同名目录中的先验和 CSV 不会覆盖远端数据。

仓库根目录的 `scripts/sync_and_restart_services.ps1` 支持 RecordReplay 专用同步：

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RecordReplayOnly
```

该入口只同步并重启 `record-replay.service`，替换前检查 RecordReplay 处于 `idle`/`waiting` 且
CameraPipeline 已在 6200 端口就绪，同步后校验代码文件清单、SHA-256、只读 `/status` 和服务
版本；不会同步或重启其它服务，也不会发送 `POST /start`。`prior_data/`、`records/`、
`runtime_state.json` 等现场数据或运行产物不纳入部署清单。仅重启而不替换文件时可使用
`-RestartOnly -RecordReplayOnly`，同样只执行只读 `/status` 就绪检查。

`-RestartOnly` 支持四个服务并按依赖顺序收敛重启：

- `-CameraPipelineOnly`：CameraPipeline → RecordReplay → API Gateway；
- `-RecordReplayOnly`：RecordReplay → API Gateway；
- `-RobotControlOnly`：RobotControl → API Gateway；
- `-ApiGatewayOnly`：仅 API Gateway；
- 不指定 `*Only`：CameraPipeline → RecordReplay → RobotControl → API Gateway。

这些重启只通过 systemd 执行，并等待各服务只读健康检查；不会发送 RecordReplay `/start`。

服务提供两个 JSON 先验替换接口：`POST /prior/ball-pose` 和
`POST /prior/charuco`。替换前旧文件会备份到服务端
`record_replay/.archive/prior_data/<时间戳>/`，不会直接删除；上传内容先通过对应文件格式校验，
校验失败不会替换现有文件。

`test/wuji/record_replay_cli.py` 是本机直连硬件的人工验证入口，与 Orin HTTP 服务的数据
位置不同：它读取本机 `record_left/`、`record_right/`，先验读取 GUI 写入的本机
`record_replay/prior_data/`。本服务的动作顺序和每项 speed/zone 只来自
`action_sequence.json`；服务不再根据 CSV 文件名推断执行顺序。
Orin HTTP 服务只读取已部署到 `/home/wuji-brain/workspace/record_replay/records/` 和
`prior_data/` 的远端副本。

人工 CLI 在启用头部 ChArUco 纠偏前读取
`record_replay/prior_data/charuco_offset_history.csv`。全部机械臂共用全局历史样本池，至少需要
6 条 `accepted=true` 的有效历史；`arm_side` 只记录数据来源，不参与筛选。xyz/rpy 各分量必须位于历史均值 ±4σ 内，平移和旋转
模长还必须同时低于历史 4σ 上界及 60 mm/5° 绝对安全上限。运行时只读该 CSV，
不会自动追加检测结果；新增实验数据必须先由人工判断可靠性，再手动录入并设置
`accepted`。单次候选 offset 未通过安全检查时会重新检测目标板并计算，最多尝试
3 次；三次均不通过才拒绝本轮执行。异常拒绝发生在 offset 写入运行时及后续纠偏
运动之前。

启用 ChArUco 纠偏时还必须部署 `left_head_base_camera.npy` 和
`right_head_base_camera.npy`，分别表示对应机械臂基坐标系下的 `T_base_camera`；
服务不会从其他目录猜测或回退读取这两个矩阵。

运行期三球纠偏使用分级检测：先移除标定 HSV 限制，以 HEX 推导的宽范围确认三个球均
可检出；随后使用标定窄范围复检。窄范围三球球心与宽范围逐球差异不超过 8 mm 时采用
窄范围结果，否则记录明确告警并回退宽范围结果。宽、窄请求始终使用同一组物理直径、
模型坐标和先验文件声明的颜色顺序，因此回退不会绕过尺寸或三球几何约束。每个宽、窄
阶段最多尝试 3 次，降低稳定帧中偶发遮挡或检测波动造成的整轮失败。

## 循环状态

```text
idle
  -> busy               (AGV、设备准备、CSV 执行和资源清理)
  -> idle
  -> rapid_stop         (人工 stop 或运动阶段失败，等待人工 reset)
```

状态快照另外提供 `execution_phase`，用于区分 `busy` 内部的具体阶段：
`agv_navigation`、`preparing_devices`、`initializing_charuco`、`waiting_action_start`、
`executing_action`、`updating_offset` 和 `releasing_resources`。`state` 仍表示顶层服务状态，
GUI 不需要根据当前动作字段反推阶段。

正常完成后回到 `idle`；人工 stop 或运动阶段失败时先停止 AGV/已连接左右 AR5，再进入 `rapid_stop`，并将错误文本写入
`ReplayContext.snapshot()`。`run_once()` 执行边界也会拒绝已锁存的 `rapid_stop` 或停止事件。
只有人工处理后调用 `POST /reset` 才能恢复 `idle`，不会自动续跑。
如果进程在 `busy` 时因原生 SDK 致命信号等原因异常退出，重启后会锁存为 `rapid_stop`，并在
`error_code/error_text` 中说明这是持久化状态恢复结果；必须结合上一进程 journal 排查后人工复位。
每侧 AR5 的准备、轨迹队列提交和 `robot.stop()` 由独立命令锁串行化；运动等待不持有该锁，
以避免停止请求与新的 `moveAppend`/`moveStart` 并发交错。AGV 与左右 AR5 的 Stop 调用并行发起，
避免单个 Stop 超时延后其它设备停止。

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
| `agv_navigation.py` / `cycle_service.py` | AGV 到位确认与单次执行。 |

## 状态查询

调用方通过 `ReplayContext.snapshot()` 获得 `ReplayStatusSnapshot`：

- `state`：当前服务阶段；
- `execution_phase`：当前具体执行阶段；空闲为 `idle`，异常锁存为 `rapid_stop`；
- `left_csv_state`：当前左臂命名动作对应的 CSV 文件 stem；
- `plan_index`：当前左臂计划下标；
- `error_text`：失败原因；
- `error_code`：稳定机器可读错误码；正常为 `null`，错误说明见 API Reference；
- `left_csv_files` / `right_csv_files`：本轮 JSON 实际引用的左右臂 CSV 文件名与数据行数；未引用的可选 index CSV 不计入本轮摘要；
- `execution_tasks`：按实际执行顺序展开的任务清单；每个一级任务同时给出
  `left_csv`、`right_csv` 和 `synchronized`，单臂阶段另一侧为空；
- `current_task_sequence`：当前执行任务在 `execution_tasks` 中的序号，从 1 开始；
- `current_task_active`：当前任务序号对应的任务是否仍在执行；任务完成后到下一任务开始前
  为 `false`，避免界面把 AGV 返航等阶段误显示为仍在执行最后一个 CSV；
- `total_execution_count`：服务进程本次启动以来累计成功完成的执行次数；服务重启后从
  0 重新计数；每次 start 仅在执行完成时加一；
- `old_tray_current_index` / `old_tray_put_index`：本次执行使用的旧托盘当前位置和放置位置 index；
- `new_tray_current_index` / `new_tray_put_index`：本次执行使用的新托盘当前位置和放置位置 index；
- `agv_navigation_enabled` / `agv_target`：本次执行的 AGV 开关与目标；
- `current_left_csv` / `current_right_csv`：左右臂当前正在处理的 CSV；
- `current_left_action_name` / `current_right_action_name`：左右臂当前命名动作；
- `current_left_action_index` / `current_right_action_index`：多目标动作 index，普通动作为空；
- `current_left_row` / `current_right_row`：左右臂当前处理的 CSV 源数据行；
- `current_left_total_rows` / `current_right_total_rows`：当前 CSV 总数据行数。
- `offset_statuses`：固定包含 `head` 和 `three_ball` 两项，分别显示本轮 offset 是否可用及
  当前动作是否应用；同一动作不会同时应用两种 offset。

部署清单在服务启动和每轮开始前刷新。当前行用于界面定位服务正在调度或处理的源数据行；
连续机械臂轨迹会批量提交给控制器，因此该值不表示控制器已经物理到达对应轨迹点。

## 测试安全红线

`record_replay` 会直接控制机械臂、AGV、夹爪、M6 和升降机构。禁止 Codex、CI、
hook 或其他无人值守流程运行任何 record_replay 测试、启动 service 冒烟，或发送
`POST /start`。即使测试当前使用 fake，也只能做静态检查，不能自动执行。

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

所有会影响执行时序、重试、容差、三球采样和 AGV 轮询的默认策略定义在
`settings.py`；允许现场修改的运行参数由 `service/config_store.py` 持久化：

- `ReplayArmSettings`：NRT、tool/wobj、reset 与机械臂型号；
- `ReplayHandSettings`：夹爪/M6/升降动作与容差；
- `ReplayOffsetSettings`：offset 触发、采样、ChArUco 安全门和三球鲁棒聚合；
- `OffsetConfig`：相机名、先验捕获与手眼结果路径；
- `ReplayServiceSettings`：AGV、触发文件及非运动调用重试。

业务模块通过统一动作 JSON 冻结后的 `ReplayContext.config.settings` 或
`ReplayRuntime.settings` 读取这些数据，
禁止重新声明模块级调试常量。动作 speed/zone 的现场修改直接编辑
`action_sequence.json`，服务只在 `idle` 状态的下一次 `POST /start` 前重新读取并完整校验；
进入 `busy` 后使用已冻结的内存计划，运行中修改磁盘文件不会改变当前轮次。
该计划同时保存 JSON 引用的 CSV 行快照，执行器不会在 busy 期间重新读取这些 CSV。
右手 M6 每次下发目标后按 0.1 s 轮询六轴实际状态，连续 3 次采样中每个轴的最大值与最小值
差值均不超过 0.1 时认为运动结束；5 s 内未稳定只记录告警并继续后续 CSV，不与下发目标值比较。
顺序 JSON 的左右数组都必须非空；fast 动作要求非零 zone，precise 动作固定 zone 为 0，
capture 必须显式提供 `final_speed` 与 `settle_delay`，且普通动作不得携带这些专用字段。
SDK 原始 speed 边界为 `(0, 4000]` mm/s，服务为现场安全将 JSON speed/final_speed 收紧为 `[5, 4000]` mm/s，
zone 限制为 `[0, 200]` mm；
SDK 内部还会把 speed 映射为 5 档（`<100`、`100~200`、`200~500`、`500~800`、`>800` mm/s），
把 zone 映射为 4 档（`<1`、`1~20`、`20~60`、`>60` mm）；JSON 保留原始数值，允许每个动作独立调整。
这些是 SDK 运动接口的工程边界和分档规则，不代表现场已经确认每个动作的安全值。
fast 的每个 arm 点使用动作项 zone，precise 的每个 arm 点固定使用 `zone=0`；capture 的前置 arm 点使用动作项 zone，
最终拍摄点固定使用 `zone=0` 并使用 `final_speed`。capture 的通用到位和稳定等待已实现；`calibration` 到位后调用已有 CameraPipeline 三球检测，
`calibration_new_tray` 当前绑定空 CSV 并留空。新增算法入口必须保持显式函数调用，不得改成字符串回调表。

## Orin 服务 API

服务入口为 `python -m record_replay.service`，默认监听 `http://0.0.0.0:6300`。
进程启动只建立 API 监听，不会自动执行机械臂动作，并处于 `idle` 状态。调用 `start` 被接受后
立即进入 `busy`，直到 AGV、设备准备、CSV 回放和资源清理全部结束，再恢复 `idle`。业务线程执行
一次既有的 AGV、CSV、双臂和 offset 流程；重复 `start` 会返回 HTTP `400` 的 JSON 错误对象，
不会并发控制同一组设备。`status` 可查询当前阶段、部署的左右臂 CSV、左右臂对齐的
实际执行任务、当前任务序号、服务累计执行次数、各臂当前 CSV、源数据行进度、计划下标
和错误文本。
GUI 可以轮询只读 `/status`，也可以订阅 `wss://<orin-host>/api/v1/record-replay-ws`。
连接建立后立即收到当前状态，后续推送 `record_replay.status`；一次成功完成时额外推送
`record_replay.completed` 结束事件，计数已同步加一。状态和结束事件同时包含 `error_code` 与中文
`error_text`。HTTP 非 2xx 响应统一返回 `{"error_code":"...","error_text":"中文说明"}`，
其中 `400` 表示请求或业务拒绝、`404` 表示路径不存在、`405` 表示方法不适用于已知路径、
`500` 表示服务内部错误。响应头包含 `X-Request-ID`；未知异常的 JSON 会给出同一请求 ID、异常类型和限长原因，
服务端日志保留完整堆栈。短暂断网后重新连接即可恢复；GUI
负责根据四个托盘位置 index 编排下一次 start，服务不提供循环执行。

HTTP API：`GET /health`、`GET /status`、`GET /plan?old_tray_current_index={old_current}&old_tray_put_index={old_put}&new_tray_current_index={new_current}&new_tray_put_index={new_put}`、`GET /config`、`GET /device-status`、`POST /config`、
`POST /prior/ball-pose`、`POST /prior/charuco`、`POST /start`、`POST /stop`、
`POST /reset`。`GET /plan` 只在 `idle` 时读取并校验本次动作 JSON 与实际 CSV，返回 GUI 执行前展示所需的 CSV、动作类型、speed、zone、index 和行数；它不连接设备、不创建线程，也不允许通过 HTTP 修改这些字段。配置更新 body
只包含非动作数字参数；动作 speed/zone 不通过 HTTP 任意修改，必须编辑顺序 JSON，且服务 `busy` 期间拒绝修改配置。两个 prior 接口接收完整 JSON，
仅在服务 `idle` 且没有活动回放线程时允许替换；校验通过后原子替换并将旧文件备份到服务端 `.archive/prior_data/<时间戳>/`。`POST /start` 必须提供
`{"old_tray_current_index": 1, "old_tray_put_index": 4, "new_tray_current_index": 1, "new_tray_put_index": 1, "enable_agv_navigation": false, "agv_target": "1"}`；服务只执行这一组四位置对应的单次计划。为 `false` 时不执行回放前导航；为 `true` 时导航到传入的 `agv_target`，回放完成后不自动返航。调用 `/start` 前会全量检查所有运行先验，缺失或无效文件会逐项写入
`error_text`；服务启动本身不会因先验缺失失败。HTTP 非 2xx 响应统一使用
`{"error_code":"...","error_text":"中文说明"}`。

四个 index 分别绑定 `get_tray`、`put_tray`、`get_new_tray`、`put_new_tray`。四个位置的下一次
取值和异常恢复由 GUI 决定，服务本身不循环。

正式客户端必须通过统一 Gateway 访问本服务：HTTP 使用 `https://<orin-host>` 配置
`/api/v1/record-replay` 前缀，状态订阅使用 `wss://<orin-host>` 配置
`/api/v1/record-replay-ws` 前缀。Gateway 只统一客户端地址和 URL 前缀，不会合并进程，
也不会移除 RecordReplay 实际监听的 `6300` 端口；6300 仅用于人工测试、Orin 本地只读
诊断和故障排查，不作为正式客户端入口。完整映射见
[`api_gateway/README.md`](../api_gateway/README.md)。RecordReplay 的 `POST /start`
仍然只能由现场人员明确手动发起，不能通过统一入口被自动化测试调用。客户端首次使用前
必须安装并信任 CasiaHand Root CA，安装指南见
[`api_gateway/certificates/README.md`](../api_gateway/certificates/README.md)；不得
关闭证书校验。

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
systemd 进程启动、停止上限均为 10 秒；服务使用 `Type=simple`，重启脚本另外在
10 秒内等待 HTTP `/status` 真正可用。独立重启脚本只重启
`record-replay.service`，不会发送回放 `/start` 请求，且必须由现场人员交互确认安全；
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
直接回车继续，输入 `n` 取消；非交互环境直接拒绝。它会检查左右臂 CSV，
并在已有服务状态不是 `idle` 或状态不可查询时拒绝停止进程。停止阶段只发送 SIGTERM；
若服务不能干净退出则停止操作，不使用 SIGKILL。脚本只重启 HTTP API 服务，不发送
`POST /start`，因此不会主动开始一轮回放。

## 本机与 Orin 文件一致性

本机 `record_replay/` 是唯一源代码。禁止只修改 Orin 上的代码副本。每次部署后必须对
本机与 `/home/wuji-brain/workspace/record_replay/` 生成相对文件清单和 SHA-256，排除
`prior_data/`、`records/`、`__pycache__`、`.pyc`、日志和运行产物；只有纳入部署范围的文件
清单与每个哈希都一致时才能报告代码同步完成。
机械臂 xCoreSDK 连接由 RobotControl 独占管理。RecordReplay 通过
`http://127.0.0.1:6500/api/v1/ar5/{side}/{operation}` 调用显式 SDK 封装，不再直接连接控制器；
连续轨迹、CSV 顺序、offset 和错误恢复仍由 RecordReplay 编排。
