# 双臂记录回放服务

当前业务语义版本：`1.11.0`，对应人工验证入口
`test/wuji/record_replay_cli.py` 中的 `RECORD_REPLAY_CLI_VERSION`。

面向 GUI 和其它项目的完整 HTTP 契约见 [API Reference](API%20Reference.md)，机器可读描述见
[OpenAPI](openapi.yaml)。

本服务从 `test/wuji/record_replay_cli.py` 拆分而来，位于仓库根目录，和
`camera_pipeline` 同级。业务代码不导入 `test`，在 Orin 上直接连接机械臂、
qmlinker、AGV 和 Orin 本机部署的 camera_pipeline，并由 HTTP API 触发一轮执行。
三球和 Board 检测全部通过 `CameraPipelineClient` 的业务接口完成；本服务不导入、配置或
描述 camera_pipeline 内部使用的传输实现，也不自行订阅相机帧或实现检测算法。

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

部署时必须把代码、先验 JSON、ChArUco offset 历史 CSV、两侧相机外参和两侧回放 CSV 作为同一个
`record_replay/` 目录同步到 Orin，并参与文件清单和 SHA-256 一致性校验。

服务提供两个 JSON 先验替换接口：`POST /prior/ball-pose` 和
`POST /prior/charuco`。替换前旧文件会备份到服务端
`record_replay/.archive/prior_data/<时间戳>/`，不会直接删除；上传内容先通过对应文件格式校验，
校验失败不会替换现有文件。

`test/wuji/record_replay_cli.py` 是本机直连硬件的人工验证入口，与 Orin HTTP 服务的数据
位置不同：它读取本机 `record_left/`、`record_right/`，先验读取 GUI 写入的本机
`record_replay/prior_data/`。该入口默认关闭 AGV、使用左臂单臂模式，只加载文件名首段
为纯数字的 CSV，并按该数字的数值顺序执行；文件名首段不是数字的 CSV 不参与回放。
人工 CLI 的 MoveAbsJ 末端线速度和 zone 都使用 CSV 数字前缀到数值的字典，并为左右臂
分别提供显式命名的配置。每套字典的键 `-1` 是该侧机械臂默认值，其余整数键覆盖该侧
对应 CSV；并行回放时左右臂各自读取自己的速度和 zone 配置。
三球 offset 触发 CSV 不再使用独立的临时 MoveAbsJ 速度，直接遵循该侧 CSV 速度字典。
零字节或只有表头的 CSV 可作为序号占位文件参与排序和双臂计划构建；人工 CLI 在执行到
该文件时记录警告并整文件跳过，不触发该 CSV 对应的 offset 或设备动作。
Orin HTTP 服务只读取已部署到 `/home/wuji-brain/workspace/record_replay/records/` 和
`prior_data/` 的远端副本。

人工 CLI 在启用头部 ChArUco 纠偏前读取
`record_replay/prior_data/charuco_offset_history.csv`。同侧机械臂至少需要 6 条
`accepted=true` 的有效历史；xyz/rpy 各分量必须位于历史均值 ±4σ 内，平移和旋转
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
```

任一阶段失败时，服务先完成运行资源清理，再回到 `idle`，并将错误文本写入
`ReplayContext.snapshot()`。下一轮必须重新发出触发指令。

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
- `error_text`：失败原因；
- `left_csv_files` / `right_csv_files`：左右臂部署目录中的 CSV 文件名与数据行数；
- `execution_tasks`：按实际执行顺序展开的任务清单；每个一级任务同时给出
  `left_csv`、`right_csv` 和 `synchronized`，单臂阶段另一侧为空；
- `current_task_sequence`：当前执行任务在 `execution_tasks` 中的序号，从 1 开始；
- `current_task_active`：当前任务序号对应的任务是否仍在执行；任务完成后到下一任务开始前
  为 `false`，避免界面把 AGV 返航等阶段误显示为仍在执行最后一个 CSV；
- `total_execution_count`：服务进程本次启动以来累计接受的执行请求次数；服务重启后从
  0 重新计数，不代表任务清单长度；
- `current_left_csv` / `current_right_csv`：左右臂当前正在处理的 CSV；
- `current_left_row` / `current_right_row`：左右臂当前处理的 CSV 源数据行；
- `current_left_total_rows` / `current_right_total_rows`：当前 CSV 总数据行数。

部署清单在服务启动和每轮开始前刷新。当前行用于界面定位服务正在调度或处理的源数据行；
连续机械臂轨迹会批量提交给控制器，因此该值不表示控制器已经物理到达对应轨迹点。

## 测试安全红线

`record_replay` 会直接控制机械臂、AGV、夹爪、M11 和升降机构。禁止 Codex、CI、
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

所有会影响执行时序、速度、重试、容差、三球采样和 AGV 轮询的默认策略定义在
`settings.py`；允许现场修改的运行参数由 `service/config_store.py` 持久化：

- `ReplayArmSettings`：NRT、tool/wobj、MoveAbsJ、reset 与机械臂型号；
- `ReplayHandSettings`：夹爪/M11/升降动作与容差；
- `ReplayOffsetSettings`：offset 触发、采样、ChArUco 安全门和三球鲁棒聚合；
- `OffsetConfig`：相机名、先验捕获与手眼结果路径；
- `ReplayServiceSettings`：AGV、触发文件及非运动调用重试。

业务模块通过 `ReplayContext.config.settings` 或 `ReplayRuntime.settings` 读取这些数据，
禁止重新声明模块级调试常量。若需现场调参，请在本机入口构造
`ReplayServiceSettings` 的定制实例，再传入 `ReplayCycleConfig`。
MoveAbsJ 末端线速度和中间点 zone 按左右臂及 CSV 数字序号配置；`-1` 是本侧默认级别，
其余序号覆盖对应 CSV。offset 触发 CSV 不再使用独立临时速度。Orin 服务通过 `/config`
读取和修改这四组映射，只有 `idle` 状态允许修改。

## Orin 服务 API

服务入口为 `python -m record_replay.service`，默认监听 `http://0.0.0.0:6300`。
进程启动只建立 API 监听，不会自动执行机械臂动作，并处于 `idle` 状态。调用 `start` 被接受后
立即进入 `busy`，直到 AGV、设备准备、CSV 回放和资源清理全部结束，再恢复 `idle`。业务线程执行
一轮既有的 AGV、CSV、双臂和 offset 流程；重复 `start` 会返回 `accepted=false`，
不会并发控制同一组设备。`status` 可查询当前阶段、部署的左右臂 CSV、左右臂对齐的
实际执行任务、当前任务序号、服务累计执行次数、各臂当前 CSV、源数据行进度、计划下标
和错误文本。
GUI 可按 1 秒周期轮询该只读接口；不需要维持 SSE/WebSocket 长连接，短暂断网后下一次
轮询会自然恢复。

HTTP API：`GET /status`、`GET /config`、`GET /device-status`、`POST /config`、
`POST /prior/ball-pose`、`POST /prior/charuco`、`POST /start`。配置更新 body
是字段到数值或 CSV 序号映射的 JSON object；服务 `busy` 期间拒绝修改配置。两个 prior 接口接收完整 JSON，
校验通过后原子替换并将旧文件备份到服务端 `.archive/prior_data/<时间戳>/`。`POST /start` 必须提供
`{"enable_agv_navigation": true|false}`；为 `false` 时不执行回放前导航，双臂 CSV
回放仍会执行；为 `true` 时只在回放前导航到站点 `1`，回放完成后不自动返航，与已验证
人工 CLI 一致。调用 `/start` 前会全量检查所有运行先验，缺失或无效文件会逐项写入
`error_text`；服务启动本身不会因先验缺失失败。

正式客户端必须通过统一 Gateway 访问本服务：使用 `https://<orin-host>`，并配置
`/api/v1/record-replay` 前缀。Gateway 只统一客户端地址和 URL 前缀，不会合并进程，
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

本机 `record_replay/` 是唯一源代码。禁止只修改 Orin 上的副本。每次部署后必须对
本机与 `/home/wuji-brain/workspace/record_replay/` 生成相对文件清单和 SHA-256，排除
`__pycache__`、`.pyc`、日志和运行产物；只有文件清单与每个哈希都一致时才能报告同步完成。
