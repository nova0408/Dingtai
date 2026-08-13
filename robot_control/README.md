# RobotControl 统一机器人控制服务

`robot_control` 将项目现有的 qmlinker 设备客户端和 AR5 xCoreSDK 客户端统一到一个 HTTP 服务边界。
服务负责客户端生命周期、状态读取、单位收窄、串行访问和控制 API；底层实时控制仍由 qmlinker/xCoreSDK 完成。

完整接口说明见 [`API Reference.md`](API%20Reference.md)，机器可读契约见
[`openapi.yaml`](openapi.yaml)。任何接口、状态字段、配置或部署边界变更，必须同步更新这两份
文档以及 [`CHANGELOG.md`](CHANGELOG.md)。

## 当前边界

- qmlinker：head、lift、可选的腰部 Pitch 控制与状态、左夹爪、右手和 AGV；不再提供 qmlinker 左右臂部件。
- AR5：左右控制器状态、上下电、工作模式、急停恢复、伺服报警清除、NRT waypoint/碰撞进度、拖动、Jog、MoveAbsJ、MoveL、elbow 和 stop。
- HTTP：默认监听 `127.0.0.1:6500`，适合由 SSH 隧道或现场人工配置的内网访问。
- 硬件客户端：第一次 GET 或人工控制请求到达时才延迟创建。
- AR5 控制客户端默认使用 `initialize_toolset=False`，连接和控制请求都不会自动写入
  tool/wobj；需要固定坐标系的 RecordReplay 由自身流程显式配置。

## 启动

```powershell
python -m robot_control.service --host 127.0.0.1 --port 6500
```

服务同时写入 journald 控制台日志和独立的 `logs/robot_control.log`。文件日志每小时轮转、
ZIP 压缩并保留 7 天；可用 `--log-path` 覆盖路径，不与其它服务合并存储。

当前机型支持腰部时使用默认配置；不支持腰部的后续机型启动时使用：

```text
python -m robot_control.service --host 127.0.0.1 --port 6500 --no-qmlinker-waist
```

关闭后不会创建腰部客户端，状态订阅的 `devices` 数组中也不会出现 `qmlinker_waist`。

启动服务本身不会执行运动。RobotControl 也不会在运动请求前自动写入默认 tool/wobj；
服务部署前必须由现场人员确认控制器当前坐标系、网络、SDK 环境和设备安全边界。

## 统一客户端入口

正式客户端必须访问 Gateway 的 HTTPS `443` 端口，并使用
`/api/v1/robot-control/*` 前缀；Gateway 再转发到本服务的 `6500` 端口。
这只统一客户端地址和 URL 前缀，不会合并进程，也不会移除 RobotControl 的内部端口；
`6500` 仍是服务实际监听端口，仅用于人工测试、Orin 本地只读诊断和故障排查。完整映射见
[`api_gateway/README.md`](../api_gateway/README.md)。客户端首次使用前必须安装并信任
CasiaHand Root CA，安装指南见
[`api_gateway/certificates/README.md`](../api_gateway/certificates/README.md)；不得关闭证书校验。

```python
from robot_control.service.client import RobotControlClient

client = RobotControlClient(
    "https://<orin-host>",
    api_prefix="/api/v1/robot-control",
)
```

## 只读状态接口

```text
GET /api/v1/health
GET /api/v1/status
GET /api/v1/devices
GET /api/v1/agv/targets
GET /api/v1/agv/base-state
GET /api/v1/agv/base-mode
GET /api/v1/agv/base-operation-state
GET /api/v1/agv/base-task-process
GET /api/v1/agv/base-battery
GET /api/v1/ar5/{side}/soft-limits
GET /api/v1/ar5/{side}/motion-progress
GET /api/v1/status/stream?interval_s=0.2
```

`/status` 和 `/devices` 会读取现场设备状态；它们不是离线接口。只有用户明确授权现场只读检查时，才可调用这些 GET 接口；本回合未连接现场服务。

AGV 聚合状态使用 qmlinker `1.0.16` 的 `GetBaseState` 与 `GetBaseBattery`，在
`qmlinker_agv.data` 中返回 `robot_state`、三态 `initialized`、`power` 和 `charge_state`。
`initialized=null` 表示底层状态未知，`false` 表示未初始化，`true` 表示已完成初始化；
`charge_state` 的 `0`、`1`、`2`、`3` 分别表示未知、未充电、手动充电和自动充电。
连接成功不等于初始化完成，充电状态也不能由可能同时保持空闲的 `robot_state` 推断。

`qmlinker_right_hand` 的 `actuator_count` 来自运行时手部规格，`positions` 必须完整包含对应的
`right_hand_a0` 到 `right_hand_aN`。如果 qmlinker 返回的实际执行器集合不完整或包含未知轴，
该设备会以 `connected=false`、`error` 带有 expected/actual/missing/unexpected 详情、空 `data`
返回，避免客户端误用部分 M6 状态。

`/status/stream` 是只读 Server-Sent Events（SSE）订阅接口。连接建立后立即推送一条
完整状态快照，之后默认每 `0.2` 秒推送一条 `event: robot_status`；可通过
`interval_s` 指定 `0.05` 到 `5.0` 秒的间隔。每条 `data` 都包含当前启用的 qmlinker、AR5、
头部、升降、可选腰部 Pitch、夹爪、右手和 AGV 状态。AR5 状态按 `identity`、`joints`、
`tcp`、`elbow` 和 `status` 分组；客户端断开后服务会停止该订阅线程。

`/qmlinker/agv/targets` 通过 qmlinker 的只读地图服务读取当前 Woosh 地图和可用导航点，
不会发送导航或其他运动请求。返回的 `x_m`、`y_m` 单位为 m，`yaw_rad` 单位为 rad；
`resolution` 保留底盘地图接口的原始值。

AGV 的 `base-state`、`base-mode`、`base-operation-state`、`base-task-process` 和
`base-battery` 是按 Woosh SDK 状态对象拆分的只读查询，业务可按需调用；不会发送运动或控制请求。

`/api/v1/ar5/{side}/soft-limits` 读取指定 AR5 控制器的七个轴软限位，`side` 为 `left` 或
`right`。返回 `enabled`、`axis_count` 和 `limits_rad`；每个轴包含 `axis_index`、`lower_rad`
和 `upper_rad`，上下限单位为 rad。接口只调用 xCoreSDK `getSoftLimit`，不会改变电源、工作模式
或拖动状态；控制器返回的软限位数量不是七个时，服务会返回带原始错误信息的 `503`。

统一 Gateway 访问路径为：
`GET https://<orin-host>/api/v1/robot-control/status/stream?interval_s=0.2`。
Gateway 只转发事件流，不改变订阅语义。

Python 客户端示例：

```python
for snapshot in client.subscribe_status(interval_s=0.2):
    dashboard.update(snapshot)
```

## 控制接口

控制接口按设备直接位于 `/api/v1/...`，使用 POST。HTTP 契约不暴露内部 qmlinker 实现层；左右臂
控制接口；新增的夹爪、右手、
AGV、AR5 急停恢复、拖动和 Jog 路径及字段见 `openapi.yaml` 与 `API Reference.md`。

腰部控制接口为 `POST /api/v1/waist`；支持该能力时，`qmlinker_waist` 报告 `enabled` 和
`pitch_deg`。升降 `height_mm` 会按 qmlinker CLI 的规则四舍五入为整毫米目标。不支持时完全省略
`qmlinker_waist` 设备，腰部控制请求返回 `503`。AGV
`translate` 是持续实时平移请求，必须由现场人员显式调用
`/api/v1/agv/stop` 停止；该停止语义是软件停止，不等同于硬件急停。

RecordReplay 通过 `motion-progress` 将 `moveExecution` 的 `cmdID`、最后到达 waypoint 与
`30400.collision_fc` 锁存关联起来；碰撞恢复使用 `clear-servo-alarm` 一对一调用
xCoreSDK `clearServoAlarm()`。该 POST 可能改变真实控制器故障状态，只能由明确的现场回放流程调用。

RecordReplay 专用的批量 `move-append` 当前使用固定实验运动倍率。AR5 设备适配器在同一 SDK
锁内尽力执行 `getAcceleration` 前读、`adjustAcceleration(0.5, 0.5)` 和后读；第一个设置参数
是系统预设加/减速度倍率，第二个是系统预设加加速度（jerk）倍率。每一步失败都只记录告警，
原批次 `moveAppend` 始终继续，实验能力不能干扰既有回放。读取成功时日志记录设置前后值，
汇总日志同时记录请求值、前后实际值与设置是否成功。两个值均不来自 HTTP 请求，当前也没有
读写倍率的公开路由。普通批次、M6 前补位与碰撞恢复重发均会重新尝试应用固定值。

这些接口可能使机械臂、AGV、夹爪、头部或升降机构动作。禁止 Codex、CI、hook 或自动化脚本发送控制 POST；只能由现场人员手动发起和验证。

控制或设备读取失败时，服务不会只返回裸 `503`。JSON 响应包含 `error`、`error_type`、`message`、
`status`、`method`、`path` 和 `stage`；其中 `message` 保留底层 SDK 的原始错误文本，服务日志记录完整异常堆栈。

## 部署依赖

服务使用本包 `robot_control/devices/` 内的 qmlinker、AGV、右手和 AR5 适配器，以及远端 Linux
`sdk/xcoresdk`、qmlinker 环境；运行时不引用仓库 `src/`。推荐使用下面的脚本只部署本服务；
脚本会同步所需 Python 源码，但不会把本机 Windows xCoreSDK 二进制覆盖到 Orin：

```powershell
pwsh -NoProfile -File .\scripts\sync_and_restart_services.ps1 -RobotControlOnly
```

当前版本不创建新的 SSH 隧道，SSH 仅可作为 HTTP 服务的外层访问通道。

## 验证边界

允许：ruff、pyright、UTF-8 扫描、`py_compile`、协议对象离线构造和 GET 请求的人工只读验证。

禁止：发送任意控制 POST、调用 `RobotControlGateway` 控制方法做测试、运行运动冒烟、启动后自动探测并操作真实设备。
