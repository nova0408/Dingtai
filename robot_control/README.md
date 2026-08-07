# RobotControl 统一机器人控制服务

`robot_control` 将项目现有的 qmlinker 设备客户端和 AR5 xCoreSDK 客户端统一到一个 HTTP 服务边界。
服务负责客户端生命周期、状态读取、单位收窄、串行访问和控制 API；底层实时控制仍由 qmlinker/xCoreSDK 完成。

完整接口说明见 [`API Reference.md`](API%20Reference.md)，机器可读契约见
[`openapi.yaml`](openapi.yaml)。任何接口、状态字段、配置或部署边界变更，必须同步更新这两份
文档以及 [`CHANGELOG.md`](CHANGELOG.md)。

## 当前边界

- qmlinker：head、lift、可选的腰部 Pitch 只读状态、左夹爪、右手和 AGV；不再提供 qmlinker 左右臂部件。
- AR5：左右控制器状态、上下电、工作模式、急停恢复、拖动、Jog、MoveAbsJ、MoveL、elbow 和 stop。
- HTTP：默认监听 `127.0.0.1:6500`，适合由 SSH 隧道或现场人工配置的内网访问。
- 硬件客户端：第一次 GET 或人工控制请求到达时才延迟创建。
- AR5 控制客户端默认使用 `initialize_toolset=False`，连接和控制请求都不会自动写入
  tool/wobj；需要固定坐标系的 RecordReplay 由自身流程显式配置。

## 启动

```powershell
python -m robot_control.service --host 127.0.0.1 --port 6500
```

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
GET /api/v1/qmlinker/agv/targets
GET /api/v1/qmlinker/agv/base-state
GET /api/v1/qmlinker/agv/base-mode
GET /api/v1/qmlinker/agv/base-operation-state
GET /api/v1/qmlinker/agv/base-task-process
GET /api/v1/qmlinker/agv/base-battery
GET /api/v1/ar5/{side}/soft-limits
GET /api/v1/status/stream?interval_s=0.2
```

`/status` 和 `/devices` 会读取现场设备状态；它们不是离线接口。只有用户明确授权现场只读检查时，才可调用这些 GET 接口；本回合未连接现场服务。

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

控制接口统一位于 `/api/v1/qmlinker/...` 和 `/api/v1/ar5/...`，使用 POST。qmlinker 不再提供左右臂
控制接口；新增的夹爪、右手、
AGV、AR5 急停恢复、拖动和 Jog 路径及字段见 `openapi.yaml` 与 `API Reference.md`。

腰部不提供任何控制接口；支持该能力时，`qmlinker_waist` 报告 `enabled` 和 `pitch_deg`。
不支持时完全省略 `qmlinker_waist` 设备，因为设计上将取消腰部这个自由度。AGV
`translate` 是持续实时平移请求，必须由现场人员显式调用
`/api/v1/qmlinker/agv/stop` 停止；该停止语义是 qmlinker 的软件停止，不等同于硬件急停。

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
