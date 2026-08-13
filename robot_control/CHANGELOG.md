# RobotControl 版本日志

当前版本：`0.16.1`

## 0.16.1 - 2026-08-13

### 修复

- `getAcceleration` 前读、`adjustAcceleration(0.5, 0.5)` 和后读均改为尽力执行；任一步失败
  只记录告警，原批次 `moveAppend` 始终继续，避免实验参数干扰既有回放。
- 成功读取时记录设置前后的 acc/jerk 倍率，并统一记录请求值、实际前后值和设置是否成功。

## 0.16.0 - 2026-08-13

### 变更

- RecordReplay 使用的批量 AR5 `move-append` 在持有同一 SDK 锁期间，先固定调用
  `adjustAcceleration(0.5, 0.5)`，成功后再提交 `MoveAbsJCommand` 队列。
- 加/减速度与加加速度倍率暂不增加 HTTP 请求字段；普通回放、M6 前单点补位和碰撞恢复重发
  统一使用本次实验默认值，并在实际 `moveAppend` 日志中记录。

## 0.15.1 - 2026-08-13

### 修复

- `GET /api/v1/status`、`GET /api/v1/devices` 和 SSE 状态流的
  `qmlinker_agv.data` 聚合 `GetBaseBattery` 返回的 `power` 与 `charge_state`。
- 明确 Woosh `charge_state` 枚举：`0` 未知、`1` 未充电、`2` 手动充电、`3` 自动充电；
  充电状态不再依赖可能同时保持 `idle` 的 `robot_state` 推断。

## 0.15.0 - 2026-08-13

### 变更

- 本机开发环境和 AGV 适配基线升级到 qmlinker `1.0.16`，使用最新 `GetBaseState` 等只读 RPC。
- `qmlinker_agv.data` 增加 Woosh 原始 `robot_state` 和三态 `initialized`；`0` 映射为未知、`1` 映射为未初始化、其它当前 Woosh 状态映射为已初始化。
- 独立 `GET /api/v1/agv/base-state` 同步返回 `robot_state` 与 `initialized`，不再要求客户端用连接状态猜测初始化状态。

## 0.14.1 - 2026-08-13

### 优化

- 增加独立的 `logs/robot_control.log`，每小时轮转、ZIP 压缩并保留 7 天。
- 日志补充服务版本、启动配置与耗时、未处理异常堆栈、关闭耗时，以及 HTTP 请求关联标识、客户端、状态和耗时；设备控制语义不变。

## 0.14.0 - 2026-08-12

### 新增

- xCoreSDK `moveExecution` 回调按 `cmdID` 锁存最后一个明确到达的 waypoint，下标从 0 开始。
- `logReporter` 收到 `30400.collision_fc` 时为当前路径锁存碰撞状态。
- 新增只读 `GET /api/v1/ar5/{side}/motion-progress`，返回路径 ID、目标数、最后到达点和碰撞信息。
- 新增 `POST /api/v1/ar5/{side}/clear-servo-alarm`，一对一调用 SDK `clearServoAlarm()`；调用成功后清除碰撞锁存。
- `move-append` 响应改为返回当前路径进度对象，供 RecordReplay 精确关联本次批量轨迹。

## 0.13.0 - 2026-08-12

### 变更

- 每次读取 AR5 软限位时均从 xCoreSDK 获取最新配置并覆盖客户端缓存，供
  RecordReplay 在每次 `moveStart` 前使用。
- `moveAppend(MoveAbsJ)` 调用前记录实际交给 SDK 的每个目标关节值（rad/deg）、速度和 zone。
- 注册 xCoreSDK `logReporter` 与 `moveExecution` 事件回调，原样记录控制器
  `ecode/edetail` 及路径执行事件，避免异步规划错误只出现在控制器侧。

## 0.12.0 - 2026-08-12

### 新增

- 新增只读 `GET /api/v1/ar5/{side}/joint-position`，返回 xCoreSDK `jointPos` 的七轴实时
  关节位置，单位为 rad，供 RecordReplay 判断 MoveAbsJ 目标是否到位。

## 0.11.0 - 2026-08-11

### 变更

- HTTP 路径移除无业务意义的 `/qmlinker` 资源层，头部、升降、手部和 AGV 改为
  `/api/v1/{device}`，不保留旧路径兼容。
- 新增 `/api/v1/ar5/{side}` 下 xCoreSDK 原操作的显式封装，覆盖 RecordReplay 所需的
  robotInfo、状态读取、toolset、NRT/自动/电机设置、TCP、IK、MoveAbsJ 队列和停止清理。
- xCoreSDK 连接继续只由 RobotControl 延迟创建并持有，避免 RecordReplay 与状态流建立重复连接。

## 0.10.1 - 2026-08-07

### 修复

- 将 RobotControl 使用的 qmlinker、AGV、右手和 AR5 适配器收回 `robot_control/devices/`，服务部署不再依赖仓库 `src/`。

## 0.10.0 - 2026-08-07

### 新增

- 新增只读 `GET /api/v1/ar5/{side}/soft-limits`，读取左右 AR5 控制器七个轴的软限位上下限和
  软限位使能状态，单位为 rad。
- xCoreSDK 返回软限位数量异常时，服务返回包含具体轴数和读取阶段的结构化错误，不暴露不完整数据。

## 0.9.1 - 2026-08-06

### 修复

- M6 右手状态读取新增执行器集合完整性校验；实际位置数量与运行时规格不一致时，
  `qmlinker_right_hand` 返回设备读取错误，不再以 `connected=true` 暴露部分状态。
- 错误信息包含 expected、actual、missing 和 unexpected，便于定位 qmlinker 状态响应缺失。

## 0.9.0 - 2026-08-06

### 变更

- HTTP 错误响应新增 `error_type`、`message`、`status`、`method`、`path` 和 `stage` 字段。
- RobotControl 服务记录控制请求失败的路径、阶段和完整异常；AR5 SDK 原始错误信息不再只表现为裸 `503`。
- 客户端将结构化错误上下文合并到可复制的异常文本中。

## 0.8.0 - 2026-08-06

### 新增

- 新增按 Woosh SDK 状态对象拆分的 AGV 只读查询：底盘状态、底盘模式、底盘运行位、底盘任务进度和底盘电池。
- qmlinker 客户端、RobotControl HTTP GET、README、API Reference 和 OpenAPI 同步新增上述接口。

## 版本号规则

版本号使用 `a.b.c`：

- `a`：重大架构或部署边界变化。
- `b`：HTTP API、设备能力或控制语义变化。
- `c`：不改变控制边界的缺陷修复和文档改进。

## 0.7.0 - 2026-08-05

### 变更

- 移除 qmlinker 左右臂状态设备、控制路由、客户端和延迟客户端创建；当前可用机械臂状态统一由 `ar5_left`/`ar5_right` 提供。
- AR5 状态由扁平快照字段改为 `identity`、`joints`、`tcp`、`elbow` 和 `status` 分组，保留完整关节、TCP、臂角、控制器状态和身份数据。
- qmlinker 腰部改为真正的可选设备：未启用时从 `devices` 和 SSE 状态订阅中完全省略，不再返回 `available=false` 占位记录。
- 同步更新 RobotControl README、API Reference 和 OpenAPI 契约。

## 0.6.0 - 2026-08-03

### 新增

- 新增只读 `GET /api/v1/qmlinker/agv/targets`，返回 Woosh 当前地图元数据和可用导航点。
- 目标点返回名称、底盘点位 ID、米制坐标和弧度制航向；查询不发送任何运动控制请求。
- qmlinker AGV 客户端新增地图结构读取接口，并保留目标名称列表接口供现有 GUI 使用。

## 0.5.0 - 2026-08-03

### 变更

- `qmlinker_waist` 改为可选状态能力，支持通过 `--no-qmlinker-waist` 适配不含腰部的后续机型。
- 不支持腰部时保留稳定设备记录并返回 `data.available=false`，不会创建腰部客户端。
- 同步更新 RobotControl 的配置说明、API Reference 和 OpenAPI 契约。

## 0.4.0 - 2026-08-03

### 新增

- 新增 AR5 急停恢复、拖动开关和单轴 Jog 控制接口。
- 新增 qmlinker 夹爪使能/校准、右手使能、AGV 使能、实时平移和软件停止接口。
- 状态新增 `qmlinker_waist` 腰部 Pitch 只读设备，并为 `qmlinker_right_hand` 增加 `enabled` 字段。

### 约束

- 腰部不提供使能或角度控制接口；腰部自由度按设计取消，仅保留只读状态。
- AGV 实时平移必须由现场人员显式调用 AGV stop；软件停止不等同于硬件急停。

## 0.1.0 - 2026-07-31

### 新增

- 新增 qmlinker 与 AR5 统一控制服务目录。
- 新增健康检查、设备只读状态和设备状态别名接口。
- 新增显式 qmlinker/AR5 控制路由；控制接口只保留给现场人员手动发起。
- 新增延迟硬件客户端创建和 AR5 只读连接模式，避免 GET 状态读取隐式写入 tool/wobj。

## 0.2.0 - 2026-07-31

### 变更

- AR5 控制客户端默认不再写入默认 tool/wobj。
- RobotControl 的 AR5 运动请求不再自动配置 tool/wobj；RecordReplay 保持由自身回放流程显式配置。
- `sync_and_restart_services.ps1 -RobotControlOnly` 可独立同步依赖源码、安装并重启 RobotControl，启动检查只调用 `/api/v1/health`。

## 0.3.0 - 2026-07-31

### 新增

- 新增只读 SSE 状态订阅接口 `GET /api/v1/status/stream`，按固定间隔推送全部 qmlinker 与 AR5 设备快照。
- RobotControl 客户端新增 `subscribe_status()`，支持直连或统一 Gateway 前缀访问。
- 状态订阅只读取状态，不发送任何控制请求；控制 POST 仍只能由现场人员手动发起。

## 0.3.1 - 2026-08-03

### 文档

- 补齐 `API Reference.md`，覆盖只读接口、SSE 订阅、状态字段、单位和控制请求格式。
- 完善 `openapi.yaml` 的状态响应与控制请求 schema，并将 README、API Reference、OpenAPI、CHANGELOG
  设为每次公开契约变更必须同步更新的文档集合。
