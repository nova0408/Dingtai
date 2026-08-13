# RobotControl API Reference

当前契约版本：`0.16.1`
HTTP API 主版本：`1`

本文档描述 `robot_control` 对外 HTTP API 的实际调用方式、状态字段、单位和安全边界。
机器可读契约见 [`openapi.yaml`](openapi.yaml)；服务概览、部署和验证边界见
[`README.md`](README.md)。三份文档必须随接口、状态、配置、部署或安全边界变更同步更新。

## 1. 访问地址

RobotControl 独立端口仅用于人工测试、Orin 本地只读诊断和故障排查：

```text
http://127.0.0.1:6500
```

正式客户端必须通过统一 Gateway 访问：

```text
https://<orin-host>/api/v1/robot-control
```

客户端首次使用前必须安装并信任 CasiaHand Root CA，安装指南见
[`api_gateway/certificates/README.md`](../api_gateway/certificates/README.md)。不得关闭证书校验。

Gateway 只移除 `/api/v1/robot-control` 前缀后转发到 RobotControl 的
`/api/v1`，不会合并进程，也不会移除内部 `6500` 端口。GUI 和其它正式客户端不得把
`6500` 作为默认访问入口。

## 2. 通用响应和错误

所有正常 JSON 响应使用 UTF-8，状态接口不额外包裹 `ok` 字段。

### 2.1 健康响应

```json
{
  "service_version": "0.16.1",
  "api_version": "1",
  "hardware_access": "lazy"
}
```

`GET /api/v1/health` 不创建 qmlinker 或 AR5 硬件客户端，不访问设备。

### 2.2 状态响应

```json
{
  "service_version": "0.16.1",
  "api_version": "1",
  "devices": [
    {
      "name": "ar5_left",
      "backend": "xcoresdk",
      "connected": true,
      "error": null,
      "data": {
        "identity": {"robot_type": "AR5", "robot_uid": "left"},
        "joints": {"count": 7, "angle_deg": [0, 0, 0, 0, 0, 0, 0]},
        "tcp": {"pose_matrix_m": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], "xyz_mm": [0, 0, 0], "rpy_deg": [0, 0, 0]},
        "elbow": {"angle_deg": 0, "available": true},
        "status": {"operation_state": "idle", "operate_mode": "manual", "power_state": "off"}
      }
    }
  ]
}
```

每个设备独立携带 `connected` 和 `error`。单个设备读取失败不会遮蔽其它设备；失败时
`data` 为空对象。状态读取会创建并访问相应硬件客户端，因此不是离线健康检查。

异常响应统一为：

```json
{
  "ok": false,
  "error": "RuntimeError: setPowerState(False) 失败: ec=-2, message=General failure",
  "error_type": "RuntimeError",
  "message": "setPowerState(False) 失败: ec=-2, message=General failure",
  "status": 503,
  "method": "POST",
  "path": "/api/v1/ar5/right/drag",
  "stage": "hardware_control"
}
```

`error` 保留完整错误摘要；`message` 是原始异常文本，`stage` 表示失败阶段，`path` 是实际服务路径。
控制失败时服务日志还会记录完整异常堆栈。客户端应同时保留 HTTP 状态码和响应体，不能只显示 `503`。

常见 HTTP 状态：`200` 成功，`202` 控制请求已接受，`400` 请求字段错误，`404` 路径不支持，
`503` 服务或设备读取/控制失败。

## 3. 只读接口

### 3.1 `GET /api/v1/health`

读取服务健康信息，不访问硬件。

### 3.2 `GET /api/v1/status`

读取 qmlinker 与 AR5 的一次完整状态快照。

### 3.3 `GET /api/v1/devices`

`/api/v1/status` 的只读别名，响应结构相同。

### 3.4 `GET /api/v1/agv/targets`

读取当前 Woosh 地图和可用于 `POST /api/v1/agv/navigate` 的目标点列表。
此接口只读取地图缓存，不发送导航、停止、使能或速度控制请求。

响应示例：

```json
{
  "map": {
    "name": "DingTaiNJ",
    "id": 0,
    "resolution": 0.03
  },
  "targets": [
    {
      "name": "3",
      "id": 2905130067,
      "x_m": 0.79,
      "y_m": 0.36,
      "yaw_rad": -3.0
    }
  ]
}
```

`targets[].name` 是导航请求使用的目标字符串；`id` 是底盘点位 ID；`x_m`、`y_m` 单位为
m；`yaw_rad` 单位为 rad；`resolution` 保留 Woosh 地图接口的原始值。若远端地图服务
不可用或尚未获得地图数据，HTTP 返回 `503`。

### 3.5 `GET /api/v1/agv/base-state`

读取 Woosh SDK 的底盘状态枚举，返回 `robot_state` 和三态 `initialized`。`robot_state=0`
表示状态未知，此时 `initialized=null`；`robot_state=1` 表示未初始化，此时为 `false`；
当前其它 Woosh 状态 `2`–`9` 均表示已经完成初始化，此时为 `true`。查询失败返回 `503`。

### 3.6 `GET /api/v1/agv/base-mode`

读取 Woosh SDK 的底盘控制模式和工作模式，返回 `robot_mode`、`work_mode`。

### 3.7 `GET /api/v1/agv/base-operation-state`

读取底盘原始运行位，返回 `nav_bits`、`robot_bits`；业务可按 Woosh SDK 位定义自行解析。

### 3.8 `GET /api/v1/agv/base-task-process`

读取当前任务和动作进度，返回 `task_id`、`task_type`、`task_state`、`action_type`、`action_state`、
`wait_id`、`dest`、`msg`、`time`。

### 3.9 `GET /api/v1/agv/base-battery`

读取底盘电量和充电状态，返回 `power`、`charge_state`。Woosh `charge_state=0` 表示未知、
`1` 表示未充电、`2` 表示手动充电、`3` 表示自动充电。

以上接口均为独立只读 GET，按需调用，不发送导航、停止、使能或速度控制请求。

### 3.10 `GET /api/v1/status/stream`

以 Server-Sent Events（SSE）持续推送完整状态快照。连接建立后立即推送第一条消息，
之后按间隔推送；这是只读 GET，不发送控制请求。

查询参数：

| 参数 | 类型 | 默认值 | 范围 | 说明 |
| --- | --- | ---: | ---: | --- |
| `interval_s` | number | `0.2` | `0.05`–`5.0` | 两次状态读取之间的间隔，单位秒 |

请求示例：

```text
GET /api/v1/status/stream?interval_s=0.2
Accept: text/event-stream
```

事件格式：

```text
event: robot_status
id: 0
data: {"service_version":"0.16.1","api_version":"1","devices":[]}

```

`data` 是与 `GET /api/v1/status` 相同的完整 JSON 状态对象。客户端断开后，服务停止该
连接对应的订阅线程。一个 GUI 仪表盘通常只应建立一个状态订阅连接。

Python 客户端：

```python
from robot_control.service.client import RobotControlClient

client = RobotControlClient("https://<orin-host>", api_prefix="/api/v1/robot-control")
for snapshot in client.subscribe_status(interval_s=0.2):
    update_dashboard(snapshot)
```

### 3.11 `GET /api/v1/ar5/{side}/soft-limits`

读取指定 AR5 控制器的七个轴软限位。`side` 必须为 `left` 或 `right`；上下限沿用 xCoreSDK
单位，为弧度 rad。每次请求均从 xCoreSDK 读取最新配置并覆盖当前客户端缓存；该接口只读，
不切换电源、工作模式或拖动状态。

请求示例：

```text
GET /api/v1/ar5/right/soft-limits
Accept: application/json
```

响应示例：

```json
{
  "side": "right",
  "enabled": true,
  "axis_count": 7,
  "limits_rad": [
    {"axis_index": 0, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 1, "lower_rad": -2.0, "upper_rad": 2.0},
    {"axis_index": 2, "lower_rad": -2.0, "upper_rad": 2.0},
    {"axis_index": 3, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 4, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 5, "lower_rad": -3.14, "upper_rad": 3.14},
    {"axis_index": 6, "lower_rad": -3.14, "upper_rad": 3.14}
  ]
}
```

`limits_rad` 固定包含七项，`axis_index` 从 0 到 6。若 SDK 读取失败或控制器返回的轴数不为
七个，服务返回 `503`，响应中的 `error`、`message` 和 `stage=hardware_read` 保留具体错误。

Python 客户端：

```python
limits = client.get_ar5_soft_limits("right")
```

## 4. 设备状态字段

`devices` 中的 `name` 是稳定设备标识，当前包括：

| name | backend | `data` 内容 |
| --- | --- | --- |
| `qmlinker_head` | `qmlinker` | `enabled`、`yaw_deg`、`pitch_deg` |
| `qmlinker_lift` | `qmlinker` | `enabled`、`height_mm` |
| `qmlinker_waist` | `qmlinker` | `enabled`、`pitch_deg`（仅启用腰部能力时出现） |
| `qmlinker_gripper` | `qmlinker` | `online`、`calibrated`、`enabled`、`position`、`state` |
| `qmlinker_right_hand` | `qmlinker` | `actuator_count`、`enabled`、`positions` |
| `qmlinker_agv` | `qmlinker` | `enabled`、`robot_state`、`initialized`、`power`、`charge_state`、`runtime` |
| `ar5_left` | `xcoresdk` | `identity`、`joints`、`tcp`、`elbow`、`status` |
| `ar5_right` | `xcoresdk` | `identity`、`joints`、`tcp`、`elbow`、`status` |

`qmlinker_right_hand.data.actuator_count` 是运行时手部规格中的期望数量，`positions` 必须完整包含
对应的 `right_hand_a0` 到 `right_hand_aN`。如果 qmlinker 返回的执行器集合不完整或包含未知轴，
该设备会返回 `connected=false`、带有 `expected`、`actual`、`missing`、`unexpected` 详情的
`error`，并将 `data` 置为空对象；客户端不得继续使用部分位置数据。

`qmlinker_agv.data.robot_state` 直接来自 qmlinker `1.0.16` 的 `GetBaseState`：`0` 为
undefined、`1` 为 uninitialized、`2` 为 idle、`3` 为 parking、`4` 为 task、`5` 为 warning、
`6` 为 fault、`7` 为 following、`8` 为 charging、`9` 为 mapping。客户端必须使用同级
`initialized` 展示初始化三态，不得把设备 `connected` 当作已初始化。

`qmlinker_agv.data.power` 与 `charge_state` 直接来自 `GetBaseBattery`。充电状态必须使用
`charge_state`：`0` 为未知、`1` 为未充电、`2` 为手动充电、`3` 为自动充电；不能根据
`robot_state` 推断，因为底盘充电时机器人状态可能仍为 `idle`。

### 4.1 AR5

AR5 `data` 使用结构化分组，完整示例：

```json
{
  "identity": {"robot_type": "AR5", "robot_uid": "left"},
  "joints": {"count": 7, "angle_deg": [0, 0, 0, 0, 0, 0, 0]},
  "tcp": {
    "pose_matrix_m": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "xyz_mm": [0, 0, 0],
    "rpy_deg": [0, 0, 0]
  },
  "elbow": {"angle_deg": 0, "available": true},
  "status": {"operation_state": "idle", "operate_mode": "manual", "power_state": "off"}
}
```

- `identity`：控制器型号和唯一标识；
- `joints.angle_deg`：七个关节角，单位 deg；
- `tcp.pose_matrix_m`：4×4 齐次矩阵，平移单位 m；`xyz_mm` 和 `rpy_deg` 分别为 mm、deg，姿态使用 SciPy 小写外禀 xyz；
- `elbow.angle_deg`：臂角，单位 deg；`available` 表示当前点位是否携带臂角约束；
- `status`：操作状态、工作模式和电机状态。

RecordReplay 使用的 AR5 只读细粒度接口还包括：

```text
GET /api/v1/ar5/{side}/joint-position
```

响应为 `{"joints_rad": [j1, j2, j3, j4, j5, j6, j7]}`，读取 xCoreSDK `jointPos` 的实时
七轴关节位置，单位为 rad。

### 4.2 qmlinker 腰部可选能力

RobotControl 默认按当前机型声明腰部能力可用。后续不支持腰部的机型应使用：

```text
python -m robot_control.service --no-qmlinker-waist
```

不支持时 `devices` 数组中不包含 `qmlinker_waist`。支持时才出现该设备，`data` 包含
`enabled` 和 `pitch_deg`。腰部始终没有使能或角度控制接口。

## 5. qmlinker 控制接口

以下接口均为 `POST`，可能操作真实设备，只能由现场人员明确手动发起。RobotControl、
Codex、CI、hook 和自动化测试不得调用这些接口。

### 5.1 头部

```text
POST /api/v1/head
```

至少提供一个字段：

```json
{"enable": true}
```

可用字段为 `enable`、`yaw_deg`、`pitch_deg`。

### 5.2 升降

```text
POST /api/v1/lift
```

至少提供一个字段：

```json
{"height_mm": 100.0}
```

可用字段为 `enable`、`height_mm`；高度单位 mm。

### 5.3 左夹爪

```text
POST /api/v1/gripper
```

```json
{"position": 0}
```

#### 夹爪使能

```text
POST /api/v1/gripper/enable
```

```json
{"enabled": true}
```

#### 夹爪校准

```text
POST /api/v1/gripper/calibrate
```

请求体可为空对象 `{}`。校准是否完成必须通过状态读取确认。

### 5.4 右手

```text
POST /api/v1/right-hand
```

```json
{"positions": [0.0, 0.0, 0.0]}
```

`positions` 为归一化执行器位置，范围由现场 qmlinker 设备契约约束。

#### 右手使能

```text
POST /api/v1/right-hand/enable
```

```json
{"enabled": true}
```

### 5.5 AGV 导航

```text
POST /api/v1/agv/navigate
```

```json
{"target": "1"}
```

#### AGV 使能

```text
POST /api/v1/agv/enable
```

```json
{"enabled": true}
```

#### AGV 实时平移

```text
POST /api/v1/agv/translate
```

```json
{"speed_mps": 0.3, "direction_deg": 0.0}
```

`speed_mps` 必须大于 0，单位为 m/s。方向约定为 `0` 前进、`90` 左移、`180` 后退、
`270` 右移，角度为 deg。该请求启动持续实时平移，`202` 只表示 qmlinker 已接受请求，
不表示移动完成，也不会自动超时停止。

#### AGV 实时停止

```text
POST /api/v1/agv/stop
```

请求 qmlinker 停止当前导航或实时平移。它是软件停止语义，不等同于硬件急停；现场仍须
按照设备安全规程处理急停。

腰部只提供前述 `qmlinker_waist` 只读状态，不提供腰部使能或角度控制接口。

## 6. AR5 控制接口

以下接口均为现场人工控制接口。AR5 关节角使用 deg，服务内部转换为 SDK 所需的 rad；
笛卡尔平移使用 mm，姿态使用 deg 的小写外禀 xyz；速度使用 mm/s，zone 使用 mm。

### 6.1 上下电

```text
POST /api/v1/ar5/{side}/power
```

`side`：`left` 或 `right`。

```json
{"enabled": true}
```

### 6.2 工作模式

```text
POST /api/v1/ar5/{side}/mode
```

```json
{"automatic": true}
```

### 6.3 急停恢复

```text
POST /api/v1/ar5/{side}/recover-estop
```

请求恢复 AR5 控制器的急停状态；不会自动上电，恢复后的设备状态必须通过 GET 或 SSE 确认。

### 6.4 拖动开关

```text
POST /api/v1/ar5/{side}/drag
```

```json
{"enabled": true}
```

该调用遵循 xCoreSDK 的拖动模式语义，可能改变控制器电源/工作模式状态。

### 6.5 Jog

```text
POST /api/v1/ar5/{side}/jog
```

```json
{
  "space": "joint",
  "axis_index": 0,
  "direction_positive": true,
  "rate": 0.2,
  "step": 1.0
}
```

`space` 为 `cartesian` 或 `joint`；笛卡尔轴索引为 `0`–`5`，关节轴索引为 `0`–`6`。
`rate` 范围为 `0.01`–`1.0`，`step` 在笛卡尔平移时为 mm，在笛卡尔旋转和关节空间时为 deg。

### 6.6 软件停止

```text
POST /api/v1/ar5/{side}/stop
```

该接口表示软件停止请求，不等同于硬件急停。

### 6.7 关节运动

```text
POST /api/v1/ar5/{side}/move-joints
```

```json
{
  "joint_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "speed_mm_s": 1000.0,
  "zone_mm": 10.0
}
```

### 6.8 笛卡尔运动

```text
POST /api/v1/ar5/{side}/move-cartesian
```

```json
{
  "xyz_mm": [0.0, 0.0, 0.0],
  "rpy_deg": [0.0, 0.0, 0.0],
  "elbow_deg": 0.0,
  "speed_mm_s": 1000.0,
  "zone_mm": 10.0
}
```

### 6.9 臂角运动

```text
POST /api/v1/ar5/{side}/move-elbow
```

```json
{
  "elbow_deg": 0.0,
  "speed_mm_s": 1000.0,
  "zone_mm": 10.0
}
```

## 7. 控制响应

### 7.1 RecordReplay 使用的 AR5 SDK 显式接口

RobotControl 是唯一 xCoreSDK 对象所有者。RecordReplay 只访问以下本机接口：

- GET：`robot-info`、`operation-state`、`operate-mode`、`power-state`、`cart-posture`、
  `motion-progress`；
- POST：`stop`、`disable-drag`、`set-motion-control-mode`、`set-operate-mode`、
  `set-power-state`、`set-default-conf-opt`、`set-default-speed`、`set-default-zone`、
  `set-toolset`、`calc-ik`、`move-reset`、`move-append`、`move-start`、
  `clear-servo-alarm`。

路径统一为 `/api/v1/ar5/{side}/{operation}`。`cart-posture` 使用 `trans_m`、`rpy_rad`、
`elbow_rad` 和 `conf_data`；`calc-ik` 使用相同字段并返回 `joints_rad`；`move-append`
接收 `targets[]`，每项包含七维 `joints_rad`、`speed_mm_s` 和 `zone_mm`。这些接口是
底层 SDK 操作的一对一显式封装，不包含 RecordReplay 动作顺序或 CSV 业务逻辑。

`move-append` 当前在同一 AR5 SDK 锁内尽力执行 `getAcceleration` 前读、
`adjustAcceleration(0.5, 0.5)` 和后读，然后无条件提交 `MoveAbsJCommand` 队列。读取或设置
失败只记录告警，不能让实验能力阻止既有轨迹；读取成功时记录设置前后值。`0.5` 分别表示系统
预设加/减速度与加加速度的倍率，不是绝对物理单位；请求体不接受倍率字段，客户端也不能通过
本接口覆盖实验默认值。SDK 声明的 acc 范围是 `[0.2, 1.5]`、jerk 范围是 `[0.1, 2.0]`，
运动中修改只对后续指令生效。

`move-append` 成功响应的 `data` 与 `motion-progress` 结构一致：

```json
{
  "command_id": "absj#270",
  "target_count": 7,
  "last_reached_waypoint_index": 1,
  "collision_detected": true,
  "collision_code": 30400,
  "collision_detail": "30400.collision_fc"
}
```

`last_reached_waypoint_index` 只接受当前 `command_id` 的 `moveExecution` 且
`reachTarget=true` 事件；尚未确认任何点时为 `null`。`clear-servo-alarm` 调用 SDK
`clearServoAlarm()`，成功后清除当前碰撞锁存，不自动执行 `moveStart`、上电或恢复轨迹。


控制请求成功只表示服务已接受并完成对应调用，不表示动作已经物理完成：

```json
{
  "service_version": "0.16.1",
  "api_version": "1",
  "accepted": true,
  "data": {
    "action": "ar5_move_joints",
    "device": "ar5_left"
  }
}
```

动作完成、设备状态和故障必须通过 `GET /api/v1/status` 或 SSE 状态流读取。

## 8. 安全与测试边界

- 服务默认只绑定 `127.0.0.1`；外部访问应使用用户明确配置的网络边界或 SSH 隧道。
- GET 状态读取会访问真实设备，只有用户明确授权现场只读检查时才可执行。
- 任何控制 POST、运动、上电、使能、导航、夹爪、升降和拖动测试只能由现场人员手动发起。
- RobotControl 默认不写入 tool/wobj；RecordReplay 需要固定坐标系时由自身流程显式配置。
- 本文档中的控制示例仅描述请求格式，不构成自动化测试授权。
