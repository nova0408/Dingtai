# RecordReplay 动作抽象、单次托盘交换与 Rapid Stop 升级 TODO

> 状态：主流程代码已完成静态检查；`calibration` 三球算法已绑定，`calibration_new_tray` 按要求留空，部署已完成，现场验收仍未完成  
> 范围：`record_replay/` 服务的 CSV 解析、动作执行、双臂同步、单次执行、运行状态及 AR5/AGV 停止
> 不在范围：修改先验录制流程、修改 CSV 内容格式、为 AR5/AGV 之外的设备设计急停

## 1. 已确认的设计约束

本次升级采用最小、显式、可追踪的设计，不建设通用工作流引擎。

- [x] 将底层运动抽象为三类：拍摄动作、快速动作、精确动作；
- [x] 再按现有 `record_left` 业务名称做第二层显式封装；
- [x] 当前业务顺序由服务内版本化 JSON 顺序列表表达，允许通过修改列表调整顺序、动作、速度和 zone；
- [x] JSON 显式保存 `function_name`，但执行器禁止使用 `getattr`、反射、插件注册表或任意字符串动态调用；
- [x] CSV 不再使用数字顺序前缀；
- [x] CSV 保留动作名、机械臂侧别和录制时间戳；
- [x] 仅多目标动作增加位置 index，例如 `get_tray_1`；
- [x] 双臂只在 `open_door` 和 `close_door` 动作的起点同步；
- [x] 双臂不设置动作终点屏障，一侧完成后不等待另一侧即可继续本侧后续动作；
- [x] Rapid Stop 只主动停止 AGV 和左右 AR5；
- [x] 收到停止后仍必须禁止所有后续动作下发，包括夹爪、灵巧手、升降和头部的后续指令；
- [x] 先验数据、先验录制方法、CSV 列、CSV 行数据和拖动示教采样逻辑保持不变。

建议将本次实现作为新的主版本发布。旧的序号编排、`Sxx` 同步和新的动作名编排不应在运行时长期并存。

## 2. 当前逻辑与本次替换边界

### 2.1 当前实现

当前服务：

- 只发现带数字前缀的 CSV；
- 根据数字前缀决定左右臂整体执行顺序；
- 根据左臂文件名中的 `Sxx` 查找右臂同步序号；
- 根据 CSV 数字序号选择速度、zone、偏移应用和算法触发点；
- 双臂同步执行采用线程配对，并等待两侧都执行完成后再继续计划；
- 服务状态只有 `idle` 和 `busy`，没有操作员可调用的 stop/reset。

本次需要替换：

```text
数字序号 -> JSON 中明确排列的命名动作顺序
Sxx      -> open_door / close_door 起点同步
按序号选择速度和 zone -> JSON 每个动作项明确填写 type、speed 和 zone
按序号触发拍摄/算法  -> 拍摄动作完成到位后显式触发
```

### 2.2 当前录制文件的命名基线

当前服务迁移基线中的左臂文件顺序为：

```text
01_go_out_left_...
02_S02_open_door_left_...
03_calibration_left_...
04_get_tray_left_...
05_get_new_tray_left_...
06_put_new_tray_left_...
07_calibration_new_tray_left_...
08_after_put_new_tray_left_...
09_S03_close_door_left_...
10_S04_go_home_left_...
```

当前服务迁移基线中的右臂文件顺序为：

```text
01_go_out_right_...
02_open_door_right_...
03_close_door_right_...
04_go_home_right_...
```

这些业务名称继续保留。需要删除的只有数字前缀和 `Sxx` 同步标记。

仓库中的 `record_left/` 是独立的录制源目录，当前还包含
`before_calibration`、`after_get_tray`、`put_tray`、`before_get_new_tray`、
`before_put_new_tray` 和 `after_put_new_tray` 等更细的动作文件；本次不修改该目录及其
CSV 内容。服务部署目录 `record_replay/records/` 与录制源目录不是自动镜像关系，只有实际
部署到服务目录的 CSV 才能被 `action_sequence.json` 绑定。

## 3. 新 CSV 命名规则

### 3.1 文件名格式

普通动作：

```text
<action_name>_<arm_side>_<timestamp>.csv
```

多目标动作：

```text
<action_name>_<index>_<arm_side>_<timestamp>.csv
```

约束：

- `arm_side` 只允许 `left` 或 `right`；
- `index` 是从 1 开始的正整数；
- 只有 `get_tray`、`put_tray`、`get_new_tray`、`put_new_tray` 允许 index；
- 无 index 的多目标动作应在启动前报错；
- 普通动作携带 index 应在启动前报错；
- 同一个动作、index 和 arm 在部署目录中必须唯一；
- 不根据时间戳静默选择“最新文件”，重复文件应作为部署错误报告；
- 不再解析文件名前缀数字，也不再解析 `Sxx`。

### 3.2 迁移示例

| 现有文件名 | 新文件名示例 | 说明 |
| --- | --- | --- |
| `01_go_out_left_20260626_104916.csv` | `go_out_left_20260626_104916.csv` | 删除数字前缀 |
| `02_S02_open_door_left_20260629_142056.csv` | `open_door_left_20260629_142056.csv` | 删除数字和 Sxx |
| `04_get_tray_left_20260630_154830.csv` | `get_tray_1_left_20260630_154830.csv` | 增加抓取位置 index |
| `05_get_new_tray_left_20260630_155520.csv` | `get_new_tray_1_left_20260630_155520.csv` | 增加新托盘位置 index |
| `06_put_new_tray_left_20260630_155941.csv` | `put_new_tray_1_left_20260630_155941.csv` | 增加放置位置 index |
| `09_S03_close_door_left_20260629_143547.csv` | `close_door_left_20260629_143547.csv` | 删除数字和 Sxx |
| `02_open_door_right_20260703_172135.csv` | `open_door_right_20260703_172135.csv` | 删除数字前缀 |

迁移只重命名文件，不改写 CSV 内容。迁移前必须保存 `.archive/record_replay/records/` 快照，并在迁移后比较每个 CSV 的 SHA-256 内容哈希。

服务动作绑定仍采用新命名；为方便继续录制，允许 CSV 文件名最前面保留纯数字前缀，
仅在动作名匹配时忽略该前缀，实际执行仍使用原始文件名。

## 4. 三类底层动作

只定义三种明确动作类型及三个明确执行入口。

### 4.1 拍摄动作 CaptureAction

用途：移动到拍摄位置，稳定到位后调用相机和算法。

执行语义：

1. 按拍摄动作的普通速度和 zone 执行前面的 AR5 点；
2. CSV 中最后一个 AR5 点使用独立的慢速参数；
3. 等待该机械臂实际完成最后一个点；
4. 执行必要的稳定等待；
5. 调用该命名动作明确绑定的拍摄方法；
6. 调用该命名动作明确绑定的算法方法；
7. 算法成功后动作才算完成。

注意：“最后一个点”指 CSV 中最后一条 `type=arm` 记录，不是简单取 CSV 最后一行。CSV 末尾可能存在其它设备记录。

JSON 动作项参数：

- `speed`：前段运动速度，单位 mm/s；
- `zone`：前段 zone，单位 mm；
- `final_speed`：最后一个 AR5 点的慢速，单位 mm/s；
- `settle_delay`：到位后拍摄前稳定时间，单位 s。

拍摄动作必须等最后一个 AR5 点到位后才调用相机，但不额外要求另一只机械臂同时到位。

### 4.2 快速动作 FastAction

用途：开门、关门、离开、回位和不要求精密接触的转移动作。

执行语义：

- 使用当前 JSON 动作项的 `speed`；
- 使用当前 JSON 动作项的非零 `zone`；
- 不调用相机或算法；
- `open_door`、`close_door` 在开始前进入双臂起点屏障；
- 其它快速动作按本臂固定顺序执行。

速度和 zone 均按动作项独立配置，因此 `open_door`、`close_door`、`go_out` 等快速动作可以使用不同参数。

### 4.3 精确动作 PreciseAction

用途：抓取、放置等必须精确到位的动作。

执行语义：

- 使用当前 JSON 动作项的 `speed`；
- 所有 AR5 运动命令的 `zone` 固定为 `0.0`；
- `zone=0.0` 不提供运行时修改入口；
- 动作完成以本臂到位为准；
- 不隐式调用拍摄算法。

JSON 中仍必须显式写出 `zone: 0.0`，启动前校验其确实为 0；执行入口再次强制使用 0，避免配置校验遗漏。

### 4.4 参数更新规则

速度和 zone 由 JSON 顺序列表中的每个动作项独立维护：

| 参数 | idle 可修改 | busy 可修改 | rapid_stop 可修改 |
| --- | --- | --- | --- |
| JSON 动作项的 speed/zone | 是 | 否 | 否 |
| 拍摄动作的 final_speed/settle_delay | 是 | 否 | 否 |
| 精确动作 zone | 否，固定 0 | 否 | 否 |

删除当前按左/右臂 CSV 数字序号维护的速度和 zone 字典。左右臂动作分别出现在 JSON 的左右顺序列表中，因此天然支持同名动作使用不同速度和 zone。

JSON 只允许在 `idle` 时替换或部署。每次 start 都重新读取并校验文件；校验成功后把动作列表复制为本轮不可变执行计划。运行期间即使磁盘文件被修改，也不能改变当前 `busy` 任务。

## 5. 基于 record_left 名称的二次封装

### 5.1 命名动作与类别

第一版显式定义以下封装，不通过字典查找函数：

| 命名动作 | 动作类别 | index | 附加行为 |
| --- | --- | --- | --- |
| `go_out` | FastAction | 不允许 | 无 |
| `open_door` | FastAction | 不允许 | 双臂起点同步 |
| `before_calibration` | FastAction | 不允许 | 拍摄前转移 |
| `calibration` | CaptureAction | 不允许 | 到位后执行明确的拍摄和算法调用 |
| `get_tray` | PreciseAction | 必须 | 读取 `get_tray_<index>` |
| `after_get_tray` | FastAction | 不允许 | 取托盘后的转移 |
| `put_tray` | PreciseAction | 必须 | 读取 `put_tray_<index>` |
| `before_get_new_tray` | FastAction | 不允许 | 取新托盘前转移 |
| `get_new_tray` | PreciseAction | 必须 | 读取 `get_new_tray_<index>` |
| `before_put_new_tray` | FastAction | 不允许 | 放新托盘前转移 |
| `put_new_tray` | PreciseAction | 必须 | 读取 `put_new_tray_<index>` |
| `calibration_new_tray` | CaptureAction | 不允许 | 当前绑定空 CSV，保持留空 |
| `after_put_new_tray` | FastAction | 不允许 | 放新托盘后的转移 |
| `close_door` | FastAction | 不允许 | 双臂起点同步 |
| `go_home` | FastAction | 不允许 | 无 |

`calibration` 到位后直接调用已有 CameraPipeline 三球检测入口；`calibration_new_tray` 当前明确留空，
对应 CSV 必须为空且不调用算法。两者都必须通过直接函数调用或明确留空实现，禁止用字符串算法名或回调注册表选择。

服务目录中的历史 `finish_new_tray` 资产已按人工确认迁移为 `after_put_new_tray`；它不是新增录制
命名规范，服务不会自动猜测或兼容旧名称。

### 5.2 服务内 JSON 顺序文件

顺序文件固定放在：

```text
record_replay/action_sequence.json
```

该文件是服务部署源文件，必须纳入版本管理、部署清单和 SHA-256 一致性校验。第一版不增加在线流程编辑器，也不通过 HTTP 接收任意动作列表；顺序调整通过修改并部署该 JSON 完成。

JSON 顶层保存版本、统一 deployment 配置和左右臂两个顺序列表。服务不在 JSON 或进程内保存循环次数；GUI
通过每次 `start` 传入旧托盘和新托盘 index 编排循环。每个动作项至少明确写出：

- `function_name`：封闭白名单中的命名动作；
- `type`：`capture`、`fast` 或 `precise`；
- `speed`：该动作的速度，单位 mm/s；
- `zone`：该动作的 zone，单位 mm；
- `index`：仅多目标动作必填；
- `final_speed`、`settle_delay`：仅拍摄动作必填。

以下仅为结构示例，数值不是已确认的现场参数：

```json
{
  "schema_version": 4,
  "deployment": {
    "prior_files": {
      "ball_pose": "prior_data/ball_pose_prior.json",
      "hand_eye_result": "prior_data/hand_eye_result.txt",
      "charuco_board": "prior_data/charuco_board_prior.json",
      "charuco_history": "prior_data/charuco_offset_history.csv",
      "left_head_base_camera": "prior_data/left_head_base_camera.npy",
      "right_head_base_camera": "prior_data/right_head_base_camera.npy"
    },
    "offset": {
      "camera_name": "left_hand_camera",
      "calculate_after_action_name": "calibration",
      "target_action_names": ["get_tray", "put_new_tray"],
      "left_charuco_target_action_names": ["open_door", "close_door"],
      "right_charuco_target_action_names": []
    }
  },
  "left": [
    {"function_name": "go_out", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "open_door", "type": "fast", "speed": 1000.0, "zone": 80.0},
    {"function_name": "before_calibration", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {
      "function_name": "calibration",
      "type": "capture",
      "speed": 500.0,
      "zone": 10.0,
      "final_speed": 100.0,
      "settle_delay": 0.5
    },
    {"function_name": "get_tray", "type": "precise", "speed": 200.0, "zone": 0.0},
    {"function_name": "after_get_tray", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "put_tray", "type": "precise", "speed": 200.0, "zone": 0.0},
    {"function_name": "before_get_new_tray", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "get_new_tray", "type": "precise", "speed": 200.0, "zone": 0.0},
    {"function_name": "before_put_new_tray", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "put_new_tray", "type": "precise", "speed": 200.0, "zone": 0.0},
    {
      "function_name": "calibration_new_tray",
      "type": "capture",
      "speed": 500.0,
      "zone": 10.0,
      "final_speed": 100.0,
      "settle_delay": 0.5
    },
    {"function_name": "after_put_new_tray", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "close_door", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "go_home", "type": "fast", "speed": 1000.0, "zone": 10.0}
  ],
  "right": [
    {"function_name": "go_out", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "open_door", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "close_door", "type": "fast", "speed": 1000.0, "zone": 10.0},
    {"function_name": "go_home", "type": "fast", "speed": 1000.0, "zone": 10.0}
  ]
}
```

要改成从一个位置取、到另一个位置放，可直接在左臂列表中保留 `get_tray` 并增加或替换为 `put_tray`，分别设置源位置和目标位置的 index。要删除、重复或调整已知动作，也通过显式修改数组顺序完成。

### 5.3 最小数据结构与显式调用

只需要：

- 一个三值 `ActionKind`；
- 一个严格的 `ActionSequenceConfig`；
- 一个严格的 `ActionItem`；
- 一个 JSON 加载和校验模块；
- 三个明确执行方法：`execute_capture_action()`、`execute_fast_action()`、`execute_precise_action()`。

`function_name` 是配置标识，不是可执行 Python 表达式。实现使用封闭的显式分支，例如：

```text
match function_name:
    case "go_out": ...
    case "open_door": ...
    case "calibration": ...
    case "get_tray": ...
    ...
    case _: raise ConfigError(...)
```

每个分支只构造已知命名动作，再由 `type` 进入三个底层执行入口。禁止：

- `getattr(executor, function_name)`；
- `globals()[function_name]`；
- `eval` 或 `exec`；
- `dict[str, Callable]` 动作注册表；
- 从 JSON 或 CSV 文件名导入任意模块或方法。

命名动作仍保留必要约束：`calibration*` 必须是 capture，`get*/put*` 必须是 precise，`open_door/close_door` 必须是 fast。JSON 必须显式写出 type，用于可读性和双重校验，但不能绕过这些安全约束。

### 5.4 start 前校验和执行冻结

`POST /start` 的顺序必须调整为：

```text
确认当前为 idle
-> 读取 action_sequence.json
-> 完整校验 JSON 和 CSV 对应关系
-> 完整校验先验
-> 生成不可变的本轮左右臂执行列表和配置 SHA-256
-> 状态切换为 busy
-> 建立设备连接并开始执行
```

start 前至少检查：

- JSON 是合法 UTF-8、合法 JSON，且 `schema_version` 受支持；
- `old_tray_current_index`、`old_tray_put_index`、`new_tray_current_index`、`new_tray_put_index`
  由每次 start 传入且必须是正整数；
- 左右列表非空，动作项字段完整且无未知字段；
- `function_name`、`type` 均在封闭白名单内且彼此匹配；
- `speed`、`zone`、`final_speed`、`settle_delay` 是有限数值且在允许范围内；
- precise 的 zone 必须为 0；
- capture 必须提供 final_speed 和 settle_delay，非 capture 不得携带这两个字段；
- 动作项不得携带 index；四个多目标动作的 index 必须由每次 plan/start 请求传入；
- 每个动作项根据 function_name、index、arm 精确匹配且只匹配一个 CSV；
- open_door 和 close_door 在左右列表中的出现次数相同，能够逐次配对起点同步；
- JSON 引用的全部 CSV 均能成功解析，拍摄动作至少存在一条 arm 记录；
- 先验和算法依赖满足当前列表中实际出现的拍摄动作。

`deployment` 与动作列表由同一个 JSON 读取器校验：其中 `offset` 承载三球触发、头部/三球应用目标及检测参数，
`prior_files` 承载本轮引用的先验文件相对路径。`start` 时将这些内容和 CSV 行一起冻结，busy 期间不重新读取配置文件。

允许 records 目录保存未被本轮 JSON 引用的其它 index CSV，它们是可选动作资产，不应因为未使用而报错。
动作顺序读取阶段只解析当前 JSON 动作的候选 CSV，`refresh_deployment_status()` 只读取已绑定路径；
未引用文件不会进入本轮部署摘要或解析流程，重复匹配仍明确报错而不会静默选择最新时间戳文件。

任何校验失败都保持 `idle`，一次性返回完整错误列表，不连接 AGV、AR5 或其它设备。进入 `busy` 后只使用内存中的不可变计划；状态响应记录本轮 JSON SHA-256，便于定位实际执行的是哪一版顺序。

## 6. 双臂同步新语义

### 6.1 同步点

同步动作仅有：

- `open_door`；
- `close_door`。

不再同步：

- CSV 数字序号；
- `S02/S03/S04`；
- `go_out` 或 `go_home`；
- 任意动作终点。

### 6.2 只同步起点

左右臂各自按本轮已冻结的 JSON 序列执行。到达 `open_door` 或 `close_door` 时：

1. 先完成本臂之前的动作；
2. 在对应命名屏障等待另一臂也准备开始同名动作；
3. 两臂从屏障同时释放并分别下发本臂动作；
4. 屏障立即完成，不等待两臂同时结束；
5. 每只机械臂只等待自己的动作完成；
6. 某一臂先完成后，立即继续该臂的下一动作，不等待另一臂。

实现只需要两个显式同步屏障，例如 `open_door_start_barrier`、`close_door_start_barrier`。不要实现通用“按任意名称创建屏障”的动态同步框架。

重要风险：取消终点同步后，一只机械臂可能在另一只机械臂仍执行开/关门时进入下一动作。实施前必须确认两侧工作空间和后续动作不会干涉，并将这一项作为现场验收门槛。

### 6.3 失败语义

- 任一臂在到达起点屏障前失败：解除另一臂等待并进入 `rapid_stop`；
- 任一臂在同步动作启动后失败：立即进入 `rapid_stop`；
- stop 到来时：两个屏障都应被唤醒并退出，不能永久等待；
- 不允许失败的一侧通过屏障后继续发送动作。

## 7. Rapid Stop 的最小实现

### 7.1 状态机

```text
idle --start--> busy --正常完成--> idle
idle --stop---------------------> rapid_stop
busy --stop/运动阶段失败--------> rapid_stop
rapid_stop --人工处理 + reset--> idle
```

规则：

- `start` 之前为 `idle`；
- 正常开始后为 `busy`；
- 正常执行完一次计划后恢复 `idle`，并将进程内完成次数加一；
- 收到停止信号时立即锁存 `rapid_stop`；
- `rapid_stop` 下拒绝 start、参数修改和所有普通动作；
- 只有人工处理后显式 `POST /reset` 才能恢复 `idle`；
- reset 只修改服务状态，不自动上电、不自动导航、不自动恢复动作、不自动续跑。

为防止重启绕过人工复位，使用一个简单的持久状态文件记录 `idle/busy/rapid_stop` 即可，不引入数据库或事件溯源。服务启动发现上次为 `busy` 或 `rapid_stop` 时，应进入 `rapid_stop`。

### 7.2 Stop 范围

收到 stop 后按固定代码直接执行：

1. 原子设置停止标志和 `rapid_stop`；
2. 禁止执行器发送任何后续动作；
3. 调用 AGV Stop；
4. 调用左 AR5 `robot.stop()`；
5. 调用右 AR5 `robot.stop()`；
6. 防止已有缓冲动作再次 `moveStart()`；
7. 等待当前工作线程退出；
8. 保持 `rapid_stop`，等待 reset。

不为夹爪、灵巧手、升降、头部增加专用急停逻辑，也不建设通用设备停止注册表。但停止标志生效后，这些设备同样不能收到任何新的普通指令。

AGV 与左右 AR5 使用显式字段和显式 stop 调用；不循环遍历动态设备列表。一个 stop 调用失败不能阻止另外两个 stop 调用。

### 7.3 AR5 与 AGV 边界

- 当前 AR5 `robot.stop()` 是 stop2 规划停止、不断电，不等同于硬件安全急停；
- 必须阻止 stop 后再次调用 `moveAppend()`、`moveStart()`；
- AR5 未执行队列是否需要在 stop 后调用 `moveReset()`，实施前按 SDK 时序做人工低速确认；
- AGV `BaseService.Stop` 需要增加明确 RPC 超时并确认返回值；
- 本次不扩展为 PLC、安全继电器或其它设备的急停项目。

## 8. 录制与先验必须保持不变

### 8.1 先验录制

以下内容不修改：

- `prior_data/ball_pose_prior.json`；
- `prior_data/charuco_board_prior.json`；
- 手眼结果和相机外参文件；
- `test/wuji/prior_record.py` 的采集方法、算法和文件内容；
- start 前的先验完整性检查。

### 8.2 CSV 录制

以下内容不修改：

- CSV 表头 `timestamp,type,joints,pose`；
- arm、gripper、m11、lift 等行的录制方式；
- 关节、位姿、单位和时间戳；
- 拖动示教的采样过程；
- 已录制 CSV 的行顺序和数值。

唯一命名变化：

- 不再生成数字顺序前缀；
- 不再生成 `Sxx`；
- 多目标动作在动作名后增加 index；
- 其余动作名、arm 侧别和时间戳保持。

如果录制工具和回放服务共享文件名生成函数，只修改该函数的输出格式；不得借此重构录制控制流。

### 8.3 拍摄动作与先验录制的区别

回放中的 `CaptureAction` 是“移动到当前工作位置后调用已有拍摄和算法接口”，不是重新录制先验。它只能生成本轮运行结果或偏移，不能覆盖先验文件。

原来按 CSV 序号触发的拍摄、偏移计算和偏移应用，必须逐项迁移到明确的命名动作，禁止继续保留隐藏序号条件。

## 9. 分阶段实施

### 阶段 1：命名解析、JSON 配置和离线迁移

- [x] 定义严格文件名解析结果：action、index、arm、timestamp；
- [x] 删除数字序号和 `Sxx` 解析；
- [x] 定义允许 index 的四个动作；
- [x] 新增 `record_replay/action_sequence.json` 和严格配置类型；
- [x] 实现 JSON UTF-8 读取、schema 校验和完整错误收集；
- [x] 实现 function_name/type 的封闭白名单校验；
- [x] 启动前检查缺失和重复文件；
- [x] 为 records 生成快照；
- [x] 只重命名现有 CSV，不修改内容；
- [x] 比较重命名前后 CSV 内容哈希；
- [x] 离线输出解析后的动作资产清单；

验收：JSON 引用的所有文件都能唯一解析；CSV 内容哈希不变；先验文件无变化；非法配置不会建立设备连接。

### 阶段 2：三类动作和命名封装

- [x] 实现 CaptureAction、FastAction、PreciseAction；
- [x] 实现三个显式执行入口；
- [x] 将速度/zone 从 CSV 序号映射改为 JSON 每个动作项的独立参数；
- [x] 精确动作强制 zone=0；
- [x] 拍摄动作最后一个 arm 点使用慢速；
- [x] `calibration` 到位后直接调用 CameraPipeline 三球检测并更新三球 offset；
- [x] `calibration_new_tray` 明确保持未实现并绑定空 CSV，不触发算法；
- [x] 为所有 record_left 名称增加二次封装；
- [x] 禁止动态动作注册和反射。

验收：离线计划能够显示每个命名动作的类别、CSV、index、速度和 zone；无法构造未知动作。

当前 `calibration` 使用现有三球 offset updater；`calibration_new_tray` 允许表头-only CSV，
执行时记录并跳过，不移动、不等待、不触发算法。

### 阶段 3：JSON 顺序列表和多点参数

- [x] 通过左右 JSON 数组承载当前动作顺序；
- [x] 允许通过版本化 JSON 调整、删除、重复已知动作；
- [x] index 由每次 plan/start 请求传入，不写入动作项；
- [x] index 精确解析到唯一 CSV；
- [x] 读取并校验 JSON 中明确的单次动作顺序；
- [x] start 成功前冻结本轮执行列表和 JSON SHA-256；
- [x] 不提供 HTTP 任意动作排序 API；
- [x] 状态进度显示动作名和 index，不显示旧 CSV 序号。
- [x] 状态响应增加 `offset_statuses`，分别显示头部与三球 offset 的可用/应用状态，并拒绝同一动作重叠配置。

验收：同一套代码可通过 JSON 顺序和请求 index 选择不同动作及抓取/放置 CSV，顺序不受目录枚举顺序影响；busy 期间修改磁盘 JSON 不影响当前轮次。

### 阶段 4：双臂起点同步

- [x] 删除 CsvExecutionPlan 的 Sxx 同步模型；
- [x] 左右臂按各自冻结后的 JSON 顺序执行；
- [x] 只增加 open_door 和 close_door 两个起点屏障；
- [x] 删除同步动作终点 join 屏障；
- [x] 一侧完成后允许本侧继续；
- [x] stop/异常可以解除屏障等待；
- [x] 做双臂时间线离线断言：start 前要求左右 `open_door/close_door` 出现次数和相对顺序一致。

验收：两臂 open/close 的开始时间满足同步要求；任一侧后续动作不等待另一侧结束。

### 阶段 5：状态、Stop 和 Reset

- [x] 增加 `rapid_stop`；
- [x] 增加 `POST /stop`；
- [x] 增加 `POST /reset`；
- [x] stop 先锁存状态，再显式停止 AGV 和左右 AR5；
- [x] stop 后阻止所有普通指令，并用每个 AR5 的命令锁串行化队列提交与 robot.stop；
- [x] 增加最小持久状态文件；
- [x] 处理服务信号退出和工作线程退出；
- [x] 不增加其它设备急停网关。

实现核对补充：非运动重试、机械臂运动轮询、AGV 到位轮询、手部/升降轮询及
ChArUco 稳定等待均已接入停止事件；停止事件生效后不会继续重试或下发后续普通指令。
这只是代码和静态层面的核对，尚未替代现场设备停止验证。

验收：停止标志生效后没有任何后续动作发送；reset 不产生硬件命令。

### 阶段 6：文档、版本与人工验证

- [x] 更新 README、API Reference 和 OpenAPI；
- [x] 更新参数说明和 CSV 命名示例；
- [x] 记录 action_sequence.json schema、字段单位和完整示例；
- [x] 升级 RecordReplay 主版本并更新 CHANGELOG；
- [x] 增加状态 WebSocket：内部 `ws://127.0.0.1:6301/api/v1/ws`，外部通过 Gateway 的
  `wss://<orin-host>/api/v1/record-replay-ws` 订阅，连接后立即发送当前快照，状态变化只保留最新消息；
- [x] 运行 UTF-8、ruff、pyright、compileall 等静态检查；
- [x] 不自动运行任何 record_replay 测试或 start；
- [x] 部署前后比较 record_replay 文件清单和 SHA-256；全量四服务部署清单包含 `record_replay/`，远端替换前后
  的 `sha256sum --check` 均通过；
- [ ] 由现场人员依次验证单臂、双臂起点同步、不同 index、GUI 外部循环和 Rapid Stop。

## 10. 主要风险与控制

| 风险 | 等级 | 控制措施 |
| --- | --- | --- |
| 文件名去序号后目录排序不再代表业务顺序 | 高 | 只使用已校验 JSON 数组顺序，不使用目录顺序 |
| JSON 中 function_name 被当成任意方法调用 | 极高 | 封闭白名单和显式 match，禁止 getattr/eval/动态导入 |
| JSON 在 busy 期间被修改 | 高 | start 前复制为不可变计划并记录 SHA-256 |
| JSON 速度或 zone 超出安全范围 | 极高 | start 前做有限值、范围和动作类别约束校验 |
| 多个时间戳文件匹配同一动作/index | 高 | 启动前报重复错误，不自动选最新 |
| 拍摄发生在最后一个 arm 点真正到位前 | 高 | 单独识别最后 arm 点，等待本臂完成后再拍摄 |
| 精确动作仍继承非零 zone | 高 | 精确执行入口内固定 zone=0，不读取外部 zone |
| 双臂只同步起点后发生空间干涉 | 极高 | 现场检查时间线和工作空间，作为发布阻断项 |
| stop 后旧缓冲动作再次启动 | 极高 | 停止标志前置检查，禁止后续 moveStart，确认队列清理时序 |
| 为动作抽象引入通用框架 | 中 | 仅三类动作、封闭命名列表、JSON 顺序数组和显式 match/if |
| 改名过程误改 CSV 内容 | 高 | 快照、只重命名、迁移前后内容 SHA-256 对比 |
| CaptureAction 错误覆盖先验 | 极高 | 回放拍摄结果与 prior_data 写入完全隔离 |

## 11.1 当前代码证据核对

- `test/wuji/record_replay_cli.py` 当前明确：三球 offset 在 `calibration` 后计算，并应用到
  `get_tray`、`put_new_tray`；左臂 ChArUco 应用于 `open_door`、`close_door`。服务配置已按此修正。
- 对参考 CLI 做了 AST 级离线常量对比：左臂 15 个业务动作的旧序号映射、三球触发点、三球目标和头部
  offset 目标均已对应到 JSON；`go_out` 不再同步、只同步 `open_door/close_door` 是本次新设计的明确差异。
  当前 JSON 将四个精确取放动作 speed 设为 200 mm/s，而 CLI 旧默认为 1000 mm/s；这是可调配置，现场安全评审
  仍未完成，不能声称数值行为已经与旧 CLI 完全相同。
- 当前服务已将 `calibration` 与 `calibration_new_tray` 分开处理：统一 JSON 的 offset 触发项默认是
  `calibration`，显式动作分发完成后调用三球 updater；后者空 CSV 留空。
- 三球 updater 在采样前使用统一 JSON 的 `capture_settle_delay_s`，等待期间收到停止信号会立即退出，
  不再发送后续检测请求。
- 命名动作分发后对空 CSV 统一直接返回，offset updater 不会被空的 `calibration_new_tray` 触发。
- AGV 导航提交与 Stop 共享单次命令锁；锁只覆盖 RPC 提交，不覆盖到位轮询，避免停止时等待整段导航。
- Git 历史中的旧回放 CLI 只在旧序号 `calibration` 位置触发三球 offset；当前服务已按最新确认
  将该入口绑定到 `calibration`，而 `calibration_new_tray` 保持空动作，不从旧 CLI 推断新算法。
- 当前代码检索确认 `moveReset`、`moveAppend`、`moveStart` 和直接 `robot.stop()` 均位于
  同侧 AR5 `command_lock` 保护范围内；这只是静态竞态审计，未替代真实停止时序验证。
- 当前执行器在创建左右 runtime、准备设备和进入每个命名动作前均检查共享停止事件；停止竞态
  不会继续建立后续 runtime 或进入后续动作。这仍是静态控制流证据，未替代现场停止验证。
- 已补强建连竞态：左右 runtime 在建连返回后先登记到 `ReplayContext`，再检查停止事件；
  stop 请求落在建连完成与准备阶段之间时，`stop_devices()` 仍可找到对应 AR5 并调用停止路径。
  该修复已通过 RecordReplay 目录级 ruff/pyright、compileall、UTF-8 完整性和差异空白检查，
  仍需现场低速验证停止时序。
- `stop_devices()` 明确并行发起 AGV、左 AR5、右 AR5 的停止调用；一个调用失败不会阻断其它
  停止调用，且每个 AR5 仍由自身 `command_lock` 串行化 `robot.stop()`。
- `run_once()` 的 AGV 导航或回放异常路径会先锁存停止事件并调用同一 `stop_devices()`，再发布
  `rapid_stop`；人工 stop 已锁存 `rapid_stop` 时不会重复提交停止调用。
- `run_once()` 执行入口再次拒绝 `rapid_stop` 和已置位的停止事件，不能从应用层之外绕过状态门。
- WSS 状态快照已补齐 HTTP `/status` 的 `accepted=true` 与 `parameters=null` 字段，保持订阅契约一致。
- WSS 快照离线契约检查已通过：构造 `busy` 且零基 `current_task_index=2` 时，对外
  `current_task_sequence=3`，并包含状态、任务、CSV、offset 和 `accepted` 字段；未启动 WebSocket 服务。
- 被 JSON 引用的 CSV 行已在动作计划构建阶段冻结，执行器只消费计划内存快照，不在 busy
  期间重新读取引用 CSV；部署摘要的行数也直接取同一冻结快照，不再二次读取磁盘。
- 依据参考目录 `C:\Project Documents\鼎泰项目\珞石AR5-5LR\xcoresdk_python-v0.7.1.ar_5.a\example` 的
  `move_example.py` 和同目录 Release stub，`MoveAbsJCommand(target, speed, zone)` 使用独立的
  speed/zone 参数；示例同时出现普通动作 `speed=1000, zone=10`、默认速度 `200`、默认 zone `50`，
  不能把默认值当作所有动作的固定值。Release stub 给出的 SDK 边界为 speed `(0, 4000]` mm/s、
  zone `[0, 200]` mm；服务额外将 speed 下限收紧为 `5` mm/s，并按动作类别校验，具体现场安全值仍待人工评审。
- `record_left/13_calibration_new_tray_left.csv` 当前只有表头；服务对应资产也保持表头-only，
  作为明确留空的 CaptureAction 计划项，不执行动作或算法。
- 统一 JSON 的 `deployment.prior_files` 已接入 start 前先验校验；当前本机离线预检查仍失败：
  `hand_eye_result.txt`、左右 `T_base_camera.npy` 不存在，且 ChArUco 历史中右臂有效样本为 0。
  该失败保持 `idle`，不连接设备；待部署先验补齐后再进行现场验证。两个先验上传接口也已复用
  同一份 JSON 目标路径，不再出现“校验路径”和“写回路径”分离。
- 两个先验替换接口仅允许在 `idle` 且没有活动 worker 时执行，并与 `start` 共用应用锁，
  busy 期间不会改变当前轮次可能读取的先验文件。
- `record_replay/records/` 是独立服务部署资产；当前本机服务目录包含 15 个左臂 CSV 和 4 个右臂 CSV，
  并已按本机 `record_left/`、`record_right/` 的对应录制源更新内容，仅保留服务端动作名、时间戳和多目标
  index 文件名。19 对源文件/服务文件逐一 SHA-256 比较全部一致，未修改任何录制源 CSV 内容；其中新增的
  `before_calibration_left.csv`、`after_get_tray_left.csv`、`before_get_new_tray_left.csv`、
  `before_put_new_tray_left.csv` 也已包含在该 19 对校验中。
- 当前任务状态以左臂 JSON 动作序列为全局 `current_task_index/current_task_sequence` 基准；右臂不再伪造
  追加阶段，而是通过 `current_right_*` 字段发布其独立动作。双臂仅在 `open_door`、`close_door` 起点屏障同步。
- 执行器已在左臂每个动作开始时调用 `advance_execution_task`，在动作和其 offset 处理完成后调用
  `complete_execution_task`；该边界会校验 JSON 任务清单中的左右 CSV 配对和单次顺序，WSS/HTTP 发布的
  `current_task_active` 因此不再只是覆盖式索引；状态更新未提供新 `plan_index` 时会保留当前索引，
  进入 idle 时才清除本轮索引。
- 纯离线不变量断言已通过：新 index 文件名正确解析，capture 末点慢速且 `zone=0` 生效；当前版本另外兼容
  纯数字前缀 CSV。capture 前段和 fast arm 点使用动作项 zone，precise 强制 `zone=0`，状态订阅只保留最新快照，idle 清除任务进度，
  stop 标志置位时 AGV 不发送导航；单次计划只展开一份动作列表。该结果不替代现场运动和停止验证。
- 纯 AST 护栏检查已通过：`record_replay/` 未使用 `getattr`、`eval`、`exec` 或 `__import__` 等动态分发，
  显式动作分发仍使用 `match`；当前 19 个服务 CSV 均通过新命名解析，`get_tray_1_left` 和
  `put_tray_4_left` 的 index 绑定均通过。
- 当前本机 `action_sequence.json` 使用 schema 4，动作项不保存 index；每次 plan/start 的
  `old_tray_current_index`、`old_tray_put_index`、`new_tray_current_index`、`new_tray_put_index`
  分别绑定 `get_tray`、`put_tray`、`get_new_tray`、`put_new_tray`，GUI 负责决定下一次参数。
- 2026-08-07 对 Orin 当前部署文件做只读权限核对：`action_sequence.json` 为 `wuji-brain:wuji-brain`、
  权限 `664`；当前状态只能证明服务用户可读写，不能证明人工修改、评审和部署职责已隔离，因此权限项继续保留待确认。
- 2026-08-07 14:12:44 全量部署已将当前本机版本同步至 Orin；远端暂存区共 205 个文件，
  替换前后清单校验通过，CameraPipeline、RecordReplay、RobotControl 和 API Gateway 的版本检查均通过。
  只读检查确认四个 systemd 服务均为 active，RecordReplay 为 `idle`；本次未执行 `/start` 或回放动作。

## 11.2 实现前仍需现场确认

- [x] `calibration` 对应现有 CameraPipeline 三球检测入口；
- [x] `calibration_new_tray` 明确暂未实现并使用空 CSV；
- [x] 历史 `finish_new_tray` 迁移为 `after_put_new_tray`；
- [x] 当前取放异点使用 `get_tray_1_left` 与 `put_tray_4_left`；
- [x] 多目标 index 按动作独立映射到 CSV 文件名后缀，不要求四类动作共用位置编号；当前为
  `get_tray_1_left` 和 `put_tray_4_left`；
- [x] 拍摄动作前段不使用全局统一值，直接使用对应 JSON 动作项的 `zone`；最终拍摄点固定为 `zone=0`；
- [x] 软件层 speed/zone/final_speed 范围已按 xCoreSDK 工程边界校验：speed `[5, 4000]` mm/s、
  zone `[0, 200]` mm、capture `final_speed <= speed`；具体现场安全值仍由人工评审；
- [ ] action_sequence.json 的人工修改、评审和部署权限；
- [ ] 取消 open/close 终点同步后，两臂工作空间是否始终安全；
- [ ] GUI 外部循环中四个托盘位置 index 的现场业务顺序和异常恢复策略。

这些确认项只影响显式 JSON 字段和顺序，不应引入反射式动态调用或通用流程编辑器。
