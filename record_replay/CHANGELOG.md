# RecordReplay 版本日志

当前版本：`3.19.0`

## 3.19.0 - 2026-08-13

- `POST /start` 在创建后台线程和连接设备前完整冻结 CSV；表头、动作类型、有限数、arm joints
  以及 pose 的 9 元 elbow/confData 契约任一不满足时立即以 `invalid_plan` 拒绝启动。
- 删除 6 元 pose 兼容和逐 TCP waypoint 的实时 `cartPosture` 读取；所有 pose 必须直接使用
  CSV 记录的 `has_elbow`、`elbow` 与 8 元 `confData`。
- ChArUco offset 就绪后一次性预编译全部当前条件已满足的 waypoint；三球 offset 在
  `calibration` 后得到时立即一次性预编译全部剩余 waypoint。预编译统一完成最终 TCP、IK、
  跳变门控、软限位、speed 与 zone 冻结，物理 segment 执行期只提交已编译命令。
- 日志以 `precompile_batch_id` 记录预编译开始、完成、逐点最终参数和待 offset 点数；执行期
  仍以 `segment_id` 逐点复述最终 TCP/joints/speed/zone，并记录 append、同步 ready、start 与完成。

## 3.18.1 - 2026-08-13

- 修复跨动作轨迹合并后 `open_door`、`close_door` 同步屏障只约束动作调度、未约束真实
  `moveStart` 的安全问题。
- 同步动作首段改为两阶段提交：左右臂分别完成目标编译、`moveReset` 和 `moveAppend`，
  两侧均 ready 后由第二阶段屏障共同释放 `moveStart`；任一侧失败会打破屏障，禁止另一侧启动。
- 同步动作首段不得跨出该同步动作；等待另一臂 ready 时释放机械臂 command lock，屏障释放后
  重新加锁并复查 stop，避免同步等待阻塞现场停止。
- 日志新增同步首段绑定、ready、屏障释放及实际 `moveStart` 下发开始记录，可按 action、
  arm_side、segment_id 和 command_id 对齐两臂启动时间。

## 3.18.0 - 2026-08-13

- 连续 arm waypoint 不再因命名动作或 CSV 边界强制结束；执行器跨动作聚合最大连续轨迹段，
  只在 gripper、M6、lift、capture、offset 更新、双臂同步屏障或动作序列结束时 flush。
- 保持每个 waypoint 原有动作的 speed、zone、capture 最终点和 offset 选择逻辑；整段在
  `moveAppend` 前一次性完成 TCP offset、IK、跳变门控和软限位处理。
- 为每条物理轨迹段增加本轮唯一 `segment_id`，日志完整记录 flush 原因、动作跨度、每个
  waypoint 的 action/CSV/row、offset 后 TCP、最终 joints、speed、zone、目标来源、钳制结果、command_id、
  append、start 和执行完成阶段。

## 3.17.1 - 2026-08-13

- 修正 acc/jerk 实验的失败语义：读取或设置失败仅记录日志，不能阻止 MoveAbsJ 批次提交。
- RobotControl 尽力读取并记录设置前后的 acc/jerk 实际值，便于现场比较控制器是否接受设置。

## 3.17.0 - 2026-08-13

- 每批 MoveAbsJ 在 `moveAppend` 前由 RobotControl 固定应用 `acc=0.5`、`jerk=0.5`；
  普通连续段、M6 前单点补位和碰撞恢复重发使用相同实验参数。
- 本次参数不加入 `POST /start`、配置更新或动作 JSON，对外契约暂不提供调整入口。
- README 补充 MoveAbsJ speed、zone、加/减速度倍率和加加速度倍率的 SDK 分段、范围、
  生效时机、运动影响与现场调参边界。

## 3.16.1 - 2026-08-13

- 增加独立的 `logs/record_replay.log`，每小时轮转、ZIP 压缩并保留 7 天。
- 日志补充服务版本、启动配置与耗时、未处理异常堆栈和关闭耗时。

## 3.16.0 - 2026-08-12

- MoveAbsJ 等待期间读取 RobotControl 当前 `cmdID` 的 waypoint 到达进度和 `collision_fc` 锁存。
- 收到碰撞后等待 `1 s`，调用 SDK `clearServoAlarm()`，恢复 NRT、自动模式和电机上电状态。
- 从最后一个 `reachTarget=true` waypoint 的下一点重新下发未完成轨迹，完成后继续后续 CSV 操作。
- 每个 waypoint 最多自动恢复一次；同一位置再次碰撞时停止并进入现有失败/rapid_stop 流程。

## 3.15.0 - 2026-08-12

- IK 跳变门控改为以当前 CSV 行记录的原始 joints 作为判断基准，不再比较前后轨迹点。
- TCP 微调候选同样逐一与当前行原始 joints 比较；三次失败后回退该行原始 joints。

## 3.14.0 - 2026-08-12

- 增加 IK 关节跳变门控，按同一命名动作内上一条已下发目标检查单轴最大跳变 `45 deg`。
- 检测到异常跳变时，依次沿基坐标 TCP 的 X/Y/Z 轴各微调 `1 mm` 重算 IK，最多尝试 3 次。
- 三次微调仍无法得到连续 IK 解时，回退该行 CSV 记录的原始 joints，禁止将异常 IK 结果加入 MoveAbsJ 队列。

## 3.13.0 - 2026-08-12

- 每个 AR5 连续段在 `moveAppend`/`moveStart` 前通过 RobotControl 重新读取并缓存七轴软限位。
- append 目标超出软限位时，分别钳制为下限 `+1 deg` 或上限 `-1 deg`；未越界目标不改变。
- 每一行记录最终实际 append 的七轴 rad/deg、目标来源、速度、zone 和发生钳制的轴号。

## 3.12.0 - 2026-08-12

- 左右 AR5 的 `moveStart` 返回成功后，增加 `0.2 s` 状态确认；若仍为 `idle`，按 `0.2 s`
  间隔最多重发 3 次，且每次重试记录启动确认状态和实际等待时间。
- 重发只再次调用 `moveStart`，不重复追加当前 MoveAbsJ 轨迹，避免重复排队。

## 3.11.0 - 2026-08-12

- M6 下发前新增机械臂实时七轴位置检查，必须确认已到达 M6 前的最后一个 arm 轨迹点。
- 连续 arm 段默认仍使用批量 MoveAbsJ；位置未到位时改用单点 MoveAbsJ 补位，补位后再次确认，
  仍未到位则拒绝下发 M6。

## 3.10.0 - 2026-08-12

- MoveAbsJ 首次 1 s 未观察到 `moving` 后，以 0.1 s 间隔读取实时七轴关节位置，最多确认 5 s。
- 5 s 内观察到 `moving` 时继续等待到 `idle`；实时关节位置到达最终目标时完成；仍无法确认时记录 warning 并放行。
- RobotControl 新增只读 `joint-position` 接口，位置到位容差为 `0.2 rad`。

## 3.9.5 - 2026-08-12

- 增加每个 M6 动作的开始、完成、超时和失败日志。
- 机械臂 MoveAbsJ 等待要求先观察到 `moving` 再接受 `idle`；未观察到 `moving` 时至少等待 `1 s`。

## 3.9.4 - 2026-08-12

- 人工复位时同步将 `execution_phase` 清为 `idle`，避免复位后残留 `rapid_stop` 阶段。

## 3.9.3 - 2026-08-12

- 稳定判定严格按连续 3 次实际采样中每个轴最大值与最小值差值 `<= 0.1` 执行，移除额外的运动变化门槛。
- 5 s 内未稳定时记录告警并放行后续流程，不再无限等待。

## 3.9.2 - 2026-08-12

- 右手 M6 下发后固定等待 `0.5 s`，再以 `0.2 s` 间隔读取实际状态判断稳定。

## 3.9.1 - 2026-08-12

- M6 稳定判定先确认实际状态发生变化，再要求连续 3 次采样每轴最大值与最小值差值均不超过 `0.1`。
- 超过 5 s 只记录告警并继续等待，手掌稳定前禁止进入下一条手臂运动指令。

## 3.9.0 - 2026-08-12

### 变更

- 右手 M6 每次下发目标后按 0.1 s 读取一次六轴实际状态，连续 3 次采样中每个轴的最大值与最小值
  差值均不超过 0.1 时认为运动结束；5 s 内未稳定时只记录告警并继续后续流程，不再与下发目标值比较。

## 3.8.1 - 2026-08-12

### 修复

- `moveStart` 遇到 xCoreSDK `ec=-17` 电机使能状态错误时，重试前恢复 NRT 运动模式、自动模式
  和电机上电状态，并确认 `power_state=on`；恢复过程不清空已排入的 MoveAbsJ 队列。

## 3.8.0 - 2026-08-12

### 变更

- 状态快照和状态 WebSocket 新增 `execution_phase`，细分 AGV 导航、设备准备、ChArUco
  初始化、动作起点等待、动作执行、offset 更新和资源释放阶段；顶层 `state` 仍用于表示
  `idle`、`busy` 和 `rapid_stop`。
- HTTP `/status`、`/config` 与 WebSocket 使用同一份具体阶段快照，客户端不需要根据当前动作
  字段自行推断服务阶段。

## 3.7.3 - 2026-08-11

### 修复

- 命名动作失败不再只返回“命名动作执行失败”；错误现在包含臂侧、动作序号、动作名、CSV、
  执行阶段、异常类型和底层 RobotControl/xCoreSDK 原始原因。
- 机械臂连续段只在明确读取到 `idle` 后才允许进入下一段；不再把 `unknown` 当作运动完成，
  也不再在等待 `moveReset` 超时后覆盖上一段运动。
- 针对 xCoreSDK `ec=-60611`（控制器正在保存诊断数据）的短暂 `moveStart` 拒绝增加有限重试，
  避免右臂上一段刚结束时被控制器临时占用导致整轮回放失败。

## 3.7.2 - 2026-08-11

### 修复

- 新增只读 `GET /health`，返回 `service_version`、API 主版本和当前回放状态，供 GUI
  在启动前校验 RecordReplay 版本；健康检查不访问现场设备。

## 3.7.1 - 2026-08-11

### 修复

- 修复右手回放仍按旧 M11 记录类型解析，导致当前 M6 CSV 被拒绝为不支持类型的问题。
  `m11` 已正式替换为 `m6`：M6 CSV 必须包含 6 个 0-1 归一化执行器目标，回放按
  `xcoresdk_arm_cli_test.py` 的状态校验和速度/力控参数下发逻辑执行。
- 同步人工回放 CLI 的 patch 版本为 `1.10.1`，确保本机验证入口与 M6 记录语义一致。

## 3.7.0 - 2026-08-11

### 变更

- RecordReplay 不再创建或持有 xCoreSDK 机器人对象，机械臂身份、状态、TCP、IK、NRT 设置和
  MoveAbsJ 队列全部通过 Orin 本机 RobotControl `/api/v1/ar5/{side}` 显式接口调用。
- 保留原有 `moveReset → 批量 moveAppend → moveStart` 连续轨迹、offset、重试和停止编排；
  HTTP、状态码、响应字段与底层 SDK 错误现在携带明确操作名和原始响应内容。
- 设备先验检查中的左右臂状态也改由 RobotControl 读取，彻底消除第二个 AR5 SDK 连接源。

## 版本号规则

版本号使用 `a.b.c`：

- `a`：重大重构或重大更新。
- `b`：功能调整，包括新增、删除或改变 API、回放流程及设备行为。
- `c`：缺陷修复和不改变功能边界的优化。

每次更新 API 或功能时，必须同步更新当前版本号，并在本文件顶部追加带日期的版本记录。同一批改动只升级一次，按影响最大的改动选择版本位。

## 3.6.1 - 2026-08-11

### 变更

- 启用 `faulthandler`，原生 SDK 触发 `SIGABRT`、段错误等致命信号时向 systemd journal 输出全部 Python 线程栈。
- 补充双臂执行器、左右 runtime、xCoreSDK 构造与每项 NRT 调用、qmlinker 附属设备、ChArUco 头部姿态与检测、命名动作和资源释放的调用前/成功日志，用最后一条日志定位原生进程中止边界。
- WebSocket 增加握手、订阅建立、控制帧关闭、连接异常和清理日志，便于区分客户端主动断开、网络断开与服务进程退出。
- 服务从非 idle 持久化状态恢复时，为 `rapid_stop` 补充稳定 `error_code` 和人工处理说明，不再出现 `rapid_stop` 但错误字段为空。

## 3.6.0 - 2026-08-11

### 变更

- HTTP 请求增加 `X-Request-ID`，未知异常响应包含请求 ID、异常类型和限长后的具体原因，服务日志保留完整堆栈，不再返回无法定位的“服务内部处理失败”。
- 已知路径使用错误 HTTP 方法时返回 JSON `405 method_not_allowed`；请求拒绝、计划冻结、线程启动、状态持久化、执行及设备停止均增加关键日志。
- 启动初始化只把动作计划校验错误归类为 `invalid_plan`；工厂、状态存储和线程启动等真实内部异常不再伪装成计划错误，并在启动失败时回滚 busy 状态。
- 后台线程兜底会把未被执行层记录的异常锁存为 `rapid_stop/internal_error`，状态持久化失败也会写入实时状态并输出完整日志。

## 3.5.0 - 2026-08-11

### 变更

- ChArUco offset 历史安全检查改为使用全部 `accepted=true` 记录组成的全局样本池，不再按
  `arm_side` 分组；启动前只要求全局有效样本达到配置阈值，左右臂运行时共用同一组统计范围。
- CSV 的 `arm_side` 字段继续作为记录来源信息保留，但不再参与有效样本筛选。

## 3.4.0 - 2026-08-11

### 变更

- `action_sequence.json` schema 升级为 4，动作项不再接受或保存 `index`；四个托盘位置 index
  只由每次 `GET /plan` 或 `POST /start` 请求传入并冻结，移除 JSON 默认值回退和双数据源。
- 先验替换只读取统一 JSON 的部署配置，不再为读取先验路径构建缺少运行时 index 的动作计划；
  执行服务只接受已经由请求参数完整冻结的计划。

## 3.3.0 - 2026-08-10

### 变更

- HTTP 非 2xx 响应统一返回独立 JSON 错误对象 `{error_code, error_text}`；OpenAPI 明确声明
  400、404 和 500 响应及其错误模型。业务拒绝不再复用完整状态响应。

## 3.2.0 - 2026-08-10

### 新增

- HTTP、状态和 WSS 响应增加稳定 `error_code`；`error_text` 统一返回中文说明，并区分请求、index、计划、状态和执行错误。

## 3.1.0 - 2026-08-10

### 变更

- `GET /plan` 和 `POST /start` 改为接收四个托盘位置 index：旧托盘当前位置、旧托盘放置位置、
  新托盘当前位置、新托盘放置位置，分别绑定 `get_tray`、`put_tray`、`get_new_tray`、`put_new_tray`。

## 3.0.0 - 2026-08-10

### 变更

- 服务改为单次执行：移除 `loop_count` 和服务内部循环；GUI 负责在托盘位置 index 之间编排下一次 `start`。
- `GET /plan` 与 `POST /start` 接收托盘位置参数；`start` 另接收 `enable_agv_navigation` 和
  `agv_target`，共同冻结本次计划。
- `total_execution_count` 改为进程启动以来成功完成的执行次数，仅在执行完成时递增。
- WebSocket 在完成时推送一次 `event=record_replay.completed` 且 `completed=true` 的结束消息，消息中的计数已递增。

## 2.4.0 - 2026-08-10

### 新增

- 新增只读 `GET /plan`，在启动前返回左右臂 CSV、动作类型、speed、zone、index、末点速度、稳定等待和 CSV 行数。
- GUI 可在执行前展示服务端计划；`POST /start` 被接受后继续通过 Gateway WSS 接收实时执行状态。

## 2.3.0 - 2026-08-10

### 新增

- 服务读取左右臂录制 CSV 时兼容 `01_`、`10_`、`11_` 等纯数字首段前缀；前缀只用于动作名匹配，实际执行、状态展示和 CSV 行记录仍保留并使用原始文件名。

## 2.2.1 - 2026-08-07

### 修复

- 将 CameraPipeline 的检测协议和 HTTP 客户端收回 RecordReplay 服务目录；运行时不再导入 CameraPipeline Python 包或仓库 `src/`。
- 保持 RecordReplay 对外 HTTP API 和回放动作语义不变。

## 2.2.0 - 2026-08-07

### 修复

- 修正拍摄动作的 zone 语义：前置拍摄点直接使用动作 JSON 中配置的 zone，最终拍摄点固定使用 `zone=0`；拍摄动作的最终点仍使用 `final_speed`。

## 2.1.10 - 2026-08-07

### 新增

- 根目录同步脚本增加 `-RecordReplayOnly` 专用部署入口，只同步并重启 RecordReplay，执行 staging、文件清单、SHA-256、旧目录归档和只读 `/status`/版本校验；不会操作其它服务或发送 `/start`。该入口会先确认 CameraPipeline 已在 6200 就绪，并排除运行时 `runtime_state.json`。

## 2.1.9 - 2026-08-07

### 修复

- 先验替换接口现在仅允许在 `idle` 且无活动 worker 时执行，并与 `start` 共用应用锁，
  防止 busy 期间改变当前回放使用的先验文件。

## 2.1.8 - 2026-08-07

### 修复

- 状态 WebSocket 快照补齐 HTTP `/status` 的 `accepted=true` 与 `parameters=null` 字段，
  与 API Reference 声明的状态字段保持一致。

## 2.1.7 - 2026-08-07

### 修复

- 在 `RecordReplayCycleService.run_once()` 执行边界再次拒绝 `rapid_stop` 或已置位的停止事件，
  防止非 HTTP 调用路径绕过应用层状态门。

## 2.1.6 - 2026-08-07

### 修复

- AGV 导航或回放阶段异常时，服务先锁存停止事件并自动调用 AGV/左右 AR5 的停止流程，
  再发布 `rapid_stop`；人工 stop 已锁存状态时不重复提交停止调用。

## 2.1.5 - 2026-08-07

### 修复

- 部署摘要的 CSV 行数改为直接读取 start 前冻结的 `preloaded_rows_by_path`，不再重新读取磁盘，
  确保状态清单与实际执行快照一致。

## 2.1.4 - 2026-08-07

### 修复

- 修正 arm 段 zone 选择：快速动作和拍摄动作的所有 arm 点均使用 JSON 动作项的非零 zone，
  只有精确动作在执行入口强制使用 `zone=0`；capture 仍仅在最后 arm 点切换 `final_speed`。

## 2.1.3 - 2026-08-07

### 修复

- 动作顺序读取阶段只解析 JSON 当前引用动作的候选 CSV；未引用的其它 index 或非法命名资产不会阻塞本轮启动，
  同一 `(function_name, index, arm)` 的重复候选仍会明确报错。

## 2.1.2 - 2026-08-07

### 修复

- 部署摘要只读取当前 `action_sequence.json` 实际引用的 CSV；`records` 目录中未引用的其它 index CSV
  作为可选动作资产保留，不再阻塞本轮启动或被错误解析。

## 2.1.1 - 2026-08-07

### 修复

- 先验上传接口现在复用 `action_sequence.json` 的 `deployment.prior_files` 目标路径，
  与 `start` 前校验和回放执行使用同一份统一配置。
- 执行器现在在左臂每个 JSON 任务开始和完成时原子更新任务状态，并校验实际 CSV 顺序与已发布任务清单一致；
  右臂继续只发布独立动作进度。
- 修正状态更新时的可选 `plan_index` 语义：未提供索引时保留当前运行索引，进入 idle 时仍清除本轮任务进度。
- 使统一 JSON 中的 `capture_settle_delay_s` 真正作用于三球采样前稳定等待，并支持被停止事件打断。
- 空 CSV 动作在命名动作分发后直接结束，不再继续进入 offset updater；因此表头-only 的
  `calibration_new_tray` 始终保持留空语义。
- AGV `navigate_to` 与 Stop 现在共享单次命令提交锁，停止事件置位后不会再与新的导航提交竞态；
  到位轮询不持有该锁，Stop 不会等待整段导航完成。

## 2.1.0 - 2026-08-07

### 变更

- 补齐本机 `record_left` 中的四个转移 CSV，并恢复服务端左臂与人工 CLI 一致的 15 项动作顺序；历史无时间戳 CSV 允许使用 `<action>_<arm>.csv` 命名。
- 按本机 `record_left`、`record_right` 录制源刷新服务端 19 个动作 CSV，保留服务端动作名、时间戳和 index 文件名，逐文件 SHA-256 全部一致。
- `action_sequence.json` 升级为统一部署配置，集中读取动作、速度、zone、index、offset 策略和先验文件入口；`start` 前冻结本轮配置。
- `start` 现在先冻结 JSON，再按 JSON 中的 `deployment.prior_files` 校验先验；缺失或无效时保持 idle 并发布错误状态。
- AGV 是否导航继续由 `POST /start` 的 `enable_agv_navigation` 参数决定。
- 新增 RecordReplay 状态 WebSocket；服务内部监听 6301，正式客户端通过 Gateway 的 `wss://<orin-host>/api/v1/record-replay-ws` 订阅状态变化。

## 2.0.2 - 2026-08-07

### 修复

- Stop 请求现在并行发起 AGV、左 AR5、右 AR5 的显式停止调用，避免单个 AGV Stop 超时延后
  其它已连接机械臂的 `robot.stop()`；失败仍会汇总并锁存 `rapid_stop`。

## 2.0.1 - 2026-08-07

### 修复

- 在创建左右机械臂 runtime 和进入每个命名动作前增加停止闸门，避免停止请求与 worker
  初始化竞态导致停止后继续准备设备或进入后续动作。
- 修正文档与当前 CaptureAction 语义：`calibration` 使用 CameraPipeline 三球检测，
  `calibration_new_tray` 绑定空 CSV 并明确留空。

## 2.0.0 - 2026-08-06

### 变更

- 将回放资产从数字前缀/Sxx 阶段改为显式命名 CSV；多目标动作使用动作名后的 index。
- 新增 `action_sequence.json`，以左右有序列表承载 `function_name`、`type`、`speed`、`zone`、
  `index` 和 capture 慢速参数；`POST /start` 前完成 UTF-8、schema、白名单、范围、CSV 唯一映射
  和 capture 依赖校验，成功后冻结 JSON SHA-256 对应的内存计划。
- 执行器按 capture、fast、precise 三类动作工作；precise 强制 zone=0，capture 最后 arm 点
  使用 final_speed，并在到位后调用显式算法入口；open_door/close_door 只同步双臂起点，
  start 前额外校验两臂同步动作的出现次数和相对顺序。
- 按当前 `record_left` 录制命名补齐 before/after 转移动作和 `put_tray` 的显式封装；同时按
  人工 CLI 语义修正三球 offset 目标为 `get_tray`、`put_new_tray`，左臂 ChArUco 目标为
  `open_door`、`close_door`。
- Rapid Stop 的停止事件已覆盖非运动重试、机械臂/AGV 到位轮询、手部/升降等待和 ChArUco
  稳定等待；停止锁存后不再继续重试或发送后续普通指令。
- 补充 start 前左右 JSON 动作列表非空校验，清理旧的 CSV 状态前缀兼容语义。
- 修正 rapid_stop 下配置更新的状态门，避免先持久化参数后才因状态拒绝请求。
- `execution_tasks` 状态摘要按 JSON 的有限 `loop_count` 展开，和实际循环执行顺序一致。
- Rapid Stop 的设备停止调用失败原因会锁存到状态响应的 `error_text`，便于现场处理。
- 明确 `calibration` 到位后调用 CameraPipeline 三球检测；`calibration_new_tray` 保持空 CSV
  留空，不调用算法；历史 `finish_new_tray` 资产迁移为 `after_put_new_tray`。
- 将当前左臂多点计划绑定为 `get_tray_1_left` 与 `put_tray_4_left`，并在状态响应中列出头部
  与三球 offset 的可用/应用状态，拒绝同一动作同时使用两种 offset。
- 为每个 AR5 增加显式命令锁，串行化准备、MoveReset/MoveAppend/MoveStart 和 robot.stop，
  降低停止请求与队列提交并发交错的风险；运动等待不持有该锁。
- README HTTP API 总览补充 `POST /stop` 与 `POST /reset`，与 OpenAPI 和 API Reference 对齐。
- 被 JSON 引用的 CSV 行在 start 前冻结进动作计划，执行期不再重新读取磁盘文件。
- 原有 14 个服务 CSV 迁移时只改文件名，不修改内容；`put_tray_4_left` 原样复制已确认录制，
  `calibration_new_tray` 服务资产按最新要求保持表头-only；先验文件和先验录制流程未改动。
- 按 xCoreSDK 协作机器人接口文档收紧动作计划 zone 上限为 200 mm；speed 上限保持 4000 mm/s。
- 文档补充 xCoreSDK speed 五档和 zone 四档映射；JSON 仍按动作保存可调原始数值。

## 1.12.0 - 2026-08-05

### 变更

- 按 xCoreSDK `MoveAbsJCommand(target, speed, zone)` 的分层语义，将速度和 zone 配置改为
  左右臂分别维护、且按 CSV 数字序号覆盖；`-1` 为对应机械臂默认级别，默认值与
  `test/wuji/record_replay_cli.py` 一致。
- `GET /config` 和 `POST /config` 暴露上述四组映射，服务仅在 `idle` 状态允许修改，校验
  速度范围、zone 范围和默认级别后原子持久化。
- 服务状态统一为 `idle`/`busy`：服务启动和每轮执行前为 `idle`，接受 `start` 后立即进入
  `busy`，包括 AGV、设备准备、CSV 回放和资源清理，直到本轮结束后恢复 `idle`。

## 1.11.0 - 2026-08-03

### 变更

- RecordReplay 服务启动时只建立 HTTP 监听，不再因缺少先验或本地调试 overlay 文件而启动失败。
- `POST /start` 在创建回放线程前一次性检查全部运行先验，并在 `error_text` 中逐项报告缺失或无效项目。
- 新增 `POST /prior/ball-pose` 和 `POST /prior/charuco`，支持校验后原子替换两个 JSON 先验；旧文件备份到服务端 `record_replay/.archive/prior_data/<时间戳>/`。
- `ball_debug_overlay.jpg` 明确为本地调试证据，不再作为远端运行依赖；远端部署清单不包含服务端 `.archive` 备份目录。

## 1.10.0 - 2026-07-31

### 变更

- RecordReplay 服务执行语义与已验证的 `test/wuji/record_replay_cli.py` 对齐，人工 CLI
  同步增加 `RECORD_REPLAY_CLI_VERSION = "1.10.0"`，服务包同步提供
  `RECORD_REPLAY_VERSION = "1.10.0"`。
- 服务 CSV 发现改为只接收纯数字首段并按数值排序；零字节或只有表头的 CSV 继续参与
  计划和状态清单，但执行时只记录警告并跳过，不触发动作或 offset。
- 服务在创建设备 runtime 前预解析执行计划中的全部 CSV；MoveAbsJ 速度和 zone 按左右臂
  与 CSV 序号字典选择，三球 offset 触发文件不再切换临时速度。
- 服务补齐 ChArUco 头部姿态、稳定帧检测、同侧历史统计安全门和最多三次重新检测，
  并按测试脚本的左右臂 CSV 序号应用缓存 offset；offset 后 IK 失败时回退到原 CSV joints。
- 删除服务运行参数中的独立 `offset_trigger_speed_mm_s`，避免与测试脚本已经验证的
  CSV 速度语义分叉。
- AGV 行为与人工 CLI 对齐：启用时仅在回放前导航到站点 `1`，不再在回放完成后自动
  导航到终点。

## 1.9.0 - 2026-07-30

### 变更

- 人工 CLI 支持使用零字节或只有表头的 CSV 作为序号占位文件；占位文件继续参与排序
  和双臂执行计划构建，执行时记录警告并跳过，不触发对应 offset 或设备动作。
- 删除人工 CLI 遗留的 offset 触发 CSV 固定 700 mm/s 临时速度；触发文件现在直接
  使用对应机械臂按 CSV 序号配置的 MoveAbsJ 速度。
- 人工 CLI 的右手 M11 执行器索引改为入口内明确配置，不再从已切换为 M6 手部的
  `xcoresdk_arm_cli_test.py` 导入已经删除的旧符号。
- 人工 CLI 的 ChArUco offset 候选未通过历史安全检查时重新检测并计算，最多尝试
  3 次；仅当三次候选均被安全门拒绝时才终止本轮回放。
- 根据左臂 9 条有效历史与现场三次检测结果，将 ChArUco offset 统计范围从 3σ
  调整为 4σ、平移绝对上限从 50 mm 调整为 60 mm；旋转绝对上限仍保持 5°，
  接纳轻微边界波动，同时继续拒绝明显异常检测。

## 1.8.0 - 2026-07-29

### 变更

- 人工 CLI 在接受 ChArUco offset 前新增同侧历史样本统计安全门：xyz/rpy 分量采用
  均值 ±3σ，平移和旋转模长同时受历史 3σ 与 50 mm/5° 绝对上限约束。
- 新增 `prior_data/charuco_offset_history.csv`，汇总现有 10 次 `T_charuco_off`；
  已知撞机样本标记为拒绝。运行时只读该表，后续数据必须经人工判断后手动录入。
- 异常 offset 在写入 runtime 和进入后续纠偏运动前抛出错误，不会自动修改历史 CSV。
- 人工 CLI 的 MoveAbsJ 末端线速度与 zone 改为按 CSV 数字前缀配置的字典，左右臂
  分别维护显式命名的配置且各以 `-1` 为本侧默认值；删除旧的左臂 CSV 末尾强制零
  zone 列表，特殊 CSV 直接在对应机械臂配置中设置速度和 zone。

## 1.7.0 - 2026-07-29

### 变更

- 三球 offset 检测改为宽 HSV 基准与窄 HSV 复检的分级策略；窄范围未检出或逐球
  球心差异超过 8 mm 时告警并回退宽范围；宽、窄阶段各允许 3 次完整检测尝试，
  避免单一窄范围或瞬时稳定帧波动导致整轮回放失败。
- 宽窄检测共用先验物理直径、模型坐标和动态颜色顺序；人工 CLI 直接复用正式服务
  检测器，不再维护固定黄、红、蓝排序和重复聚合实现。
- 人工 CLI 的手部纠偏恢复使用三球检测生成的 `global_cartesian_offset`，不再临时
  复用头部 ChArUco offset。
- GUI 三球先验改用 30 帧 HSV 聚合，Hue 使用环形均值、6–8 自适应半宽并支持红色
  179/0 跨界双范围；运行期 S/V 至少放宽至 140/120 下限，避免球面掩码碎片化。
- `test/wuji/ball_pose_detection.py` 增加宽窄分级策略的无运动相机验证阶段。

## 1.6.1 - 2026-07-29

### 修复

- GUI 三球先验保存器补充 `tcp_pose_matrix`，修复其生成的
  `ball_pose_prior.json` 无法被人工 CLI 和 RecordReplay 服务加载的问题。
- TCP 齐次矩阵在 AR5 状态读取时直接由 SDK 原始 `trans(m)+rpy(rad)` 按小写外禀
  `xyz` 构造，避免从 GUI 的 mm/deg 展示字段反向恢复计算矩阵。

## 1.6.0 - 2026-07-29

### 变更

- 本机 `test/wuji/record_replay_cli.py` 默认使用 GUI 写入
  `record_replay/prior_data/` 的三球与 ChArUco 先验，不再读取旧测试归档。
- 人工 CLI 默认关闭 AGV 并保持左臂单臂模式；CSV 仅接受纯数字首段文件名，并按
  前缀数值顺序执行，非数字前缀文件不再进入回放计划。
- README 明确区分本机人工 CLI 与 Orin HTTP 服务各自读取的 CSV 和先验位置。

## 1.5.1 - 2026-07-29

### 修复

- CameraPipeline 部署流程在同步两个服务后统一恢复并验证 RecordReplay，不再因
  `-CameraPipelineOnly` 提前退出而把业务服务永久留在停止状态。
- CameraPipeline 公共 API 或线协议更新后强制重启 RecordReplay，使常驻进程加载
  同一批部署的客户端代码，避免服务端与业务端协议版本不一致。
- `-CameraPipelineOnly` 仅保留为纯重启模式的服务选择器；分阶段部署改用语义明确的
  `-SkipRecordReplayOverlayCheck`，该参数只跳过产物门槛，不改变双服务恢复流程。
- 本机总控脚本可通过显式 `--non-interactive` 安全重启 RecordReplay；该模式只执行
  systemd 重启和只读 `/status` 检查，同时修复重启脚本遗漏 Python 路径的问题。
- 两个服务的独立重启脚本改为等待 systemd 重启事务完成后再检查业务就绪，避免旧
  进程端口尚未释放时被误判为新版本已经加载。

## 1.5.0 - 2026-07-28

### 变更

- 三球纠偏不再在配置或网关中硬编码黄、红、紫颜色；服务启动时从先验 JSON 的
  `local_coordinate_frame` 读取原点、X 轴和平面提示球颜色。
- 加载后的三球先验按坐标语义排序并原样传入 CameraPipeline，运行期检测只依据先验
  中的颜色与专属 HSV 范围。

## 1.4.0 - 2026-07-27

### 新增

- `GET /status` 增加左右臂已部署 CSV 清单、每个文件的数据行数、按实际顺序展开并
  左右对齐的任务清单、当前任务序号，以及服务进程启动以来累计接受的执行请求次数。
- 状态同时提供左右臂当前 CSV、当前源数据行和总行数，供 GUI 主页轮询展示执行位置；
  同步运动位于同一个一级任务，单臂运动在另一侧留空。
- 服务启动及每轮执行前刷新部署清单；每次实际执行阶段开始前切换当前任务，双臂并行
  回放时分别发布各自进度；任务完成后单独清除活跃标记，避免返航阶段继续指示最后一个
  CSV 正在执行。
- GUI 主页改为左右 AR5 独立任务组件，增加 AGV 导航选项和经安全确认的启动按钮；
  停止按钮仅预留界面，不发送未定义的停止请求。

### 说明

- 当前行表示服务正在调度或处理的 CSV 源数据行。连续机械臂轨迹会批量提交给控制器，
  因此该字段不表示控制器已经物理到达对应轨迹点。

## 1.3.4 - 2026-07-23

### 部署

- 修复主线程退出 HTTP 请求循环后调用 `HTTPServer.shutdown()` 造成的停机自锁；服务
  启停日志增加配置、先验加载、HTTP socket 关闭、业务线程等待及实际耗时。
- RecordReplay systemd 进程启动、停止上限和 HTTP 业务就绪等待均为 10 秒；
  `Type=simple` 拉起进程后仍以 `/status` 响应作为独立的业务就绪条件。
- 独立重启脚本改为只管理 `record-replay.service`，不再创建脱离 systemd cgroup 的
  `setsid` 后台进程；仍保留交互式现场安全确认，且不会发送 `/start`。

## 1.3.3 - 2026-07-23

### 部署

- 本机同步脚本移除 `DEPLOY` 人工确认，sudo 密码已配置时可直接完成打包、校验、
  远端备份、文件替换和 systemd 重启；仍保留 RecordReplay `waiting` 状态硬校验。

## 1.3.2 - 2026-07-23

### 修复

- 将 `record-replay.service` 的停止超时由 300 秒缩短为 10 秒；服务已经完成业务清理
  但仍有第三方后台线程未退出时，由 systemd 在 10 秒后清理进程组，避免部署长时间阻塞。

## 1.3.1 - 2026-07-23

### 部署

- 新增本机一键同步与重启脚本，完整同步 `record_replay/` 部署源文件、生成远端备份，
  并在替换前后执行文件清单与 SHA-256 校验。
- 脚本仅在 RecordReplay 处于 `waiting` 时允许停止服务；默认要求新先验 debug overlay，
  重启后校验 systemd、6300 端口和只读 `/status`，不会发送 `/start`。

## 1.3.0 - 2026-07-23

### 变更

- RecordReplay 加载三球先验时强制校验三个颜色、有效相对位置关系及每球 30 帧标定
  `hsv_ranges`，并校验 30 帧/最少 24 个保留帧元数据和 debug overlay 核验图；无效
  文件直接阻止服务启动，不再回退到会被误认为先验采集的占位球心。
- 本机回放 CLI 同步要求完整记录先验，常规 offset 检测继续关闭 debug。

## 1.2.0 - 2026-07-23

### 变更

- 加载三球先验时同步读取每球 `hsv_ranges` 并传给 CameraPipeline；现场重新执行
  30 帧先验记录后，offset 检测优先使用标定窄颜色范围。

## 1.1.0 - 2026-07-23

### 变更

- 三球先验尺寸字段由 `radius_mm` 迁移为 `diameter_mm`，适配 CameraPipeline 协议
  版本 7；现有部署先验中的半径测量值已转换为直径值。

## 1.0.0 - 2026-07-21

### 初始基线

- 提供独立的 RecordReplay HTTP 服务和公共客户端。
- 提供回放启动、运行参数和设备静态状态接口。
- 直接使用 qmlinker 设备对象连接双臂、夹爪、头部和升降机构。
- 支持可选 AGV 的回放流程，以及固定先验与左右臂 CSV 记录部署。
