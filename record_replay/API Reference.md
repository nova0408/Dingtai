# RecordReplay API Reference

文档版本：`3.3.0`（2026-08-10）
服务业务版本：`3.3.0`
默认监听：`http://<orin>:6300`

机器可读文件：[OpenAPI 3.1](openapi.yaml)。

## 1. 重要安全边界

RecordReplay 会控制机械臂、AGV、夹爪、M11 和升降机构。服务启动只建立 HTTP 监听，
不会自动开始回放；`POST /start` 会启动真实业务线程。GUI 可以读取状态和配置，但不能
把 `start` 当作普通查询接口调用。任何真实启动都必须由现场人员确认设备区域安全后手动发起。

RecordReplay 通过本服务内的 `camera_client.py` 调用 CameraPipeline 的 HTTP 检测接口，
不导入 CameraPipeline Python 包，也不直接使用其 ZMQ wire codec。GUI 和其它项目只需要
访问 RecordReplay 的 HTTP API。

## 2. 通用约定

- Content-Type：`application/json; charset=utf-8`。
- 当前服务版本：`3.3.0`，与 `record_replay/CHANGELOG.md` 一致。
- 根目录同步脚本支持 `-RecordReplayOnly`，只替换并重启 RecordReplay；替换前检查 RecordReplay
  为 `idle`/`waiting` 且 CameraPipeline 在 6200 端口就绪，替换后校验文件清单、SHA-256、只读
  `/status` 和版本，不发送 `/start`；`runtime_state.json` 等运行产物不纳入清单。
- 服务启动只建立 HTTP 监听，不加载先验；`POST /start` 前执行完整先验检查。
- 正式客户端必须访问 API Gateway：`https://<orin-host>/api/v1/record-replay/*`。首次使用前
  必须安装并信任 CasiaHand Root CA，不得关闭证书校验。RecordReplay
  独立的 `6300` 端口只用于人工测试、Orin 本地只读诊断和故障排查；GUI 或其它正式客户端
  不得将 `6300` 作为默认访问入口。Gateway 只转发请求，不改变本服务 API 语义。
- `/status`、`/config`、`/device-status` 均为只读请求，但 `/device-status` 会连接并读取现场设备。
- 服务提供状态 WebSocket；正式客户端通过 Gateway 的 `wss://<orin-host>/api/v1/record-replay-ws` 订阅，后端内部端口为 `6301`。
- HTTP 非 2xx 响应统一为独立 JSON 错误对象，不复用状态快照：

  ```json
  {"error_code":"invalid_request","error_text":"请求 body 必须是 JSON object"}
  ```

  `error_code` 提供稳定机器判断值，`error_text` 提供中文说明。
- HTTP 状态码约定：`200` 表示同步请求成功，`202` 表示 `/start` 已接受并创建执行任务，
  `400` 表示请求参数或业务状态拒绝，`404` 表示路径不存在，`500` 表示服务内部错误。

## 3. API 总览

| 方法 | 路径 | HTTP 成功码 | 作用 |
| --- | --- | ---: | --- |
| GET | `/status` | 200 | 读取回放阶段、命名 CSV 清单和任务进度 |
| GET | `/plan?old_tray_current_index={old_current}&old_tray_put_index={old_put}&new_tray_current_index={new_current}&new_tray_put_index={new_put}` | 200 | 启动前读取本次 CSV、动作参数和行数 |
| GET | `/config` | 200 | 读取可调运行参数 |
| POST | `/config` | 200 | 修改并持久化可调运行参数 |
| GET | `/device-status` | 200 | 读取机械臂、夹爪、头部和升降状态 |
| POST | `/prior/ball-pose` | 200 | 校验、备份并替换三球 JSON 先验 |
| POST | `/prior/charuco` | 200 | 校验、备份并替换 ChArUco JSON 先验 |
| POST | `/start` | 202 | 接受一次回放请求；返回启动时状态快照 |
| POST | `/stop` | 200 | 停止 AGV/当前左右 AR5，并锁存 `rapid_stop` |
| POST | `/reset` | 200 | 人工处理后清除 `rapid_stop`，恢复 `idle` |

状态 WebSocket：

```text
wss://<orin-host>/api/v1/record-replay-ws
```

连接建立后立即收到当前状态，之后每次状态快照变化推送一条 `event=record_replay.status` 消息。
一次执行成功完成时，额外推送一条 `event=record_replay.completed` 且 `completed=true` 的结束消息；
该消息中的 `total_execution_count` 已经加一，客户端可使用结束事件或计数判断本次完成。
慢客户端只保留最新状态，不会阻塞回放线程；GUI 负责决定下一次 start 的托盘 index，不由服务循环。

未列出的路径返回 `404` 和上述 JSON 错误对象。`/stop` 不会恢复动作，`/reset` 不会自动上电、导航或续跑。

## 4. GET /status

```http
GET /status
```

响应对象字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `state` | enum | `idle`、`busy` 或 `rapid_stop`；停止后必须人工 reset 才能回到 `idle` |
| `accepted` | boolean | 请求是否被接受；正常查询为 `true` |
| `error_code` | string/null | 稳定错误码；正常为 `null` |
| `action_sequence_sha256` | string/null | 当前冻结动作顺序 JSON 的 SHA-256 |
| `left_csv_state` | string/null | 当前左臂命名动作对应的 CSV 状态名 |
| `plan_index` | integer/null | 当前执行计划索引，从 `0` 开始 |
| `error_text` | string/null | 失败原因；正常为 `null` |
| `left_csv_files` | array | 本轮 JSON 实际引用的左侧 CSV 清单 |
| `right_csv_files` | array | 本轮 JSON 实际引用的右侧 CSV 清单 |
| `execution_tasks` | array | 按实际执行顺序对齐的左右臂任务 |
| `current_task_sequence` | integer | 当前任务序号，从 `1` 开始；无任务为 `0` |
| `current_task_active` | boolean | 当前任务是否仍在执行 |
| `total_execution_count` | integer | 本次服务进程累计成功完成的执行次数；每次 start 成功完成时加一，服务重启后归零 |
| `old_tray_current_index` | integer/null | 本次执行使用的旧托盘当前位置 index |
| `old_tray_put_index` | integer/null | 本次执行使用的旧托盘放置位置 index |
| `new_tray_current_index` | integer/null | 本次执行使用的新托盘当前位置 index |
| `new_tray_put_index` | integer/null | 本次执行使用的新托盘放置位置 index |
| `agv_navigation_enabled` | boolean/null | 本次执行是否启用 AGV 导航 |
| `agv_target` | string/null | 本次执行请求的 AGV 目标 |
| `current_left_csv` / `current_right_csv` | string/null | 当前正在处理的 CSV |
| `current_left_action_name` / `current_right_action_name` | string/null | 当前命名动作 |
| `current_left_action_index` / `current_right_action_index` | integer/null | 当前多目标动作 index；普通动作为空 |
| `current_left_row` / `current_right_row` | integer/null | 当前 CSV 源数据行，从 `1` 开始 |
| `current_left_total_rows` / `current_right_total_rows` | integer/null | 当前 CSV 总数据行数 |
| `parameters` | object/null | `/status` 中为 `null`；`/config` 响应中填充 |

`left_csv_files[]` / `right_csv_files[]`：

```json
{"name":"get_tray_1_left_20260630_154830.csv","row_count":120}
```

录制 CSV 文件名允许在动作名前增加纯数字首段前缀，例如
`01_go_out_left.csv`、`10_get_new_tray_left.csv`。前缀只用于动作名匹配；响应中的文件名、
当前处理文件和 CSV 行记录均保留实际文件名，不会改写或复制文件。

`execution_tasks[]`：

```json
{
  "sequence": 1,
  "left_csv": "get_tray_1_left_20260630_154830.csv",
  "right_csv": null,
  "synchronized": false
}
```

连续轨迹会批量提交给控制器，`current_*_row` 表示服务正在调度或处理的 CSV 源数据行，
不表示控制器已经物理到达该点。

状态流程：

```text
    idle -> busy -> idle
    人工 stop 或任一阶段异常 -> rapid_stop -> 人工 reset -> idle
```

启用 AGV 时只在回放前导航到请求中的 `agv_target`；回放完成后不自动导航到终点。
Rapid Stop 期间，AR5 的队列提交与 `robot.stop()` 使用同侧命令锁串行化；该锁只保护指令
提交，不会让运动完成等待阻塞停止流程。

## 5. GET /plan

```http
GET /plan?old_tray_current_index=1&old_tray_put_index=4&new_tray_current_index=1&new_tray_put_index=1
```

该接口只在服务处于 `idle` 且没有活动回放线程时读取
`record_replay/action_sequence.json`，复用 `POST /start` 的动作 JSON、CSV 唯一映射和 CSV
行预加载校验。不连接机械臂、AGV、相机或其它现场设备，不创建执行线程，也不提供修改参数的
请求字段。

响应中的 `left` 和 `right` 是本次单次执行的实际顺序。四个 index 分别应用于 `get_tray`、
`put_tray`、`get_new_tray`、`put_new_tray`；`csv` 始终是磁盘上的实际
文件名，因此 `01_`、`10_`、`11_` 等录制前缀会原样显示。每项字段如下：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `state` | enum | 当前服务状态；正常预览为 `idle` |
| `accepted` | boolean | 是否成功读取计划 |
| `error_code` | string/null | 稳定错误码；正常为 `null` |
| `action_sequence_sha256` | string/null | 本次读取的顺序 JSON SHA-256 |
| `old_tray_current_index` / `old_tray_put_index` | integer/null | 本次预览使用的旧托盘当前位置/放置位置 index |
| `new_tray_current_index` / `new_tray_put_index` | integer/null | 本次预览使用的新托盘当前位置/放置位置 index |
| `left` / `right` | array | 按实际执行顺序展开的动作清单 |
| `error_text` | string/null | 校验失败或非 idle 时的原因 |

计划动作项示例：

```json
{
  "sequence": 1,
  "csv": "01_go_out_left.csv",
  "action_name": "go_out",
  "action_type": "fast",
  "speed": 800.0,
  "zone": 50.0,
  "index": null,
  "final_speed": null,
  "settle_delay": null,
  "row_count": 120
}
```

`speed`、`zone`、`index`、`final_speed` 和 `settle_delay` 仅供查看；要修改它们必须由人工
编辑部署目录中的 `action_sequence.json`，然后重新读取计划。GUI 不提供直接编辑控件。

错误码约定：`invalid_request` 表示请求结构或参数类型错误，`invalid_index` 表示托盘 index
不是正整数或查询值不合法，`invalid_plan` 表示动作 JSON、CSV 或先验校验失败，`busy` 表示
已有执行线程，`rapid_stop` 表示必须先人工 reset，`invalid_state` 表示当前状态不允许该操作，
`stop_failed` 表示停止设备失败，`execution_failed` 表示执行阶段失败，`not_found` 表示路径不存在，
`internal_error` 表示服务内部错误。`error_text` 始终使用中文。

## 6. GET /config

```http
GET /config
```

响应除包含完整 `/status` 字段外，`parameters` 必定填充：

| 参数 | 类型 | 默认值 | 范围/说明 |
| --- | --- | ---: | --- |
| `agv_navigation_timeout_s` | number | 600.0 | 大于 `0` 秒 |
| `agv_navigation_poll_interval_s` | number | 1.0 | 大于 `0` 秒 |
| `non_motion_retry_count` | integer | 3 | 大于 `0` |
| `non_motion_retry_delay_s` | number | 0.5 | 不小于 `0` 秒 |

这些参数只影响后续回放轮次，不暴露现场 IP、端口、先验路径、CSV 路径或机械臂型号。

## 7. POST /config

请求 body 是“字段到数字”的 JSON object，可以只更新部分字段；动作 speed/zone 不在此接口修改：

```json
{
  "non_motion_retry_count": 4
}
```

约束：

- 动作顺序、function_name、type、speed、zone 和 index 必须编辑 `record_replay/action_sequence.json`；该文件只在 idle 到下一次 start 前读取。
- 配置接口中的参数必须是 JSON number，boolean 会被拒绝。
- 未知字段被拒绝。
- 空 object 表示保存当前配置并返回完整配置。
- `state=busy` 期间修改会被拒绝；只有 `state=idle` 时允许修改。
- 校验通过后立即以 UTF-8 JSON 原子写入 `record_replay/config.json`，并更新下一轮运行配置。

成功返回 HTTP `200` 和完整 `RecordReplayResponse`，其中 `parameters` 为保存后的值。

## 8. 动作顺序 JSON

服务在 `POST /start` 建立设备 runtime 前读取并校验固定文件
`record_replay/action_sequence.json`。根节点包含 `schema_version`、
`deployment`、`left` 和 `right` 两个有序数组；`deployment` 统一承载 offset 策略和先验
文件入口。每项必须包含 `function_name`、`type`、`speed`、`zone`，
两个数组都不能为空；多目标动作还必须包含正整数 `index`，普通动作不得携带 `index`。
capture 另外必须包含 `final_speed` 和 `settle_delay`，非 capture 不得携带这两个字段；
服务 speed/final_speed 必须在 `[5, 4000]` mm/s 内（SDK 原始边界为 `(0, 4000]`），zone 必须在 `[0, 200]` mm 内；
fast 的 zone 必须大于 `0`，precise 的 zone 必须为 `0`。fast 的每个 arm 点使用动作项 zone，capture 的前置
arm 点使用动作项 zone，最终拍摄点固定使用 `zone=0`，并在该点使用慢速后进入拍摄/算法阶段；`calibration` 到位后直接调用现有
CameraPipeline 三球检测并更新三球 offset，`calibration_new_tray` 暂未实现且必须绑定空 CSV，
不会调用算法。
SDK 会将 speed 按 `<100`、`100~200`、`200~500`、`500~800`、`>800` mm/s 分为 5 档，
将 zone 按 `<1`、`1~20`、`20~60`、`>60` mm 分为 4 档；JSON 仍保存动作级原始数值。

动作名是封闭白名单，当前包括 `go_out`、`open_door`、`before_calibration`、`calibration`、
`get_tray`、`after_get_tray`、`put_tray`、`before_get_new_tray`、`get_new_tray`、
`before_put_new_tray`、`put_new_tray`、`calibration_new_tray`、`after_put_new_tray`、
`close_door` 和 `go_home`。左右 `open_door`、`close_door` 仅同步起点，
不等待终点。校验失败时保持 idle、返回完整错误列表，并不创建运动 runtime；校验成功后
本轮使用 JSON SHA-256 对应的内存冻结计划，并使用计划构建阶段冻结的引用 CSV 行。

响应中的 `offset_statuses` 固定列出 `head`（头部 ChArUco）和 `three_ball`（三球）两种来源，
并分别给出 `available`、`applied`；同一动作配置重叠时，start 前直接拒绝。

## 9. GET /device-status

```http
GET /device-status
```

该接口只执行现场设备只读诊断，不调用上电、使能、标定、运动或回放指令。回放运行期间
拒绝并发读取，通常返回 `400`。单项设备失败不会遮蔽其它设备结果。

响应：

```json
{
  "all_connected": true,
  "left_arm": {
    "connected": true,
    "error": null,
    "ip": "192.168.100.161",
    "expected_type": "AR5-5_0.8L-W4C1C9-ZY2",
    "robot_type": "AR5-5_0.8L-W4C1C9-ZY2",
    "robot_uid": "...",
    "operate_mode": "...",
    "operation_state": "...",
    "power_state": "...",
    "powered_on": true
  },
  "right_arm": {},
  "gripper": {},
  "head": {},
  "lift": {}
}
```

机械臂字段：`connected`、`error`、`ip`、`expected_type`、`robot_type`、`robot_uid`、
`operate_mode`、`operation_state`、`power_state`、`powered_on`。  
夹爪字段：`connected`、`error`、`online`、`calibrated`、`enabled`、`position`、`state`。  
头部字段：`connected`、`error`、`enabled`、`yaw_deg`、`pitch_deg`。  
升降字段：`connected`、`error`、`enabled`、`height_mm`。

可选数值读取失败时返回 `null`；整个设备项读取失败时 `connected=false` 并填充 `error`。

## 10. POST /prior/ball-pose

仅在 `state=idle` 且没有活动回放线程时，替换统一动作 JSON `deployment.prior_files.ball_pose` 指定的服务内先验文件；默认路径为
`record_replay/prior_data/ball_pose_prior.json`。请求 body 必须是完整的三球 JSON object；
服务会先将内容写入临时文件并执行同样的格式校验，校验失败不会改变当前先验。

```http
POST /prior/ball-pose
Content-Type: application/json

{ ...完整 ball_pose_prior.json 内容... }
```

成功时，旧文件会在服务端备份到
`record_replay/.archive/prior_data/<时间戳>/ball_pose_prior.json`，然后以原子替换方式写入
新文件。若旧文件不存在，`backup_file` 为 `null`。接口只替换 JSON 先验，不接收或生成
`ball_debug_overlay.jpg`；该图片是本地调试证据，不是远端运行依赖。

响应：

```json
{
  "accepted": true,
  "file_name": "ball_pose_prior.json",
  "backup_file": ".archive/prior_data/20260803-153000/ball_pose_prior.json"
}
```

## 11. POST /prior/charuco

仅在 `state=idle` 且没有活动回放线程时，替换统一动作 JSON `deployment.prior_files.charuco_board` 指定的服务内先验文件；默认路径为
`record_replay/prior_data/charuco_board_prior.json`。请求 body 必须是完整的 ChArUco JSON object，
替换前会执行格式校验，并按与三球先验相同的规则备份旧文件和原子替换。
该接口不上传 ChArUco 历史 CSV、手眼结果或相机外参；这些部署先验仍由 `/start` 全量检查。

```http
POST /prior/charuco
Content-Type: application/json

{ ...完整 charuco_board_prior.json 内容... }
```

## 12. POST /start

```http
POST /start
Content-Type: application/json

{"old_tray_current_index": 1, "old_tray_put_index": 4, "new_tray_current_index": 1, "new_tray_put_index": 1, "enable_agv_navigation": false, "agv_target": "1"}
```

body 必须且只能包含 `old_tray_current_index`、`old_tray_put_index`、`new_tray_current_index`、
`new_tray_put_index`、`enable_agv_navigation`、`agv_target`。四个 index 必须是正整数，导航
flag 必须是 boolean，目标必须是非空字符串。

| HTTP 状态 | `accepted` | 语义 |
| ---: | --- | --- |
| `202` | `true` | 已创建唯一回放业务线程 |
| `400` | 无 `accepted` 字段 | body 非法或服务拒绝启动，响应为 `{"error_code":"...","error_text":"中文说明"}` |

调用前会一次性检查全部运行先验，包括两个 JSON、手眼结果、ChArUco 历史和两侧相机外参。
缺失或格式无效的项目会在错误对象的 `error_text` 中逐项列出，缺少先验时不会创建回放线程，也不会
触发设备动作。上传接口替换的 JSON 会在下一次人工调用 `/start` 时重新加载。

`enable_agv_navigation=true` 时，回放前导航到 `agv_target`；回放完成后不自动返航。
`false` 时跳过回放前导航，但双臂 CSV 回放仍然执行。服务进程启动和 HTTP 就绪不会触发
该接口，也不会自动执行任何动作。

## 13. 客户端示例

```python
from record_replay.client import RecordReplayClient

client = RecordReplayClient(
    "https://<orin-host>",
    api_prefix="/api/v1/record-replay",
)
print(client.get_status())
print(client.get_plan(1, 4, 1, 1))
print(client.get_config())
```

`RecordReplayClient.start()` 只应由现场人员在安全确认后手动调用。GUI 应在 start 前读取
`get_plan(old_tray_current_index, old_tray_put_index, new_tray_current_index, new_tray_put_index)` 展示只读动作信息；start 被接受后切换到 Gateway WSS，
根据状态快照中的 `state`、`current_task_*`、结束事件和 `error_text` 更新界面。GUI 自行编排
四个托盘位置 index 的循环，服务每次只执行一份冻结计划。

## 14. 文档变更记录

| 文档版本 | 日期 | 内容 |
| --- | --- | --- |
| `3.2.0` | 2026-08-10 | 增加稳定 `error_code`，并统一中文 `error_text` |
| `3.1.0` | 2026-08-10 | 增加旧/新托盘当前位置与放置位置四个 index 参数 |
| `3.0.0` | 2026-08-10 | 改为单次 start；增加 AGV 目标参数、完成次数和 WebSocket 结束事件 |
| `2.4.0` | 2026-08-10 | 新增启动前只读 `GET /plan`，明确 GUI 预览与启动后 WSS 状态链路 |
| `2.3.0` | 2026-08-10 | 兼容录制 CSV 的纯数字首段前缀，并保留实际文件名处理 |
| `2.0.2` | 2026-08-07 | AGV 与左右 AR5 Stop 调用并行发起，避免单点超时延后其它停止调用 |
| `2.0.1` | 2026-08-07 | 补充停止竞态闸门，明确 calibration 三球检测与 calibration_new_tray 空动作语义 |
| `2.0.0` | 2026-08-06 | 引入命名动作、JSON 顺序、多点 index、Rapid Stop 和 offset 状态 |
| `1.0.5` | 2026-08-05 | 增加左右臂/CSV 分级速度与 zone 配置，并统一 `idle`/`busy` 状态 |
| `1.0.4` | 2026-08-03 | 正式 Gateway 入口改为 HTTPS 443，并要求客户端安装 CasiaHand CA |
| `1.0.3` | 2026-08-03 | 明确正式客户端必须通过 API Gateway，6300 独立端口仅用于测试和诊断 |
| `1.0.2` | 2026-08-03 | 增加先验替换接口，明确启动时不加载先验、`/start` 全量校验及 `.archive` 备份语义 |
| `1.0.1` | 2026-07-31 | 统一 OpenAPI operationId 为 `snake_case` |
| `1.0.0` | 2026-07-31 | 补齐五个 HTTP API、状态/设备/配置字段与硬件安全边界 |
