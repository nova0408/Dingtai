# RecordReplay API Reference

文档版本：`1.0.5`（2026-08-05）
服务业务版本：`1.12.0`
默认监听：`http://<orin>:6300`

机器可读文件：[OpenAPI 3.1](openapi.yaml)。

## 1. 重要安全边界

RecordReplay 会控制机械臂、AGV、夹爪、M11 和升降机构。服务启动只建立 HTTP 监听，
不会自动开始回放；`POST /start` 会启动真实业务线程。GUI 可以读取状态和配置，但不能
把 `start` 当作普通查询接口调用。任何真实启动都必须由现场人员确认设备区域安全后手动发起。

RecordReplay 通过内部 `CameraPipelineClient` 调用 CameraPipeline 的 ZMQ 业务 client，
不直接使用 CameraPipeline 的 ZMQ wire codec。GUI 和其它项目只需要访问 RecordReplay
的 HTTP API。

## 2. 通用约定

- Content-Type：`application/json; charset=utf-8`。
- 当前服务版本：`1.11.0`，与 `record_replay/CHANGELOG.md` 一致。
- 服务启动只建立 HTTP 监听，不加载先验；`POST /start` 前执行完整先验检查。
- 正式客户端必须访问 API Gateway：`https://<orin-host>/api/v1/record-replay/*`。首次使用前
  必须安装并信任 CasiaHand Root CA，不得关闭证书校验。RecordReplay
  独立的 `6300` 端口只用于人工测试、Orin 本地只读诊断和故障排查；GUI 或其它正式客户端
  不得将 `6300` 作为默认访问入口。Gateway 只转发请求，不改变本服务 API 语义。
- `/status`、`/config`、`/device-status` 均为只读请求，但 `/device-status` 会连接并读取现场设备。
- 服务没有 SSE/WebSocket 状态接口；GUI 建议每 `1` 秒轮询 `/status`。
- HTTP 错误响应仍为 JSON，但只保证 `accepted=false` 和 `error_text` 有值；状态字段来自
  发生错误时的只读快照。

## 3. API 总览

| 方法 | 路径 | HTTP 成功码 | 作用 |
| --- | --- | ---: | --- |
| GET | `/status` | 200 | 读取回放阶段、CSV 清单和任务进度 |
| GET | `/config` | 200 | 读取可调运行参数 |
| POST | `/config` | 200 | 修改并持久化可调运行参数 |
| GET | `/device-status` | 200 | 读取机械臂、夹爪、头部和升降状态 |
| POST | `/prior/ball-pose` | 200 | 校验、备份并替换三球 JSON 先验 |
| POST | `/prior/charuco` | 200 | 校验、备份并替换 ChArUco JSON 先验 |
| POST | `/start` | 202 | 接受一轮回放请求；返回启动时状态快照 |

未列出的路径返回 `404` 和 `accepted=false`。服务当前没有停止回放 API。

## 4. GET /status

```http
GET /status
```

响应对象字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `state` | enum | `idle` 或 `busy`；接受 start 后立即为 `busy`，本轮执行和资源清理完成后为 `idle` |
| `accepted` | boolean | 请求是否被接受；正常查询为 `true` |
| `left_csv_state` | string/null | 当前左臂 CSV 去掉 `left_` 前缀后的状态名 |
| `plan_index` | integer/null | 当前执行计划索引，从 `0` 开始 |
| `error_text` | string/null | 失败原因；正常为 `null` |
| `left_csv_files` | array | 左侧部署 CSV 清单 |
| `right_csv_files` | array | 右侧部署 CSV 清单 |
| `execution_tasks` | array | 按实际执行顺序对齐的左右臂任务 |
| `current_task_sequence` | integer | 当前任务序号，从 `1` 开始；无任务为 `0` |
| `current_task_active` | boolean | 当前任务是否仍在执行 |
| `total_execution_count` | integer | 本次服务进程累计接受的 start 请求数 |
| `current_left_csv` / `current_right_csv` | string/null | 当前正在处理的 CSV |
| `current_left_row` / `current_right_row` | integer/null | 当前 CSV 源数据行，从 `1` 开始 |
| `current_left_total_rows` / `current_right_total_rows` | integer/null | 当前 CSV 总数据行数 |
| `parameters` | object/null | `/status` 中为 `null`；`/config` 响应中填充 |

`left_csv_files[]` / `right_csv_files[]`：

```json
{"name":"001_demo.csv","row_count":120}
```

`execution_tasks[]`：

```json
{
  "sequence": 1,
  "left_csv": "001_demo.csv",
  "right_csv": null,
  "synchronized": false
}
```

连续轨迹会批量提交给控制器，`current_*_row` 表示服务正在调度或处理的 CSV 源数据行，
不表示控制器已经物理到达该点。

状态流程：

```text
    idle -> busy -> idle
    任一阶段异常在清理完成后回到 idle，并在 error_text 保留错误
```

启用 AGV 时只在回放前导航到站点 `1`；回放完成后不自动导航到终点。

## 5. GET /config

```http
GET /config
```

响应除包含完整 `/status` 字段外，`parameters` 必定填充：

| 参数 | 类型 | 默认值 | 范围/说明 |
| --- | --- | ---: | --- |
| `left_move_abs_j_end_linear_speed_mm_s_by_csv_sequence` | object | `{"-1":1000,"4":200}` | 左臂速度级别；键为 CSV 数字序号，`-1` 为默认值，范围 `5.0` 至 `4000.0` mm/s |
| `right_move_abs_j_end_linear_speed_mm_s_by_csv_sequence` | object | `{"-1":1000}` | 右臂速度级别；键为 CSV 数字序号，`-1` 为默认值，范围 `5.0` 至 `4000.0` mm/s |
| `left_move_abs_j_zone_mm_by_csv_sequence` | object | `{"-1":10,"2":80,"4":0,"15":80}` | 左臂 zone 级别；键为 CSV 数字序号，`-1` 为默认值，不能小于 `0` mm |
| `right_move_abs_j_zone_mm_by_csv_sequence` | object | `{"-1":10}` | 右臂 zone 级别；键为 CSV 数字序号，`-1` 为默认值，不能小于 `0` mm |
| `agv_navigation_timeout_s` | number | 600.0 | 大于 `0` 秒 |
| `agv_navigation_poll_interval_s` | number | 1.0 | 大于 `0` 秒 |
| `non_motion_retry_count` | integer | 3 | 大于 `0` |
| `non_motion_retry_delay_s` | number | 0.5 | 不小于 `0` 秒 |

这些参数只影响后续回放轮次，不暴露现场 IP、端口、先验路径、CSV 路径或机械臂型号。

## 6. POST /config

请求 body 是“字段到数字”的 JSON object，可以只更新部分字段：

```json
{
  "left_move_abs_j_zone_mm_by_csv_sequence": {"-1": 10.0, "2": 60.0},
  "non_motion_retry_count": 4
}
```

约束：

- 速度/zone 参数是“CSV 数字序号到 JSON number”的 object；其它参数是 JSON number，boolean 会被拒绝。
- 未知字段被拒绝。
- 空 object 表示保存当前配置并返回完整配置。
- `state=busy` 期间修改会被拒绝；只有 `state=idle` 时允许修改。
- 校验通过后立即以 UTF-8 JSON 原子写入 `record_replay/config.json`，并更新下一轮运行配置。

成功返回 HTTP `200` 和完整 `RecordReplayResponse`，其中 `parameters` 为保存后的值。

## 7. GET /device-status

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

## 8. POST /prior/ball-pose

替换固定文件 `record_replay/prior_data/ball_pose_prior.json`。请求 body 必须是完整的
三球 JSON object；服务会先将内容写入临时文件并执行同样的格式校验，校验失败不会改变
当前先验。

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

## 9. POST /prior/charuco

替换固定文件 `record_replay/prior_data/charuco_board_prior.json`。请求 body 必须是完整的
ChArUco JSON object，替换前会执行格式校验，并按与三球先验相同的规则备份旧文件和原子替换。
该接口不上传 ChArUco 历史 CSV、手眼结果或相机外参；这些部署先验仍由 `/start` 全量检查。

```http
POST /prior/charuco
Content-Type: application/json

{ ...完整 charuco_board_prior.json 内容... }
```

## 10. POST /start

```http
POST /start
Content-Type: application/json

{"enable_agv_navigation": false}
```

body 必须且只能包含 `enable_agv_navigation`，类型必须为 boolean。

| HTTP 状态 | `accepted` | 语义 |
| ---: | --- | --- |
| `202` | `true` | 已创建唯一回放业务线程 |
| `202` | `false` | 已有回放运行，拒绝重复启动；不会并发控制设备 |
| `400` | `false` | body 非法或服务拒绝启动 |

调用前会一次性检查全部运行先验，包括两个 JSON、手眼结果、ChArUco 历史和两侧相机外参。
缺失或格式无效的项目会在 `error_text` 中逐项列出，缺少先验时不会创建回放线程，也不会
触发设备动作。上传接口替换的 JSON 会在下一次人工调用 `/start` 时重新加载。

`enable_agv_navigation=true` 时，回放前导航到站点 `1`；回放完成后不自动返航。
`false` 时跳过回放前导航，但双臂 CSV 回放仍然执行。服务进程启动和 HTTP 就绪不会触发
该接口，也不会自动执行任何动作。

## 11. 客户端示例

```python
from record_replay.client import RecordReplayClient

client = RecordReplayClient(
    "https://<orin-host>",
    api_prefix="/api/v1/record-replay",
)
print(client.get_status())
print(client.get_config())
```

`RecordReplayClient.start()` 只应由现场人员在安全确认后手动调用。GUI 首页推荐只轮询
`get_status()`，根据 `state`、`current_task_*` 和 `error_text` 更新界面。

## 12. 文档变更记录

| 文档版本 | 日期 | 内容 |
| --- | --- | --- |
| `1.0.5` | 2026-08-05 | 增加左右臂/CSV 分级速度与 zone 配置，并统一 `idle`/`busy` 状态 |
| `1.0.4` | 2026-08-03 | 正式 Gateway 入口改为 HTTPS 443，并要求客户端安装 CasiaHand CA |
| `1.0.3` | 2026-08-03 | 明确正式客户端必须通过 API Gateway，6300 独立端口仅用于测试和诊断 |
| `1.0.2` | 2026-08-03 | 增加先验替换接口，明确启动时不加载先验、`/start` 全量校验及 `.archive` 备份语义 |
| `1.0.1` | 2026-07-31 | 统一 OpenAPI operationId 为 `snake_case` |
| `1.0.0` | 2026-07-31 | 补齐五个 HTTP API、状态/设备/配置字段与硬件安全边界 |
