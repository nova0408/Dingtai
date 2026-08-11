# Calibration Service API Reference

文档版本：`1.3.0`（2026-08-11）
默认内部地址：`http://127.0.0.1:6600`  
正式外部前缀：`https://<orin-host>/api/v1/calibration`

机器可读协议见 [OpenAPI](openapi.yaml)。标定服务只读取 RobotControl 状态、调用
CameraPipeline 拍摄和执行计算，不发送设备控制请求。设备姿态由其它服务或现场人员负责。

## GET 接口

| 方法 | 内部路径 | 作用 |
| --- | --- | --- |
| GET | `/api/v1/status` | 读取任务状态、当前标定类型和样本数量 |
| GET | `/api/v1/results/hand-eye` | 读取左手眼在手上 `T_tool_cam` |
| GET | `/api/v1/results/head-eye/{arm_side}` | 读取头部眼在手外 `T_base_camera` |
| GET | `/api/v1/results/prior/head` | 读取头部 ChArUco 先验 JSON |
| GET | `/api/v1/results/prior/hand` | 读取手部三球先验 JSON |
| GET | `/api/v1/hand-eye/config` | 读取手眼 ChArUco 默认参数和当前 OpenCV 可用字典 |

通过 Gateway 访问时，把内部 `/api/v1` 替换为 `/api/v1/calibration`，例如：
`GET /api/v1/calibration/results/head-eye/left`。

结果路径固定为：

- `record_replay/prior_data/hand_eye_result.txt`：左手眼在手上，矩阵语义 `T_tool_cam`；
- `record_replay/prior_data/left_head_base_camera.npy` 或
  `right_head_base_camera.npy`：头部眼在手外，矩阵语义 `T_base_camera`；
- `record_replay/prior_data/charuco_board_prior.json`、`ball_pose_prior.json`：头部和手部先验。

不存在的结果返回 `accepted=false` 和可复制的错误文本。

## 写接口

| 方法 | 内部路径 | 作用 |
| --- | --- | --- |
| POST | `/api/v1/start` | 提示标定开始并清空对应内存样本，不拍摄、不控制设备 |
| POST | `/api/v1/end` | 提示标定结束并封存样本，不自动求解 |
| POST | `/api/v1/cancel` | 丢弃样本和待确认结果，不替换正式文件；待确认结果超时 30 秒也会自动丢弃 |
| POST | `/api/v1/replacements/confirm` | 30 秒内二次确认后替换正式文件，并重命名保留旧文件 |
| POST | `/api/v1/prior/head` | 拍摄并计算头部先验，结果进入 30 秒待确认缓存 |
| POST | `/api/v1/prior/hand` | 读取当前 AR5 状态，拍摄并计算手部先验，结果进入 30 秒待确认缓存 |
| POST | `/api/v1/hand-eye/sample` | 采集左手眼在手上样本 |
| POST | `/api/v1/head-eye/sample` | 采集指定侧头部眼在手外样本 |
| POST | `/api/v1/hand-eye/solve` | 求解并缓存 `T_tool_cam`，等待二次确认 |
| POST | `/api/v1/head-eye/solve` | 求解并缓存指定侧 `T_base_camera`，等待二次确认 |
| PATCH | `/api/v1/hand-eye/config` | 部分修改手眼 ChArUco 默认参数，不触发拍摄 |

手眼采样的默认板参数为：

```json
{
  "dictionary_name": "DICT_APRILTAG_16H5",
  "squares_x": 4,
  "squares_y": 4,
  "square_length_mm": 20.0,
  "marker_length_mm": 14.0,
  "min_charuco_corners": 6,
  "max_frames": 300,
  "stable_timeout_s": 10.0,
  "enable_debug": false
}
```

`GET /api/v1/hand-eye/config` 返回 `data.config` 和
`data.available_dictionary_names`。字典列表由服务运行时的
`cv2.aruco` 预定义字典动态生成，更新时 `dictionary_name` 必须来自该列表；其它字段可在
`PATCH` body 中按需提供。配置只在当前服务进程内生效，服务重启后恢复上述默认值。配置更新
不会连接设备或触发拍摄；拍摄任务执行期间更新会返回 `409`。

`start/end` body 使用：

```json
{"calibration_kind":"left_eye_in_hand","arm_side":"left"}
```

或：

```json
{"calibration_kind":"head_eye_to_hand","arm_side":"right"}
```

每次 sample 响应包含样本序号、相机名、marker/ChArUco 数量和重投影误差；所有会产生文件的
prior/solve 响应都包含目标路径、`replacement_id`、确认状态和完整的 `data.result`，但不会替换正式文件。
头部 ChArUco 与手部三球先验的 `result` 是待确认目录中对应 JSON 的完整内容；手眼在手上和头部
眼在手外的 `result` 包含矩阵、`translation_m`、`translation_mm`、`rpy_deg`、样本数及求解指标。
待确认响应同时包含 `expires_at`，表示本地时间的确认截止时刻。
前端确认结果后，必须使用 `POST /api/v1/replacements/confirm` 并提交
`{"replacement_id":"...","confirmed":true}`。取消时调用 `POST /api/v1/cancel`；取消只
清理缓存，正式文件保持不变。确认替换时，旧文件会按
`文件名_yymmdd_hhmmss.扩展名` 重命名保留，不执行删除。样本和待确认结果只保存在服务内存/临时
缓存中；生成结果后 30 秒内未确认或取消时，服务自动清理待确认缓存，正式文件保持不变。
服务重启后需要重新 start 和采样。`solve` 也可在 body 中提供显式 `samples` 数组，以便离线复算。
