# Calibration Service

`calibration_service` 是 RecordReplay 的先验准备服务，默认监听 Orin 本机
`http://127.0.0.1:6600`，正式客户端通过 Gateway 的
`/api/v1/calibration` 前缀访问。

服务只做三类工作：读取 RobotControl 的 AR5 状态、按需调用 CameraPipeline 拍摄和检测、
执行离线计算并保存结果。它不导入 qmlinker 或 xCoreSDK，也不发送任何设备控制 POST；
机械臂、头部和其它设备必须由 RobotControl 或现场人工操作接口负责。

服务同时写入 journald 控制台日志和独立的 `logs/calibration_service.log`。文件日志每小时
轮转、ZIP 压缩并保留 7 天；可用 `--log-path` 覆盖路径，不与其它服务合并存储。

systemd 单元通过 `Requires`、`After` 和 `PartOf` 关联 CameraPipeline 与 RobotControl：依赖
服务执行 `restart` 时会同步重启 Calibration Service。官方单服务部署脚本在依赖服务就绪后
还会显式恢复 Calibration Service，并通过本机 `GET /api/v1/status` 确认状态为 `idle`。

结果固定写入同级 RecordReplay 服务目录：

- `record_replay/prior_data/charuco_board_prior.json`：`POST /api/v1/prior/head`；
- `record_replay/prior_data/ball_pose_prior.json`：`POST /api/v1/prior/hand`；
- `record_replay/prior_data/hand_eye_result.txt`：`POST /api/v1/hand-eye/solve`。
- `record_replay/prior_data/left_head_base_camera.npy`：`POST /api/v1/head-eye/solve`，
  `arm_side=left`；右侧同理写入 `right_head_base_camera.npy`。

头部和手部先验分别使用两个接口。手部先验当前按 RecordReplay 语义固定使用左手
相机和左臂状态。手眼样本保存在服务内存中，服务重启后需重新调用
`POST /api/v1/hand-eye/sample`；也可以在 solve body 中直接提供样本数组。
手眼 ChArUco 板参数通过 `GET/PATCH /api/v1/hand-eye/config` 查询和修改，字典选项与当前
`cv2.aruco` 可用预定义字典一致。配置只在当前服务进程内生效，重启后恢复默认值。

## 手动流程

1. 由现场人员通过 RobotControl 接口或其它安全操作方式把设备放到拍摄姿态。
2. 调用 `POST /api/v1/prior/head` 或 `POST /api/v1/prior/hand`；接口只生成待确认缓存，
   不会立即替换 RecordReplay 正式文件。响应包含 `replacement_id`、`expires_at` 和完整的
   `data.result`；头部与手部先验的 `result` 就是临时目录中对应 JSON 的完整内容；
   必须在结果生成后的 30 秒内确认或取消，超时后服务自动丢弃待确认结果。
3. 左手眼在手上标定时，调用 `POST /api/v1/start`（经 Gateway 为
   `/api/v1/calibration/start`），
   `calibration_kind=left_eye_in_hand`；现场人员每次调整左臂姿态后调用
   `POST /api/v1/hand-eye/sample`，至少采集三组。
4. 调用 `POST /api/v1/end`（经 Gateway 为 `/api/v1/calibration/end`），
   再调用 `POST /api/v1/hand-eye/solve`。solve 只生成待确认缓存；前端展示结果后，
   调用 `POST /api/v1/replacements/confirm` 才会保存 RecordReplay 可读取的 `T_tool_cam`。
   取消时调用 `POST /api/v1/cancel`，结果通过 `GET /api/v1/results/hand-eye` 读取。
5. 头部眼在手外标定使用 `calibration_kind=head_eye_to_hand`，采样接口改为
   `POST /api/v1/head-eye/sample`，求解接口为 `POST /api/v1/head-eye/solve`；同样需要
   `POST /api/v1/replacements/confirm` 二次确认，结果通过
   `GET /api/v1/results/head-eye/{arm_side}` 读取。

修改手眼采样默认板参数示例：

```http
PATCH /api/v1/hand-eye/config
Content-Type: application/json

{"dictionary_name":"DICT_5X5_1000","squares_x":12,"squares_y":9}
```

先读取配置响应中的 `available_dictionary_names`，再选择字典；服务不会接受当前
OpenCV 未提供的字典名称。

只读 `GET /api/v1/health` 返回 `service_version` 和当前任务状态，不访问设备；正式客户端可用
该字段进行版本校验。标定会话的 `start/end` 只用于提示、清空和封存本次内存样本，不会隐式移动设备、拍摄或
求解。`cancel` 会丢弃样本和待确认缓存，不执行替换。正式替换时，旧文件会按
`文件名_yymmdd_hhmmss.扩展名` 重命名保留，不会直接删除。`GET /api/v1/results/prior/head`
与 `/results/prior/hand` 分别读取两个已确认的先验 JSON。

所有待确认结果的确认窗口均为 30 秒。超时后临时结果和 `replacement_id` 失效，
RecordReplay 正式文件保持不变；`GET /api/v1/status` 的 `pending_replacement` 会变为 `null`。

待确认响应统一保留原有的 `calibration_kind`、`result_path`、`message`、状态和确认字段，
并增加未写入正式文件的 `result`。例如头部先验为：

```json
{
  "data": {
    "calibration_kind": "head",
    "result": {
      "marker_count": 7,
      "charuco_count": 7,
      "reprojection_error_px": 0.12,
      "camera_board_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "translation_mm": [0, 0, 0],
      "rpy_deg": [0, 0, 0]
    },
    "replacement_id": "...",
    "requires_confirmation": true
  }
}
```

这些接口只触发拍摄和计算，不替代设备控制接口。重启服务不会自动拍摄、计算或启动
RecordReplay 回放。
