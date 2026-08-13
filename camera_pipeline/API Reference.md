# CameraPipeline API Reference

文档版本：`1.0.4`（2026-08-03）
CameraPipeline 功能版本：`1.11.1`
内部 ZMQ 线协议：`CPW1` / `PROTOCOL_VERSION=10`
外部 WebSocket 帧协议：`CPWS1` / `protocol_version=1`

机器可读文件：

- [OpenAPI 3.1 HTTP API](openapi.yaml)
- [AsyncAPI 3.0 WebSocket API](asyncapi.yaml)

## 1. 传输边界

| 用途 | 协议 | 默认地址 | 调用方 |
| --- | --- | --- | --- |
| 服务内部 RPC | ZMQ REQ/REP + `CPW1` | `tcp://<orin>:6200` | 内部 Python client、RecordReplay |
| 服务内部帧流 | ZMQ XPUB + `CPW1` | `6201/6202/6203` | 兼容旧内部调用方 |
| 服务 HTTP 后端 | HTTP/JSON | `http://<orin>:6400` | Gateway、人工测试与诊断 |
| 服务 WebSocket 后端 | WebSocket + `CPWS1` | `ws://<orin>:6401` | Gateway、人工测试与诊断 |

HTTP/WebSocket 适配层只桥接现有 `CameraPipelineApplication` 和 `PipelineContext`，不复制
算法，也不改变内部 ZMQ 协议。GUI 不需要实现 `CPW1`。

正式客户端访问必须经过统一 API Gateway：HTTP 使用
`https://<orin-host>/api/v1/camera/*`，图像 WebSocket 使用
`wss://<orin-host>/api/v1/camera-ws/*`。客户端首次使用前必须安装并信任 CasiaHand Root CA；
不得关闭证书校验。`6400`、`6401` 以及内部 ZMQ 端口只用于服务内部
联调、人工测试和故障诊断；不得作为 GUI 或其它正式客户端的默认访问地址。Gateway 只做
转发，不合并 CameraPipeline 进程。

URL 统一使用小写和短横线；路径参数使用简短资源名；JSON body 字段继续使用既有
`snake_case`，避免把 URL 命名规则和数据字段规则混在一起。

## 2. HTTP 通用约定

成功响应统一为：

```json
{
  "ok": true,
  "service_version": "1.11.1",
  "zmq_protocol_version": 10,
  "data": {},
  "error": null
}
```

错误响应统一为：

```json
{
  "ok": false,
  "service_version": "1.11.1",
  "zmq_protocol_version": 10,
  "data": null,
  "error": {"code": "invalid_request", "message": "ValueError: ..."}
}
```

| HTTP 状态码 | code | 语义 |
| ---: | --- | --- |
| `400` | `invalid_request` | JSON、路径、类型、数值或参数非法 |
| `404` | `not_found` | 路径或相机枚举不支持 |
| `503` | `service_error` | 相机不可用、内部业务异常或 payload 缺失 |
| `504` | `timeout` | 等待相机帧或稳定帧超时 |

“未检测到目标”不是 HTTP 错误；算法允许空结果时仍返回 `200`，由 `status` 或
`detected` 表示。

支持的 `camera_name`：`head_camera`、`chest_camera`、`left_hand_camera`、
`right_hand_camera`。相机清单接口只返回当前已配置、已连接且已有最新帧的相机；
未连接或未配置的安装位不会出现在清单中。

## 3. HTTP 接口

### 3.1 健康检查

```http
GET /api/v1/health
```

`data`：`{"service_version":"1.11.1","zmq_protocol_version":10}`。该接口只表示
HTTP 适配层可响应，不保证任意相机已有首帧；请使用相机状态接口判断相机可用性。

### 3.2 相机清单

```http
GET /api/v1/cameras
```

`data`：

```json
{
  "cameras": [
    {
      "service_version": "1.11.1",
      "camera_name": "left_hand_camera",
      "camera_id": "LEFT",
      "camera_model": "unknown",
      "width": 1280,
      "height": 720,
      "color_enabled": true,
      "depth_enabled": true,
      "online": true,
      "error": null
    }
  ]
}
```

数组只包含已配置、已连接且已有最新帧的相机；没有可用相机时返回空数组，不返回错误。

### 3.3 相机状态

```http
GET /api/v1/cameras/{camera}/status?timeout_s=10
```

`timeout_s` 可选，默认 `10.0` 秒，必须为正数。`data` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `service_version` | string | 远端功能版本 |
| `camera_name` | string | 实际逻辑相机名 |
| `camera_id` | string | 上游相机 ID/SN |
| `camera_model` | string | 当前无法取得时为 `unknown` |
| `width` / `height` | integer | 彩色帧尺寸，pixel |
| `color_enabled` / `depth_enabled` | boolean | 彩色/深度流开关状态 |
| `online` | boolean | 首帧是否可用 |
| `error` | null/string | 成功时为 `null`；保留的协议字段 |

没有首帧时不会返回伪造的在线状态，而是返回 `503`。

### 3.4 相机内参

```http
GET /api/v1/cameras/{camera}/intrinsics?timeout_s=10
```

`data`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `camera_name` | string | 逻辑相机名 |
| `fx` / `fy` | number | 焦距，pixel |
| `cx` / `cy` | number | 主点，pixel |
| `distortion` | number[8] | OpenCV `(k1,k2,p1,p2,k3,k4,k5,k6)` |
| `width` / `height` | integer | 彩色图尺寸，pixel |
| `error` | null/string | 成功时为 `null`；保留的协议字段 |

### 3.5 稳定帧

```http
POST /api/v1/cameras/{camera}/stable-frame
Content-Type: application/json

{"timeout_s": 10.0}
```

body 可以为空对象 `{}`，`timeout_s` 默认 `10.0` 秒。`data`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `frame_id` | integer | 稳定窗口中点的实际帧号 |
| `camera_name` | string | 实际相机名 |
| `timestamp_ms` | number | 采集时间戳，ms |
| `error` | null/string | 成功时为 `null`；保留的协议字段 |

稳定窗口未形成或帧没有递增时返回 `504` 或 `503`。

### 3.6 三球位姿检测

```http
POST /api/v1/detections/ball
Content-Type: application/json
```

请求：

```json
{
  "request_id": 1,
  "camera_name": "left_hand_camera",
  "frame_id": -1,
  "enable_debug": false,
  "priors": [
    {
      "color_hex": "#FFFF00",
      "diameter_mm": 50.0,
      "model_center_mm": [0.0, 0.0, 0.0],
      "hsv_ranges": [[20, 80, 80, 40, 255, 255]]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | integer | 调用方编号，原样返回 |
| `camera_name` | enum | 逻辑相机名 |
| `frame_id` | integer | 正数精确取缓存帧；非正数选择稳定帧 |
| `enable_debug` | boolean | 是否返回 debug 图像，默认 `false` |
| `priors` | array | 先验顺序决定结果顺序；为空时不检测球 |
| `priors[].color_hex` | string | 带 `#` 的六位 RGB 颜色身份 |
| `priors[].diameter_mm` | number | 球物理直径，mm |
| `priors[].model_center_mm` | number[3] | 参考坐标系 `(x,y,z)`，mm |
| `priors[].hsv_ranges` | array | 每项六整数 `(hmin,smin,vmin,hmax,smax,vmax)` |

响应 `data`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` / `frame_id` | integer | 请求编号 / 实际计算帧号 |
| `camera_name` | string | 实际相机名 |
| `timestamp_ms` | number | 实际帧时间戳，ms |
| `elapsed_ms` | number | 检测耗时，ms |
| `matched_count` | integer | 有效三维球数量 |
| `detections` | array | 与 `priors` 同顺序 |
| `debug_artifacts` | array | debug 开启并生成成功时通常一个元素 |

`detections[]`：`color_hex`、`detected`、`center_px`（`u,v` pixel）、`center_mm`
（相机坐标 `x,y,z` mm）、`diameter_mm`、`radius_px`、`center_norm`、`radius_norm`、
`point_count`、`status`（如 `detected`/`depth_weak`/`missing`）和 `observed_hsv`。
无效坐标字段为空数组。`enable_debug=false` 时 `debug_artifacts` 为空数组。

### 3.7 ChArUco 位姿检测

```http
POST /api/v1/detections/charuco
Content-Type: application/json
```

请求：

```json
{
  "camera_name": "head_camera",
  "dictionary_name": "DICT_APRILTAG_16H5",
  "squares_x": 4,
  "squares_y": 4,
  "square_length_mm": 20.0,
  "marker_length_mm": 14.0,
  "min_charuco_corners": 6,
  "max_frames": 5,
  "stable_timeout_s": 10.0,
  "enable_debug": false
}
```

`dictionary_name` 必须是当前 OpenCV `cv2.aruco` 暴露的预定义字典；横纵方格至少为 `2`；方格和 marker 边长必须满足
`0 < marker_length_mm < square_length_mm`；最少角点至少为 `4`；`max_frames` 和
`stable_timeout_s` 必须大于 `0`。服务端自行构造 Board、等待纯彩色稳定帧并执行 PnP。

响应 `data`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | `detected` 或 `missing` |
| `camera_name` | string | 实际相机名 |
| `t_cam_board_mm` | number[4][4] | `p_camera = T_camera_board @ p_board`，平移 mm |
| `error_px` | number/null | 平均重投影误差，pixel；请优先判断 `status` |
| `marker_num` / `charuco_num` | integer | marker / ChArUco 角点数量 |
| `overlay_bgr` | object | debug 图像；关闭 debug 时为空图像对象 |

JSON 不允许标准外的 `Infinity`。内部无结果的 `inf` 会以 JSON `null` 兼容返回，GUI
不得用 `error_px` 是否无穷判断成功，必须使用 `status`。

## 4. WebSocket 图像订阅

```text
ws://<orin>:6401/api/v1/ws/cameras/{camera}/color
ws://<orin>:6401/api/v1/ws/cameras/{camera}/depth
ws://<orin>:6401/api/v1/ws/cameras/{camera}/rgbd
```

服务端只发送 binary message，不发送 Base64。服务重启、断线或无首帧时，客户端应重新
连接。每个连接只保留最新帧，慢客户端会跳过中间帧。

### 4.1 CPWS1 消息格式

```text
offset  size  meaning
0       5     ASCII magic: CPWS1
5       4     unsigned big-endian metadata_length
9       N     UTF-8 JSON metadata
9+N     ...   binary array area
```

示例 metadata：

```json
{
  "protocol": "camera_pipeline.websocket",
  "protocol_version": 1,
  "packet_type": "color_frame",
  "fields": {
    "frame_id": 123,
    "camera_name": "head_camera",
    "timestamp_ms": 1720000000000.0,
    "fx": 600.0,
    "fy": 600.0,
    "cx": 320.0,
    "cy": 240.0,
    "distortion": [0, 0, 0, 0, 0, 0, 0, 0],
    "color_bgr": {"encoding":"raw","dtype":"|u1","shape":[480,640,3],"offset":0,"nbytes":921600}
  }
}
```

`offset`/`nbytes` 相对于 metadata 后的 binary area；数组为 C contiguous。流字段：

| URL 尾段 | packet_type | 数组 |
| --- | --- | --- |
| `color` | `color_frame` | `color_bgr`: `uint8`、`(H,W,3)`、BGR |
| `depth` | `depth_frame` | `depth_mm`: `uint16`、`(H,W)`、mm、0 无效 |
| `rgbd` | `rgbd_frame` | 上述两个数组 |

所有流都包含 `frame_id`、`camera_name`、`timestamp_ms`、`fx`、`fy`、`cx`、`cy` 和
`distortion`。

## 5. 内部 ZMQ 兼容接口

RecordReplay 和同一 Python 部署包内的旧调用方继续使用：

```python
from camera_pipeline.client import CameraName, CameraPipelineClient

client = CameraPipelineClient("tcp://127.0.0.1:6200")
try:
    status = client.get_camera_status(CameraName.HEAD)
finally:
    client.close()
```

内部 operation 为：`camera_summary`、`camera_intrinsics`、`camera_status`、`stable_frame`、
`camera_frame_subscribe`、`camera_color_frame_subscribe`、`camera_depth_frame_subscribe`、
`detect_ball`、`detect_charuco`。`tray_detection` 和 `opening_detection` 当前不属于协议。

## 6. 文档变更记录

| 文档版本 | 日期 | 内容 |
| --- | --- | --- |
| `1.0.4` | 2026-08-03 | 正式 Gateway 入口改为 HTTPS/WSS 443，并要求客户端安装 CasiaHand CA |
| `1.0.3` | 2026-08-03 | 明确正式客户端必须通过 API Gateway，独立端口仅用于测试和诊断 |
| `1.0.2` | 2026-07-31 | 增加只返回当前可用相机的 `GET /api/v1/cameras` 清单接口 |
| `1.0.1` | 2026-07-31 | 统一 URL、OpenAPI operationId 和 AsyncAPI 标识命名 |
| `1.0.0` | 2026-07-31 | 新增 HTTP/JSON、CPWS1 WebSocket、OpenAPI 和 AsyncAPI 说明 |

服务功能版本见 `camera_pipeline/CHANGELOG.md`；当前日志轮转优化使功能版本升级为
`1.11.1`，内部 ZMQ `PROTOCOL_VERSION=10` 不变。
