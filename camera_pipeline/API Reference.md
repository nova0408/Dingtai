# CameraPipeline API Reference

## 公共入口与通用约定

用户统一从 `camera_pipeline.client` 导入 `CameraPipelineClient`：

```python
from camera_pipeline.client import CameraName, CameraPipelineClient

client = CameraPipelineClient(service_addr="tcp://127.0.0.1:6200")
try:
    intrinsics = client.get_camera_intrinsics(CameraName.HEAD)
finally:
    client.close()
```

当前协议版本为 `9`。RPC 失败统一写入
`CameraPipelineServiceResponse.error`，client 将其转换为 `RuntimeError`。
网络超时、服务未启动、协议版本不匹配、相机未连接或目标 payload 缺失都不作为成功结果返回。

支持的逻辑相机名与上游端口为：

| 安装位 | 枚举 | 协议值 | 上游端口 |
| --- | --- | --- | ---: |
| 头部 | `CameraName.HEAD` | `head_camera` | 5560 |
| 胸腔 | `CameraName.CHEST` | `chest_camera` | 5561 |
| 左臂 | `CameraName.LEFT_ARM` | `left_hand_camera` | 5562 |
| 右臂 | `CameraName.RIGHT_ARM` | `right_hand_camera` | 5563 |

## 生命周期 API

### `CameraPipelineClient(service_addr="tcp://127.0.0.1:6200", timeout_ms=30000)`

创建 RPC client。`service_addr` 是统一 REQ/REP 服务地址；`timeout_ms` 同时用于 RPC 发送与接收超时，单位 ms。

### `close() -> None`

关闭 RPC socket。帧订阅返回独立迭代器，调用方结束订阅时也应关闭该迭代器。

## 相机查询 API

| API | 返回类型 | 成功语义 | 失败条件 |
| --- | --- | --- | --- |
| `get_camera_summary(camera_name, timeout_s=10.0)` | `CameraSummaryResponse` | 返回指定相机的最新帧摘要 | 首帧未就绪、帧字段非法 |
| `get_camera_intrinsics(camera_name, timeout_s=10.0)` | `CameraIntrinsicsResponse` | 读取指定相机内参 | 相机未就绪、未连接或内参查询失败 |
| `get_camera_status(camera_name, timeout_s=10.0)` | `CameraStatusResponse` | 返回指定相机状态 | 首帧不可用、相机流异常 |
| `get_stable_frame(camera_name, timeout_s=10.0)` | `StableFrameResponse` | 返回指定相机稳定窗口中点帧 | 相机未连接、稳定等待超时、目标帧已淘汰 |

### 相机查询响应字段

- `CameraSummaryResponse`：`frame_id`、`camera_name`、`timestamp_ms`、
  `color_shape`、`depth_shape`、`fx`、`fy`、`cx`、`cy`。
- `CameraIntrinsicsResponse`：`camera_name`、`fx`、`fy`、`cx`、`cy`、
  `distortion`、`width`、`height`。焦距和主点单位为像素。
- `CameraStatusResponse`：`service_version`、`camera_name`、`camera_id`、`camera_model`、
  `width`、`height`、`color_enabled`、`depth_enabled`、`online`。
- `StableFrameResponse`：`frame_id`、`camera_name`、`timestamp_ms`。

## 参数化帧订阅 API

参数化 API 主要供通用工具和测试使用：

| API | 迭代元素 | 发布端口 |
| --- | --- | ---: |
| `subscribe_camera_frames(camera_name)` | `CameraFramePacket` | 6201 |
| `subscribe_camera_color_frames(camera_name)` | `CameraColorFramePacket` | 6202 |
| `subscribe_camera_depth_frames(camera_name)` | `CameraDepthFramePacket` | 6203 |

这三个方法先通过 RPC 获取发布地址，再按 `camera_name + NUL` topic
订阅指定相机。合法订阅不存在“空结果”；首帧等待或 socket 超时会抛异常。
服务端通过 XPUB 跟踪实际订阅与取消订阅事件，只为当前存在订阅者的
相机和帧类型编码数据；所有订阅结束后不会继续占用 CPU 编码无人接收的帧。

### 帧数据包字段

- `CameraFramePacket`：`frame_id`、`camera_name`、`timestamp_ms`、
  `color_bgr`、`depth_mm`、`fx`、`fy`、`cx`、`cy`、`distortion`。
- `CameraColorFramePacket`：`frame_id`、`camera_name`、`timestamp_ms`、
  `color_bgr`、`fx`、`fy`、`cx`、`cy`、`distortion`。
- `CameraDepthFramePacket`：`frame_id`、`camera_name`、`timestamp_ms`、
  `depth_mm`、`fx`、`fy`、`cx`、`cy`、`distortion`。

`color_bgr` 形状为 `(H, W, 3)`、dtype 为 `uint8`、通道顺序 BGR；
`depth_mm` 形状为 `(H, W)`、dtype 为 `uint16`、单位 mm，零值表示无效深度。
`distortion` 是彩色相机 OpenCV 8 参数畸变系数元组，固定按
`(k1, k2, p1, p2, k3, k4, k5, k6)` 排列。上游 ZMQ 的 `data["dist"]`
使用 Orbbec SDK 顺序 `(k1, k2, k3, k4, k5, k6, p1, p2)`；该差异已在
`CameraStreamRuntime` 的控制协议边界完成转换，不向算法和 RPC 客户端传播。

## ChArUco Board 检测客户端 API

### `CameraPipelineClient.detect_charuco(request) -> CharucoDetectionResponse`

用户必须显式传入完整 Board 几何和检测边界。CameraPipeline 不提供默认板型：

```python
from camera_pipeline.client import CameraName
from camera_pipeline.service.protocol import CharucoDetectionRequest

request = CharucoDetectionRequest(
    camera_name=CameraName.HEAD,
    dictionary_name="DICT_APRILTAG_16H5",
    squares_x=4,
    squares_y=4,
    square_length_mm=20.0,
    marker_length_mm=14.0,
    min_charuco_corners=6,
    max_frames=300,
    stable_timeout_s=10.0,
    enable_debug=True,
)
result = client.detect_charuco(request)
```

协议中的所有参数均为必填：

- `camera_name`：逻辑相机名。
- `dictionary_name`：Board 使用的 ArUco 字典；当前支持 `DICT_APRILTAG_16H5`。
- `squares_x`、`squares_y`：横向、纵向方格数量，均至少为 2。
- `square_length_mm`：方格边长，单位 mm，必须大于 0。
- `marker_length_mm`：marker 边长，单位 mm，必须大于 0 且小于方格边长。
- `min_charuco_corners`：进入 PnP 的最少角点数，至少为 4。
- `max_frames`：最多尝试的纯彩色稳定帧数量，必须大于 0。
- `stable_timeout_s`：每次等待纯彩色稳定帧的超时，单位 s，必须大于 0；
  ChArUco 不读取深度帧，也不执行深度稳定阈值。
- `enable_debug`：是否返回最终检测帧的 marker、角点和坐标轴 overlay。

服务端只校验协议、构造本次请求的
`cv2.aruco.CharucoBoard`，然后由 `PipelineContext.detect_charuco()` 获取纯彩色
稳定帧并调用
`camera_pipeline.charuco_detection.CharucoDetector`；服务端不保存任何固定板型。

返回 `CharucoDetectionResponse`：

- `status`：`detected` 或 `missing`。
- `camera_name`：实际检测相机名。
- `t_cam_board_mm`：`T_camera_board`；满足 `p_camera = T_camera_board @ p_board`，
  平移单位 mm。空结果时为空元组。
- `error_px`：平均重投影误差，单位 pixel；空结果时为正无穷。
- `marker_num`、`charuco_num`：最终融合的 marker 和 ChArUco 角点数量。
- `overlay_bgr`：`enable_debug=True` 时返回最终检测帧的 BGR 叠加图；关闭时为空数组。

相机不可用、稳定帧超时、板参数非法或服务协议错误由客户端转换为 `RuntimeError`。
达到 `max_frames` 仍未得到位姿属于合法空结果，返回 `status="missing"`。

## 算法请求 API

`request_tray_detection()` 和 `request_opening_detection()` 已暂时移除。协议版本 8
不再接受 `tray_detection`、`opening_detection` operation，也不再暴露对应的请求、
响应字段或 wire codec 类型。调用方不得继续构造旧版 operation；需要恢复时应同步
恢复子模块入口、服务编排、协议、codec、client API 和测试后再提升协议版本。

### `detect_ball(request) -> BallPoseDetectionResponse`

请求类型为 `BallPoseDetectionRequest`，主要字段为 `request_id`、
`camera_name`、`frame_id`、`enable_debug`、`priors`。每个 `BallPosePriorInfo` 包含
`color_hex`、`diameter_mm`、`model_center_mm`、`hsv_ranges`。`color_hex` 是稳定
身份和首次记录参考色；非空 `hsv_ranges` 是记录先验得到的专属窄范围，优先于参考色
宽范围。`diameter_mm` 表示球的物理直径，单位 mm。成功响应包含 `matched_count`、
`detections` 和可选 debug 产物。无先验时 `detections=()`；单球漏检时对应
`BallDetectionInfo.detected=False`，坐标元组为空；有效结果的 `observed_hsv` 是候选
内部颜色像素的实测 HSV 中心。

## 统一响应边界

- 算法成功响应只描述算法结果，不重复定义服务级 `error` 字段。
- “没有检测到目标”只有在对应算法明确允许时才是成功空结果。
- `debug_artifacts=()` 表示调用方关闭 debug 或本次未生成调试载荷，不表示算法失败。
- `frame_id` 是实际计算帧号。使用稳定帧时，响应帧号是最终证据帧，不是请求发出时的最新帧。
