# CameraPipeline API Reference

## 公共入口与通用约定

用户统一从 `camera_pipeline.client` 导入 `CameraPipelineClient`：

```python
from camera_pipeline.client import CameraPipelineClient

client = CameraPipelineClient(service_addr="tcp://127.0.0.1:6200")
try:
    intrinsics = client.get_head_camera_intrinsics()
finally:
    client.close()
```

当前协议版本为 `3`。RPC 失败统一写入
`CameraPipelineServiceResponse.error`，client 将其转换为 `RuntimeError`。
网络超时、服务未启动、协议版本不匹配、相机未连接或目标 payload 缺失都不作为成功结果返回。

支持的逻辑相机名与上游端口为：

| 安装位 | `camera_name` | 上游端口 | 当前状态 |
| --- | --- | ---: | --- |
| 头部 | `head_camera` | 5560 | 已连接 |
| 胸腔 | `chest_camera` | 5561 | 已连接 |
| 左臂 | `left_hand_camera` | 5562 | 已连接，默认算法相机 |
| 右臂 | `right_hand_camera` | 5563 | API 已保留，当前未连接 |

## 生命周期 API

### `CameraPipelineClient(service_addr="tcp://127.0.0.1:6200", timeout_ms=30000)`

创建 RPC client。`service_addr` 是统一 REQ/REP 服务地址；`timeout_ms` 同时用于 RPC 发送与接收超时，单位 ms。

### `close() -> None`

关闭 RPC socket。帧订阅返回独立迭代器，调用方结束订阅时也应关闭该迭代器。

## 相机查询 API

| API | 返回类型 | 成功语义 | 失败条件 |
| --- | --- | --- | --- |
| `get_camera_summary(timeout_s=10.0)` | `CameraSummaryResponse` | 返回默认左臂相机的最新帧摘要 | 首帧未就绪、帧字段非法 |
| `get_camera_intrinsics(camera_name="left_hand_camera", timeout_s=10.0)` | `CameraIntrinsicsResponse` | 按参数读取指定相机内参 | 相机未就绪、未连接或内参查询失败 |
| `get_head_camera_intrinsics(timeout_s=10.0)` | `CameraIntrinsicsResponse` | 读取头部相机内参 | 同上 |
| `get_chest_camera_intrinsics(timeout_s=10.0)` | `CameraIntrinsicsResponse` | 读取胸腔相机内参 | 同上 |
| `get_left_arm_camera_intrinsics(timeout_s=10.0)` | `CameraIntrinsicsResponse` | 读取左臂相机内参 | 同上 |
| `get_right_arm_camera_intrinsics(timeout_s=10.0)` | `CameraIntrinsicsResponse` | 保留的右臂内参入口 | 当前固定报右臂未连接 |
| `get_camera_status(timeout_s=10.0)` | `CameraStatusResponse` | 返回默认左臂相机状态 | 首帧不可用、相机流异常 |
| `get_stable_frame(camera_name="left_hand_camera", timeout_s=10.0)` | `StableFrameResponse` | 按参数返回指定相机稳定窗口中点帧 | 相机未连接、稳定等待超时、目标帧已淘汰 |
| `get_head_camera_stable_frame(timeout_s=10.0)` | `StableFrameResponse` | 返回头部相机稳定帧 | 同上 |
| `get_chest_camera_stable_frame(timeout_s=10.0)` | `StableFrameResponse` | 返回胸腔相机稳定帧 | 同上 |
| `get_left_arm_camera_stable_frame(timeout_s=10.0)` | `StableFrameResponse` | 返回左臂相机稳定帧 | 同上 |
| `get_right_arm_camera_stable_frame(timeout_s=10.0)` | `StableFrameResponse` | 保留的右臂稳定帧入口 | 当前固定报右臂未连接 |

### 相机查询响应字段

- `CameraSummaryResponse`：`frame_id`、`camera_name`、`timestamp_ms`、
  `color_shape`、`depth_shape`、`fx`、`fy`、`cx`、`cy`。
- `CameraIntrinsicsResponse`：`camera_name`、`fx`、`fy`、`cx`、`cy`、
  `distortion`、`width`、`height`。焦距和主点单位为像素。
- `CameraStatusResponse`：`camera_name`、`camera_id`、`camera_model`、
  `width`、`height`、`color_enabled`、`depth_enabled`、`online`。
- `StableFrameResponse`：`frame_id`、`camera_name`、`timestamp_ms`。

## 参数化帧订阅 API

参数化 API 主要供通用工具和测试使用：

| API | 迭代元素 | 发布端口 |
| --- | --- | ---: |
| `subscribe_camera_frames(camera_name="left_hand_camera")` | `CameraFramePacket` | 6201 |
| `subscribe_camera_color_frames(camera_name="left_hand_camera")` | `CameraColorFramePacket` | 6202 |
| `subscribe_camera_depth_frames(camera_name="left_hand_camera")` | `CameraDepthFramePacket` | 6203 |

这三个方法先通过 RPC 获取发布地址，再按 `camera_name + NUL` topic
订阅指定相机。合法订阅不存在“空结果”；首帧等待或 socket 超时会抛异常。
服务端通过 XPUB 跟踪实际订阅与取消订阅事件，只为当前存在订阅者的
相机和帧类型编码数据；所有订阅结束后不会继续占用 CPU 编码无人接收的帧。

## 明确安装位帧订阅 API

下列 API 是业务代码的主要入口，避免在调用点传入无语义字符串。

| 安装位 | RGBD | 彩色 | 深度 |
| --- | --- | --- | --- |
| 头部 | `subscribe_head_camera_frames()` | `subscribe_head_camera_color_frames()` | `subscribe_head_camera_depth_frames()` |
| 胸腔 | `subscribe_chest_camera_frames()` | `subscribe_chest_camera_color_frames()` | `subscribe_chest_camera_depth_frames()` |
| 左臂 | `subscribe_left_arm_camera_frames()` | `subscribe_left_arm_camera_color_frames()` | `subscribe_left_arm_camera_depth_frames()` |
| 右臂 | `subscribe_right_arm_camera_frames()` | `subscribe_right_arm_camera_color_frames()` | `subscribe_right_arm_camera_depth_frames()` |

右臂方法保留公开签名，但当前在首次迭代时抛出“configured but not connected”服务错误。

### 帧数据包字段

- `CameraFramePacket`：`frame_id`、`camera_name`、`timestamp_ms`、
  `color_bgr`、`depth_mm`、`fx`、`fy`、`cx`、`cy`。
- `CameraColorFramePacket`：`frame_id`、`camera_name`、`timestamp_ms`、
  `color_bgr`、`fx`、`fy`、`cx`、`cy`。
- `CameraDepthFramePacket`：`frame_id`、`camera_name`、`timestamp_ms`、
  `depth_mm`、`fx`、`fy`、`cx`、`cy`。

`color_bgr` 形状为 `(H, W, 3)`、dtype 为 `uint8`、通道顺序 BGR；
`depth_mm` 形状为 `(H, W)`、dtype 为 `uint16`、单位 mm，零值表示无效深度。

## 算法请求 API

### `request_tray_detection(request) -> OrinTrayDetectionResponse`

请求类型为 `OrinTrayDetectionRequest`，主要字段为 `request_id`、
`camera_name`、`frame_id`、`enable_debug`。成功响应包含实际帧号、相机名、
时间戳、耗时、`tray_count`、`tray_results`和 `debug_artifacts`。
未检测到托盘是合法空结果：`tray_count=0`、`tray_results=()`。
关闭 debug 时 `debug_artifacts=()`。帧不可回取、模型加载/推理失败或协议非法会抛出 `RuntimeError`。

### `request_opening_detection(request) -> OpeningDetectionPipelineResponse`

请求类型为 `OpeningDetectionPipelineRequest`，主要字段为 `request_id`、
`camera_name`、`frame_id`、`target_tray_index`、`enable_debug`。成功响应包含
托盘列表、选中索引、`selected_result`、`all_tray_results` 和可选 debug 产物。
该 API 没有合法空结果；目标托盘不存在、mask 无效、深度点不足或位姿无法计算均为失败。

### `request_ball_pose_detection(request) -> BallPoseDetectionResponse`

请求类型为 `BallPoseDetectionRequest`，主要字段为 `request_id`、
`camera_name`、`frame_id`、`enable_debug`、`priors`。每个 `BallPosePriorInfo` 包含
`color_hex`、`radius_mm`、`model_center_mm`。成功响应包含 `matched_count`、
`detections` 和可选 debug 产物。无先验时 `detections=()`；单球漏检时对应
`BallDetectionInfo.detected=False`，坐标元组为空。

## 统一响应边界

- 算法成功响应只描述算法结果，不重复定义服务级 `error` 字段。
- “没有检测到目标”只有在对应算法明确允许时才是成功空结果。
- `debug_artifacts=()` 表示调用方关闭 debug 或本次未生成调试载荷，不表示算法失败。
- `frame_id` 是实际计算帧号。使用稳定帧时，响应帧号是最终证据帧，不是请求发出时的最新帧。
