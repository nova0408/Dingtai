# CameraPipeline Service

## 单一职责

`camera_pipeline.service` 是 CameraPipeline 唯一的网络服务子模块，负责把协议请求桥接到 `PipelineContext` 和纯计算算法。算法目录不创建 socket、不监听端口，也不管理服务进程。

## 模块结构

| 文件 | 职责 |
| --- | --- |
| `config.py` | 服务监听、帧发布地址和请求循环超时配置 |
| `protocol.py` | 统一请求、响应、operation 和协议版本 |
| `wire_codec.py` | 白名单协议对象、JSON 元数据和 NumPy 原始字节块编解码 |
| `transport.py` | ZMQ REQ/REP 收发与消息类型校验 |
| `client.py` | 外接开发机和 Orin 本地业务服务共用的客户端实现 |
| `frame_publisher.py` | 单线程发布 RGBD、彩色、深度最新帧 |
| `application.py` | 相机请求和 tray/opening/ball 业务调用编排 |
| `server.py` | 通用 REP 请求循环和统一异常边界 |
| `__main__.py` | 参数解析、信号处理、对象组装和资源释放 |

## 数据流

```text
CameraPipelineClient
  -> REQ/REP transport
  -> CameraPipelineServer
  -> CameraPipelineApplication
  -> PipelineContext.resolve_frame()
  -> pure algorithm service.compute()
  -> protocol response
```

opening 请求属于业务组合流程：application 使用同一个稳定帧先执行 tray，取得目标托盘 mask 后执行 opening。该组合不进入算法实现，也不进入网络 transport。

## 部署拓扑

服务端默认绑定：

```text
tcp://0.0.0.0:6200
```

Orin 本地业务服务使用公共入口：

```python
from camera_pipeline.client import CameraPipelineClient

client = CameraPipelineClient()  # 默认 tcp://127.0.0.1:6200
```

外接开发机显式指定 Orin 地址：

```python
client = CameraPipelineClient(service_addr="tcp://<orin-ip>:6200")
```

启动命令：

```bash
python -m camera_pipeline.service \
  --bind-addr tcp://0.0.0.0:6200 \
  --control-port 5570 \
  --stream-port 5562 \
  --camera-id LEFT \
  --camera-name left_hand_camera
```

systemd 模板位于 `camera-pipeline.service`。Orin 本地其他业务服务应声明 `Requires=camera-pipeline.service` 和 `After=camera-pipeline.service`，并在启动后调用 `get_camera_status()` 判断首帧是否真正可用。

## 端口

- `6200`：统一请求响应服务。
- `6201`：完整 RGBD 帧。
- `6202`：彩色帧。
- `6203`：深度帧。

帧订阅响应返回服务端 bind 地址；客户端会将 `0.0.0.0` 或 `127.0.0.1` 替换为当前服务主机，因此同时支持外接开发机和 Orin loopback。

## 生命周期

`__main__.py` 按以下顺序释放资源：

1. 停止并关闭帧发布器。
2. 关闭 REP transport。
3. 关闭 `PipelineContext` 和相机运行时。

发布端口只在首次订阅时绑定。未订阅帧流时不会创建 PUB socket 或发布线程。

## 协议与安全边界

请求包含 `protocol_version`，当前版本为 `1`。客户端与服务端版本不一致时服务端明确拒绝。

REQ/REP 和三路帧流使用同一显式二进制协议：固定头部携带 JSON 元数据长度，
元数据只允许白名单中的协议 dataclass、基础类型和元组，NumPy 图像与 mask 以
连续原始字节块附加，并在元数据中记录 `dtype`、`shape`、偏移和长度。协议不使用
Python pickle，不依赖 dataclass 的 Python 版本内存布局，可在 Windows Python
3.10+ 客户端与 Orin Python 3.8 服务之间互通。未知类型、未知协议标识、越界数组
和字段不匹配均会明确拒绝。

## 错误约定

请求解码失败和业务异常均由 `CameraPipelineServer` 捕获，写入统一响应的 `error`
字段；单个非法请求不会终止服务循环。客户端发现顶层错误或目标 payload 缺失时
抛出 `RuntimeError`。算法模块不负责网络错误转换。

## 用户请求与可能响应

用户只需要调用 `camera_pipeline.client.CameraPipelineClient`；下面的表描述每个公开请求的成功结果、业务空结果和失败行为。所有请求都可能额外出现网络超时、服务未启动或协议版本不匹配，此时客户端抛出 `RuntimeError`。

| 客户端方法 | 成功响应 | 合法的业务空结果 | 服务错误条件 |
| --- | --- | --- | --- |
| `get_camera_summary()` | `CameraSummaryResponse`，包含首帧形状、内参和帧号 | 无；没有首帧视为服务错误 | 相机未启动、首帧超时、帧字段非法 |
| `get_camera_intrinsics()` | `CameraIntrinsicsResponse` | 无；内参不可用视为服务错误 | 相机未就绪、控制查询失败 |
| `get_camera_status()` | `CameraStatusResponse`，`online=True` 表示服务已取得有效帧 | 不在线状态不会作为成功响应返回 | 首帧不可用、相机流异常 |
| `get_stable_frame()` | `StableFrameResponse`，包含稳定窗口中点附近的正数 `frame_id` | 稳定窗口未形成不会返回响应，直到超时后报错 | 稳定等待超时、目标帧已被缓存淘汰 |
| `subscribe_frames()` | `CameraFrameSubscribeResponse`，包含可连接 PUB 地址 | 无 | 发布器启动或端口绑定失败 |
| `subscribe_color_frames()` | `CameraColorFrameSubscribeResponse` | 无 | 发布器启动或端口绑定失败 |
| `subscribe_depth_frames()` | `CameraDepthFrameSubscribeResponse` | 无 | 发布器启动或端口绑定失败 |
| `request_tray_detection()` | `OrinTrayDetectionResponse` | 未检测到托盘时成功返回 `tray_count=0`、`tray_results=()`；关闭 debug 时 `debug_artifacts=()` | 帧不可回取、模型加载/推理失败、输入协议非法 |
| `request_opening_detection()` | `OpeningDetectionPipelineResponse`，包含完整 `TrayPoseInfo` | 无；目标托盘不存在、mask 无效或抓取位姿无法计算均为错误 | tray 依赖失败、深度点不足、平面/位姿计算失败 |
| `request_ball_pose_detection()` | `BallPoseDetectionResponse`，包含明确类型的 `detections` | 没有球先验时 `detections=()`；单个球漏检时对应 `detected=False`，坐标为空元组 | 帧非法、图像/深度计算异常、服务端业务异常 |

### 响应边界

- 算法成功响应只描述算法结果，不重复定义 `error` 字段。
- RPC 失败统一写入 `CameraPipelineServiceResponse.error`，客户端将其转换为 `RuntimeError`。
- “没有检测到目标”只有在对应算法明确允许时才是成功空结果；opening 和稳定帧不允许用空结果掩盖业务失败。
- `debug_artifacts` 为空元组表示调用方关闭了 debug 或该请求没有生成调试载荷，不表示算法失败。
- `frame_id` 是实际计算帧号；如果请求使用稳定帧，响应中的帧号应被视为最终证据帧，而不是请求时的最新帧。

## 测试边界

无真实设备时可以验证协议、loopback RPC、请求路由、稳定帧和资源释放。真实相机连通性、模型性能、发布帧率和现场稳定阈值必须在 Orin 上另行验证。
