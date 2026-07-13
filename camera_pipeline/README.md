# CameraPipeline

## 目标

`camera_pipeline` 是部署在 NVIDIA Orin 上的相机数据与视觉算法服务。它统一连接上游 RGBD 相机流，对外提供相机状态、稳定帧、帧订阅以及 tray/opening/ball 三类算法请求。

本项目同时支持两种消费者：

```text
外接开发机 -> tcp://<orin-ip>:6200
Orin 本地业务服务 -> tcp://127.0.0.1:6200
```

两者使用相同的公共客户端：

```python
from camera_pipeline.client import CameraPipelineClient
```

## 架构原则

1. `PipelineContext` 是唯一的相机运行时和帧生命周期管理者。
2. `service` 负责网络服务、请求路由和业务调用编排。
3. 算法子模块只接收协议输入并执行纯计算，不连接相机、不创建 RPC。
4. 帧选择统一经由 `PipelineContext.resolve_frame()`：指定正数 `frame_id` 时精确取帧，未指定时默认等待稳定帧。
5. debug 数据通过请求显式控制；关闭后算法不构造大体积调试图像。
6. 文本和协议必须明确单位、坐标系和错误语义。

## 模块结构

| 模块 | 单一职责 | 文档 |
| --- | --- | --- |
| `camera_stream` | 连接上游相机服务、解码 RGBD、维护最近帧缓存 | [camera_stream/README.md](camera_stream/README.md) |
| `stable_frame` | 根据连续 RGBD 帧判断稳定窗口并输出中点帧号 | [stable_frame/README.md](stable_frame/README.md) |
| `tray_detection` | 在单帧彩色图中检测并分割托盘 | [tray_detection/README.md](tray_detection/README.md) |
| `opening_detection` | 基于单帧 RGBD 和单个托盘 mask 计算开口及抓取位姿 | [opening_detection/README.md](opening_detection/README.md) |
| `ball_pose_detection` | 根据颜色和几何先验检测球心三维坐标 | [ball_pose_detection/README.md](ball_pose_detection/README.md) |
| `service` | 统一 RPC、帧发布、业务编排和部署入口 | [service/README.md](service/README.md) |
| `pipeline_context.py` | 组装相机运行时，解析指定帧或稳定帧 | 本文“帧选择” |
| `protocol.py` | 跨模块 RGBD 帧 Protocol 和帧传输 packet | 本文“架构原则” |
| `client.py` | 用户级 `CameraPipelineClient` facade | 本文“调用方式” |

## 数据流

### 默认算法请求

```text
CameraPipelineClient
  -> service transport
  -> CameraPipelineApplication
  -> PipelineContext.resolve_frame(frame_id <= 0)
  -> StableFrameDetector
  -> 中点 frame_id -> CameraStreamRuntime 缓存帧
  -> 纯算法 service.compute(frame, request)
  -> response
```

### 指定帧请求

调用方可以先显式获取稳定帧，再把返回的正数 `frame_id` 传给算法请求。`resolve_frame()` 会精确使用该缓存帧，不会重新等待。

### opening 业务流程

```text
resolve one frame
  -> tray detection
  -> select target tray mask
  -> opening detection
  -> compose OpeningDetectionPipelineResponse
```

该组合逻辑属于业务层，位于 `service/application.py`；两个算法仍保持独立纯计算。

## 调用方式

### Orin 本地业务服务

```python
from camera_pipeline.client import CameraPipelineClient

client = CameraPipelineClient()
status = client.get_camera_status()
```

默认连接 `tcp://127.0.0.1:6200`。

### 外接开发机

```python
client = CameraPipelineClient(
    service_addr="tcp://192.168.1.118:6200",
    timeout_ms=30_000,
)
```

外接地址必须由调用方显式提供，不写入正式业务逻辑。

## 部署

```bash
python -m camera_pipeline.service \
  --bind-addr tcp://0.0.0.0:6200 \
  --control-port 5570 \
  --stream-port 5562 \
  --camera-id LEFT \
  --camera-name left_hand_camera
```

仓库脚本：

```bash
bash scripts/restart_camera_pipeline_service.sh
```

另一个 Orin 本地业务服务可在 systemd 中声明：

```ini
Requires=camera-pipeline.service
After=camera-pipeline.service
```

`After` 只保证进程顺序，不保证相机首帧就绪。业务服务应调用 `get_camera_status()` 做明确可用性检查。

## 端口

| 端口 | 用途 |
| --- | --- |
| `6200` | 统一 REQ/REP 服务 |
| `6201` | 完整 RGBD 帧 PUB |
| `6202` | 彩色帧 PUB |
| `6203` | 深度帧 PUB |
| `5570` | 上游相机控制口 |
| `5562` | 上游相机数据流 |

服务端 bind 和客户端 connect 地址分离。`0.0.0.0` 只用于 bind，Orin 本地调用使用 `127.0.0.1`。

## 协议与安全

统一请求包含 `protocol_version`。当前网络数据使用 Python pickle，只适用于受信任的 Orin 和开发内网，不能直接暴露到不可信网络，也不提供跨语言兼容性。

## 测试

无真实设备时允许执行：

- ruff、pyright；
- 稳定帧合成数据测试；
- fake context 业务路由测试；
- 本机 in-process ZMQ 协议回环；
- 服务关闭和端口释放测试。

以下内容必须在 Orin 和真实相机上验证：

- 上游相机控制与 RGBD 解码；
- 模型文件、CUDA 和显存；
- 真实帧率下的缓存容量；
- 稳定帧阈值；
- tray/opening/ball 精度与耗时；
- 外接设备和 Orin 本地业务服务的实际连通性。
