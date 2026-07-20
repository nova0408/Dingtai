# CameraPipeline

## 相机输入模式

`config.json` 的 `camera_source_mode` 是部署时的硬编码模式开关：

- `zmq`：默认模式，继续使用 `camera_stream` 连接远端相机控制口和数据口，并维护有限历史帧缓存。
- `usb`：使用 `camera_local` 按 LEFT/RIGHT/HEAD/CHEST 的 SN 连接本机 Orbbec 相机，只保存当前最新帧，断线后持续自动重连。

四个安装位的逻辑名称、SN、启用状态、ZMQ 端口以及 USB color/depth profile 均定义在
`config.json`。默认调用相机不在 JSON 中配置，由 `PipelineContextConfig` 的代码默认值决定。
USB 模式不允许以枚举顺序代替 SN；设备缺失时按指数退避持续重试，SN 为空时线程保持空闲，
同时通过服务 Loguru 配置记录连接、断线、异常和恢复日志。

## 目标

`camera_pipeline` 是以 Python 3.12 部署在 NVIDIA Orin 上的相机数据与视觉算法服务。它统一连接上游 RGBD 相机流，对外提供相机状态、稳定帧、帧订阅以及 ball 位姿算法请求。

> `tray_detection` 和 `opening_detection` 已暂时从运行时、RPC 协议和公共 client API 移除，其目录内源码仅作为后续恢复参考，不参与当前部署。两个子目录的 `__init__.py` 已删除。

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

面向用户的全部 client API、请求/响应字段、合法空结果和服务错误统一见
[API Reference](API%20Reference.md)；算法 README 只描述算法内部数据契约和计算失败原因。

## 模块结构

| 模块 | 单一职责 | 文档 |
| --- | --- | --- |
| `camera_stream` | 连接上游相机服务、解码 RGBD、维护最近帧缓存 | [camera_stream/README.md](camera_stream/README.md) |
| `stable_frame` | 根据连续 RGBD 帧判断稳定窗口并输出中点帧号 | [stable_frame/README.md](stable_frame/README.md) |
| `charuco_detection` | 使用原图、CLAHE 和 unsharp 回退融合检测 ChArUco 位姿 | [charuco_detection/README.md](charuco_detection/README.md) |
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

### 多相机订阅与内参

主要调用入口使用明确安装位的方法名：

```python
head_intrinsics = client.get_head_camera_intrinsics()
chest_frames = client.subscribe_chest_camera_frames()
left_color_frames = client.subscribe_left_arm_camera_color_frames()
```

完整帧、彩色帧和深度帧均提供 `head/chest/left_arm/right_arm` 命名 API。
右臂 API 已保留，但当前相机未连接，调用时会得到明确的 `RuntimeError`。
`subscribe_camera_frames(camera_name)`、`subscribe_camera_color_frames(camera_name)`、
`subscribe_camera_depth_frames(camera_name)` 等参数化 API 继续保留，用于通用工具与测试。

## 部署

```bash
/home/wuji-brain/miniconda3/envs/wuji/bin/python -m camera_pipeline.service \
  --bind-addr tcp://0.0.0.0:6200 \
  --control-port 5570 \
  --stream-port 5562 \
  --camera-id LEFT \
  --camera-name left_hand_camera
```

服务默认同时写入 systemd/journald 控制台日志和
`logs/camera_pipeline_service.log` 文件日志。文件按 `20 MB` 轮转、ZIP 压缩并保留
`14 days`；可通过 `--log-path`、`--log-rotation`、`--log-retention` 覆盖。
当前最低日志级别为 `INFO`，不输出 `DEBUG`。

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
| `5560` | 上游头部相机数据流 |
| `5561` | 上游胸腔相机数据流 |
| `5562` | 上游左臂相机数据流 |
| `5563` | 上游右臂相机数据流（当前未连接） |

服务端 bind 和客户端 connect 地址分离。`0.0.0.0` 只用于 bind，Orin 本地调用使用 `127.0.0.1`。

## 协议与安全

统一请求包含 `protocol_version`。当前版本 4 已移除 tray/opening 请求和响应类型。
网络数据采用白名单协议 dataclass 的 JSON 元数据与 NumPy 原始字节块，不使用
Python pickle。服务端部署基线为 Python 3.12。解码器拒绝未知类型、非法数组范围和字段不匹配；端口仍只应开放在受控
开发网络中。

## 测试

无真实设备时允许执行：

- ruff、pyright；
- 稳定帧合成数据测试；
- fake context 业务路由测试；
- 本机 in-process ZMQ 协议回环；
- 服务关闭和端口释放测试。

以下内容必须在 Orin 和真实相机上验证：

- 上游相机控制与 RGBD 解码；
- ball 模型文件、CUDA 和显存；
- 真实帧率下的缓存容量；
- 稳定帧阈值；
- ball 精度与耗时；
- 外接设备和 Orin 本地业务服务的实际连通性。
