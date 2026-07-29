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
| `application.py` | 相机请求和 ball 业务调用编排 |
| `server.py` | 通用 REP 请求循环和统一异常边界 |
| `logging_config.py` | 服务入口唯一的 Loguru 控制台与轮转文件 sink 配置 |
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

`tray_detection` 和 `opening_detection` 已暂时从 application、统一协议、wire codec 与公共 client API 移除，不参与当前服务进程。

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
/home/wuji-brain/miniconda3/envs/wuji/bin/python -m camera_pipeline.service \
  --bind-addr tcp://0.0.0.0:6200
```

相机源、上游控制端口、各相机流端口、相机名称与启用状态统一来自
`camera_pipeline/config.json`。

可按部署环境覆盖日志参数：

```bash
/home/wuji-brain/miniconda3/envs/wuji/bin/python -m camera_pipeline.service \
  --log-path logs/camera_pipeline_service.log \
  --log-rotation "20 MB" \
  --log-retention "14 days"
```

## 日志

服务统一使用 Loguru，最低级别固定为 `INFO`，暂不启用 `DEBUG`。服务入口同时创建：

- 控制台 sink：写入 stderr，由 systemd/journald 收集；
- 文件 sink：默认写入 `logs/camera_pipeline_service.log`，UTF-8 编码；
- 文件达到 `20 MB` 后轮转，轮转文件 ZIP 压缩并保留 `14 days`；
- 两个 sink 都使用 `enqueue=True`，避免服务线程直接执行文件写入。

日志覆盖服务启动与停止、信号退出、请求 operation/耗时/结果、API 请求参数与响应摘要、
相机流启动与恢复、稳定帧判定、ChArUco 每次尝试、ball 候选与匹配摘要、解码失败和
发布器生命周期。正常逐帧数据不写日志，发布队列的预期丢帧也不逐条记录，避免长期
运行时产生高频日志。日志只记录标量、状态和计数，不写图像、mask、点云数组或完整
协议对象。日志目录无法创建或文件 sink 无法初始化时，服务启动失败。

头部、胸腔和左臂运行时按 `camera_pipeline/config.json` 中的启用状态与端点表启动。

systemd 模板位于 `camera-pipeline.service`，进程启动、停止上限分别为 20 秒和
15 秒。该 unit 使用 `Type=simple`，systemd 拉起进程不等于相机业务已经就绪；
部署与重启脚本另外最多等待 20 秒，并调用 `get_camera_status(camera_name)` 确认
目标相机首帧可用，不能只检查 6200 端口。本地其他业务服务应声明
`Requires=camera-pipeline.service` 和 `After=camera-pipeline.service`。

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
三个 XPUB 端口都使用以 `camera_name + NUL` 编码的 topic 前缀，因此头部、
胸腔和左臂可复用同一组发布端口，订阅端不会收到其他安装位的帧。
右臂 topic 和 client API 已保留，但当前端点标记为未连接，请求会明确返回服务错误。
发布器跟踪真实订阅与取消订阅事件，只编码当前存在订阅者的相机和帧类型。

## 协议与安全边界

请求包含 `protocol_version`，当前版本为 `10`。版本 8 为三球先验增加每球专属
`hsv_ranges`，并在检测结果中返回 `observed_hsv`。相机相关请求必须显式携带
`camera_name`；版本 9 在 `CameraStatusResponse` 中增加必填 `service_version`，
供客户端在连通性阶段核对远端功能版本。客户端与服务端线协议版本不一致时服务端
明确拒绝。版本 10 为 ChArUco debug 响应增加最终检测帧 overlay。

三球请求若携带球先验但不含有效毫米尺度相对位置关系，服务将其识别为先验采集，
并强制要求 `enable_debug=True`；完全不带球先验的冒烟请求不属于先验采集。

REQ/REP 和三路帧流使用同一显式二进制协议：固定头部携带 JSON 元数据长度，
元数据只允许白名单中的协议 dataclass、基础类型和元组，NumPy 图像与 mask 以
连续原始字节块附加，并在元数据中记录 `dtype`、`shape`、偏移和长度。协议不使用
Python pickle，不依赖 dataclass 的 Python 版本内存布局。Orin 服务固定使用
`/home/wuji-brain/miniconda3/envs/wuji/bin/python`（Python 3.10+），不再使用
`pytorch_38`/`py38_tourch` 环境。未知类型、未知协议标识、越界数组
和字段不匹配均会明确拒绝。

三类帧流彼此独立：彩色发布读取彩色缓存，完整帧与深度发布读取 RGBD 缓存。
因此上游暂时只有 RGB 时，`camera_color_frame_subscribe` 仍持续转发彩色帧，
不会被尚未恢复的深度流阻塞。ZMQ SUB 客户端允许发布端晚启动或重启。

## 错误约定

请求解码失败和业务异常均由 `CameraPipelineServer` 捕获，写入统一响应的 `error`
字段；单个非法请求不会终止服务循环。客户端发现顶层错误或目标 payload 缺失时
抛出 `RuntimeError`。算法模块不负责网络错误转换。

请求解码失败和业务异常写入 `ERROR`，可恢复的相机控制、重连和停止异常写入
`WARNING`；稳定帧超时、ChArUco 最终未识别和 ball 候选/匹配不足也写入 `WARNING`；
正常生命周期、API 响应、算法开始与成功摘要写入 `INFO`。

## API Reference

全部公开 client API、请求/响应字段和错误边界已迁移到
[CameraPipeline API Reference](../API%20Reference.md)。

## 测试边界

无真实设备时可以验证协议、loopback RPC、请求路由、稳定帧和资源释放。真实相机连通性、模型性能、发布帧率和现场稳定阈值必须在 Orin 上另行验证。
# Board 检测

`charuco_detection` 由 CameraPipeline 完整负责相机稳定帧获取、Board 构造、角点融合和
PnP 位姿计算。调用方必须传入字典、方格数量、方格边长、marker 边长及检测边界；服务端
不保存默认板型。调用方不订阅相机帧，也不自行实现检测。

成功时返回 `status="detected"` 和 `T_camera_board`，平移单位 mm；未形成有效位姿时返回
`status="missing"` 和空矩阵。服务级输入、相机或算法异常仍通过统一 `error` 返回。
