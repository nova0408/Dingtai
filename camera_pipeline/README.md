# Orin Runtime Architecture

## 目标

`orin/` 目录现在按“协议、IO、算法”分层：

1. `camera_stream/` 只负责采集和缓存相机流。
2. `tray_detection/` 只负责托盘检测算法。
3. `opening_detection/` 只负责开口检测和抓取位姿算法。
4. `pipeline_context.py` 只负责把相机流、托盘结果和请求/响应 IO 串起来。

核心原则是：

1. 算法模块只做计算，不直接管相机、不直接管 RPC。
2. `context` 负责维护运行时、拉流、RPC 请求和帧选择。
3. 每个模块都能独立得到自己的结果。
4. 开口计算和位姿计算只依赖远端托盘检测结果 + 本地相机流。

---

## 目录职责

```text
orin/
  camera_stream/
  tray_detection/
  opening_detection/
  pipeline_context.py
  service.py
```

### `camera_stream/`

负责：

1. 连接远端 Orbbec / wuyou 相机控制口。
2. 拉取 RGBD 数据流。
3. 维护最近帧缓存。
4. 向上提供 `CameraFramePacket`。

### `tray_detection/`

负责：

1. 读取一帧 RGB 图像。
2. 输出托盘检测结果和调试掩码。
3. 保持托盘编号从左到右排序。

### `opening_detection/`

负责：

1. 接收单个托盘掩码和相机 RGBD 数据。
2. 执行开口边缘检测。
3. 构造近邻平面和顶面掩码。
4. 计算抓取平面和平面姿态。
5. 输出开口、调试图和抓取位姿结果。

### `pipeline_context.py`

负责：

1. 维护相机流运行时。
2. 向远端托盘检测服务发起 RPC。
3. 根据 `frame_id` 选择缓存帧，或回退到最新帧。
4. 把 IO 数据交给算法模块。

---

## 数据流

### 托盘检测

```text
context -> camera_stream -> tray_detection -> tray_results
```

### 开口检测与位姿计算

```text
context -> camera_stream + tray_detection结果 -> opening_detection -> grasp_pose
```

其中：

1. `tray_detection` 先得到全部托盘结果。
2. `opening_detection` 从中选择 `target_tray_index` 对应托盘。
3. 算法只拿输入数据，不自己请求相机或托盘服务。

---

## 当前实现

### 托盘检测服务

入口：

```text
python -m orin.service tray_detection
```

运行时：

1. `PipelineContext` 负责维持相机流。
2. 服务从 `context` 取帧。
3. `OrinTrayDetectionExecutor.compute(frame, request)` 只做算法计算。

### 开口检测服务

入口：

```text
python -m orin.service opening_detection
```

运行时：

1. `PipelineContext` 先提供相机帧。
2. `PipelineContext` 再请求远端托盘检测。
3. `OpeningDetectionPipelineExecutor.compute(frame, tray_mask, ...)` 只做开口与位姿计算。

---

## 设计约束

1. 不再从 `src/` 取算法逻辑来拼远端实现。
2. 不再用 lint 注释掩盖导入边界问题。
3. 协议数据放在 `protocol.py` 和 `codec.py`。
4. 算法模块只保留计算函数和必要的几何工具。
5. `context` 是数据 IO 的唯一编排层。

---

## 说明

当前实现已经把三个模块收敛到同一条数据流上，但仍保留独立服务入口，方便单独调试：

1. 托盘检测可以单独跑。
2. 开口检测可以通过 `pipeline_context` 组合托盘结果和相机流。
3. 后续如果只需要一个总入口，也可以在 `service.py` 上再封装统一调度。

