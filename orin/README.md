# Orin Opening Detection Pipeline Design

## 目标

仓库根目录现在只围绕一种主协议设计：

1. Orin 自己独立获取 `wuyou` 相机流。
2. Orin 在一次请求内按顺序执行：
   - 托盘检测
   - 缺口识别
   - 近邻面 / 顶面构建
   - 抓取位姿求解
3. 调用方只请求一次，就能直接得到“从左到右第 `index` 个托盘”的最终抓取位姿。
4. 当请求 `enable_debug=True` 时，同一次响应里额外返回调试底图、掩码和中间结果。

---

## 当前目录结构

```text
workspace/
  README.md
  service.py
  camera_stream/
  tray_detection/
  opening_detection/
  opening_detection_pipeline/
```

说明：

1. `camera_stream/`
   负责唯一的远端相机流获取与缓存。
2. `tray_detection/`
   负责阶段 1 托盘检测逻辑，并保留独立调试服务。
3. `opening_detection/`
   只保留阶段 2 算法实现，不再提供独立对外服务。
4. `opening_detection_pipeline/`
   负责对外主协议、顺序执行阶段 1 和阶段 2，并统一返回最终结果。
5. `service.py`
   作为 workspace 统一入口，只分发：
   - `tray_detection`
   - `opening_detection_pipeline`

部署说明：

1. 远端仓库根目录对应 `/home/wuji-brain/workspace`。
2. 仓库根目录直接承载 Python 模块，不再额外保留 `orin/` 包目录。
3. 远端服务与本机脚本都以仓库根作为统一工作区，而不是再嵌套一层 `/workspace/orin/`。

---

## 设计原则

1. 仓库内部逻辑必须闭环，不依赖本机 `src/`。
2. 相机流只能由一个持久化组件获取和缓存，算法阶段不得重复采流。
3. 最终业务协议只保留一个主服务：`opening_detection_pipeline`。
4. `tray_detection` 独立服务仅用于 debug 和算法开发，不作为最终调用协议。
5. 不保留旧的 RGBD 上传式服务兼容逻辑。
6. 主服务请求必须显式携带 `target_tray_index`，托盘编号规则固定为图像中从左到右。

---

## 主服务协议

主服务模块：

- `opening_detection_pipeline/protocol.py`
- `opening_detection_pipeline/codec.py`
- `opening_detection_pipeline/transport.py`
- `opening_detection_pipeline/engine.py`
- `opening_detection_pipeline/service.py`

默认地址：

```text
tcp://0.0.0.0:6220
```

请求字段：

1. `request_id`
2. `camera_name`
3. `frame_id`
   说明：通常传 `-1`，表示使用最新缓存帧。
4. `target_tray_index`
   说明：按图像从左到右编号。
5. `enable_debug`

响应字段：

1. `frame_id`
2. `tray_results`
   说明：阶段 1 全部托盘结果。
3. `selected_result`
   说明：目标托盘最终抓取结果。
4. `all_tray_results`
   说明：全部托盘最终抓取结果。
5. `debug`
   说明：仅在 `enable_debug=True` 时返回。

`debug` 中包含：

1. `color_bgr`
2. `depth_mm`
3. `camera_intrinsics`
4. `tray_instance_masks`
5. `selected_tray_mask`
6. `near_plane_mask`
7. `no_hole_mask`
8. `overlay_bgr`
9. `contrast_bgr`

---

## 执行顺序

主服务内部固定按以下顺序执行：

1. 从 `camera_stream` 解析请求帧。
2. 执行 `tray_detection`。
3. 按从左到右顺序编号托盘。
4. 选中 `target_tray_index` 对应托盘。
5. 把该托盘 `tray_mask` 送入 `opening_detection` 阶段。
6. 返回最终抓取位姿。
7. 若 `enable_debug=True`，则同时返回 RGB、深度、掩码和诊断图。

---

## 启动方式

统一入口：

```bash
/home/wuji-brain/miniconda3/envs/py38_tourch/bin/python -m service opening_detection_pipeline \
  --bind-addr tcp://0.0.0.0:6220 \
  --host 192.168.100.60 \
  --control-port 5570 \
  --stream-port 5562 \
  --camera-id LEFT \
  --camera-name left_hand_camera
```

托盘检测调试服务：

```bash
/home/wuji-brain/miniconda3/envs/py38_tourch/bin/python -m service tray_detection \
  --bind-addr tcp://0.0.0.0:6210 \
  --host 192.168.100.60 \
  --control-port 5570 \
  --stream-port 5562 \
  --camera-id LEFT \
  --camera-name left_hand_camera
```

---

## 冒烟测试

当前保留两个本机冒烟测试：

1. `test/wuji/test_tray_detection_rpc_smoke.py`
   用于单独查看阶段 1 托盘检测结果。
2. `test/wuji/test_opening_detection_rpc_smoke.py`
   用于通过主服务直接查看托盘 `0` 的最终抓取位姿。

另外还有一个 2D + 3D 联合查看器：

3. `test/pointcloud/test_orbbec_remote_opening_detection_viewer.py`
   用于查看主服务返回的 RGB、HSV 深度、掩码、点云和最终位姿。

---

## systemd

当前远端建议注册两个 user service：

1. `orin-tray-detection.service`
2. `orin-opening-detection-pipeline.service`

其中：

- `orin-opening-detection-pipeline.service` 是实际主服务
- `orin-tray-detection.service` 仅用于阶段 1 debug
