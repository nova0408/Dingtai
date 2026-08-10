# ChArUco Detection

## 日志约定

`PipelineContext.detect_charuco()` 在业务编排边界记录日志：每次调用以 `INFO` 记录
相机、最大尝试帧数和 debug 开关；每次稳定帧检测记录 frame、marker 数、ChArUco
角点数、重投影误差和状态；成功记录最终帧与尝试次数，达到上限仍为 `missing` 时记录
`WARNING`。单帧算法不创建日志 sink，也不记录图像或角点数组。

## 职责

`charuco_detection` 接收一帧满足 `camera_pipeline.protocol.RgbdFrameProtocol`
的稳定彩色帧和调用方构造的 `cv2.aruco.CharucoBoard`，输出标定板到相机的
位姿。模块不连接相机、不等待稳定帧、不做重试、RPC、文件读写或窗口显示。

纯彩色稳定帧获取和最多 5 帧重试由 `PipelineContext.detect_charuco()` 负责。
ChArUco 只依赖彩色图、彩色相机内参和畸变参数，不读取深度图，不受深度帧缺失、
深度噪声或 RGBD 稳定阈值影响。

## 输入

- `frame.color_bgr`：`(H, W, 3)`、`uint8`、BGR 图像。
- `frame.fx/fy/cx/cy`：彩色相机针孔内参，单位 pixel。
- `frame.distortion`：OpenCV 畸变系数顺序，至少包含一个有限值。
- `board`：原生 `cv2.aruco.CharucoBoard`。棋盘格数量、字典、方格边长和
  marker 边长均由该对象提供，长度统一使用 mm。
- `CharucoDetectionConfig`：最小角点数、CLAHE、unsharp 和 debug 坐标轴参数。

HTTP 服务层通过 `dictionary_name` 从当前 OpenCV `cv2.aruco` 暴露的预定义字典中构造
`CharucoBoard`；本算法模块只接收已经构造好的 Board，不维护字典白名单。

模块只依赖 OpenCV、NumPy、`camera_pipeline.protocol` 和自身文件，不从
`src`、`experiments`、`test` 或其它算法子模块导入。

## 算法流程

1. BGR 转灰度，将当前帧内参与畸变参数写入
   `cv2.aruco.CharucoDetector`，再通过 `detectBoard()` 一次完成 marker 检测和
   ChArUco 角点插值；不再调用已废弃且部分 Linux wheel 未导出的
   `interpolateCornersCharuco()`。
2. 原始灰度角点足够且 PnP 成功时直接返回，不运行增强分支。
3. 位姿失败时，额外执行 CLAHE 和轻量 unsharp；原始分支结果不会重复计算。
4. 汇总 none、CLAHE、unsharp 三条分支。相同 ID 的像素坐标逐维取中位数，
   每个物理角点只保留一次，避免重复 ID 对 PnP 形成额外权重。
5. 从 `board.getChessboardCorners()` 按 ChArUco ID 取板坐标，调用
   `cv2.solvePnP(..., SOLVEPNP_ITERATIVE)`。
6. 计算参与 PnP 的唯一角点平均重投影误差。

未使用 Otsu。ArUco 检测器内部已有自适应阈值，外部全局二值化可能在阴影或
反光区域丢失边界。未使用 bilateral，避免高分辨率实时图像的额外开销和小 marker
边缘钝化。

## 输出

`CharucoDetectionResult` 核心字段：

- `status`：`detected` 或 `missing`。
- `t_cam_board_mm`：成功时为 `(4, 4)` `float64` 齐次矩阵，失败时为空矩阵。
- `error_px`：平均重投影误差，单位 pixel；失败时为正无穷。
- `marker_num`：最终融合后的唯一 marker ID 数量。
- `charuco_num`：最终融合后的唯一 ChArUco ID 数量。

坐标变换关系为：

```text
p_cam = T_cam_board @ p_board
```

`T_cam_board` 的旋转把板坐标轴转换到相机坐标系，平移单位与 board 长度一致，
本模块约定 board 使用 mm，因此字段名为 `t_cam_board_mm`。

## Debug

`enable_debug=False` 时：

- 不复制彩色图。
- 不绘制 marker、ChArUco 角点或 pose 坐标轴。
- `debug_artifacts=()`。

`enable_debug=True` 时返回一个 `CharucoDebugArtifacts`：

- `overlay_bgr`：最终融合 marker、ChArUco 角点和有效 pose 坐标轴叠加图。
- `marker_corners_px/marker_ids`：融合后的 marker 数据。
- `charuco_corners_px/charuco_ids`：融合后的 ChArUco 数据。

marker 和 ChArUco 数量属于低成本核心诊断值，无论 debug 是否开启都返回。

## 成功、空结果与失败

- 成功：`status="detected"`，位姿矩阵为 `(4, 4)`，误差为有限非负数。
- 未检测到或角点不足：`status="missing"`，返回当前帧融合计数、空位姿矩阵和
  正无穷误差。这是算法空结果，不是服务级异常。
- `PipelineContext.detect_charuco()`：单帧空结果时继续等待下一纯彩色稳定帧，最多尝试
  `max_frames=5`；第 5 帧仍失败则返回最后一帧的 `missing` 结果。
- 输入图像、焦距或畸变参数非法：抛出 `ValueError`。
- 稳定帧超时、缓存帧淘汰或相机不可用：由 `PipelineContext` 抛出 `RuntimeError`。

原生 `cv2.aruco.CharucoBoard` 当前作为进程内对象传入，不进入 wire codec，也不由
算法模块隐式读取外部配置文件。
