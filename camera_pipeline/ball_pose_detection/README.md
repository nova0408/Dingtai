# Ball Pose Detection

`camera_pipeline.ball_pose_detection` 是一个只负责 RGBD 多球检测与位姿求解的服务模块。

## 算法逻辑

输入是一帧 RGBD 图像和一组球先验。先验必须按实际物体上的球顺序给出，并至少包含 3 个球。

算法按照下面的真实规则构建球坐标系：

1. 记录到的第一个球作为坐标系原点。
2. 第二个球与第一个球连线方向作为 `x` 轴。
3. 第三个球与前两个球共同确定 `xoy` 平面，平面法向用于构建 `z` 轴，随后按右手系得到 `y` 轴。
4. 如果输入超过 3 个球，后续球用于修正位姿误差。
5. 检测阶段先得到球坐标系到相机坐标系的刚体变换 `T_ball_cam`。
6. 如果请求携带 `reference_relative_transform_mm`，最终输出位姿会叠加该先验变换，得到 `T_final_cam = T_ball_cam @ T_relative`。
7. 所以最终位姿 `pose_transform`、`pose_rotation`、`pose_translation_mm` 都是叠加了先验修正后的相机坐标系结果。

位姿计算时，服务会优先使用前 3 个球构建基准位姿，再将全部可用球一起参与误差修正。如果全量拟合比基准拟合更稳定，则采用全量结果，否则回退到前三球基准结果。

## 输入数据

### `BallPoseDetectionRequest`

- `request_id`: 请求编号
- `camera_name`: 相机名称
- `frame_id`: 要处理的 frame 编号
- `enable_debug`: 是否返回 debug 数据
- `priors`: 球先验列表
- `reference_relative_transform_mm`: 可选的参考相对变换，用于采集和对比，不参与核心检测

### `BallPosePriorInfo`

- `name`: 球先验名称
- `color_hex`: 球颜色，使用 HEX 码表示
- `radius_mm`: 球半径，单位 mm
- `model_center_mm`: 球在球坐标系中的模型中心，单位 mm

## 输出数据

### `BallPoseDetectionResponse`

- `pose_transform`: 4x4 刚体变换矩阵，球坐标系到相机坐标系
- `pose_rotation`: 3x3 旋转矩阵
- `pose_translation_mm`: 3 维平移
- `residual_mm`: 拟合残差
- `matched_count`: 成功匹配到并参与位姿计算的球数量
- `detections`: 每个球的检测结果摘要
- `debug`: 调试数据
- `error`: 错误信息

## Debug 数据

`BallPoseDetectionDebugArtifacts` 中包含：

- `color_bgr`: 原始彩色图
- `depth_mm`: 原始深度图
- `camera_intrinsics`: 相机内参 `(fx, fy, cx, cy)`
- `overlay_bgr`: 叠加了球检测结果和最终位姿的图
- `detection_overlay_bgr`: 仅叠加球检测结果的图
- `detections`: 每个球的 debug 字典，包含颜色、半径、中心位置、像素半径、归一化几何、点数和状态

## 说明

该模块不负责相机采集、不负责 RPC 端点配置，也不负责先验采集脚本。
先验采集和位姿对比应由 `test/wuji/collect_ball_opening_relative_pose.py` 和 `test/wuji/ball_pose_detection.py` 完成。
