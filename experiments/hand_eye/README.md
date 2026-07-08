# hand_eye 实验页

`test_charuco_hand_eye.py` 是一个 ChArUco 眼在手上手眼标定测试脚本。
`test_charuco_board_viewer.py` 是一个专门用于查看 ChArUco 标定板识别结果的实时 `cv2` 预览页。
`hand_eye_orin_left_arm_drag.py` 是一个新的左臂实采页：启动后自动打开左臂拖动，接入 Orin 左手相机流，实时叠加 ChArUco 识别结果、左臂关节角、基于 base 的 flange 位姿和采样建议，并在采样后直接更新 `T_flange_camera` 外参结果文件。
`hand_eye_orin_left_arm_board_fixed_joint_solve.py` 是一个固定标定板的联合求解页：每次按空格记录 `T_base_flange` 与 `T_camera_board`，随后持续联合优化 `T_flange_camera` 与 `T_base_board`。

已固定的板参数：

- `square_length_mm = 15.0`
- `marker_length_mm = 11.25`
- `dictionary = DICT_5X5_1000`

你还需要按实物板确认棋盘行列数。当前默认按 `9x12` 配置：

- `squares-x = 9`
- `squares-y = 12`

运行时：

- `p` 或回车：记录一帧样本
- `q` 或 `Esc`：退出

左臂实采页运行时：

- 启动即连接左臂并打开拖动示教，只用于左臂
- 使用 Orin `left_hand_camera` 相机流做实时预览
- 预览上直接显示板识别结果、机器人关节角、基于 base 的 flange 位姿、采样覆盖度和下一步拖动建议
- `Enter` / `Space` / `P`：记录当前样本
- 每次记录都会保存原始图、预览图、CSV 样本和单独的 `flange_camera_extrinsic_result.txt`
- 求解时使用：
  - A 组：`T_base_flange`
  - B 组：`T_camera_board`
  - 结果：`T_flange_camera`
  - 约束：`T_base_flange @ T_flange_camera @ T_camera_board = constant`

固定板联合求解页运行时：

- 启动后同样连接左臂并打开拖动示教
- 使用 Orin `left_hand_camera` 相机流做实时预览
- `Enter` / `Space` / `P`：记录当前样本
- 每次记录都会保存原始图、预览图、CSV 样本和单独的 `base_board_flange_camera_joint_result.txt`
- 求解时使用：
  - 已知采样：`T_base_flange`
  - 已知采样：`T_camera_board`
  - 联合未知：`T_flange_camera`、`T_base_board`
  - 约束：`T_base_flange @ T_flange_camera = T_base_board @ T_camera_board`
  - 同时让所有样本尽量满足同一链路

识别查看页运行时：

- 只显示实时 `cv2` 结果，不使用 Qt
- 显示 ChArUco 角点、标定板外接轮廓、坐标轴和重投影误差
- `L`：切换 `legacyPattern`
- `q` 或 `Esc`：退出

如果额外提供 `--robot-pose-csv`，脚本会把同编号机器人位姿一起写入样本并在样本数足够时调用现有手眼求解。
识别查看页不做手眼求解，只负责看清楚标定板识别是否稳定。
