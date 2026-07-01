# hand_eye 实验页

`test_charuco_hand_eye.py` 是一个 ChArUco 眼在手上手眼标定测试脚本。
`test_charuco_board_viewer.py` 是一个专门用于查看 ChArUco 标定板识别结果的实时 `cv2` 预览页。

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

识别查看页运行时：

- 只显示实时 `cv2` 结果，不使用 Qt
- 显示 ChArUco 角点、标定板外接轮廓、坐标轴和重投影误差
- `L`：切换 `legacyPattern`
- `q` 或 `Esc`：退出

如果额外提供 `--robot-pose-csv`，脚本会把同编号机器人位姿一起写入样本并在样本数足够时调用现有手眼求解。
识别查看页不做手眼求解，只负责看清楚标定板识别是否稳定。
