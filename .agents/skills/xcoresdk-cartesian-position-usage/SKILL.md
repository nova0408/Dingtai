---
name: xcoresdk-cartesian-position-usage
description: 规范 Dingtai 项目中 xCoreSDK `CartesianPosition` 的读取、构造与单位约定。用于机械臂实验、手眼标定、调试页或运动脚本中处理 `robot.cartPosture(...)` 返回值或主动构造 `CartesianPosition` 时，强制区分 `pos/trans/rpy` 的语义，并在当前项目阶段基于 `trans/rpy` 自行组装位姿矩阵，因为实测 `pos` 为空值不可用；同时按 pyi 约定使用 3+3、6 或 16 个数值构造 `CartesianPosition`。特别注意：读取到的原始姿态结果必须先保留其原始单位语义，再转换为内部计算矩阵，避免把显示单位误写入 `pose_matrix`。
---

# xCoreSDK CartesianPosition Usage

## 适用范围

1. 使用 `sdk.xcoresdk.xCoreSDK_python` 的机器人实验页、调试页、采样脚本、运动脚本。
2. 任何读取 `robot.cartPosture(...)` 返回的 `CartesianPosition`，或需要主动构造 `xCoreSDK_python.CartesianPosition(...)` 的代码。

## pyi 已确认的接口

`sdk/xcoresdk/xCoreSDK_python/__init__.pyi` 中 `CartesianPosition` 支持以下构造：

1. `CartesianPosition(trans, rpy)`
   - `trans`: 长度 3
   - `rpy`: 长度 3
2. `CartesianPosition(frame6)`
   - `frame6`: 长度 6，顺序 `[X, Y, Z, Rx, Ry, Rz]`
3. `CartesianPosition(matrix16)`
   - `matrix16`: 长度 16，行优先 4x4 齐次变换矩阵

## 强制规则

1. 当前项目阶段不要使用 `robot.cartPosture(...)` 返回的 `CartesianPosition.pos` 作为位姿真值；已实测该字段可能为空值，视为当前 SDK 路径不可靠。
2. `CartesianPosition.trans` 的原始单位是 `m`，`CartesianPosition.rpy` 的原始单位是 `rad`。这两个字段可以直接参与计算链路，但不应直接作为 GUI/CSV 的最终展示单位。
3. 当前项目代码读取 `cartPosture(...)` 时，必须基于 `trans + rpy` 自行重建位姿矩阵；重建时统一按 scipy `Rotation.from_euler("xyz", rpy, degrees=False)` 解释 SDK 原始 `rpy`，并让 `pose_matrix` 保持 `m` 语义。
4. 若同时需要“用于计算的位姿”和“用于给人看的数值”，必须分离来源：
   - 计算链路：`trans/rpy -> pose_matrix(m)`
   - 显示链路：`pose_matrix(m) -> mm + deg`
5. 对手眼标定场景，`T_base_flange` 或同类机器人绝对位姿在当前阶段必须来自 `trans/rpy` 拼装结果，不能再依赖 `pos`，也不能先把平移写成 `mm` 再塞回 `pose_matrix`。
6. 当代码需要显示欧拉角时，统一先把 `rpy` 转成 `deg` 再显示；但这些角度不是位姿真值来源。
7. 主动构造 `CartesianPosition` 时，必须先明确当前 API 期望的是：
   - `trans/rpy`
   - `frame6`
   - `matrix16`
   不允许靠“先试一把”猜构造方式。
8. 能直接使用 xCoreSDK 原生对象或 `numpy.ndarray` 时，就直接使用；不要再额外封装 `src.utils.datas.Transform`、`Quaternion`、`Translation` 一类项目姿态对象，避免多套数据结构产生语义漂移。

## 构造约定

1. 当下游接口明确要求位置与欧拉角时，可使用 `CartesianPosition(trans, rpy)` 或 `CartesianPosition([X, Y, Z, Rx, Ry, Rz])`。
2. 上述 `X/Y/Z` 与 `trans` 保持 SDK 原始长度单位 `m`；`Rx/Ry/Rz` 与 `rpy` 保持 SDK 原始角度单位 `rad`。
3. 当下游接口天然以齐次变换矩阵工作，优先使用 `CartesianPosition(matrix16)`，其中 `matrix16` 必须是行优先 4x4 齐次变换矩阵展开后的 16 个值。
4. 若代码中已有 `np.ndarray shape=(4, 4)` 变换矩阵，推荐先显式转成行优先 16 值列表，再传给 `CartesianPosition(matrix16)`。

## 常见单位坑

1. `CartesianPosition(trans, rpy)` 与 `CartesianPosition([X, Y, Z, Rx, Ry, Rz])` 走的是 SDK 原始单位，不是 GUI/CSV 常用单位。
2. 对这两种构造来说：
   - `X/Y/Z` 或 `trans` 必须使用 `m`
   - `Rx/Ry/Rz` 或 `rpy` 必须使用 `rad`
3. 项目里的 GUI、日志、CSV 常常显示为：
   - 平移：`mm`
   - 欧拉角：`deg`
4. 因此，以下做法是错误的：
   - 直接把 GUI 上看到的 `mm` 数值传给 `CartesianPosition([X, Y, Z, Rx, Ry, Rz])`
   - 直接把 CSV 里的 `deg` 数值传给 `CartesianPosition(trans, rpy)`
5. 若输入来源是 GUI/CSV 的 `mm/deg`，必须先显式换算回 SDK 原始单位，再构造 `CartesianPosition`。
6. 若输入来源已经是 4x4 齐次变换矩阵，并且矩阵平移项已经按目标接口单位准备好，优先使用 `CartesianPosition(matrix16)`，避免再走一遍 `mm/deg -> m/rad` 的手工拼装。
7. 任何时候都不要把“显示用单位”和“计算用单位”混进同一个 `pose_matrix` 字段里；该字段应该保留算法输入的原始语义，若内部约定是 `m`，就始终是 `m`。
8. 若代码里存在 `PoseSnapshot.pose_matrix`、`base_flange_pose`、`end_pose_matrix` 这类内部矩阵字段，必须在字段定义或构造函数旁明确写出单位约定，避免后续维护者把 GUI 的 `mm` 误当成内部矩阵单位。
9. 这条规则必须从源头执行：读取 `cartPosture(...)` 后，先基于原始 `trans(m) + rpy(rad)` 生成内部矩阵；只有在最终显示、日志或 CSV 落盘时，才把平移转成 `mm`、角度转成 `deg`。

```python
x_m = x_mm / 1000.0
y_m = y_mm / 1000.0
z_m = z_mm / 1000.0
rx_rad = np.deg2rad(rx_deg)
ry_rad = np.deg2rad(ry_deg)
rz_rad = np.deg2rad(rz_deg)

target_pose = xCoreSDK_python.CartesianPosition(
    [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
)
```

## 推荐实现片段

```python
cartesian_pose = robot.cartPosture(xCoreSDK_python.flangeInBase, ec)
if ec.get("ec", 0) != 0:
    raise RuntimeError("读取机器人位姿失败")

translation_m = np.asarray(cartesian_pose.trans, dtype=np.float64).reshape(3)
rpy_rad = np.asarray(cartesian_pose.rpy, dtype=np.float64).reshape(3)
rotation = Rotation.from_euler("XYZ", rpy_rad, degrees=False).as_matrix()
base_flange_pose = np.eye(4, dtype=np.float64)
base_flange_pose[:3, :3] = rotation
base_flange_pose[:3, 3] = translation_m

translation_mm = tuple(float(value) * 1000.0 for value in cartesian_pose.trans)
rpy_degrees = tuple(float(np.degrees(float(value))) for value in cartesian_pose.rpy)
```

上面这段示例里要特别注意：

1. `base_flange_pose` 是内部计算矩阵，所以平移项必须保持 `m`。
2. `translation_mm` 只是给界面、日志或 CSV 用的展示值，不能再反写回 `base_flange_pose`。
3. 一旦把 `translation_m * 1000.0` 写进矩阵，再参与 `T_base_tool @ T_tool_cam @ T_cam_board` 这类链式计算，就会造成数量级错误，看起来像“标定结果不稳定”，实际是单位污染。

```python
target_pose = xCoreSDK_python.CartesianPosition(
    [x_m, y_m, z_m],
    [rx_rad, ry_rad, rz_rad],
)
```

```python
target_pose = xCoreSDK_python.CartesianPosition(
    [
        r11, r12, r13, tx,
        r21, r22, r23, ty,
        r31, r32, r33, tz,
        0.0, 0.0, 0.0, 1.0,
    ]
)
```

## Jog 与 Move 指令补充

1. `startJog(...)` 的接口约束以 `sdk/xcoresdk/xCoreSDK_python/__init__.pyi` 为准：
   - 笛卡尔 jog 的 `step` 单位是 `mm`
   - 关节 jog 的 `step` 单位是 `deg`
2. `startJog(...)` 需要机器人处于手动模式；执行完 jog 后必须显式调用 `robot.stop(ec)` 收尾，不能假设 SDK 自动停稳。
3. 若需求是“根据当前感知结果一次性生成目标 TCP 并执行补偿”，优先使用 `MoveL` / `MoveAbsJ` 这类自动模式运动接口；不要把“内部矩阵单位是 `m`”错误外推成 `startJog(step=...)` 也该传 `m`。

## MoveAbsJ 的 speed 语义

1. `MoveAbsJ` 的 `speed` 不是关节速度比例，而是机器人末端线速度，单位是 `mm/s`。
2. 当前项目在调用 `MoveAbsJ` 时应默认按末端线速度理解和传递，不再使用 `(0, 1]` 这类比例值去表达运动快慢。
3. 参考手册中可用的末端线速度范围按 `5 ~ 4000 mm/s` 处理；默认回放速度可用 `2000 mm/s`，offset 触发 CSV 这类临时补偿场景可用 `700 mm/s`。
4. `MoveAbsJ` 的速度参数和 `zone` 是两类不同语义：
   - `speed`: 末端线速度，单位 `mm/s`
   - `zone`: 路径过渡/到位容差，按该接口文档对应单位单独填写
5. 代码里若同时存在 `speed`、`jointSpeed`、`zone` 三个名字，必须先确认 pyi 和手册定义，再决定哪个字段承载真实语义；不要因为历史命名沿用就把末端线速度误写成关节比例。
6. 若当前模块只需要统一一个速度字段，优先命名为能反映真实单位和语义的字段，例如 `move_abs_j_end_linear_speed_mm_s`，避免后续维护者误以为它仍是关节比例。

## Drag 模式补充

1. 若页面目标是“人工拖动机械臂，同时实时观测某个矩阵量是否恒定”，优先直接开启 drag，而不是额外下发 MoveL / Jog 补偿。
2. 当前仓库里可复用的 drag 开启顺序应保持一致：
   - 先强制固定 `tool/wobj`
   - `setMotionControlMode(NrtCommandMode)`
   - `setPowerState(False)`
   - `setOperateMode(manual)`
   - `moveReset()`
   - `enableDrag(cartesianSpace, freely, enable_drag_button=False)`
3. 对依赖 `endInRef` 的页，drag 开启前和每次读取 `cartPosture(endInRef)` 前都应重新强制固定 `toolset`，避免控制器残留状态让 `endInRef` 参考系漂移。
4. 若脚本只是做“offset 是否恒定”的验算，界面应优先显示：
   - 当前 `tcp`
   - 当前 `cam_board`
   - 当前 `base_board`
   - 当前 `offset`
   不应混入尚未确认正确性的目标 TCP 运动链。

## 输出要求

1. 明确说明当前位姿是来自 `cartPosture(...).pos` 还是来自主动构造的 `CartesianPosition(...)`。
2. 若脚本界面显示机器人姿态，当前项目统一显示平移 `mm` 与 `rpy_deg`；其中 `rpy` 仅作为基于 SDK 原始 `rpy` 的展示结果，不应和独立姿态真值来源混淆。
3. 若修改涉及 `CartesianPosition` 的读取或构造，必须同步检查同目录相关实验页，避免一处修复、另一处继续混用单位或语义。
