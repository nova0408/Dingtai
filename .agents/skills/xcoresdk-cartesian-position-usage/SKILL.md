---
name: xcoresdk-cartesian-position-usage
description: 规范 Dingtai 项目中 xCoreSDK `CartesianPosition` 的读取、构造、长度单位与欧拉角顺序。用于机械臂实验、手眼标定、调试页或运动脚本中处理 `robot.cartPosture(...)` 返回值或主动构造 `CartesianPosition` 时，强制使用 `trans(m)`、`rpy(rad)`，按 SciPy 小写外禀 `from_euler("xyz")` 重建 SDK 位姿，并严格区分项目展示用大写内禀 `as_euler("XYZ")`；禁止使用实测可能为空的 `pos`，禁止把 mm/deg 或展示欧拉角回灌到内部计算矩阵。
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

## 两套欧拉角约定必须严格分离

| 数据语义 | 长度单位 | 角度单位 | SciPy 调用 | 用途 |
| --- | --- | --- | --- | --- |
| SDK 原始 `trans/rpy` | `m` | `rad` | `Rotation.from_euler("xyz", sdk_rpy_rad, degrees=False)` | 重建计算用位姿矩阵 |
| SDK 原始 RPY 人工查看 | `mm` | `deg` | 只对原始值做 `np.degrees(sdk_rpy_rad)` | 日志/CSV，字段名必须带 `sdk_rpy_xyz_deg` |
| 项目矩阵姿态展示 | `mm` | `deg` | `Rotation.from_matrix(R).as_euler("XYZ", degrees=True)` | 项目统一的最终展示约定 |

小写 `"xyz"` 与大写 `"XYZ"` 在 SciPy 中不是同一种旋转：小写表示外禀旋转，大写表示内禀旋转。SDK 原始 `rpy` 的矩阵重建只能使用小写 `"xyz"`。项目要求的 `as_euler("XYZ")` 只用于从已经构造好的旋转矩阵生成最终展示值，绝不能用于解释 SDK 原始 `rpy`，也不能把该展示值转换后回灌 SDK 或标定计算。

## 强制规则

1. 当前项目阶段不要使用 `robot.cartPosture(...)` 返回的 `CartesianPosition.pos` 作为位姿真值；已实测该字段可能为空值，视为当前 SDK 路径不可靠。
2. `CartesianPosition.trans` 的原始单位只能按 `m` 解释，`CartesianPosition.rpy` 的原始单位只能按 `rad` 解释。变量名必须显式写成 `translation_m`、`sdk_rpy_rad`；禁止使用无单位名称继续向计算链路传递。
3. 读取 `cartPosture(...)` 后，必须直接基于原始 `trans + rpy` 重建位姿矩阵；唯一允许的 SDK RPY 解释是 `Rotation.from_euler("xyz", sdk_rpy_rad, degrees=False)`。禁止写成大写 `"XYZ"`、其它顺序或 `degrees=True`。
4. 若同时需要“用于计算的位姿”和“用于给人看的数值”，必须分离来源：
   - 计算链路：`trans(m) + sdk_rpy(rad) -> from_euler("xyz") -> pose_matrix(m)`
   - SDK 原值查看：`trans(m) -> mm`，`sdk_rpy(rad) -> np.degrees(...)`
   - 项目姿态展示：`pose_matrix(m) -> translation(mm) + as_euler("XYZ", deg)`
5. 对手眼标定场景，`T_base_flange` 或同类机器人绝对位姿在当前阶段必须来自 `trans/rpy` 拼装结果，不能再依赖 `pos`，也不能先把平移写成 `mm` 再塞回 `pose_matrix`。
6. SDK 原始 RPY 的 degree 显示与项目矩阵的 `XYZ` degree 显示必须使用不同字段名。前者命名为 `sdk_rpy_xyz_deg`，后者命名为 `pose_rpy_XYZ_deg`；禁止都写成含糊的 `rpy_degrees`。
7. 主动构造 `CartesianPosition` 时，必须先明确当前 API 期望的是：
   - `trans/rpy`
   - `frame6`
   - `matrix16`
   不允许靠“先试一把”猜构造方式。
8. 能直接使用 xCoreSDK 原生对象或 `numpy.ndarray` 时，就直接使用；不要再额外封装 `src.utils.datas.Transform`、`Quaternion`、`Translation` 一类项目姿态对象，避免多套数据结构产生语义漂移。
9. 修改任何读取 `cartPosture(...)` 的代码时，必须检查以下字面规则：
   - 不得出现 `Rotation.from_euler("XYZ", sdk_rpy_rad, ...)`。
   - 不得把 `translation_mm` 写入计算矩阵。
   - 不得把 `pose_rpy_XYZ_deg` 转成 rad 后当作 SDK 原始 RPY。
   - 手眼标定必须用至少一组历史数据或合成数据验证单位换算和旋转顺序；仅通过 ruff/pyright 不算完成。

## 构造约定

1. 当下游接口明确要求位置与欧拉角时，可使用 `CartesianPosition(trans, rpy)` 或 `CartesianPosition([X, Y, Z, Rx, Ry, Rz])`。
2. 上述 `X/Y/Z` 与 `trans` 保持 SDK 原始长度单位 `m`；`Rx/Ry/Rz` 与 `rpy` 保持 SDK 原始角度单位 `rad`，顺序固定为 SDK 外禀 `xyz`。
3. 从旋转矩阵主动构造 SDK RPY 时，必须使用 `Rotation.from_matrix(rotation).as_euler("xyz", degrees=False)`；不得使用项目展示值 `as_euler("XYZ", degrees=True)` 直接构造 `CartesianPosition`。
4. 若输入只有项目展示用 `pose_rpy_XYZ_deg`，必须先按 `Rotation.from_euler("XYZ", pose_rpy_XYZ_deg, degrees=True)` 恢复旋转矩阵，再从该矩阵使用 `as_euler("xyz", degrees=False)` 得到 SDK RPY。禁止仅做 `deg -> rad` 后直接发送。
5. 当下游接口天然以齐次变换矩阵工作，优先使用 `CartesianPosition(matrix16)`，其中 `matrix16` 必须是行优先 4x4 齐次变换矩阵展开后的 16 个值。
6. 若代码中已有 `np.ndarray shape=(4, 4)` 变换矩阵，推荐先显式转成行优先 16 值列表，再传给 `CartesianPosition(matrix16)`。

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
sdk_rpy_rad = np.asarray(cartesian_pose.rpy, dtype=np.float64).reshape(3)
rotation = Rotation.from_euler("xyz", sdk_rpy_rad, degrees=False).as_matrix()
base_flange_pose = np.eye(4, dtype=np.float64)
base_flange_pose[:3, :3] = rotation
base_flange_pose[:3, 3] = translation_m

translation_mm = tuple(float(value) * 1000.0 for value in cartesian_pose.trans)
sdk_rpy_xyz_deg = tuple(float(np.degrees(float(value))) for value in cartesian_pose.rpy)
pose_rpy_XYZ_deg = Rotation.from_matrix(base_flange_pose[:3, :3]).as_euler(
    "XYZ", degrees=True
)
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
2. 若显示 SDK 原始返回值，字段必须明确为平移 `translation_mm` 与 `sdk_rpy_xyz_deg`；这里只允许对原始 `trans/rpy` 做单位转换。
3. 若显示项目统一矩阵姿态，字段必须明确为 `translation_mm` 与 `pose_rpy_XYZ_deg`；姿态必须来自 `Rotation.from_matrix(...).as_euler("XYZ", degrees=True)`。
4. 禁止使用含糊的 `rpy_deg` 同时表示 SDK 原始 RPY 和项目矩阵展示 RPY。
5. 若修改涉及 `CartesianPosition` 的读取或构造，必须同步检查同目录相关实验页，避免一处修复、另一处继续混用单位或语义。
