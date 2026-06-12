# `src.wuji` 模块说明

## 总体定位

`src.wuji` 不是再造一套 `qmlinker`，而是把 `qmlinker` 中**项目侧必须固定、必须命名统一、必须拆分清楚**的部分收口到无际封装层。

当前原则：

- 直接复用 `qmlinker` 的原生对象能力，不再额外包一层 facade
- `base` 只负责 `channel`、连接与必要的 SSH 转发
- 机械臂、头部、底盘、右手、左手夹爪、相机分别独立代码页实现
- GUI 与 smoke 只依赖当前真实链路
- 远端 Orin 若缺少新版 `qmlinker`，会先同步本地 `env/qmlinker-1.0.15-py3-none-any.whl` 再执行

## 当前设备语义

- 左手是夹爪，单独走大寰夹爪链路
- 右手是灵巧手，对应 `QMHand`
- 机械臂对应 `QMArm`
- 头部对应 `QMHead`
- body 对应 `QMLift` / `QMWaist`
- AGV 对应 `QMMoveBase`
- 相机当前以 ZMQ 相机客户端为主，不再依赖旧的 qmlinker 相机入口

## 当前信息流

### 右手灵巧手

`GUI / smoke -> WujiRightHandClient -> QMHand -> qmlinker hand RPC`

用于：

- 读取右手执行器状态
- 读取右手执行器数量
- 读取右手使能
- 设置右手使能
- 设置右手状态

右手固定为 `HandM`，共 11 个执行器，GUI 只显示固定轴序列，不再动态兼容旧手型。

### 左手夹爪

`GUI / smoke -> DahuanGripperClient -> Orin-side qmlinker script -> QMGripper -> 夹爪 RPC`

用于：

- 读取左手夹爪状态
- 设置使能
- 设置位置、速度、力
- 校准

左手不再使用通用 `hand` 语义。
左手固定为夹爪，所有位置、速度、力与校准都走夹爪专用接口。烟雾测试直接按位置、速度和力的目标值调用，不再保留旧的兼容路径。

### 机械臂

`GUI / smoke -> WujiArmClient -> QMArm -> qmlinker arm RPC`

用于：

- 读取关节数、限位和角度
- 设置关节目标
- 使能
- FK / IK

机械臂固定为 `ArmWuJi`，共 6 个关节；`arm.set_joints` 在项目封装层仅接受角度序列，由封装层补齐原始命令结构。烟雾测试也直接按角度序列组织，再由封装层转换成 `qmlinker` 需要的关节命令。

### body / head / AGV

- body: `QMLift` + `QMWaist`
- head: `QMHead`
- AGV: `QMMoveBase`

AGV 当前可稳定完成：

- 使能读取与设置
- 状态读取
- 四方向实时移动
- 停止
- 导航指令发送

导航同步确认不再作为 smoke 成功条件。

### 相机

当前 GUI 使用 ZMQ 相机客户端读取信息与流，不再回退到旧 qmlinker 相机封装。

## 当前存在问题

### 左手夹爪的 `QMGripper` 直连封装不可用

问题现象：

- 冒烟测试与 GUI 早期实现直接把 `DahuanGripperClient` 做成 `QMGripper` 的薄封装时，`get_status()` 会返回 `None`
- 夹爪控制会报 `Gripper control (type 6) failed: ... UNIMPLEMENTED`
- 这说明 GUI 侧直连的 `QMGripper` 路径并没有落到可用的 gripper 服务实现上

问题原因：

- 本机侧直接复用 `QMGripper` 的路径，实际上走到了不稳定的夹爪服务暴露面，而不是 Orin 上已验证可用的脚本执行路径
- `QMGripper` 读写接口和现场夹爪链路的实际语义并不完全一致，尤其是状态读取字段与位置回读行为，容易造成“接口调用成功但位置没动”的假象
- 早期烟测还把“稳定后必须等于目标值”作为硬条件，但这条夹爪链路的真实表现是“位置会变化并稳定”，不保证最终回读严格等于写入目标
- 因此，直连封装看起来像是“接口可调用”，但无法证明硬件真的按预期动作

定位结果：

- `grpc_bridge_v2` 本身是启动的，但它依赖的夹爪链路没有稳定提供可用的 `type 6` 状态与控制返回
- 直接把 `QMGripper` 当作 GUI 侧唯一入口，无法稳定完成状态读取、位置设置和回读确认

解决方式：

- 恢复为显式封装版 `DahuanGripperClient`
- GUI / smoke 不再直接依赖 `QMGripper` 作为对外接口，而是统一调用 `DahuanGripperClient`
- `DahuanGripperClient` 通过 Orin SSH 执行远端脚本，在 Orin 侧调用 `create_channel(base_control_ip:gripper_port)` 直连 `50066` 的夹爪服务，再用 `QMGripper` 发送 `set_enable`、`set_speed`、`set_force`、`set_pos` 与 `get_status`
- 冒烟测试继续保留“发送后轮询当前 position，直到稳定后再额外等待 1 秒确认”的策略

当前结论：

- `QMGripper` 只能作为远端脚本里的协议实现细节使用
- 项目侧对外必须保留 `DahuanGripperClient` 显式封装，不再把 `QMGripper` 暴露成 GUI 的直接依赖
- 这次能跑通的关键，不是改成“更宽松地承认接口成功”，而是把控制落回到 Orin 上已验证的链路，并按“位置变化 + 稳定”来判断动作是否真的生效

## 模块职责

### `client_base.py`

只保留基础连接、channel、SSH 转发和底层对象创建，不再承担上层 GUI 逻辑。

### 设备专用文件

每个设备一个实现页，职责清晰：

- `arm_client.py`
- `body_client.py`
- `head_client.py`
- `right_hand_client.py`
- `dahuan_gripper_client.py`
- `agv_client.py`
- `zmq_camera_client.py`

### `device_clients.py`

这里只做组合，不做额外业务逻辑。

## 验证现状

当前已用 DingTai 环境复验：

- 机械臂信息读取通过
- body / head 信息读取通过
- 右手信息与控制通过
- 左手夹爪信息与控制通过
- AGV 信息、使能与四方向控制通过
- AGV 导航指令发送通过
- Orin 抓取位姿与托盘检测 smoke 通过

## 约束

- 不保留无效入口
- 不保留只做转发的冗余 client
- 不把 GUI 再包装成旧包装层
- 不把 `src.wuji` 做成一个新的厚 facade
