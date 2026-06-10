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

`GUI / smoke -> DahuanGripperClient -> QMGripper -> 夹爪 RPC`

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
