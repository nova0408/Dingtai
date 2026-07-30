# `src.wuji` 模块说明

## 总体定位

`src.wuji` 不是再造一套 `qmlinker`，而是把 `qmlinker` 中**项目侧必须固定、必须命名统一、必须拆分清楚**的部分收口到无际封装层。

当前原则：

- 直接复用 `qmlinker` 的原生对象能力，不再额外包一层 facade
- `session` 只负责 `channel`、连接与必要的 SSH 转发
- 机械臂、头部、底盘、右手、左手夹爪、相机分别独立代码页实现
- GUI 与 smoke 只依赖当前真实链路
- 远端 Orin 若缺少新版 `qmlinker`，会先同步本地 `env/qmlinker-1.0.15-py3-none-any.whl` 再执行

## 当前设备语义

- 左手是夹爪，单独走大寰夹爪链路，对应`QMGripper`
- 右手是灵巧手，对应 `QMHand`
- 机械臂未开发
- 头部对应 `QMHead`
- body 对应 `QMLift` / `QMWaist`
- AGV 对应 `QMMoveBase`
- 相机读取来自Orin的转发流，对应`CameraPipeline`

## 当前信息流

### 右手灵巧手


用于：

- 读取右手执行器状态
- 读取右手执行器数量
- 读取右手使能
- 设置右手使能
- 设置右手状态

当前右手硬件为 `M6`。`WujiRightHandClient` 直接读取 qmlinker `hand_info`，
根据 `actuator_count` 和 `actuator_names` 生成运行时执行器规格，不再在协议客户端中硬编码轴数。

### 左手夹爪

用于：

- 读取左手夹爪状态
- 设置使能
- 校准
- 设置位置

左手固定为夹爪，所有位置、速度、力与校准都走夹爪专用接口。冒烟测试直接按位置、速度和力的目标值调用。

### 机械臂

机械臂更新为珞石AR5七轴机械臂，直接使用SDK控制

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


### 相机

相机流通过部署在Orin中的camera_pipeline服务提供。

### 设备专用文件

每个设备一个实现页，职责清晰：

- `arm_client.py`
- `body_client.py`
- `head_client.py`
- `right_hand_client.py`
- `dahuan_gripper_client.py`
- `agv_client.py`
- `zmq_camera_client.py`

