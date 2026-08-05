# 东莞鼎泰项目

## 推荐环境版本

以下版本组合已在 2026-06-09 实测通过，可正常初始化 Qt 内嵌 Open3D 视图：

- Python 3.10
- numpy 1.26.4
- open3d 0.19.0
- PySide6 6.11.x
- plyfile < 1.1.0

全新 Orin 部署当前 CameraPipeline、RecordReplay、RobotControl 和 API Gateway 的环境、网络、证书、systemd 与验收流程见：[Orin 四服务部署指南](docs/服务部署指南.md)。

如果把 `numpy` 升到 2.x，当前项目常用的 `plyfile 1.0.x` 约束会被破坏；如果回退到 `open3d 0.18.0`，在本机 `PySide6 6.11.x` 环境下又复现过嵌入窗口初始化崩溃，因此当前仓库统一按上述组合维护。

## Open3D GUI 已知问题与修复

2026-06-09 排查到 `gui/util_components/open3d_widget.py` 在 Windows + Qt 内嵌 Open3D 场景下有两类稳定复现的问题：

1. `open3d 0.18.0 + numpy 2.2.6 + PySide6 6.11.1` 组合下，`Visualizer` 的 `ViewControl.set_lookat(...)` 会在窗口初始化阶段直接导致进程退出，没有 Python 异常栈。
2. 同一环境下，`TriangleMesh.create_coordinate_frame(...)` 也会在部分初始化路径中触发进程级崩溃。

当前修复方案：

1. 环境固定为 `open3d==0.19.0` 与 `numpy==1.26.4`。
2. GUI 中不再使用 `TriangleMesh.create_coordinate_frame(...)` 构造辅助坐标轴，统一改为 `LineSet` 实现，代码位于 `gui/util_components/open3d_geometry_utils.py`。
3. `O3DViewerWidget` 内保留了 native window 嵌入路径与相机初始化日志，后续若再次出现无栈退出，优先从 `create_window -> get_view_control -> attach native window -> apply camera view` 这一链路复查。

## 第三方库

- loguru
  用于 日志记录
  `pip install loguru`

- numpy
  用于 数组计算
  固定 `1.26.4`
  `pip install numpy==1.26.4`

- scipy
  用于 数学计算，主要是旋转矩阵的计算
  `pip install scipy`

-tomlkit
  用于 解析 TOML 文件
  `pip install tomlkit`

- opencv-python
  用于 图像处理
  `pip install opencv-python`

- open3d
  用于 3D 模型处理与 Qt 内嵌 3D 视图
  固定 `0.19.0`
  `pip install open3d==0.19.0`

- PySide6 6.11.0  
  用于 用于 GUI 界面
  `pip install PySide6==6.11.0`

- qmlinker 1.0.8
  用于 无际二次开发接口，当前 GUI 机械臂调试页通过本机 DingTai 环境中的 qmlinker 连接基础控制工控机。
  `pip install env\qmlinker-1.0.8-py3-none-any.whl`

- protobuf 6.33.6
  qmlinker 与本项目静态工具链共用的 Protocol Buffers 运行库。
  `pip install "protobuf<7.0.0,>=6.33.5"`

- plyfile 1.0.3
  当前环境固定 `numpy==1.26.4`，因此不要安装要求 `numpy>=2.0` 的 `plyfile 1.1.x`。
  `pip install "plyfile<1.1.0"`

- tomlkit
  读写配置文件
  `conda install tomlkit -y`

- paramiko
  SSH tunnel 依赖
  `conda install -c conda-forge paramiko -y`

- (可选)PyTorch 2.9.1 + Cuda 12.6

  PyPose 的基础库，可选项。

  ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126```

## 无际二次开发接口

- 协议源文件放在 `resource/protos/`：
  - `resource/protos/common.proto`
  - `resource/protos/arm_service.proto`
  - `resource/protos/head_service.proto`
  - `resource/protos/lift_service.proto`
  - `resource/protos/waist_service.proto`
- 当前来源：`E:\DingTai\无际二次开发接口文档\protos\`
- 当前实现不再生成项目内 gRPC stub，而是直接使用文档提供的 qmlinker wheel。
- 网络地址配置放在 `config/robot_network.toml`：
  - `192.168.100.60`：基础控制工控机，qmlinker 二次开发接口连接该地址。
  - `192.168.100.50`：Orin 模组固定地址，用于平板直连与边缘计算链路。
- GUI 会先 ping 用户输入的 Orin 接入 IP，再探测 `192.168.100.60`：
  - 控制地址可达时按平板部署方式直连，不创建 SSH 转发。
  - 控制地址不可达时按本机测试方式，通过输入的 DHCP 地址建立 SSH 转发。
  - 成功输入会保存在本机 Qt 设置中，并作为下次启动默认值。
- 当前 GUI 已接入官方 xCoreSDK AR5、qmlinker 身体和头部控制：
  - AR5：左右臂实时状态、七关节 MoveAbsJ、带 elbow 的 MoveL、关节/笛卡尔 Jog、
    自由拖动和保持 TCP 的独立 elbow 调整；左右臂控制器 IP 分别在对应 AR5 页设置。
  - 身体：`body_z` 升降轴、`body_ry` 俯仰旋转轴。
  - 头部：`head_yaw` 可动旋转轴。
- AGV、夹爪和相机沿用现有控制接口；M11 手页及 GUI 连接暂时取消。

## xCoreSDK 机械臂直连与 SSH 转发

两台机械臂已接入机器人本体交换机。平板位于现场固定网段时直接访问控制器；本机不在
该网段时，由 GUI 通过用户输入的 Orin DHCP 地址建立共享 SSH 本地转发：

- GUI 的 qmlinker、相机和夹爪共享转发使用 Paramiko 完成非交互认证；AR5 严格复用
  `test/wuji/xcoresdk_arm_cli_test.py` 已验证的系统 OpenSSH `ssh -N -L` 路径。
- AR5 OpenSSH 通过 `SSH_ASKPASS` 和代码库内固定凭据自动认证，不再弹出密码窗口，
  密码不会出现在 SSH 进程命令行。
- 固定密码仅用于本机测试模式；平板直连模式不会创建 SSH 连接。

- 左臂控制器：`192.168.100.161`
- 右臂控制器：`192.168.100.160`

端口必须区分官方 GUI 和当前 Python SDK 的实际连接行为：

| 使用方 | 控制器端口 | 说明 |
| --- | ---: | --- |
| 官方 GUI | `5050` | 官方 GUI 连接控制器使用的端口。 |
| 官方 GUI | `4567` | 官方 GUI 使用的升级服务端口。 |
| `xCoreSDK_python` | `6666`、`7777` | 本项目当前 SDK 在创建 `xMateErProRobot` 对象时实际连接的两个端口；缺少任一转发都会报“网络异常：网络连接错误”。 |

`5050/4567` 与 `6666/7777` 不能互相替代。仅执行下面的官方 GUI 端口转发，不足以让当前 Python SDK 创建机器人对象：

```powershell
ssh -N -L 5050:192.168.100.160:5050 -L 4567:192.168.100.160:4567 orin
```

当前正式测试入口为 `test/wuji/xcoresdk_arm_cli_test.py`。本机模式会为两台机械臂分配
不同的 loopback 地址，使两边可以同时复用相同的固定端口：

| 机械臂 | SDK 本地地址 | 控制器地址 | 转发端口 |
| --- | --- | --- | --- |
| 左臂 | `127.0.0.2` | `192.168.100.161` | `5050`、`4567`、`6666`、`7777` |
| 右臂 | `127.0.0.3` | `192.168.100.160` | `5050`、`4567`、`6666`、`7777` |

等价的手工双臂转发命令如下：

```powershell
ssh -N `
  -L 127.0.0.2:5050:192.168.100.161:5050 `
  -L 127.0.0.2:4567:192.168.100.161:4567 `
  -L 127.0.0.2:6666:192.168.100.161:6666 `
  -L 127.0.0.2:7777:192.168.100.161:7777 `
  -L 127.0.0.3:5050:192.168.100.160:5050 `
  -L 127.0.0.3:4567:192.168.100.160:4567 `
  -L 127.0.0.3:6666:192.168.100.160:6666 `
  -L 127.0.0.3:7777:192.168.100.160:7777 `
  orin
```

单独排查连接时，可运行 `test/wuji/xcoresdk_ssh.py`。默认地址 `127.0.0.1` 适用于单臂
转发；双臂模式应分别使用 `127.0.0.2` 和 `127.0.0.3`。2026-07-20 已实测两台控制器
均可成功创建对象并读取 `robotInfo()`。

## 相机链路现状

以下结论来自 2026-06-05 在本机、`orin` 与 `wuyou` 上的联合排查。

- 当前 `wuyou` 在线运行的整机桥接程序是 `grpc_bridge_v2`，它只暴露 `Arm/Head/Lift/Waist/Base/Hand/Weili` 等 gRPC 服务。
- `grpc_bridge_v2` 当前**没有注册** `CameraService`，因此对 `50062` 上的相机 `GetCameraIntrinsics`、`StreamGetImage2D`、`StreamGetRGBDImage`、`SetEnabled` 等请求会返回 `UNIMPLEMENTED`。
- 当前相机真实服务不是 qmlinker gRPC，而是 `sensors_depthcamera_ob_zmq_v2`：
  - 控制端口：`5570`
  - 数据端口：`5560`、`5561`、`5562`、`5563`
- 结论必须强调：**相机应走 ZMQ，不应再走 qmlinker CameraService。**

当前仓库内已经补了对应验证与接入：

- 独立探测脚本：`test/wuji/test_zmq_camera_probe.py`
- ZMQ 相机客户端：`src/wuji/zmq_camera_client.py`
- GUI/backend 相机链路已改为通过 ZMQ 相机客户端读取真实状态、内参和 RGB/RGBD 图像。

## 相机服务端重映射

2026-06-05 经过重新枚举、序列号核对与服务端修正后，当前 `wuyou` 上已经明确以下三路 Orbbec 相机映射：

- `HEAD` / `head_camera` -> `CPC7B530000P` -> `Orbbec Gemini 336L`
- `CHEST` / `chest_camera` -> `CP9365300011` -> `Orbbec Gemini 336`
- `LEFT` / `left_hand_camera` -> `CV2R161000B1` -> `Orbbec Gemini 305`

以上三路当前都已经进入 `sensors_depthcamera_ob_zmq_v2` 通讯服务，并能在 GUI 中显示其远端槽位名与可复制序列号。

本地实测结果已落盘：

- 原始探测：`test/wuji/artifacts/camera_probe/`
- ZMQ 探测：`test/wuji/artifacts/zmq_camera_probe_live/`
- 重新核对后的 ZMQ 探测：`test/wuji/artifacts/zmq_camera_probe_left_remap/`

当前可以确认的结论是：

- `head_camera` 已确认对应 `Orbbec Gemini 336L`，序列号 `CPC7B530000P`
- `chest_camera` 已确认对应 `Orbbec Gemini 336`，序列号 `CP9365300011`
- `left_hand_camera` 已确认对应 `Orbbec Gemini 305`，序列号 `CV2R161000B1`
- `right_hand_camera` 当前离线

### 当前 USB 设备

2026-06-05 现场重新枚举后，`wuyou` 上当前可见的业务 USB 设备如下：

- `Orbbec Gemini 336`
  - USB 枚举：`2bc5:0803`
  - 当前服务端槽位：`CHEST`
- `Orbbec Gemini 336L`
  - USB 枚举：`2bc5:0807`
  - 当前服务端槽位：`HEAD`
- `DECXIN Camera`
  - USB 枚举：`1bcf:2d4f`
  - 当前标签：头部全景相机
  - 当前服务端定义：尚未在 `sensors_depthcamera_ob_zmq_v2` 中形成独立 ZMQ 槽位
- `Orbbec Gemini 305`
  - USB 枚举：`2bc5:0840`
  - 当前服务端槽位：`LEFT`
  - 当前序列号：`CV2R161000B1`

### 设备序列号

当前 `ob_camera.yaml` 与现场可见设备对应的序列号记录如下：

- `camera_head` -> `CPC7B530000P`
- `camera_chest` -> `CP9365300011`
- `camera_left` -> `CV2R161000B1`
- `camera_right`

结合现场 USB 重新检查后的结论：

- `camera_head` 已确认对应 `Orbbec Gemini 336L`
- `camera_chest` 已确认对应 `Orbbec Gemini 336`
- `camera_left` 已确认对应 `Orbbec Gemini 305`
- `camera_right` 当前离线

## 现场 USB 与功能归属

以下结论来自 2026-06-05 在 `orin` 与 `wuyou` 上的现场检查。

- `orin` 侧仅看到蓝牙相关 USB 设备，没有相机、CAN、网卡等业务外设。
- `wuyou` 侧挂载了所有当前可见的业务 USB 外设；其中一部分设备位于 USB 拓展坞后面，Linux `lsusb -t` 里会先看到 `Genesys Logic` Hub，再看到其下游设备。
- 当前 qmlinker 服务目标是 `wuyou`，不是 `orin`。

| 逻辑项              | 当前确认的物理设备                                               | 连接机器 | 备注                                                                                |
| ------------------- | ---------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------- |
| `head_camera`       | `Orbbec Gemini 336L`                                              | `wuyou`  | 对应序列号 `CPC7B530000P`；当前已进入 ZMQ 相机服务并在线。                           |
| `chest_camera`      | `Orbbec Gemini 336`                                               | `wuyou`  | 对应序列号 `CP9365300011`；当前已进入 ZMQ 相机服务并在线。                           |
| `left_hand_camera`  | `Orbbec Gemini 305`                                               | `wuyou`  | 对应序列号 `CV2R161000B1`；2026-06-05 已补入 `LEFT` 并在 GUI 中可见。                |
| `right_hand_camera` | 当前服务端槽位为空，保留为离线占位                               | `wuyou`  | 旧 `RIGHT` 物理设备已拆除，当前应按离线状态理解。                                   |
| `left_arm`          | AR5 左臂控制器 `192.168.100.161`                                 | 交换机   | GUI 通过官方 xCoreSDK 直连，或经 Orin 转发 SDK 固定端口。                           |
| `right_arm`         | AR5 右臂控制器 `192.168.100.160`                                 | 交换机   | GUI 通过官方 xCoreSDK 直连，或经 Orin 转发 SDK 固定端口。                           |
| `body`              | qmlinker 本体服务                                                | `wuyou`  | 目前可见到 `body_z` 与 `body_ry` 控制链路。                                         |
| `head`              | qmlinker 头部服务                                                | `wuyou`  | 目前可见到 `head_yaw` 控制链路。                                                    |
| `hand`              | qmlinker 手部服务                                                | `wuyou`  | 左右手当前以逻辑执行器形式出现，仓库未找到独立 USB 枚举映射。                       |
| `agv`               | `Microchip USBCAN/CANalyst-II` / `AX88179 USB Ethernet` 相关链路 | `wuyou`  | AGV 相关链路与 USB CAN / USB 网卡有关，但当前仓库还没有把它直接绑定为单一硬件设备。 |

补充说明：

1. `wuyou` 上的相机设备都挂在 USB 拓展坞之后，实际排查时要看 `lsusb -t` 的树状层级，而不是只看顶层 `lsusb`。
2. 当前 AR5 通过官方 xCoreSDK 控制；`body`、`head`、`hand`、`agv` 仍是协议/服务层逻辑模块。
3. 当前相机链路应按 `sensors_depthcamera_ob_zmq_v2` 的 ZMQ 服务处理，不复用 AR5 SDK 链路。
4. `DECXIN Camera` 当前仍作为独立的头部全景相机理解，但它不属于当前 Orbbec ZMQ 深度相机四槽位中的任何一路。
5. 截至 2026-06-05，`HEAD / CHEST / LEFT` 三路 Orbbec 的型号与序列号已经完成固化；当前未完成确认的只剩 `RIGHT`。

## 手部夹爪负载参数

 1.51 kg 重心：（-21.83，1.56，65.10）

## 远端 CA 证书私钥

zkgj
