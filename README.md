# 东莞鼎泰项目

## 第三方库

- loguru
  用于 日志记录
  `pip install loguru`

- pythonocc 7.9.3
  用于 3D 仿真
  `conda install -c conda-forge pythonocc-core=7.9.3`
  
- numpy
  用于 数组计算
  `pip install numpy<2.0.0`

- scipy
  用于 数学计算，主要是旋转矩阵的计算
  `pip install scipy`

-tomlkit
  用于 解析 TOML 文件
  `pip install tomlkit`

- pyorbbecsdk2

奥比中光 SDK
用于 奥比中光 Gemini 305 相机的控制
`pip install pyorbbecsdk2`

- opencv-python
  用于 图像处理
  `pip install opencv-python`

- open3d
  用于 3D 模型处理
  `pip install open3d`

- PySide6 6.11.0  
  用于 用于 GUI 界面
  `pip install PySide6==6.11.0`

- qmlinker 1.0.8
  用于 无际二次开发接口，当前 GUI 机械臂调试页通过本机 DingTai 环境中的 qmlinker 连接基础控制工控机。
  `pip install E:\DingTai\无际二次开发接口文档\api\qmlinker-1.0.8-py3-none-any.whl`

- protobuf 6.33.6
  qmlinker 与本项目静态工具链共用的 Protocol Buffers 运行库。
  `pip install "protobuf<7.0.0,>=6.33.5"`

- plyfile 1.0.3
  qmlinker 间接依赖。当前项目固定 `numpy<2.0.0`，因此不要使用要求 `numpy>=2.0` 的新版 plyfile。
  `pip install "plyfile<1.1.0"`

- PyTorch 2.9.1 + Cuda 12.6

  PyPose 的基础库

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
  - `192.168.100.70`：Orin 模组，用于 SSH 登录和边缘计算链路。
- GUI 机械臂调试页默认使用 `config/robot_network.toml` 中的 qmlinker 地址创建 gRPC channel。
- 当前 GUI 已接入 qmlinker 的臂、身体和头部控制：
  - 身体：`body_z` 升降轴、`body_ry` 俯仰旋转轴。
  - 头部：`head_yaw` 可动旋转轴。
- AGV 底盘暂不接入当前 qmlinker 调试页；底盘使用另一套 SDK，后续有开发计划时再单独接入。
