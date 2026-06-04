from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tomllib

# region 数据结构

ArmSide = Literal["left", "right"]
ArmDeviceName = Literal["left_arm", "right_arm"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_NETWORK_CONFIG_PATH = PROJECT_ROOT / "config" / "robot_network.toml"


@dataclass(frozen=True, slots=True)
class WujiArmQmlinkerConfig:
    """无际机械臂 qmlinker 连接配置。

    职责边界：
    - 只描述本机 qmlinker 客户端访问基础控制工控机所需的网络参数。
    - 不负责 SSH 登录、GUI 控件、线程池调度或可视化模型。

    设计思想：
    - 直接使用接口文档提供的 qmlinker wheel，避免重复实现 SDK 内部逻辑。
    - 默认主机读取 `config/robot_network.toml` 中的基础控制工控机 IP。

    生命周期：
    - 不持有网络连接，可在线程间安全传递。

    继承关系：
    - 不继承业务基类，作为机械臂协议配置数据使用。
    """

    host: str = "192.168.100.60"
    "基础控制工控机地址，单位为 IPv4、主机名或 host:port 字符串。"

    port: int = 50062
    "机器人 ArmService 默认 gRPC 端口，单位为 TCP 端口号。"

    default_speed_ratio: float = 0.3
    "关节命令默认速度比例，范围通常为 0.0 到 1.0。"

    request_timeout_s: float = 3.0
    "普通 unary gRPC 请求超时时间，单位 s。"

    stream_first_timeout_s: float = 2.0
    "读取关节状态流首帧的超时时间，单位 s。"

    def target(self) -> str:
        """返回 gRPC 连接目标字符串。

        Returns
        -------
        str
            gRPC target，格式为 `host:port`。若 `host` 已包含端口则原样返回。
        """

        if ":" in self.host:
            return self.host
        return f"{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class WujiRobotNetworkConfig:
    """无际机器人网络地址配置。

    职责边界：
    - 只保存项目配置文件中的固定网络地址。
    - 不负责创建 qmlinker channel、SSH 连接或 GUI 控件。

    设计思想：
    - 将基础控制工控机和 Orin 模组区分保存，避免把 SSH 地址误用为 qmlinker 地址。
    - 使用不可变 dataclass，便于在 GUI 与后端之间传递。

    生命周期：
    - 每次从 `config/robot_network.toml` 读取后构造，不持有外部资源。

    继承关系：
    - 不继承业务基类，作为网络配置数据使用。
    """

    base_control_ip: str = "192.168.100.60"
    "基础控制工控机 IP，qmlinker 二次开发接口连接该地址。"

    orin_ip: str = "192.168.100.70"
    "Orin 模组 IP，用于 SSH 登录与边缘计算链路。"

    qmlinker: WujiArmQmlinkerConfig = WujiArmQmlinkerConfig()
    "qmlinker 机械臂连接配置。"


def load_wuji_robot_network_config(path: Path | None = None) -> WujiRobotNetworkConfig:
    """读取无际机器人网络配置。

    Parameters
    ----------
    path:
        可选配置路径，为 `None` 时读取项目根目录下 `config/robot_network.toml`。

    Returns
    -------
    WujiRobotNetworkConfig
        网络地址配置。配置文件不存在时返回默认值。
    """

    config_path = ROBOT_NETWORK_CONFIG_PATH if path is None else path
    if not config_path.exists():
        return WujiRobotNetworkConfig()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    network = data.get("network", {})
    qmlinker_data = data.get("qmlinker", {})
    qmlinker_config = WujiArmQmlinkerConfig(
        host=str(qmlinker_data.get("host", network.get("base_control_ip", "192.168.100.60"))),
        port=int(qmlinker_data.get("port", 50062)),
        default_speed_ratio=float(qmlinker_data.get("default_speed_ratio", 0.3)),
        request_timeout_s=float(qmlinker_data.get("request_timeout_s", 3.0)),
        stream_first_timeout_s=float(qmlinker_data.get("stream_first_timeout_s", 2.0)),
    )
    return WujiRobotNetworkConfig(
        base_control_ip=str(network.get("base_control_ip", qmlinker_config.host)),
        orin_ip=str(network.get("orin_ip", "192.168.100.70")),
        qmlinker=qmlinker_config,
    )



# endregion


# region 轴与设备映射

SUPPORTED_ARM_DEVICES: tuple[ArmDeviceName, ...] = ("left_arm", "right_arm")
"当前 arm 接口文档可真实控制的机械臂设备。"


def parse_arm_axis_name(axis_name: str) -> tuple[ArmDeviceName, int] | None:
    """解析 GUI DoF 轴名为机械臂设备与关节索引。

    Parameters
    ----------
    axis_name:
        GUI 轴名，例如 `left_j1` 或 `right_j6`。

    Returns
    -------
    tuple[ArmDeviceName, int] | None
        成功时返回设备名与 1 基关节索引；非机械臂轴返回 `None`。
    """

    if axis_name.startswith("left_j"):
        index_text = axis_name.removeprefix("left_j")
        if index_text.isdigit() and 1 <= int(index_text) <= 6:
            return "left_arm", int(index_text)
        return None
    if axis_name.startswith("right_j"):
        index_text = axis_name.removeprefix("right_j")
        if index_text.isdigit() and 1 <= int(index_text) <= 6:
            return "right_arm", int(index_text)
        return None
    return None


def axis_names_for_device(device_name: ArmDeviceName) -> tuple[str, ...]:
    """返回指定机械臂对应的 GUI 轴名序列。

    Parameters
    ----------
    device_name:
        机械臂设备名，取值为 `left_arm` 或 `right_arm`。

    Returns
    -------
    tuple[str, ...]
        GUI 轴名序列，长度为 6，单位语义为 deg。
    """

    prefix = "left" if device_name == "left_arm" else "right"
    return tuple(f"{prefix}_j{idx}" for idx in range(1, 7))


# endregion
