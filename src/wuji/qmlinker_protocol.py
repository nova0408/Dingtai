from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import tomllib

# region 数据结构

WujiQmlinkerEnableModuleName = Literal["body", "head", "left_arm", "right_arm"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_NETWORK_CONFIG_PATH = PROJECT_ROOT / "config" / "robot_network.toml"


@dataclass(frozen=True, slots=True)
class WujiQmlinkerConfig:
    """无际 qmlinker 连接配置。

    职责边界：
    - 只描述 qmlinker SDK 访问基础控制工控机所需的网络参数。
    - 不负责 GUI 控件、线程池调度、设备状态订阅或运动控制。

    设计思想：
    - 将 qmlinker 配置从 arm 模块移出，避免机械臂模块承载整机 SDK 语义。
    - 默认主机读取 `config/robot_network.toml` 中的基础控制工控机 IP。

    生命周期：
    - 不持有网络连接，可在线程间安全传递。

    继承关系：
    - 不继承业务基类，作为无际 qmlinker 集成层配置数据使用。
    """

    host: str = "192.168.100.60"
    "基础控制工控机地址，单位为 IPv4、主机名或 host:port 字符串。"

    port: int = 50062
    "qmlinker gRPC 端口，单位为 TCP 端口号。"

    default_speed_ratio: float = 0.3
    "关节命令默认速度比例，范围通常为 0.0 到 1.0。"

    request_timeout_s: float = 3.0
    "普通 unary gRPC 请求超时时间，单位 s。"

    stream_first_timeout_s: float = 2.0
    "读取流式接口首帧的超时时间，单位 s。"

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
    - 不继承业务基类，作为无际整机网络配置数据使用。
    """

    base_control_ip: str = "192.168.100.60"
    "基础控制工控机 IP，qmlinker 二次开发接口连接该地址。"

    orin_ip: str = "192.168.100.70"
    "Orin 模组 IP，用于 SSH 登录与相机、边缘计算等非 qmlinker 控制链路。"

    qmlinker: WujiQmlinkerConfig = WujiQmlinkerConfig()
    "qmlinker SDK 连接配置。"


# endregion


# region 配置

SUPPORTED_WUJI_QMLINKER_ENABLE_MODULES: tuple[WujiQmlinkerEnableModuleName, ...] = (
    "body",
    "head",
    "left_arm",
    "right_arm",
)
"当前 qmlinker 可真实读写使能状态的整机模块。手与 AGV 状态由各自子模块描述。"


# endregion


# region 主入口

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
    qmlinker_config = WujiQmlinkerConfig(
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
