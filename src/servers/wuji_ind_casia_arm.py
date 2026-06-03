from __future__ import annotations

from dataclasses import dataclass

from src.servers.arm import ArmConfig, ArmServer, default_casia_arm_config
from src.servers.body import BodyConfig, BodyServer, default_body_config

# region 数据结构


@dataclass(frozen=True, slots=True)
class WujiIndCasiaArmConfig:
    """无际工业版（CASIA 机械臂）整机配置。

    职责边界：
    - 组合非机械臂本体配置与单只 CASIA 机械臂配置。
    - 不包含底座、手部末端工具、相机、gRPC 绑定或硬件连接。

    设计思想：
    - 左右臂是镜像安装，使用同一 `ArmConfig` 创建两个 `ArmServer`。
    - 整合层只表达整机结构关系，不复制单臂关节定义。

    生命周期：
    - 可跨线程读取，不持有硬件连接。

    继承关系：
    - 不继承业务基类，保持整机配置职责。
    """

    body: BodyConfig
    "非机械臂本体配置。"

    arm: ArmConfig
    "单只 CASIA 机械臂配置，左右臂共享。"


# endregion


# region 配置


def default_wuji_ind_casia_arm_config() -> WujiIndCasiaArmConfig:
    """创建无际工业版 CASIA 机械臂整机默认配置。

    Returns
    -------
    WujiIndCasiaArmConfig
        整机默认配置。
    """

    return WujiIndCasiaArmConfig(body=default_body_config(), arm=default_casia_arm_config())


# endregion


# region 主入口


class WujiIndCasiaArmServer:
    """无际工业版 CASIA 机械臂整合服务。

    职责边界：
    - 组合非机械臂本体、左机械臂和右机械臂三个控制对象。
    - 提供整机级访问入口，让上层明确选择 `body`、`left_arm` 或 `right_arm`。
    - 不负责底座、手部末端工具、gRPC 绑定、硬件通信、轨迹规划或碰撞检测。

    设计思想：
    - 左右臂使用同一单臂配置，避免镜像机械臂重复定义两套关节限制。
    - 镜像关系通过 `ArmServer.mirror_sign` 记录，后续接入安装位姿或运动学时在整合层使用。

    生命周期：
    - 实例可在单线程服务流程中复用。
    - 当前类不持有硬件连接、线程、协程或文件句柄。
    - 多线程场景需要在外层增加锁或消息队列。

    继承关系：
    - 不继承业务基类，便于后续按真实协议适配。
    """

    def __init__(self, config: WujiIndCasiaArmConfig | None = None) -> None:
        """初始化无际工业版 CASIA 机械臂整合服务。

        Parameters
        ----------
        config:
            整机配置。为 `None` 时使用默认配置。
        """

        self._config = config or default_wuji_ind_casia_arm_config()
        self.body = BodyServer(self._config.body)
        self.left_arm = ArmServer(side="left", config=self._config.arm, mirror_sign=1)
        self.right_arm = ArmServer(side="right", config=self._config.arm, mirror_sign=-1)

    def arms(self) -> tuple[ArmServer, ArmServer]:
        """获取左右机械臂服务。

        Returns
        -------
        tuple[ArmServer, ArmServer]
            左臂和右臂服务，顺序为 `(left_arm, right_arm)`。
        """

        return self.left_arm, self.right_arm


# endregion
