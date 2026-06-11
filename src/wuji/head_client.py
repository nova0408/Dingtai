from __future__ import annotations

from google.protobuf import empty_pb2
from qmlinker import QMHead
from qmlinker.grpc_py import head_pb2

from src.wuji.client_base import WujiQmlinkerBaseClient


# region head 客户端


class WujiHeadClient(QMHead):
    """无际头部客户端。

    职责边界：
    - 直接继承 `QMHead`，负责头部 yaw 读取、使能读取和角度控制。
    - 不负责 body、机械臂、相机或 GUI 逻辑。

    设计思想：
    - 用最薄的项目侧封装保留协议一致性，同时避免把 SDK 再包装成一层厚 facade。
    - 只保留必要的统一超时控制和项目语义方法名。

    生命周期：
    - 依赖基础客户端的 channel。
    - 不额外持有后台资源。

    继承关系：
    - 直接继承 `QMHead`。
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        """创建头部客户端。"""

        super().__init__(base_client.channel)


# endregion
