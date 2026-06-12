from __future__ import annotations

from qmlinker import QMMoveBase, create_channel

from src.wuji.client_base import WujiQmlinkerBaseClient


# region agv 客户端


class WujiAgvClient(QMMoveBase):
    """无际 AGV 客户端。

    职责边界：
    - 直接继承 `QMMoveBase`，负责底盘状态读取、使能与移动控制。
    - 不负责 GUI 的卡片展示逻辑，也不负责电量、里程等字段格式化。

    设计思想：
    - 底盘是独立设备域，项目侧只需要把 SDK 绑定到统一 channel。
    - 具体的控制按钮、展示格式和接口适配留给上层 GUI 或门面类处理。

    生命周期：
    - 依赖基础客户端 channel。
    - 不持有后台 worker。

    继承关系：
    - 直接继承 `QMMoveBase`。
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        """创建 AGV 客户端。"""

        super().__init__(create_channel(base_client.move_base_target))


# endregion
