from __future__ import annotations

from typing import cast

from qmlinker import QMLift, QMWaist
from src.arm.wuji_arm_protocol import WUJI_BODY_AXIS_LIMITS

from src.wuji.client_base import WujiQmlinkerBaseClient


# region body 客户端


class WujiBodyClient:
    """无际 body 客户端。

    职责边界：
    - 直接封装 `QMLift` 和 `QMWaist`，负责升降与腰部俯仰的读写。
    - 不负责 GUI 状态同步、订阅调度或其他设备的使能逻辑。

    设计思想：
    - body 在项目语义里由两个独立执行器组成，因此这里显式持有两个 SDK 对象。
    - 只把项目侧常用的毫米/角度接口整理出来，避免上层重复处理 proto 细节。

    生命周期：
    - 依赖 `WujiQmlinkerBaseClient` 的 channel 生命周期。
    - 不持有线程，调用完成即可释放上下文。

    继承关系：
    - 不继承业务基类，避免把 lift 和 waist 强行捆成动态多态对象。
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        """创建 body 客户端。

        Parameters
        ----------
        base_client:
            qmlinker 基础客户端，负责提供底层 gRPC channel。

        Notes
        -----
        本类不拥有 channel 生命周期，关闭连接仍由 `base_client` 调用方负责。
        """

        self._lift = QMLift(base_client.channel)
        self._waist = QMWaist(base_client.channel)

    @property
    def lift(self) -> QMLift:
        """返回底层升降 SDK 对象。

        Returns
        -------
        QMLift
            qmlinker 升降执行器对象，暴露 SDK 原始接口。
        """

        return self._lift

    @property
    def waist(self) -> QMWaist:
        """返回底层腰部 SDK 对象。

        Returns
        -------
        QMWaist
            qmlinker 腰部执行器对象，暴露 SDK 原始接口。
        """

        return self._waist

# endregion
