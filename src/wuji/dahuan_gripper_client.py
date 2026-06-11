from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from src.wuji.client_base import WujiQmlinkerBaseClient

from qmlinker import QMGripper,GripperInfo



class DahuanGripperClient(QMGripper):
    """大寰夹爪远程适配器。

    职责边界：
    - 通过 `WujiQmlinkerBaseClient` 共享的 Orin SSH 上下文执行夹爪脚本。
    - 不创建 GUI 控件，不维护长连接线程。

    设计思想：
    - 夹爪在现场仍然通过 Orin 侧 `qmlinker` Python 环境运行，客户端只负责把脚本收口成
      项目侧稳定方法。
    - 只暴露新 SDK 直接支持的使能、校准、速度、力、位置与状态接口。

    生命周期：
    - 依赖外部传入的 `WujiQmlinkerBaseClient`。
    - 不持有后台线程。

    继承关系：
    - 继承QMGripper
    """

    def __init__(self, base_client: WujiQmlinkerBaseClient) -> None:
        super().__init__(base_client.channel)
    
