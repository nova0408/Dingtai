"""AGV 导航命令与 runtime info 到位确认。"""

from __future__ import annotations

import time
from typing import Protocol

from loguru import logger


class AgvClient(Protocol):
    """循环服务所需的最小 AGV 客户端接口。"""

    def navigate_to(self, target_name: str) -> object:
        """下发站点导航命令。"""

        ...

    def get_runtime_info(self) -> dict[str, object]:
        """读取包含 agv_navi_status 的运行时信息。"""

        ...


def wait_until_arrived(client: AgvClient, target: str, timeout_s: float, poll_s: float) -> None:
    """导航到目标站点并等待 AGV 从运动状态恢复为空闲状态。

    Parameters
    ----------
    client:
        提供导航命令与运行状态查询的 AGV 客户端。
    target:
        AGV 地图导航点名称。
    timeout_s:
        等待本次导航结束的最长时间，单位 s。
    poll_s:
        查询运行状态的间隔，单位 s。

    Raises
    ------
    TimeoutError
        超时前未依次观察到 ``busy`` 和 ``idel`` 时抛出。

    Notes
    -----
    AGV 协议中 ``busy`` 表示正在导航，``idel`` 表示其余状态。必须先观察到
    ``busy`` 才接受后续的 ``idel``，避免把命令刚下发时尚未切换的空闲状态误判为到位。
    """

    client.navigate_to(target)
    deadline = time.monotonic() + timeout_s
    observed_busy = False
    while time.monotonic() < deadline:
        raw_status = str(client.get_runtime_info().get("agv_navi_status", "")).strip().lower()
        logger.info("AGV 导航状态 target={} raw_status={!r}", target, raw_status)
        if raw_status == "busy":
            observed_busy = True
        elif observed_busy and raw_status == "idel":
            logger.success("AGV 导航结束 target={} raw_status={!r}", target, raw_status)
            return
        time.sleep(poll_s)
    raise TimeoutError(f"AGV 导航到位超时：target={target}, observed_busy={observed_busy}")
