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
    """导航到目标站点并等待 runtime info 明确报告到位。"""

    client.navigate_to(target)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = str(client.get_runtime_info().get("agv_navi_status", "")).lower()
        logger.info("AGV 导航状态 target={} status={}", target, status)
        if status in {"arrived", "succeeded", "success", "idle", "none", "0"}:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"AGV 导航到位超时：target={target}")
