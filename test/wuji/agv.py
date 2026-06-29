from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from qmlinker import create_channel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import AGV_HOST, DEFAULT_PORT, create_wuyou_channel, stop_ssh_process  # noqa: E402
from src.wuji.agv_client import WujiAgvClient  # noqa: E402
from src.wuji.qmlinker_session import WujiQmlinkerSession  # noqa: E402
from network_discovery import get_cached_orin_host  # noqa: E402

SSH_ALIAS = "orin"
QMLINKER_HOST = get_cached_orin_host()
# QMLINKER_HOST = "192.168.1.102"


def _smoke_direct_channel() -> None:
    """验证测试脚本直连路径可用。"""

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT, AGV_HOST)
    client = WujiAgvClient(qmlinker_channel)
    try:
        runtime_info = client.get_runtime_info()
        logger.info("AGV 直连路径运行信息 {}", runtime_info)
        logger.info("AGV 直连路径使能状态 {}", client.get_enable())
        logger.success("AGV 直连路径冒烟通过")
    finally:
        stop_ssh_process(ssh_process)
        logger.info("AGV 直连路径冒烟结束")


def _smoke_gui_session_path() -> None:
    """验证 GUI 当前使用的 session 转发路径可用。"""

    session = WujiQmlinkerSession(
        host=QMLINKER_HOST,
        port=DEFAULT_PORT,
        ssh_alias=SSH_ALIAS,
    )
    try:
        session.check_ready()
        logger.info("GUI session 摘要 {}", session.debug_connection_summary())
        client = WujiAgvClient(
            create_channel(session.move_base_target),
            request_timeout_s=session.request_timeout_s,
        )
        runtime_info = client.get_runtime_info()
        logger.info("AGV GUI 路径运行信息 {}", runtime_info)
        logger.info("AGV GUI 路径使能状态 {}", client.get_enable())
        logger.success("AGV GUI 路径冒烟通过")
    finally:
        session.close()
        logger.info("AGV GUI 路径冒烟结束")


def main() -> None:
    """验证 AGV 直连路径与 GUI session 路径均可用。"""

    _smoke_direct_channel()
    _smoke_gui_session_path()


if __name__ == "__main__":
    main()
