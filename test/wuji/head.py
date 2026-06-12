from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_orin_channel, stop_ssh_process  # noqa: E402
from src.wuji.head_client import WujiHeadClient  # noqa: E402


def main() -> None:
    """读取头部当前状态。"""

    ssh_process, qmlinker_channel = create_orin_channel(DEFAULT_PORT)
    head_client = WujiHeadClient(qmlinker_channel)
    try:
        logger.info("头部冒烟测试")
        logger.info("头部初始使能 {}", head_client.get_enable())
        head_client.set_enable(False)
        logger.info("头部关闭后使能 {}", head_client.get_enable())
        head_client.set_enable(True)
        logger.info("头部打开后使能 {}", head_client.get_enable())

        yaw = float(head_client.get_head_yaw() or 0.0)
        pitch = float(head_client.get_head_pitch() or 0.0)
        logger.info("头部初始 yaw {} deg", yaw)
        logger.info("头部初始 pitch {} deg", pitch)

        head_client.set_head_yaw(yaw + 1.0)
        head_client.set_head_pitch(pitch + 1.0)
        logger.info("头部调整后 yaw {} deg", head_client.get_head_yaw())
        logger.info("头部调整后 pitch {} deg", head_client.get_head_pitch())

        head_client.set_head_yaw(yaw)
        head_client.set_head_pitch(pitch)
        logger.info("头部恢复后 yaw {} deg", head_client.get_head_yaw())
        logger.info("头部恢复后 pitch {} deg", head_client.get_head_pitch())
        logger.success("无际头部信息冒烟通过")
    finally:
        stop_ssh_process(ssh_process)
        logger.info("无际头部信息冒烟结束")


if __name__ == "__main__":
    main()
