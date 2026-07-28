from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_wuyou_channel, stop_ssh_process
from src.wuji.head_client import WujiHeadClient


def main() -> None:
    """验证头部基础控制链路。"""

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT)
    head_client = WujiHeadClient(qmlinker_channel)
    try:
        head_client.set_enable(True)
        if not head_client.get_enable():
            raise RuntimeError("头部使能失败")

        yaw_deg = head_client.get_head_yaw()
        pitch_deg = head_client.get_head_pitch()
        logger.info("头部当前 yaw {:.1f} deg", yaw_deg)
        logger.info("头部当前 pitch {:.1f} deg", pitch_deg)

        head_client.set_head_yaw(yaw_deg)
        head_client.set_head_pitch(pitch_deg)
        logger.success("无际头部冒烟测试通过")
    finally:
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
