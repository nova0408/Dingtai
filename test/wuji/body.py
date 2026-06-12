from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_orin_channel, stop_ssh_process  # noqa: E402
from src.wuji.body_client import WujiBodyClient  # noqa: E402


def main() -> None:
    """读取身体当前状态。"""

    ssh_process, qmlinker_channel = create_orin_channel(DEFAULT_PORT)
    body_client = WujiBodyClient(qmlinker_channel)
    try:
        logger.info("身体冒烟测试")

        lift = body_client.lift
        logger.info("lift 初始使能 {}", lift.get_enable())
        lift.set_enable(False)
        logger.info("lift 关闭后使能 {}", lift.get_enable())
        lift.set_enable(True)
        logger.info("lift 打开后使能 {}", lift.get_enable())
        logger.info("lift 当前高度 {} mm", lift.get_lift_height())

        waist = body_client.waist
        logger.info("waist 初始使能 {}", waist.get_enable())
        waist.set_enable(False)
        logger.info("waist 关闭后使能 {}", waist.get_enable())
        waist.set_enable(True)
        logger.info("waist 打开后使能 {}", waist.get_enable())
        logger.info("waist 当前俯仰 {} deg", waist.get_waist_pitch())

        logger.success("无际身体信息冒烟通过")
    finally:
        stop_ssh_process(ssh_process)
        logger.info("无际身体信息冒烟结束")


if __name__ == "__main__":
    main()
