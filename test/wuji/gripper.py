from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import GRIPPER_PORT, create_orin_channel, stop_ssh_process  # noqa: E402
from src.wuji.dahuan_gripper_client import DahuanGripperClient  # noqa: E402


def main() -> None:
    """验证大寰夹爪基础控制链路。"""

    ssh_process, channel = create_orin_channel(GRIPPER_PORT)
    client = DahuanGripperClient(channel)
    try:
        status = client.get_status()
        logger.info("在线状态 {}", status.online)
        logger.info("校准状态 {}", status.calibrated)
        logger.info("使能状态 {}", status.enable)
        logger.info("当前位置 {}", status.position)
        logger.info("夹持状态码 {}", status.state)

        client.set_enable(True)
        logger.info("使能后状态 {}", client.get_status().enable)

        client.set_pos(1000)
        time.sleep(3.0)
        if client.get_status().position != 1000:
            raise RuntimeError("夹爪前进位置设置失败")

        client.set_pos(0)
        time.sleep(3.0)
        if client.get_status().position != 0:
            raise RuntimeError("夹爪回零位置设置失败")

        logger.success("大寰夹爪冒烟测试通过")
    finally:
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
