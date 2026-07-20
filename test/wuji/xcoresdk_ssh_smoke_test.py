#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdk.xcoresdk import xCoreSDK_python

# SSH 本地转发地址；5050/4567 由 SDK 使用固定端口访问。
DEFAULT_ROBOT_IP = "127.0.0.1"


def main(robot_ip: str = DEFAULT_ROBOT_IP) -> int:
    """创建机器人对象并读取控制器基本信息。"""

    logger.info("开始创建机器人对象，robot_ip={}", robot_ip)
    try:
        robot = xCoreSDK_python.xMateErProRobot(robot_ip)
        logger.success("机器人对象创建成功，robot_ip={}", robot_ip)
        ec: dict[str, object] = {}
        robot_info = robot.robotInfo(ec)
        if ec.get("ec", 0) != 0:
            logger.warning("读取机器人信息失败，robot_ip={}，ec={}", robot_ip, ec)
            return 1
        logger.success(
            "读取机器人信息成功，robot_ip={}，type={}，uid={}，version={}",
            robot_ip,
            robot_info.type,
            robot_info.id,
            robot_info.version,
        )
        return 0
    except Exception:
        logger.exception("机器人连接测试失败，robot_ip={}", robot_ip)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
