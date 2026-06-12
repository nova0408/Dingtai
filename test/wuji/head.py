from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from flask.cli import F
from loguru import logger
DEFAULT_REQUEST_TIMEOUT_S = 3.0
"身体和头部信息读取超时，单位 s。"

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
    
from src.wuji.head_client import WujiHeadClient


from src.wuji.client_base import WujiQmlinkerBaseClient  
from src.wuji.protocol import WujiQmlinkerConfig  
# region 主入口


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """读取头部当前状态。"""

    logger.info("body/head smoke stage: import ready")


    logger.info("body/head smoke stage: creating client")
    
    base = WujiQmlinkerBaseClient(WujiQmlinkerConfig(host="192.168.100.60", port=50062))
    head_client = WujiHeadClient(base)
    
    try:
        logger.info("头部冒烟测试")
        logger.info("头部初始 enable: {}", head_client.get_enable())
        head_client.set_enable(False)
        logger.info("头部set disable: {}", head_client.get_enable())
        head_client.set_enable(True)
        logger.info("头部set enable: {}", head_client.get_enable())
        yaw=head_client.get_head_yaw()
        logger.info("头部 yaw 轴 {} deg", yaw)
        head_client.set_head_yaw(yaw+1.0)
        logger.info("头部 yaw 轴 {} deg", head_client.get_head_yaw())
        pitch=head_client.get_head_pitch()
        logger.info("头部 pitch 轴 {} deg", pitch)
        head_client.set_head_pitch(pitch+1.0)
        logger.info("头部 pitch 轴 {} deg", head_client.get_head_pitch())
        head_client.set_head_yaw(yaw-1.0)
        logger.info("头部 yaw 轴 {} deg", head_client.get_head_yaw())
        head_client.set_head_pitch(pitch-1.0)
        logger.info("头部 pitch 轴 {} deg", head_client.get_head_pitch())
        logger.success("无际头部信息冒烟通过")
    finally:
        logger.info("无际头部信息冒烟结束")
    os._exit(0)


# endregion


# region CLI


def _parse_cli() -> float:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取无际身体和头部状态信息")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args()
    return float(args.request_timeout_s)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(request_timeout_s=_parse_cli())
    else:
        main()


# endregion
