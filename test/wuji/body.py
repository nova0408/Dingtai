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
    
from src.wuji.body_client import WujiBodyClient


from src.wuji.client_base import WujiQmlinkerBaseClient  
from src.wuji.protocol import WujiQmlinkerConfig  
# region 主入口


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """读取身体当前状态。"""

    
    base = WujiQmlinkerBaseClient(WujiQmlinkerConfig(host="192.168.100.60", port=50062))
    body_client = WujiBodyClient(base)
    
    try:
        logger.info("身体冒烟测试")
        lift=body_client.lift
        logger.info(f"开始测试 lift")
        logger.info(f"初始 enable: {lift.get_enable()}")
        lift.set_enable(False)
        logger.info(f"disable : {lift.get_enable()}")
        lift.set_enable(True)
        logger.info(f"enable: {lift.get_enable()}")
        logger.info(f"lift height: {lift.get_lift_height()}")
        
        logger.info(f"开始测试 waist")
        waist=body_client.waist
        logger.info(f"初始 enable: {waist.get_enable()}")
        waist.set_enable(False)
        logger.info(f"disable : {waist.get_enable()}")
        waist.set_enable(True)
        logger.info(f"enable: {waist.get_enable()}")
        logger.info(f"waist pitch: {waist.get_waist_pitch()}")
        logger.success("无际身体信息冒烟通过")
    finally:
        logger.info("无际身体信息冒烟结束")
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
