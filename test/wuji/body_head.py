from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from loguru import logger
DEFAULT_REQUEST_TIMEOUT_S = 3.0
"身体和头部信息读取超时，单位 s。"


# region 主入口


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """读取身体升降、腰部和头部当前状态。"""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    logger.info("body/head smoke stage: import ready")

    from src.wuji.client_base import WujiQmlinkerBaseClient
    from src.wuji.device_clients import WujiQmlinkerClientSet
    from src.wuji.protocol import WujiQmlinkerConfig

    logger.info("body/head smoke stage: creating client")
    client = WujiQmlinkerClientSet(
        WujiQmlinkerBaseClient(WujiQmlinkerConfig(request_timeout_s=float(request_timeout_s)))
    )
    try:
        logger.info("body/head smoke stage: client ready")
        body_z = client.get_body_z()
        body_ry = client.get_body_ry()
        head_yaw = client.get_head_yaw()
        logger.info("身体 z 轴 {} mm", body_z)
        logger.info("身体 Ry 轴 {} deg", body_ry)
        logger.info("头部 yaw 轴 {} deg", head_yaw)
        logger.success("无际身体和头部信息冒烟通过")
    finally:
        logger.info("无际身体和头部信息冒烟结束")
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
