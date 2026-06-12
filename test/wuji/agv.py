from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.wuji.agv_client import WujiAgvClient
from src.wuji.client_base import WujiQmlinkerBaseClient

def main() -> None:
    """验证 AGV 状态读取链路可用。"""

    client = WujiAgvClient(WujiQmlinkerBaseClient())
    try:
        values = client.get_base_status()
        print(values)
        print(client.get_enable())
        logger.success("AGV 信息读取冒烟通过")
    finally:
        logger.info("AGV 信息读取冒烟结束")
    os._exit(0)


def _parse_cli() -> float:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取 AGV 状态")
    args = parser.parse_args()
    return float(args.request_timeout_s)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        main()
