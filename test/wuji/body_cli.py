from __future__ import annotations

import argparse
from calendar import c
import logging
import os
import sys
from pathlib import Path
import time

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
    
    while True:
        print("控制举升：lift")
        print("控制腰部：waist")
        value=input("请输入指令: ")
        if value=="q":
            break
        if value=="lift":
            lift=body_client.lift
            lift.set_enable(True)
            print(f"当前升降高度: {lift.get_lift_height()},使能: {lift.get_enable()}")
            value=input("请输入升降高度（mm）: ")
            if value.isdigit():
                lift.set_lift_height(float(value))
                time.sleep(2)
                print(f"当前升降高度: {lift.get_lift_height()}")
            if value=="q":
                continue
        elif value=="waist":
            waist=body_client.waist
            waist.set_enable(True)
            print(f"当前腰部俯仰角度: {waist.get_waist_pitch()},使能: {waist.get_enable()}")
            value=input("请输入腰部俯仰角度（deg）: ")
            if value.isdigit():
                waist.set_waist_pitch(float(value))
                time.sleep(2)
                print(f"当前腰部俯仰角度: {waist.get_waist_pitch()}")
            if value=="q":
                continue
        else:
            continue
        waist=body_client.waist
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
