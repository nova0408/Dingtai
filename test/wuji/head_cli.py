from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import create_wuyou_channel, stop_ssh_process  # noqa: E402
from src.wuji.head_client import WujiHeadClient  # noqa: E402

DEFAULT_REQUEST_TIMEOUT_S = 3.0  # 请求超时时间，单位 s
DEFAULT_STEP_DEG = 5.0  # 默认单次调整步长，单位 deg


def _to_float(value: object) -> float:
    """将头部返回值转换为浮点数。"""

    return float(value if value is not None else 0.0)


def _read_head_state(head: WujiHeadClient) -> tuple[bool, float, float]:
    """读取头部当前状态。"""

    enabled = bool(head.get_enable())
    yaw_deg = _to_float(head.get_head_yaw())
    pitch_deg = _to_float(head.get_head_pitch())
    return enabled, yaw_deg, pitch_deg


def _control_head(head: WujiHeadClient) -> None:
    """交互式控制头部 yaw/pitch。"""

    head.set_enable(True)
    enabled, yaw_deg, pitch_deg = _read_head_state(head)
    logger.info("head 当前使能 {}", enabled)
    logger.info("head 当前 yaw {:.1f} deg", yaw_deg)
    logger.info("head 当前 pitch {:.1f} deg", pitch_deg)

    while True:
        print()
        print("控制头部：")
        print("  yaw   : 控制偏航")
        print("  pitch : 控制俯仰")
        print("  q     : 返回")
        mode = input("请选择轴: ").strip().lower()
        if mode == "q":
            return
        if mode not in {"yaw", "pitch"}:
            logger.warning("未知轴 {}", mode)
            continue

        current_value = yaw_deg if mode == "yaw" else pitch_deg
        value_text = input(f"请输入目标 {mode} 角度（deg），直接回车表示 +{DEFAULT_STEP_DEG:.1f} deg: ").strip().lower()
        if value_text == "q":
            return
        if value_text == "":
            target_deg = current_value + DEFAULT_STEP_DEG
        else:
            target_deg = float(value_text)

        logger.info("{} 当前 {:.1f} deg，目标 {:.1f} deg", mode, current_value, target_deg)
        if mode == "yaw":
            head.set_head_yaw(target_deg)
        else:
            head.set_head_pitch(target_deg)
        time.sleep(1.0)
        enabled, yaw_deg, pitch_deg = _read_head_state(head)
        logger.info("head 读回使能 {}", enabled)
        logger.info("head 读回 yaw {:.1f} deg", yaw_deg)
        logger.info("head 读回 pitch {:.1f} deg", pitch_deg)


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """头部交互式 CLI。"""

    logger.info("头部控制脚本启动，请先确认 wuyou qmlinker 连接正常。")
    logger.info("请求超时 {} s", request_timeout_s)

    ssh_process, qmlinker_channel = create_wuyou_channel()
    head_client = WujiHeadClient(qmlinker_channel)
    try:
        _control_head(head_client)
    finally:
        stop_ssh_process(ssh_process)
        os._exit(0)


def _parse_cli() -> float:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="控制无际头部 yaw/pitch")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args()
    return float(args.request_timeout_s)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(request_timeout_s=_parse_cli())
    else:
        main()
