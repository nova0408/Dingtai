from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import create_wuyou_channel, stop_ssh_process
from src.wuji.head_client import WujiHeadClient


def _require_bool_true(value: object, label: str) -> None:
    """校验布尔结果必须为真。"""

    if bool(value):
        return
    raise RuntimeError(f"{label} failed: {value!r}")


def _require_float_value(value: object, label: str) -> float:
    """校验头部角度读取成功。"""

    if value is None:
        raise RuntimeError(f"{label} returned None")
    return float(value)


def main() -> None:
    """按官方 head 示例验证 wuyou qmlinker 头部链路。"""

    ssh_process, qmlinker_channel = create_wuyou_channel()
    head_client = WujiHeadClient(qmlinker_channel)
    try:
        logger.info("头部冒烟测试")
        set_enable_result = head_client.set_enable(True)
        logger.info("头部打开使能返回 {}", set_enable_result)
        _require_bool_true(set_enable_result, "head.set_enable(True)")

        enabled = head_client.get_enable()
        logger.info("头部当前使能 {}", enabled)
        _require_bool_true(enabled, "head.get_enable()")

        pitch = _require_float_value(head_client.get_head_pitch(), "head.get_head_pitch()")
        yaw = _require_float_value(head_client.get_head_yaw(), "head.get_head_yaw()")
        logger.info("头部当前 pitch {:.3f} deg", pitch)
        logger.info("头部当前 yaw {:.3f} deg", yaw)
        logger.success("无际头部信息冒烟通过")
    finally:
        stop_ssh_process(ssh_process)
        logger.info("无际头部信息冒烟结束")


if __name__ == "__main__":
    main()
