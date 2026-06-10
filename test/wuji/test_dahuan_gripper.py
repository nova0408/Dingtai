from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.dahuan_gripper_client import DahuanGripperClient
from src.wuji.protocol import WujiQmlinkerConfig

DEFAULT_REQUEST_TIMEOUT_S = 10.0
"夹爪冒烟超时，单位 s。"


@dataclass(frozen=True, slots=True)
class GripperSmokeConfig:
    """大寰夹爪冒烟测试配置。"""

    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    "单次远端命令超时时间，单位 s。"


# region 主入口


def run_gripper_smoke(config: GripperSmokeConfig) -> None:
    """按新 qmlinker SDK 读取并控制左手夹爪。"""

    base_client = WujiQmlinkerBaseClient(WujiQmlinkerConfig(request_timeout_s=float(config.request_timeout_s)))
    gripper = DahuanGripperClient(base_client)

    try:
        info = gripper.get_gripper_info()
        logger.info("gripper online={} calibrated={} enabled={}", info.online, info.calibrated, info.enabled)
        logger.info("gripper position={} speed={} force={} state={}", info.position, info.speed, info.force, info.grip_state)

        logger.info("gripper enable -> True")
        gripper.set_enable(True)
        logger.info("gripper speed -> 10")
        gripper.set_speed(10)
        logger.info("gripper force -> 10")
        gripper.set_force(10)
        logger.info("gripper position -> 1000")
        gripper.set_pos(1000)
        logger.info("gripper calibrate -> {}", gripper.calibrate())
        logger.success("gripper smoke passed")
    finally:
        base_client.close()


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """读取夹爪当前状态并验证基础控制链路。"""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    run_gripper_smoke(GripperSmokeConfig(request_timeout_s=float(request_timeout_s)))


# endregion


# region CLI


def _parse_cli() -> float:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取并控制无际夹爪")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args()
    return float(args.request_timeout_s)


if __name__ == "__main__":
    main(request_timeout_s=_parse_cli() if len(sys.argv) > 1 else DEFAULT_REQUEST_TIMEOUT_S)


# endregion
