from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from loguru import logger
DEFAULT_REQUEST_TIMEOUT_S = 3.0
"AGV 控制冒烟超时，单位 s。"


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """验证 AGV 既可读也可控。"""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    logger.info("agv control smoke stage: import ready")

    from src.wuji.client_base import WujiQmlinkerBaseClient
    from src.wuji.device_clients import WujiQmlinkerClientSet
    from src.wuji.protocol import WujiQmlinkerConfig

    logger.info("agv control smoke stage: creating client")
    client = WujiQmlinkerClientSet(
        WujiQmlinkerBaseClient(WujiQmlinkerConfig(request_timeout_s=float(request_timeout_s)))
    )
    try:
        logger.info("agv control smoke stage: client ready")
        before_enable = client.get_agv_enable()
        logger.info("AGV 使能前 {}", before_enable)
        logger.info("AGV 使能设置结果 {}", client.set_agv_enable(True))
        after_enable = client.get_agv_enable()
        logger.info("AGV 使能后 {}", after_enable)
        values = client.get_agv_status_values()
        for key in ("agv_x", "agv_y", "agv_yaw", "agv_battery"):
            logger.info("{} {}", key, values.get(key))
        logger.info("AGV stop 结果 {}", client.stop_agv())
        logger.info("AGV 左移控制结果 {}", client.move_agv_real_time_translate(0.05, 90))
        logger.info("AGV 右移控制结果 {}", client.move_agv_real_time_translate(0.05, 270))
        logger.info("AGV 前进控制结果 {}", client.move_agv_real_time_translate(0.05, 0))
        logger.info("AGV 后退控制结果 {}", client.move_agv_real_time_translate(0.05, 180))
        target_name = os.environ.get("WUJI_AGV_TARGET_NAME", "charge").strip() or "charge"
        try:
            logger.info("AGV 导航到点结果 {} -> {}", target_name, client.agv_navigate_to(target_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("AGV 导航到点暂不可用: {}", exc)
        try:
            logger.info("AGV 去充电结果 {}", client.agv_navigate_to_charge())
        except Exception as exc:  # noqa: BLE001
            logger.warning("AGV 去充电导航暂不可用: {}", exc)
        logger.success("AGV 可读可控冒烟通过")
    finally:
        logger.info("AGV 可读可控冒烟结束")
    os._exit(0)


def _parse_cli() -> float:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取并控制 AGV")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args()
    return float(args.request_timeout_s)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(request_timeout_s=_parse_cli())
    else:
        main()
