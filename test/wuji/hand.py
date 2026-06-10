from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from qmlinker import QMHand

from src.wuji.right_hand_client import WujiRightHandClient
from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.protocol import WujiQmlinkerConfig

DEFAULT_REQUEST_TIMEOUT_S = 3.0
"右手灵巧手冒烟超时，单位 s。"


@dataclass(frozen=True, slots=True)
class HandSmokeConfig:
    """右手灵巧手冒烟测试配置。"""

    host: str = "192.168.100.60"
    "qmlinker 主机地址。"

    port: int = 50062
    "qmlinker 端口。"

    hand_id: int = QMHand.HAND_RIGHT
    "手部类型，固定右手。"

    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    "单次请求超时，单位 s。"

    def target(self) -> str:
        """返回 qmlinker 连接目标。"""

        return f"{self.host}:{self.port}"


# region 主入口


def run_hand_smoke(config: HandSmokeConfig) -> None:
    """按新 qmlinker SDK 读取并控制右手灵巧手。"""

    base_client = WujiQmlinkerBaseClient(WujiQmlinkerConfig(host=config.host, port=config.port))
    hand = WujiRightHandClient(base_client)

    logger.info("hand smoke ready: host={} hand_id={}", config.host, config.hand_id)
    logger.info("hand enable before={}", hand.get_enable())

    hand_info = hand.get_hand_info()
    if hand_info is None:
        raise RuntimeError("hand info unavailable")
    actuator_count = int(hand_info["actuator_count"])
    if actuator_count != 11:
        raise RuntimeError(f"unexpected actuator count: {actuator_count}")

    current_state = hand.get_hand_state(include_tactile=False)
    if current_state is None:
        raise RuntimeError("hand state unavailable")
    current_positions = [float(actuator["position"]) for actuator in current_state["actuators"]]
    if len(current_positions) != 11:
        raise RuntimeError(f"unexpected actuator state count: {len(current_positions)}")

    logger.info("hand set_hand_state current pose => {}", hand.set_hand_state(current_positions))
    logger.info("hand enable after set={}", hand.get_enable())
    logger.success("hand smoke passed")

    base_client.close()


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
    """读取右手当前状态并验证基础控制链路。"""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    config = HandSmokeConfig(request_timeout_s=float(request_timeout_s))
    run_hand_smoke(config)


# endregion


# region CLI


def _parse_cli() -> float:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取并控制无际右手")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    args = parser.parse_args()
    return float(args.request_timeout_s)


if __name__ == "__main__":
    main(request_timeout_s=_parse_cli() if len(sys.argv) > 1 else DEFAULT_REQUEST_TIMEOUT_S)


# endregion
