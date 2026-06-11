from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from qmlinker import QMHand

DEFAULT_REQUEST_TIMEOUT_S = 3.0
"右手灵巧手冒烟超时，单位 s。"

DEFAULT_A0_DELTA = 0.1
"a0 轴小幅运动幅度，单位 归一化比例。"


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

    a0_delta: float = DEFAULT_A0_DELTA
    "a0 轴小幅运动幅度，单位 归一化比例。"

    def target(self) -> str:
        """返回 qmlinker 连接目标。"""

        return f"{self.host}:{self.port}"


def _clamp_normalized(value: float) -> float:
    """将数值限制到 0 到 1 的归一化区间。"""

    if not math.isfinite(value):
        raise ValueError(f"目标值必须是有限数，当前为 {value!r}")
    return max(0.0, min(1.0, float(value)))


def _append_repo_root_to_sys_path() -> None:
    """允许直接运行 test 下的脚本。"""

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "src").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_append_repo_root_to_sys_path()


from src.wuji.right_hand_client import WujiRightHandClient  # noqa: E402
from src.wuji.client_base import WujiQmlinkerBaseClient  # noqa: E402
from src.wuji.protocol import WujiQmlinkerConfig  # noqa: E402


# region 主入口


def run_hand_smoke(config: HandSmokeConfig) -> None:
    """按新 qmlinker SDK 读取并控制右手灵巧手。

    这次冒烟验证只做 `right_hand_a0` 的小幅运动，避免大范围动作。
    """

    base_client = WujiQmlinkerBaseClient(WujiQmlinkerConfig(host=config.host, port=config.port))
    try:
        hand = WujiRightHandClient(base_client)

        logger.info("hand smoke ready: host={} hand_id={}", config.host, config.hand_id)
        enable_before = bool(hand.get_enable())
        logger.info("hand enable before={}", enable_before)

        if not enable_before:
            if not hand.set_enable(True):
                raise RuntimeError("hand enable failed")
            time.sleep(0.2)
            logger.info("hand enable after set={}", hand.get_enable())

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

        a0_before = float(current_positions[0])
        a0_target = _clamp_normalized(a0_before + float(config.a0_delta))
        if a0_target == a0_before:
            a0_target = _clamp_normalized(a0_before - float(config.a0_delta))

        logger.info("hand a0 before={:.3f} target={:.3f}", a0_before, a0_target)
        if not hand.set_right_hand_axis(0, a0_target):
            raise RuntimeError(f"hand a0 move failed: target={a0_target:.3f}")
        time.sleep(0.5)

        moved_state = hand.get_hand_state(include_tactile=False)
        if moved_state is None:
            raise RuntimeError("hand state unavailable after a0 move")
        moved_positions = [float(actuator["position"]) for actuator in moved_state["actuators"]]
        moved_a0 = float(moved_positions[0])
        logger.info("hand a0 after move={:.3f}", moved_a0)

        if not hand.set_right_hand_axis(0, a0_before):
            raise RuntimeError(f"hand a0 restore failed: target={a0_before:.3f}")
        time.sleep(0.5)

        restored_state = hand.get_hand_state(include_tactile=False)
        if restored_state is None:
            raise RuntimeError("hand state unavailable after a0 restore")
        restored_positions = [float(actuator["position"]) for actuator in restored_state["actuators"]]
        restored_a0 = float(restored_positions[0])

        logger.info("hand a0 restored={:.3f}", restored_a0)
        logger.success("hand smoke passed")
    finally:
        base_client.close()


def main(request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S, a0_delta: float = DEFAULT_A0_DELTA) -> None:
    """读取右手当前状态并验证基础控制链路。"""

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    config = HandSmokeConfig(request_timeout_s=float(request_timeout_s), a0_delta=float(a0_delta))
    run_hand_smoke(config)


# endregion


# region CLI


def _parse_cli() -> tuple[float, float]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="读取并控制无际右手")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--a0-delta", type=float, default=DEFAULT_A0_DELTA)
    args = parser.parse_args()
    return float(args.request_timeout_s), float(args.a0_delta)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        request_timeout_s, a0_delta = _parse_cli()
        main(request_timeout_s=request_timeout_s, a0_delta=a0_delta)
    else:
        main()


# endregion
