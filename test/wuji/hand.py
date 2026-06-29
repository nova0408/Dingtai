from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_wuyou_channel, stop_ssh_process  # noqa: E402
from src.wuji.right_hand_client import WujiRightHandClient  # noqa: E402

DEFAULT_A0_DELTA = 0.1


def _clamp_normalized(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main() -> None:
    """读取右手当前状态并验证基础控制链路。"""

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT)
    hand = WujiRightHandClient(qmlinker_channel)
    try:
        logger.info("右手冒烟测试")

        if not hand.get_enable():
            if not hand.set_enable(True):
                raise RuntimeError("右手使能失败")
            time.sleep(0.2)

        hand_info = hand.get_hand_info()
        if hand_info is None:
            raise RuntimeError("右手信息不可用")
        actuator_count = int(hand_info["actuator_count"])
        if actuator_count != 11:
            raise RuntimeError(f"右手 actuator 数量异常: {actuator_count}")

        current_state = hand.get_hand_state(include_tactile=False)
        if current_state is None:
            raise RuntimeError("右手状态不可用")
        current_positions = [float(actuator["position"]) for actuator in current_state["actuators"]]
        if len(current_positions) != 11:
            raise RuntimeError(f"右手状态数量异常: {len(current_positions)}")

        a0_before = float(current_positions[0])
        a0_target = _clamp_normalized(a0_before + DEFAULT_A0_DELTA)
        if a0_target == a0_before:
            a0_target = _clamp_normalized(a0_before - DEFAULT_A0_DELTA)

        logger.info("右手 a0 初始值 {} 目标值 {}", a0_before, a0_target)
        if not hand.set_right_hand_axis(0, a0_target):
            raise RuntimeError("右手 a0 运动失败")
        time.sleep(0.5)

        moved_state = hand.get_hand_state(include_tactile=False)
        if moved_state is None:
            raise RuntimeError("右手运动后状态不可用")
        moved_a0 = float(moved_state["actuators"][0]["position"])
        logger.info("右手 a0 运动后 {}", moved_a0)

        restore_positions = list(current_positions)
        restore_positions[0] = a0_before
        if not hand.set_hand_state(restore_positions):
            raise RuntimeError("右手 a0 恢复失败")
        time.sleep(0.5)

        restored_state = hand.get_hand_state(include_tactile=False)
        if restored_state is None:
            raise RuntimeError("右手恢复后状态不可用")
        restored_a0 = float(restored_state["actuators"][0]["position"])
        logger.info("右手 a0 恢复后 {}", restored_a0)
        logger.success("右手冒烟测试通过")
    finally:
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
