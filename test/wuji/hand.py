from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, close_wuyou_channel, create_wuyou_channel, stop_ssh_process

from src.wuji.right_hand_client import WujiRightHandClient

# M6 冒烟测试的 0 号执行器位移，单位为归一化比例。
DEFAULT_A0_DELTA = 0.1
# 目标位置允许误差，单位为归一化比例。
DEFAULT_POSITION_TOLERANCE = 0.03
# 单次运动等待超时，单位 s。
DEFAULT_MOVE_TIMEOUT_S = 3.0
# 当前右手期望型号，用于避免把其他手型误当作 M6 执行运动测试。
DEFAULT_EXPECTED_MODEL_NAME = "CasiaHand-6D-5F"


def _clamp_normalized(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _read_positions(hand: WujiRightHandClient, actuator_count: int) -> list[float]:
    """读取并校验 M6 当前执行器位置。"""

    state = hand.get_hand_state(include_tactile=False)
    if state is None:
        raise RuntimeError("右手状态不可用")
    positions = [float(actuator["position"]) for actuator in state["actuators"]]
    if len(positions) != actuator_count:
        raise RuntimeError(
            f"右手状态数量与 hand_info 不一致: state={len(positions)} info={actuator_count}"
        )
    return positions


def _wait_for_axis_position(
    hand: WujiRightHandClient,
    actuator_count: int,
    actuator_id: int,
    target: float,
) -> list[float]:
    """等待指定执行器进入目标位置容差。"""

    deadline = time.monotonic() + DEFAULT_MOVE_TIMEOUT_S
    last_positions: list[float] = []
    while time.monotonic() < deadline:
        last_positions = _read_positions(hand, actuator_count)
        if abs(last_positions[actuator_id] - target) <= DEFAULT_POSITION_TOLERANCE:
            return last_positions
        time.sleep(0.1)
    current = last_positions[actuator_id] if last_positions else None
    raise RuntimeError(
        f"右手 a{actuator_id} 未在超时内到位: "
        f"target={target:.4f} current={current} timeout={DEFAULT_MOVE_TIMEOUT_S:.1f} s"
    )


def main() -> None:
    """读取 M6 当前状态，执行小幅运动并恢复初始位置。"""

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT)
    hand = WujiRightHandClient(qmlinker_channel)
    try:
        logger.info("M6 右手冒烟测试")

        if not hand.get_enable():
            if not hand.set_enable(True):
                raise RuntimeError("右手使能失败")
            time.sleep(0.2)

        hand_info = hand.get_hand_info()
        if hand_info is None:
            raise RuntimeError("右手信息不可用")
        model_name = str(hand_info["model_name"])
        logger.info("M6 右手型号 {}", model_name)
        if model_name != DEFAULT_EXPECTED_MODEL_NAME:
            raise RuntimeError(f"右手型号异常: {model_name}")
        actuator_count = int(hand_info["actuator_count"])
        runtime_actuator_count = hand.get_right_hand_actuator_count()
        if actuator_count != runtime_actuator_count:
            raise RuntimeError(
                f"右手信息与运行时规格数量不一致: info={actuator_count} runtime={runtime_actuator_count}"
            )
        logger.info(
            "M6 信息 model={} actuator_count={} names={}",
            model_name,
            actuator_count,
            hand_info["actuator_names"],
        )

        current_positions = _read_positions(hand, actuator_count)
        a0_before = float(current_positions[0])
        a0_target = _clamp_normalized(a0_before + DEFAULT_A0_DELTA)
        if a0_target == a0_before:
            a0_target = _clamp_normalized(a0_before - DEFAULT_A0_DELTA)

        logger.info("M6 a0 初始值 {} 目标值 {}", a0_before, a0_target)
        if not hand.set_right_hand_axis(0, a0_target):
            raise RuntimeError("M6 a0 运动指令下发失败")
        try:
            moved_positions = _wait_for_axis_position(hand, actuator_count, 0, a0_target)
            moved_a0 = moved_positions[0]
            if abs(moved_a0 - a0_before) <= DEFAULT_POSITION_TOLERANCE:
                raise RuntimeError(f"M6 a0 未发生有效运动: before={a0_before:.4f} moved={moved_a0:.4f}")
            logger.info("M6 a0 运动后 {}", moved_a0)
        finally:
            if not hand.set_hand_state(current_positions):
                raise RuntimeError("M6 初始位置恢复指令下发失败")
            restored_positions = _wait_for_axis_position(hand, actuator_count, 0, a0_before)
            logger.info("M6 a0 恢复后 {}", restored_positions[0])

        logger.success("M6 右手冒烟测试通过")
    finally:
        close_wuyou_channel(qmlinker_channel)
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
