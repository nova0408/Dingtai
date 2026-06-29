from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PORT, create_wuyou_channel, stop_ssh_process
from src.wuji.right_hand_client import WujiRightHandClient
from src.wuji.right_hand_specs import (
    RIGHT_HAND_ACTUATOR_SPECS,
)

PRINTED_AXIS_ROWS: tuple[tuple[int, ...], ...] = (
    (1, 2, 4, 6, 8, 10),
    (0, 3, 5, 7, 9),
)
PAIR_GROUPS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("f", "前端", (3, 5, 7, 9)),
    ("e", "末端", (4, 6, 8, 10)),
)


def _clamp_normalized(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _prompt(text: str) -> str:
    return input(text).strip()


def _find_current_positions(hand: WujiRightHandClient) -> list[float]:
    state = hand.get_hand_state(include_tactile=False)
    if state is None:
        raise RuntimeError("右手状态不可用")
    positions = [float(actuator["position"]) for actuator in state["actuators"]]
    if len(positions) != 11:
        raise RuntimeError(f"右手状态数量异常: {len(positions)}")
    return positions


def _build_axis_positions(hand: WujiRightHandClient) -> list[float]:
    """获取 11 轴当前位置。"""

    return _find_current_positions(hand)


def _read_enabled(hand: WujiRightHandClient) -> bool:
    return bool(hand.get_enable())


def _toggle_enable(hand: WujiRightHandClient) -> None:
    enabled = _read_enabled(hand)
    target = not enabled
    print(f"当前使能状态：{enabled}")
    print(f"切换目标状态：{target}")
    if not hand.set_enable(target):
        raise RuntimeError("右手切换使能失败")
    time.sleep(0.2)
    enabled_after = _read_enabled(hand)
    print(f"切换后状态：{enabled_after}")


def _print_axis_rows(positions: list[float]) -> None:
    """按指定行序打印右手轴状态。"""

    print("")
    print("当前右手各轴状态：")
    for row in PRINTED_AXIS_ROWS:
        line = "  ".join(f"{actuator_id}:{positions[actuator_id]:.6f}" for actuator_id in row)
        print(line)


def _print_state(hand: WujiRightHandClient) -> None:
    hand_info = hand.get_hand_info()
    if hand_info is None:
        raise RuntimeError("右手信息不可用")
    print("")
    print("右手信息：")
    print(f"  hand_id        : {hand_info['hand_id']}")
    print(f"  model_name     : {hand_info['model_name']}")
    print(f"  actuator_count : {hand_info['actuator_count']}")
    print(f"  has_tactile    : {hand_info['has_tactile']}")
    print(f"  enabled        : {_read_enabled(hand)}")
    _print_axis_rows(_build_axis_positions(hand))


def _set_single_axis(hand: WujiRightHandClient) -> None:
    while True:
        print("")
        print("可选轴号：1-10")
        user_axis = _prompt("axis> ")
        if user_axis in {"back", "b"}:
            return
        if user_axis == "q":
            return
        if not user_axis.isdigit():
            print("轴号必须是数字。")
            continue
        actuator_id = int(user_axis)
        if actuator_id < 0 or actuator_id >= len(RIGHT_HAND_ACTUATOR_SPECS):
            print("轴号超出范围。")
            continue

        current_positions = _build_axis_positions(hand)
        _print_axis_rows(current_positions)

        while True:
            target_text = _prompt(f"right_hand_a{actuator_id} target> ")
            if target_text in {"back", "b"}:
                break
            if target_text == "q":
                break
            try:
                target_value = _clamp_normalized(float(target_text))
            except ValueError:
                print("目标值不是有效数字。")
                continue

            current_positions[actuator_id] = target_value
            if not hand.set_hand_state(current_positions):
                raise RuntimeError("右手单轴下发失败")
            time.sleep(1.0)
            _print_axis_rows(_build_axis_positions(hand))
            continue

        continue


def _set_four_finger_pair(hand: WujiRightHandClient) -> None:
    while True:
        print("")
        print("选择四指联动目标：")
        print("  f : 根部")
        print("  e : 指尖")
        print("  q : 返回上一层")
        user_choice = _prompt("pair> ")
        if user_choice in {"q", "back", "b"}:
            return

        target_group: tuple[int, ...] | None = None
        target_label = ""
        for choice, label, group in PAIR_GROUPS:
            if user_choice == choice:
                target_group = group
                target_label = label
                break
        if target_group is None:
            print(f"未知选择：{user_choice}")
            continue

        current_positions = _build_axis_positions(hand)
        _print_axis_rows(current_positions)
        while True:
            value_text = _prompt(f"{target_label} target> ")
            if value_text in {"back", "b"}:
                break
            if value_text == "q":
                break
            try:
                target_value = _clamp_normalized(float(value_text))
            except ValueError:
                print("目标值不是有效数字。")
                continue

            for actuator_id in target_group:
                current_positions[actuator_id] = target_value
            if not hand.set_hand_state(current_positions):
                raise RuntimeError("四指联动下发失败")
            time.sleep(1.0)
            _print_axis_rows(_build_axis_positions(hand))
            continue

        continue


def main() -> None:
    """右手交互式 smoke。"""

    ssh_process, qmlinker_channel = create_wuyou_channel(DEFAULT_PORT)
    hand = WujiRightHandClient(qmlinker_channel)
    try:
        logger.info("右手交互测试")
        _print_state(hand)

        while True:
            print("")
            print("========== 右手菜单 ==========")
            print("t : 切换使能")
            print("s : 打印状态")
            print("a : 单轴控制")
            print("p : 四指联动")
            print("q : 退出")

            command = _prompt("main> ")

            if command in {"q", "quit", "exit"}:
                break
            if command == "t":
                _toggle_enable(hand)
                continue
            if command == "s":
                _print_state(hand)
                continue
            if command == "a":
                _set_single_axis(hand)
                continue
            if command == "p":
                _set_four_finger_pair(hand)
                continue
            if command == "":
                continue
            print(f"未知命令：{command}")
    finally:
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
