from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import create_wuyou_channel, stop_ssh_process
from src.wuji.head_client import WujiHeadClient

DEFAULT_STEP_DEG = 5.0


def _prompt(text: str) -> str:
    return input(text).strip().lower()


def _read_state(head: WujiHeadClient) -> tuple[bool, float, float]:
    enabled = bool(head.get_enable())
    yaw_deg = float(head.get_head_yaw() or 0.0)
    pitch_deg = float(head.get_head_pitch() or 0.0)
    return enabled, yaw_deg, pitch_deg


def _print_state(head: WujiHeadClient) -> tuple[bool, float, float]:
    enabled, yaw_deg, pitch_deg = _read_state(head)
    print("")
    print(f"使能状态: {enabled}")
    print(f"yaw     : {yaw_deg:.1f} deg")
    print(f"pitch   : {pitch_deg:.1f} deg")
    return enabled, yaw_deg, pitch_deg


def _control_axis(head: WujiHeadClient, axis: str) -> None:
    while True:
        _, yaw_deg, pitch_deg = _print_state(head)
        current_value = yaw_deg if axis == "yaw" else pitch_deg
        value_text = input(
            f"请输入目标 {axis} 角度（deg），直接回车表示 +{DEFAULT_STEP_DEG:.1f} deg, q 返回: "
        ).strip().lower()
        if value_text == "q":
            return
        if value_text == "":
            target_deg = current_value + DEFAULT_STEP_DEG
        else:
            try:
                target_deg = float(value_text)
            except ValueError:
                print("目标值不是有效数字。")
                continue

        logger.info("{} 当前 {:.1f} deg，目标 {:.1f} deg", axis, current_value, target_deg)
        if axis == "yaw":
            head.set_head_yaw(target_deg)
        else:
            head.set_head_pitch(target_deg)
        time.sleep(1.0)
        _print_state(head)


def _print_menu() -> None:
    print("")
    print("========== 头部主菜单 ==========")
    print("state : 打印当前状态")
    print("yaw   : 控制 yaw")
    print("pitch : 控制 pitch")
    print("q     : 退出")


def main() -> None:
    """头部交互式 CLI。"""

    logger.info("头部控制脚本启动，请先确认 wuyou qmlinker 连接正常。")
    ssh_process, qmlinker_channel = create_wuyou_channel()
    head_client = WujiHeadClient(qmlinker_channel)
    try:
        _print_state(head_client)
        while True:
            _print_menu()
            command = _prompt("main> ")
            if command == "q":
                break
            if command == "state":
                _print_state(head_client)
                continue
            if command in {"yaw", "pitch"}:
                _control_axis(head_client, command)
                continue
            if command == "":
                continue
            print(f"未知命令：{command}")
    finally:
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
