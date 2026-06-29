from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger
from qmlinker import GripperInfo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import GRIPPER_PORT, create_wuyou_channel, stop_ssh_process
from src.wuji.dahuan_gripper_client import DahuanGripperClient

DEFAULT_REQUEST_TIMEOUT_S = 10.0
POSITION_CONFIRM_S = 1.0
POSITION_POLL_INTERVAL_S = 0.2


def _prompt(text: str) -> str:
    return input(text).strip().lower()


def _print_status(status: GripperInfo) -> None:
    print("")
    print("========== 当前夹爪状态 ==========")
    print(f"在线状态  : {status.online}")
    print(f"校准状态  : {status.calibrated}")
    print(f"使能状态  : {status.enable}")
    print(f"当前位置  : {status.position}")
    print(f"夹持状态码: {status.state}")


def _read_status(client: DahuanGripperClient) -> GripperInfo:
    status = client.get_status()
    _print_status(status)
    return status


def _toggle_enable(client: DahuanGripperClient) -> None:
    status = client.get_status()
    target = not bool(status.enable)
    print(f"当前使能状态：{bool(status.enable)}")
    print(f"切换目标状态：{target}")
    if not client.set_enable(target):
        raise RuntimeError("夹爪切换使能失败")
    time.sleep(0.2)
    _print_status(client.get_status())


def _calibrate(client: DahuanGripperClient) -> None:
    """执行夹爪校准。"""

    print("开始校准夹爪。")
    if not client.calibrate():
        raise RuntimeError("夹爪校准失败")
    time.sleep(1.0)
    _print_status(client.get_status())


def _wait_for_stable_position(client: DahuanGripperClient, target_position: int, timeout_s: float) -> None:
    deadline = time.monotonic() + max(float(timeout_s), POSITION_CONFIRM_S)
    last_position: int | None = None
    stable_since: float | None = None
    while True:
        current_position = int(client.get_status().position or 0)
        if current_position != last_position:
            last_position = current_position
            stable_since = None
        elif stable_since is None:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= POSITION_CONFIRM_S:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"夹爪位置未在 {timeout_s:.1f}s 内稳定到 {target_position}")
        time.sleep(POSITION_POLL_INTERVAL_S)


def _control_position(client: DahuanGripperClient) -> None:
    while True:
        _read_status(client)
        print("")
        print("输入目标位置 0-1000，直接回车保持当前层，q 返回上一层")
        raw = input("pos> ").strip().lower()
        if raw in {"q", "back", "b"}:
            return
        if raw == "":
            continue
        try:
            target_position = int(raw)
        except ValueError:
            print("目标位置必须是整数。")
            continue
        if target_position < 0 or target_position > 1000:
            print("目标位置超出范围。")
            continue
        if not client.set_pos(target_position):
            raise RuntimeError("夹爪位置下发失败")
        _wait_for_stable_position(client, target_position, DEFAULT_REQUEST_TIMEOUT_S)
        _print_status(client.get_status())


def _batch_control(client: DahuanGripperClient) -> None:
    while True:
        print("")
        print("批量控制模式：")
        print("  1 : 先使能后到 1000")
        print("  2 : 回零到 0")
        print("  q : 返回上一层")
        choice = _prompt("batch> ")
        if choice in {"q", "back", "b"}:
            return
        if choice == "1":
            if not client.set_enable(True):
                raise RuntimeError("夹爪使能失败")
            client.set_pos(1000)
            _wait_for_stable_position(client, 1000, DEFAULT_REQUEST_TIMEOUT_S)
            _print_status(client.get_status())
            continue
        if choice == "2":
            client.set_pos(0)
            _wait_for_stable_position(client, 0, DEFAULT_REQUEST_TIMEOUT_S)
            _print_status(client.get_status())
            continue
        print(f"未知选择：{choice}")


def main() -> None:
    """大寰夹爪交互式 CLI。"""

    logger.info("夹爪交互式 CLI 启动")
    ssh_process, channel = create_wuyou_channel(GRIPPER_PORT)
    client = DahuanGripperClient(channel)
    try:
        _print_status(client.get_status())
        while True:
            print("")
            print("========== 夹爪主菜单 ==========")
            print("toggle   : 切换使能")
            print("status   : 打印状态")
            print("calib    : 执行校准")
            print("pos      : 位置控制")
            print("batch    : 批量控制")
            print("q        : 退出")

            command = _prompt("gripper> ")
            if command in {"q", "quit", "exit"}:
                break
            if command == "toggle":
                _toggle_enable(client)
                continue
            if command == "status":
                _read_status(client)
                continue
            if command == "calib":
                _calibrate(client)
                continue
            if command == "pos":
                _control_position(client)
                continue
            if command == "batch":
                _batch_control(client)
                continue
            if command == "":
                continue
            print(f"未知命令：{command}")
    finally:
        stop_ssh_process(ssh_process)


if __name__ == "__main__":
    main()
