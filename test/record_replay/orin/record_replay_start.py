"""Orin 上人工确认后启动一次 RecordReplay。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from record_replay.client import RecordReplayClient

SERVICE_ADDR = "http://127.0.0.1:6300"
"已部署 RecordReplay 服务在 Orin 本机的固定地址。"
DEFAULT_ENABLE_AGV_NAVIGATION = False
"默认不移动 AGV；可修改此常量或用 --agv/--no-agv 覆盖。"
DEFAULT_OLD_TRAY_CURRENT_INDEX = 1
"本次执行的旧托盘当前位置 index 默认值。"
DEFAULT_OLD_TRAY_PUT_INDEX = 4
"本次执行的旧托盘放置位置 index 默认值。"
DEFAULT_NEW_TRAY_CURRENT_INDEX = 1
"本次执行的新托盘当前位置 index 默认值。"
DEFAULT_NEW_TRAY_PUT_INDEX = 1
"本次执行的新托盘放置位置 index 默认值。"
DEFAULT_AGV_TARGET = "1"
"启用 AGV 时的默认目标。"
HARDWARE_CONFIRMATION = "我已确认现场安全并同意执行RecordReplay回放"
"启动真实设备回放前必须完整输入的确认文本。"


def main(
    enable_agv_navigation: bool = DEFAULT_ENABLE_AGV_NAVIGATION,
    old_tray_current_index: int = DEFAULT_OLD_TRAY_CURRENT_INDEX,
    old_tray_put_index: int = DEFAULT_OLD_TRAY_PUT_INDEX,
    new_tray_current_index: int = DEFAULT_NEW_TRAY_CURRENT_INDEX,
    new_tray_put_index: int = DEFAULT_NEW_TRAY_PUT_INDEX,
    agv_target: str = DEFAULT_AGV_TARGET,
    service_addr: str = SERVICE_ADDR,
) -> int:
    """确认现场安全后，显式指定托盘 index 和 AGV 选项并发送 start。"""

    logger.warning(
        "本次回放将控制双臂、夹爪、M6 和 Lift，旧托盘当前位置={} 放置位置={}，新托盘当前位置={} 放置位置={}，AGV 导航={} 目标={}",
        old_tray_current_index,
        old_tray_put_index,
        new_tray_current_index,
        new_tray_put_index,
        "启用" if enable_agv_navigation else "禁用",
        agv_target,
    )
    confirmation = input(f"请输入“{HARDWARE_CONFIRMATION}”继续：").strip()
    if confirmation != HARDWARE_CONFIRMATION:
        logger.warning("未获得完整现场安全确认，不发送 start")
        return 1
    client = RecordReplayClient(service_addr)
    response = client.start(
        old_tray_current_index,
        old_tray_put_index,
        new_tray_current_index,
        new_tray_put_index,
        enable_agv_navigation,
        agv_target,
    )
    logger.info("RecordReplay start 响应\n{}", json.dumps(response, ensure_ascii=False, indent=2))
    if response.get("accepted") is not True:
        logger.warning("RecordReplay 未接受本次 start")
        return 1
    logger.success(
        "RecordReplay 已接受 start，旧托盘当前位置={} 放置位置={}，新托盘当前位置={} 放置位置={}，AGV 导航={} 目标={}",
        old_tray_current_index,
        old_tray_put_index,
        new_tray_current_index,
        new_tray_put_index,
        "启用" if enable_agv_navigation else "禁用",
        agv_target,
    )
    return 0


def _parse_cli() -> argparse.Namespace:
    """解析本轮执行的托盘 index 和 AGV 选项。"""

    parser = argparse.ArgumentParser(description="人工启动 Orin RecordReplay")
    parser.add_argument(
        "--agv",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_AGV_NAVIGATION,
        help="启用或禁用本轮 AGV 去程与返程导航",
    )
    parser.add_argument("--old-tray-current-index", type=int, default=DEFAULT_OLD_TRAY_CURRENT_INDEX)
    parser.add_argument("--old-tray-put-index", type=int, default=DEFAULT_OLD_TRAY_PUT_INDEX)
    parser.add_argument("--new-tray-current-index", type=int, default=DEFAULT_NEW_TRAY_CURRENT_INDEX)
    parser.add_argument("--new-tray-put-index", type=int, default=DEFAULT_NEW_TRAY_PUT_INDEX)
    parser.add_argument("--agv-target", default=DEFAULT_AGV_TARGET)
    return parser.parse_args()


if __name__ == "__main__":
    selected = _parse_cli() if len(sys.argv) > 1 else argparse.Namespace(
        agv=DEFAULT_ENABLE_AGV_NAVIGATION,
        old_tray_current_index=DEFAULT_OLD_TRAY_CURRENT_INDEX,
        old_tray_put_index=DEFAULT_OLD_TRAY_PUT_INDEX,
        new_tray_current_index=DEFAULT_NEW_TRAY_CURRENT_INDEX,
        new_tray_put_index=DEFAULT_NEW_TRAY_PUT_INDEX,
        agv_target=DEFAULT_AGV_TARGET,
    )
    raise SystemExit(
        main(
            enable_agv_navigation=bool(selected.agv),
            old_tray_current_index=selected.old_tray_current_index,
            old_tray_put_index=selected.old_tray_put_index,
            new_tray_current_index=selected.new_tray_current_index,
            new_tray_put_index=selected.new_tray_put_index,
            agv_target=selected.agv_target,
        )
    )
