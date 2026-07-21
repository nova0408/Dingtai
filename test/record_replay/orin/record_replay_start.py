"""Orin 上人工确认后启动一轮 RecordReplay。"""

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
HARDWARE_CONFIRMATION = "我已确认现场安全并同意执行RecordReplay回放"
"启动真实设备回放前必须完整输入的确认文本。"


def main(
    enable_agv_navigation: bool = DEFAULT_ENABLE_AGV_NAVIGATION,
    service_addr: str = SERVICE_ADDR,
) -> int:
    """确认现场安全后，显式指定 AGV 选项并发送 start。"""

    logger.warning(
        "本次回放将控制双臂、夹爪、M11 和 Lift，AGV 导航={}",
        "启用" if enable_agv_navigation else "禁用",
    )
    confirmation = input(f"请输入“{HARDWARE_CONFIRMATION}”继续：").strip()
    if confirmation != HARDWARE_CONFIRMATION:
        logger.warning("未获得完整现场安全确认，不发送 start")
        return 1
    client = RecordReplayClient(service_addr)
    response = client.start(enable_agv_navigation=enable_agv_navigation)
    logger.info("RecordReplay start 响应\n{}", json.dumps(response, ensure_ascii=False, indent=2))
    if response.get("accepted") is not True:
        logger.warning("RecordReplay 未接受本次 start")
        return 1
    logger.success(
        "RecordReplay 已接受 start，AGV 导航={}",
        "启用" if enable_agv_navigation else "禁用",
    )
    return 0


def _parse_cli() -> bool:
    """解析是否为本轮回放启用 AGV 导航。"""

    parser = argparse.ArgumentParser(description="人工启动 Orin RecordReplay")
    parser.add_argument(
        "--agv",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ENABLE_AGV_NAVIGATION,
        help="启用或禁用本轮 AGV 去程与返程导航",
    )
    return bool(parser.parse_args().agv)


if __name__ == "__main__":
    selected_enable_agv = _parse_cli() if len(sys.argv) > 1 else DEFAULT_ENABLE_AGV_NAVIGATION
    raise SystemExit(main(enable_agv_navigation=selected_enable_agv))
