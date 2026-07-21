"""Orin 上只读获取 RecordReplay 服务与现场设备状态。"""

from __future__ import annotations

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


def main(service_addr: str = SERVICE_ADDR) -> int:
    """只读输出服务状态和全部设备诊断状态。"""

    client = RecordReplayClient(service_addr)
    service_status = client.get_status()
    device_status = client.get_device_status()
    logger.info("RecordReplay 服务状态\n{}", json.dumps(service_status, ensure_ascii=False, indent=2))
    logger.info("RecordReplay 设备状态\n{}", json.dumps(device_status, ensure_ascii=False, indent=2))
    if device_status.get("all_connected") is True:
        logger.success("双臂、Gripper、Head 和 Lift 状态读取成功")
        return 0
    logger.warning("存在未正确连接或状态读取失败的设备，请检查各项 error")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
