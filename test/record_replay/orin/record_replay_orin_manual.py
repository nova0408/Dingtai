"""Orin 上人工触发已部署 RecordReplay API，禁止自动运行。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from record_replay.client import RecordReplayClient

SERVICE_ADDR = "http://127.0.0.1:6300"
HARDWARE_CONFIRMATION = "我已确认现场安全并同意设备运动"


def main() -> int:
    """由现场人员确认后发送一次 start，并显示服务响应。"""

    print("警告：该操作会触发机械臂、AGV、手部或升降设备运动。")
    confirmation = input(f"请输入“{HARDWARE_CONFIRMATION}”继续：").strip()
    if confirmation != HARDWARE_CONFIRMATION:
        print("未获得完整现场安全确认，不发送启动请求。")
        return 1
    client = RecordReplayClient(SERVICE_ADDR)
    old_tray_current_index = int(input("旧托盘当前位置 index：").strip())
    old_tray_put_index = int(input("旧托盘放置位置 index：").strip())
    new_tray_current_index = int(input("新托盘当前位置 index：").strip())
    new_tray_put_index = int(input("新托盘放置位置 index：").strip())
    enable_agv_navigation = input("是否启用 AGV 导航（y/N）：").strip().lower() == "y"
    agv_target = input("AGV 目标：").strip()
    print(
        client.start(
            old_tray_current_index,
            old_tray_put_index,
            new_tray_current_index,
            new_tray_put_index,
            enable_agv_navigation,
            agv_target,
        )
    )
    print(client.get_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
