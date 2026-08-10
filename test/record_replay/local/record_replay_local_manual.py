"""本机经 Orin 管理网 IP 人工管理 RecordReplay，禁止自动运行。"""

from __future__ import annotations

import json
from pathlib import Path

from record_replay.client import RecordReplayClient

SERVICE_ADDR = "http://192.168.1.128:6300"
"Orin 在本机管理网中的 RecordReplay 服务地址。"
HARDWARE_CONFIRMATION = "我已确认现场安全并同意设备运动"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_CONFIG_PATH = PROJECT_ROOT / "record_replay" / "config.json"


def main() -> int:
    """直连 Orin service API，并提供配置、状态和人工启动菜单。"""

    client = RecordReplayClient(SERVICE_ADDR)
    while True:
        print("1: 获取状态  2: 获取配置  3: 修改配置  4: 启动一次  5: 设备诊断  q: 退出")
        choice = input("请选择：").strip().lower()
        if choice == "q":
            return 0
        if choice == "1":
            print(json.dumps(client.get_status(), ensure_ascii=False, indent=2))
        elif choice == "2":
            print(json.dumps(client.get_config(), ensure_ascii=False, indent=2))
        elif choice == "3":
            key = input("参数名：").strip()
            value = float(input("新数值：").strip())
            response = client.update_config({key: value})
            print(json.dumps(response, ensure_ascii=False, indent=2))
            parameters = response.get("parameters")
            if isinstance(parameters, dict):
                LOCAL_CONFIG_PATH.write_text(
                    json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(f"本机默认配置已同步：{LOCAL_CONFIG_PATH}")
        elif choice == "4":
            print("警告：start 会触发机械臂、AGV、手部或升降设备运动。")
            confirmation = input(f"请输入“{HARDWARE_CONFIRMATION}”继续：").strip()
            if confirmation != HARDWARE_CONFIRMATION:
                print("未获得完整现场安全确认，不发送启动请求。")
                continue
            old_tray_current_index = int(input("旧托盘当前位置 index：").strip())
            old_tray_put_index = int(input("旧托盘放置位置 index：").strip())
            new_tray_current_index = int(input("新托盘当前位置 index：").strip())
            new_tray_put_index = int(input("新托盘放置位置 index：").strip())
            enable_agv_navigation = input("是否启用 AGV 导航（y/N）：").strip().lower() == "y"
            agv_target = input("AGV 目标：").strip()
            print(
                json.dumps(
                    client.start(
                        old_tray_current_index,
                        old_tray_put_index,
                        new_tray_current_index,
                        new_tray_put_index,
                        enable_agv_navigation,
                        agv_target,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif choice == "5":
            print("设备诊断只读取状态，不下发上电、使能或运动指令。")
            print(json.dumps(client.get_device_status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
