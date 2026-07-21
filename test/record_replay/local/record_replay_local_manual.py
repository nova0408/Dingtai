"""本机经 SSH 转发人工管理 Orin RecordReplay 服务，禁止自动运行。"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from record_replay.service.client import RecordReplayClient

SSH_ALIAS = "orin"
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 6300
REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 6300
HARDWARE_CONFIRMATION = "我已确认现场安全并同意设备运动"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_CONFIG_PATH = PROJECT_ROOT / "record_replay" / "config.json"


def main() -> int:
    """建立 service API 转发，并提供配置、状态和人工启动菜单。"""

    process = subprocess.Popen(
        [
            "ssh",
            "-o",
            "ExitOnForwardFailure=yes",
            "-N",
            "-L",
            f"{LOCAL_HOST}:{LOCAL_PORT}:{REMOTE_HOST}:{REMOTE_PORT}",
            SSH_ALIAS,
        ],
        stderr=subprocess.PIPE,
    )
    time.sleep(0.5)
    if process.poll() is not None:
        stderr = b"" if process.stderr is None else process.stderr.read()
        raise RuntimeError(stderr.decode("utf-8", errors="replace").strip() or "SSH 转发启动失败")
    client = RecordReplayClient(f"http://{LOCAL_HOST}:{LOCAL_PORT}")
    try:
        while True:
            print("1: 获取状态  2: 获取配置  3: 修改配置  4: 启动一轮  q: 退出")
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
                print(json.dumps(client.start(), ensure_ascii=False, indent=2))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3.0)


if __name__ == "__main__":
    raise SystemExit(main())
