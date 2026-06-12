from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

DEFAULT_REQUEST_TIMEOUT_S = 10.0
"""单次状态读取超时，单位 s。"""

DEFAULT_ORIN_SSH_ALIAS = "orin"
"""本机 SSH 配置中的 Orin 别名。"""

DEFAULT_REMOTE_HOST = "192.168.100.60"
"""夹爪远端服务地址。"""

DEFAULT_REMOTE_PORT = 50066
"""夹爪远端服务端口。"""

POSITION_POLL_INTERVAL_S = 0.2
"""位置轮询间隔，单位 s。"""

POSITION_STABLE_CONFIRM_S = 1.0
"""位置连续稳定确认时长，单位 s。"""

current_file = Path(__file__).resolve()
for parent in current_file.parents:
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

from qmlinker import GripperInfo  # noqa: E402

from src.wuji.client_base import WujiQmlinkerBaseClient  # noqa: E402
from src.wuji.dahuan_gripper_client import DahuanGripperClient  # noqa: E402
from src.wuji.protocol import WujiQmlinkerConfig  # noqa: E402


def _print_status(status: GripperInfo) -> None:
    print("")
    print("========== 当前夹爪状态 ==========")
    print(f"在线状态      : {status.online}")
    print(f"校准状态      : {status.calibrated}")
    print(f"使能状态      : {status.enable}")
    print(f"当前位置      : {status.position}")
    print(f"夹持状态码    : {status.state}")
    print("", flush=True)


def _wait_for_stable_position(
    client: DahuanGripperClient,
    target_position: int,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + max(float(timeout_s), POSITION_STABLE_CONFIRM_S)
    last_position: int | None = None
    stable_since: float | None = None
    while True:
        status = client.get_status()
        current_position = int(status.position or 0)
        if current_position != last_position:
            last_position = current_position
            stable_since = None
        elif stable_since is None:
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= POSITION_STABLE_CONFIRM_S:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"夹爪位置未在 {timeout_s:.1f}s 内稳定到 {target_position}")
        time.sleep(POSITION_POLL_INTERVAL_S)


def run_interactive_cli(
    request_timeout_s: float,
    ssh_alias: str,
    remote_host: str,
    remote_port: int,
) -> None:
    base_client = WujiQmlinkerBaseClient(
        WujiQmlinkerConfig(
            host=str(remote_host),
            port=int(remote_port),
            request_timeout_s=float(request_timeout_s),
        )
    )
    client = DahuanGripperClient(
        base_client,
        ssh_alias=ssh_alias,
        remote_host=remote_host,
        remote_port=remote_port,
    )
    try:
        logger.info("夹爪交互式 CLI 已启动 host={} port={}", remote_host, remote_port)
        _print_status(client.get_status())
        while True:
            raw = input("gripper> ").strip()
            if not raw:
                continue
            parts = raw.split()
            command = parts[0].lower()
            if command in {"quit", "exit", "q"}:
                return
            if command in {"status", "show", "s"}:
                _print_status(client.get_status())
                continue
            if command in {"help", "h", "?"}:
                print("status | pos <值> | quit | help", flush=True)
                continue
            if command in {"pos", "p"} and len(parts) == 2:
                target_position = int(parts[1])
                client.set_pos(target_position)
                _wait_for_stable_position(client, target_position, float(request_timeout_s))
                _print_status(client.get_status())
                continue
            logger.warning("未知命令：{}", raw)
    finally:
        base_client.close()


def _parse_cli() -> tuple[float, str, str, int]:
    parser = argparse.ArgumentParser(description="大寰夹爪交互式控制台")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--ssh-alias", type=str, default=DEFAULT_ORIN_SSH_ALIAS)
    parser.add_argument("--remote-host", type=str, default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    args = parser.parse_args()
    return float(args.request_timeout_s), str(args.ssh_alias), str(args.remote_host), int(args.remote_port)


def main(
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ssh_alias: str = DEFAULT_ORIN_SSH_ALIAS,
    remote_host: str = DEFAULT_REMOTE_HOST,
    remote_port: int = DEFAULT_REMOTE_PORT,
) -> None:
    run_interactive_cli(request_timeout_s, ssh_alias, remote_host, remote_port)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_request_timeout_s, cli_ssh_alias, cli_remote_host, cli_remote_port = _parse_cli()
        main(
            request_timeout_s=cli_request_timeout_s,
            ssh_alias=cli_ssh_alias,
            remote_host=cli_remote_host,
            remote_port=cli_remote_port,
        )
    else:
        main()
