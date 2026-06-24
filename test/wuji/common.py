from __future__ import annotations

import subprocess
import time

from loguru import logger
from qmlinker import create_channel

GRIPPER_PORT = 50066
DEFAULT_PORT = 50062
DATA_PORT = 50061
ORIN_HOST = "192.168.100.60"
WUYOU_HOST = "192.168.1.114"
AGV_HOST = "192.168.100.70"
SSH_ALIAS = "orin"
WUYOU_SSH_ALIAS = "wuyou"
TUNNEL_WAIT_S = 1.0


def create_orin_channel(remote_port: int, remote_host: str = ORIN_HOST) -> tuple[subprocess.Popen[bytes], object]:
    """启动到 Orin 的 SSH 转发，并返回本机测试 channel。"""

    process = start_ssh_tunnel(remote_port, remote_host)
    time.sleep(TUNNEL_WAIT_S)
    return process, create_channel(f"127.0.0.1:{int(remote_port) - 1}")


def stop_ssh_process(process: subprocess.Popen[bytes]) -> None:
    """停止 SSH 转发进程。"""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except Exception:  # noqa: BLE001
        process.kill()
        process.wait(timeout=3.0)


def start_ssh_tunnel(
    remote_port: int,
    remote_host: str = ORIN_HOST,
    ssh_alias: str = SSH_ALIAS,
) -> subprocess.Popen[bytes]:
    """启动固定端口 SSH 本地转发。"""

    command = [
        "ssh",
        "-N",
        "-L",
        f"127.0.0.1:{int(remote_port) - 1}:{remote_host}:{int(remote_port)}",
        ssh_alias,
    ]
    logger.info(
        "启动 SSH 隧道-本地 127.0.0.1:{} | 远端 {}:{} | 跳板 {}",
        int(remote_port) - 1,
        remote_host,
        int(remote_port),
        ssh_alias,
    )
    return subprocess.Popen(command)
