from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from loguru import logger
from qmlinker import create_channel
from network_discovery import (
    DEFAULT_ORIN_FALLBACKS,
    DEFAULT_ORIN_SSH_ALIAS,
    DEFAULT_WUYOU_FALLBACKS,
    DEFAULT_WUYOU_SSH_ALIAS,
    get_cached_orin_host,
    get_cached_wuyou_host,
    iter_candidate_hosts,
    remember_host,
)

GRIPPER_PORT = 50066
DEFAULT_PORT = 50062
DATA_PORT = 50061
ORIN_HOST = get_cached_orin_host()
WUYOU_HOST = get_cached_wuyou_host()
AGV_HOST = "192.168.100.70"
SSH_ALIAS = "orin"
WUYOU_SSH_ALIAS = "wuyou"
TUNNEL_WAIT_S = 1.0


@dataclass(slots=True)
class SshTunnelGroup:
    """一组需要统一关闭的 SSH 转发进程。"""

    processes: tuple[subprocess.Popen[bytes], ...]


def create_orin_channel(remote_port: int, remote_host: str = ORIN_HOST) -> tuple[SshTunnelGroup, object]:
    """按 qmlinker 官方示例创建本机调试 channel。

    Notes
    -----
    `qmlinker.create_channel("host")` 会同时使用：
    - `DEFAULT` 通道：`50062`
    - `DATA` 通道：`50061`

    因此本机 SSH 调试不能把 `50062` 错映射到 `50061` 再直连单端口，
    而应保持本地端口号与远端一致，再调用 `create_channel("127.0.0.1")`。
    """

    if int(remote_port) != DEFAULT_PORT:
        process = start_ssh_tunnel(remote_port, remote_host)
        time.sleep(TUNNEL_WAIT_S)
        return SshTunnelGroup((process,)), create_channel(f"127.0.0.1:{int(remote_port) - 1}")

    default_process = _start_ssh_tunnel_with_local_port(
        remote_port=DEFAULT_PORT,
        remote_host=remote_host,
        local_port=DEFAULT_PORT,
        ssh_alias=SSH_ALIAS,
    )
    data_process = _start_ssh_tunnel_with_local_port(
        remote_port=DATA_PORT,
        remote_host=remote_host,
        local_port=DATA_PORT,
        ssh_alias=SSH_ALIAS,
        allow_discovery=False,
    )
    return SshTunnelGroup((default_process, data_process)), create_channel("127.0.0.1")


def create_wuyou_channel(remote_port: int = DEFAULT_PORT, remote_host: str = WUYOU_HOST) -> tuple[SshTunnelGroup, object]:
    """为 wuyou 上的 qmlinker 服务创建本机调试 channel。"""

    if int(remote_port) != DEFAULT_PORT:
        process = start_ssh_tunnel(remote_port, remote_host=remote_host, ssh_alias=WUYOU_SSH_ALIAS)
        time.sleep(TUNNEL_WAIT_S)
        return SshTunnelGroup((process,)), create_channel(f"127.0.0.1:{int(remote_port) - 1}")

    default_process = _start_ssh_tunnel_with_local_port(
        remote_port=DEFAULT_PORT,
        remote_host=remote_host,
        local_port=DEFAULT_PORT,
        ssh_alias=WUYOU_SSH_ALIAS,
        allow_discovery=True,
    )
    return SshTunnelGroup((default_process,)), create_channel("127.0.0.1")


def stop_ssh_process(process: subprocess.Popen[bytes] | SshTunnelGroup) -> None:
    """停止 SSH 转发进程。"""

    if isinstance(process, SshTunnelGroup):
        for item in process.processes:
            _stop_single_ssh_process(item)
        return
    _stop_single_ssh_process(process)


def start_ssh_tunnel(
    remote_port: int,
    remote_host: str = ORIN_HOST,
    ssh_alias: str = SSH_ALIAS,
) -> subprocess.Popen[bytes]:
    """启动固定端口 SSH 本地转发。

    Notes
    -----
    默认只尝试缓存主机，避免每次启动都做探测。
    仅当 SSH 进程提前退出时，才按候选列表逐个重试并回写成功主机。
    """

    return _start_ssh_tunnel_with_local_port(
        remote_port=int(remote_port),
        remote_host=str(remote_host),
        local_port=int(remote_port) - 1,
        ssh_alias=str(ssh_alias),
    )


def _start_ssh_tunnel_with_local_port(
    remote_port: int,
    remote_host: str,
    local_port: int,
    ssh_alias: str,
    allow_discovery: bool = True,
) -> subprocess.Popen[bytes]:
    """按指定本地端口启动 SSH 转发。"""

    candidate_hosts = _build_tunnel_candidates(
        remote_host=str(remote_host),
        ssh_alias=str(ssh_alias),
        allow_discovery=bool(allow_discovery),
    )
    last_process: subprocess.Popen[bytes] | None = None
    for index, candidate_host in enumerate(candidate_hosts):
        if index > 0:
            logger.warning(
                "SSH 隧道首选主机失败，尝试候选 {}:{} -> {}",
                ssh_alias,
                int(remote_port),
                candidate_host,
            )
        process = _spawn_ssh_tunnel(
            remote_port=int(remote_port),
            remote_host=candidate_host,
            local_port=int(local_port),
            ssh_alias=str(ssh_alias),
        )
        time.sleep(TUNNEL_WAIT_S)
        if process.poll() is None:
            _remember_alias_host(ssh_alias=str(ssh_alias), remote_host=candidate_host)
            return process
        _read_early_exit_error(process)
        last_process = process

    if last_process is None:
        raise RuntimeError(f"ssh tunnel start failed: remote={remote_host}:{int(remote_port)} alias={ssh_alias}")
    raise RuntimeError(
        f"ssh tunnel exited early after retries: alias={ssh_alias} remote_port={int(remote_port)} host={remote_host}"
    )


def _spawn_ssh_tunnel(remote_port: int, remote_host: str, local_port: int, ssh_alias: str) -> subprocess.Popen[bytes]:
    """创建单次 SSH 隧道子进程。"""

    command = [
        "ssh",
        "-N",
        "-L",
        f"127.0.0.1:{int(local_port)}:{remote_host}:{int(remote_port)}",
        ssh_alias,
    ]
    logger.info(
        "启动 SSH 隧道-本地 127.0.0.1:{} | 远端 {}:{} | 跳板 {}",
        int(local_port),
        remote_host,
        int(remote_port),
        ssh_alias,
    )
    return subprocess.Popen(command, stderr=subprocess.PIPE)


def _build_tunnel_candidates(remote_host: str, ssh_alias: str, allow_discovery: bool) -> tuple[str, ...]:
    """按当前目标类型生成 SSH 转发重试候选。"""

    if not allow_discovery:
        return (str(remote_host),)
    if ssh_alias == DEFAULT_ORIN_SSH_ALIAS and remote_host == ORIN_HOST:
        return iter_candidate_hosts(ssh_alias, DEFAULT_ORIN_FALLBACKS, preferred_host=remote_host)
    if ssh_alias == DEFAULT_WUYOU_SSH_ALIAS and remote_host == WUYOU_HOST:
        return iter_candidate_hosts(ssh_alias, DEFAULT_WUYOU_FALLBACKS, preferred_host=remote_host)
    return (str(remote_host),)


def _remember_alias_host(ssh_alias: str, remote_host: str) -> None:
    """仅为自动探测场景回写缓存。"""

    if ssh_alias == DEFAULT_ORIN_SSH_ALIAS:
        remember_host(ssh_alias, remote_host)
        return
    if ssh_alias == DEFAULT_WUYOU_SSH_ALIAS:
        remember_host(ssh_alias, remote_host)


def _read_early_exit_error(process: subprocess.Popen[bytes]) -> None:
    """读取 SSH 进程的提前退出错误日志。"""

    if process.stderr is None:
        return
    try:
        stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return
    if stderr_text:
        logger.warning("SSH 隧道提前退出: {}", stderr_text)


def _stop_single_ssh_process(process: subprocess.Popen[bytes]) -> None:
    """停止单个 SSH 转发进程。"""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except Exception:  # noqa: BLE001
        process.kill()
        process.wait(timeout=3.0)
