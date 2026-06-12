from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

DEFAULT_REQUEST_TIMEOUT_S = 3.0
"""夹爪状态读取超时时间，单位 s。"""

DEFAULT_ORIN_SSH_ALIAS = "orin"
"""本机 SSH 配置中的 Orin 别名。"""

DEFAULT_REMOTE_HOST = "192.168.100.60"
"""夹爪远端服务地址。"""

DEFAULT_REMOTE_PORT = 50066
"""夹爪远端服务端口。"""

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.wuji.client_base import WujiQmlinkerBaseClient  # noqa: E402
from src.wuji.dahuan_gripper_client import DahuanGripperClient  # noqa: E402
from src.wuji.protocol import WujiQmlinkerConfig  # noqa: E402
from qmlinker import QMGripper  # noqa: E402


# region 工具


def _format_status(status: object) -> str:
    """格式化夹爪状态对象，便于打印关键字段。"""

    if status is None:
        return "status=None"

    field_names = (
        "timestamp_ms",
        "position",
        "speed",
        "force",
        "grip_state",
        "enabled",
        "state",
    )
    parts: list[str] = []
    for field_name in field_names:
        if hasattr(status, field_name):
            parts.append(f"{field_name}={getattr(status, field_name)!r}")
    if not parts:
        return repr(status)
    return ", ".join(parts)


# endregion


# region 主入口


def main(
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ssh_alias: str = DEFAULT_ORIN_SSH_ALIAS,
    remote_host: str = DEFAULT_REMOTE_HOST,
    remote_port: int = DEFAULT_REMOTE_PORT,
) -> None:
    """经 Orin SSH 转发后读取夹爪当前状态。"""

    logger.info("夹爪冒烟测试启动")
    logger.info("SSH 别名 {} 远端地址 {} 端口 {}", ssh_alias, remote_host, remote_port)

    base_client = WujiQmlinkerBaseClient(
        WujiQmlinkerConfig(
            host=str(remote_host),
            port=int(remote_port),
            request_timeout_s=float(request_timeout_s),
        )
    )
    gripper = DahuanGripperClient(
        base_client,
        ssh_alias=ssh_alias,
        remote_host=remote_host,
        remote_port=remote_port,
    )
    logger.info("请求超时 {} s", request_timeout_s)

    try:
        status: object | None = None
        status = gripper._send_control(QMGripper.STATUS)
        logger.info("_send_control STATUS 结果 {}", status)
        
        if status is None:
            raise RuntimeError("夹爪状态读取失败：接口返回 None。")

        gripper.set_pos(1000)
        time.sleep(3)
        status = gripper._send_control(QMGripper.STATUS)
        if not isinstance(status, dict):
            raise RuntimeError(f"夹爪状态读取失败：位置 1000 后返回值非法 {status!r}")
        pos = status["position"]
        logger.info(f"pos:{pos}")
        gripper.set_pos(0)
        time.sleep(3)
        status = gripper._send_control(QMGripper.STATUS)
        if not isinstance(status, dict):
            raise RuntimeError(f"夹爪状态读取失败：位置 0 后返回值非法 {status!r}")
        pos = status["position"]
        logger.info(f"pos:{pos}")
        logger.success("无际夹爪冒烟测试通过")
    finally:
        base_client.close()
        logger.info("无际夹爪冒烟测试结束")


# endregion


# region CLI


def _parse_cli() -> tuple[float, str, str, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="经 Orin SSH 转发读取无际夹爪状态")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--ssh-alias", type=str, default=DEFAULT_ORIN_SSH_ALIAS)
    parser.add_argument("--remote-host", type=str, default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    args = parser.parse_args()
    return (
        float(args.request_timeout_s),
        str(args.ssh_alias),
        str(args.remote_host),
        int(args.remote_port),
    )


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


# endregion
