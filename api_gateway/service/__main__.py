"""统一入口服务启动入口。"""

from __future__ import annotations

import argparse
import logging
import ssl
from collections.abc import Sequence
from pathlib import Path

from aiohttp import web

from .. import API_GATEWAY_VERSION
from ..config import GatewaySettings
from ..server import create_app, on_cleanup, on_startup

DEFAULT_TLS_CERT_PATH = Path(
    "/etc/dingtai/api-gateway/tls/api-gateway.fullchain.pem"
)
DEFAULT_TLS_KEY_PATH = Path("/etc/dingtai/api-gateway/tls/api-gateway.key.pem")
_DEFAULT_LISTEN_HOSTS = ("0.0.0.0", "::")


def main(argv: Sequence[str] | None = None) -> int:
    """启动统一入口，不主动连接任意后端。"""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args(argv)
    settings = GatewaySettings(host=args.host, port=args.port)
    ssl_context = _create_ssl_context(args.tls_cert, args.tls_key)
    app = create_app(settings)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    listen_host: str | tuple[str, ...] = settings.host
    if settings.host == "0.0.0.0":
        listen_host = _DEFAULT_LISTEN_HOSTS
    logging.getLogger(__name__).info(
        "API Gateway 启动 version=%s host=%s port=%s tls_cert=%s",
        API_GATEWAY_VERSION,
        listen_host,
        settings.port,
        args.tls_cert,
    )
    web.run_app(
        app,
        host=listen_host,
        port=settings.port,
        handle_signals=True,
        ssl_context=ssl_context,
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """解析统一入口监听参数。"""

    parser = argparse.ArgumentParser(description="Dingtai API Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=443)
    parser.add_argument("--tls-cert", type=Path, default=DEFAULT_TLS_CERT_PATH)
    parser.add_argument("--tls-key", type=Path, default=DEFAULT_TLS_KEY_PATH)
    return parser.parse_args(argv)


def _create_ssl_context(cert_path: Path, key_path: Path) -> ssl.SSLContext:
    """创建 aiohttp 服务端 TLS 上下文，并在启动前检查证书文件。"""

    missing_paths = [path for path in (cert_path, key_path) if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"API Gateway TLS 文件不存在：{missing_text}")

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=cert_path, keyfile=key_path)
    return context


if __name__ == "__main__":
    raise SystemExit(main())
