"""RobotControl HTTP 服务启动入口。"""

from __future__ import annotations

import argparse
import signal
from collections.abc import Sequence
from types import FrameType

from loguru import logger

from ..config import RobotControlSettings
from ..gateway import RobotControlGateway
from .application import RobotControlApplication
from .server import RobotControlServer


def main(argv: Sequence[str] | None = None) -> int:
    """启动 RobotControl 服务。

    服务启动不会主动执行运动；硬件对象按第一次 GET 或人工控制请求延迟创建。
    """

    args = _parse_args(argv)
    settings = RobotControlSettings(
        http_host=args.host,
        http_port=args.port,
        qmlinker_waist_available=args.qmlinker_waist,
    )
    application = RobotControlApplication(RobotControlGateway(settings))
    server = RobotControlServer(
        args.host,
        args.port,
        application,
        status_stream_interval_s=settings.status_stream_interval_s,
    )

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        """把终止信号转换为服务主循环退出。"""

        logger.warning("robot control service received stop signal={}", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    logger.info("robot control service started http://{}:{}", args.host, args.port)
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("robot control service stopping")
    finally:
        server.close()
        application.close()
        logger.info("robot control service stopped")
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """解析服务监听参数。"""

    parser = argparse.ArgumentParser(description="Dingtai RobotControl service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6500)
    parser.add_argument(
        "--qmlinker-waist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="声明当前机型是否支持 qmlinker 腰部状态能力。",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
