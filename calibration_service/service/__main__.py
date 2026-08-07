"""手眼标定与先验记录服务入口。"""

from __future__ import annotations

import argparse
import signal
from collections.abc import Sequence
from types import FrameType

from loguru import logger

from .application import CalibrationApplication
from .camera_client import CameraPipelineHttpClient
from .server import CalibrationServer


def main(argv: Sequence[str] | None = None) -> int:
    """启动只负责拍摄和计算的常驻 HTTP 服务。"""

    args = _parse_args(argv)
    application = CalibrationApplication(
        lambda: CameraPipelineHttpClient(
            base_url=args.camera_url,
            timeout_s=args.camera_timeout_s,
        )
    )
    server = CalibrationServer(args.host, args.port, application)

    def stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("calibration service received stop signal={}", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    logger.info("calibration service started http://{}:{}", args.host, args.port)
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("calibration service stopping")
    finally:
        server.close()
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dingtai calibration service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6600)
    parser.add_argument("--camera-url", default="http://127.0.0.1:6400")
    parser.add_argument("--camera-timeout-s", type=float, default=30.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
