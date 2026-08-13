"""手眼标定与先验记录服务入口。"""

from __future__ import annotations

import argparse
import signal
import time
from collections.abc import Sequence
from types import FrameType

from loguru import logger

from .. import CALIBRATION_SERVICE_VERSION
from .application import CalibrationApplication
from .camera_client import CameraPipelineHttpClient
from .logging_config import (
    DEFAULT_LOG_PATH,
    configure_service_logging,
    shutdown_service_logging,
)
from .server import CalibrationServer


def main(argv: Sequence[str] | None = None) -> int:
    """启动只负责拍摄和计算的常驻 HTTP 服务。"""

    startup_started_at = time.perf_counter()
    args = _parse_args(argv)
    log_path = configure_service_logging(args.log_path)
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
    logger.info(
        "calibration service started version={} http://{}:{} camera_url={} camera_timeout_s={} log_path={} startup_elapsed_ms={:.3f}",
        CALIBRATION_SERVICE_VERSION,
        args.host,
        args.port,
        args.camera_url,
        args.camera_timeout_s,
        log_path,
        (time.perf_counter() - startup_started_at) * 1000.0,
    )
    try:
        server.serve()
    except KeyboardInterrupt:
        logger.info("calibration service stopping")
    except Exception:
        logger.exception("calibration service terminated by unhandled error")
        raise
    finally:
        shutdown_started_at = time.perf_counter()
        server.close()
        logger.info(
            "calibration service stopped shutdown_elapsed_ms={:.3f}",
            (time.perf_counter() - shutdown_started_at) * 1000.0,
        )
        shutdown_service_logging()
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dingtai calibration service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6600)
    parser.add_argument("--camera-url", default="http://127.0.0.1:6400")
    parser.add_argument("--camera-timeout-s", type=float, default=30.0)
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
