from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

DEFAULT_LOG_PATH = "logs/api_gateway.log"
LOG_BACKUP_COUNT = 24 * 7

_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-7s | "
    "%(process)d:%(threadName)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_service_logging(log_path: str) -> Path:
    """初始化 API Gateway 独立的控制台和按小时轮转文件日志。"""

    resolved_path = Path(log_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = TimedRotatingFileHandler(
        resolved_path,
        when="H",
        interval=1,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=False,
        utc=False,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d_%H"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)
    logging.getLogger(__name__).info(
        "service logging initialized service=api_gateway path=%s rotation=1_hour retention=7_days level=INFO",
        resolved_path,
    )
    return resolved_path


def shutdown_service_logging() -> None:
    """刷新并关闭 API Gateway 服务入口创建的全部日志 handler。"""

    logging.getLogger(__name__).info(
        "service logging shutting down service=api_gateway"
    )
    logging.shutdown()
