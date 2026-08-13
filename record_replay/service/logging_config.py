from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

DEFAULT_LOG_PATH = "logs/record_replay.log"
LOG_ROTATION = "1 hour"
LOG_RETENTION = "7 days"

_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | "
    "{process.id}:{thread.name} | {name}:{function}:{line} | {message}"
)


def configure_service_logging(log_path: str) -> Path:
    """初始化 RecordReplay 独立的控制台和按小时轮转文件日志。"""

    resolved_path = Path(log_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=_LOG_FORMAT,
        colorize=False,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )
    logger.add(
        resolved_path,
        level="INFO",
        format=_LOG_FORMAT,
        encoding="utf-8",
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        compression="zip",
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )
    logger.info(
        "service logging initialized service=record_replay path={} rotation={} retention={} level=INFO",
        resolved_path,
        LOG_ROTATION,
        LOG_RETENTION,
    )
    return resolved_path


def shutdown_service_logging() -> None:
    """刷新并关闭 RecordReplay 服务入口创建的全部日志 sink。"""

    logger.info("service logging shutting down service=record_replay")
    logger.remove()
