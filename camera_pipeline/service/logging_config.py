from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | "
    "{process.id}:{thread.name} | {name}:{function}:{line} | {message}"
)


def configure_service_logging(
    log_path: str,
    *,
    rotation: str,
    retention: str,
) -> Path:
    """初始化 CameraPipeline 服务唯一的控制台和轮转文件日志 sink。"""

    resolved_path = Path(log_path).expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.enable("camera_pipeline")
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
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=False,
        diagnose=False,
        enqueue=True,
    )
    logger.info(
        "service logging initialized path={} rotation={} retention={} level=INFO",
        resolved_path,
        rotation,
        retention,
    )
    return resolved_path


def shutdown_service_logging() -> None:
    """刷新并关闭服务入口创建的全部 Loguru sink。"""

    logger.info("service logging shutting down")
    logger.remove()
    logger.disable("camera_pipeline")
