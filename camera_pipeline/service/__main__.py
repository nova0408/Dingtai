from __future__ import annotations

import argparse
import signal
import threading
from collections.abc import Sequence
from types import FrameType

from loguru import logger

from ..pipeline_context import PipelineContext, PipelineContextConfig
from .application import CameraPipelineApplication
from .config import CameraPipelineServiceConfig
from .frame_publisher import CameraFramePublisher
from .logging_config import configure_service_logging, shutdown_service_logging
from .server import CameraPipelineServer
from .transport import CameraPipelineRpcServer, ZmqSocketOptions

def main(argv: Sequence[str] | None = None) -> int:
    """启动 Orin 本地 CameraPipeline 统一服务。"""

    args = _parse_args(argv)
    log_path = configure_service_logging(
        args.log_path,
        rotation=args.log_rotation,
        retention=args.log_retention,
    )
    logger.info(
        "camera pipeline service initializing bind_addr={} camera_name={} camera_id={} log_path={}",
        args.bind_addr,
        args.camera_name,
        args.camera_id,
        log_path,
    )
    stop_event = threading.Event()
    pipeline_context = PipelineContext(
        PipelineContextConfig(
            camera_control_port=args.control_port,
            camera_stream_port=args.stream_port,
            camera_id=args.camera_id,
            camera_name=args.camera_name,
        )
    )
    service_config = CameraPipelineServiceConfig(service_bind_addr=args.bind_addr)
    frame_publisher = CameraFramePublisher(
        pipeline_context,
        frame_bind_addr=service_config.frame_bind_addr,
        color_bind_addr=service_config.color_bind_addr,
        depth_bind_addr=service_config.depth_bind_addr,
    )
    application = CameraPipelineApplication(pipeline_context, frame_publisher)
    transport = CameraPipelineRpcServer(
        args.bind_addr,
        options=ZmqSocketOptions(
            receive_timeout_ms=service_config.request_receive_timeout_ms,
            send_timeout_ms=service_config.response_send_timeout_ms,
        ),
    )
    server = CameraPipelineServer(transport, application)

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.warning("camera pipeline service received stop signal={}", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        pipeline_context.start()
        logger.info("camera pipeline service started bind_addr={}", args.bind_addr)
        server.serve(stop_event)
    except Exception:
        logger.exception("camera pipeline service terminated by unhandled error")
        raise
    finally:
        logger.info("camera pipeline service stopping")
        try:
            frame_publisher.close()
            transport.close()
            pipeline_context.close()
            logger.info("camera pipeline service stopped")
        finally:
            shutdown_service_logging()
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    service_defaults = CameraPipelineServiceConfig()
    camera_defaults = PipelineContextConfig()
    parser = argparse.ArgumentParser(description="Orin CameraPipeline unified service")
    parser.add_argument("--bind-addr", default=service_defaults.service_bind_addr)
    parser.add_argument(
        "--control-port", type=int, default=camera_defaults.camera_control_port
    )
    parser.add_argument(
        "--stream-port", type=int, default=camera_defaults.camera_stream_port
    )
    parser.add_argument("--camera-id", default=camera_defaults.camera_id)
    parser.add_argument("--camera-name", default=camera_defaults.camera_name)
    parser.add_argument("--log-path", default=service_defaults.log_path)
    parser.add_argument("--log-rotation", default=service_defaults.log_rotation)
    parser.add_argument("--log-retention", default=service_defaults.log_retention)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
