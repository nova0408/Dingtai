from __future__ import annotations

import argparse
import logging
import signal
import threading
from collections.abc import Sequence

from ..pipeline_context import PipelineContext, PipelineContextConfig
from .application import CameraPipelineApplication
from .config import CameraPipelineServiceConfig
from .frame_publisher import CameraFramePublisher
from .server import CameraPipelineServer
from .transport import CameraPipelineRpcServer, ZmqSocketOptions

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """启动 Orin 本地 CameraPipeline 统一服务。"""

    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
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

    def _handle_signal(signum: int, _frame: object) -> None:
        LOGGER.info("received stop signal %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        pipeline_context.start()
        LOGGER.info("camera pipeline service started at %s", args.bind_addr)
        server.serve(stop_event)
    finally:
        frame_publisher.close()
        transport.close()
        pipeline_context.close()
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
