from __future__ import annotations

import argparse
import logging
import signal
import pickle
import threading
import time

import zmq

from .pipeline_context import PipelineContext, PipelineContextConfig
from .ports import (
    CAMERA_PIPELINE_FRAME_STREAM_BIND_ADDR,
    CAMERA_PIPELINE_SERVICE_BIND_ADDR,
    DEFAULT_CAMERA_ID,
    DEFAULT_CAMERA_NAME,
    DEFAULT_CONTROL_PORT,
    DEFAULT_STREAM_PORT,
)
from .unified_protocol import (
    CameraFrameSubscribeResponse,
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraPipelineServiceResponse,
    CameraStatusResponse,
    CameraSummaryResponse,
)
from .unified_transport import CameraPipelineRpcServer, ZmqSocketOptions

LOGGER = logging.getLogger("..unified_service")


class CameraPipelineUnifiedService:
    """统一远端 camera pipeline 服务。"""

    def __init__(self, context: PipelineContext) -> None:
        self._context = context
        self._frame_pub_socket = zmq.Context.instance().socket(zmq.PUB)
        self._frame_pub_socket.bind(CAMERA_PIPELINE_FRAME_STREAM_BIND_ADDR)
        self._frame_pub_running = False
        self._frame_pub_thread: threading.Thread | None = None
        self._tray_service = None
        self._opening_service = None
        self._ball_service = None

    def handle(self, request: CameraPipelineServiceRequest) -> CameraPipelineServiceResponse:
        if request.operation == "camera_summary":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_summary=self._handle_camera_summary(request),
            )
        if request.operation == "camera_intrinsics":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_intrinsics=self._handle_camera_intrinsics(request),
            )
        if request.operation == "camera_status":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_status=self._handle_camera_status(request),
            )
        if request.operation == "camera_frame_subscribe":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_frame_subscribe=self._handle_camera_frame_subscribe(request),
            )
        if request.operation == "tray_detection":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                tray_detection=self._handle_tray_detection(request),
            )
        if request.operation == "opening_detection":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                opening_detection=self._handle_opening_detection(request),
            )
        if request.operation == "ball_pose_detection":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                ball_pose_detection=self._handle_ball_pose_detection(request),
            )
        raise RuntimeError(f"unsupported operation: {request.operation}")

    def _handle_camera_summary(self, request: CameraPipelineServiceRequest) -> CameraSummaryResponse:
        camera_request = request.camera_summary
        if camera_request is None:
            raise RuntimeError("camera_summary payload missing")
        if not self._context.wait_until_ready(timeout_s=float(camera_request.timeout_s)):
            raise RuntimeError(f"camera first frame not ready within {float(camera_request.timeout_s):.1f}s")
        frame = self._context.get_latest_frame()
        if frame is None:
            raise RuntimeError("camera first frame is still empty after ready wait")
        return CameraSummaryResponse(
            frame_id=int(frame.frame_id),
            camera_name=str(frame.camera_name),
            timestamp_ms=float(frame.timestamp_ms),
            color_shape=(int(frame.color_bgr.shape[0]), int(frame.color_bgr.shape[1]), int(frame.color_bgr.shape[2])),
            depth_shape=(int(frame.depth_mm.shape[0]), int(frame.depth_mm.shape[1])),
            fx=float(frame.fx),
            fy=float(frame.fy),
            cx=float(frame.cx),
            cy=float(frame.cy),
            source_meta=dict(frame.source_meta),
            error=None,
        )

    def _handle_camera_intrinsics(self, request: CameraPipelineServiceRequest) -> CameraIntrinsicsResponse:
        intrinsics_request = request.camera_intrinsics
        if intrinsics_request is None:
            raise RuntimeError("camera_intrinsics payload missing")
        if not self._context.wait_until_ready(timeout_s=float(intrinsics_request.timeout_s)):
            raise RuntimeError(f"camera intrinsics not ready within {float(intrinsics_request.timeout_s):.1f}s")
        frame = self._context.get_latest_frame()
        if frame is None:
            raise RuntimeError("camera intrinsics unavailable")
        return CameraIntrinsicsResponse(
            camera_name=str(frame.camera_name),
            fx=float(frame.fx),
            fy=float(frame.fy),
            cx=float(frame.cx),
            cy=float(frame.cy),
            distortion=tuple(),
            width=int(frame.color_bgr.shape[1]),
            height=int(frame.color_bgr.shape[0]),
            error=None,
        )

    def _handle_camera_status(self, request: CameraPipelineServiceRequest) -> CameraStatusResponse:
        status_request = request.camera_status
        if status_request is None:
            raise RuntimeError("camera_status payload missing")
        if not self._context.wait_until_ready(timeout_s=float(status_request.timeout_s)):
            raise RuntimeError(f"camera status not ready within {float(status_request.timeout_s):.1f}s")
        frame = self._context.get_latest_frame()
        if frame is None:
            raise RuntimeError("camera status unavailable")
        return CameraStatusResponse(
            camera_name=str(frame.camera_name),
            camera_id=str(self._context.get_camera_id()),
            camera_model=str(frame.source_meta.get("camera_model", "unknown")),
            width=int(frame.color_bgr.shape[1]),
            height=int(frame.color_bgr.shape[0]),
            color_enabled=True,
            depth_enabled=True,
            online=True,
            source_meta=dict(frame.source_meta),
            error=None,
        )

    def _handle_camera_frame_subscribe(self, request: CameraPipelineServiceRequest) -> CameraFrameSubscribeResponse:
        subscribe_request = request.camera_frame_subscribe
        if subscribe_request is None:
            raise RuntimeError("camera_frame_subscribe payload missing")
        self._ensure_frame_publisher_started()
        return CameraFrameSubscribeResponse(
            stream_addr=CAMERA_PIPELINE_FRAME_STREAM_BIND_ADDR,
            camera_name=str(subscribe_request.camera_name),
            error=None,
        )

    def _ensure_frame_publisher_started(self) -> None:
        if self._frame_pub_running:
            return
        self._frame_pub_running = True
        self._frame_pub_thread = threading.Thread(target=self._frame_pub_loop, name="camera-frame-pub", daemon=True)
        self._frame_pub_thread.start()

    def _frame_pub_loop(self) -> None:
        last_frame_id = -1
        while self._frame_pub_running:
            frame = self._context.get_latest_frame()
            if frame is None or int(frame.frame_id) == int(last_frame_id):
                time.sleep(0.02)
                continue
            last_frame_id = int(frame.frame_id)
            try:
                self._frame_pub_socket.send(pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL), flags=zmq.NOBLOCK)
            except zmq.error.Again:
                pass
            time.sleep(0.01)

    def close(self) -> None:
        self._frame_pub_running = False
        if self._frame_pub_thread is not None:
            self._frame_pub_thread.join(timeout=1.0)
            self._frame_pub_thread = None
        self._frame_pub_socket.close(linger=0)

    def _handle_tray_detection(self, request: CameraPipelineServiceRequest) -> OrinTrayDetectionResponse:
        from .tray_detection.protocol import OrinTrayDetectionRequest, OrinTrayDetectionResponse
        from .tray_detection.service import OrinTrayDetectionService

        if self._tray_service is None:
            self._tray_service = OrinTrayDetectionService()
        tray_request = request.tray_detection
        if tray_request is None:
            raise RuntimeError("tray_detection payload missing")
        frame = self._context.resolve_frame(tray_request.frame_id)
        return self._tray_service.compute(frame, tray_request)

    def _handle_opening_detection(self, request: CameraPipelineServiceRequest) -> OpeningDetectionPipelineResponse:
        from .opening_detection.protocol import DebugArtifacts, OpeningDetectionPipelineResponse, TrayPoseInfo
        from .opening_detection.service import OpeningDetectionPipelineService
        from .tray_detection.protocol import OrinTrayDetectionRequest

        if self._opening_service is None:
            self._opening_service = OpeningDetectionPipelineService()
        if self._tray_service is None:
            from .tray_detection.service import OrinTrayDetectionService

            self._tray_service = OrinTrayDetectionService()
        opening_request = request.opening_detection
        if opening_request is None:
            raise RuntimeError("opening_detection payload missing")
        frame = self._context.resolve_frame(opening_request.frame_id)
        tray_response = self._tray_service.compute(
            frame,
            OrinTrayDetectionRequest(
                request_id=int(opening_request.request_id),
                camera_name=str(opening_request.camera_name),
                frame_id=int(opening_request.frame_id),
                enable_debug=bool(opening_request.enable_debug),
            ),
        )
        if tray_response.error is not None:
            raise RuntimeError(tray_response.error)
        target_tray_index = int(opening_request.target_tray_index)
        if target_tray_index < 0 or target_tray_index >= len(tray_response.tray_results):
            raise RuntimeError(f"目标托盘索引越界：{target_tray_index}")
        debug = tray_response.debug
        if debug is None or len(debug.tray_masks) <= target_tray_index:
            raise RuntimeError("tray detection debug masks unavailable")
        tray_pose, opening_debug = self._opening_service.compute(
            frame=frame,
            tray_mask=debug.tray_masks[target_tray_index],
            request_id=int(opening_request.request_id),
            target_tray_index=target_tray_index,
            enable_debug=bool(opening_request.enable_debug),
        )
        if not isinstance(tray_pose, TrayPoseInfo):
            raise RuntimeError("opening detection returned invalid tray pose result")
        if opening_debug is not None and not isinstance(opening_debug, DebugArtifacts):
            raise RuntimeError("opening detection returned invalid debug artifacts")
        return OpeningDetectionPipelineResponse(
            request_id=int(opening_request.request_id),
            frame_id=int(tray_response.frame_id),
            camera_name=str(tray_response.camera_name),
            timestamp_ms=float(tray_response.timestamp_ms),
            source_meta=dict(tray_response.source_meta),
            elapsed_ms=float(tray_response.elapsed_ms),
            tray_count=int(tray_response.tray_count),
            tray_results=tray_response.tray_results,
            selected_tray_index=target_tray_index,
            selected_result=tray_pose,
            all_tray_results=(tray_pose,),
            debug=opening_debug,
            error=None,
        )

    def _handle_ball_pose_detection(self, request: CameraPipelineServiceRequest) -> BallPoseDetectionResponse:
        from .ball_pose_detection.protocol import BallPoseDetectionResponse
        from .ball_pose_detection.service import BallPoseDetectionService

        if self._ball_service is None:
            self._ball_service = BallPoseDetectionService()
        ball_request = request.ball_pose_detection
        if ball_request is None:
            raise RuntimeError("ball_pose_detection payload missing")
        frame = self._context.resolve_frame(ball_request.frame_id)
        return self._ball_service.compute(frame, ball_request)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Unified camera pipeline service")
    parser.add_argument("--bind-addr", type=str, default=CAMERA_PIPELINE_SERVICE_BIND_ADDR)
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT)
    parser.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT)
    parser.add_argument("--camera-id", type=str, default=DEFAULT_CAMERA_ID)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    context = PipelineContext(
        PipelineContextConfig(
            camera_control_port=int(args.control_port),
            camera_stream_port=int(args.stream_port),
            camera_id=str(args.camera_id),
            camera_name=str(args.camera_name),
        )
    )
    service = CameraPipelineUnifiedService(context=context)
    server = CameraPipelineRpcServer(str(args.bind_addr), options=ZmqSocketOptions())
    running = True

    def _handle_signal(signum, _frame) -> None:  # noqa: ANN001
        nonlocal running
        LOGGER.info("received stop signal %s", signum)
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        context.start()
        LOGGER.info("camera pipeline unified service started")
        while running:
            try:
                request = server.recv_request()
            except zmq.error.Again:
                continue
            try:
                response = service.handle(request)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("camera pipeline unified service failed: %s", exc)
                response = CameraPipelineServiceResponse(
                    operation=request.operation,
                    error="{0}: {1}".format(type(exc).__name__, exc),
                )
            server.send_response(response)
    finally:
        server.close()
        context.close()
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
