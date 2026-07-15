from __future__ import annotations

from typing import TYPE_CHECKING

from ..ball_pose_detection.protocol import BallPoseDetectionResponse
from ..protocol import RgbdFrameProtocol
from ..opening_detection.protocol import (
    OpeningDetectionPipelineResponse,
)
from ..pipeline_context import PipelineContext
from ..tray_detection.protocol import OrinTrayDetectionRequest
from ..tray_detection.protocol import OrinTrayDetectionResponse
from .frame_publisher import CameraFramePublisher
from .protocol import (
    PROTOCOL_VERSION,
    CameraColorFrameSubscribeResponse,
    CameraDepthFrameSubscribeResponse,
    CameraFrameSubscribeResponse,
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraPipelineServiceResponse,
    CameraStatusResponse,
    CameraSummaryResponse,
    StableFrameResponse,
)

if TYPE_CHECKING:
    from ..ball_pose_detection.service import BallPoseDetectionService
    from ..opening_detection.service import OpeningDetectionPipelineService
    from ..tray_detection.service import OrinTrayDetectionService


class CameraPipelineApplication:
    """编排 CameraPipeline 请求、帧选择和纯计算算法调用。

    本类不创建网络请求 socket、不运行服务主循环，也不直接访问相机
    上游协议。所有帧访问经由 `PipelineContext`，所有网络发布经由
    `CameraFramePublisher`。
    """

    def __init__(
        self,
        pipeline_context: PipelineContext,
        frame_publisher: CameraFramePublisher,
    ) -> None:
        self._pipeline_context = pipeline_context
        self._frame_publisher = frame_publisher
        self._tray_service: OrinTrayDetectionService | None = None
        self._opening_service: OpeningDetectionPipelineService | None = None
        self._ball_service: BallPoseDetectionService | None = None

    # region 请求分发

    def handle(
        self, request: CameraPipelineServiceRequest
    ) -> CameraPipelineServiceResponse:
        """显式分发一个经过传输层解码的请求。"""

        if request.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"unsupported protocol version {request.protocol_version}; expected {PROTOCOL_VERSION}"
            )
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
        if request.operation == "stable_frame":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                stable_frame=self._handle_stable_frame(request),
            )
        if request.operation == "camera_frame_subscribe":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_frame_subscribe=self._handle_frame_subscribe(request),
            )
        if request.operation == "camera_color_frame_subscribe":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_color_frame_subscribe=self._handle_color_frame_subscribe(
                    request
                ),
            )
        if request.operation == "camera_depth_frame_subscribe":
            return CameraPipelineServiceResponse(
                operation=request.operation,
                camera_depth_frame_subscribe=self._handle_depth_frame_subscribe(
                    request
                ),
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

    # endregion

    # region 相机查询与稳定帧

    def _handle_camera_summary(
        self, request: CameraPipelineServiceRequest
    ) -> CameraSummaryResponse:
        payload = request.camera_summary
        if payload is None:
            raise RuntimeError("camera_summary payload missing")
        frame = self._wait_for_latest_frame(payload.timeout_s, "camera first frame")
        return CameraSummaryResponse(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
            color_shape=(
                frame.color_bgr.shape[0],
                frame.color_bgr.shape[1],
                frame.color_bgr.shape[2],
            ),
            depth_shape=(frame.depth_mm.shape[0], frame.depth_mm.shape[1]),
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
        )

    def _handle_camera_intrinsics(
        self, request: CameraPipelineServiceRequest
    ) -> CameraIntrinsicsResponse:
        payload = request.camera_intrinsics
        if payload is None:
            raise RuntimeError("camera_intrinsics payload missing")
        frame = self._wait_for_latest_frame(
            payload.timeout_s,
            "camera intrinsics",
            camera_name=payload.camera_name,
        )
        return CameraIntrinsicsResponse(
            camera_name=frame.camera_name,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
            distortion=tuple(),
            width=frame.color_bgr.shape[1],
            height=frame.color_bgr.shape[0],
        )

    def _handle_camera_status(
        self, request: CameraPipelineServiceRequest
    ) -> CameraStatusResponse:
        payload = request.camera_status
        if payload is None:
            raise RuntimeError("camera_status payload missing")
        frame = self._wait_for_latest_frame(payload.timeout_s, "camera status")
        return CameraStatusResponse(
            camera_name=frame.camera_name,
            camera_id=self._pipeline_context.get_camera_id(frame.camera_name),
            camera_model="unknown",
            width=frame.color_bgr.shape[1],
            height=frame.color_bgr.shape[0],
            color_enabled=True,
            depth_enabled=True,
            online=True,
        )

    def _handle_stable_frame(
        self, request: CameraPipelineServiceRequest
    ) -> StableFrameResponse:
        payload = request.stable_frame
        if payload is None:
            raise RuntimeError("stable_frame payload missing")
        frame = self._pipeline_context.wait_for_stable_frame(
            timeout_s=payload.timeout_s,
            camera_name=payload.camera_name,
        )
        return StableFrameResponse(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
        )

    def _wait_for_latest_frame(
        self,
        timeout_s: float,
        description: str,
        camera_name: str | None = None,
    ) -> RgbdFrameProtocol:
        if not self._pipeline_context.wait_until_ready(
            timeout_s=timeout_s,
            camera_name=camera_name,
        ):
            raise RuntimeError(f"{description} not ready within {timeout_s:.1f}s")
        frame = self._pipeline_context.get_latest_frame(camera_name)
        if frame is None:
            raise RuntimeError(f"{description} unavailable")
        return frame

    # endregion

    # region 帧订阅

    def _handle_frame_subscribe(
        self, request: CameraPipelineServiceRequest
    ) -> CameraFrameSubscribeResponse:
        payload = request.camera_frame_subscribe
        if payload is None:
            raise RuntimeError("camera_frame_subscribe payload missing")
        self._pipeline_context.get_camera_runtime(payload.camera_name)
        self._frame_publisher.start()
        return CameraFrameSubscribeResponse(
            stream_addr=self._frame_publisher.frame_bind_addr,
            camera_name=payload.camera_name,
        )

    def _handle_color_frame_subscribe(
        self,
        request: CameraPipelineServiceRequest,
    ) -> CameraColorFrameSubscribeResponse:
        payload = request.camera_color_frame_subscribe
        if payload is None:
            raise RuntimeError("camera_color_frame_subscribe payload missing")
        self._pipeline_context.get_camera_runtime(payload.camera_name)
        self._frame_publisher.start()
        return CameraColorFrameSubscribeResponse(
            stream_addr=self._frame_publisher.color_bind_addr,
            camera_name=payload.camera_name,
        )

    def _handle_depth_frame_subscribe(
        self,
        request: CameraPipelineServiceRequest,
    ) -> CameraDepthFrameSubscribeResponse:
        payload = request.camera_depth_frame_subscribe
        if payload is None:
            raise RuntimeError("camera_depth_frame_subscribe payload missing")
        self._pipeline_context.get_camera_runtime(payload.camera_name)
        self._frame_publisher.start()
        return CameraDepthFrameSubscribeResponse(
            stream_addr=self._frame_publisher.depth_bind_addr,
            camera_name=payload.camera_name,
        )

    # endregion

    # region 算法业务编排

    def _handle_tray_detection(
        self, request: CameraPipelineServiceRequest
    ) -> OrinTrayDetectionResponse:
        payload = request.tray_detection
        if payload is None:
            raise RuntimeError("tray_detection payload missing")
        frame = self._pipeline_context.resolve_frame(payload.frame_id)
        return self._get_tray_service().compute(frame, payload)

    def _handle_opening_detection(
        self,
        request: CameraPipelineServiceRequest,
    ) -> OpeningDetectionPipelineResponse:
        payload = request.opening_detection
        if payload is None:
            raise RuntimeError("opening_detection payload missing")
        frame = self._pipeline_context.resolve_frame(payload.frame_id)
        tray_response = self._get_tray_service().compute(
            frame,
            OrinTrayDetectionRequest(
                request_id=payload.request_id,
                camera_name=payload.camera_name,
                frame_id=frame.frame_id,
                enable_debug=True,
            ),
        )
        target_index = payload.target_tray_index
        if target_index < 0 or target_index >= len(tray_response.tray_results):
            raise RuntimeError(f"target tray index out of range: {target_index}")
        if not tray_response.debug_artifacts:
            raise RuntimeError("tray detection masks unavailable for opening detection")
        tray_debug = tray_response.debug_artifacts[0]
        if len(tray_debug.tray_masks) <= target_index:
            raise RuntimeError("target tray mask unavailable for opening detection")

        tray_pose, opening_debug_artifacts = self._get_opening_service().compute(
            frame=frame,
            tray_mask=tray_debug.tray_masks[target_index],
            request_id=payload.request_id,
            target_tray_index=target_index,
            enable_debug=payload.enable_debug,
        )
        return OpeningDetectionPipelineResponse(
            request_id=payload.request_id,
            frame_id=tray_response.frame_id,
            camera_name=tray_response.camera_name,
            timestamp_ms=tray_response.timestamp_ms,
            elapsed_ms=tray_response.elapsed_ms,
            tray_count=tray_response.tray_count,
            tray_results=tray_response.tray_results,
            selected_tray_index=target_index,
            selected_result=tray_pose,
            all_tray_results=(tray_pose,),
            debug_artifacts=opening_debug_artifacts,
        )

    def _handle_ball_pose_detection(
        self, request: CameraPipelineServiceRequest
    ) -> BallPoseDetectionResponse:
        payload = request.ball_pose_detection
        if payload is None:
            raise RuntimeError("ball_pose_detection payload missing")
        frame = self._pipeline_context.resolve_frame(payload.frame_id)
        return self._get_ball_service().compute(frame, payload)

    def _get_tray_service(self) -> OrinTrayDetectionService:
        if self._tray_service is None:
            from ..tray_detection.service import OrinTrayDetectionService

            self._tray_service = OrinTrayDetectionService()
        return self._tray_service

    def _get_opening_service(self) -> OpeningDetectionPipelineService:
        if self._opening_service is None:
            from ..opening_detection.service import OpeningDetectionPipelineService

            self._opening_service = OpeningDetectionPipelineService()
        return self._opening_service

    def _get_ball_service(self) -> BallPoseDetectionService:
        if self._ball_service is None:
            from ..ball_pose_detection.service import BallPoseDetectionService

            self._ball_service = BallPoseDetectionService()
        return self._ball_service

    # endregion
