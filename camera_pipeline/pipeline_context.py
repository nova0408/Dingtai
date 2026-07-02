from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from .camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig
from .ports import BALL_POSE_DETECTION_CONNECT_ADDR, TRAY_DETECTION_CONNECT_ADDR
from .ball_pose_detection.protocol import BallPoseDetectionRequest, BallPoseDetectionResponse
from .ball_pose_detection.transport import BallPoseDetectionRpcClient, ZmqSocketOptions as BallZmqSocketOptions
from .tray_detection.protocol import OrinTrayDetectionRequest, OrinTrayDetectionResponse
from .tray_detection.transport import OrinTrayDetectionRpcClient, ZmqSocketOptions as TrayZmqSocketOptions


@dataclass(frozen=True)
class PipelineContextConfig:
    """总流程上下文配置。"""

    camera_runtime: CameraStreamRuntimeConfig
    tray_rpc_addr: str = TRAY_DETECTION_CONNECT_ADDR
    tray_rpc_timeout_ms: int = 30000
    ball_rpc_addr: str = BALL_POSE_DETECTION_CONNECT_ADDR
    ball_rpc_timeout_ms: int = 30000


class PipelineContext:
    """统一管理相机流、托盘 RPC 和数据输入输出的上下文。"""

    def __init__(self, config: PipelineContextConfig) -> None:
        self._config = config
        self._frame_runtime = CameraStreamRuntime(config.camera_runtime)
        self._ball_client = BallPoseDetectionRpcClient(
            connect_addr=str(config.ball_rpc_addr),
            options=BallZmqSocketOptions(
                recv_timeout_ms=int(config.ball_rpc_timeout_ms),
                send_timeout_ms=int(config.ball_rpc_timeout_ms),
            ),
        )
        self._tray_client = OrinTrayDetectionRpcClient(
            connect_addr=str(config.tray_rpc_addr),
            options=TrayZmqSocketOptions(
                recv_timeout_ms=int(config.tray_rpc_timeout_ms),
                send_timeout_ms=int(config.tray_rpc_timeout_ms),
            ),
        )

    def start(self) -> None:
        self._frame_runtime.start()

    def close(self) -> None:
        self._ball_client.close()
        self._tray_client.close()
        self._frame_runtime.stop()

    def wait_until_ready(self, timeout_s: float = 8.0) -> bool:
        return self._frame_runtime.wait_until_ready(timeout_s=timeout_s)

    def get_camera_runtime(self) -> CameraStreamRuntime:
        return self._frame_runtime

    def get_latest_frame(self):
        return self._frame_runtime.get_latest_frame()

    def get_frame_by_id(self, frame_id: int):
        return self._frame_runtime.get_frame_by_id(frame_id)

    def resolve_frame(self, frame_id: int):
        if int(frame_id) > 0:
            frame = self.get_frame_by_id(int(frame_id))
            if frame is not None:
                return frame
        frame = self.get_latest_frame()
        if frame is None:
            raise RuntimeError("camera frame not ready")
        return frame

    def request_tray_detection(self, request: OrinTrayDetectionRequest) -> OrinTrayDetectionResponse:
        return self._tray_client.call(request)

    def build_tray_detection_request(self, request_id: int, camera_name: str, frame_id: int, enable_debug: bool = True) -> OrinTrayDetectionRequest:
        return OrinTrayDetectionRequest(
            request_id=int(request_id),
            camera_name=str(camera_name),
            frame_id=int(frame_id),
            enable_debug=bool(enable_debug),
        )

    def request_tray_detection_for_frame(
        self,
        request_id: int,
        camera_name: str,
        frame_id: int,
        enable_debug: bool = True,
    ) -> OrinTrayDetectionResponse:
        return self.request_tray_detection(
            self.build_tray_detection_request(
                request_id=request_id,
                camera_name=camera_name,
                frame_id=frame_id,
                enable_debug=enable_debug,
            )
        )

    def request_ball_pose_detection(self, request: BallPoseDetectionRequest) -> BallPoseDetectionResponse:
        return self._ball_client.call(request)

    def request_ball_pose_detection_for_frame(
        self,
        request_id: int,
        camera_name: str,
        frame_id: int,
        priors: tuple,
        reference_relative_transform_mm: Optional[Tuple[Tuple[float, float, float, float], ...]] = None,
        enable_debug: bool = True,
    ) -> BallPoseDetectionResponse:
        return self.request_ball_pose_detection(
            BallPoseDetectionRequest(
                request_id=int(request_id),
                camera_name=str(camera_name),
                frame_id=int(frame_id),
                enable_debug=bool(enable_debug),
                priors=tuple(priors),
                reference_relative_transform_mm=reference_relative_transform_mm,
            )
        )
