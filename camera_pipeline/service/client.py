from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

import zmq

from ..ball_pose_detection.protocol import (
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
)
from ..protocol import CameraColorFramePacket, CameraDepthFramePacket, CameraFramePacket
from .protocol import (
    CameraColorFrameSubscribeRequest,
    CameraDepthFrameSubscribeRequest,
    CameraFrameSubscribeRequest,
    CameraIntrinsicsRequest,
    CameraIntrinsicsResponse,
    CameraPipelineServiceRequest,
    CameraStatusRequest,
    CameraStatusResponse,
    CameraSummaryRequest,
    CameraSummaryResponse,
    StableFrameRequest,
    StableFrameResponse,
    build_camera_stream_topic,
)
from .transport import CameraPipelineRpcClient, ZmqSocketOptions
from .wire_codec import decode_wire

PacketT = TypeVar(
    "PacketT",
    bound=CameraFramePacket | CameraColorFramePacket | CameraDepthFramePacket,
)
_TCP_PREFIX = "tcp://"
_HEAD_CAMERA_NAME = "head_camera"
_CHEST_CAMERA_NAME = "chest_camera"
_LEFT_ARM_CAMERA_NAME = "left_hand_camera"
_RIGHT_ARM_CAMERA_NAME = "right_hand_camera"


class CameraPipelineClient:
    """外接开发机和 Orin 本地业务服务共用的公共客户端。"""

    def __init__(
        self,
        service_addr: str = "tcp://127.0.0.1:6200",
        timeout_ms: int = 30_000,
    ) -> None:
        self._service_addr = service_addr
        self._rpc_client = CameraPipelineRpcClient(
            connect_addr=service_addr,
            options=ZmqSocketOptions(
                receive_timeout_ms=timeout_ms,
                send_timeout_ms=timeout_ms,
            ),
        )

    def close(self) -> None:
        self._rpc_client.close()

    # region 相机查询

    def get_camera_summary(self, timeout_s: float = 10.0) -> CameraSummaryResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_summary",
                camera_summary=CameraSummaryRequest(timeout_s=timeout_s),
            )
        )
        if response.error is not None or response.camera_summary is None:
            raise RuntimeError(response.error or "camera summary response missing")
        return response.camera_summary

    def get_camera_intrinsics(
        self,
        camera_name: str = _LEFT_ARM_CAMERA_NAME,
        timeout_s: float = 10.0,
    ) -> CameraIntrinsicsResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_intrinsics",
                camera_intrinsics=CameraIntrinsicsRequest(
                    camera_name=camera_name,
                    timeout_s=timeout_s,
                ),
            )
        )
        if response.error is not None or response.camera_intrinsics is None:
            raise RuntimeError(response.error or "camera intrinsics response missing")
        return response.camera_intrinsics

    def get_head_camera_intrinsics(
        self,
        timeout_s: float = 10.0,
    ) -> CameraIntrinsicsResponse:
        """读取头部相机内参。"""

        return self.get_camera_intrinsics(_HEAD_CAMERA_NAME, timeout_s)

    def get_chest_camera_intrinsics(
        self,
        timeout_s: float = 10.0,
    ) -> CameraIntrinsicsResponse:
        """读取胸腔相机内参。"""

        return self.get_camera_intrinsics(_CHEST_CAMERA_NAME, timeout_s)

    def get_left_arm_camera_intrinsics(
        self,
        timeout_s: float = 10.0,
    ) -> CameraIntrinsicsResponse:
        """读取左臂相机内参。"""

        return self.get_camera_intrinsics(_LEFT_ARM_CAMERA_NAME, timeout_s)

    def get_right_arm_camera_intrinsics(
        self,
        timeout_s: float = 10.0,
    ) -> CameraIntrinsicsResponse:
        """读取右臂相机内参；未连接时服务端明确报错。"""

        return self.get_camera_intrinsics(_RIGHT_ARM_CAMERA_NAME, timeout_s)

    def get_camera_status(self, timeout_s: float = 10.0) -> CameraStatusResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_status",
                camera_status=CameraStatusRequest(timeout_s=timeout_s),
            )
        )
        if response.error is not None or response.camera_status is None:
            raise RuntimeError(response.error or "camera status response missing")
        return response.camera_status

    def get_stable_frame(
        self,
        camera_name: str = _LEFT_ARM_CAMERA_NAME,
        timeout_s: float = 10.0,
    ) -> StableFrameResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="stable_frame",
                stable_frame=StableFrameRequest(
                    camera_name=camera_name,
                    timeout_s=timeout_s,
                ),
            )
        )
        if response.error is not None or response.stable_frame is None:
            raise RuntimeError(response.error or "stable frame response missing")
        return response.stable_frame

    def get_head_camera_stable_frame(
        self,
        timeout_s: float = 10.0,
    ) -> StableFrameResponse:
        """等待并返回头部相机稳定帧。"""

        return self.get_stable_frame(_HEAD_CAMERA_NAME, timeout_s)

    def get_chest_camera_stable_frame(
        self,
        timeout_s: float = 10.0,
    ) -> StableFrameResponse:
        """等待并返回胸腔相机稳定帧。"""

        return self.get_stable_frame(_CHEST_CAMERA_NAME, timeout_s)

    def get_left_arm_camera_stable_frame(
        self,
        timeout_s: float = 10.0,
    ) -> StableFrameResponse:
        """等待并返回左臂相机稳定帧。"""

        return self.get_stable_frame(_LEFT_ARM_CAMERA_NAME, timeout_s)

    def get_right_arm_camera_stable_frame(
        self,
        timeout_s: float = 10.0,
    ) -> StableFrameResponse:
        """等待右臂相机稳定帧；未连接时服务端明确报错。"""

        return self.get_stable_frame(_RIGHT_ARM_CAMERA_NAME, timeout_s)

    # endregion

    # region 帧订阅

    def subscribe_camera_frames(
        self, camera_name: str = "left_hand_camera"
    ) -> Iterator[CameraFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_frame_subscribe",
                camera_frame_subscribe=CameraFrameSubscribeRequest(
                    camera_name=camera_name
                ),
            )
        )
        if response.error is not None or response.camera_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_frame_subscribe.stream_addr,
            camera_name,
            CameraFramePacket,
        )

    def subscribe_camera_color_frames(
        self,
        camera_name: str = "left_hand_camera",
    ) -> Iterator[CameraColorFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_color_frame_subscribe",
                camera_color_frame_subscribe=CameraColorFrameSubscribeRequest(
                    camera_name=camera_name
                ),
            )
        )
        if response.error is not None or response.camera_color_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera color frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_color_frame_subscribe.stream_addr,
            camera_name,
            CameraColorFramePacket,
        )

    def subscribe_camera_depth_frames(
        self,
        camera_name: str = "left_hand_camera",
    ) -> Iterator[CameraDepthFramePacket]:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="camera_depth_frame_subscribe",
                camera_depth_frame_subscribe=CameraDepthFrameSubscribeRequest(
                    camera_name=camera_name
                ),
            )
        )
        if response.error is not None or response.camera_depth_frame_subscribe is None:
            raise RuntimeError(
                response.error or "camera depth frame subscribe response missing"
            )
        yield from self._subscribe_stream(
            response.camera_depth_frame_subscribe.stream_addr,
            camera_name,
            CameraDepthFramePacket,
        )

    # region 明确安装位订阅 API

    def subscribe_head_camera_frames(self) -> Iterator[CameraFramePacket]:
        """订阅头部相机 RGBD 帧。"""

        return self.subscribe_camera_frames(_HEAD_CAMERA_NAME)

    def subscribe_chest_camera_frames(self) -> Iterator[CameraFramePacket]:
        """订阅胸腔相机 RGBD 帧。"""

        return self.subscribe_camera_frames(_CHEST_CAMERA_NAME)

    def subscribe_left_arm_camera_frames(self) -> Iterator[CameraFramePacket]:
        """订阅左臂相机 RGBD 帧。"""

        return self.subscribe_camera_frames(_LEFT_ARM_CAMERA_NAME)

    def subscribe_right_arm_camera_frames(self) -> Iterator[CameraFramePacket]:
        """订阅右臂相机 RGBD 帧；未连接时服务端明确报错。"""

        return self.subscribe_camera_frames(_RIGHT_ARM_CAMERA_NAME)

    def subscribe_head_camera_color_frames(self) -> Iterator[CameraColorFramePacket]:
        """订阅头部相机彩色帧。"""

        return self.subscribe_camera_color_frames(_HEAD_CAMERA_NAME)

    def subscribe_chest_camera_color_frames(self) -> Iterator[CameraColorFramePacket]:
        """订阅胸腔相机彩色帧。"""

        return self.subscribe_camera_color_frames(_CHEST_CAMERA_NAME)

    def subscribe_left_arm_camera_color_frames(
        self,
    ) -> Iterator[CameraColorFramePacket]:
        """订阅左臂相机彩色帧。"""

        return self.subscribe_camera_color_frames(_LEFT_ARM_CAMERA_NAME)

    def subscribe_right_arm_camera_color_frames(
        self,
    ) -> Iterator[CameraColorFramePacket]:
        """订阅右臂相机彩色帧；未连接时服务端明确报错。"""

        return self.subscribe_camera_color_frames(_RIGHT_ARM_CAMERA_NAME)

    def subscribe_head_camera_depth_frames(self) -> Iterator[CameraDepthFramePacket]:
        """订阅头部相机深度帧。"""

        return self.subscribe_camera_depth_frames(_HEAD_CAMERA_NAME)

    def subscribe_chest_camera_depth_frames(self) -> Iterator[CameraDepthFramePacket]:
        """订阅胸腔相机深度帧。"""

        return self.subscribe_camera_depth_frames(_CHEST_CAMERA_NAME)

    def subscribe_left_arm_camera_depth_frames(
        self,
    ) -> Iterator[CameraDepthFramePacket]:
        """订阅左臂相机深度帧。"""

        return self.subscribe_camera_depth_frames(_LEFT_ARM_CAMERA_NAME)

    def subscribe_right_arm_camera_depth_frames(
        self,
    ) -> Iterator[CameraDepthFramePacket]:
        """订阅右臂相机深度帧；未连接时服务端明确报错。"""

        return self.subscribe_camera_depth_frames(_RIGHT_ARM_CAMERA_NAME)

    # endregion

    def _subscribe_stream(
        self,
        stream_addr: str,
        camera_name: str,
        packet_type: type[PacketT],
    ) -> Iterator[PacketT]:
        connect_addr = self._resolve_stream_connect_addr(stream_addr)
        topic = build_camera_stream_topic(camera_name)
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVHWM, 1)
        socket.setsockopt(zmq.SUBSCRIBE, topic)
        socket.setsockopt(zmq.RCVTIMEO, 10_000)
        socket.connect(connect_addr)
        try:
            while True:
                message = socket.recv()
                if not message.startswith(topic):
                    raise RuntimeError(f"camera stream topic mismatch: {camera_name}")
                yield decode_wire(message[len(topic) :], packet_type)
        finally:
            socket.close(linger=0)

    def _resolve_stream_connect_addr(self, stream_addr: str) -> str:
        service_addr = self._service_addr
        if service_addr.startswith(_TCP_PREFIX):
            service_addr = service_addr[len(_TCP_PREFIX) :]
        service_host = service_addr.split(":")[0]
        if stream_addr.startswith("tcp://0.0.0.0:") or stream_addr.startswith(
            "tcp://127.0.0.1:"
        ):
            return stream_addr.replace("0.0.0.0", service_host, 1).replace(
                "127.0.0.1", service_host, 1
            )
        return stream_addr or "tcp://127.0.0.1:6201"

    # endregion

    # region 算法请求

    def request_ball_pose_detection(
        self, request: BallPoseDetectionRequest
    ) -> BallPoseDetectionResponse:
        response = self._rpc_client.call(
            CameraPipelineServiceRequest(
                operation="ball_pose_detection", ball_pose_detection=request
            )
        )
        if response.error is not None or response.ball_pose_detection is None:
            raise RuntimeError(response.error or "ball pose detection response missing")
        return response.ball_pose_detection

    # endregion
