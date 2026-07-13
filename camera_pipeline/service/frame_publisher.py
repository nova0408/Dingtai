from __future__ import annotations

import threading

import zmq

from ..pipeline_context import PipelineContext
from ..protocol import CameraColorFramePacket, CameraDepthFramePacket, RgbdFrameProtocol
from .wire_codec import encode_wire


class CameraFramePublisher:
    """通过一个后台线程发布 RGBD、彩色和深度三类最新帧。"""

    def __init__(
        self,
        pipeline_context: PipelineContext,
        *,
        frame_bind_addr: str = "tcp://0.0.0.0:6201",
        color_bind_addr: str = "tcp://0.0.0.0:6202",
        depth_bind_addr: str = "tcp://0.0.0.0:6203",
    ) -> None:
        self._pipeline_context = pipeline_context
        self._frame_bind_addr = frame_bind_addr
        self._color_bind_addr = color_bind_addr
        self._depth_bind_addr = depth_bind_addr
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_socket: zmq.Socket | None = None
        self._color_socket: zmq.Socket | None = None
        self._depth_socket: zmq.Socket | None = None

    @property
    def frame_bind_addr(self) -> str:
        return self._frame_bind_addr

    @property
    def color_bind_addr(self) -> str:
        return self._color_bind_addr

    @property
    def depth_bind_addr(self) -> str:
        return self._depth_bind_addr

    def start(self) -> None:
        """按需绑定发布端口并启动单一发布线程。"""

        if self._thread is not None:
            return
        try:
            self._frame_socket = self._create_socket(self._frame_bind_addr)
            self._color_socket = self._create_socket(self._color_bind_addr)
            self._depth_socket = self._create_socket(self._depth_bind_addr)
        except Exception:
            self.close()
            raise
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._publish_loop, name="camera-frame-publisher", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """停止发布线程并释放全部 PUB socket。"""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        for socket in (self._frame_socket, self._color_socket, self._depth_socket):
            if socket is not None:
                socket.close(linger=0)
        self._frame_socket = None
        self._color_socket = None
        self._depth_socket = None

    def _create_socket(self, bind_addr: str) -> zmq.Socket:
        socket = zmq.Context.instance().socket(zmq.PUB)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.bind(bind_addr)
        return socket

    def _publish_loop(self) -> None:
        last_frame_id = -1
        while not self._stop_event.is_set():
            frame = self._pipeline_context.get_latest_frame()
            if frame is None or frame.frame_id == last_frame_id:
                self._stop_event.wait(0.02)
                continue
            last_frame_id = frame.frame_id
            self._publish_frame(frame)
            self._stop_event.wait(0.01)

    def _publish_frame(self, frame: RgbdFrameProtocol) -> None:
        if (
            self._frame_socket is None
            or self._color_socket is None
            or self._depth_socket is None
        ):
            raise RuntimeError("camera frame publisher sockets are not ready")
        packets = (
            (self._frame_socket, frame),
            (self._color_socket, self._build_color_packet(frame)),
            (self._depth_socket, self._build_depth_packet(frame)),
        )
        for socket, packet in packets:
            try:
                socket.send(encode_wire(packet), flags=zmq.NOBLOCK)
            except zmq.error.Again:
                continue

    @staticmethod
    def _build_color_packet(frame: RgbdFrameProtocol) -> CameraColorFramePacket:
        return CameraColorFramePacket(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
            color_bgr=frame.color_bgr,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
        )

    @staticmethod
    def _build_depth_packet(frame: RgbdFrameProtocol) -> CameraDepthFramePacket:
        return CameraDepthFramePacket(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
            depth_mm=frame.depth_mm,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
        )
