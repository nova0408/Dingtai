from __future__ import annotations

import threading

import zmq
from loguru import logger

from ..pipeline_context import PipelineContext
from ..protocol import (
    CameraColorFramePacket,
    CameraDepthFramePacket,
    CameraFramePacket,
    ColorFrameProtocol,
    RgbdFrameProtocol,
)
from .protocol import build_camera_stream_topic
from .wire_codec import encode_wire


class CameraFramePublisher:
    """通过一个后台线程按相机 topic 发布三类最新帧。"""

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
        self._frame_subscriptions: dict[bytes, int] = {}
        self._color_subscriptions: dict[bytes, int] = {}
        self._depth_subscriptions: dict[bytes, int] = {}

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
            logger.exception("camera frame publisher socket initialization failed")
            self.close()
            raise
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._publish_loop, name="camera-frame-publisher", daemon=True
        )
        self._thread.start()
        logger.info(
            "camera frame publisher started frame_addr={} color_addr={} depth_addr={}",
            self._frame_bind_addr,
            self._color_bind_addr,
            self._depth_bind_addr,
        )

    def close(self) -> None:
        """停止发布线程并释放全部 PUB socket。"""

        was_running = self._thread is not None
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
        self._frame_subscriptions.clear()
        self._color_subscriptions.clear()
        self._depth_subscriptions.clear()
        if was_running:
            logger.info("camera frame publisher stopped")

    def _create_socket(self, bind_addr: str) -> zmq.Socket:
        socket = zmq.Context.instance().socket(zmq.XPUB)
        socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.bind(bind_addr)
        return socket

    def _publish_loop(self) -> None:
        """分别转发 RGBD、彩色和深度缓存，避免三类订阅互相阻塞。"""

        last_rgbd_frame_ids: dict[str, int] = {}
        last_color_frame_ids: dict[str, int] = {}
        while not self._stop_event.is_set():
            self._drain_subscription_events()
            if not self._has_any_subscriber():
                self._stop_event.wait(0.02)
                continue
            published = False
            for camera_name in self._pipeline_context.get_connected_camera_names():
                topic = build_camera_stream_topic(camera_name)
                if not self._has_topic_subscriber(topic):
                    continue
                if self._has_rgbd_subscriber(topic):
                    frame = self._pipeline_context.get_latest_frame(camera_name)
                    if (
                        frame is not None
                        and frame.frame_id != last_rgbd_frame_ids.get(camera_name)
                    ):
                        last_rgbd_frame_ids[camera_name] = frame.frame_id
                        self._publish_rgbd_frame(frame, topic)
                        published = True
                if self._color_subscriptions.get(topic, 0) > 0:
                    color_frame = self._pipeline_context.get_latest_color_frame(
                        camera_name
                    )
                    if (
                        color_frame is not None
                        and color_frame.frame_id
                        != last_color_frame_ids.get(camera_name)
                    ):
                        last_color_frame_ids[camera_name] = color_frame.frame_id
                        self._publish_color_frame(color_frame, topic)
                        published = True
            if not published:
                self._stop_event.wait(0.02)
                continue
            self._stop_event.wait(0.01)

    def _publish_rgbd_frame(self, frame: RgbdFrameProtocol, topic: bytes) -> None:
        """按订阅类型发布完整 RGBD 或独立深度数据。"""

        if (
            self._frame_socket is None
            or self._depth_socket is None
        ):
            raise RuntimeError("camera frame publisher sockets are not ready")
        packets: list[
            tuple[
                zmq.Socket,
                CameraFramePacket | CameraColorFramePacket | CameraDepthFramePacket,
            ]
        ] = []
        if self._frame_subscriptions.get(topic, 0) > 0:
            packets.append((self._frame_socket, self._build_frame_packet(frame)))
        if self._depth_subscriptions.get(topic, 0) > 0:
            packets.append((self._depth_socket, self._build_depth_packet(frame)))
        for socket, packet in packets:
            try:
                socket.send(topic + encode_wire(packet), flags=zmq.NOBLOCK)
            except zmq.error.Again:
                continue

    def _publish_color_frame(
        self,
        frame: ColorFrameProtocol,
        topic: bytes,
    ) -> None:
        """发布不依赖深度缓存的彩色数据。"""

        if self._color_socket is None:
            raise RuntimeError("camera color frame publisher socket is not ready")
        try:
            self._color_socket.send(
                topic + encode_wire(self._build_color_packet(frame)),
                flags=zmq.NOBLOCK,
            )
        except zmq.error.Again:
            return

    @staticmethod
    def _build_frame_packet(frame: RgbdFrameProtocol) -> CameraFramePacket:
        """在传输边界把算法帧协议转换为完整 RGBD packet。"""

        return CameraFramePacket(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
            color_bgr=frame.color_bgr,
            depth_mm=frame.depth_mm,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
            distortion=frame.distortion,
        )

    def _drain_subscription_events(self) -> None:
        """消费 XPUB 订阅事件并更新每个相机 topic 的订阅者数量。"""

        if (
            self._frame_socket is None
            or self._color_socket is None
            or self._depth_socket is None
        ):
            return
        socket_states = (
            (self._frame_socket, self._frame_subscriptions),
            (self._color_socket, self._color_subscriptions),
            (self._depth_socket, self._depth_subscriptions),
        )
        for socket, subscriptions in socket_states:
            while True:
                try:
                    event = socket.recv(flags=zmq.NOBLOCK)
                except zmq.error.Again:
                    break
                if len(event) <= 1:
                    continue
                topic = event[1:]
                if event[0] == 1:
                    subscriptions[topic] = subscriptions.get(topic, 0) + 1
                    continue
                remaining = subscriptions.get(topic, 0) - 1
                if remaining > 0:
                    subscriptions[topic] = remaining
                else:
                    subscriptions.pop(topic, None)

    def _has_any_subscriber(self) -> bool:
        """返回任一帧类型是否存在活跃订阅者。"""

        return bool(
            self._frame_subscriptions
            or self._color_subscriptions
            or self._depth_subscriptions
        )

    def _has_topic_subscriber(self, topic: bytes) -> bool:
        """返回指定相机 topic 是否存在任一帧类型订阅者。"""

        return (
            self._frame_subscriptions.get(topic, 0) > 0
            or self._color_subscriptions.get(topic, 0) > 0
            or self._depth_subscriptions.get(topic, 0) > 0
        )

    def _has_rgbd_subscriber(self, topic: bytes) -> bool:
        """返回该相机是否需要读取 RGBD 缓存。"""

        return (
            self._frame_subscriptions.get(topic, 0) > 0
            or self._depth_subscriptions.get(topic, 0) > 0
        )

    @staticmethod
    def _build_color_packet(frame: ColorFrameProtocol) -> CameraColorFramePacket:
        return CameraColorFramePacket(
            frame_id=frame.frame_id,
            camera_name=frame.camera_name,
            timestamp_ms=frame.timestamp_ms,
            color_bgr=frame.color_bgr,
            fx=frame.fx,
            fy=frame.fy,
            cx=frame.cx,
            cy=frame.cy,
            distortion=frame.distortion,
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
            distortion=frame.distortion,
        )
