"""CameraPipeline 对外 WebSocket 最新帧订阅服务。

该实现只依赖 Python 标准库：服务端发送二进制帧，客户端通过 URL 选择相机和
帧类型。每个连接只保留最新帧，慢客户端不会在服务端形成无界队列。
"""

from __future__ import annotations

import base64
import hashlib
import select
import socket
import socketserver
import struct
import threading
from urllib.parse import unquote, urlsplit

from loguru import logger

from ..protocol import (
    CameraColorFramePacket,
    CameraDepthFramePacket,
    CameraFramePacket,
    CameraName,
    ColorFrameProtocol,
    RgbdFrameProtocol,
)
from ..pipeline_context import PipelineContext
from .external_codec import encode_stream_packet

_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_HANDSHAKE_BYTES = 16 * 1024


class CameraPipelineWebSocketServer:
    """按相机和帧类型提供 WebSocket 二进制最新帧订阅。"""

    def __init__(
        self,
        host: str,
        port: int,
        pipeline_context: PipelineContext,
        stop_event: threading.Event,
    ) -> None:
        self._pipeline_context = pipeline_context
        self._stop_event = stop_event
        self._server = _ThreadingTcpServer(
            (host, port),
            self._build_handler(),
        )
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """后台启动 WebSocket 服务。"""

        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="camera-pipeline-websocket",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "camera pipeline WebSocket API started address={}",
            self._server.server_address,
        )

    def close(self) -> None:
        """停止 WebSocket listener；连接线程由 daemon socket handler 退出。"""

        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._server.server_close()

    def _build_handler(self) -> type[socketserver.BaseRequestHandler]:
        pipeline_context = self._pipeline_context
        stop_event = self._stop_event

        class Handler(socketserver.BaseRequestHandler):
            """处理一个相机流 WebSocket 连接。"""

            def handle(self) -> None:
                try:
                    self.request.settimeout(5.0)
                    path, headers = _read_handshake(self.request)
                    stream = _parse_stream_path(path)
                    if stream is None or "sec-websocket-key" not in headers:
                        _send_http_error(self.request, 400, "invalid websocket request")
                        return
                    _send_handshake(self.request, headers["sec-websocket-key"])
                    self.request.settimeout(1.0)
                    self._stream_frames(stream[0], stream[1])
                except (ConnectionError, OSError, TimeoutError):
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("camera pipeline WebSocket connection failed: {}", exc)

            def _stream_frames(self, camera_name: CameraName, frame_kind: str) -> None:
                last_frame_id: int | None = None
                while not stop_event.is_set():
                    if not _handle_control_frame(self.request):
                        return
                    packet = _latest_packet(pipeline_context, camera_name, frame_kind)
                    if packet is None or packet.frame_id == last_frame_id:
                        stop_event.wait(0.02)
                        continue
                    last_frame_id = packet.frame_id
                    payload = _packet_payload(packet, frame_kind)
                    _send_websocket_frame(self.request, payload)

        return Handler


class _ThreadingTcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _parse_stream_path(path: str) -> tuple[CameraName, str] | None:
    parts = tuple(part for part in urlsplit(path).path.split("/") if part)
    if len(parts) != 6 or parts[:4] != ("api", "v1", "ws", "cameras"):
        return None
    try:
        camera_name = CameraName(unquote(parts[4]))
    except ValueError:
        return None
    if parts[5] not in {"color", "depth", "rgbd"}:
        return None
    return camera_name, parts[5]


def _latest_packet(
    pipeline_context: PipelineContext,
    camera_name: CameraName,
    frame_kind: str,
) -> CameraFramePacket | CameraColorFramePacket | CameraDepthFramePacket | None:
    if frame_kind == "color":
        frame = pipeline_context.get_latest_color_frame(camera_name.value)
        return None if frame is None else _build_color_packet(frame)
    frame = pipeline_context.get_latest_frame(camera_name.value)
    if frame is None:
        return None
    if frame_kind == "depth":
        return _build_depth_packet(frame)
    return _build_rgbd_packet(frame)


def _build_rgbd_packet(frame: RgbdFrameProtocol) -> CameraFramePacket:
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


def _packet_payload(
    packet: CameraFramePacket | CameraColorFramePacket | CameraDepthFramePacket,
    frame_kind: str,
) -> bytes:
    fields = {
        "frame_id": packet.frame_id,
        "camera_name": packet.camera_name,
        "timestamp_ms": packet.timestamp_ms,
        "fx": packet.fx,
        "fy": packet.fy,
        "cx": packet.cx,
        "cy": packet.cy,
        "distortion": packet.distortion,
    }
    if frame_kind in {"color", "rgbd"}:
        if isinstance(packet, CameraDepthFramePacket):
            raise TypeError("depth packet cannot provide color_bgr")
        fields["color_bgr"] = packet.color_bgr
    if frame_kind in {"depth", "rgbd"}:
        if isinstance(packet, CameraColorFramePacket):
            raise TypeError("color packet cannot provide depth_mm")
        fields["depth_mm"] = packet.depth_mm
    packet_type = "rgbd_frame" if frame_kind == "rgbd" else f"{frame_kind}_frame"
    return encode_stream_packet(packet_type, fields)


def _read_handshake(request: socket.socket) -> tuple[str, dict[str, str]]:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = request.recv(4096)
        if not chunk:
            raise ConnectionError("websocket handshake closed")
        data.extend(chunk)
        if len(data) > _MAX_HANDSHAKE_BYTES:
            raise ValueError("websocket handshake is too large")
    header_text = bytes(data).split(b"\r\n\r\n", 1)[0].decode("iso-8859-1")
    lines = header_text.split("\r\n")
    request_line = lines[0].split(" ")
    if len(request_line) != 3 or request_line[0] != "GET":
        raise ValueError("websocket handshake must use GET")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    return request_line[1], headers


def _send_handshake(request: socket.socket, key: str) -> None:
    accept = base64.b64encode(
        hashlib.sha1((key + _WEBSOCKET_GUID).encode("ascii")).digest()
    ).decode("ascii")
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode("ascii")
    request.sendall(response)


def _send_http_error(request: socket.socket, status: int, message: str) -> None:
    body = message.encode("utf-8")
    request.sendall(
        f"HTTP/1.1 {status} Bad Request\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    )


def _send_websocket_frame(request: socket.socket, payload: bytes) -> None:
    length = len(payload)
    if length < 126:
        header = bytes((0x82, length))
    elif length <= 0xFFFF:
        header = bytes((0x82, 126)) + struct.pack("!H", length)
    else:
        header = bytes((0x82, 127)) + struct.pack("!Q", length)
    request.sendall(header + payload)


def _handle_control_frame(request: socket.socket) -> bool:
    readable, _, _ = select.select([request], [], [], 0.0)
    if not readable:
        return True
    first_two = _recv_exact(request, 2)
    fin = first_two[0] & 0x80
    opcode = first_two[0] & 0x0F
    masked = first_two[1] & 0x80
    length = first_two[1] & 0x7F
    if not fin or not masked:
        return False
    if length == 126:
        length = struct.unpack("!H", _recv_exact(request, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _recv_exact(request, 8))[0]
    if length > 64 * 1024:
        return False
    mask = _recv_exact(request, 4)
    payload = bytes(
        value ^ mask[index % 4]
        for index, value in enumerate(_recv_exact(request, length))
    )
    if opcode == 0x8:
        request.sendall(b"\x88\x00")
        return False
    if opcode == 0x9:
        _send_control_frame(request, 0xA, payload)
    return True


def _send_control_frame(request: socket.socket, opcode: int, payload: bytes) -> None:
    if len(payload) >= 126:
        return
    request.sendall(bytes((0x80 | opcode, len(payload))) + payload)


def _recv_exact(request: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = request.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("websocket connection closed")
        chunks.extend(chunk)
    return bytes(chunks)
