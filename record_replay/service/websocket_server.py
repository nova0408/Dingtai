"""RecordReplay 状态 WebSocket 订阅服务。

后端只提供内部 ``ws`` 端口；正式客户端通过 API Gateway 的 TLS 终止层使用 ``wss``。
每个订阅者只保留最新状态，避免慢客户端拖住回放线程或形成无界队列。
"""

from __future__ import annotations

import base64
import hashlib
import json
import select
import socket
import socketserver
import struct
import threading
from queue import Empty
from urllib.parse import urlsplit

from loguru import logger

from ..context import ReplayContext
from ..contracts import ReplayExecutionCompletedEvent, ReplayStatusSnapshot

_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_MAX_HANDSHAKE_BYTES = 16 * 1024


class RecordReplayWebSocketServer:
    """按状态快照变化向客户端推送 JSON 文本消息。"""

    def __init__(self, host: str, port: int, context: ReplayContext) -> None:
        self._context = context
        self._stop_event = threading.Event()
        self._server = _ThreadingTcpServer((host, port), self._build_handler())
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """后台启动状态订阅监听。"""

        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="record-replay-websocket",
            daemon=True,
        )
        self._thread.start()
        logger.info("record replay WebSocket API started address={}", self._server.server_address)

    def close(self) -> None:
        """停止监听并释放 WebSocket 端口。"""

        self._stop_event.set()
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2.0)
            self._thread = None
        self._server.server_close()

    def _build_handler(self) -> type[socketserver.BaseRequestHandler]:
        context = self._context
        stop_event = self._stop_event

        class Handler(socketserver.BaseRequestHandler):
            """处理一个 RecordReplay 状态订阅连接。"""

            def handle(self) -> None:
                subscriber = None
                client_address = self.client_address
                try:
                    logger.info("RecordReplay WebSocket 握手开始 client={}", client_address)
                    self.request.settimeout(5.0)
                    path, headers = _read_handshake(self.request)
                    if urlsplit(path).path != "/api/v1/ws" or "sec-websocket-key" not in headers:
                        logger.warning(
                            "RecordReplay WebSocket 握手拒绝 client={} path={}",
                            client_address,
                            path,
                        )
                        _send_http_error(self.request, 400, "invalid record replay websocket request")
                        return
                    _send_handshake(self.request, headers["sec-websocket-key"])
                    self.request.settimeout(1.0)
                    subscriber = context.subscribe_status()
                    logger.info(
                        "RecordReplay WebSocket 订阅已建立 client={} path={}",
                        client_address,
                        path,
                    )
                    while not stop_event.is_set():
                        if not _handle_control_frame(self.request):
                            logger.info(
                                "RecordReplay WebSocket 收到关闭或无效控制帧 client={}",
                                client_address,
                            )
                            return
                        try:
                            message = subscriber.get(timeout=0.2)
                        except Empty:
                            continue
                        if isinstance(message, ReplayExecutionCompletedEvent):
                            payload = _completion_json(message.snapshot)
                        else:
                            payload = _snapshot_json(message)
                        _send_text_frame(self.request, payload)
                except (ConnectionError, OSError, TimeoutError) as error:
                    logger.info(
                        "RecordReplay WebSocket 连接结束 client={} type={} detail={}",
                        client_address,
                        type(error).__name__,
                        error,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "RecordReplay WebSocket 未处理异常 client={} type={} detail={}",
                        client_address,
                        type(exc).__name__,
                        exc,
                    )
                finally:
                    if subscriber is not None:
                        context.unsubscribe_status(subscriber)
                    logger.info("RecordReplay WebSocket 连接清理完成 client={}", client_address)

        return Handler


class _ThreadingTcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _snapshot_json(snapshot: ReplayStatusSnapshot) -> str:
    """将不可变状态快照编码为稳定的 JSON 文本。"""

    return _status_json(snapshot, "record_replay.status", completed=False)


def _completion_json(snapshot: ReplayStatusSnapshot) -> str:
    """编码一次性执行完成事件；事件中的计数已经递增。"""

    return _status_json(snapshot, "record_replay.completed", completed=True)


def _status_json(
    snapshot: ReplayStatusSnapshot,
    event_name: str,
    *,
    completed: bool,
) -> str:
    """编码状态或完成事件的共同字段。"""

    payload = {
        "event": event_name,
        "completed": completed,
        "state": snapshot.state.value,
        "accepted": True,
        "error_code": snapshot.error_code,
        "total_execution_count": snapshot.total_execution_count,
        "old_tray_current_index": snapshot.old_tray_current_index,
        "old_tray_put_index": snapshot.old_tray_put_index,
        "new_tray_current_index": snapshot.new_tray_current_index,
        "new_tray_put_index": snapshot.new_tray_put_index,
        "agv_navigation_enabled": snapshot.agv_navigation_enabled,
        "agv_target": snapshot.agv_target,
        "action_sequence_sha256": snapshot.action_sequence_sha256,
        "left_csv_state": snapshot.left_csv_state,
        "plan_index": snapshot.plan_index,
        "error_text": snapshot.error_text,
        "left_csv_files": [
            {"name": item.name, "row_count": item.row_count}
            for item in snapshot.left_csv_files
        ],
        "right_csv_files": [
            {"name": item.name, "row_count": item.row_count}
            for item in snapshot.right_csv_files
        ],
        "execution_tasks": [
            {
                "sequence": item.sequence,
                "left_csv": item.left_csv,
                "right_csv": item.right_csv,
                "synchronized": item.synchronized,
            }
            for item in snapshot.execution_tasks
        ],
        "current_task_sequence": (
            0 if snapshot.current_task_index is None else snapshot.current_task_index + 1
        ),
        "current_task_active": snapshot.current_task_active,
        "current_left_csv": snapshot.current_left_csv,
        "current_left_action_name": snapshot.current_left_action_name,
        "current_left_action_index": snapshot.current_left_action_index,
        "current_right_csv": snapshot.current_right_csv,
        "current_right_action_name": snapshot.current_right_action_name,
        "current_right_action_index": snapshot.current_right_action_index,
        "current_left_row": snapshot.current_left_row,
        "current_right_row": snapshot.current_right_row,
        "current_left_total_rows": snapshot.current_left_total_rows,
        "current_right_total_rows": snapshot.current_right_total_rows,
        "offset_statuses": [
            {
                "source": item.source,
                "available": item.available,
                "applied": item.applied,
            }
            for item in snapshot.offset_statuses
        ],
        "parameters": None,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


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
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    return request_line[1], headers


def _send_handshake(request: socket.socket, key: str) -> None:
    accept = base64.b64encode(hashlib.sha1((key + _WEBSOCKET_GUID).encode("ascii")).digest()).decode("ascii")
    request.sendall(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii")
    )


def _send_http_error(request: socket.socket, status: int, message: str) -> None:
    body = message.encode("utf-8")
    request.sendall(
        f"HTTP/1.1 {status} Bad Request\r\nContent-Length: {len(body)}\r\n\r\n".encode("ascii")
        + body
    )


def _send_text_frame(request: socket.socket, payload: str) -> None:
    encoded = payload.encode("utf-8")
    length = len(encoded)
    if length < 126:
        header = bytes((0x81, length))
    elif length <= 0xFFFF:
        header = bytes((0x81, 126)) + struct.pack("!H", length)
    else:
        header = bytes((0x81, 127)) + struct.pack("!Q", length)
    request.sendall(header + encoded)


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
    payload = bytes(value ^ mask[index % 4] for index, value in enumerate(_recv_exact(request, length)))
    if opcode == 0x8:
        request.sendall(b"\x88\x00")
        return False
    if opcode == 0x9:
        _send_control_frame(request, 0xA, payload)
    return True


def _send_control_frame(request: socket.socket, opcode: int, payload: bytes) -> None:
    if len(payload) < 126:
        request.sendall(bytes((0x80 | opcode, len(payload))) + payload)


def _recv_exact(request: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = request.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("websocket connection closed")
        chunks.extend(chunk)
    return bytes(chunks)
