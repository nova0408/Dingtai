from __future__ import annotations
# pyright: reportMissingImports=false

import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import lz4.block
import numpy as np
import zmq

from src.wuji.camera_protocol import WujiCameraFrame, WujiCameraName

# region 配置

_FRAME_HEADER_STRUCT = struct.Struct("<4sBBBBIIIIIIIQI")
_FRAME_HEADER_SIZE = _FRAME_HEADER_STRUCT.size

# endregion


# region 数据结构


@dataclass(frozen=True, slots=True)
class _ZmqFrameHeader:
    """ZMQ 相机二进制帧头。

    职责边界：
    - 只表达单帧消息头中的结构化字段。
    - 不负责 socket 读写、JPEG 解码或深度图解压。

    设计思想：
    - 将原始 `struct.unpack()` 结果命名化，避免解码过程散落大量魔法索引。
    - 只保留当前项目消费链路需要的字段，不在协议层推导 GUI 或业务状态。

    生命周期：
    - 每次解码单帧消息时构造，不持有外部资源。

    继承关系：
    - 不继承业务基类，作为内部协议数据结构使用。
    """

    magic: bytes
    version: int
    flags: int
    color_format: int
    depth_format: int
    color_width: int
    color_height: int
    color_data_size: int
    depth_width: int
    depth_height: int
    depth_data_size: int
    depth_original_size: int
    timestamp_us: int
    sequence: int


# endregion


# region 基础工具


def build_zmq_tcp_address(host: str, port: int) -> str:
    """构造 ZMQ TCP 地址字符串。"""

    return f"tcp://{host}:{int(port)}"


def decode_zmq_camera_frame(
    camera_name: WujiCameraName,
    raw_message: bytes,
    expect_depth: bool,
) -> WujiCameraFrame:
    """解析单条 ZMQ 相机消息为结构化帧。

    Parameters
    ----------
    camera_name:
        当前逻辑相机名。
    raw_message:
        ZMQ SUB 收到的原始二进制消息。
    expect_depth:
        `True` 表示调用方期待当前消息包含深度载荷。

    Returns
    -------
    WujiCameraFrame
        解码后的结构化相机帧。
    """

    if len(raw_message) < _FRAME_HEADER_SIZE:
        raise RuntimeError(f"ZMQ camera frame too short: {len(raw_message)}")

    header = _ZmqFrameHeader(*_FRAME_HEADER_STRUCT.unpack(raw_message[:_FRAME_HEADER_SIZE]))
    if header.magic != b"ZCAM":
        raise RuntimeError(f"invalid ZMQ camera frame magic: {header.magic!r}")

    color_start = _FRAME_HEADER_SIZE
    color_end = color_start + int(header.color_data_size)
    color_jpeg = raw_message[color_start:color_end]
    color_bgr = cv2.imdecode(np.frombuffer(color_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise RuntimeError("camera jpeg decode failed")

    depth_array: np.ndarray | None = None
    if int(header.depth_format) != 0 and int(header.depth_data_size) > 0:
        depth_start = color_end
        depth_end = depth_start + int(header.depth_data_size)
        depth_bytes = raw_message[depth_start:depth_end]
        depth_raw = lz4.block.decompress(
            depth_bytes,
            uncompressed_size=int(header.depth_original_size),
        )
        depth_array = np.frombuffer(depth_raw, dtype=np.uint16).reshape(
            (int(header.depth_height), int(header.depth_width))
        ).copy()
    elif expect_depth:
        raise RuntimeError("camera rgbd frame missing depth payload")

    return WujiCameraFrame(
        camera_name=camera_name,
        color_bgr=np.asarray(color_bgr, dtype=np.uint8).copy(),
        timestamp=int(header.timestamp_us),
        sequence_id=int(header.sequence),
        depth=depth_array,
    )


# endregion


# region ZMQ 传输


class WujiZmqCameraTransport:
    """无际 ZMQ 相机底层传输器。

    职责边界：
    - 只负责 ZMQ 控制命令收发与图像数据流订阅。
    - 不负责逻辑相机清单、GUI 展示模型、远端 SSH 探测或业务语义拼装。

    设计思想：
    - 将纯 ZMQ 协议层从上层 client 中剥离，避免业务入口继续膨胀。
    - 控制口使用短连接 REQ/REP，数据流使用按需创建的 SUB socket。

    生命周期：
    - 可由上层 client 长期持有。
    - `close()` 只负责终止自身 ZMQ Context。

    继承关系：
    - 不继承业务基类，作为 ZMQ 协议传输实现使用。

    线程/异步语义：
    - 每次流式读取都单独创建 SUB socket，适合由外部线程按需消费。
    """

    def __init__(
        self,
        host: str,
        *,
        control_port: int = 5570,
        request_timeout_ms: int = 3000,
        stream_timeout_ms: int = 5000,
    ) -> None:
        """创建 ZMQ 相机传输器。"""

        self._host = str(host)
        self._control_port = int(control_port)
        self._request_timeout_ms = int(request_timeout_ms)
        self._stream_timeout_ms = int(stream_timeout_ms)
        self._context = zmq.Context()

    def close(self) -> None:
        """关闭 ZMQ Context。"""

        self._context.term()

    def send_control_command(
        self,
        camera_id: str,
        command_name: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """向控制口发送单次命令。"""

        payload = {
            "cmd": str(command_name),
            "camera": str(camera_id),
            "params": {} if params is None else params,
        }
        response: object | None = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            socket_obj = self._context.socket(zmq.REQ)
            socket_obj.setsockopt(zmq.RCVTIMEO, self._request_timeout_ms)
            socket_obj.setsockopt(zmq.SNDTIMEO, self._request_timeout_ms)
            socket_obj.connect(build_zmq_tcp_address(self._host, self._control_port))
            try:
                socket_obj.send_json(payload)
                response = socket_obj.recv_json()
                last_error = None
                break
            except zmq.error.Again as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.2 * attempt)
            finally:
                socket_obj.close(linger=0)
        if last_error is not None:
            raise last_error
        if not isinstance(response, dict):
            raise RuntimeError(f"invalid camera control response: {response!r}")
        if not bool(response.get("success", False)):
            raise RuntimeError(str(response.get("error", "unknown camera control error")))
        return {str(key): value for key, value in response.items()}

    def stream_frames(
        self,
        camera_name: WujiCameraName,
        *,
        stream_port: int,
        expect_depth: bool,
    ) -> Iterator[WujiCameraFrame]:
        """订阅指定端口并持续读取帧。"""

        socket_obj = self._context.socket(zmq.SUB)
        socket_obj.setsockopt(zmq.CONFLATE, 1)
        socket_obj.setsockopt(zmq.RCVHWM, 1)
        socket_obj.setsockopt_string(zmq.SUBSCRIBE, "")
        socket_obj.setsockopt(zmq.RCVTIMEO, self._stream_timeout_ms)
        socket_obj.connect(build_zmq_tcp_address(self._host, int(stream_port)))
        try:
            while True:
                raw_message = socket_obj.recv()
                yield decode_zmq_camera_frame(
                    camera_name=camera_name,
                    raw_message=raw_message,
                    expect_depth=expect_depth,
                )
        finally:
            socket_obj.close(linger=0)


# endregion
