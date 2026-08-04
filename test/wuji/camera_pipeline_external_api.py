"""CameraPipeline 外部 HTTP/WebSocket API 双端冒烟测试。

版本：1.0.0

该脚本只查询相机状态、内参并订阅只读彩色流，不调用检测和 RecordReplay，不触发
机械臂或 AGV 动作。Windows 默认访问 Orin 外部地址，Linux/Orin 默认访问本机服务，
也可以通过 CLI 显式覆盖地址后直接运行。
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from types import GeneratorType

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "camera_pipeline" / "client.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraName
from camera_pipeline.protocol import CameraFramePacket
from camera_pipeline.service.http_client import CameraPipelineHttpClient

CAMERA_PIPELINE_EXTERNAL_API_TEST_VERSION = "1.0.0"
DEFAULT_CAMERA_NAME = CameraName.LEFT_ARM
DEFAULT_MAX_FRAMES = 3
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_HTTP_BASE_URL = (
    "http://192.168.1.128:6400"
    if platform.system() == "Windows"
    else "http://127.0.0.1:6400"
)
DEFAULT_WEBSOCKET_URL = (
    "ws://192.168.1.128:6401"
    if platform.system() == "Windows"
    else "ws://127.0.0.1:6401"
)


def main(
    http_base_url: str = DEFAULT_HTTP_BASE_URL,
    websocket_url: str = DEFAULT_WEBSOCKET_URL,
    camera_name: CameraName = DEFAULT_CAMERA_NAME,
    max_frames: int = DEFAULT_MAX_FRAMES,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> None:
    """执行外部 API 健康、状态、内参和 WebSocket 彩色流检查。"""

    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    client = CameraPipelineHttpClient(
        base_url=http_base_url,
        websocket_url=websocket_url,
        timeout_s=timeout_s,
    )
    health = client.get_health()
    if health.get("service_version") != client.expected_service_version:
        raise RuntimeError(f"unexpected service version: {health!r}")
    status = client.get_camera_status(camera_name, timeout_s=timeout_s)
    if not status.online or not status.color_enabled:
        raise RuntimeError(f"camera is not ready for color stream: {status}")
    intrinsics = client.get_camera_intrinsics(camera_name, timeout_s=timeout_s)
    if intrinsics.width <= 0 or intrinsics.height <= 0:
        raise RuntimeError(f"invalid camera resolution: {intrinsics}")

    color_frame_ids = _read_color_frames(client, camera_name, max_frames)
    rgbd_frame_ids = _read_rgbd_frames(client, camera_name, max_frames)
    client.close()
    print(
        "CameraPipeline external API smoke passed: "
        f"version={health['service_version']} camera={camera_name.value} "
        f"resolution={intrinsics.width}x{intrinsics.height} "
        f"color_frames={color_frame_ids} rgbd_frames={rgbd_frame_ids}"
    )


def _read_color_frames(
    client: CameraPipelineHttpClient,
    camera_name: CameraName,
    max_frames: int,
) -> list[int]:
    stream = client.subscribe_camera_color_frames(camera_name)
    frame_ids: list[int] = []
    try:
        for _ in range(max_frames):
            frame = next(stream)
            if frame.camera_name != camera_name.value:
                raise RuntimeError(f"unexpected camera in color frame: {frame.camera_name}")
            if frame.color_bgr.ndim != 3 or frame.color_bgr.shape[2] != 3:
                raise RuntimeError(f"unexpected color shape: {frame.color_bgr.shape}")
            frame_ids.append(frame.frame_id)
    finally:
        if isinstance(stream, GeneratorType):
            stream.close()
    _check_frame_ids(frame_ids, "color")
    return frame_ids


def _read_rgbd_frames(
    client: CameraPipelineHttpClient,
    camera_name: CameraName,
    max_frames: int,
) -> list[int]:
    stream = client.subscribe_camera_frames(camera_name)
    frame_ids: list[int] = []
    try:
        for _ in range(max_frames):
            frame = next(stream)
            if frame.camera_name != camera_name.value:
                raise RuntimeError(f"unexpected camera in RGBD frame: {frame.camera_name}")
            if not isinstance(frame, CameraFramePacket):
                raise RuntimeError("RGBD endpoint returned a non-RGBD frame")
            if frame.color_bgr.ndim != 3 or frame.color_bgr.shape[2] != 3:
                raise RuntimeError(f"unexpected RGBD color shape: {frame.color_bgr.shape}")
            if frame.depth_mm.ndim != 2:
                raise RuntimeError(f"unexpected depth shape: {frame.depth_mm.shape}")
            frame_ids.append(frame.frame_id)
    finally:
        if isinstance(stream, GeneratorType):
            stream.close()
    _check_frame_ids(frame_ids, "RGBD")
    return frame_ids


def _check_frame_ids(frame_ids: list[int], stream_name: str) -> None:
    if len(frame_ids) != len(set(frame_ids)) or frame_ids != sorted(frame_ids):
        raise RuntimeError(f"{stream_name} frame ids are not strictly increasing: {frame_ids}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-base-url", default=DEFAULT_HTTP_BASE_URL)
    parser.add_argument("--websocket-url", default=DEFAULT_WEBSOCKET_URL)
    parser.add_argument("--camera-name", choices=[item.value for item in CameraName], default=DEFAULT_CAMERA_NAME.value)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        http_base_url=args.http_base_url,
        websocket_url=args.websocket_url,
        camera_name=CameraName(args.camera_name),
        max_frames=args.max_frames,
        timeout_s=args.timeout_s,
    )
