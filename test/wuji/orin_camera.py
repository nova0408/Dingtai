from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path
import sys
from typing import TypeVar, cast

from loguru import logger

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "camera_pipeline" / "client.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient
from camera_pipeline.protocol import (
    CameraColorFramePacket,
    CameraDepthFramePacket,
    CameraFramePacket,
)
from camera_pipeline.service.protocol import (
    CameraIntrinsicsResponse,
    StableFrameResponse,
)

# Windows 从开发机访问 Orin，Linux 在 Orin 本机访问统一服务。
DEFAULT_ORIN_SERVICE_ADDR = (
    "tcp://192.168.1.121:6200"
    if sys.platform == "win32"
    else "tcp://127.0.0.1:6200"
)
# RPC 与首帧等待超时，单位 s。
DEFAULT_TIMEOUT_S = 10.0
# client socket 收发超时，单位 ms。
DEFAULT_CLIENT_TIMEOUT_MS = 30_000

FrameT = TypeVar("FrameT")


# region 主流程


def main() -> int:
    """验证 CameraPipeline 多相机查询、订阅与右臂保留 API。"""

    logger.info(
        "硬件测试开始：服务地址 {} 超时 {} s",
        DEFAULT_ORIN_SERVICE_ADDR,
        DEFAULT_TIMEOUT_S,
    )
    client = CameraPipelineClient(
        service_addr=DEFAULT_ORIN_SERVICE_ADDR,
        timeout_ms=DEFAULT_CLIENT_TIMEOUT_MS,
    )
    try:
        _check_queries(client)
        _check_named_streams(client)
        _check_parameterized_streams(client)
        _check_disconnected_right_arm_streams(client)
        _check_stable_frames(client)
    finally:
        client.close()

    logger.success("CameraPipeline 多相机 API 与帧数据验证通过")
    logger.warning("本测试不执行 tray/opening/ball 算法推理")
    return 0


# endregion


# region 相机查询


def _check_queries(client: CameraPipelineClient) -> None:
    """验证默认相机摘要、状态与分相机内参。"""

    summary = client.get_camera_summary(timeout_s=DEFAULT_TIMEOUT_S)
    if summary.camera_name != "left_hand_camera":
        raise RuntimeError(f"unexpected summary camera: {summary.camera_name}")
    if summary.frame_id <= 0 or summary.color_shape[2] != 3:
        raise RuntimeError(f"invalid camera summary: {summary!r}")
    logger.info(
        "默认相机摘要：相机 {} 帧号 {} 彩色尺寸 {} 深度尺寸 {}",
        summary.camera_name,
        summary.frame_id,
        summary.color_shape,
        summary.depth_shape,
    )

    status = client.get_camera_status(timeout_s=DEFAULT_TIMEOUT_S)
    if status.camera_name != "left_hand_camera" or not status.online:
        raise RuntimeError(f"invalid camera status: {status!r}")
    logger.info(
        "默认相机状态：相机 {} 在线 {} 彩色 {} 深度 {}",
        status.camera_name,
        status.online,
        status.color_enabled,
        status.depth_enabled,
    )

    _validate_intrinsics(
        client.get_head_camera_intrinsics(timeout_s=DEFAULT_TIMEOUT_S),
        "head_camera",
    )
    _validate_intrinsics(
        client.get_chest_camera_intrinsics(timeout_s=DEFAULT_TIMEOUT_S),
        "chest_camera",
    )
    _validate_intrinsics(
        client.get_left_arm_camera_intrinsics(timeout_s=DEFAULT_TIMEOUT_S),
        "left_hand_camera",
    )
    _validate_intrinsics(
        client.get_camera_intrinsics(
            "chest_camera",
            timeout_s=DEFAULT_TIMEOUT_S,
        ),
        "chest_camera",
    )
    try:
        client.get_right_arm_camera_intrinsics(timeout_s=DEFAULT_TIMEOUT_S)
    except RuntimeError as exc:
        _validate_disconnected_error(exc, "右臂内参")
    else:
        raise RuntimeError("未连接右臂相机不应返回内参")


def _validate_intrinsics(
    intrinsics: CameraIntrinsicsResponse,
    expected_camera_name: str,
) -> None:
    """校验指定相机的内参归属与数值有效性。"""

    if intrinsics.camera_name != expected_camera_name:
        raise RuntimeError(
            f"intrinsics camera mismatch: {intrinsics.camera_name} != {expected_camera_name}"
        )
    if min(intrinsics.fx, intrinsics.fy, intrinsics.width, intrinsics.height) <= 0:
        raise RuntimeError(f"invalid camera intrinsics: {intrinsics!r}")
    logger.info(
        "相机内参：相机 {} 分辨率 {}x{} 焦距 ({:.3f}, {:.3f}) 像素",
        intrinsics.camera_name,
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
    )


# endregion


# region 帧订阅


def _check_named_streams(client: CameraPipelineClient) -> None:
    """验证头部、胸腔和左臂的明确命名订阅 API。"""

    _validate_rgbd_frame(
        _read_one(client.subscribe_head_camera_frames()),
        "head_camera",
    )
    _validate_rgbd_frame(
        _read_one(client.subscribe_chest_camera_frames()),
        "chest_camera",
    )
    _validate_rgbd_frame(
        _read_one(client.subscribe_left_arm_camera_frames()),
        "left_hand_camera",
    )

    _validate_color_frame(
        _read_one(client.subscribe_head_camera_color_frames()),
        "head_camera",
    )
    _validate_color_frame(
        _read_one(client.subscribe_chest_camera_color_frames()),
        "chest_camera",
    )
    _validate_color_frame(
        _read_one(client.subscribe_left_arm_camera_color_frames()),
        "left_hand_camera",
    )

    _validate_depth_frame(
        _read_one(client.subscribe_head_camera_depth_frames()),
        "head_camera",
    )
    _validate_depth_frame(
        _read_one(client.subscribe_chest_camera_depth_frames()),
        "chest_camera",
    )
    _validate_depth_frame(
        _read_one(client.subscribe_left_arm_camera_depth_frames()),
        "left_hand_camera",
    )


def _check_parameterized_streams(client: CameraPipelineClient) -> None:
    """验证保留的参数化订阅 API。"""

    _validate_rgbd_frame(
        _read_one(client.subscribe_camera_frames("left_hand_camera")),
        "left_hand_camera",
    )
    _validate_color_frame(
        _read_one(client.subscribe_camera_color_frames("left_hand_camera")),
        "left_hand_camera",
    )
    _validate_depth_frame(
        _read_one(client.subscribe_camera_depth_frames("left_hand_camera")),
        "left_hand_camera",
    )


def _read_one(stream: Iterator[FrameT]) -> FrameT:
    """读取单帧并立即关闭订阅 socket。"""

    generator = cast(Generator[FrameT, None, None], stream)
    try:
        return next(generator)
    finally:
        generator.close()


def _validate_rgbd_frame(
    frame: CameraFramePacket,
    expected_camera_name: str,
) -> None:
    """校验 RGBD 帧的相机归属、形状和内参。"""

    if frame.camera_name != expected_camera_name:
        raise RuntimeError(
            f"RGBD camera mismatch: {frame.camera_name} != {expected_camera_name}"
        )
    if frame.color_bgr.ndim != 3 or frame.color_bgr.shape[2] != 3:
        raise RuntimeError(f"invalid color frame shape: {frame.color_bgr.shape}")
    if frame.depth_mm.ndim != 2 or frame.depth_mm.shape != frame.color_bgr.shape[:2]:
        raise RuntimeError(
            f"RGBD shape mismatch: color={frame.color_bgr.shape}, depth={frame.depth_mm.shape}"
        )
    if frame.frame_id <= 0 or min(frame.fx, frame.fy) <= 0.0:
        raise RuntimeError(f"invalid RGBD frame metadata: {frame!r}")
    logger.info(
        "RGBD 帧：相机 {} 帧号 {} 尺寸 {}x{} 有效深度点 {} 点",
        frame.camera_name,
        frame.frame_id,
        frame.color_bgr.shape[1],
        frame.color_bgr.shape[0],
        int((frame.depth_mm > 0).sum()),
    )


def _validate_color_frame(
    frame: CameraColorFramePacket,
    expected_camera_name: str,
) -> None:
    """校验彩色帧的相机归属和图像形状。"""

    if frame.camera_name != expected_camera_name:
        raise RuntimeError(
            f"color camera mismatch: {frame.camera_name} != {expected_camera_name}"
        )
    if frame.color_bgr.ndim != 3 or frame.color_bgr.shape[2] != 3:
        raise RuntimeError(f"invalid color frame shape: {frame.color_bgr.shape}")
    logger.info(
        "彩色帧：相机 {} 帧号 {} 尺寸 {}x{} 像素",
        frame.camera_name,
        frame.frame_id,
        frame.color_bgr.shape[1],
        frame.color_bgr.shape[0],
    )


def _validate_depth_frame(
    frame: CameraDepthFramePacket,
    expected_camera_name: str,
) -> None:
    """校验深度帧的相机归属和图像形状。"""

    if frame.camera_name != expected_camera_name:
        raise RuntimeError(
            f"depth camera mismatch: {frame.camera_name} != {expected_camera_name}"
        )
    if frame.depth_mm.ndim != 2:
        raise RuntimeError(f"invalid depth frame shape: {frame.depth_mm.shape}")
    logger.info(
        "深度帧：相机 {} 帧号 {} 尺寸 {}x{} 有效深度点 {} 点",
        frame.camera_name,
        frame.frame_id,
        frame.depth_mm.shape[1],
        frame.depth_mm.shape[0],
        int((frame.depth_mm > 0).sum()),
    )


# endregion


# region 稳定帧


def _check_stable_frames(client: CameraPipelineClient) -> None:
    """在内参与帧数据验证完成后，最后验证分相机稳定帧 API。"""

    _validate_stable_frame(
        client.get_head_camera_stable_frame(timeout_s=DEFAULT_TIMEOUT_S),
        "head_camera",
    )
    _validate_stable_frame(
        client.get_chest_camera_stable_frame(timeout_s=DEFAULT_TIMEOUT_S),
        "chest_camera",
    )
    _validate_stable_frame(
        client.get_left_arm_camera_stable_frame(timeout_s=DEFAULT_TIMEOUT_S),
        "left_hand_camera",
    )
    _validate_stable_frame(
        client.get_stable_frame(
            "chest_camera",
            timeout_s=DEFAULT_TIMEOUT_S,
        ),
        "chest_camera",
    )

    try:
        client.get_right_arm_camera_stable_frame(timeout_s=DEFAULT_TIMEOUT_S)
    except RuntimeError as exc:
        _validate_disconnected_error(exc, "右臂稳定帧")
    else:
        raise RuntimeError("未连接右臂相机不应返回稳定帧")


def _validate_stable_frame(
    stable_frame: StableFrameResponse,
    expected_camera_name: str,
) -> None:
    """校验指定相机的稳定帧归属与帧号。"""

    if stable_frame.camera_name != expected_camera_name:
        raise RuntimeError(
            f"stable frame camera mismatch: {stable_frame.camera_name} != {expected_camera_name}"
        )
    if stable_frame.frame_id <= 0:
        raise RuntimeError(f"invalid stable frame: {stable_frame!r}")
    logger.info(
        "稳定帧：相机 {} 帧号 {} 时间戳 {:.3f} ms",
        stable_frame.camera_name,
        stable_frame.frame_id,
        stable_frame.timestamp_ms,
    )


# endregion


# region 未连接右臂帧


def _check_disconnected_right_arm_streams(client: CameraPipelineClient) -> None:
    """验证右臂帧 API 存在，且当前以明确错误表达未连接。"""

    _expect_stream_disconnected(client.subscribe_right_arm_camera_frames(), "右臂 RGBD")
    _expect_stream_disconnected(
        client.subscribe_right_arm_camera_color_frames(),
        "右臂彩色",
    )
    _expect_stream_disconnected(
        client.subscribe_right_arm_camera_depth_frames(),
        "右臂深度",
    )


def _expect_stream_disconnected(
    stream: Iterator[FrameT],
    api_name: str,
) -> None:
    """确认未连接相机的订阅在首次迭代时失败。"""

    generator = cast(Generator[FrameT, None, None], stream)
    try:
        next(generator)
    except RuntimeError as exc:
        _validate_disconnected_error(exc, api_name)
    else:
        raise RuntimeError(f"{api_name} 未连接时不应返回帧")
    finally:
        generator.close()


def _validate_disconnected_error(exc: RuntimeError, api_name: str) -> None:
    """校验右臂保留 API 的错误语义。"""

    if "not connected" not in str(exc):
        raise RuntimeError(f"{api_name} 返回了非预期错误: {exc}") from exc
    logger.info("{} API 保留验证通过：{}", api_name, exc)


# endregion


if __name__ == "__main__":
    raise SystemExit(main())
