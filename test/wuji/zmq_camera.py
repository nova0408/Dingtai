from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import WUYOU_HOST, WUYOU_SSH_ALIAS, start_ssh_tunnel, stop_ssh_process  # noqa: E402
from src.wuji import SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL, WujiCameraName, WujiZmqCameraClient  # noqa: E402

# endregion


# region 默认参数

DEFAULT_CAMERA_NAME: WujiCameraName = "left_hand_camera"
DEFAULT_CONTROL_PORT = 5570  # ZMQ 控制口端口，单位 端口号
DEFAULT_LEFT_HAND_STREAM_PORT = 5562  # 左手相机数据口端口，单位 端口号
DEFAULT_REQUEST_TIMEOUT_MS = 3000  # 控制命令超时，单位 ms
DEFAULT_STREAM_TIMEOUT_MS = 5000  # 首帧等待超时，单位 ms
DEFAULT_FORWARD_WAIT_S = 1.0  # SSH 转发建立等待时间，单位 s

# endregion


# region 主流程


def main() -> None:
    """测试左手 ZMQ 相机是否能连上并获取到首帧数据。"""

    logger.info("硬件测试脚本：未连通 orin 或真实左手相机时会失败")
    logger.info("测试相机 {}", DEFAULT_CAMERA_NAME)
    logger.info("相机服务远端地址 {}", WUYOU_HOST)

    control_tunnel_process = start_ssh_tunnel(
        DEFAULT_CONTROL_PORT,
        remote_host=WUYOU_HOST,
        ssh_alias=WUYOU_SSH_ALIAS,
    )
    stream_tunnel_process = start_ssh_tunnel(
        DEFAULT_LEFT_HAND_STREAM_PORT,
        remote_host=WUYOU_HOST,
        ssh_alias=WUYOU_SSH_ALIAS,
    )
    time.sleep(DEFAULT_FORWARD_WAIT_S)

    client = WujiZmqCameraClient(
        host="127.0.0.1",
        control_port=DEFAULT_CONTROL_PORT - 1,
        request_timeout_ms=DEFAULT_REQUEST_TIMEOUT_MS,
        stream_timeout_ms=DEFAULT_STREAM_TIMEOUT_MS,
        camera_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
    )
    try:
        status = client.get_camera_status(DEFAULT_CAMERA_NAME)
        logger.info(
            "相机状态 online {} color_enabled {} depth_enabled {}",
            status.online,
            status.color_enabled,
            status.depth_enabled,
        )
        if not status.online:
            raise RuntimeError("左手相机离线，无法继续取数")

        frame = next(client.stream_camera_rgb_frames(DEFAULT_CAMERA_NAME))
        frame_height, frame_width = frame.color_bgr.shape[:2]
        logger.success("左手相机 RGB 首帧获取成功")
        logger.info("图像宽度 {} 像素", frame_width)
        logger.info("图像高度 {} 像素", frame_height)
        logger.info("图像数据类型 {}", frame.color_bgr.dtype)
    finally:
        client.close()
        stop_ssh_process(control_tunnel_process)
        stop_ssh_process(stream_tunnel_process)


# endregion


if __name__ == "__main__":
    main()
