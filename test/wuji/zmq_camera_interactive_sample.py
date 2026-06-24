from __future__ import annotations

# region 依赖导入
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import WUYOU_HOST, WUYOU_SSH_ALIAS, start_ssh_tunnel, stop_ssh_process  # noqa: E402
from src.wuji import SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL  # noqa: E402
from src.wuji import WujiCameraFrame  # noqa: E402
from src.wuji import WujiCameraName  # noqa: E402
from src.wuji import WujiZmqCameraClient  # noqa: E402

# endregion


# region 默认参数

DEFAULT_CAMERA_NAME: WujiCameraName = "left_hand_camera"  # 默认采样相机名
DEFAULT_ZMQ_HOST = WUYOU_HOST  # ZMQ 相机服务主机地址
DEFAULT_CONTROL_PORT = 5570  # ZMQ 控制口端口，单位 端口号
DEFAULT_LEFT_HAND_STREAM_PORT = 5562  # 左手相机数据口端口，单位 端口号
DEFAULT_REQUEST_TIMEOUT_MS = 3000  # 控制命令超时，单位 ms
DEFAULT_STREAM_TIMEOUT_MS = 5000  # 首帧等待超时，单位 ms
DEFAULT_FORWARD_WAIT_S = 1.0  # SSH 转发建立等待时间，单位 s
DEFAULT_ROOT_DIR = Path(__file__).resolve().parent  # 采样结果根目录，同级目录
DEFAULT_WINDOW_NAME = "Wuji ZMQ left-hand sampler"  # 预览窗口名
DEFAULT_RGB_EXT = ".jpg"  # RGB 图片后缀
DEFAULT_DEPTH_EXT = ".npy"  # 深度矩阵后缀

# endregion


# region 主流程


def main() -> None:
    """交互式采样左手 ZMQ 相机流。"""

    logger.info("硬件测试脚本：未连通 wuyou 或真实左手相机时会失败")
    logger.info("测试相机 {}", DEFAULT_CAMERA_NAME)
    logger.info("相机服务远端地址 {}", DEFAULT_ZMQ_HOST)

    session_dir = DEFAULT_ROOT_DIR / datetime.now().strftime("%m%d-%H%M")
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info("采样输出目录 {}", session_dir)

    control_tunnel_process = start_ssh_tunnel(
        DEFAULT_CONTROL_PORT,
        remote_host=DEFAULT_ZMQ_HOST,
        ssh_alias=WUYOU_SSH_ALIAS,
    )
    stream_tunnel_process = start_ssh_tunnel(
        DEFAULT_LEFT_HAND_STREAM_PORT,
        remote_host=DEFAULT_ZMQ_HOST,
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
        _run_interactive_capture(client=client, camera_name=DEFAULT_CAMERA_NAME, session_dir=session_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("相机交互采样失败: {}", exc)
        raise
    finally:
        client.close()
        stop_ssh_process(control_tunnel_process)
        stop_ssh_process(stream_tunnel_process)


# endregion


# region 交互采样


def _run_interactive_capture(client: WujiZmqCameraClient, camera_name: WujiCameraName, session_dir: Path) -> None:
    """等待空格开始连续采样，并将每帧 RGB 和深度矩阵写入同一目录。"""

    status = client.get_camera_status(camera_name)
    logger.info(
        "相机状态 online {} color_enabled {} depth_enabled {}",
        status.online,
        status.color_enabled,
        status.depth_enabled,
    )
    if not status.online:
        raise RuntimeError("左手相机离线，无法继续取数")

    intrinsics = client.get_camera_intrinsics(camera_name)
    intrinsics_path = session_dir / "intrinsics.json"
    intrinsics_payload = {
        "camera_name": camera_name,
        "host": DEFAULT_ZMQ_HOST,
        "capture_started_at": datetime.now().isoformat(timespec="seconds"),
        "intrinsics": asdict(intrinsics),
    }
    intrinsics_path.write_text(json.dumps(intrinsics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("内参已写入 {}", intrinsics_path)

    logger.info("按键说明：空格 采集当前帧；Q 或 ESC 退出。")
    logger.info("每按一次空格，只保存一帧 RGB 和一帧深度矩阵，避免连续采样导致排队。")

    frame_index = 0
    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            preview_frame = next(client.stream_camera_rgb_frames(camera_name))
            preview_bgr = _ensure_bgr_uint8(preview_frame)
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_bgr)
            key = cv2.waitKey(1 if frame_index > 0 else 30) & 0xFF
            if key in (ord("q"), 27):
                logger.warning("收到退出指令，结束采样")
                break
            if key != 32:
                continue

            logger.info("收到空格指令，开始抓取第 {} 帧", frame_index)
            frame = next(client.stream_camera_rgbd_frames(camera_name))
            rgb_path = _save_rgb_frame(session_dir, frame_index, frame)
            depth_path = _save_depth_frame(session_dir, frame_index, frame)
            logger.success("已保存 frame {} rgb {} depth {}", frame_index, rgb_path.name, depth_path.name)
            frame_index += 1
    finally:
        cv2.destroyWindow(DEFAULT_WINDOW_NAME)


# endregion


# region 文件输出


def _save_rgb_frame(session_dir: Path, frame_index: int, frame: WujiCameraFrame) -> Path:
    """保存 RGB 图像。"""

    rgb_path = session_dir / f"rgb_{frame_index:04d}{DEFAULT_RGB_EXT}"
    if not cv2.imwrite(str(rgb_path), frame.color_bgr):
        raise RuntimeError(f"rgb image write failed: {rgb_path}")
    return rgb_path


def _save_depth_frame(session_dir: Path, frame_index: int, frame: WujiCameraFrame) -> Path:
    """保存深度矩阵。"""

    if frame.depth is None:
        raise RuntimeError("rgbd frame depth is None")
    depth_matrix = np.asarray(frame.depth, dtype=np.uint16)
    depth_path = session_dir / f"d_{frame_index:04d}{DEFAULT_DEPTH_EXT}"
    np.save(depth_path, depth_matrix)
    return depth_path


# endregion


# region 基础工具


def _ensure_bgr_uint8(frame: WujiCameraFrame) -> np.ndarray:
    """确保预览帧是可直接显示的 BGR 图像。"""

    color_bgr = np.asarray(frame.color_bgr)
    if color_bgr.dtype != np.uint8:
        color_bgr = np.clip(color_bgr, 0, 255).astype(np.uint8)
    return color_bgr


# endregion


if __name__ == "__main__":
    main()
