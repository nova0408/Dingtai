from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orin.tray_detection.protocol import OrinTrayDetectionRequest  # noqa: E402
from orin.tray_detection.transport import OrinTrayDetectionRpcClient, ZmqSocketOptions  # noqa: E402

# region 默认参数
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.116:6210"  # Orin 托盘检测服务地址
DEFAULT_CAMERA_NAME = "left_hand_camera"  # 默认逻辑相机名
DEFAULT_RPC_TIMEOUT_MS = 60_000  # Orin RPC 超时，单位 ms
DEFAULT_MAX_FRAMES = 0  # 最多验证帧数，0 表示持续运行直到用户退出
DEFAULT_WINDOW_NAME = "Wuyou tray rpc merged"  # 2D 合并预览窗口名
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 窗口最小长边，单位 像素
DEBUG_COLORS_BGR = (
    (0, 220, 255),
    (80, 200, 120),
    (255, 170, 0),
    (255, 110, 180),
    (140, 180, 255),
    (200, 120, 255),
)
# endregion


# region 主入口
def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    rpc_timeout_ms: int = DEFAULT_RPC_TIMEOUT_MS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> None:
    """验证 Orin 独立托盘检测服务的返回结果与 2D 可视化效果。"""

    logger.info("启动 Orin 托盘检测 RPC 冒烟测试")
    logger.info("service {} camera {}", service_addr, camera_name)
    logger.warning("当前脚本依赖 Orin 侧持久化相机流 service 已经正常运行。")
    rpc_client = OrinTrayDetectionRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=int(rpc_timeout_ms), send_timeout_ms=int(rpc_timeout_ms)),
    )
    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_size_initialized = False
    frame_count = 0
    fps_start = time.perf_counter()
    fps_frames = 0

    try:
        while True:
            frame_count += 1
            fps_frames += 1
            response = rpc_client.call(
                OrinTrayDetectionRequest(
                    request_id=int(frame_count),
                    camera_name=str(camera_name),
                    frame_id=-1,
                    enable_debug=True,
                )
            )
            merged = _build_preview(response)
            if not window_size_initialized:
                image_h, image_w = merged.shape[:2]
                win_w, win_h = _compute_preview_window_size(image_w, image_h, DEFAULT_MIN_2D_WINDOW_LONG_SIDE)
                cv2.resizeWindow(DEFAULT_WINDOW_NAME, win_w, win_h)
                window_size_initialized = True
            cv2.imshow(DEFAULT_WINDOW_NAME, merged)

            key = cv2.waitKey(1)
            if key == 27 or key == ord("q") or key == ord("Q"):
                logger.info("收到用户中断，准备退出。")
                break
            if _cv_window_closed(DEFAULT_WINDOW_NAME):
                logger.info("预览窗口关闭，准备退出。")
                break

            if frame_count % 10 == 0:
                now = time.perf_counter()
                elapsed = max(1e-6, now - fps_start)
                logger.info(
                    "帧数 {} 帧 tray_count {} frame_id {} elapsed {:.1f} ms 预览帧率 {:.1f} fps",
                    frame_count,
                    response.tray_count,
                    response.frame_id,
                    response.elapsed_ms,
                    fps_frames / elapsed,
                )
                fps_start = now
                fps_frames = 0

            if int(max_frames) > 0 and frame_count >= int(max_frames):
                logger.success("达到最大帧数 {} 帧，测试结束。", max_frames)
                break
    finally:
        rpc_client.close()
        _safe_destroy_cv_window(DEFAULT_WINDOW_NAME)


def _parse_cli(argv: list[str]) -> tuple[str, str, int, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="验证 Orin 托盘检测 RPC 结果")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--rpc-timeout-ms", type=int, default=DEFAULT_RPC_TIMEOUT_MS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = parser.parse_args(argv)
    return (
        str(args.service_addr),
        str(args.camera_name),
        int(args.rpc_timeout_ms),
        int(args.max_frames),
    )


# endregion


# region 预览工具
def _build_preview(response) -> np.ndarray:
    """构造 Orin 返回结果的 2D 合并预览。"""

    overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask_preview = np.zeros_like(overlay)
    if response.debug is not None:
        if response.debug.overlay_bgr is not None:
            overlay = response.debug.overlay_bgr
        if response.debug.mask_bgr is not None:
            mask_preview = response.debug.mask_bgr
        for tray_id, tray_mask in enumerate(response.debug.tray_masks):
            _draw_mask_outline(overlay, tray_mask, _debug_color_bgr(tray_id))
    for tray_info in response.tray_results:
        x, y, w, h = tray_info.bbox_xywh
        color = _debug_color_bgr(tray_info.tray_id)
        cv2.rectangle(
            overlay,
            (int(x), int(y)),
            (int(x + w - 1), int(y + h - 1)),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"tray_{tray_info.tray_id} conf {tray_info.confidence_2d:.2f}",
            (int(x), max(14, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"tray_count {response.tray_count} elapsed {response.elapsed_ms:.1f} ms",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        mask_preview,
        f"frame_id {response.frame_id}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if overlay.shape[:2] != mask_preview.shape[:2]:
        mask_preview = cv2.resize(mask_preview, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.hstack([overlay, mask_preview])


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    """在图像上绘制二值掩码外轮廓。"""

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or not np.any(mask_u8 > 0):
        return
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _compute_preview_window_size(src_w: int, src_h: int, min_long_side: int) -> tuple[int, int]:
    """根据最小长边约束计算 2D 预览窗口尺寸。"""

    long_side = max(1, src_w, src_h)
    if long_side >= min_long_side:
        return max(1, src_w), max(1, src_h)
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _debug_color_bgr(index: int) -> tuple[int, int, int]:
    return DEBUG_COLORS_BGR[int(index) % len(DEBUG_COLORS_BGR)]


def _safe_destroy_cv_window(window_name: str) -> None:
    """安全销毁 cv2 窗口。"""

    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def _cv_window_closed(window_name: str) -> bool:
    """判断 cv2 窗口是否已被用户关闭。"""

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


# endregion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        service_addr_arg, camera_name_arg, rpc_timeout_arg, max_frames_arg = _parse_cli(sys.argv[1:])
        main(
            service_addr=service_addr_arg,
            camera_name=camera_name_arg,
            rpc_timeout_ms=rpc_timeout_arg,
            max_frames=max_frames_arg,
        )
    else:
        main()
