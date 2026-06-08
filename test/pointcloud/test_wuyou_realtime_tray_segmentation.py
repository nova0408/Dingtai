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

from src.pointcloud.tray_detection import TrayDetectionPipeline, TrayRuntimeState  # noqa: E402
from src.wuji import WujiZmqCameraClient, WujiZmqCameraConfig, load_wuji_robot_network_config  # noqa: E402
from src.wuji.camera_protocol import WujiCameraName  # noqa: E402

# region 默认参数
DEFAULT_CAMERA_NAME = "left_hand_camera"  # 默认测试相机名
DEFAULT_HOST = load_wuji_robot_network_config().base_control_ip  # wuyou 相机服务主机
DEFAULT_REQUEST_TIMEOUT_MS = 3000  # 控制命令超时，单位 ms
DEFAULT_STREAM_TIMEOUT_MS = 8000  # 图像流超时，单位 ms
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 窗口最小长边，单位 像素
DEFAULT_2D_WINDOW_NAME = "Wuyou tray segmentation merged"  # 2D 预览窗口名
DEFAULT_MAX_FRAMES = 0  # 最多处理帧数，0 表示持续运行直到用户中断
# endregion


# region 主入口
def main(
    camera_name: str = DEFAULT_CAMERA_NAME,
    host: str = DEFAULT_HOST,
    request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    stream_timeout_ms: int = DEFAULT_STREAM_TIMEOUT_MS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> None:
    """验证 wuyou 相机实时托盘分割效果。

    Parameters
    ----------
    camera_name:
        待测试逻辑相机名。
    host:
        wuyou 相机服务主机地址。
    request_timeout_ms:
        控制命令超时，单位 ms。
    stream_timeout_ms:
        图像流等待超时，单位 ms。
    max_frames:
        最多处理帧数，0 表示持续运行直到用户中断。

    Notes
    -----
    本脚本直接使用 `src.pointcloud.tray_detection` 子模块中的托盘分割算法。
    该脚本依赖真实 wuyou 相机链路，当前只做代码静态验证，未在本轮实际连接硬件运行。
    """

    logger.info("启动 wuyou 实时托盘分割测试")
    logger.info("相机 {} host {}", camera_name, host)
    logger.warning("本脚本依赖真实 wuyou 相机服务与 RGBD 数据流，未连硬件时会失败。")

    pipeline = TrayDetectionPipeline(TrayDetectionPipeline.build_default_detector())
    state = TrayRuntimeState()
    client = WujiZmqCameraClient(
        WujiZmqCameraConfig(
            host=str(host),
            request_timeout_ms=int(request_timeout_ms),
            stream_timeout_ms=int(stream_timeout_ms),
        )
    )

    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_size_initialized = False

    frame_count = 0
    fps_start = time.perf_counter()
    fps_frames = 0

    try:
        for frame in client.stream_camera_rgbd_frames(_to_camera_name(camera_name)):
            frame_count += 1
            fps_frames += 1

            detections, from_detector = pipeline.segment_trays(np.asarray(frame.color_bgr, dtype=np.uint8), state)

            merged = _build_2d_preview(
                color_bgr=np.asarray(frame.color_bgr, dtype=np.uint8),
                detections=detections,
                from_detector=bool(from_detector),
                frame_id=frame_count,
            )
            if not window_size_initialized:
                image_h, image_w = merged.shape[:2]
                win_w, win_h = _compute_preview_window_size(image_w, image_h, DEFAULT_MIN_2D_WINDOW_LONG_SIDE)
                cv2.resizeWindow(DEFAULT_2D_WINDOW_NAME, win_w, win_h)
                window_size_initialized = True
            cv2.imshow(DEFAULT_2D_WINDOW_NAME, merged)

            key = cv2.waitKey(1)
            if key == 27 or key == ord("q") or key == ord("Q"):
                logger.info("收到用户中断，准备退出。")
                break
            if _cv_window_closed(DEFAULT_2D_WINDOW_NAME):
                logger.info("预览窗口关闭，准备退出。")
                break

            if frame_count % 10 == 0:
                now = time.perf_counter()
                elapsed = max(1e-6, now - fps_start)
                logger.info(
                    "帧数 {} 帧 检测数 {} 个 来源 {} 预览帧率 {:.1f} fps",
                    frame_count,
                    len(detections),
                    "detector" if from_detector else "fast",
                    fps_frames / elapsed,
                )
                fps_start = now
                fps_frames = 0

            if int(max_frames) > 0 and frame_count >= int(max_frames):
                logger.success("达到最大帧数 {} 帧，测试结束。", max_frames)
                break
    finally:
        try:
            client.stop_camera_depth_stream(_to_camera_name(camera_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("关闭深度流失败：{}", exc)
        client.close()
        _safe_destroy_cv_window(DEFAULT_2D_WINDOW_NAME)


def _parse_cli(argv: list[str]) -> tuple[str, str, int, int, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="测试来自 wuyou 相机流数据的实时托盘分割")
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--request-timeout-ms", type=int, default=DEFAULT_REQUEST_TIMEOUT_MS)
    parser.add_argument("--stream-timeout-ms", type=int, default=DEFAULT_STREAM_TIMEOUT_MS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = parser.parse_args(argv)
    return (
        str(args.camera_name),
        str(args.host),
        int(args.request_timeout_ms),
        int(args.stream_timeout_ms),
        int(args.max_frames),
    )


# endregion


# region 2D 预览
def _build_2d_preview(
    color_bgr: np.ndarray,
    detections: list,
    from_detector: bool,
    frame_id: int,
) -> np.ndarray:
    """构造与现有实时测试风格一致的左右并排 2D 预览。"""

    overlay = np.asarray(color_bgr, dtype=np.uint8).copy()
    mask_preview = np.zeros_like(overlay)
    for index, det in enumerate(detections):
        color_bgr_tuple = _palette_bgr(index)
        mask = np.asarray(det.mask, dtype=np.uint8)
        overlay = _blend_mask_overlay(overlay, mask, color_bgr_tuple, 0.26)
        _draw_mask_outline(overlay, mask, color_bgr_tuple)
        _draw_mask_outline(mask_preview, mask, color_bgr_tuple)
        contour = np.asarray(det.contour, dtype=np.int32)
        if contour.shape[0] >= 3:
            cv2.polylines(overlay, [contour], True, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.polylines(mask_preview, [contour], True, (255, 255, 255), 1, cv2.LINE_AA)
        x, y, w, h = cv2.boundingRect(contour.reshape(-1, 1, 2)) if contour.shape[0] >= 3 else (0, 0, 1, 1)
        text = f"tray_{index} conf {float(det.confidence_2d):.2f}"
        cv2.putText(overlay, text, (int(x), max(14, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr_tuple, 1, cv2.LINE_AA)
        cv2.putText(mask_preview, text, (int(x), max(14, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr_tuple, 1, cv2.LINE_AA)
        cv2.putText(mask_preview, f"area {int(np.count_nonzero(mask))} px", (int(x), min(mask_preview.shape[0] - 8, int(y + h + 16))), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr_tuple, 1, cv2.LINE_AA)

    cv2.putText(
        overlay,
        f"frame {int(frame_id)} source {'detector' if from_detector else 'fast'} trays {len(detections)}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        mask_preview,
        "Tray mask preview",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.hstack([overlay, mask_preview])


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    """在图像上绘制二值 mask 外轮廓。"""

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or not np.any(mask_u8 > 0):
        return
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _blend_mask_overlay(
    base_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    """将托盘掩码按指定颜色半透明叠加到彩色图像。"""

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or base_bgr.shape[:2] != mask_u8.shape[:2]:
        return base_bgr
    mask_bool = mask_u8 > 0
    if not np.any(mask_bool):
        return base_bgr
    result = np.asarray(base_bgr, dtype=np.float32).copy()
    color = np.asarray(color_bgr, dtype=np.float32)
    result[mask_bool] = result[mask_bool] * (1.0 - float(alpha)) + color * float(alpha)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# endregion


# region 基础工具
def _to_camera_name(camera_name: str) -> WujiCameraName:
    """将字符串相机名转换为项目内逻辑相机名类型。"""

    return camera_name  # type: ignore[return-value]


def _palette_bgr(index: int) -> tuple[int, int, int]:
    """生成托盘实例使用的 BGR 颜色。"""

    hue = (37 * int(index)) % 180
    hsv = np.asarray([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0, :]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _compute_preview_window_size(src_w: int, src_h: int, min_long_side: int) -> tuple[int, int]:
    """根据最小长边约束计算 2D 预览窗口尺寸。"""

    long_side = max(1, src_w, src_h)
    if long_side >= min_long_side:
        return max(1, src_w), max(1, src_h)
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


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
        camera_name_arg, host_arg, request_timeout_arg, stream_timeout_arg, max_frames_arg = _parse_cli(sys.argv[1:])
        main(
            camera_name=camera_name_arg,
            host=host_arg,
            request_timeout_ms=request_timeout_arg,
            stream_timeout_ms=stream_timeout_arg,
            max_frames=max_frames_arg,
        )
    else:
        main()
