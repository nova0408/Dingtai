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

from orin.grasp_pose_pipeline.protocol import GraspPosePipelineRequest  # noqa: E402
from orin.grasp_pose_pipeline.transport import GraspPosePipelineRpcClient, ZmqSocketOptions  # noqa: E402


# region 默认参数
DEFAULT_PIPELINE_SERVICE_ADDR = "tcp://192.168.1.116:6220"  # Orin 抓取位姿主服务地址
DEFAULT_CAMERA_NAME = "left_hand_camera"  # 默认逻辑相机名
DEFAULT_TARGET_TRAY_INDEX = 0  # 默认目标托盘编号，按从左到右编号
DEFAULT_RPC_TIMEOUT_MS = 60_000  # RPC 超时，单位 ms
DEFAULT_WINDOW_NAME = "Wuyou tray0 grasp pose merged"  # 2D 预览窗口名
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 窗口最小长边，单位 像素
DEFAULT_MAX_FRAMES = 0  # 最多验证帧数，0 表示持续运行直到用户退出
# endregion


# region 主入口
def main(
    pipeline_service_addr: str = DEFAULT_PIPELINE_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    target_tray_index: int = DEFAULT_TARGET_TRAY_INDEX,
    rpc_timeout_ms: int = DEFAULT_RPC_TIMEOUT_MS,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> None:
    """验证统一 pipeline 对托盘 0 的最终抓取位姿与调试可视化。"""

    logger.info("启动 grasp_pose_pipeline 抓取位姿冒烟测试")
    logger.warning("当前脚本默认只访问 Orin grasp_pose_pipeline 单端口，并直接展示目标托盘最终位姿。")
    rpc_client = GraspPosePipelineRpcClient(
        connect_addr=str(pipeline_service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=int(rpc_timeout_ms), send_timeout_ms=int(rpc_timeout_ms)),
    )
    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    frame_count = 0
    fps_start = time.perf_counter()
    fps_frames = 0
    window_size_initialized = False
    try:
        while True:
            frame_count += 1
            fps_frames += 1
            response = rpc_client.call(
                GraspPosePipelineRequest(
                    request_id=int(frame_count),
                    camera_name=str(camera_name),
                    frame_id=-1,
                    target_tray_index=int(target_tray_index),
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
                break
            if _cv_window_closed(DEFAULT_WINDOW_NAME):
                break
            if frame_count % 10 == 0:
                now = time.perf_counter()
                elapsed = max(1e-6, now - fps_start)
                logger.info(
                    "帧数 {} 帧 frame_id {} elapsed {:.1f} ms 预览帧率 {:.1f} fps",
                    frame_count,
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


def _parse_cli(argv: list[str]) -> tuple[str, str, int, int, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="验证 grasp_pose_pipeline 返回的抓取位姿")
    parser.add_argument("--pipeline-service-addr", type=str, default=DEFAULT_PIPELINE_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--target-tray-index", type=int, default=DEFAULT_TARGET_TRAY_INDEX)
    parser.add_argument("--rpc-timeout-ms", type=int, default=DEFAULT_RPC_TIMEOUT_MS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = parser.parse_args(argv)
    return str(args.pipeline_service_addr), str(args.camera_name), int(args.target_tray_index), int(args.rpc_timeout_ms), int(args.max_frames)


# endregion


# region 预览工具
def _build_preview(response) -> np.ndarray:
    """构造远端 RGB + HSV 深度的合并预览。"""

    color_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth_bgr = np.zeros_like(color_bgr)
    if response.debug is not None:
        if response.debug.color_bgr is not None:
            color_bgr = np.asarray(response.debug.color_bgr, dtype=np.uint8)
        if response.debug.depth_mm is not None:
            depth_bgr = _build_hsv_depth_view(np.asarray(response.debug.depth_mm, dtype=np.uint16))
    return np.hstack([_draw_result_overlay(color_bgr, response), _draw_result_overlay(depth_bgr, response)])


def _draw_result_overlay(base_bgr: np.ndarray, response) -> np.ndarray:
    """在底图上叠加托盘框、缺口、缺口中心与最终位姿文本。"""

    overlay = np.asarray(base_bgr, dtype=np.uint8).copy()
    debug = response.debug
    if debug is not None:
        for tray_id, tray_mask in enumerate(debug.tray_instance_masks):
            _draw_mask_outline(overlay, tray_mask, _tray_color_bgr(tray_id))
    for tray in response.tray_results:
        x, y, w, h = tray.bbox_xywh
        color = (0, 255, 255) if int(tray.tray_id) == int(response.selected_tray_index) else _tray_color_bgr(tray.tray_id)
        cv2.rectangle(overlay, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), color, 2 if int(tray.tray_id) == int(response.selected_tray_index) else 1, cv2.LINE_AA)
        cv2.putText(overlay, f"tray_{tray.tray_id}", (int(x), max(16, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    if response.selected_result is not None and debug is not None:
        result = response.selected_result
        if debug.selected_tray_mask is not None:
            _blend_mask(overlay, debug.selected_tray_mask, (0, 180, 180), 0.18)
        if debug.near_plane_mask is not None:
            _blend_mask(overlay, debug.near_plane_mask, (0, 0, 255), 0.34)
            _draw_mask_outline(overlay, debug.near_plane_mask, (0, 0, 255))
        if debug.no_hole_mask is not None:
            _blend_mask(overlay, debug.no_hole_mask, (255, 0, 0), 0.30)
            _draw_mask_outline(overlay, debug.no_hole_mask, (255, 0, 0))
        if result.opening_quad_uv is not None:
            quad = np.round(np.asarray(result.opening_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(overlay, [quad], True, (0, 255, 0), 1, cv2.LINE_AA)
        if result.top_quad_uv is not None:
            top_quad = np.round(np.asarray(result.top_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(overlay, [top_quad], True, (255, 0, 0), 1, cv2.LINE_AA)
        if result.opening_center_uv is not None:
            u = int(round(float(result.opening_center_uv[0])))
            v = int(round(float(result.opening_center_uv[1])))
            cv2.circle(overlay, (u, v), 4, (0, 255, 0), -1)
        if result.pose is not None:
            pose_text = "grasp ({0:.1f}, {1:.1f}, {2:.1f}) mm".format(
                float(result.pose.grasp_point_mm[0]),
                float(result.pose.grasp_point_mm[1]),
                float(result.pose.grasp_point_mm[2]),
            )
            cv2.putText(overlay, pose_text, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"frame_id {response.frame_id} elapsed {response.elapsed_ms:.1f} ms", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def _build_hsv_depth_view(depth_mm: np.ndarray) -> np.ndarray:
    """把远端深度图转换为 HSV 着色。"""

    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1.0)
    hsv = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if np.any(valid):
        z_min = float(np.percentile(depth[valid], 2))
        z_max = float(np.percentile(depth[valid], 98))
        norm = np.clip((depth - z_min) / max(1e-6, z_max - z_min), 0.0, 1.0)
        hsv[..., 0] = np.where(valid, np.rint((1.0 - norm) * 120.0), 0).astype(np.uint8)
        hsv[..., 1] = np.where(valid, 255, 0).astype(np.uint8)
        hsv[..., 2] = np.where(valid, 255, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _blend_mask(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    """把二值掩码以半透明形式叠加到底图。"""

    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(mask_bool):
        return
    base = np.asarray(base_bgr, dtype=np.float32)
    color = np.asarray(color_bgr, dtype=np.float32)
    base[mask_bool] = base[mask_bool] * (1.0 - float(alpha)) + color * float(alpha)
    base_bgr[:, :, :] = np.clip(base, 0.0, 255.0).astype(np.uint8)


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    """在图像上绘制二值掩码外轮廓。"""

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or not np.any(mask_u8 > 0):
        return
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _tray_color_bgr(index: int) -> tuple[int, int, int]:
    palette = (
        (0, 220, 255),
        (80, 200, 120),
        (255, 170, 0),
        (255, 110, 180),
        (140, 180, 255),
        (200, 120, 255),
    )
    return palette[int(index) % len(palette)]


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
        pipeline_service_addr_arg, camera_name_arg, target_tray_index_arg, rpc_timeout_arg, max_frames_arg = _parse_cli(sys.argv[1:])
        main(pipeline_service_addr_arg, camera_name_arg, target_tray_index_arg, rpc_timeout_arg, max_frames_arg)
    else:
        main()
