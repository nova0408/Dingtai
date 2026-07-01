from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from types import ModuleType
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TEST_WUJI_ROOT = PROJECT_ROOT / "test" / "wuji"
if str(TEST_WUJI_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_WUJI_ROOT))

from common import WUYOU_HOST, WUYOU_SSH_ALIAS, start_ssh_tunnel, stop_ssh_process
from orin.tray_detection.protocol import OrinTrayDetectionRequest, OrinTrayDetectionResponse
from orin.tray_detection.transport import OrinTrayDetectionRpcClient, ZmqSocketOptions
from src.wuji import SUPPORTED_WUJI_ZMQ_CAMERAS, SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL, WujiZmqCameraClient
from src.wuji.camera_protocol import WujiCameraFrame, WujiCameraIntrinsicsInfo, WujiCameraName
from src.wuji.zmq_camera_catalog import get_wuji_zmq_camera_endpoint

# region 默认参数

DEFAULT_SERVICE_ADDR = "tcp://127.0.0.1:6210"
"远端托盘检测服务地址。"

DEFAULT_CAMERA_NAME: WujiCameraName = "left_hand_camera"
"默认相机逻辑名称。"

DEFAULT_CONTROL_PORT = 5570
"无际 ZMQ 相机控制口端口号。"

DEFAULT_FORWARD_WAIT_S = 1.0
"SSH 转发建立等待时间，单位 秒。"

DEFAULT_TARGET_TRAY_INDEX = 0
"默认目标托盘索引，按远端服务返回顺序。"

DEFAULT_COMPUTE_INTERVAL_MS = 120
"远端托盘检测请求间隔，单位 ms。"

DEFAULT_RPC_TIMEOUT_MS = 30_000
"远端托盘检测 RPC 超时，单位 ms。"

DEFAULT_FRAME_CACHE_SIZE = 24
"本地帧缓存数量，用于按 frame_id 对齐远端托盘结果。"

DEFAULT_WINDOW_NAME = "opening_detection_local"
"cv2 预览窗口名称。"

DEFAULT_MIN_WINDOW_LONG_SIDE = 960
"cv2 预览窗口最小长边，单位 像素。"

DEFAULT_MAX_DEPTH_MM = 5000.0
"点云构造使用的最大深度阈值，单位 mm。"

DEFAULT_MIN_LOCAL_POINTS = 80
"开口局部点云最小点数阈值，单位 点。"

DEFAULT_INIT_RETRY_INTERVAL_S = 1.5
"相机流初始化失败后的重试间隔，单位 秒。"

# endregion


# region 数据结构


@dataclass(frozen=True)
class OpeningDetectionRuntime:
    """按文件加载的开口检测运行时入口集合。"""

    opening_pipeline_cls: type[Any]
    "开口检测流程类。"

    pose_estimator_cls: type[Any]
    "抓取位姿估计器类。"

    temporal_state_cls: type[Any]
    "位姿时序稳定状态类。"


@dataclass(frozen=True)
class ProjectionIntrinsics:
    """本地投影内参代理。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ComputeResult:
    """单次本地开口检测与位姿计算结果。"""

    frame_id: int
    target_tray_id: int
    tray_response: OrinTrayDetectionResponse
    opening: Any | None
    plane: Any | None
    grasp: Any | None
    near_plane_mask: np.ndarray | None
    no_hole_mask: np.ndarray | None
    error: str | None


@dataclass(frozen=True)
class CameraRuntimeState:
    """相机流初始化结果。"""

    client: WujiZmqCameraClient | None
    intrinsics: WujiCameraIntrinsicsInfo | None
    error: str | None


# endregion


# region 主流程


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: WujiCameraName = DEFAULT_CAMERA_NAME,
    target_tray_index: int = DEFAULT_TARGET_TRAY_INDEX,
    compute_interval_ms: int = DEFAULT_COMPUTE_INTERVAL_MS,
) -> int:
    logger.info("启动 opening_detection_local")
    opening_runtime = _load_opening_detection_runtime()
    remote_endpoint = get_wuji_zmq_camera_endpoint(
        camera_name,
        supported_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS,
    )
    local_endpoint = get_wuji_zmq_camera_endpoint(
        camera_name,
        supported_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
    )
    tray_client = OrinTrayDetectionRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=DEFAULT_RPC_TIMEOUT_MS, send_timeout_ms=DEFAULT_RPC_TIMEOUT_MS),
    )
    opening_pipeline = opening_runtime.opening_pipeline_cls()
    pose_estimator = opening_runtime.pose_estimator_cls()
    temporal_state = opening_runtime.temporal_state_cls()

    try:
        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        latest_result: ComputeResult | None = None
        preview_frames = 0
        fps_t0 = time.perf_counter()
        window_ready = False
        while True:
            init_state = _open_wuji_camera_runtime(
                camera_name=camera_name,
                remote_endpoint_stream_port=int(remote_endpoint.stream_port),
            )
            if init_state.error is not None or init_state.client is None or init_state.intrinsics is None:
                logger.warning("相机流初始化失败：{}", init_state.error)
                wait_canvas = _build_status_canvas(init_state.error or "unknown camera init error")
                if not window_ready:
                    win_w, win_h = _compute_preview_window_size(
                        wait_canvas.shape[1],
                        wait_canvas.shape[0],
                        DEFAULT_MIN_WINDOW_LONG_SIDE,
                    )
                    cv2.resizeWindow(DEFAULT_WINDOW_NAME, win_w, win_h)
                    window_ready = True
                cv2.imshow(DEFAULT_WINDOW_NAME, wait_canvas)
                key = cv2.waitKey(max(1, int(round(DEFAULT_INIT_RETRY_INTERVAL_S * 1000.0))))
                _close_camera_runtime(init_state)
                if key in (27, ord("q"), ord("Q")) or _cv_window_closed(DEFAULT_WINDOW_NAME):
                    break
                continue

            camera_client = init_state.client
            intrinsics = init_state.intrinsics
            logger.info(
                "相机初始化成功 host {} control_local {} stream_local {} fx {:.3f} px fy {:.3f} px",
                "127.0.0.1",
                DEFAULT_CONTROL_PORT - 1,
                int(local_endpoint.stream_port),
                intrinsics.fx,
                intrinsics.fy,
            )

            frame_cache: dict[int, WujiCameraFrame] = {}
            frame_order: deque[int] = deque(maxlen=max(1, int(DEFAULT_FRAME_CACHE_SIZE)))
            latest_frame: WujiCameraFrame | None = None
            last_request_ts = 0.0
            try:
                for frame in camera_client.stream_camera_rgbd_frames(camera_name):
                    latest_frame = frame
                    _cache_frame(frame_cache, frame_order, frame)
                    now = time.perf_counter()
                    if now - last_request_ts >= max(0.02, float(compute_interval_ms) / 1000.0):
                        latest_result = _request_and_compute_once(
                            tray_client=tray_client,
                            opening_pipeline=opening_pipeline,
                            pose_estimator=pose_estimator,
                            temporal_state=temporal_state,
                            camera_name=camera_name,
                            intrinsics=intrinsics,
                            target_tray_index=int(target_tray_index),
                            frame_cache=frame_cache,
                            latest_frame=latest_frame,
                        )
                        last_request_ts = now
                        _log_compute_result(latest_result)

                    canvas = _build_preview_canvas(frame, latest_result)
                    if not window_ready:
                        win_w, win_h = _compute_preview_window_size(
                            canvas.shape[1],
                            canvas.shape[0],
                            DEFAULT_MIN_WINDOW_LONG_SIDE,
                        )
                        cv2.resizeWindow(DEFAULT_WINDOW_NAME, win_w, win_h)
                        window_ready = True
                    cv2.imshow(DEFAULT_WINDOW_NAME, canvas)
                    preview_frames += 1
                    if preview_frames % 30 == 0:
                        elapsed = max(1e-6, time.perf_counter() - fps_t0)
                        logger.info("预览帧率 {:.1f} fps", preview_frames / elapsed)
                    key = cv2.waitKey(1)
                    if key in (27, ord("q"), ord("Q")) or _cv_window_closed(DEFAULT_WINDOW_NAME):
                        return 0
            except Exception as exc:  # noqa: BLE001
                stream_error = f"{type(exc).__name__}: {exc}"
                logger.warning("相机流暂不可用：{}", stream_error)
                wait_canvas = _build_status_canvas(stream_error)
                cv2.imshow(DEFAULT_WINDOW_NAME, wait_canvas)
                key = cv2.waitKey(max(1, int(round(DEFAULT_INIT_RETRY_INTERVAL_S * 1000.0))))
                if key in (27, ord("q"), ord("Q")) or _cv_window_closed(DEFAULT_WINDOW_NAME):
                    break
            finally:
                _close_camera_runtime(init_state)
    finally:
        tray_client.close()
        _safe_destroy_cv_window(DEFAULT_WINDOW_NAME)
    return 0


# endregion


# region 远端托盘结果接入


def _request_and_compute_once(
    tray_client: OrinTrayDetectionRpcClient,
    opening_pipeline: Any,
    pose_estimator: Any,
    temporal_state: Any,
    camera_name: WujiCameraName,
    intrinsics: WujiCameraIntrinsicsInfo,
    target_tray_index: int,
    frame_cache: dict[int, WujiCameraFrame],
    latest_frame: WujiCameraFrame | None,
) -> ComputeResult:
    tray_response = tray_client.call(
        OrinTrayDetectionRequest(
            request_id=int(time.time() * 1000.0),
            camera_name=str(camera_name),
            frame_id=-1,
            enable_debug=True,
        )
    )
    matched_frame = frame_cache.get(int(tray_response.frame_id))
    if matched_frame is None:
        matched_frame = latest_frame
    if matched_frame is None:
        return ComputeResult(
            frame_id=int(tray_response.frame_id),
            target_tray_id=int(target_tray_index),
            tray_response=tray_response,
            opening=None,
            plane=None,
            grasp=None,
            near_plane_mask=None,
            no_hole_mask=None,
            error="本地尚未缓存到可用于对齐的相机帧",
        )
    return _compute_from_remote_tray_result(
        frame=matched_frame,
        tray_response=tray_response,
        opening_pipeline=opening_pipeline,
        pose_estimator=pose_estimator,
        temporal_state=temporal_state,
        intrinsics=intrinsics,
        target_tray_index=target_tray_index,
    )


def _cache_frame(frame_cache: dict[int, WujiCameraFrame], frame_order: deque[int], frame: WujiCameraFrame) -> None:
    frame_id = _resolve_frame_id(frame)
    frame_cache[frame_id] = frame
    frame_order.append(frame_id)
    while len(frame_order) > frame_order.maxlen:
        expired = frame_order.popleft()
        frame_cache.pop(expired, None)
    if len(frame_order) == frame_order.maxlen:
        oldest = frame_order[0]
        stale_ids = [cached_id for cached_id in frame_cache.keys() if cached_id < oldest]
        for stale_id in stale_ids:
            frame_cache.pop(stale_id, None)


# endregion


# region 本机开口检测与位姿计算


def _compute_from_remote_tray_result(
    frame: WujiCameraFrame,
    tray_response: OrinTrayDetectionResponse,
    opening_pipeline: Any,
    pose_estimator: Any,
    temporal_state: Any,
    intrinsics: WujiCameraIntrinsicsInfo,
    target_tray_index: int,
) -> ComputeResult:
    if tray_response.error is not None:
        return ComputeResult(
            frame_id=int(tray_response.frame_id),
            target_tray_id=int(target_tray_index),
            tray_response=tray_response,
            opening=None,
            plane=None,
            grasp=None,
            near_plane_mask=None,
            no_hole_mask=None,
            error=str(tray_response.error),
        )
    if tray_response.debug is None or len(tray_response.tray_results) == 0:
        return ComputeResult(
            frame_id=int(tray_response.frame_id),
            target_tray_id=int(target_tray_index),
            tray_response=tray_response,
            opening=None,
            plane=None,
            grasp=None,
            near_plane_mask=None,
            no_hole_mask=None,
            error="远端 tray_detection 未返回可用调试掩码",
        )
    target_result_index = -1
    for result_index, tray_info in enumerate(tray_response.tray_results):
        if int(tray_info.tray_id) == int(target_tray_index):
            target_result_index = result_index
            break
    if target_result_index < 0:
        return ComputeResult(
            frame_id=int(tray_response.frame_id),
            target_tray_id=int(target_tray_index),
            tray_response=tray_response,
            opening=None,
            plane=None,
            grasp=None,
            near_plane_mask=None,
            no_hole_mask=None,
            error=f"未找到目标托盘 tray_id={target_tray_index}",
        )

    tray_mask = np.asarray(tray_response.debug.tray_masks[int(target_result_index)], dtype=np.uint8)
    color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
    depth = frame.depth
    if depth is None:
        return ComputeResult(
            frame_id=int(tray_response.frame_id),
            target_tray_id=int(target_tray_index),
            tray_response=tray_response,
            opening=None,
            plane=None,
            grasp=None,
            near_plane_mask=None,
            no_hole_mask=None,
            error="本地 RGBD 流当前帧不包含深度图",
        )
    depth_mm = np.asarray(depth, dtype=np.float64)

    try:
        _, hp_gray, hp_edge = _compute_high_contrast_domain(color_bgr)
        opening = opening_pipeline.detect_opening(color_bgr, tray_mask, hp_gray)
        near_plane_mask, no_hole_mask = opening_pipeline.compute_mask_pipeline(
            tray_mask,
            True,
            opening,
            hp_gray,
            hp_edge,
        )
        xyz, rgb = _rgbd_to_points(
            depth_mm=depth_mm,
            color_bgr=color_bgr,
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.cx),
            cy=float(intrinsics.cy),
        )
        uv, valid = _project_points_to_image(
            xyz=xyz,
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.cx),
            cy=float(intrinsics.cy),
            width=int(color_bgr.shape[1]),
            height=int(color_bgr.shape[0]),
        )
        xyz_local = opening_pipeline.filter_opening_local_points(
            xyz=xyz,
            rgb=rgb,
            opening=opening,
            img_w=int(color_bgr.shape[1]),
            img_h=int(color_bgr.shape[0]),
            uv=uv,
            valid=valid,
        )
        if xyz_local.shape[0] < int(DEFAULT_MIN_LOCAL_POINTS):
            return ComputeResult(
                frame_id=int(tray_response.frame_id),
                target_tray_id=int(target_tray_index),
                tray_response=tray_response,
                opening=opening,
                plane=None,
                grasp=None,
                near_plane_mask=near_plane_mask,
                no_hole_mask=no_hole_mask,
                error=f"开口局部点不足：{xyz_local.shape[0]} 点",
            )
        plane = pose_estimator.estimate_plane(xyz_local)
        top_normal = opening_pipeline.estimate_top_plane_normal(xyz, no_hole_mask, uv, valid)
        top_normal = pose_estimator.stabilize_top_normal(top_normal, temporal_state)
        grasp = pose_estimator.compute_grasp(
            opening=opening,
            plane=plane,
            intrinsics=ProjectionIntrinsics(
                width=int(color_bgr.shape[1]),
                height=int(color_bgr.shape[0]),
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                cx=float(intrinsics.cx),
                cy=float(intrinsics.cy),
            ),
            top_ref_normal=top_normal,
        )
        grasp = pose_estimator.stabilize_grasp_result(grasp, temporal_state)
        return ComputeResult(
            frame_id=int(tray_response.frame_id),
            target_tray_id=int(target_tray_index),
            tray_response=tray_response,
            opening=opening,
            plane=plane,
            grasp=grasp,
            near_plane_mask=near_plane_mask,
            no_hole_mask=no_hole_mask,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return ComputeResult(
            frame_id=int(tray_response.frame_id),
            tray_response=tray_response,
            opening=None,
            plane=None,
            grasp=None,
            near_plane_mask=None,
            no_hole_mask=None,
            error=f"{type(exc).__name__}: {exc}",
        )


# endregion


# region 预览绘制


def _build_preview_canvas(
    frame: WujiCameraFrame,
    result: ComputeResult | None,
) -> np.ndarray:
    left = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
    if frame.depth is None:
        right = np.zeros_like(left)
    else:
        right = _build_depth_view(np.asarray(frame.depth, dtype=np.uint16))
    if result is not None:
        left = _draw_remote_overlay(left, result.tray_response, result.target_tray_id)
        if result.near_plane_mask is not None:
            _blend_mask(left, result.near_plane_mask, (0, 140, 255), 0.22)
        if result.no_hole_mask is not None:
            _blend_mask(left, result.no_hole_mask, (255, 120, 0), 0.18)
        if result.opening is not None:
            _draw_opening(left, result.opening)
        if result.grasp is not None:
            _draw_grasp_text(left, result.grasp)
        if (
            getattr(result.tray_response, "debug", None) is not None
            and result.tray_response.debug.mask_bgr is not None
        ):
            right = np.asarray(result.tray_response.debug.mask_bgr, dtype=np.uint8).copy()
            if result.near_plane_mask is not None:
                _draw_mask_outline(right, result.near_plane_mask, (0, 140, 255))
            if result.no_hole_mask is not None:
                _draw_mask_outline(right, result.no_hole_mask, (255, 120, 0))
            if result.opening is not None:
                _draw_opening(right, result.opening)
    merged = np.hstack([left, right])
    cv2.putText(
        merged,
        f"local_frame { _resolve_frame_id(frame) }  tray_frame { -1 if result is None else result.frame_id }",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        merged,
        "left: tray/opening/grasp  right: remote mask preview",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if result is not None and result.error is not None:
        cv2.putText(
            merged,
            result.error[:120],
            (12, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )
    return merged


def _draw_remote_overlay(
    base_bgr: np.ndarray,
    tray_response: OrinTrayDetectionResponse,
    target_tray_id: int,
) -> np.ndarray:
    out = np.asarray(base_bgr, dtype=np.uint8).copy()
    debug = tray_response.debug
    if debug is not None:
        for idx, tray_info in enumerate(tray_response.tray_results):
            if int(tray_info.tray_id) != int(target_tray_id):
                continue
            if idx < len(debug.tray_masks):
                _draw_mask_outline(out, debug.tray_masks[idx], _tray_color_bgr(int(tray_info.tray_id)))
    for tray in tray_response.tray_results:
        if int(tray.tray_id) != int(target_tray_id):
            continue
        x, y, w, h = tray.bbox_xywh
        color = _tray_color_bgr(int(tray.tray_id))
        cv2.rectangle(out, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            f"tray_{tray.tray_id}",
            (int(x), max(18, int(y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _draw_opening(image_bgr: np.ndarray, opening: Any) -> None:
    quad = np.round(np.asarray(opening.quad_uv, dtype=np.float64)).astype(np.int32)
    cv2.polylines(image_bgr, [quad], True, (0, 0, 255), 1, cv2.LINE_AA)
    u = int(round(float(opening.center_uv[0])))
    v = int(round(float(opening.center_uv[1])))
    cv2.circle(image_bgr, (u, v), 3, (0, 255, 0), -1)
    cv2.putText(
        image_bgr,
        f"opening {opening.score:.2f}",
        (u + 6, v - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_grasp_text(image_bgr: np.ndarray, grasp: Any) -> None:
    rpy = _rotation_matrix_to_rpy_deg(grasp.rotation)
    cv2.putText(
        image_bgr,
        f"grasp {grasp.grasp_point[0]:.1f}, {grasp.grasp_point[1]:.1f}, {grasp.grasp_point[2]:.1f} mm",
        (12, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        f"rpy {rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f} deg",
        (12, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _build_depth_view(depth_mm: np.ndarray) -> np.ndarray:
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
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    if not np.any(mask_bool):
        return
    base = np.asarray(base_bgr, dtype=np.float32)
    color = np.asarray(color_bgr, dtype=np.float32)
    base[mask_bool] = base[mask_bool] * (1.0 - float(alpha)) + color * float(alpha)
    base_bgr[:, :, :] = np.clip(base, 0.0, 255.0).astype(np.uint8)


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


# endregion


# region 数学与工具


def _compute_high_contrast_domain(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blur = cv2.GaussianBlur(bgr, (0, 0), 2.6)
    highpass = cv2.addWeighted(bgr, 1.90, blur, -0.90, 0.0)
    gray = cv2.cvtColor(highpass, cv2.COLOR_BGR2GRAY)
    gray_f = cv2.bilateralFilter(gray, d=7, sigmaColor=42, sigmaSpace=42)
    gray_f = cv2.morphologyEx(gray_f, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    edge = cv2.Canny(gray_f, 42, 118)
    return highpass, gray_f, edge


def _rgbd_to_points(
    depth_mm: np.ndarray,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_mm.shape[:2]
    v, u = np.indices((h, w))
    z = np.asarray(depth_mm, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1.0) & (z < float(DEFAULT_MAX_DEPTH_MM))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    rgb = np.asarray(color_bgr, dtype=np.float64) / 255.0
    pts = np.stack(
        [
            x[valid],
            y[valid],
            z[valid],
            rgb[..., 0][valid],
            rgb[..., 1][valid],
            rgb[..., 2][valid],
        ],
        axis=1,
    )
    return pts[:, :3], pts[:, 3:]


def _project_points_to_image(
    xyz: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    z = xyz[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)
    if np.any(valid):
        uu = np.rint(xyz[valid, 0] * fx / z[valid] + cx).astype(np.int32)
        vv = np.rint(xyz[valid, 1] * fy / z[valid] + cy).astype(np.int32)
        in_bounds = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        idx = np.where(valid)[0][in_bounds]
        u[idx] = uu[in_bounds]
        v[idx] = vv[in_bounds]
    return np.stack([u, v], axis=1), (u >= 0) & (v >= 0)


def _rotation_matrix_to_rpy_deg(rot: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.asarray([roll, pitch, yaw], dtype=np.float64))


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


def _resolve_frame_id(frame: WujiCameraFrame) -> int:
    if frame.sequence_id is not None:
        return int(frame.sequence_id)
    timestamp = frame.timestamp
    if isinstance(timestamp, (int, np.integer)):
        return int(timestamp)
    return int(time.time() * 1000.0)


def _compute_preview_window_size(src_w: int, src_h: int, min_long_side: int) -> tuple[int, int]:
    long_side = max(1, src_w, src_h)
    if long_side >= min_long_side:
        return max(1, src_w), max(1, src_h)
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _safe_destroy_cv_window(window_name: str) -> None:
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def _cv_window_closed(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _log_compute_result(result: ComputeResult) -> None:
    if result.error is not None:
        logger.warning("托盘帧 {} 计算失败：{}", result.frame_id, result.error)
        return
    if result.opening is None or result.grasp is None:
        logger.warning("托盘帧 {} 未得到完整开口或抓取结果", result.frame_id)
        return
    rpy = _rotation_matrix_to_rpy_deg(result.grasp.rotation)
    logger.info(
        "托盘帧 {} 开口中心 {:.1f}, {:.1f} px 抓取点 {:.1f}, {:.1f}, {:.1f} mm RPY {:.1f}, {:.1f}, {:.1f} deg",
        result.frame_id,
        float(result.opening.center_uv[0]),
        float(result.opening.center_uv[1]),
        float(result.grasp.grasp_point[0]),
        float(result.grasp.grasp_point[1]),
        float(result.grasp.grasp_point[2]),
        float(rpy[0]),
        float(rpy[1]),
        float(rpy[2]),
    )


def _open_wuji_camera_runtime(
    camera_name: WujiCameraName,
    remote_endpoint_stream_port: int,
) -> CameraRuntimeState:
    """严格按 `test/wuji/zmq_camera.py` 打开本地转发后的 ZMQ 相机链路。"""

    control_tunnel_process: Popen[bytes] | None = None
    stream_tunnel_process: Popen[bytes] | None = None
    camera_client: WujiZmqCameraClient | None = None
    try:
        control_tunnel_process = start_ssh_tunnel(
            DEFAULT_CONTROL_PORT,
            remote_host=WUYOU_HOST,
            ssh_alias=WUYOU_SSH_ALIAS,
        )
        stream_tunnel_process = start_ssh_tunnel(
            int(remote_endpoint_stream_port),
            remote_host=WUYOU_HOST,
            ssh_alias=WUYOU_SSH_ALIAS,
        )
        time.sleep(DEFAULT_FORWARD_WAIT_S)
        camera_client = WujiZmqCameraClient(
            host="127.0.0.1",
            control_port=DEFAULT_CONTROL_PORT - 1,
            request_timeout_ms=3000,
            stream_timeout_ms=8000,
            camera_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
        )
        setattr(camera_client, "_opening_local_control_tunnel", control_tunnel_process)
        setattr(camera_client, "_opening_local_stream_tunnel", stream_tunnel_process)
        intrinsics = camera_client.get_camera_intrinsics(camera_name)
        return CameraRuntimeState(
            client=camera_client,
            intrinsics=intrinsics,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        if camera_client is not None:
            camera_client.close()
        if control_tunnel_process is not None:
            stop_ssh_process(control_tunnel_process)
        if stream_tunnel_process is not None:
            stop_ssh_process(stream_tunnel_process)
        return CameraRuntimeState(
            client=None,
            intrinsics=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _close_camera_runtime(camera_state: CameraRuntimeState) -> None:
    """关闭按 `zmq_camera.py` 打开的相机资源。"""

    client = camera_state.client
    if client is None:
        return
    try:
        client.close()
    finally:
        control_tunnel_process = getattr(client, "_opening_local_control_tunnel", None)
        stream_tunnel_process = getattr(client, "_opening_local_stream_tunnel", None)
        if control_tunnel_process is not None:
            stop_ssh_process(control_tunnel_process)
        if stream_tunnel_process is not None:
            stop_ssh_process(stream_tunnel_process)


def _build_status_canvas(message: str) -> np.ndarray:
    """构造初始化等待或重连状态画面。"""

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        "opening_detection_local waiting for camera control/stream",
        (40, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message[:140],
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 180, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Press ESC or Q to quit. The script will keep retrying.",
        (40, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _load_opening_detection_runtime() -> OpeningDetectionRuntime:
    """按文件加载 opening_detection 子模块，避免触发 `src.pointcloud.__init__`。"""

    package_name = "_opening_detection_local_runtime"
    opening_dir = PROJECT_ROOT / "src" / "pointcloud" / "opening_detection"
    if not opening_dir.is_dir():
        raise RuntimeError(f"opening_detection 目录不存在：{opening_dir}")
    _ensure_runtime_package(package_name, opening_dir)
    _load_runtime_module(package_name, opening_dir, "types")
    opening_pipeline_module = _load_runtime_module(package_name, opening_dir, "opening_pipeline")
    pose_pipeline_module = _load_runtime_module(package_name, opening_dir, "pose_pipeline")
    return OpeningDetectionRuntime(
        opening_pipeline_cls=opening_pipeline_module.OpeningDetectionPipeline,
        pose_estimator_cls=pose_pipeline_module.GraspPoseEstimator,
        temporal_state_cls=pose_pipeline_module.TemporalFilterState,
    )


def _ensure_runtime_package(package_name: str, package_dir: Path) -> None:
    """创建仅供测试脚本使用的临时包命名空间。"""

    if package_name in sys.modules:
        return
    package_module = ModuleType(package_name)
    package_module.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package_module


def _load_runtime_module(package_name: str, package_dir: Path, module_name: str) -> Any:
    """从 `opening_detection` 目录按文件加载单模块。"""

    full_name = f"{package_name}.{module_name}"
    cached_module = sys.modules.get(full_name)
    if cached_module is not None:
        return cached_module
    module_path = package_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法创建模块 spec：{module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


# endregion


# region CLI


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="opening_detection_local smoke test")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--target-tray-index", type=int, default=DEFAULT_TARGET_TRAY_INDEX)
    parser.add_argument("--compute-interval-ms", type=int, default=DEFAULT_COMPUTE_INTERVAL_MS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = _parse_cli(sys.argv[1:])
        raise SystemExit(
            main(
                service_addr=str(args.service_addr),
                camera_name=args.camera_name,
                target_tray_index=int(args.target_tray_index),
                compute_interval_ms=int(args.compute_interval_ms),
            )
        )
    raise SystemExit(main())


# endregion
