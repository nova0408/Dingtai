from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import zmq
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orin.grasp_pose_pipeline.protocol import GraspPosePipelineRequest  # noqa: E402
from orin.grasp_pose_pipeline.transport import GraspPosePipelineRpcClient, ZmqSocketOptions  # noqa: E402


# region 默认参数
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.116:6220"  # Orin 抓取位姿主服务地址
DEFAULT_CAMERA_NAME = "left_hand_camera"  # 逻辑相机名
DEFAULT_TARGET_TRAY_INDEX = 0  # 目标托盘编号，默认展示最左托盘
DEFAULT_RPC_TIMEOUT_MS = 10_000  # 单次 RPC 超时，单位 ms
DEFAULT_COMPUTE_MIN_INTERVAL_S = 0.20  # 请求间隔，单位 s
DEFAULT_MAX_FRAMES = 0  # 最多验证帧数，0 表示持续运行直到用户退出
DEFAULT_MAX_PREVIEW_POINTS = 100_000  # 3D 预览点云最大点数
DEFAULT_WINDOW_WIDTH = 1440  # 3D 窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 3D 窗口高度，单位 像素
DEFAULT_POINT_SIZE = 1.5  # 点云点大小
DEFAULT_MIN_2D_WINDOW_LONG_SIDE = 800  # 2D 窗口最小长边，单位 像素
DEFAULT_2D_WINDOW_NAME = "Orin tray0 pipeline merged"  # 2D 预览窗口名
DEFAULT_3D_WINDOW_NAME = "Orin tray0 pipeline 3D"  # 3D 预览窗口名
# endregion


# region 数据结构
@dataclass(frozen=True)
class ProjectionIntrinsics:
    """图像投影内参。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class PipelinePoller:
    """后台异步拉取 grasp_pose_pipeline 结果，避免窗口线程被 RPC 阻塞。"""

    def __init__(
        self,
        service_addr: str,
        camera_name: str,
        target_tray_index: int,
        rpc_timeout_ms: int,
        compute_min_interval_s: float,
    ) -> None:
        self._service_addr = str(service_addr)
        self._camera_name = str(camera_name)
        self._target_tray_index = int(target_tray_index)
        self._rpc_timeout_ms = int(rpc_timeout_ms)
        self._compute_min_interval_s = float(compute_min_interval_s)
        self._lock = threading.Lock()
        self._latest_response = None
        self._latest_error: Optional[str] = None
        self._request_count = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="orin-grasp-pose-pipeline-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_snapshot(self) -> tuple[Any, Optional[str], int]:
        with self._lock:
            return self._latest_response, self._latest_error, int(self._request_count)

    def _run_loop(self) -> None:
        client = self._create_client()
        last_warn_error: Optional[str] = None
        try:
            while self._running:
                self._request_count += 1
                try:
                    response = client.call(
                        GraspPosePipelineRequest(
                            request_id=int(self._request_count),
                            camera_name=self._camera_name,
                            frame_id=-1,
                            target_tray_index=self._target_tray_index,
                            enable_debug=True,
                        )
                    )
                    with self._lock:
                        self._latest_response = response
                        self._latest_error = response.error
                    last_warn_error = None
                except zmq.error.Again:
                    with self._lock:
                        self._latest_error = "grasp_pose_pipeline rpc timeout"
                    if last_warn_error != "grasp_pose_pipeline rpc timeout":
                        logger.warning("grasp_pose_pipeline 请求超时，窗口保持可拖动并等待下次结果。")
                        last_warn_error = "grasp_pose_pipeline rpc timeout"
                    client.close()
                    client = self._create_client()
                except Exception as exc:  # noqa: BLE001
                    error_text = "{0}: {1}".format(type(exc).__name__, exc)
                    with self._lock:
                        self._latest_error = error_text
                    if last_warn_error != error_text:
                        logger.warning("grasp_pose_pipeline 请求失败：{}", error_text)
                        last_warn_error = error_text
                    client.close()
                    client = self._create_client()
                time.sleep(max(0.02, self._compute_min_interval_s))
        finally:
            client.close()

    def _create_client(self) -> GraspPosePipelineRpcClient:
        return GraspPosePipelineRpcClient(
            connect_addr=self._service_addr,
            options=ZmqSocketOptions(recv_timeout_ms=self._rpc_timeout_ms, send_timeout_ms=self._rpc_timeout_ms),
        )


# endregion


# region 主入口
def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    target_tray_index: int = DEFAULT_TARGET_TRAY_INDEX,
    rpc_timeout_ms: int = DEFAULT_RPC_TIMEOUT_MS,
    compute_min_interval_s: float = DEFAULT_COMPUTE_MIN_INTERVAL_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> None:
    """查看 Orin grasp_pose_pipeline 返回的托盘 0 最终抓取位姿与调试结果。"""

    logger.info("启动 Orin grasp_pose_pipeline 远端查看器")
    logger.warning("当前查看器只访问一个主服务端口，默认直接展示 tray 0 的最终抓取位姿。")
    poller = PipelinePoller(
        service_addr=str(service_addr),
        camera_name=str(camera_name),
        target_tray_index=int(target_tray_index),
        rpc_timeout_ms=int(rpc_timeout_ms),
        compute_min_interval_s=float(compute_min_interval_s),
    )
    poller.start()
    vis, stop_flag, raw_pcd, plane_pcd, frame_mesh, grasp_line = _init_3d_viewer()
    cv2.namedWindow(DEFAULT_2D_WINDOW_NAME, cv2.WINDOW_NORMAL)
    window_size_initialized = False
    shown_frames = 0
    fps_start = time.perf_counter()
    try:
        while True:
            if stop_flag["flag"]:
                break
            latest_response, latest_error, request_count = poller.get_snapshot()
            merged = _build_waiting_view(latest_error, request_count)
            if latest_response is not None and latest_response.debug is not None:
                if latest_response.debug.color_bgr is not None and latest_response.debug.depth_mm is not None:
                    color_bgr = np.asarray(latest_response.debug.color_bgr, dtype=np.uint8)
                    depth_mm = np.asarray(latest_response.debug.depth_mm, dtype=np.uint16)
                    intrinsics = _decode_intrinsics(latest_response, color_bgr, depth_mm)
                    rgb_view = _draw_remote_result(color_bgr, latest_response)
                    depth_view = _draw_remote_result(_build_hsv_depth_view(depth_mm), latest_response)
                    merged = np.hstack([rgb_view, depth_view])
                    xyz, rgb = _rgbd_to_points(color_bgr, depth_mm, intrinsics)
                    preview_xyz, preview_rgb = _downsample_xyzrgb(xyz, rgb, DEFAULT_MAX_PREVIEW_POINTS)
                    _update_raw_cloud(raw_pcd, preview_xyz, preview_rgb)
                    _update_remote_plane_cloud(plane_pcd, xyz, intrinsics, latest_response)
                    _update_remote_pose(frame_mesh, grasp_line, latest_response)
                    vis.update_geometry(raw_pcd)
                    vis.update_geometry(plane_pcd)
                    vis.update_geometry(frame_mesh)
                    vis.update_geometry(grasp_line)
                    shown_frames += 1
                    if shown_frames % 10 == 0:
                        now = time.perf_counter()
                        elapsed = max(1e-6, now - fps_start)
                        logger.info(
                            "显示帧数 {} 帧 frame_id {} elapsed {:.1f} ms 显示帧率 {:.1f} fps",
                            shown_frames,
                            latest_response.frame_id,
                            latest_response.elapsed_ms,
                            shown_frames / elapsed,
                        )
            if not window_size_initialized:
                image_h, image_w = merged.shape[:2]
                win_w, win_h = _compute_preview_window_size(image_w, image_h, DEFAULT_MIN_2D_WINDOW_LONG_SIDE)
                cv2.resizeWindow(DEFAULT_2D_WINDOW_NAME, win_w, win_h)
                window_size_initialized = True
            cv2.imshow(DEFAULT_2D_WINDOW_NAME, merged)
            if _should_exit(vis):
                break
            if int(max_frames) > 0 and shown_frames >= int(max_frames):
                logger.success("达到最大显示帧数 {} 帧，测试结束。", max_frames)
                break
        poller.stop()
    finally:
        poller.stop()
        vis.destroy_window()
        _safe_destroy_cv_window(DEFAULT_2D_WINDOW_NAME)


def _parse_cli(argv: list[str]) -> tuple[str, str, int, int, float, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="查看 Orin grasp_pose_pipeline 返回结果")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--target-tray-index", type=int, default=DEFAULT_TARGET_TRAY_INDEX)
    parser.add_argument("--rpc-timeout-ms", type=int, default=DEFAULT_RPC_TIMEOUT_MS)
    parser.add_argument("--compute-min-interval-s", type=float, default=DEFAULT_COMPUTE_MIN_INTERVAL_S)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    args = parser.parse_args(argv)
    return str(args.service_addr), str(args.camera_name), int(args.target_tray_index), int(args.rpc_timeout_ms), float(args.compute_min_interval_s), int(args.max_frames)


# endregion


# region 2D 预览
def _build_waiting_view(latest_error: Optional[str], request_count: int) -> np.ndarray:
    """构造等待远端结果时的占位画面。"""

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    message = "waiting grasp_pose_pipeline result..."
    cv2.putText(canvas, message, (48, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, "target tray index 0", (48, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "request count {0}".format(int(request_count)), (48, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (180, 180, 180), 1, cv2.LINE_AA)
    if latest_error:
        cv2.putText(canvas, latest_error[:80], (48, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 120, 255), 1, cv2.LINE_AA)
    return np.hstack([canvas, canvas.copy()])


def _draw_remote_result(base_bgr: np.ndarray, response) -> np.ndarray:
    """在远端底图上叠加阶段 1、阶段 2 与最终位姿结果。"""

    out = np.asarray(base_bgr, dtype=np.uint8).copy()
    debug = response.debug
    if debug is not None:
        for tray_id, tray_mask in enumerate(debug.tray_instance_masks):
            _draw_mask_outline(out, tray_mask, _tray_color_bgr(tray_id))
    for tray in response.tray_results:
        x, y, w, h = tray.bbox_xywh
        color = (0, 255, 255) if int(tray.tray_id) == int(response.selected_tray_index) else _tray_color_bgr(tray.tray_id)
        cv2.rectangle(out, (int(x), int(y)), (int(x + w - 1), int(y + h - 1)), color, 2 if int(tray.tray_id) == int(response.selected_tray_index) else 1, cv2.LINE_AA)
        cv2.putText(out, f"tray_{tray.tray_id}", (int(x), max(16, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    selected = response.selected_result
    if selected is not None and debug is not None:
        if debug.selected_tray_mask is not None:
            _blend_mask(out, debug.selected_tray_mask, (0, 180, 180), 0.18)
        if debug.near_plane_mask is not None:
            _blend_mask(out, debug.near_plane_mask, (0, 0, 255), 0.34)
            _draw_mask_outline(out, debug.near_plane_mask, (0, 0, 255))
        if debug.no_hole_mask is not None:
            _blend_mask(out, debug.no_hole_mask, (255, 0, 0), 0.30)
            _draw_mask_outline(out, debug.no_hole_mask, (255, 0, 0))
        if selected.opening_quad_uv is not None:
            quad = np.round(np.asarray(selected.opening_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(out, [quad], True, (0, 255, 0), 1, cv2.LINE_AA)
        if selected.top_quad_uv is not None:
            top_quad = np.round(np.asarray(selected.top_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(out, [top_quad], True, (255, 0, 0), 1, cv2.LINE_AA)
        if selected.opening_center_uv is not None:
            u = int(round(float(selected.opening_center_uv[0])))
            v = int(round(float(selected.opening_center_uv[1])))
            cv2.circle(out, (u, v), 4, (0, 255, 0), -1)
        if selected.pose is not None:
            pose_text = "grasp ({0:.1f}, {1:.1f}, {2:.1f}) mm".format(
                float(selected.pose.grasp_point_mm[0]),
                float(selected.pose.grasp_point_mm[1]),
                float(selected.pose.grasp_point_mm[2]),
            )
            cv2.putText(out, pose_text, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"frame_id {response.frame_id} elapsed {response.elapsed_ms:.1f} ms", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _build_hsv_depth_view(depth_mm: np.ndarray) -> np.ndarray:
    """把远端深度图转换为 HSV 着色预览。"""

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
    """绘制二值掩码外轮廓。"""

    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _compute_preview_window_size(src_w: int, src_h: int, min_long_side: int) -> tuple[int, int]:
    """按最小长边约束计算 2D 窗口尺寸。"""

    long_side = max(1, src_w, src_h)
    if long_side >= min_long_side:
        return max(1, src_w), max(1, src_h)
    scale = float(min_long_side) / float(long_side)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


# endregion


# region 3D 预览
def _init_3d_viewer():
    """初始化 Open3D 预览窗口。"""

    vis = o3d.visualization.VisualizerWithKeyCallback()  # pyright: ignore[reportAttributeAccessIssue]
    vis.create_window(DEFAULT_3D_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = DEFAULT_POINT_SIZE
        render_option.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)
    stop_flag = {"flag": False}
    vis.register_key_callback(256, lambda _vis: stop_flag.__setitem__("flag", True) or False)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    raw_pcd = o3d.geometry.PointCloud()
    plane_pcd = o3d.geometry.PointCloud()
    frame_mesh = _empty_mesh()
    grasp_line = o3d.geometry.LineSet()
    grasp_line.points = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0]], dtype=np.float64))
    grasp_line.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    grasp_line.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.8, 1.0]], dtype=np.float64))
    for geometry in (axis, raw_pcd, plane_pcd, frame_mesh, grasp_line):
        vis.add_geometry(geometry)
    return vis, stop_flag, raw_pcd, plane_pcd, frame_mesh, grasp_line


def _decode_intrinsics(response, color_bgr: np.ndarray, depth_mm: np.ndarray) -> ProjectionIntrinsics:
    """从响应中恢复远端内参。"""

    raw_intrinsics = None if response.debug is None else response.debug.camera_intrinsics
    height, width = color_bgr.shape[:2]
    if (height, width) != depth_mm.shape[:2]:
        raise RuntimeError("远端 color/depth 尺寸不一致")
    if raw_intrinsics is None:
        raise RuntimeError("远端响应缺少相机内参")
    return ProjectionIntrinsics(
        width=int(width),
        height=int(height),
        fx=float(raw_intrinsics[0]),
        fy=float(raw_intrinsics[1]),
        cx=float(raw_intrinsics[2]),
        cy=float(raw_intrinsics[3]),
    )


def _rgbd_to_points(color_bgr: np.ndarray, depth_mm: np.ndarray, intrinsics: ProjectionIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """把远端 RGBD 转换为本地点云。"""

    depth = np.asarray(depth_mm, dtype=np.float32)
    vv, uu = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    z = depth[valid]
    x = (uu[valid] - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (vv[valid] - float(intrinsics.cy)) * z / float(intrinsics.fy)
    xyz = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    rgb = np.asarray(color_bgr[valid][:, ::-1], dtype=np.float32) / 255.0
    return xyz, np.clip(rgb, 0.0, 1.0).astype(np.float64, copy=False)


def _downsample_xyzrgb(xyz: np.ndarray, rgb: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """限制 3D 预览点数。"""

    if xyz.shape[0] <= int(max_points):
        return xyz, rgb
    step = max(1, int(np.ceil(xyz.shape[0] / float(max_points))))
    return xyz[::step], rgb[::step]


def _update_raw_cloud(pcd: o3d.geometry.PointCloud, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """更新背景点云。"""

    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(0.35 * rgb + 0.15, 0.0, 1.0))


def _update_remote_plane_cloud(plane_pcd: o3d.geometry.PointCloud, xyz: np.ndarray, intrinsics: ProjectionIntrinsics, response) -> None:
    """更新近邻面与顶面着色点云。"""

    if response is None or response.debug is None:
        plane_pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        plane_pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return
    near_mask = response.debug.near_plane_mask
    top_mask = response.debug.no_hole_mask
    if near_mask is None and top_mask is None:
        plane_pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        plane_pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return
    uv, valid = _project_points_to_image(xyz, intrinsics)
    idx = np.where(valid)[0]
    if idx.size == 0:
        plane_pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        plane_pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return
    u = uv[idx, 0]
    v = uv[idx, 1]
    near_sel = np.zeros((idx.size,), dtype=bool) if near_mask is None else (np.asarray(near_mask, dtype=np.uint8)[v, u] > 0)
    top_sel = np.zeros((idx.size,), dtype=bool) if top_mask is None else (np.asarray(top_mask, dtype=np.uint8)[v, u] > 0)
    show = near_sel | top_sel
    chosen = idx[show]
    colors = np.zeros((chosen.size, 3), dtype=np.float64)
    colors[near_sel[show]] = np.asarray([1.0, 0.15, 0.15], dtype=np.float64)
    colors[top_sel[show] & (~near_sel[show])] = np.asarray([0.15, 0.35, 1.0], dtype=np.float64)
    plane_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz[chosen], dtype=np.float64))
    plane_pcd.colors = o3d.utility.Vector3dVector(colors)


def _project_points_to_image(xyz: np.ndarray, intrinsics: ProjectionIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """把点云投影到图像平面。"""

    z = xyz[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    u = np.full((xyz.shape[0],), -1, dtype=np.int32)
    v = np.full((xyz.shape[0],), -1, dtype=np.int32)
    if np.any(valid):
        x = xyz[valid, 0]
        y = xyz[valid, 1]
        zz = z[valid]
        uu = np.rint(float(intrinsics.fx) * x / zz + float(intrinsics.cx)).astype(np.int32)
        vv = np.rint(float(intrinsics.fy) * y / zz + float(intrinsics.cy)).astype(np.int32)
        in_bounds = (uu >= 0) & (uu < intrinsics.width) & (vv >= 0) & (vv < intrinsics.height)
        idx = np.where(valid)[0][in_bounds]
        u[idx] = uu[in_bounds]
        v[idx] = vv[in_bounds]
    return np.stack([u, v], axis=1), (u >= 0) & (v >= 0)


def _update_remote_pose(frame_mesh: o3d.geometry.TriangleMesh, grasp_line: o3d.geometry.LineSet, response) -> None:
    """更新最终抓取位姿坐标系与预抓取连线。"""

    pose = None if response is None or response.selected_result is None else response.selected_result.pose
    if pose is None:
        return
    rotation = np.asarray(pose.rotation, dtype=np.float64)
    grasp_point = np.asarray(pose.grasp_point_mm, dtype=np.float64)
    pre_grasp_point = np.asarray(pose.pre_grasp_point_mm, dtype=np.float64)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40.0, origin=[0.0, 0.0, 0.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = grasp_point
    frame.transform(transform)
    frame_mesh.vertices = frame.vertices
    frame_mesh.triangles = frame.triangles
    frame_mesh.vertex_colors = frame.vertex_colors
    frame_mesh.vertex_normals = frame.vertex_normals
    grasp_line.points = o3d.utility.Vector3dVector(np.vstack([grasp_point, pre_grasp_point]).astype(np.float64))
    grasp_line.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
    grasp_line.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.8, 1.0]], dtype=np.float64))


def _empty_mesh() -> o3d.geometry.TriangleMesh:
    """构造离屏占位 mesh。"""

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0], [0.0, 1.0, -10000.0]], dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32))
    return mesh


# endregion


# region 通用工具
def _tray_color_bgr(index: int) -> tuple[int, int, int]:
    """为托盘编号分配固定颜色。"""

    palette = (
        (0, 220, 255),
        (80, 200, 120),
        (255, 170, 0),
        (255, 110, 180),
        (140, 180, 255),
        (200, 120, 255),
    )
    return palette[int(index) % len(palette)]


def _poll_viewer(vis: Any) -> bool:
    """刷新 Open3D 事件循环。"""

    alive = vis.poll_events()
    vis.update_renderer()
    return bool(alive)


def _safe_destroy_cv_window(window_name: str) -> None:
    """安全销毁 cv2 窗口。"""

    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def _cv_window_closed(window_name: str) -> bool:
    """判断 cv2 窗口是否已关闭。"""

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def _should_exit(vis: Any) -> bool:
    """判断 2D/3D 预览是否需要退出。"""

    key = cv2.waitKey(1)
    if key == 27 or key == ord("q") or key == ord("Q"):
        return True
    if (not _poll_viewer(vis)) or _cv_window_closed(DEFAULT_2D_WINDOW_NAME):
        return True
    return False


# endregion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        service_addr_arg, camera_name_arg, target_tray_index_arg, rpc_timeout_ms_arg, compute_min_interval_s_arg, max_frames_arg = _parse_cli(sys.argv[1:])
        main(service_addr_arg, camera_name_arg, target_tray_index_arg, rpc_timeout_ms_arg, compute_min_interval_s_arg, max_frames_arg)
    else:
        main()
