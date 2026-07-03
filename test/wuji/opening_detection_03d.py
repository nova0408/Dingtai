from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from loguru import logger

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient  # noqa: E402
from camera_pipeline.opening_detection.protocol import OpeningDetectionPipelineRequest  # noqa: E402
from camera_pipeline.tray_detection.protocol import OrinTrayDetectionRequest  # noqa: E402

# region 默认参数
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6200"
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_WAIT_AFTER_DETECTED_S = 20.0
DEFAULT_MAX_FRAMES = 240
DEFAULT_POINT_SIZE = 1.5
DEFAULT_VIS_WINDOW_WIDTH = 1440
DEFAULT_VIS_WINDOW_HEIGHT = 900
# endregion


# region 数据结构
@dataclass(frozen=True)
class OpeningDetectionSnapshot:
    frame_id: int
    camera_name: str
    camera_status: dict[str, Any]
    camera_intrinsics: dict[str, Any]
    opening_detection: dict[str, Any]
    raw_pose: dict[str, Any] | None
    pose_transform_camera_frame: list[list[float]] | None
    error: str | None
    depth_shape: tuple[int, int] | None


# endregion


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    wait_after_detected_s: float = DEFAULT_WAIT_AFTER_DETECTED_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> int:
    logger.info("opening_detection_03d 启动：service_addr={} camera_name={}", service_addr, camera_name)

    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=int(float(timeout_s) * 1000.0))
    vis, stop_flag = _init_visualizer()
    cloud = o3d.geometry.PointCloud()
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
    opening_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=80.0, origin=[0.0, 0.0, 0.0])
    vis.add_geometry(cloud)
    vis.add_geometry(camera_frame)
    vis.add_geometry(opening_frame)
    cv2.namedWindow("opening_detection_03d_depth", cv2.WINDOW_NORMAL)

    snapshot: OpeningDetectionSnapshot | None = None
    try:
        status = client.get_camera_status(timeout_s=float(timeout_s))
        intrinsics = client.get_camera_intrinsics(timeout_s=float(timeout_s))
        logger.info(
            "相机状态 online={} color_enabled={} depth_enabled={} model={} size={}x{}",
            status.online,
            status.color_enabled,
            status.depth_enabled,
            status.camera_model,
            status.width,
            status.height,
        )
        logger.info(
            "相机内参 fx={:.3f} fy={:.3f} cx={:.3f} cy={:.3f}",
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )

        stream = client.subscribe_camera_frames(camera_name=camera_name)
        for frame_idx, frame in enumerate(stream, start=1):
            if stop_flag["flag"]:
                break
            depth_mm = np.asarray(frame.depth_mm, dtype=np.float64)
            valid_mask = np.isfinite(depth_mm) & (depth_mm > 1.0)
            valid_count = int(np.count_nonzero(valid_mask))
            if valid_count == 0:
                logger.warning(
                    "frame {} 深度有效点为 0，shape={} min={} max={} ，先继续等待",
                    int(frame.frame_id),
                    tuple(int(v) for v in depth_mm.shape[:2]),
                    float(np.nanmin(depth_mm)) if depth_mm.size > 0 else float("nan"),
                    float(np.nanmax(depth_mm)) if depth_mm.size > 0 else float("nan"),
                )
                if frame_idx >= int(max_frames):
                    logger.warning("达到最大帧数 {} 仍未拿到有效深度", int(max_frames))
                    break
                continue

            _update_cloud(cloud, depth_mm, frame.color_bgr, frame.fx, frame.fy, frame.cx, frame.cy)
            cv2.imshow("opening_detection_03d_depth", _build_depth_view(np.asarray(frame.depth_mm, dtype=np.uint16)))
            cv2.waitKey(1)
            vis.update_geometry(cloud)
            _poll_viewer(vis)

            tray_response = client.request_tray_detection(
                OrinTrayDetectionRequest(
                    request_id=frame_idx,
                    camera_name=str(camera_name),
                    frame_id=int(frame.frame_id),
                    enable_debug=True,
                )
            )
            if tray_response.error is not None or tray_response.tray_count <= 0:
                logger.warning("frame {} tray 检测失败：{}", frame.frame_id, tray_response.error)
                if frame_idx >= int(max_frames):
                    logger.warning("达到最大帧数 {} 仍未稳定识别到托盘", int(max_frames))
                    break
                continue

            response = None
            for tray_index in range(int(tray_response.tray_count)):
                response = client.request_opening_detection(
                    OpeningDetectionPipelineRequest(
                        request_id=frame_idx,
                        camera_name=str(camera_name),
                        frame_id=int(tray_response.frame_id),
                        target_tray_index=int(tray_index),
                        enable_debug=True,
                    )
                )
                if response.error is None and response.selected_result is not None and response.selected_result.pose is not None:
                    break
            if response is None or response.error is not None or response.selected_result is None or response.selected_result.pose is None:
                logger.warning("frame {} opening 检测失败：{}", frame.frame_id, None if response is None else response.error)
                if frame_idx >= int(max_frames):
                    logger.warning("达到最大帧数 {} 仍未稳定识别到开口", int(max_frames))
                    break
                continue

            pose = response.selected_result.pose
            transform = _pose_to_transform(pose.rotation, pose.grasp_point_mm)
            _update_pose_frame(opening_frame, transform)
            vis.update_geometry(opening_frame)
            _poll_viewer(vis)

            snapshot = OpeningDetectionSnapshot(
                frame_id=int(response.frame_id),
                camera_name=str(response.camera_name),
                camera_status={
                    "camera_name": status.camera_name,
                    "camera_id": status.camera_id,
                    "camera_model": status.camera_model,
                    "width": int(status.width),
                    "height": int(status.height),
                    "color_enabled": bool(status.color_enabled),
                    "depth_enabled": bool(status.depth_enabled),
                    "online": bool(status.online),
                    "source_meta": dict(status.source_meta),
                },
                camera_intrinsics={
                    "camera_name": intrinsics.camera_name,
                    "fx": float(intrinsics.fx),
                    "fy": float(intrinsics.fy),
                    "cx": float(intrinsics.cx),
                    "cy": float(intrinsics.cy),
                    "distortion": list(intrinsics.distortion),
                    "width": int(intrinsics.width),
                    "height": int(intrinsics.height),
                },
                opening_detection=_serialize_response(response),
                raw_pose=None if response.selected_result is None or response.selected_result.pose is None else {
                    "grasp_point_mm": [float(v) for v in response.selected_result.pose.grasp_point_mm],
                    "pre_grasp_point_mm": [float(v) for v in response.selected_result.pose.pre_grasp_point_mm],
                    "rotation": None
                    if response.selected_result.pose.rotation is None
                    else [[float(v) for v in row] for row in response.selected_result.pose.rotation],
                    "rpy_deg": None
                    if response.selected_result.pose.rpy_deg is None
                    else [float(v) for v in response.selected_result.pose.rpy_deg],
                },
                pose_transform_camera_frame=None if transform is None else np.asarray(transform, dtype=np.float64).tolist(),
                error=None,
                depth_shape=(int(frame.depth_mm.shape[0]), int(frame.depth_mm.shape[1])),
            )

            _show_snapshot(snapshot)
            logger.success("frame {} 已识别开口并显示结果，等待 {:.1f} s 后关闭窗口", frame.frame_id, float(wait_after_detected_s))
            deadline = time.perf_counter() + float(wait_after_detected_s)
            while time.perf_counter() < deadline:
                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.03)
            break
    finally:
        vis.destroy_window()
        _safe_destroy_cv_window("opening_detection_03d_depth")
        client.close()

    return 0


def _update_cloud(
    cloud: o3d.geometry.PointCloud,
    depth_mm: np.ndarray,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    xyz, rgb = _rgbd_to_points(
        np.asarray(depth_mm, dtype=np.float64),
        np.asarray(color_bgr, dtype=np.uint8),
        fx,
        fy,
        cx,
        cy,
    )
    if xyz.shape[0] == 0:
        return
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(rgb, dtype=np.float64))


def _update_pose_frame(frame_mesh: o3d.geometry.TriangleMesh, transform: np.ndarray | None) -> None:
    if transform is None:
        return
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=80.0, origin=[0.0, 0.0, 0.0])
    mesh.transform(np.asarray(transform, dtype=np.float64))
    frame_mesh.vertices = mesh.vertices
    frame_mesh.triangles = mesh.triangles
    frame_mesh.vertex_colors = mesh.vertex_colors
    frame_mesh.vertex_normals = mesh.vertex_normals


def _pose_to_transform(rotation: Any, translation_mm: Any) -> np.ndarray | None:
    if rotation is None or translation_mm is None:
        return None
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_mm, dtype=np.float64).reshape(3)
    return transform


def _rgbd_to_points(
    depth_mm: np.ndarray,
    color_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_mm.shape[:2]
    vv, uu = np.indices((height, width))
    z = np.asarray(depth_mm, dtype=np.float64)
    valid = np.isfinite(z) & (z > 1.0)
    x = (uu - float(cx)) * z / max(1e-9, float(fx))
    y = (vv - float(cy)) * z / max(1e-9, float(fy))
    pts = np.stack([x[valid], y[valid], z[valid]], axis=1)
    color_rgb = np.asarray(color_bgr, dtype=np.float64) / 255.0
    colors = np.stack([color_rgb[..., 2][valid], color_rgb[..., 1][valid], color_rgb[..., 0][valid]], axis=1)
    return pts, colors


def _serialize_response(response: Any) -> dict[str, Any]:
    selected = None
    if response.selected_result is not None:
        selected = {
            "tray_index": int(response.selected_result.tray_index),
            "tray_bbox_xywh": list(response.selected_result.tray_bbox_xywh),
            "tray_center_uv": list(response.selected_result.tray_center_uv),
            "opening_center_uv": None if response.selected_result.opening_center_uv is None else list(response.selected_result.opening_center_uv),
            "opening_quad_uv": None
            if response.selected_result.opening_quad_uv is None
            else [list(item) for item in response.selected_result.opening_quad_uv],
            "top_quad_uv": None if response.selected_result.top_quad_uv is None else [list(item) for item in response.selected_result.top_quad_uv],
            "pose": None
            if response.selected_result.pose is None
            else {
                "grasp_point_mm": [float(v) for v in response.selected_result.pose.grasp_point_mm],
                "pre_grasp_point_mm": [float(v) for v in response.selected_result.pose.pre_grasp_point_mm],
                "rotation": None
                if response.selected_result.pose.rotation is None
                else [[float(v) for v in row] for row in response.selected_result.pose.rotation],
                "rpy_deg": None if response.selected_result.pose.rpy_deg is None else [float(v) for v in response.selected_result.pose.rpy_deg],
            },
        }
    return {
        "frame_id": int(response.frame_id),
        "camera_name": str(response.camera_name),
        "timestamp_ms": float(response.timestamp_ms),
        "elapsed_ms": float(response.elapsed_ms),
        "tray_count": int(response.tray_count),
        "selected_tray_index": int(response.selected_tray_index),
        "selected_result": selected,
        "error": response.error,
    }


def _show_snapshot(snapshot: OpeningDetectionSnapshot) -> None:
    logger.success("最终结果：{}", snapshot)


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


def _init_visualizer() -> tuple[o3d.visualization.VisualizerWithKeyCallback, dict[str, bool]]:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("opening_detection_03d", width=DEFAULT_VIS_WINDOW_WIDTH, height=DEFAULT_VIS_WINDOW_HEIGHT)
    opt = vis.get_render_option()
    if opt is not None:
        opt.point_size = float(DEFAULT_POINT_SIZE)
        opt.background_color = np.asarray([0.02, 0.02, 0.02], dtype=np.float64)

    stop_flag = {"flag": False}

    def _on_escape(_vis):  # noqa: ANN001
        stop_flag["flag"] = True
        return False

    vis.register_key_callback(256, _on_escape)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    vis.add_geometry(axis)
    view = vis.get_view_control()
    if view is not None:
        view.set_lookat([0.0, 0.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
    return vis, stop_flag


def _poll_viewer(vis: o3d.visualization.VisualizerWithKeyCallback) -> bool:
    alive = vis.poll_events()
    vis.update_renderer()
    return bool(alive)


def _safe_destroy_cv_window(window_name: str) -> None:
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
