from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.test_gui.test_wuji_camera_tab import ImagePreviewLabel
from gui.wuji.pose_context import WujiPoseExecutionContext, WujiPoseExecutionResult
from src.wuji import load_wuji_robot_network_config
from src.wuji.camera_protocol import WujiCameraFrame, WujiCameraIntrinsicsInfo

POSE_DEPENDENCY_ERROR: str | None = None

try:
    import open3d as o3d
except Exception as exc:  # noqa: BLE001
    o3d = None
    POSE_DEPENDENCY_ERROR = f"open3d unavailable: {type(exc).__name__}: {exc}"

try:
    from gui.util_components.o3d_viewer_window import O3DViewerWindow
except Exception as exc:  # noqa: BLE001
    O3DViewerWindow = None
    if POSE_DEPENDENCY_ERROR is None:
        POSE_DEPENDENCY_ERROR = f"o3d viewer unavailable: {type(exc).__name__}: {exc}"

try:
    from orin.grasp_pose.protocol import GraspPoseResponse
except Exception as exc:  # noqa: BLE001
    GraspPoseResponse = object
    if POSE_DEPENDENCY_ERROR is None:
        POSE_DEPENDENCY_ERROR = f"orin grasp rpc unavailable: {type(exc).__name__}: {exc}"

LEFT_CAMERA_NAME = "left_hand_camera"
DEFAULT_ORIN_SERVICE_ADDR = "tcp://{0}:6200".format(load_wuji_robot_network_config().orin_ip)
DEFAULT_MAX_PREVIEW_POINTS = 100_000
DEFAULT_POINT_SIZE = 1.5


class _RpcResultPacket:
    def __init__(
        self,
        frame_idx: int,
        frame: WujiCameraFrame,
        intrinsics: WujiCameraIntrinsicsInfo,
        target_tray_index: int,
        response: Any | None,
        error: str | None,
    ) -> None:
        self.frame_idx = int(frame_idx)
        self.frame = frame
        self.intrinsics = intrinsics
        self.target_tray_index = int(target_tray_index)
        self.response = response
        self.error = error


class WujiPoseTabWidget(QWidget):
    def __init__(self, parent: QWidget | None = None, service_addr: str = DEFAULT_ORIN_SERVICE_ADDR) -> None:
        super().__init__(parent)
        self._service_addr = str(service_addr)
        self._context = WujiPoseExecutionContext(service_addr=self._service_addr)
        self._result_timer = QTimer(self)
        self._result_timer.setInterval(30)
        self._result_timer.timeout.connect(self._poll_runtime_result)
        self._frame_index = 0
        self._active = False
        self._intrinsics: WujiCameraIntrinsicsInfo | None = None
        self._latest_result: _RpcResultPacket | None = None
        self._latest_rendered_frame_idx: int | None = None

        self.status_label = QLabel("pose tab 未激活", self)
        self.service_label = QLabel("Orin RPC: {0}".format(self._service_addr), self)
        self.runtime_button = QPushButton("启动测试", self)
        self.runtime_button.setCheckable(True)
        self.runtime_button.setChecked(False)
        self.runtime_button.toggled.connect(self._on_runtime_toggled)
        self.tray_index_spinbox = QSpinBox(self)
        self.tray_index_spinbox.setRange(0, 31)
        self.tray_index_spinbox.setValue(0)
        self.interval_spinbox = QSpinBox(self)
        self.interval_spinbox.setRange(100, 5000)
        self.interval_spinbox.setSingleStep(100)
        self.interval_spinbox.setValue(1000)
        self.tray_count_label = QLabel("托盘数: -", self)
        self.selected_index_label = QLabel("选中托盘: -", self)
        self.pending_frame_label = QLabel("待处理帧: -", self)
        self.completed_frame_label = QLabel("完成帧: -", self)
        self.submitted_label = QLabel("提交数: -", self)
        self.completed_count_label = QLabel("完成数: -", self)
        self.dropped_label = QLabel("丢弃数: -", self)
        self.elapsed_label = QLabel("耗时: -", self)
        self.error_label = QLabel("错误: -", self)
        self.rgb_preview = ImagePreviewLabel("RGB pose result", self)
        self.rgbd_preview = ImagePreviewLabel("RGBD pose result", self)
        self.result_tabs = QTabWidget(self)
        self._o3d_error_label = QLabel(self)
        self._o3d_error_label.setWordWrap(True)
        self._o3d_error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.o3d_window = None if O3DViewerWindow is None else O3DViewerWindow(title="pose o3d", size=(960, 720))
        self._raw_cloud = None if o3d is None else o3d.geometry.PointCloud()
        self._pose_line = None if o3d is None else o3d.geometry.LineSet()
        self._pose_axis_name = "pose_axis_frame"
        if self._pose_line is not None:
            assert o3d is not None
            self._pose_line.points = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0]], dtype=np.float64))
            self._pose_line.lines = o3d.utility.Vector2iVector(np.asarray([[0, 1]], dtype=np.int32))
            self._pose_line.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 0.8, 1.0]], dtype=np.float64))

        self._build_layout()
        if self._o3d_ready():
            self._init_o3d_scene()
        elif POSE_DEPENDENCY_ERROR is not None:
            self.status_label.setText("pose 依赖缺失: {0}".format(POSE_DEPENDENCY_ERROR))
        self._result_timer.start()
        self._runtime_enabled = False
        self._o3d_view_fitted = False

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._context.stop_continuous()
        super().closeEvent(event)

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if POSE_DEPENDENCY_ERROR is not None:
            self.status_label.setText("pose 依赖缺失: {0}".format(POSE_DEPENDENCY_ERROR))
            return
        if self._active:
            self.status_label.setText("pose tab 已激活，点击“启动测试”后将抓取 LEFT 相机数据并请求 Orin 结果")
        else:
            self.status_label.setText("pose tab 未激活")

    def set_runtime_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._runtime_enabled:
            return
        self._runtime_enabled = enabled
        if enabled:
            self._context.start_continuous(
                target_tray_index=int(self.tray_index_spinbox.value()),
                enable_debug=True,
                compute_interval_ms=int(self.interval_spinbox.value()),
            )
            self._latest_rendered_frame_idx = None
            self._o3d_view_fitted = False
            self.runtime_button.setText("停止测试")
            self.runtime_button.blockSignals(True)
            self.runtime_button.setChecked(True)
            self.runtime_button.blockSignals(False)
            self.status_label.setText("pose rpc 测试已启动")
        else:
            self._context.stop_continuous()
            self.runtime_button.setText("启动测试")
            self.runtime_button.blockSignals(True)
            self.runtime_button.setChecked(False)
            self.runtime_button.blockSignals(False)
            self.status_label.setText("pose rpc 测试已停止")

    def update_intrinsics(self, intrinsics: WujiCameraIntrinsicsInfo) -> None:
        if intrinsics.camera_name != LEFT_CAMERA_NAME:
            return
        self._intrinsics = intrinsics
        if POSE_DEPENDENCY_ERROR is not None:
            return
        self.status_label.setText(
            "LEFT intrinsics: {0}x{1}, fx={2:.1f}, fy={3:.1f}".format(
                intrinsics.width,
                intrinsics.height,
                intrinsics.fx,
                intrinsics.fy,
            )
        )

    def update_frame(self, frame: WujiCameraFrame) -> None:
        _ = frame
        return

    def _build_layout(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left_placeholder = QWidget(self)
        left_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.status_label)

        form_panel = QWidget(self)
        form_layout = QFormLayout(form_panel)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.addRow("服务地址", self.service_label)
        form_layout.addRow("测试开关", self.runtime_button)
        form_layout.addRow("目标托盘编号", self.tray_index_spinbox)
        form_layout.addRow("提交间隔(ms)", self.interval_spinbox)
        form_layout.addRow("当前检测托盘数", self.tray_count_label)
        form_layout.addRow("选中托盘", self.selected_index_label)
        form_layout.addRow("待处理帧", self.pending_frame_label)
        form_layout.addRow("完成帧", self.completed_frame_label)
        form_layout.addRow("提交数", self.submitted_label)
        form_layout.addRow("完成数", self.completed_count_label)
        form_layout.addRow("丢弃数", self.dropped_label)
        form_layout.addRow("耗时", self.elapsed_label)
        form_layout.addRow("错误", self.error_label)
        right_layout.addWidget(form_panel)

        rgb_page = QWidget(self)
        rgb_layout = QVBoxLayout(rgb_page)
        rgb_layout.setContentsMargins(0, 0, 0, 0)
        rgb_layout.addWidget(self.rgb_preview)

        rgbd_page = QWidget(self)
        rgbd_layout = QVBoxLayout(rgbd_page)
        rgbd_layout.setContentsMargins(0, 0, 0, 0)
        rgbd_layout.addWidget(self.rgbd_preview)

        o3d_page = QWidget(self)
        o3d_layout = QVBoxLayout(o3d_page)
        o3d_layout.setContentsMargins(0, 0, 0, 0)
        if self.o3d_window is not None:
            o3d_layout.addWidget(self.o3d_window)
        else:
            self._o3d_error_label.setText(POSE_DEPENDENCY_ERROR or "o3d viewer unavailable")
            o3d_layout.addWidget(self._o3d_error_label)

        self.result_tabs.addTab(rgb_page, "rgb")
        self.result_tabs.addTab(rgbd_page, "rgbd")
        self.result_tabs.addTab(o3d_page, "o3d")
        right_layout.addWidget(self.result_tabs, 1)

        main_layout.addWidget(left_placeholder, 1)
        main_layout.addWidget(right_panel, 2)

    def _on_runtime_toggled(self, checked: bool) -> None:
        self.set_runtime_enabled(bool(checked))

    def _init_o3d_scene(self) -> None:
        if not self._o3d_ready():
            return
        assert self.o3d_window is not None
        assert self._raw_cloud is not None
        assert self._pose_line is not None
        assert o3d is not None
        viewer = self.o3d_window.viewer
        viewer.set_background_color(0.02, 0.02, 0.02)
        viewer.set_point_size(DEFAULT_POINT_SIZE)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0.0, 0.0, 0.0])
        viewer.add_helper_geometry("pose_world_axis", axis, helper_type="others")
        viewer.add_point_cloud("pose_raw_cloud", self._raw_cloud, reset_view=True)
        viewer.add_helper_geometry("pose_grasp_line", self._pose_line, helper_type="others")
        viewer.add_helper_geometry(self._pose_axis_name, o3d.geometry.TriangleMesh.create_coordinate_frame(size=80.0), helper_type="others")
        QTimer.singleShot(200, self._apply_reference_view)

    def _apply_reference_view(self) -> None:
        if not self._o3d_ready():
            return
        assert self.o3d_window is not None
        viewer = self.o3d_window.viewer
        if viewer.vis is None:
            QTimer.singleShot(200, self._apply_reference_view)
            return
        view = viewer.vis.get_view_control()
        if view is None:
            return
        view.set_lookat([0.0, 0.0, 0.0])
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
        viewer.vis.update_renderer()

    def _poll_runtime_result(self) -> None:
        if POSE_DEPENDENCY_ERROR is not None:
            return
        self._context.update_target(
            int(self.tray_index_spinbox.value()),
            enable_debug=True,
            compute_interval_ms=int(self.interval_spinbox.value()),
        )
        status = self._context.snapshot_status()
        self._pending_frame_label_update(status.pending_frame_id)
        self._completed_frame_label_update(status.completed_frame_id)
        self._submitted_label_update(status.submitted_count)
        self._completed_count_label_update(status.completed_count)
        self._dropped_label_update(status.dropped_count)
        if not status.busy and status.last_error:
            self._error_label_update(status.last_error)
        preview = self._context.get_latest_preview()
        if preview is not None and self._latest_rendered_frame_idx is None:
            self._render_preview_snapshot(preview.frame, preview.intrinsics, preview.frame_id)
        result = self._context.poll_result()
        if result is None:
            return
        result_packet = _convert_context_result(result)
        self._latest_result = result_packet
        if self._latest_rendered_frame_idx is not None and result_packet.frame_idx <= self._latest_rendered_frame_idx:
            return
        self._latest_rendered_frame_idx = int(result_packet.frame_idx)
        self._render_result(result_packet)

    def _render_preview_snapshot(
        self,
        frame: WujiCameraFrame,
        intrinsics: WujiCameraIntrinsicsInfo,
        frame_idx: int,
    ) -> None:
        color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
        depth_bgr = _depth_to_color(frame.depth) if frame.depth is not None else color_bgr
        self.rgb_preview.set_preview_pixmap(_bgr_to_pixmap(color_bgr))
        self.rgbd_preview.set_preview_pixmap(_bgr_to_pixmap(np.hstack([color_bgr, depth_bgr])))
        if self._o3d_ready():
            preview_packet = _RpcResultPacket(
                frame_idx=int(frame_idx),
                frame=frame,
                intrinsics=intrinsics,
                target_tray_index=int(self.tray_index_spinbox.value()),
                response=None,
                error=None,
            )
            self._update_o3d_result(preview_packet)

    def _render_result(self, result_packet: _RpcResultPacket) -> None:
        frame = result_packet.frame
        response = result_packet.response
        overlay_bgr = np.asarray(frame.color_bgr, dtype=np.uint8).copy()
        contrast_bgr = _depth_to_color(frame.depth) if frame.depth is not None else overlay_bgr
        if response is not None and response.debug is not None:
            if response.debug.overlay_bgr is not None:
                overlay_bgr = np.asarray(response.debug.overlay_bgr, dtype=np.uint8)
            if response.debug.contrast_bgr is not None:
                contrast_bgr = np.asarray(response.debug.contrast_bgr, dtype=np.uint8)
            elif response.debug.selected_tray_mask is not None:
                contrast_bgr = _mask_to_bgr(response.debug.selected_tray_mask)
        overlay_bgr = _compose_debug_overlay(overlay_bgr, response)
        contrast_bgr = _compose_contrast_overlay(contrast_bgr, response)
        self.rgb_preview.set_preview_pixmap(_bgr_to_pixmap(overlay_bgr))
        merged = np.hstack([overlay_bgr, contrast_bgr])
        self.rgbd_preview.set_preview_pixmap(_bgr_to_pixmap(merged))
        self._tray_count_label_update(response)
        self._selected_index_label_update(response)
        self._completed_frame_label_update(result_packet.frame_idx)
        self._elapsed_label_update(response)
        self._error_label_update(result_packet.error if result_packet.error else (None if response is None else response.error))
        self._update_o3d_result(result_packet)
        if result_packet.error:
            logger.error("pose rpc error: frame={} target={} error={}", result_packet.frame_idx, result_packet.target_tray_index, result_packet.error)
        elif response is not None and response.error:
            logger.error("pose response error: frame={} target={} error={}", response.frame_id, response.selected_tray_index, response.error)

        if result_packet.error:
            self.status_label.setText("pose failed: {0}".format(result_packet.error))
            return
        if response is None:
            self.status_label.setText("pose failed: Orin service returned empty response")
            return
        selected = response.selected_result
        if selected is None or selected.pose is None:
            self.status_label.setText(
                "pose ready | tray_count={0} | target={1} | no valid pose".format(
                    response.tray_count,
                    response.selected_tray_index,
                )
            )
            return
        point = np.asarray(selected.pose.grasp_point_mm, dtype=np.float64)
        self.status_label.setText(
            "pose ready | frame={0} | tray={1}/{2} | XYZ=({3:.1f}, {4:.1f}, {5:.1f}) mm | elapsed={6:.1f} ms".format(
                result_packet.frame_idx,
                response.selected_tray_index,
                max(0, response.tray_count - 1),
                point[0],
                point[1],
                point[2],
                response.elapsed_ms,
            )
        )

    def _tray_count_label_update(self, response: Any | None) -> None:
        if response is None:
            self.tray_count_label.setText("托盘数: -")
            return
        self.tray_count_label.setText("托盘数: {0}".format(response.tray_count))

    def _selected_index_label_update(self, response: Any | None) -> None:
        if response is None:
            self.selected_index_label.setText("选中托盘: -")
            return
        self.selected_index_label.setText("选中托盘: {0}".format(response.selected_tray_index))

    def _elapsed_label_update(self, response: Any | None) -> None:
        if response is None:
            self.elapsed_label.setText("耗时: -")
            return
        self.elapsed_label.setText("耗时: {0:.1f} ms".format(float(response.elapsed_ms)))

    def _error_label_update(self, error: str | None) -> None:
        self.error_label.setText("错误: {0}".format(error or "-"))

    def _pending_frame_label_update(self, frame_idx: int | None) -> None:
        self.pending_frame_label.setText("待处理帧: {0}".format("-" if frame_idx is None else frame_idx))

    def _completed_frame_label_update(self, frame_idx: int | None) -> None:
        self.completed_frame_label.setText("完成帧: {0}".format("-" if frame_idx is None else frame_idx))

    def _submitted_label_update(self, value: int) -> None:
        self.submitted_label.setText("提交数: {0}".format(int(value)))

    def _completed_count_label_update(self, value: int) -> None:
        self.completed_count_label.setText("完成数: {0}".format(int(value)))

    def _dropped_label_update(self, value: int) -> None:
        self.dropped_label.setText("丢弃数: {0}".format(int(value)))

    def _update_o3d_result(self, result_packet: _RpcResultPacket) -> None:
        if not self._o3d_ready():
            return
        assert self.o3d_window is not None
        assert self._pose_line is not None
        assert o3d is not None
        frame = result_packet.frame
        intrinsics = result_packet.intrinsics
        points, colors = _rgbd_to_xyz_rgb(frame.depth, frame.color_bgr, intrinsics)
        if points.shape[0] > DEFAULT_MAX_PREVIEW_POINTS:
            sample_idx = np.linspace(0, points.shape[0] - 1, DEFAULT_MAX_PREVIEW_POINTS, dtype=np.int64)
            points = points[sample_idx]
            colors = colors[sample_idx]
        viewer = self.o3d_window.viewer
        viewer.update_point_cloud("pose_raw_cloud", points=points, colors=colors)
        if not self._o3d_view_fitted and points.shape[0] > 0:
            self._fit_o3d_view_to_cloud(points)
            self._o3d_view_fitted = True

        pose_matrix = np.eye(4, dtype=np.float64)
        hidden_matrix = np.eye(4, dtype=np.float64)
        hidden_matrix[:3, 3] = np.array([0.0, 0.0, -10000.0], dtype=np.float64)
        line_points = np.asarray([[0.0, 0.0, -10000.0], [1.0, 0.0, -10000.0]], dtype=np.float64)

        response = result_packet.response
        if response is not None and response.selected_result is not None and response.selected_result.pose is not None:
            pose = response.selected_result.pose
            rotation = np.asarray(pose.rotation, dtype=np.float64)
            grasp_point = np.asarray(pose.grasp_point_mm, dtype=np.float64)
            pre_grasp_point = np.asarray(pose.pre_grasp_point_mm, dtype=np.float64)
            pose_matrix[:3, :3] = rotation
            pose_matrix[:3, 3] = grasp_point
            line_points = np.vstack([grasp_point, pre_grasp_point]).astype(np.float64)
            viewer.transform_helper_geometry(self._pose_axis_name, pose_matrix, absolute=True)
        else:
            viewer.transform_helper_geometry(self._pose_axis_name, hidden_matrix, absolute=True)

        self._pose_line.points = o3d.utility.Vector3dVector(line_points)
        if viewer.vis is not None:
            viewer.vis.update_geometry(self._pose_line)

    def _fit_o3d_view_to_cloud(self, points: np.ndarray) -> None:
        if not self._o3d_ready() or points.size == 0:
            return
        assert self.o3d_window is not None
        viewer = self.o3d_window.viewer
        if viewer.vis is None:
            return
        min_xyz = np.min(points, axis=0)
        max_xyz = np.max(points, axis=0)
        center = (min_xyz + max_xyz) * 0.5
        extent = max(float(np.linalg.norm(max_xyz - min_xyz)), 1.0)
        view = viewer.vis.get_view_control()
        if view is None:
            return
        view.set_lookat(center.tolist())
        view.set_front([0.0, 0.0, -1.0])
        view.set_up([0.0, -1.0, 0.0])
        zoom = max(0.02, min(0.8, 3000.0 / extent))
        view.set_zoom(float(zoom))
        viewer.vis.poll_events()
        viewer.vis.update_renderer()

    def _o3d_ready(self) -> bool:
        return self.o3d_window is not None and self._raw_cloud is not None and self._pose_line is not None


def _convert_context_result(result: WujiPoseExecutionResult) -> _RpcResultPacket:
    return _RpcResultPacket(
        frame_idx=int(result.frame_id),
        frame=result.frame,
        intrinsics=result.intrinsics,
        target_tray_index=int(result.request.target_tray_index),
        response=result.response,
        error=result.error,
    )


def _rgbd_to_xyz_rgb(
    depth: np.ndarray | None,
    color_bgr: np.ndarray,
    intrinsics: WujiCameraIntrinsicsInfo,
) -> tuple[np.ndarray, np.ndarray]:
    if depth is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2 or depth_array.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    image_h, image_w = depth_array.shape
    color = np.asarray(color_bgr, dtype=np.uint8)
    if color.shape[:2] != (image_h, image_w):
        color = cv2.resize(color, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
    vv, uu = np.indices((image_h, image_w), dtype=np.float32)
    valid = np.isfinite(depth_array) & (depth_array > 1.0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    z = depth_array[valid]
    x = (uu[valid] - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (vv[valid] - float(intrinsics.cy)) * z / float(intrinsics.fy)
    points = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    colors = color[valid][:, ::-1].astype(np.float64, copy=False) / 255.0
    return points, np.clip(colors, 0.0, 1.0)


def _depth_to_color(depth: np.ndarray | None) -> np.ndarray:
    if depth is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    depth_array = np.asarray(depth)
    valid = np.isfinite(depth_array) & (depth_array > 0)
    gray = np.zeros(depth_array.shape[:2], dtype=np.uint8)
    if np.any(valid):
        valid_values = depth_array[valid].astype(np.float32, copy=False)
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        if max_value > min_value:
            gray[valid] = np.clip(
                (valid_values - min_value) * 255.0 / (max_value - min_value),
                0.0,
                255.0,
            ).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def _compose_debug_overlay(base_bgr: np.ndarray, response: Any | None) -> np.ndarray:
    """将 Orin 响应重绘为接近原测试脚本风格的 RGB 辅助图。"""

    overlay = np.asarray(base_bgr, dtype=np.uint8).copy()
    if overlay.ndim != 3 or overlay.shape[2] != 3 or response is None:
        return overlay
    debug_artifacts = response.debug
    if debug_artifacts is not None and debug_artifacts.tray_instance_masks:
        for index, mask in enumerate(debug_artifacts.tray_instance_masks):
            tray_mask = np.asarray(mask, dtype=np.uint8)
            overlay = _blend_mask_overlay(overlay, tray_mask, (0, 180, 180), 0.18)
            _draw_mask_outline(overlay, tray_mask, (0, 220, 220))
            tray_info = response.all_tray_results[index] if index < len(response.all_tray_results) else None
            if tray_info is not None:
                _draw_text(
                    overlay,
                    "Tray {0}".format(tray_info.tray_index),
                    (int(tray_info.tray_bbox_xywh[0]), max(14, int(tray_info.tray_bbox_xywh[1]) - 4)),
                    (0, 220, 220),
                )
    if debug_artifacts is not None and debug_artifacts.near_plane_mask is not None:
        near_mask = np.asarray(debug_artifacts.near_plane_mask, dtype=np.uint8)
        overlay = _blend_mask_overlay(overlay, near_mask, (255, 120, 0), 0.32)
        _draw_mask_outline(overlay, near_mask, (255, 170, 0))
        _draw_first_mask_label(overlay, near_mask, "Opening Plane", (255, 170, 0))
    if debug_artifacts is not None and debug_artifacts.no_hole_mask is not None:
        top_mask = np.asarray(debug_artifacts.no_hole_mask, dtype=np.uint8)
        overlay = _blend_mask_overlay(overlay, top_mask, (0, 200, 0), 0.26)
        _draw_mask_outline(overlay, top_mask, (0, 255, 0))
        _draw_first_mask_label(overlay, top_mask, "Top Plane", (0, 255, 0))
    if debug_artifacts is not None and debug_artifacts.selected_tray_mask is not None:
        _draw_mask_outline(overlay, np.asarray(debug_artifacts.selected_tray_mask, dtype=np.uint8), (0, 0, 255))
    _draw_response_geometry(overlay, response)
    return overlay


def _compose_contrast_overlay(base_bgr: np.ndarray, response: Any | None) -> np.ndarray:
    contrast = np.asarray(base_bgr, dtype=np.uint8).copy()
    if contrast.ndim != 3 or contrast.shape[2] != 3 or response is None or response.debug is None:
        return contrast
    debug_artifacts = response.debug
    if debug_artifacts.near_plane_mask is not None:
        _draw_mask_outline(contrast, np.asarray(debug_artifacts.near_plane_mask, dtype=np.uint8), (0, 165, 255))
    if debug_artifacts.no_hole_mask is not None:
        _draw_mask_outline(contrast, np.asarray(debug_artifacts.no_hole_mask, dtype=np.uint8), (0, 255, 0))
    _draw_response_geometry(contrast, response)
    _draw_text(contrast, "High-contrast retain + edges", (16, 26), (255, 255, 255), thickness=2)
    return contrast


def _draw_response_geometry(image_bgr: np.ndarray, response: Any) -> None:
    for tray_info in getattr(response, "all_tray_results", tuple()):
        if tray_info.top_quad_uv is not None:
            top_quad = np.round(np.asarray(tray_info.top_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(image_bgr, [top_quad], True, (0, 255, 0), 1, cv2.LINE_AA)
        if tray_info.opening_quad_uv is not None:
            opening_quad = np.round(np.asarray(tray_info.opening_quad_uv, dtype=np.float64)).astype(np.int32)
            cv2.polylines(image_bgr, [opening_quad], True, (0, 0, 255), 1, cv2.LINE_AA)
            qx, qy, _, _ = cv2.boundingRect(opening_quad)
            _draw_text(image_bgr, "Opening", (qx, max(14, qy - 4)), (0, 0, 255))
        if tray_info.opening_center_uv is not None:
            center = np.round(np.asarray(tray_info.opening_center_uv, dtype=np.float64)).astype(np.int32)
            cv2.circle(image_bgr, (int(center[0]), int(center[1])), 3, (0, 255, 0), -1)
            _draw_text(image_bgr, "Center", (int(center[0]) + 4, int(center[1]) - 4), (0, 255, 0))
        if tray_info.pose is not None:
            _draw_pose_annotation(image_bgr, tray_info.pose)
    _draw_text(
        image_bgr,
        "tray={0}".format("multi-instance" if getattr(response, "tray_count", 0) > 1 else "single"),
        (12, 66),
        (60, 255, 60),
    )
    _draw_text(image_bgr, "near-plane(orange) -> top-plane(green)", (12, 84), (255, 255, 255))


def _draw_pose_annotation(image_bgr: np.ndarray, pose: Any) -> None:
    grasp_point = np.asarray(pose.grasp_point_mm, dtype=np.float64)
    rpy = np.asarray(pose.rpy_deg, dtype=np.float64)
    _draw_text(
        image_bgr,
        "grasp XYZ {0:.1f}, {1:.1f}, {2:.1f} mm".format(grasp_point[0], grasp_point[1], grasp_point[2]),
        (12, 24),
        (255, 255, 255),
    )
    _draw_text(
        image_bgr,
        "RPY {0:.1f}, {1:.1f}, {2:.1f} deg".format(rpy[0], rpy[1], rpy[2]),
        (12, 46),
        (255, 255, 255),
    )


def _draw_text(
    image_bgr: np.ndarray,
    text: str,
    origin_xy: tuple[int, int],
    color_bgr: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        image_bgr,
        str(text),
        (int(origin_xy[0]), int(origin_xy[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 if thickness == 1 else 0.62,
        color_bgr,
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_mask_outline(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> None:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or not np.any(mask_u8 > 0):
        return
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image_bgr, contours, -1, color_bgr, 1, cv2.LINE_AA)


def _draw_first_mask_label(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    label: str,
    color_bgr: tuple[int, int, int],
) -> None:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or not np.any(mask_u8 > 0):
        return
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0:
        return
    _draw_text(image_bgr, label, (int(np.min(xs)), max(14, int(np.min(ys)) - 4)), color_bgr)


def _blend_mask_overlay(base_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    """将二值 mask 以指定颜色叠加到 BGR 图像上。"""

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.ndim != 2 or base_bgr.ndim != 3 or base_bgr.shape[:2] != mask_u8.shape[:2]:
        return base_bgr
    mask_bool = mask_u8 > 0
    if not np.any(mask_bool):
        return base_bgr
    result = np.asarray(base_bgr, dtype=np.float32).copy()
    color = np.asarray(color_bgr, dtype=np.float32)
    result[mask_bool] = result[mask_bool] * (1.0 - float(alpha)) + color * float(alpha)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _bgr_to_pixmap(image_bgr: np.ndarray) -> QPixmap:
    image_rgb = cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    qimage = QImage(image_rgb.data, width, height, int(image_rgb.strides[0]), QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
