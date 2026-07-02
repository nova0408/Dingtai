from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from .opening_pipeline import OpeningDetectionPipeline
from .pose_pipeline import GraspPoseEstimator, GraspPoseEstimatorConfig, TemporalFilterState
from .protocol import DebugArtifacts, GraspPoseInfo, TrayPoseInfo
from .types import GraspResult, OpeningDetection


class CameraIntrinsicsProtocol(Protocol):
    """抓取位姿计算所需的相机内参协议。"""

    @property
    def fx(self) -> float: ...

    @property
    def fy(self) -> float: ...

    @property
    def cx(self) -> float: ...

    @property
    def cy(self) -> float: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...


@dataclass(frozen=True)
class OpeningDetectionPipelineExecutorConfig:
    """开口检测与位姿计算执行器配置。"""

    pose_config: GraspPoseEstimatorConfig = field(default_factory=GraspPoseEstimatorConfig)


class OpeningDetectionPipelineExecutor:
    """开口检测与抓取位姿纯计算执行器。

    职责边界：
    - 只接收已经解码好的单帧 RGBD 数据和托盘掩码。
    - 不负责相机采集、托盘 RPC、请求轮询和服务监听。
    - 只输出开口检测、局部平面和抓取位姿结果。
    """

    def __init__(self, config: OpeningDetectionPipelineExecutorConfig | None = None) -> None:
        self._config = OpeningDetectionPipelineExecutorConfig() if config is None else config
        self._opening_pipeline = OpeningDetectionPipeline()
        self._pose_estimator = GraspPoseEstimator(self._config.pose_config)
        self._temporal_state = TemporalFilterState()

    def compute(
        self,
        frame,
        tray_mask: np.ndarray,
        request_id: int,
        target_tray_index: int,
        enable_debug: bool = True,
    ) -> tuple[TrayPoseInfo, DebugArtifacts | None]:
        """基于单帧图像和单个托盘掩码计算开口与抓取位姿。"""

        color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
        depth_mm = np.asarray(frame.depth_mm, dtype=np.float64)
        tray_mask_u8 = np.asarray(tray_mask, dtype=np.uint8)
        _, hp_gray, hp_edge = self._opening_pipeline.build_high_contrast_domain(color_bgr)
        opening = self._opening_pipeline.detect_opening(color_bgr, tray_mask_u8, hp_gray, hp_edge)
        near_plane_mask, no_hole_mask = self._opening_pipeline.compute_mask_pipeline(
            tray_mask_u8,
            True,
            opening,
            hp_gray,
            hp_edge,
        )
        xyz, rgb = self._rgbd_to_points(
            depth_mm,
            color_bgr,
            float(frame.fx),
            float(frame.fy),
            float(frame.cx),
            float(frame.cy),
        )
        uv, valid = self._project_points_to_image(
            xyz,
            float(frame.fx),
            float(frame.fy),
            float(frame.cx),
            float(frame.cy),
            int(color_bgr.shape[1]),
            int(color_bgr.shape[0]),
        )
        xyz_local = self._opening_pipeline.filter_opening_local_points(
            xyz=xyz,
            rgb=rgb,
            opening=opening,
            img_w=int(color_bgr.shape[1]),
            img_h=int(color_bgr.shape[0]),
            uv=uv,
            valid=valid,
        )
        if xyz_local.shape[0] < 80:
            raise RuntimeError(f"开口局部点不足：{xyz_local.shape[0]} 点")
        plane = self._pose_estimator.estimate_plane(xyz_local)
        top_normal = self._opening_pipeline.estimate_top_plane_normal(xyz, no_hole_mask, uv, valid)
        top_normal = self._pose_estimator.stabilize_top_normal(top_normal, self._temporal_state)
        grasp = self._pose_estimator.compute_grasp(
            opening=opening,
            plane=plane,
            intrinsics=_FrameIntrinsics(
                width=int(color_bgr.shape[1]),
                height=int(color_bgr.shape[0]),
                fx=float(frame.fx),
                fy=float(frame.fy),
                cx=float(frame.cx),
                cy=float(frame.cy),
            ),
            top_ref_normal=top_normal,
        )
        grasp = self._pose_estimator.stabilize_grasp_result(grasp, self._temporal_state)
        tray_pose = self._build_tray_pose_info(
            target_tray_index=int(target_tray_index),
            opening=opening,
            grasp=grasp,
        )
        debug = self._build_debug_artifacts(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            tray_mask=tray_mask_u8,
            hp_gray=hp_gray,
            near_plane_mask=near_plane_mask,
            no_hole_mask=no_hole_mask,
            opening=opening,
            grasp=grasp,
            frame_intrinsics=(float(frame.fx), float(frame.fy), float(frame.cx), float(frame.cy)),
        ) if bool(enable_debug) else None
        return tray_pose, debug

    @staticmethod
    def _rgbd_to_points(
        depth_mm: np.ndarray,
        color_bgr: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = depth_mm.shape[:2]
        uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        z = np.asarray(depth_mm, dtype=np.float64)
        valid = np.isfinite(z) & (z > 0.0)
        x = (uu - float(cx)) * z / max(1e-9, float(fx))
        y = (vv - float(cy)) * z / max(1e-9, float(fy))
        xyz = np.stack([x, y, z], axis=-1).reshape((-1, 3))
        rgb = np.asarray(color_bgr, dtype=np.float64).reshape((-1, 3)) / 255.0
        valid_flat = valid.reshape((-1,))
        return xyz[valid_flat], rgb[valid_flat]

    @staticmethod
    def _project_points_to_image(
        xyz: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = np.asarray(xyz[:, 2], dtype=np.float64)
        valid = np.isfinite(z) & (z > 0.0)
        u = (xyz[:, 0] * float(fx)) / np.maximum(1e-9, z) + float(cx)
        v = (xyz[:, 1] * float(fy)) / np.maximum(1e-9, z) + float(cy)
        uv = np.column_stack([u, v]).astype(np.int32)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < int(width)) & (uv[:, 1] >= 0) & (uv[:, 1] < int(height))
        return uv, valid

    @staticmethod
    def _build_tray_pose_info(
        target_tray_index: int,
        opening: OpeningDetection,
        grasp: GraspResult | None,
    ) -> TrayPoseInfo:
        pose = None
        if grasp is not None:
            pose = GraspPoseInfo(
                grasp_point_mm=tuple(float(v) for v in np.asarray(grasp.grasp_point, dtype=np.float64)),
                pre_grasp_point_mm=tuple(float(v) for v in np.asarray(grasp.pre_grasp_point, dtype=np.float64)),
                rotation=tuple(tuple(float(v) for v in row) for row in np.asarray(grasp.rotation, dtype=np.float64)),
                rpy_deg=None,
            )
        return TrayPoseInfo(
            tray_index=int(target_tray_index),
            tray_bbox_xywh=tuple(int(v) for v in opening.bbox_xywh),
            tray_center_uv=tuple(float(v) for v in np.asarray(opening.center_uv, dtype=np.float64)),
            opening_center_uv=tuple(float(v) for v in np.asarray(opening.center_uv, dtype=np.float64)),
            opening_quad_uv=tuple(tuple(float(v) for v in row) for row in np.asarray(opening.quad_uv, dtype=np.float64)),
            top_quad_uv=None,
            pose=pose,
        )

    @staticmethod
    def _build_debug_artifacts(
        color_bgr: np.ndarray,
        depth_mm: np.ndarray,
        tray_mask: np.ndarray,
        hp_gray: np.ndarray,
        near_plane_mask: np.ndarray | None,
        no_hole_mask: np.ndarray | None,
        opening: OpeningDetection,
        grasp: GraspResult | None,
        frame_intrinsics: tuple[float, float, float, float],
    ) -> DebugArtifacts:
        overlay = np.asarray(color_bgr, dtype=np.uint8).copy()
        cv2.polylines(overlay, [np.round(opening.quad_uv).astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "opening", tuple(np.round(opening.center_uv).astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        contrast = cv2.cvtColor(np.asarray(hp_gray, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        return DebugArtifacts(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            camera_intrinsics=frame_intrinsics,
            overlay_bgr=overlay,
            contrast_bgr=contrast,
            tray_instance_masks=(tray_mask,),
            selected_tray_mask=tray_mask,
            near_plane_mask=near_plane_mask,
            no_hole_mask=no_hole_mask,
            opening_center_uv=tuple(float(v) for v in np.asarray(opening.center_uv, dtype=np.float64)),
            opening_quad_uv=tuple(tuple(float(v) for v in row) for row in np.asarray(opening.quad_uv, dtype=np.float64)),
            opening_bbox_xywh=tuple(int(v) for v in opening.bbox_xywh),
            opening_score=float(opening.score),
            grasp_point_mm=None if grasp is None else tuple(float(v) for v in np.asarray(grasp.grasp_point, dtype=np.float64)),
            pre_grasp_point_mm=None if grasp is None else tuple(float(v) for v in np.asarray(grasp.pre_grasp_point, dtype=np.float64)),
            rotation=None if grasp is None else tuple(tuple(float(v) for v in row) for row in np.asarray(grasp.rotation, dtype=np.float64)),
        )


@dataclass(frozen=True)
class _FrameIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
