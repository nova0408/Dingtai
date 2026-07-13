from __future__ import annotations

# pyright: reportMissingImports=false

from dataclasses import dataclass, field
import concurrent.futures
from typing import Protocol

import cv2
import numpy as np

from .opening_pipeline import OpeningDetectionPipeline
from .pose_pipeline import (
    GraspPoseEstimator,
    GraspPoseEstimatorConfig,
    TemporalFilterState,
)
from .protocol import DebugArtifacts, GraspPoseInfo, TrayPoseInfo
from .types import GraspResult, OpeningDetection, OpeningDetectionConfig


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


@dataclass(frozen=True, slots=True)
class OpeningDetectionPipelineExecutorConfig:
    """开口检测与位姿计算执行器配置。"""

    opening_config: OpeningDetectionConfig = field(
        default_factory=OpeningDetectionConfig
    )
    "开口图像预处理和区域生长参数。"
    pose_config: GraspPoseEstimatorConfig = field(
        default_factory=GraspPoseEstimatorConfig
    )
    "抓取几何和时序平滑参数。"


class OpeningDetectionPipelineExecutor:
    """开口检测与抓取位姿纯计算执行器。

    职责边界：
    - 只接收已经解码好的单帧 RGBD 数据和托盘掩码。
    - 不负责相机采集、托盘 RPC、请求轮询和服务监听。
    - 只输出开口检测、局部平面和抓取位姿结果。
    """

    def __init__(
        self, config: OpeningDetectionPipelineExecutorConfig | None = None
    ) -> None:
        self._config = (
            OpeningDetectionPipelineExecutorConfig() if config is None else config
        )
        self._opening_pipeline = OpeningDetectionPipeline(self._config.opening_config)
        self._pose_estimator = GraspPoseEstimator(self._config.pose_config)
        self._temporal_state = TemporalFilterState()

    def compute(
        self,
        frame,
        tray_mask: np.ndarray,
        request_id: int,
        target_tray_index: int,
        enable_debug: bool = True,
    ) -> tuple[TrayPoseInfo, tuple[DebugArtifacts, ...]]:
        """基于单帧图像和单个托盘掩码计算开口与抓取位姿。"""

        color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
        depth_mm = np.asarray(frame.depth_mm, dtype=np.float64)
        tray_mask_u8 = np.asarray(tray_mask, dtype=np.uint8)
        _, hp_gray, hp_edge = self._opening_pipeline.build_high_contrast_domain(
            color_bgr
        )
        opening = self._opening_pipeline.detect_opening(
            color_bgr, tray_mask_u8, hp_gray, hp_edge
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
        intrinsics = _FrameIntrinsics(
            width=int(color_bgr.shape[1]),
            height=int(color_bgr.shape[0]),
            fx=float(frame.fx),
            fy=float(frame.fy),
            cx=float(frame.cx),
            cy=float(frame.cy),
        )
        grasp = self._pose_estimator.compute_grasp(
            opening=opening,
            plane=plane,
            intrinsics=intrinsics,
            top_ref_normal=None,
        )
        near_plane_mask: np.ndarray | None = None
        no_hole_mask: np.ndarray | None = None
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="opening_mask"
            ) as executor:
                mask_future = executor.submit(
                    self._opening_pipeline.compute_mask_pipeline,
                    tray_mask_u8,
                    True,
                    opening,
                    hp_gray,
                    hp_edge,
                )
                near_plane_mask, no_hole_mask = mask_future.result(timeout=0.06)
                top_normal = self._opening_pipeline.estimate_top_plane_normal(
                    xyz, no_hole_mask, uv, valid
                )
                top_normal = self._pose_estimator.stabilize_top_normal(
                    top_normal, self._temporal_state
                )
                grasp = self._pose_estimator.compute_grasp(
                    opening=opening,
                    plane=plane,
                    intrinsics=intrinsics,
                    top_ref_normal=top_normal,
                )
        except concurrent.futures.TimeoutError:
            pass
        except Exception:
            pass
        grasp = self._pose_estimator.stabilize_grasp_result(grasp, self._temporal_state)
        if grasp is None:
            raise RuntimeError("无法从开口平面计算有效抓取位姿")
        tray_pose = self._build_tray_pose_info(
            target_tray_index=int(target_tray_index),
            opening=opening,
            grasp=grasp,
        )
        debug_artifacts = (
            (
                self._build_debug_artifacts(
                    color_bgr=color_bgr,
                    depth_mm=depth_mm,
                    tray_mask=tray_mask_u8,
                    hp_gray=hp_gray,
                    near_plane_mask=near_plane_mask,
                    no_hole_mask=no_hole_mask,
                    opening=opening,
                    grasp=grasp,
                    frame_intrinsics=(
                        float(frame.fx),
                        float(frame.fy),
                        float(frame.cx),
                        float(frame.cy),
                    ),
                ),
            )
            if bool(enable_debug)
            else ()
        )
        return tray_pose, debug_artifacts

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
        uu, vv = np.meshgrid(
            np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
        )
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
        valid &= (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < int(width))
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < int(height))
        )
        return uv, valid

    @staticmethod
    def _build_tray_pose_info(
        target_tray_index: int,
        opening: OpeningDetection,
        grasp: GraspResult,
    ) -> TrayPoseInfo:
        pose = GraspPoseInfo(
            grasp_point_mm=_point3(grasp.grasp_point),
            pre_grasp_point_mm=_point3(grasp.pre_grasp_point),
            rotation=_matrix3(grasp.rotation),
        )
        return TrayPoseInfo(
            tray_index=int(target_tray_index),
            tray_bbox_xywh=_bbox4(opening.bbox_xywh),
            tray_center_uv=_point2(opening.center_uv),
            opening_center_uv=_point2(opening.center_uv),
            opening_quad_uv=_quad2(opening.quad_uv),
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
        grasp: GraspResult,
        frame_intrinsics: tuple[float, float, float, float],
    ) -> DebugArtifacts:
        overlay = np.asarray(color_bgr, dtype=np.uint8).copy()
        cv2.polylines(
            overlay,
            [np.round(opening.quad_uv).astype(np.int32)],
            True,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "opening",
            tuple(np.round(opening.center_uv).astype(np.int32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        contrast = cv2.cvtColor(np.asarray(hp_gray, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        return DebugArtifacts(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            camera_intrinsics=frame_intrinsics,
            overlay_bgr=overlay,
            contrast_bgr=contrast,
            tray_instance_masks=(tray_mask,),
            selected_tray_mask=tray_mask,
            near_plane_masks=() if near_plane_mask is None else (near_plane_mask,),
            no_hole_masks=() if no_hole_mask is None else (no_hole_mask,),
            opening_center_uv=_point2(opening.center_uv),
            opening_quad_uv=_quad2(opening.quad_uv),
            opening_bbox_xywh=_bbox4(opening.bbox_xywh),
            opening_score=float(opening.score),
            grasp_point_mm=_point3(grasp.grasp_point),
            pre_grasp_point_mm=_point3(grasp.pre_grasp_point),
            rotation=_matrix3(grasp.rotation),
        )


def _point2(values: np.ndarray) -> tuple[float, float]:
    return float(values[0]), float(values[1])


def _point3(values: np.ndarray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _quad2(
    values: np.ndarray,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return (
        _point2(values[0]),
        _point2(values[1]),
        _point2(values[2]),
        _point2(values[3]),
    )


def _matrix3(
    values: np.ndarray,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return _point3(values[0]), _point3(values[1]), _point3(values[2])


def _bbox4(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return values[0], values[1], values[2], values[3]


@dataclass(frozen=True, slots=True)
class _FrameIntrinsics:
    """由当前 RGBD 帧提取的针孔相机内参。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
