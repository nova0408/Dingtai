from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensorFrustumConfig:
    """传感器理论视锥参数（单位：毫米）。"""

    min_depth_mm: float = 70.0
    max_depth_mm: float = 430.0
    near_width_mm: float = 117.0
    near_height_mm: float = 89.0
    far_width_mm: float = 839.0
    far_height_mm: float = 637.0


@dataclass(frozen=True)
class SessionOptions:
    """Orbbec 会话运行参数。"""

    timeout_ms: int = 120
    enable_frame_sync: bool = True
    require_full_frame_when_color: bool = True
    preferred_capture_fps: int | None = None


@dataclass(frozen=True)
class IntrinsicPatch:
    """相机内参微调补丁。"""

    fx_scale: float = 1.0
    fy_scale: float = 1.0
    cx_offset: float = 0.0
    cy_offset: float = 0.0


@dataclass(frozen=True)
class DistortionPatch:
    """相机畸变参数微调补丁。"""

    k1_offset: float = 0.0
    k2_offset: float = 0.0
    p1_offset: float = 0.0
    p2_offset: float = 0.0


@dataclass(frozen=True)
class CameraParamPatch:
    """相机参数补丁聚合对象。"""

    depth: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    color: IntrinsicPatch = field(default_factory=IntrinsicPatch)
    depth_dist: DistortionPatch = field(default_factory=DistortionPatch)
    color_dist: DistortionPatch = field(default_factory=DistortionPatch)
    d2c_translation_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
