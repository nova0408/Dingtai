from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .types import GraspResult, OpeningDetection, PlaneResult

# region 数据结构


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


@dataclass
class TemporalFilterState:
    """抓取时序稳定状态。

    职责边界：
    - 仅保存时序平滑所需的历史状态。
    - 不负责计算抓取结果，不持有相机、模型或线程资源。
    """

    last_top_normal: np.ndarray | None = None
    "上一次顶面法向量，形状 `(3,)`，dtype `float64`，单位向量。"
    last_grasp: GraspResult | None = None
    "上一次输出抓取结果，用于跨帧平滑。"
    point_hist: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=5))
    "抓取点历史队列，元素形状 `(3,)`，用于中值滤波抑制跳变。"


@dataclass(frozen=True)
class GraspPoseEstimatorConfig:
    """抓取位姿估计器参数配置。

    职责边界：
    - 仅承载算法参数，不包含运行时状态。
    - 作为显式数据契约传入估计器，避免参数散落在 class 属性中。
    """

    top_normal_alpha: float = 0.35
    "顶面法向量 EMA 系数，范围 `0-1`。"
    top_normal_max_angle_deg: float = 12.0
    "顶面法向量单帧最大融合角度，单位 deg。"
    grasp_point_alpha: float = 0.30
    "抓取点平滑系数，范围 `0-1`。"
    grasp_axis_alpha: float = 0.28
    "抓取轴向量平滑系数，范围 `0-1`。"
    grasp_max_translation_mm: float = 40.0
    "抓取点单帧允许的最大平移，单位 mm。"
    grasp_max_axis_angle_deg: float = 15.0
    "抓取轴单帧允许的最大旋转角，单位 deg。"
    grasp_translation_soft_mm: float = 1.5
    "抓取点软门控阈值，单位 mm。"
    grasp_axis_soft_deg: float = 1.2
    "抓取轴软门控阈值，单位 deg。"


# endregion

# region 位姿估计类


class GraspPoseEstimator:
    """开口局部平面与抓取位姿估计器。

    设计说明：
    - 该类聚焦“几何估计与时序稳定”单一职责，不包含图像分割与相机采集逻辑。
    - 通过显式方法暴露平面拟合、抓取求解、法线稳定、姿态稳定，便于测试与替换。
    - 不继承业务基类，避免隐式生命周期和魔术分发。
    """

    def __init__(self, config: GraspPoseEstimatorConfig | None = None) -> None:
        """初始化估计器。

        Parameters
        ----------
        config:
            抓取位姿估计参数配置。为空时使用默认 `GraspPoseEstimatorConfig`。
        """
        self._config = config if config is not None else GraspPoseEstimatorConfig()

    def estimate_plane(self, xyz_local: np.ndarray) -> PlaneResult:
        """拟合开口局部平面。

        Parameters
        ----------
        xyz_local:
            局部点云，形状 `(N, 3)`，dtype `float64/float32`，相机坐标系，单位 mm。

        Returns
        -------
        plane:
            平面模型 `n·x + d = 0`。
        """
        pts = np.asarray(xyz_local, dtype=np.float64)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 50:
            raise RuntimeError(f"滤波后点数太少：{pts.shape[0]}")
        center = np.mean(pts, axis=0)
        q = pts - center.reshape(1, 3)
        cov = (q.T @ q) / max(1, q.shape[0])
        vals, vecs = np.linalg.eigh(cov)
        n = np.asarray(vecs[:, int(np.argmin(vals))], dtype=np.float64)
        n = _normalize(n)
        d = -float(np.dot(n, center))
        if np.dot(n, np.array([0.0, 0.0, -1.0], dtype=np.float64)) < 0.0:
            n = -n
            d = -d
        return PlaneResult(normal=n, d=d)

    def compute_grasp(
        self,
        opening: OpeningDetection,
        plane: PlaneResult,
        intrinsics: CameraIntrinsicsProtocol,
        top_ref_normal: np.ndarray | None,
    ) -> GraspResult:
        """基于开口与平面估计抓取位姿。

        Parameters
        ----------
        opening:
            开口检测结果。
        plane:
            开口局部平面模型。
        intrinsics:
            相机针孔内参协议对象，需提供 `fx/fy/cx/cy/width/height` 字段，单位 像素。
        top_ref_normal:
            顶面参考法向量，形状 `(3,)`，dtype `float64`。为空时回退到默认方向。
        """
        grasp_point = _pixel_ray_intersect_plane(
            opening.center_uv[0], opening.center_uv[1], plane.normal, plane.d, intrinsics
        )
        quad = np.asarray(opening.quad_uv, dtype=np.float64)
        lengths = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
        best_i = int(np.argmax(np.asarray(lengths, dtype=np.float64)))
        edge_dir = _normalize(quad[(best_i + 1) % 4] - quad[best_i])
        half = 0.45 * float(lengths[best_i])
        left_uv = opening.center_uv - edge_dir * half
        right_uv = opening.center_uv + edge_dir * half
        p_left = _pixel_ray_intersect_plane(left_uv[0], left_uv[1], plane.normal, plane.d, intrinsics)
        p_right = _pixel_ray_intersect_plane(right_uv[0], right_uv[1], plane.normal, plane.d, intrinsics)
        x_axis = _normalize(p_right - p_left)
        if float(np.dot(x_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))) < 0.0:
            x_axis = -x_axis
        y_ref = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        if top_ref_normal is not None and float(np.linalg.norm(top_ref_normal)) > 1e-9:
            y_axis = _normalize(np.asarray(top_ref_normal, dtype=np.float64))
            y_axis = y_axis - float(np.dot(y_axis, x_axis)) * x_axis
            y_axis = (
                _normalize(y_axis)
                if float(np.linalg.norm(y_axis)) > 1e-9
                else _normalize(np.cross(_normalize(plane.normal), x_axis))
            )
        else:
            y_axis = y_ref - float(np.dot(y_ref, x_axis)) * x_axis
            y_axis = (
                _normalize(y_axis)
                if float(np.linalg.norm(y_axis)) > 1e-9
                else _normalize(np.cross(_normalize(plane.normal), x_axis))
            )
        if float(np.dot(y_axis, y_ref)) < 0.0:
            y_axis = -y_axis
        z_axis = _normalize(np.cross(x_axis, y_axis))
        y_axis = _normalize(np.cross(z_axis, x_axis))
        rotation = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
        return GraspResult(grasp_point=grasp_point, pre_grasp_point=grasp_point + z_axis * 80.0, rotation=rotation)

    def stabilize_top_normal(self, n: np.ndarray | None, state: TemporalFilterState) -> np.ndarray | None:
        """平滑顶面法线。"""
        if n is None:
            return state.last_top_normal
        nn = _normalize(np.asarray(n, dtype=np.float64))
        if state.last_top_normal is None:
            state.last_top_normal = nn
            return nn
        state.last_top_normal = _blend_unit_vector(
            state.last_top_normal, nn, self._config.top_normal_alpha, self._config.top_normal_max_angle_deg
        )
        return state.last_top_normal

    def stabilize_grasp_result(self, grasp: GraspResult | None, state: TemporalFilterState) -> GraspResult | None:
        """平滑抓取位姿。"""
        if grasp is None:
            return state.last_grasp
        if state.last_grasp is None:
            state.point_hist.clear()
            state.point_hist.append(np.asarray(grasp.grasp_point, dtype=np.float64))
            state.last_grasp = grasp
            return grasp
        prev = state.last_grasp
        curr_p = np.asarray(grasp.grasp_point, dtype=np.float64)
        prev_p = np.asarray(prev.grasp_point, dtype=np.float64)
        dp = curr_p - prev_p
        dp_norm = float(np.linalg.norm(dp))
        p_alpha = self._config.grasp_point_alpha * _soft_gate_gain(dp_norm, self._config.grasp_translation_soft_mm)
        if dp_norm > self._config.grasp_max_translation_mm:
            p_alpha *= self._config.grasp_max_translation_mm / max(1e-9, dp_norm)
        p_new = prev_p + p_alpha * dp
        state.point_hist.append(p_new)
        if len(state.point_hist) >= 3:
            p_new = np.median(np.asarray(list(state.point_hist), dtype=np.float64), axis=0)
        prev_r = np.asarray(prev.rotation, dtype=np.float64)
        curr_r = np.asarray(grasp.rotation, dtype=np.float64)
        x_new = _blend_unit_vector(
            prev_r[:, 0],
            curr_r[:, 0],
            self._config.grasp_axis_alpha
            * _soft_gate_gain(_vector_angle_deg(prev_r[:, 0], curr_r[:, 0]), self._config.grasp_axis_soft_deg),
            self._config.grasp_max_axis_angle_deg,
        )
        y_new = _blend_unit_vector(
            prev_r[:, 1],
            curr_r[:, 1],
            self._config.grasp_axis_alpha
            * _soft_gate_gain(_vector_angle_deg(prev_r[:, 1], curr_r[:, 1]), self._config.grasp_axis_soft_deg),
            self._config.grasp_max_axis_angle_deg,
        )
        y_new = y_new - float(np.dot(y_new, x_new)) * x_new
        y_new = _normalize(y_new) if float(np.linalg.norm(y_new)) > 1e-9 else _normalize(np.cross(prev_r[:, 2], x_new))
        z_new = _normalize(np.cross(x_new, y_new))
        y_new = _normalize(np.cross(z_new, x_new))
        pre_dist = float(np.linalg.norm(np.asarray(grasp.pre_grasp_point, dtype=np.float64) - curr_p))
        pre_dist = pre_dist if np.isfinite(pre_dist) and pre_dist >= 1e-6 else 80.0
        out = GraspResult(
            grasp_point=p_new,
            pre_grasp_point=p_new + z_new * pre_dist,
            rotation=np.column_stack([x_new, y_new, z_new]).astype(np.float64),
        )
        state.last_grasp = out
        return out


# endregion

# region 基础数学工具


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """向量归一化。"""
    n = float(np.linalg.norm(v))
    if n < eps:
        raise RuntimeError(f"向量归一化失败，norm={n}")
    return np.asarray(v, dtype=np.float64) / n


def _pixel_ray_intersect_plane(
    u: float, v: float, n: np.ndarray, d: float, intrinsics: CameraIntrinsicsProtocol
) -> np.ndarray:
    """求像素射线与平面交点。"""
    ray = _normalize(
        np.array(
            [
                (float(u) - float(intrinsics.cx)) / float(intrinsics.fx),
                (float(v) - float(intrinsics.cy)) / float(intrinsics.fy),
                1.0,
            ],
            dtype=np.float64,
        )
    )
    nn = _normalize(n)
    denom = float(np.dot(nn, ray))
    if abs(denom) < 1e-9:
        raise RuntimeError("射线与平面近平行")
    t = -float(d) / denom
    if t <= 0:
        raise RuntimeError(f"交点无效 t={t:.6f}")
    return t * ray


def _blend_unit_vector(prev: np.ndarray, curr: np.ndarray, alpha: float, max_angle_deg: float) -> np.ndarray:
    """在单位球面上平滑两个方向向量。"""
    p = _normalize(prev)
    c = _normalize(curr)
    if float(np.dot(p, c)) < 0.0:
        c = -c
    ang = float(np.degrees(np.arccos(float(np.clip(np.dot(p, c), -1.0, 1.0)))))
    aa = float(np.clip(alpha, 0.0, 1.0))
    if ang > max_angle_deg:
        aa *= float(max_angle_deg / max(1e-6, ang))
    return _normalize((1.0 - aa) * p + aa * c)


def _vector_angle_deg(v0: np.ndarray, v1: np.ndarray) -> float:
    """计算两向量夹角，单位 deg。"""
    return float(np.degrees(np.arccos(float(np.clip(np.dot(_normalize(v0), _normalize(v1)), -1.0, 1.0)))))


def _soft_gate_gain(magnitude: float, soft_threshold: float) -> float:
    """软门控增益函数。"""
    x = max(0.0, float(magnitude)) / max(1e-6, float(soft_threshold))
    return float(x / (1.0 + x))


# endregion
