from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.utils.datas import Quaternion, Transform, Translation
from src.utils.datas.kinematics.transform_protocol import MatrixSerializable

PoseLike = MatrixSerializable | NDArray[np.floating]
PairMode = Literal["adjacent", "all"]


@dataclass(frozen=True, slots=True)
class HandEyeResidualStats:
    """手眼标定残差统计。"""

    sample_count: int
    rotation_rmse_deg: float
    rotation_max_deg: float
    translation_rmse: float
    translation_max: float


@dataclass(frozen=True, slots=True)
class HandEyeCalibrationResult:
    """手眼标定结果。"""

    transform: Transform
    residual: HandEyeResidualStats


def _to_se3(pose: PoseLike) -> NDArray[np.float64]:
    if isinstance(pose, np.ndarray):
        mat = pose.astype(np.float64, copy=False)
    else:
        mat = pose.as_SE3().astype(np.float64, copy=False)

    if mat.shape != (4, 4):
        raise ValueError(f"期望 4x4 SE(3) 矩阵，实际形状为 {mat.shape}")
    return mat


def _inv_se3(t: NDArray[np.float64]) -> NDArray[np.float64]:
    r = t[:3, :3]
    p = t[:3, 3]

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -(r.T @ p)
    return out


def _project_to_so3(r: NDArray[np.float64]) -> NDArray[np.float64]:
    u, _, vt = np.linalg.svd(r)
    r_proj = u @ vt
    if np.linalg.det(r_proj) < 0.0:
        u[:, -1] *= -1.0
        r_proj = u @ vt
    return r_proj


def _rotation_error_deg(r_err: NDArray[np.float64]) -> float:
    trace = np.trace(r_err)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _build_relative_pair(t_i: NDArray[np.float64], t_j: NDArray[np.float64]) -> NDArray[np.float64]:
    return _inv_se3(t_i) @ t_j


def make_relative_motion_pairs(
    group_a_poses: list[PoseLike], group_b_poses: list[PoseLike], mode: PairMode = "all"
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """
    从两组同步位姿构造相对运动对 (A_k, B_k)，用于方程 A_k X = X B_k。

    - mode='adjacent': 使用相邻帧构造
    - mode='all': 使用 i<j 的全部组合构造
    """
    if len(group_a_poses) != len(group_b_poses):
        raise ValueError("两组位姿长度必须一致")
    if len(group_a_poses) < 2:
        raise ValueError("至少需要 2 组同步位姿")
    if mode not in ("adjacent", "all"):
        raise ValueError("mode 仅支持 'adjacent' 或 'all'")

    a_abs = [_to_se3(p) for p in group_a_poses]
    b_abs = [_to_se3(p) for p in group_b_poses]

    a_rel: list[NDArray[np.float64]] = []
    b_rel: list[NDArray[np.float64]] = []

    if mode == "adjacent":
        for i in range(len(a_abs) - 1):
            a_rel.append(_build_relative_pair(a_abs[i], a_abs[i + 1]))
            b_rel.append(_build_relative_pair(b_abs[i], b_abs[i + 1]))
        return a_rel, b_rel

    for i in range(len(a_abs) - 1):
        for j in range(i + 1, len(a_abs)):
            a_rel.append(_build_relative_pair(a_abs[i], a_abs[j]))
            b_rel.append(_build_relative_pair(b_abs[i], b_abs[j]))
    return a_rel, b_rel


def calibrate_hand_eye_ax_xb(
    a_motions: list[PoseLike], b_motions: list[PoseLike], min_required_samples: int = 3
) -> Transform:
    """
    纯数学 AX=XB 手眼标定求解。

    输入为相对运动对 A_k、B_k，满足 A_k X = X B_k。
    """
    if len(a_motions) != len(b_motions):
        raise ValueError("A/B 运动对数量必须一致")
    if len(a_motions) < min_required_samples:
        raise ValueError(f"至少需要 {min_required_samples} 组运动对")

    a_seq = [_to_se3(a) for a in a_motions]
    b_seq = [_to_se3(b) for b in b_motions]

    rot_equations: list[NDArray[np.float64]] = []
    for a_mat, b_mat in zip(a_seq, b_seq, strict=True):
        r_a = a_mat[:3, :3]
        r_b = b_mat[:3, :3]
        rot_equations.append(np.kron(np.eye(3), r_a) - np.kron(r_b.T, np.eye(3)))

    m = np.vstack(rot_equations)
    _, _, vt = np.linalg.svd(m)
    r_x_vec = vt[-1]
    r_x_raw = r_x_vec.reshape((3, 3), order="F")
    r_x = _project_to_so3(r_x_raw)

    lhs_blocks: list[NDArray[np.float64]] = []
    rhs_blocks: list[NDArray[np.float64]] = []
    for a_mat, b_mat in zip(a_seq, b_seq, strict=True):
        r_a = a_mat[:3, :3]
        t_a = a_mat[:3, 3]
        t_b = b_mat[:3, 3]
        lhs_blocks.append(r_a - np.eye(3))
        rhs_blocks.append((r_x @ t_b - t_a).reshape(3, 1))

    lhs = np.vstack(lhs_blocks)
    rhs = np.vstack(rhs_blocks).reshape(-1)
    t_x, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)

    x = np.eye(4, dtype=np.float64)
    x[:3, :3] = r_x
    x[:3, 3] = t_x
    return Transform(translation=Translation(*t_x.tolist()), rotation=Quaternion.from_SO3(r_x))


def evaluate_hand_eye_solution(
    a_motions: list[PoseLike], b_motions: list[PoseLike], x: PoseLike
) -> HandEyeResidualStats:
    """评估 A X 与 X B 的闭环残差。"""
    if len(a_motions) != len(b_motions):
        raise ValueError("A/B 运动对数量必须一致")
    if not a_motions:
        raise ValueError("至少需要 1 组运动对")

    x_mat = _to_se3(x)
    rot_errors_deg: list[float] = []
    trans_errors: list[float] = []

    for a_pose, b_pose in zip(a_motions, b_motions, strict=True):
        a_mat = _to_se3(a_pose)
        b_mat = _to_se3(b_pose)
        delta = _inv_se3(a_mat @ x_mat) @ (x_mat @ b_mat)
        rot_errors_deg.append(_rotation_error_deg(delta[:3, :3]))
        trans_errors.append(float(np.linalg.norm(delta[:3, 3])))

    rot_arr = np.array(rot_errors_deg, dtype=np.float64)
    trans_arr = np.array(trans_errors, dtype=np.float64)

    return HandEyeResidualStats(
        sample_count=len(rot_errors_deg),
        rotation_rmse_deg=float(np.sqrt(np.mean(rot_arr**2))),
        rotation_max_deg=float(np.max(rot_arr)),
        translation_rmse=float(np.sqrt(np.mean(trans_arr**2))),
        translation_max=float(np.max(trans_arr)),
    )


def calibrate_hand_eye_from_pose_sequences(
    group_a_poses: list[PoseLike], group_b_poses: list[PoseLike], pair_mode: PairMode = "all"
) -> HandEyeCalibrationResult:
    """
    从同步位姿序列直接完成手眼标定。

    内部流程：
    1) 构造相对运动对 (A_k, B_k)
    2) 求解 X
    3) 输出残差
    """
    a_motions, b_motions = make_relative_motion_pairs(group_a_poses, group_b_poses, mode=pair_mode)
    x = calibrate_hand_eye_ax_xb(a_motions, b_motions)
    residual = evaluate_hand_eye_solution(a_motions, b_motions, x)
    return HandEyeCalibrationResult(transform=x, residual=residual)


def _random_rotation(rng: np.random.Generator) -> NDArray[np.float64]:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _random_transform(rng: np.random.Generator, t_scale: float) -> NDArray[np.float64]:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = _random_rotation(rng)
    t[:3, 3] = rng.uniform(-t_scale, t_scale, size=3)
    return t


def _noise_transform(rng: np.random.Generator, rot_noise_deg: float, trans_noise: float) -> NDArray[np.float64]:
    n = np.eye(4, dtype=np.float64)
    if rot_noise_deg > 0.0:
        axis = rng.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis /= axis_norm
        angle = np.deg2rad(rng.normal(0.0, rot_noise_deg))
        k = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        n[:3, :3] = np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)
    if trans_noise > 0.0:
        n[:3, 3] = rng.normal(0.0, trans_noise, size=3)
    return n


def generate_synthetic_motion_pairs(
    sample_count: int = 30,
    translation_scale: float = 300.0,
    rotation_noise_deg: float = 0.0,
    translation_noise: float = 0.0,
    seed: int | None = None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], Transform]:
    """
    生成可用于手眼标定算法自测的运动对数据。

    返回：(A_motions, B_motions, X_true)
    """
    if sample_count < 3:
        raise ValueError("sample_count 至少为 3")

    rng = np.random.default_rng(seed)
    x_true = _random_transform(rng, t_scale=translation_scale * 0.2)
    x_inv = _inv_se3(x_true)

    a_motions: list[NDArray[np.float64]] = []
    b_motions: list[NDArray[np.float64]] = []

    for _ in range(sample_count):
        a = _random_transform(rng, t_scale=translation_scale)
        b = x_inv @ a @ x_true
        b = _noise_transform(rng, rotation_noise_deg, translation_noise) @ b
        a_motions.append(a)
        b_motions.append(b)

    x_true_transform = Transform.from_SE3(x_true)
    return a_motions, b_motions, x_true_transform
