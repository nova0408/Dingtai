from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.utils.datas import Transform
from src.utils.protocol import MatrixSerializable

PoseLike = MatrixSerializable | NDArray[np.floating]
PairMode = Literal["adjacent", "all"]
HandEyeMethodName = Literal[
    "closed_form",
    "opencv_tsai",
    "opencv_park",
    "opencv_horaud",
    "opencv_daniilidis",
]


@dataclass(frozen=True, slots=True)
class HandEyeResidualStats:
    """手眼标定残差统计。"""

    sample_count: int
    rotation_rmse_deg: float
    rotation_max_deg: float
    translation_rmse: float
    translation_max: float


@dataclass(frozen=True, slots=True)
class HandEyeCrossValidationStats:
    """手眼标定交叉验证统计。"""

    fold_count: int
    train_rotation_rmse_deg_mean: float
    train_translation_rmse_mean: float
    val_rotation_rmse_deg_mean: float
    val_translation_rmse_mean: float
    val_rotation_rmse_deg_max: float
    val_translation_rmse_max: float


@dataclass(frozen=True, slots=True)
class HandEyeStabilityStats:
    """手眼标定外参稳定性统计。"""

    fold_count: int
    rotation_mean_pairwise_deg: float
    rotation_max_pairwise_deg: float
    translation_mean_pairwise: float
    translation_max_pairwise: float
    rotation_std_deg: float
    translation_std: float


@dataclass(frozen=True, slots=True)
class HandEyeCalibrationResult:
    """手眼标定结果。"""

    transform: Transform
    residual: HandEyeResidualStats


@dataclass(frozen=True, slots=True)
class HandEyeMethodResult:
    """单个候选方法的求解与评估结果。"""

    method_name: HandEyeMethodName
    transform: Transform | None
    residual: HandEyeResidualStats | None
    cv_residual: HandEyeCrossValidationStats | None
    stability: HandEyeStabilityStats | None
    score: float | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class HandEyeMultiMethodResult:
    """多方法手眼标定总结果。"""

    best_method: HandEyeMethodName | None
    best_result: HandEyeMethodResult | None
    candidates: tuple[HandEyeMethodResult, ...]


def _to_se3(pose: PoseLike) -> NDArray[np.float64]:
    if isinstance(pose, np.ndarray):
        mat = pose.astype(np.float64, copy=False)
    else:
        mat = pose.as_SE3().astype(np.float64, copy=False)

    if mat.shape != (4, 4):
        raise ValueError(f"期望 4x4 SE(3) 矩阵，实际形状为 {mat.shape}")
    return mat


def _to_transform(x: NDArray[np.float64]) -> Transform:
    return Transform.from_SE3(np.asarray(x, dtype=np.float64).reshape(4, 4))


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


def _method_to_cv_code(method_name: HandEyeMethodName) -> int:
    mapping = {
        "opencv_tsai": cv2.CALIB_HAND_EYE_TSAI,
        "opencv_park": cv2.CALIB_HAND_EYE_PARK,
        "opencv_horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "opencv_daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if method_name not in mapping:
        raise ValueError(f"不支持的 OpenCV 手眼方法: {method_name}")
    return mapping[method_name]


def _sample_folds(sample_count: int, cv_folds: int) -> list[np.ndarray]:
    indices = np.arange(sample_count, dtype=np.int64)
    return [fold for fold in np.array_split(indices, cv_folds) if len(fold) > 0]


def _pair_subset(
    a_motions: Sequence[PoseLike],
    b_motions: Sequence[PoseLike],
    indices: Sequence[int],
    mode: PairMode,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    a_seq = [_to_se3(a_motions[index]) for index in indices]
    b_seq = [_to_se3(b_motions[index]) for index in indices]
    if len(a_seq) < 2:
        raise ValueError("交叉验证子集至少需要 2 个样本")
    return make_relative_motion_pairs(a_seq, b_seq, mode=mode)


def _solve_closed_form_from_motions(
    a_motions: Sequence[PoseLike],
    b_motions: Sequence[PoseLike],
    min_required_samples: int = 3,
) -> NDArray[np.float64]:
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
    return x


def make_relative_motion_pairs(
    group_a_poses: Sequence[PoseLike], group_b_poses: Sequence[PoseLike], mode: PairMode = "all"
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """
    从两组同步位姿构造相对运动对 (A_k, B_k)，用于方程 A_k X = X B_k。
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
    a_motions: Sequence[PoseLike], b_motions: Sequence[PoseLike], min_required_samples: int = 3
) -> Transform:
    """
    纯数学 AX=XB 手眼标定求解。
    """
    x = _solve_closed_form_from_motions(a_motions, b_motions, min_required_samples=min_required_samples)
    return _to_transform(x)


def evaluate_hand_eye_solution(
    a_motions: Sequence[PoseLike], b_motions: Sequence[PoseLike], x: PoseLike
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


def _solve_opencv_hand_eye(
    method_name: HandEyeMethodName,
    group_a_poses: Sequence[PoseLike],
    group_b_poses: Sequence[PoseLike],
) -> NDArray[np.float64]:
    robot_matrices = [_inv_se3(_to_se3(pose)) for pose in group_a_poses]
    board_matrices = [_to_se3(pose) for pose in group_b_poses]
    r_gripper2base = [matrix[:3, :3] for matrix in robot_matrices]
    t_gripper2base = [matrix[:3, 3].reshape(3, 1) for matrix in robot_matrices]
    r_target2cam = [matrix[:3, :3] for matrix in board_matrices]
    t_target2cam = [matrix[:3, 3].reshape(3, 1) for matrix in board_matrices]

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=_method_to_cv_code(method_name),
    )
    raw_rotation = np.asarray(r_cam2gripper, dtype=np.float64).reshape(3, 3)
    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :3] = _project_to_so3(raw_rotation)
    transform_matrix[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
    return transform_matrix


def _estimate_cv_folds(sample_count: int) -> int:
    if sample_count < 6:
        return 1
    if sample_count < 10:
        return 3
    return 5


def _compute_stability_stats(transforms: Sequence[NDArray[np.float64]]) -> HandEyeStabilityStats | None:
    if len(transforms) < 2:
        return None
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for i in range(len(transforms) - 1):
        x_i = np.asarray(transforms[i], dtype=np.float64).reshape(4, 4)
        for j in range(i + 1, len(transforms)):
            x_j = np.asarray(transforms[j], dtype=np.float64).reshape(4, 4)
            delta = _inv_se3(x_i) @ x_j
            rotation_errors.append(_rotation_error_deg(delta[:3, :3]))
            translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
    rotation_arr = np.asarray(rotation_errors, dtype=np.float64)
    translation_arr = np.asarray(translation_errors, dtype=np.float64)
    return HandEyeStabilityStats(
        fold_count=len(transforms),
        rotation_mean_pairwise_deg=float(np.mean(rotation_arr)),
        rotation_max_pairwise_deg=float(np.max(rotation_arr)),
        translation_mean_pairwise=float(np.mean(translation_arr)),
        translation_max_pairwise=float(np.max(translation_arr)),
        rotation_std_deg=float(np.std(rotation_arr)),
        translation_std=float(np.std(translation_arr)),
    )


def _cross_validate_method(
    method_name: HandEyeMethodName,
    group_a_poses: Sequence[PoseLike],
    group_b_poses: Sequence[PoseLike],
    cv_folds: int | None,
    pair_mode: PairMode,
) -> tuple[HandEyeCrossValidationStats | None, HandEyeStabilityStats | None]:
    sample_count = len(group_a_poses)
    folds = 1 if sample_count < 6 else (cv_folds or _estimate_cv_folds(sample_count))
    if folds < 2:
        return None, None
    split_folds = _sample_folds(sample_count, folds)
    if len(split_folds) < 2:
        return None, None

    train_rot: list[float] = []
    train_trans: list[float] = []
    val_rot: list[float] = []
    val_trans: list[float] = []
    fold_transforms: list[NDArray[np.float64]] = []

    all_indices = np.arange(sample_count, dtype=np.int64)
    for val_fold in split_folds:
        train_indices = [int(index) for index in all_indices if int(index) not in set(int(v) for v in val_fold)]
        if len(train_indices) < 2:
            continue
        train_a = [_to_se3(group_a_poses[index]) for index in train_indices]
        train_b = [_to_se3(group_b_poses[index]) for index in train_indices]
        train_a_motions, train_b_motions = make_relative_motion_pairs(train_a, train_b, mode=pair_mode)
        if len(train_a_motions) < 3:
            continue
        if method_name == "closed_form":
            x_train = _solve_closed_form_from_motions(train_a_motions, train_b_motions)
        else:
            x_train = _solve_opencv_hand_eye(method_name, train_a, train_b)
        fold_transforms.append(x_train)
        train_residual = evaluate_hand_eye_solution(train_a_motions, train_b_motions, x_train)
        val_a = [_to_se3(group_a_poses[index]) for index in val_fold]
        val_b = [_to_se3(group_b_poses[index]) for index in val_fold]
        if len(val_a) >= 2:
            val_a_motions, val_b_motions = make_relative_motion_pairs(val_a, val_b, mode=pair_mode)
            val_residual = evaluate_hand_eye_solution(val_a_motions, val_b_motions, x_train)
            val_rot.append(val_residual.rotation_rmse_deg)
            val_trans.append(val_residual.translation_rmse)
        train_rot.append(train_residual.rotation_rmse_deg)
        train_trans.append(train_residual.translation_rmse)

    if not train_rot or not val_rot:
        return None, _compute_stability_stats(fold_transforms)

    return (
        HandEyeCrossValidationStats(
            fold_count=len(train_rot),
            train_rotation_rmse_deg_mean=float(np.mean(np.asarray(train_rot, dtype=np.float64))),
            train_translation_rmse_mean=float(np.mean(np.asarray(train_trans, dtype=np.float64))),
            val_rotation_rmse_deg_mean=float(np.mean(np.asarray(val_rot, dtype=np.float64))),
            val_translation_rmse_mean=float(np.mean(np.asarray(val_trans, dtype=np.float64))),
            val_rotation_rmse_deg_max=float(np.max(np.asarray(val_rot, dtype=np.float64))),
            val_translation_rmse_max=float(np.max(np.asarray(val_trans, dtype=np.float64))),
        ),
        _compute_stability_stats(fold_transforms),
    )


def _score_method(
    residual: HandEyeResidualStats | None,
    cv_residual: HandEyeCrossValidationStats | None,
    stability: HandEyeStabilityStats | None,
    error_message: str | None,
) -> float:
    if error_message is not None or residual is None:
        return float("inf")
    train_rot = residual.rotation_rmse_deg
    train_trans = residual.translation_rmse
    val_rot = cv_residual.val_rotation_rmse_deg_mean if cv_residual is not None else train_rot
    val_trans = cv_residual.val_translation_rmse_mean if cv_residual is not None else train_trans
    stab_rot = stability.rotation_mean_pairwise_deg if stability is not None else 0.0
    stab_trans = stability.translation_mean_pairwise if stability is not None else 0.0
    return (
        4.0 * val_rot
        + 2.0 * val_trans
        + 1.5 * stab_rot
        + 1.0 * stab_trans
        + 0.3 * train_rot
        + 0.1 * train_trans
    )


def _solve_method_result(
    method_name: HandEyeMethodName,
    group_a_poses: Sequence[PoseLike],
    group_b_poses: Sequence[PoseLike],
    cv_folds: int | None,
    pair_mode: PairMode,
) -> HandEyeMethodResult:
    error_message: str | None = None
    transform: Transform | None = None
    residual: HandEyeResidualStats | None = None
    cv_residual: HandEyeCrossValidationStats | None = None
    stability: HandEyeStabilityStats | None = None
    try:
        if method_name == "closed_form":
            a_motions, b_motions = make_relative_motion_pairs(group_a_poses, group_b_poses, mode=pair_mode)
            transform_matrix = _solve_closed_form_from_motions(a_motions, b_motions)
        else:
            transform_matrix = _solve_opencv_hand_eye(method_name, group_a_poses, group_b_poses)
        transform = _to_transform(transform_matrix)
        a_motions, b_motions = make_relative_motion_pairs(group_a_poses, group_b_poses, mode=pair_mode)
        residual = evaluate_hand_eye_solution(a_motions, b_motions, transform_matrix)
        cv_residual, stability = _cross_validate_method(method_name, group_a_poses, group_b_poses, cv_folds, pair_mode)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
    score = _score_method(residual, cv_residual, stability, error_message)
    return HandEyeMethodResult(
        method_name=method_name,
        transform=transform,
        residual=residual,
        cv_residual=cv_residual,
        stability=stability,
        score=score,
        error_message=error_message,
    )


def calibrate_hand_eye_multi_method(
    group_a_poses: Sequence[PoseLike],
    group_b_poses: Sequence[PoseLike],
    pair_mode: PairMode = "all",
    methods: Sequence[HandEyeMethodName] | None = None,
    cv_folds: int | None = None,
) -> HandEyeMultiMethodResult:
    """
    多方法手眼标定统一入口。
    """
    if len(group_a_poses) != len(group_b_poses):
        raise ValueError("两组位姿长度必须一致")
    if len(group_a_poses) < 2:
        raise ValueError("至少需要 2 组同步位姿")
    if pair_mode not in ("adjacent", "all"):
        raise ValueError("pair_mode 仅支持 'adjacent' 或 'all'")

    _ = make_relative_motion_pairs(group_a_poses, group_b_poses, mode=pair_mode)
    method_names = tuple(methods) if methods is not None else (
        "closed_form",
        "opencv_tsai",
        "opencv_park",
        "opencv_horaud",
        "opencv_daniilidis",
    )
    candidates = tuple(
        _solve_method_result(method_name, group_a_poses, group_b_poses, cv_folds=cv_folds, pair_mode=pair_mode)
        for method_name in method_names
    )
    valid_candidates = [candidate for candidate in candidates if candidate.transform is not None]
    best_result = min(valid_candidates, key=lambda item: float(item.score or float("inf")), default=None)
    best_method = None if best_result is None else best_result.method_name
    return HandEyeMultiMethodResult(
        best_method=best_method,
        best_result=best_result,
        candidates=candidates,
    )


def calibrate_hand_eye_from_pose_sequences(
    group_a_poses: Sequence[PoseLike],
    group_b_poses: Sequence[PoseLike],
    pair_mode: PairMode = "all",
    method: Literal["closed_form", "multi_method"] = "closed_form",
) -> HandEyeCalibrationResult:
    """
    从同步位姿序列直接完成手眼标定。
    """
    if method == "multi_method":
        multi_result = calibrate_hand_eye_multi_method(
            group_a_poses=group_a_poses,
            group_b_poses=group_b_poses,
            pair_mode=pair_mode,
        )
        if multi_result.best_result is None or multi_result.best_result.transform is None or multi_result.best_result.residual is None:
            raise RuntimeError("多方法手眼标定失败，没有可用的候选结果")
        return HandEyeCalibrationResult(
            transform=multi_result.best_result.transform,
            residual=multi_result.best_result.residual,
        )

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
