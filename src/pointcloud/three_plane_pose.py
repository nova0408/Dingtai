from __future__ import annotations

import numpy as np
import open3d as o3d

from src.utils.datas import Axis, Point, Vector

from .three_plane_types import (
    CoordinateFramePose,
    PlanePatch,
    PlanePoseConfig,
    ThreePlanePoseResult,
)


# region 主入口
def make_coordinate_frame_pose(
    origin: np.ndarray,
    rotation: np.ndarray,
    rpy_deg: np.ndarray,
    residual_mm: float,
) -> CoordinateFramePose:
    """从 NumPy 位姿分量构造项目统一坐标系位姿。

    Parameters
    ----------
    origin_mm:
        坐标系原点，形状为 `(3,)`，单位 mm。
    rotation:
        旋转矩阵，形状为 `(3, 3)`。
    rpy_deg:
        欧拉角数组，形状为 `(3,)`，单位 deg。
    residual_mm:
        三平面交点残差，单位 mm。

    Returns
    -------
    pose:
        以 `src.utils.datas.Axis` 为唯一位姿源数据的坐标系位姿。

    Notes
    -----
    输入旋转矩阵只用于构造 `Axis`。最终旋转矩阵语义由 `Axis.to_transform().as_SE3()`
    决定，避免算法层手工拼接出与项目 Transform 协议不一致的矩阵。
    """
    axis = Axis(
        origin=Point.from_array(np.asarray(origin, dtype=np.float64)),
        x_axis=Vector.from_array(np.asarray(rotation, dtype=np.float64)[:, 0]),
        z_axis=Vector.from_array(np.asarray(rotation, dtype=np.float64)[:, 2]),
    )
    return CoordinateFramePose(
        axis=axis,
        rpy_deg=np.asarray(rpy_deg, dtype=np.float64),
        residual=float(residual_mm),
    )


def estimate_three_plane_pose(
    xyz: np.ndarray,
    *,
    excluded_mask: np.ndarray | None = None,
    config: PlanePoseConfig | None = None,
) -> ThreePlanePoseResult:
    """从裁切点云估计三平面和测试坐标系。

    Parameters
    ----------
    xyz:
        裁切后的点云数组，形状为 `(N, 3)` 或 `(N, C)`，单位 mm。仅前 3 列 XYZ 参与几何计算。
    excluded_mask:
        可选点云排除掩码，形状为 `(N,)`，dtype 为 `bool`。True 表示该点不参与 RANSAC 和 PCA。
    config:
        三平面位姿估计配置。为 None 时使用 `PlanePoseConfig()` 默认值。

    Returns
    -------
    result:
        三平面估计结果。`labels` 与输入点云逐点对齐，`planes` 是排序后的平面模型，
        `pose` 是由三个平面求交和法线方向构造出的坐标系，失败时为 None。

    Notes
    -----
    该函数只处理几何算法，不访问相机、不加载模型、不做 Open3D 预览。
    """
    cfg = config or PlanePoseConfig()
    # xyz 输入可带颜色列；pts 固定截取为 (N, 3) float64，后续距离计算统一按毫米坐标处理。
    pts = np.asarray(xyz[:, :3], dtype=np.float64)
    # excluded: (N,) bool；True 表示该点来自料盘区域，平面拟合和标签回写都应跳过。
    excluded = np.zeros((pts.shape[0],), dtype=bool) if excluded_mask is None else np.asarray(excluded_mask, dtype=bool)
    # excluded=True 的点来自料盘识别结果，不参与三平面 RANSAC。
    candidate_xyz = pts[~excluded]
    plane_models = _segment_plane_models(candidate_xyz, cfg)
    # labels: (N,) int32；-1 未分配，0/1/2 是三平面候选，-2 是料盘排除点。
    labels = _assign_points_to_planes(pts, plane_models, cfg.plane_distance_mm)
    labels[excluded] = -2
    if cfg.refine_plane_models:
        plane_models = _refine_plane_models_by_pca(
            xyz=pts,
            labels=labels,
            plane_models=plane_models,
            max_points=cfg.plane_refine_max_points,
        )
        labels = _assign_points_to_planes(pts, plane_models, cfg.plane_distance_mm)
        labels[excluded] = -2

    ordered = _order_planes(plane_models=plane_models, labels=labels, xyz=pts, bottom_axis=cfg.bottom_axis)
    labels = _remap_labels(labels=labels, ordered_old_ids=[old_id for old_id, _ in ordered])
    planes: list[PlanePatch] = []
    for new_id, (old_id, label) in enumerate(ordered):
        # labels == new_id 产生 (N,) bool 掩码，用于统计当前重排后平面的点数。
        count = int(np.count_nonzero(labels == new_id))
        if count <= 0:
            continue
        planes.append(
            PlanePatch(
                label=label,
                model=plane_models[old_id],
                inlier_count=count,
            )
        )
    return ThreePlanePoseResult(labels=labels, planes=planes, pose=compute_coordinate_frame_pose(planes, cfg))


# endregion


# region 坐标系与位姿工具
def compute_coordinate_frame_pose(planes: list[PlanePatch], cfg: PlanePoseConfig) -> CoordinateFramePose | None:
    """根据三个已排序平面构造测试坐标系。

    Parameters
    ----------
    planes:
        平面列表，需要至少包含 `bottom`、`left_side`、`right_side` 三个语义平面。
        每个平面的 `model` 形状为 `(4,)`，表示 `ax + by + cz + d = 0`。
    cfg:
        坐标系方向配置，主要使用 `frame_z_hint`、`frame_x_hint` 和 `use_fixed_x_hint_axis`。

    Returns
    -------
    pose:
        计算出的测试坐标系位姿。原点为三平面交点，方向由 `Axis` 表达；
        平面缺失或三平面退化时返回 None。

    Notes
    -----
    Z 轴默认来自底面法线并按 `frame_z_hint` 定向；X 轴优先使用投影到切平面的 `frame_x_hint`。
    """
    if len(planes) < 3:
        return None
    by_label = {p.label: p for p in planes}
    bottom = by_label.get("bottom")
    left = by_label.get("left_side")
    right = by_label.get("right_side")
    if bottom is None or left is None or right is None:
        return None
    origin, residual = _intersect_three_planes([bottom.model, left.model, right.model])
    if origin is None:
        return None

    # plane model 为 [a, b, c, d]，前三项是平面法线；底面法线作为测试坐标系 Z 轴。
    z_axis = _orient_axis_to_hint(_plane_normal_vector(bottom.model), _coerce_hint_vector(cfg.frame_z_hint))
    frame_x_hint = _coerce_hint_vector(cfg.frame_x_hint)
    # 将 X 参考方向投影到底面切平面内，避免与 Z 轴不正交导致姿态抖动。
    x_hint = (frame_x_hint - z_axis * frame_x_hint.dot(z_axis)).normalized()
    if cfg.use_fixed_x_hint_axis and x_hint.length >= 1e-8:
        x_axis = x_hint
    else:
        # 两个侧面法线叉乘得到侧面交线方向，作为备选 X 轴方向。
        x_axis = _plane_normal_vector(left.model).cross(_plane_normal_vector(right.model)).normalized()
        x_axis = (x_axis - z_axis * x_axis.dot(z_axis)).normalized()
        if x_axis.length < 1e-8:
            left_normal = _plane_normal_vector(left.model)
            x_axis = (left_normal - z_axis * left_normal.dot(z_axis)).normalized()
        if x_hint.length >= 1e-8 and x_axis.dot(x_hint) < 0.0:
            x_axis = x_axis.negated()
    axis = Axis(origin=Point.from_array(origin), x_axis=x_axis, z_axis=z_axis)
    rotation = axis.to_transform().as_SE3()[:3, :3]
    return make_coordinate_frame_pose(
        origin=axis.origin.as_array(),
        rotation=np.asarray(rotation, dtype=np.float64),
        rpy_deg=rotation_matrix_to_rpy_deg(rotation),
        residual_mm=float(residual),
    )


def pose_to_matrix(pose: CoordinateFramePose) -> np.ndarray:
    """将坐标系位姿转换为 4x4 齐次矩阵。

    Parameters
    ----------
    pose:
        测试坐标系位姿。位姿源数据为 `pose.axis`。

    Returns
    -------
    matrix:
        齐次变换矩阵，形状为 `(4, 4)`，dtype 为 `float64`，平移单位 mm。
    """
    return np.asarray(pose.axis.to_transform().as_SE3(), dtype=np.float64)


def relative_pose(reference: CoordinateFramePose, current: CoordinateFramePose) -> CoordinateFramePose:
    """计算当前坐标系相对参考坐标系的位姿差。

    Parameters
    ----------
    reference:
        参考坐标系位姿。
    current:
        当前坐标系位姿。

    Returns
    -------
    delta_pose:
        `current` 在 `reference` 坐标系下的相对位姿。平移单位 mm，RPY 单位 deg。
    """
    # T_delta = inv(T_ref) @ T_current，得到 current 在参考坐标系下的相对位姿。
    delta = np.linalg.inv(pose_to_matrix(reference)) @ pose_to_matrix(current)
    return make_coordinate_frame_pose(
        origin=delta[:3, 3],
        rotation=delta[:3, :3],
        rpy_deg=rotation_matrix_to_rpy_deg(delta[:3, :3]),
        residual_mm=current.residual,
    )


def rotation_matrix_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为 roll/pitch/yaw 欧拉角。

    Parameters
    ----------
    rotation:
        旋转矩阵，形状为 `(3, 3)`，dtype 可为任意浮点类型。

    Returns
    -------
    rpy_deg:
        欧拉角数组，形状为 `(3,)`，顺序为 roll/pitch/yaw，单位 deg。

    Notes
    -----
    当 pitch 接近奇异位姿时使用简化分支，并将 yaw 置为 0。
    """
    r = np.asarray(rotation, dtype=np.float64)
    # sy 是 R 的第一列在 XY 平面的长度，用于判断 pitch 接近 +/-90 度的奇异情况。
    sy = float(np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0]))
    if sy >= 1e-8:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.asarray([roll, pitch, yaw], dtype=np.float64))


# endregion


# region 平面拟合
def _segment_plane_models(xyz: np.ndarray, cfg: PlanePoseConfig) -> list[np.ndarray]:
    """使用 Open3D RANSAC 从点云中分割最多三个平面模型。

    Parameters
    ----------
    xyz:
        候选点云数组，形状为 `(N, 3)`，单位 mm，通常已排除料盘点。
    cfg:
        三平面估计配置，使用其中的距离阈值、最小点数、迭代次数和采样上限。

    Returns
    -------
    models:
        平面模型列表。每个模型形状为 `(4,)`，dtype 为 `float64`，表示 `ax + by + cz + d = 0`。

    Notes
    -----
    输入点数超过 `cfg.max_ransac_points` 时使用固定步长降采样，避免 RANSAC 计算过慢。
    """
    if xyz.shape[0] < cfg.plane_min_points:
        return []
    sample = _downsample_points(xyz, max_points=cfg.max_ransac_points)
    remain = o3d.geometry.PointCloud()
    # Open3D Vector3dVector 需要连续的 (M, 3) float64 数组，ascontiguousarray 避免视图步长问题。
    remain.points = o3d.utility.Vector3dVector(np.ascontiguousarray(sample, dtype=np.float64))
    out: list[np.ndarray] = []
    for _ in range(3):
        if len(remain.points) < cfg.plane_min_points:
            break
        model, inliers = remain.segment_plane(
            distance_threshold=float(cfg.plane_distance_mm),
            ransac_n=3,
            num_iterations=int(cfg.ransac_iterations),
        )
        if len(inliers) < cfg.plane_min_points:
            break
        out.append(_normalize_plane_model(np.asarray(model, dtype=np.float64)))
        remain = remain.select_by_index(inliers, invert=True)
    return out


def _assign_points_to_planes(xyz: np.ndarray, plane_models: list[np.ndarray], distance_mm: float) -> np.ndarray:
    """把点云分配给最近的候选平面。

    Parameters
    ----------
    xyz:
        点云数组，形状为 `(N, 3)`，单位 mm。
    plane_models:
        候选平面模型列表，每个模型形状为 `(4,)`。
    distance_mm:
        点到平面的最大归属距离，单位 mm。

    Returns
    -------
    labels:
        逐点标签数组，形状为 `(N,)`，dtype 为 `int32`。-1 表示未分配，非负值表示平面索引。
    """
    # labels 与输入点云一一对应，形状 (N,)；默认 -1 表示不属于任何候选平面。
    labels = np.full((xyz.shape[0],), -1, dtype=np.int32)
    if len(plane_models) == 0:
        return labels
    # dist_mat 形状为 N x P：每一行是一个点到各候选平面的绝对距离。
    # xyz @ normal: (N, 3) @ (3,) -> (N,)，再加 d 得到每个点到单个平面的有符号距离。
    dist = [np.abs(xyz @ m[:3] + float(m[3])) for m in plane_models]
    dist_mat = np.stack(dist, axis=1)
    # best/best_dist 都是 (N,)；np.arange(N) 用于逐行取每个点最近平面的距离。
    best = np.argmin(dist_mat, axis=1)
    best_dist = dist_mat[np.arange(xyz.shape[0]), best]
    labels[best_dist <= float(distance_mm)] = best[best_dist <= float(distance_mm)].astype(np.int32)
    return labels


def _refine_plane_models_by_pca(
    xyz: np.ndarray,
    labels: np.ndarray,
    plane_models: list[np.ndarray],
    max_points: int,
) -> list[np.ndarray]:
    """用 PCA 根据已分配内点精修平面模型。

    Parameters
    ----------
    xyz:
        全量点云数组，形状为 `(N, 3)`，单位 mm。
    labels:
        逐点平面标签，形状为 `(N,)`，dtype 为 `int32`。
    plane_models:
        RANSAC 得到的初始平面模型列表，每个模型形状为 `(4,)`。
    max_points:
        每个平面参与 PCA 的最大点数，单位 点。

    Returns
    -------
    refined_models:
        精修后的平面模型列表，长度与 `plane_models` 相同。

    Notes
    -----
    PCA 法线方向会按 seed model 对齐，避免符号翻转影响后续坐标系方向。
    """
    refined: list[np.ndarray] = []
    for plane_id, seed_model in enumerate(plane_models):
        # labels == plane_id 是 (N,) bool 掩码，直接从 xyz 中切出该平面的点集 (K, 3)。
        pts = xyz[labels == int(plane_id)]
        if pts.shape[0] < 3:
            refined.append(seed_model)
            continue
        model = _fit_plane_model_pca(_downsample_points(pts, max_points=max_points), seed_model=seed_model)
        refined.append(model if model is not None else seed_model)
    return refined


def _fit_plane_model_pca(points: np.ndarray, seed_model: np.ndarray) -> np.ndarray | None:
    """对单个平面点集执行 PCA 平面拟合。

    Parameters
    ----------
    points:
        单平面内点数组，形状为 `(K, 3)` 或 `(K, C)`，单位 mm。
    seed_model:
        初始平面模型，形状为 `(4,)`，用于约束法线符号方向。

    Returns
    -------
    model:
        PCA 拟合后的平面模型，形状为 `(4,)`，dtype 为 `float64`；点数不足时返回 None。
    """
    if points.shape[0] < 3:
        return None
    pts = np.asarray(points[:, :3], dtype=np.float64)
    # center: (3,)；对每一列 XYZ 分别求均值。
    center = np.mean(pts, axis=0)
    # 先中心化点集，再用协方差矩阵的最小特征值方向作为平面法线。
    # center.reshape(1, 3) 与 pts: (K, 3) 广播相减，得到 centered: (K, 3)。
    centered = pts - center.reshape(1, 3)
    # cov: (3, 3)，centered.T @ centered 是三维坐标的协方差未归一化形式。
    cov = centered.T @ centered / max(1, pts.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = Vector.from_array(eigvecs[:, int(np.argmin(eigvals))]).normalized()
    seed_normal = _plane_normal_vector(seed_model)
    if normal.dot(seed_normal) < 0.0:
        normal = normal.negated()
    normal_array = normal.as_array()
    d = -float(normal.dot(Vector.from_array(center)))
    return _normalize_plane_model(np.asarray([normal_array[0], normal_array[1], normal_array[2], d], dtype=np.float64))


def _order_planes(
    plane_models: list[np.ndarray],
    labels: np.ndarray,
    xyz: np.ndarray,
    bottom_axis: Vector | tuple[float, float, float] | list[float] | np.ndarray,
) -> list[tuple[int, str]]:
    """把候选平面排序并赋予 bottom/left/right 语义标签。

    Parameters
    ----------
    plane_models:
        候选平面模型列表，每个模型形状为 `(4,)`。
    labels:
        逐点候选平面标签，形状为 `(N,)`。
    xyz:
        点云数组，形状为 `(N, 3)`，单位 mm。
    bottom_axis:
        用于识别底面的相机坐标轴方向。通常来自 `PlanePoseConfig.bottom_axis`。

    Returns
    -------
    ordered:
        `(old_plane_id, label)` 列表。第一个优先为 bottom，侧面按 X 坐标均值从小到大排序。
    """
    if len(plane_models) == 0:
        return []
    axis = _coerce_hint_vector(bottom_axis)
    scores = [abs(_plane_normal_vector(model).dot(axis)) for model in plane_models]
    bottom_old = int(np.argmax(scores))
    side_ids = [idx for idx in range(len(plane_models)) if idx != bottom_old]
    side_ids.sort(key=lambda idx: _plane_centroid_x(labels=labels, xyz=xyz, plane_id=idx))
    ordered = [(bottom_old, "bottom")]
    if len(side_ids) >= 1:
        ordered.append((side_ids[0], "left_side"))
    if len(side_ids) >= 2:
        ordered.append((side_ids[1], "right_side"))
    return ordered


def _intersect_three_planes(models: list[np.ndarray]) -> tuple[np.ndarray | None, float]:
    """计算三个平面的交点。

    Parameters
    ----------
    models:
        三个平面模型列表，每个模型形状为 `(4,)`，表示 `ax + by + cz + d = 0`。

    Returns
    -------
    origin:
        三平面交点，形状为 `(3,)`，单位 mm；平面退化时为 None。
    residual:
        线性方程残差，单位 mm；退化时为 `inf`。
    """
    # a: (3, 3) 三个平面法线按行排列；b: (3,) 为 -d。
    a = np.asarray([m[:3] for m in models], dtype=np.float64)
    b = -np.asarray([float(m[3]) for m in models], dtype=np.float64)
    if np.linalg.matrix_rank(a) < 3:
        return None, float("inf")
    # 三个平面写成 A @ origin = b，直接求交点作为测试坐标系原点。
    origin = np.linalg.solve(a, b)
    return origin, float(np.linalg.norm(a @ origin - b))


# endregion


# region 基础数学工具
def _plane_centroid_x(labels: np.ndarray, xyz: np.ndarray, plane_id: int) -> float:
    """计算指定平面内点的 X 坐标均值。

    Parameters
    ----------
    labels:
        逐点标签数组，形状为 `(N,)`，dtype 为 `int32`。
    xyz:
        点云坐标数组，形状为 `(N, 3)`，单位 mm。
    plane_id:
        需要统计的平面编号。

    Returns
    -------
    mean_x:
        平面内点 X 坐标均值，单位 mm；该平面无点时返回 0。
    """
    # mask: (N,) bool；用于按点云 X 坐标均值区分左右侧面。
    mask = labels == int(plane_id)
    return 0.0 if not np.any(mask) else float(np.mean(xyz[mask, 0]))


def _remap_labels(labels: np.ndarray, ordered_old_ids: list[int]) -> np.ndarray:
    """按排序结果重映射逐点平面标签。

    Parameters
    ----------
    labels:
        原始逐点标签数组，形状为 `(N,)`，dtype 为 `int32`。
    ordered_old_ids:
        排序后的旧平面编号列表。

    Returns
    -------
    remapped:
        重映射后的标签数组，形状为 `(N,)`。-2 料盘排除点保持不变，未分配点保持 -1。
    """
    # out 保持 (N,) 标签数组形状不变，只把旧平面编号重映射到 bottom/left/right 顺序。
    out = np.full_like(labels, -1)
    out[labels == -2] = -2
    for new_id, old_id in enumerate(ordered_old_ids):
        out[labels == int(old_id)] = int(new_id)
    return out


def _normalize_plane_model(model: np.ndarray) -> np.ndarray:
    """归一化平面模型。

    Parameters
    ----------
    model:
        平面模型数组，形状为 `(4,)`，表示 `ax + by + cz + d = 0`。

    Returns
    -------
    normalized:
        法线长度归一化后的平面模型，形状为 `(4,)`。若法线长度过小则原样返回。

    Notes
    -----
    当前实现会把 Z 分量为负的法线翻转到 Z 非负方向，后续坐标系仍会再按 hint 定向。
    """
    # model: (4,) 表示 ax + by + cz + d = 0；整体除以法线长度后距离单位保持为 mm。
    n = model[:3]
    norm = float(np.linalg.norm(n))
    if norm <= 1e-12:
        return model
    out = model / norm
    if out[2] < 0:
        out = -out
    return out


def _plane_normal_vector(model: np.ndarray) -> Vector:
    """从平面模型中提取单位法线向量。

    Parameters
    ----------
    model:
        平面模型数组，形状为 `(4,)`，dtype 通常为 `float64`，表示 `ax + by + cz + d = 0`。

    Returns
    -------
    normal:
        平面单位法线，项目统一 `Vector` 对象。若模型法线长度接近 0，则返回零向量。

    Notes
    -----
    该函数只读取前三个法线分量，不处理 `d`。点到平面距离仍由 NumPy 模型数组计算，
    方向、点积和叉乘逻辑则交给 `Vector`，避免在姿态构造路径重复实现向量运算。
    """
    return Vector.from_array(np.asarray(model[:3], dtype=np.float64)).normalized()


def _orient_axis_to_hint(axis: Vector, hint: Vector) -> Vector:
    """把轴向量定向到与参考方向同半球。

    Parameters
    ----------
    axis:
        待定向轴向量，项目统一 `Vector` 对象。
    hint:
        参考方向向量，项目统一 `Vector` 对象。

    Returns
    -------
    oriented:
        与 `hint` 点积非负的单位向量，项目统一 `Vector` 对象。
    """
    out = axis.normalized()
    hint_n = hint.normalized()
    if out.length < 1e-8 or hint_n.length < 1e-8:
        return out
    return out.negated() if out.dot(hint_n) < 0.0 else out


def _coerce_hint_vector(value: Vector | tuple[float, float, float] | list[float] | np.ndarray) -> Vector:
    """把用户给定 hint 转换为项目统一单位向量。

    Parameters
    ----------
    value:
        参考方向，可为 `Vector`、tuple、list 或 ndarray，形状为 `(3,)`。

    Returns
    -------
    vector:
        单位向量，项目统一 `Vector` 对象。

    Raises
    ------
    ValueError
        当输入向量长度接近 0 时抛出。
    """
    vector = value.normalized() if isinstance(value, Vector) else Vector.from_array(value).normalized()
    if vector.length <= 1e-8:
        raise ValueError("hint vector must not be zero")
    return vector


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """按固定步长降采样点云数组。

    Parameters
    ----------
    points:
        输入点云或点集数组，形状为 `(N, C)`。
    max_points:
        输出点数上限，单位 点。

    Returns
    -------
    sampled:
        降采样后的数组，形状约为 `(M, C)`，通常是原数组视图而非复制。
    """
    if len(points) <= int(max_points):
        return points
    step = max(1, int(np.ceil(len(points) / float(max_points))))
    # 使用固定步长切片保留输入顺序；返回仍是 (M, C) 视图，不复制大数组。
    return points[::step]


# endregion
