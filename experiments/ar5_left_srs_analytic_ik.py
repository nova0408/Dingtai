#!/usr/bin/env python3
"""
AR5-5_07L-W4C4A2-S1 七轴 SRS 机械臂解析逆解

适用的 URDF 运动链（关节 origin.rpy 均为 0）：

    J1: Rz(q1)
    J2: Tz(0.1745) * Ry(q2)
    J3: Tz(0.3140) * Rz(q3)
    J4: Tx(0.0105) * Ry(q4)
    J5: T(-0.0105, 0, 0.2600) * Rz(q5)
    J6: Ry(q6)
    J7: Rx(q7)
    TCP: Tz(0.0920)

输入
----
1. pose_target:
   4x4 齐次变换矩阵，表示 base -> TCP。
   平移单位必须是米，旋转矩阵必须是正交右手矩阵。

2. psi:
   肘部绕“肩球心 -> 腕球心”连线的旋臂角，单位 rad。
   psi 不改变 TCP 位姿，但会选择七轴冗余机械臂的一种肘部位置。

3. joint_current:
   可选，当前七关节角，shape=(7,)，单位 rad。
   提供后会从有效解析解中选择与当前关节角加权角距离最小的一组。

输出
----
IKResult:
    solutions:
        所有满足几何、关节限位和正解回代误差要求的解析解。
        每一行是 [q1, q2, q3, q4, q5, q6, q7]，单位 rad。
    selected:
        若提供 joint_current，返回最接近当前角度的解；
        否则返回 solutions[0]；无解时为 None。
    position_errors:
        每组解正解回代后的 TCP 位置误差，单位 m。
    orientation_errors:
        每组解正解回代后的姿态角误差，单位 rad。

说明
----
- 该解析解按修正后的 URDF 几何参数推导，不使用标准化的“无偏置两直杆”近似。
- 10.5 mm 的 J4/J5 横向偏置被保留在解析几何中。
- TCP 的 92 mm 固定长度会先从目标 TCP 位姿中扣除，得到腕球心。
- 肩部为 Z-Y-Z，肘部为 Y，腕部为 Z-Y-X。
- 不处理自碰撞；如需自碰撞检查，应在获得候选解后交给 MoveIt、Pinocchio
  或专门的碰撞检测模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, pi, sin
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RobotGeometry:
    """与 URDF 一致的运动学几何参数，单位为米。"""

    shoulder_height: float = 0.1745
    upper_z: float = 0.3140
    elbow_x: float = 0.0105
    forearm_x: float = -0.0105
    forearm_z: float = 0.2600
    tcp_z: float = 0.0920

    @property
    def shoulder_to_elbow_local(self) -> FloatArray:
        # 在 frame_3 中，从肩球心指向 J4 轴线参考点。
        return np.array([self.elbow_x, 0.0, self.upper_z], dtype=np.float64)

    @property
    def elbow_to_wrist_local(self) -> FloatArray:
        # 在经过 J4 旋转后的 frame_4 中，从 J4 轴线参考点指向腕球心。
        return np.array([self.forearm_x, 0.0, self.forearm_z], dtype=np.float64)

    @property
    def shoulder_position_base(self) -> FloatArray:
        return np.array([0.0, 0.0, self.shoulder_height], dtype=np.float64)

    @property
    def tcp_offset_link7(self) -> FloatArray:
        return np.array([0.0, 0.0, self.tcp_z], dtype=np.float64)


@dataclass(frozen=True)
class JointLimits:
    """来自 URDF 的七个关节限位，单位 rad。"""

    lower: FloatArray
    upper: FloatArray

    @staticmethod
    def from_urdf() -> "JointLimits":
        return JointLimits(
            lower=np.array(
                [-3.1067, -2.0944, -3.1067, -1.0472, -3.1067, -1.0472, -1.0472],
                dtype=np.float64,
            ),
            upper=np.array(
                [3.1067, 2.0944, 3.1067, 2.5307, 3.1067, 1.0472, 1.0472],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class IKSolution:
    joints: FloatArray
    position_error: float
    orientation_error: float


@dataclass(frozen=True)
class IKResult:
    solutions: FloatArray
    selected: FloatArray | None
    position_errors: FloatArray
    orientation_errors: FloatArray


def rot_x(angle: float) -> FloatArray:
    c = cos(angle)
    s = sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def rot_y(angle: float) -> FloatArray:
    c = cos(angle)
    s = sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def rot_z(angle: float) -> FloatArray:
    c = cos(angle)
    s = sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def normalize(vector: FloatArray, *, eps: float = 1e-12) -> FloatArray:
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        raise ValueError("无法归一化接近零长度的向量。")
    return vector / norm


def wrap_to_pi(angle: float) -> float:
    return float(atan2(sin(angle), cos(angle)))


def wrap_joints(joints: FloatArray) -> FloatArray:
    return np.arctan2(np.sin(joints), np.cos(joints)).astype(np.float64)


def skew_rotation_error(r_actual: FloatArray, r_target: FloatArray) -> float:
    relative = r_actual.T @ r_target
    cos_angle = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(acos(cos_angle))


def validate_pose(pose: FloatArray, *, atol: float = 1e-6) -> None:
    if pose.shape != (4, 4):
        raise ValueError(f"pose_target 必须是 4x4，实际 shape={pose.shape}。")
    if not np.all(np.isfinite(pose)):
        raise ValueError("pose_target 包含 NaN 或 Inf。")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("pose_target 最后一行必须是 [0, 0, 0, 1]。")

    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("pose_target 的旋转部分不是正交矩阵。")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("pose_target 的旋转部分不是右手旋转矩阵，det(R) 应为 1。")


def forward_kinematics(
    joints: Iterable[float],
    geometry: RobotGeometry = RobotGeometry(),
) -> FloatArray:
    """按照 URDF 链计算 base -> TCP 正运动学。"""

    q = np.asarray(tuple(joints), dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"joints 必须是 shape=(7,)，实际 shape={q.shape}。")

    q1, q2, q3, q4, q5, q6, q7 = q

    rotation_03 = rot_z(q1) @ rot_y(q2) @ rot_z(q3)

    shoulder = geometry.shoulder_position_base
    elbow = shoulder + rotation_03 @ geometry.shoulder_to_elbow_local

    rotation_04 = rotation_03 @ rot_y(q4)
    wrist = elbow + rotation_04 @ geometry.elbow_to_wrist_local

    rotation_07 = rotation_04 @ rot_z(q5) @ rot_y(q6) @ rot_x(q7)
    tcp = wrist + rotation_07 @ geometry.tcp_offset_link7

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_07
    pose[:3, 3] = tcp
    return pose


def decompose_zyz(rotation: FloatArray, *, singular_eps: float = 1e-9) -> list[FloatArray]:
    """
    分解 R = Rz(q1) Ry(q2) Rz(q3)。

    非奇异时返回两个等价欧拉分支。
    """

    c2 = float(np.clip(rotation[2, 2], -1.0, 1.0))
    q2_abs = acos(c2)

    solutions: list[FloatArray] = []
    for q2 in (q2_abs, -q2_abs):
        s2 = sin(q2)
        if abs(s2) < singular_eps:
            # ZYZ 奇异：q1、q3 只有和或差可确定。
            # 固定 q3=0，令 q1 承担总旋转。
            if c2 > 0.0:
                q1 = atan2(rotation[1, 0], rotation[0, 0])
            else:
                q1 = atan2(-rotation[1, 0], -rotation[0, 0])
            q3 = 0.0
        else:
            q1 = atan2(rotation[1, 2] / s2, rotation[0, 2] / s2)
            q3 = atan2(rotation[2, 1] / s2, -rotation[2, 0] / s2)

        solutions.append(wrap_joints(np.array([q1, q2, q3], dtype=np.float64)))

    return solutions


def decompose_zyx_wrist(
    rotation: FloatArray,
    *,
    singular_eps: float = 1e-9,
) -> list[FloatArray]:
    """
    分解 R = Rz(q5) Ry(q6) Rx(q7)。

    返回一个或两个欧拉分支。
    """

    s6 = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    q6_a = np.arcsin(s6)
    c6_a = cos(float(q6_a))

    solutions: list[FloatArray] = []

    if abs(c6_a) < singular_eps:
        # 万向节锁：q5 与 q7 耦合。固定 q7=0。
        q5 = atan2(-rotation[0, 1], rotation[1, 1])
        q7 = 0.0
        solutions.append(wrap_joints(np.array([q5, float(q6_a), q7], dtype=np.float64)))
        return solutions

    q5_a = atan2(rotation[1, 0], rotation[0, 0])
    q7_a = atan2(rotation[2, 1], rotation[2, 2])
    solutions.append(wrap_joints(np.array([q5_a, float(q6_a), q7_a], dtype=np.float64)))

    # Tait-Bryan 的第二分支。
    q6_b = pi - float(q6_a) if q6_a >= 0.0 else -pi - float(q6_a)
    q5_b = q5_a + pi
    q7_b = q7_a + pi
    solutions.append(wrap_joints(np.array([q5_b, q6_b, q7_b], dtype=np.float64)))

    return solutions


def make_swivel_basis(sw_direction: FloatArray) -> tuple[FloatArray, FloatArray]:
    """
    构造与肩腕连线垂直的稳定二维基。

    psi=0 时，肘部位于由 sw_direction 和参考轴形成的基准平面。
    """

    candidates = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    reference = min(candidates, key=lambda axis: abs(float(np.dot(axis, sw_direction))))

    basis_u = normalize(np.cross(sw_direction, reference))
    basis_v = normalize(np.cross(sw_direction, basis_u))
    return basis_u, basis_v


def elbow_position_from_swivel(
    shoulder: FloatArray,
    wrist: FloatArray,
    upper_length: float,
    forearm_length: float,
    psi: float,
    *,
    reach_tolerance: float = 1e-9,
) -> FloatArray:
    """由肩球心、腕球心和旋臂角计算肘部参考点。"""

    sw = wrist - shoulder
    distance = float(np.linalg.norm(sw))

    minimum = abs(upper_length - forearm_length)
    maximum = upper_length + forearm_length
    if distance < minimum - reach_tolerance or distance > maximum + reach_tolerance:
        raise ValueError(
            "目标腕心超出机械臂位置可达范围：" f"|SW|={distance:.6f} m，允许范围=[{minimum:.6f}, {maximum:.6f}] m。"
        )
    if distance < reach_tolerance:
        raise ValueError("肩球心与腕球心重合，旋臂角 psi 无法定义。")

    sw_direction = sw / distance

    center_distance = (upper_length * upper_length - forearm_length * forearm_length + distance * distance) / (
        2.0 * distance
    )

    circle_radius_sq = upper_length * upper_length - center_distance * center_distance
    circle_radius = float(np.sqrt(max(0.0, circle_radius_sq)))

    circle_center = shoulder + center_distance * sw_direction
    basis_u, basis_v = make_swivel_basis(sw_direction)

    radial = cos(psi) * basis_u + sin(psi) * basis_v
    return circle_center + circle_radius * radial


def solve_elbow_angle(
    rotation_03: FloatArray,
    elbow_to_wrist_base: FloatArray,
    geometry: RobotGeometry,
) -> float:
    """
    求解 R03 * Ry(q4) * forearm_local = elbow_to_wrist_base。
    """

    target_local_3 = rotation_03.T @ elbow_to_wrist_base
    target_angle = atan2(float(target_local_3[0]), float(target_local_3[2]))

    source = geometry.elbow_to_wrist_local
    source_angle = atan2(float(source[0]), float(source[2]))

    return wrap_to_pi(target_angle - source_angle)


def within_limits(
    joints: FloatArray,
    limits: JointLimits,
    *,
    tolerance: float = 1e-9,
) -> bool:
    return bool(np.all(joints >= limits.lower - tolerance) and np.all(joints <= limits.upper + tolerance))


def angular_distance(
    candidate: FloatArray,
    current: FloatArray,
    weights: FloatArray,
) -> float:
    delta = np.arctan2(np.sin(candidate - current), np.cos(candidate - current))
    return float(np.sum(weights * np.abs(delta)))


def deduplicate_solutions(
    solutions: list[IKSolution],
    *,
    joint_tolerance: float = 1e-7,
) -> list[IKSolution]:
    unique: list[IKSolution] = []
    for solution in solutions:
        duplicate = False
        for existing in unique:
            delta = np.arctan2(
                np.sin(solution.joints - existing.joints),
                np.cos(solution.joints - existing.joints),
            )
            if float(np.max(np.abs(delta))) <= joint_tolerance:
                duplicate = True
                break
        if not duplicate:
            unique.append(solution)
    return unique


def swivel_angle_from_joints(
    joints: Iterable[float],
    geometry: RobotGeometry = RobotGeometry(),
) -> float:
    """
    根据一组关节角计算与 inverse_kinematics() 使用同一定义的旋臂角 psi。

    该函数适合：
    - 从当前机械臂关节角获得连续的 psi 初值；
    - 验证 FK -> psi -> IK 闭环；
    - 上层轨迹规划时维持冗余参数连续。

    返回
    ----
    psi:
        单位 rad，范围 [-pi, pi]。
    """

    q = np.asarray(tuple(joints), dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"joints 必须是 shape=(7,)，实际 shape={q.shape}。")

    q1, q2, q3, q4 = q[:4]

    rotation_03 = rot_z(q1) @ rot_y(q2) @ rot_z(q3)
    shoulder = geometry.shoulder_position_base
    elbow = shoulder + rotation_03 @ geometry.shoulder_to_elbow_local

    rotation_04 = rotation_03 @ rot_y(q4)
    wrist = elbow + rotation_04 @ geometry.elbow_to_wrist_local

    sw = wrist - shoulder
    sw_direction = normalize(sw)

    upper_length = float(np.linalg.norm(geometry.shoulder_to_elbow_local))
    forearm_length = float(np.linalg.norm(geometry.elbow_to_wrist_local))
    distance = float(np.linalg.norm(sw))

    center_distance = (upper_length * upper_length - forearm_length * forearm_length + distance * distance) / (
        2.0 * distance
    )
    circle_center = shoulder + center_distance * sw_direction

    radial = elbow - circle_center
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-10:
        raise ValueError("当前关节位形接近肘部伸直/折叠奇异，psi 无法稳定计算。")
    radial = radial / radial_norm

    basis_u, basis_v = make_swivel_basis(sw_direction)
    return wrap_to_pi(atan2(float(np.dot(radial, basis_v)), float(np.dot(radial, basis_u))))


def inverse_kinematics(
    pose_target: FloatArray,
    psi: float,
    joint_current: FloatArray | None = None,
    *,
    geometry: RobotGeometry = RobotGeometry(),
    limits: JointLimits | None = None,
    position_tolerance: float = 1e-7,
    orientation_tolerance: float = 1e-7,
    joint_weights: FloatArray | None = None,
) -> IKResult:
    """
    求解给定 base -> TCP 位姿和旋臂角 psi 的七轴解析逆解。
    """

    pose = np.asarray(pose_target, dtype=np.float64)
    validate_pose(pose)

    if limits is None:
        limits = JointLimits.from_urdf()

    if joint_current is not None:
        current = np.asarray(joint_current, dtype=np.float64)
        if current.shape != (7,):
            raise ValueError(f"joint_current 必须是 shape=(7,)，实际 shape={current.shape}。")
    else:
        current = None

    if joint_weights is None:
        weights = np.ones(7, dtype=np.float64)
    else:
        weights = np.asarray(joint_weights, dtype=np.float64)
        if weights.shape != (7,):
            raise ValueError(f"joint_weights 必须是 shape=(7,)，实际 shape={weights.shape}。")
        if np.any(weights < 0.0):
            raise ValueError("joint_weights 不能包含负数。")

    target_rotation = pose[:3, :3]
    target_tcp = pose[:3, 3]

    # TCP 固定偏移沿 link7 的 +Z，因此先得到腕球心位置。
    wrist = target_tcp - target_rotation @ geometry.tcp_offset_link7
    shoulder = geometry.shoulder_position_base

    upper_local = geometry.shoulder_to_elbow_local
    forearm_local = geometry.elbow_to_wrist_local
    upper_length = float(np.linalg.norm(upper_local))
    forearm_length = float(np.linalg.norm(forearm_local))

    elbow = elbow_position_from_swivel(
        shoulder=shoulder,
        wrist=wrist,
        upper_length=upper_length,
        forearm_length=forearm_length,
        psi=float(psi),
    )

    shoulder_to_elbow = elbow - shoulder
    elbow_to_wrist = wrist - elbow
    se_direction = normalize(shoulder_to_elbow)

    plane_normal_raw = np.cross(shoulder_to_elbow, elbow_to_wrist)
    if float(np.linalg.norm(plane_normal_raw)) < 1e-10:
        raise ValueError("目标处于伸直或折叠肘奇异位形，肩部绕上臂轴的方向无法唯一确定。")
    plane_normal = normalize(plane_normal_raw)

    upper_direction_local = normalize(upper_local)
    elbow_axis_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_third = normalize(np.cross(upper_direction_local, elbow_axis_local))
    local_basis = np.column_stack((upper_direction_local, elbow_axis_local, local_third))

    candidates: list[IKSolution] = []

    # 肘轴法向存在正负两个解析分支。
    for elbow_axis_sign in (1.0, -1.0):
        elbow_axis_base = elbow_axis_sign * plane_normal
        world_third = normalize(np.cross(se_direction, elbow_axis_base))
        world_basis = np.column_stack((se_direction, elbow_axis_base, world_third))

        rotation_03_nominal = world_basis @ local_basis.T

        for shoulder_joints in decompose_zyz(rotation_03_nominal):
            q1, q2, q3 = shoulder_joints
            rotation_03 = rot_z(q1) @ rot_y(q2) @ rot_z(q3)

            q4 = solve_elbow_angle(rotation_03, elbow_to_wrist, geometry)
            rotation_04 = rotation_03 @ rot_y(q4)

            wrist_rotation = rotation_04.T @ target_rotation
            for wrist_joints in decompose_zyx_wrist(wrist_rotation):
                q5, q6, q7 = wrist_joints
                joints = wrap_joints(np.array([q1, q2, q3, q4, q5, q6, q7], dtype=np.float64))

                if not within_limits(joints, limits):
                    continue

                reconstructed = forward_kinematics(joints, geometry)
                position_error = float(np.linalg.norm(reconstructed[:3, 3] - target_tcp))
                orientation_error = skew_rotation_error(reconstructed[:3, :3], target_rotation)

                if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
                    candidates.append(
                        IKSolution(
                            joints=joints,
                            position_error=position_error,
                            orientation_error=orientation_error,
                        )
                    )

    candidates = deduplicate_solutions(candidates)

    if not candidates:
        empty = np.empty((0, 7), dtype=np.float64)
        return IKResult(
            solutions=empty,
            selected=None,
            position_errors=np.empty((0,), dtype=np.float64),
            orientation_errors=np.empty((0,), dtype=np.float64),
        )

    solutions_array = np.vstack([item.joints for item in candidates])
    position_errors = np.array(
        [item.position_error for item in candidates],
        dtype=np.float64,
    )
    orientation_errors = np.array(
        [item.orientation_error for item in candidates],
        dtype=np.float64,
    )

    if current is None:
        selected = solutions_array[0].copy()
    else:
        scores = np.array(
            [angular_distance(solution, current, weights) for solution in solutions_array],
            dtype=np.float64,
        )
        selected = solutions_array[int(np.argmin(scores))].copy()

    return IKResult(
        solutions=solutions_array,
        selected=selected,
        position_errors=position_errors,
        orientation_errors=orientation_errors,
    )


def _demo() -> None:
    """
    自检示例：先用一组已知关节角生成目标位姿，再执行逆解。
    """

    np.set_printoptions(precision=8, suppress=True)

    joint_reference = np.deg2rad(np.array([20.0, -30.0, 35.0, 45.0, 25.0, -20.0, 15.0]))
    pose_target = forward_kinematics(joint_reference)

    # 从当前关节角计算与本解析器一致的 psi。
    found_psi = swivel_angle_from_joints(joint_reference)
    found = inverse_kinematics(
        pose_target=pose_target,
        psi=found_psi,
        joint_current=joint_reference,
    )

    print("目标 base -> TCP 位姿：")
    print(pose_target)
    print()

    if found.selected is None:
        print("示例未找到满足关节限位的解析解。")
        return

    print(f"使用 psi = {found_psi:.8f} rad")
    print(f"有效解数量 = {len(found.solutions)}")
    print("全部解，单位 deg：")
    print(np.rad2deg(found.solutions))
    print()
    print("选择解，单位 rad：")
    print(found.selected)
    print("选择解，单位 deg：")
    print(np.rad2deg(found.selected))
    print("位置回代误差，m：")
    print(found.position_errors)
    print("姿态回代误差，rad：")
    print(found.orientation_errors)


if __name__ == "__main__":
    _demo()
