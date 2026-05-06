from __future__ import annotations

from dataclasses import dataclass
from math import atan2

import numpy as np

from src.robotics.kinematic_models import ArmMountState
from src.simulation.protocols import ArmSimulationBinding, ChainSnapshot, JointAxisGlyph
from src.utils.datas import Axis, Degree, Point, Quaternion, Radian, Transform, Translation, Vector


def _as_float_joint(value: Degree | Radian | float) -> float:
    """将角度/位移类型统一转换为 `float`"""

    if isinstance(value, Degree):
        return value.value
    if isinstance(value, Radian):
        return Degree.from_radians(value.value).value
    return float(value)


# region 三维运动学实现


@dataclass(slots=True)
class SpatialArmKinematics:
    """单自由度转动/滑动副串联三维运动学

    Parameters
    ----------
    name:
        机构模型名称
    link_vectors:
        每级连杆向量（局部坐标系），形状语义 `(N, 3)`，单位 米
    joint_axes_local:
        每级关节局部轴向，形状语义 `(N, 3)`
    joint_limits:
        每级关节限制，形状语义 `(N, 2)`
    joint_types:
        关节类型序列，元素为 `revolute` 或 `prismatic`，长度为 `N`

    Notes
    -----
    - `revolute` 关节值单位：度
    - `prismatic` 关节值单位：米
    """

    name: str
    """机构模型名称"""

    link_vectors: tuple[tuple[float, float, float], ...]
    """连杆向量序列"""

    joint_axes_local: tuple[tuple[float, float, float], ...]
    """局部关节轴向序列"""

    joint_limits: tuple[tuple[Degree | float, Degree | float], ...]
    """关节约束区间序列"""

    joint_types: tuple[str, ...] | None = None
    """关节类型序列"""

    def __post_init__(self) -> None:
        """构造后一致性校验"""

        count = len(self.link_vectors)
        if count == 0:
            raise ValueError("至少需要 1 个关节")
        if len(self.joint_axes_local) != count or len(self.joint_limits) != count:
            raise ValueError("link_vectors / joint_axes_local / joint_limits 长度必须一致")
        if self.joint_types is None:
            self.joint_types = tuple("revolute" for _ in range(count))
        if len(self.joint_types) != count:
            raise ValueError("joint_types 长度必须与关节数量一致")
        for joint_type in self.joint_types:
            if joint_type not in {"revolute", "prismatic"}:
                raise ValueError(f"不支持的关节类型：{joint_type}")

    def _clamp(self, joint_positions: tuple[float, ...]) -> tuple[float, ...]:
        """按关节限制裁剪输入关节值"""

        if len(joint_positions) != len(self.link_vectors):
            raise ValueError(f"关节数量不匹配：expected={len(self.link_vectors)}, actual={len(joint_positions)}")
        values: list[float] = []
        for value, limit in zip(joint_positions, self.joint_limits, strict=True):
            low = float(limit[0].value) if isinstance(limit[0], Degree) else float(limit[0])
            high = float(limit[1].value) if isinstance(limit[1], Degree) else float(limit[1])
            values.append(min(high, max(low, value)))
        return tuple(values)

    def _forward_kinematics(
        self, joint_positions: tuple[float, ...]
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """计算关节原点、轴向和节点轨迹

        Returns
        -------
        origins_world:
            每级关节原点，形状语义 `(N, 3)`
        axes_world:
            每级关节轴向，形状语义 `(N, 3)`
        points_world:
            链节点坐标，形状语义 `(N+1, 3)`
        """

        joints = self._clamp(joint_positions)
        pos = Point.Zero()
        rot = Quaternion.Identity()

        origins_world: list[np.ndarray] = []
        axes_world: list[np.ndarray] = []
        points_world: list[np.ndarray] = [np.array(pos.to_tuple(), dtype=np.float64)]

        for joint_value, axis_local, link_local, joint_type in zip(
            joints, self.joint_axes_local, self.link_vectors, self.joint_types, strict=True
        ):
            axis_local_vec = Vector.from_list(axis_local).normalized()
            axis_world_vec = axis_local_vec.transformed(rot).normalized()

            origins_world.append(np.array(pos.to_tuple(), dtype=np.float64))
            axes_world.append(axis_world_vec.as_array())

            if joint_type == "revolute":
                rot = rot * Quaternion.from_axis_angle(axis_local_vec, joint_value)
            else:
                # prismatic：沿当前局部轴方向平移（经当前姿态映射到世界）
                delta_world = axis_world_vec * joint_value
                pos = pos + delta_world

            link_world = Vector.from_list(link_local).transformed(rot)
            pos = pos + link_world
            points_world.append(np.array(pos.to_tuple(), dtype=np.float64))

        return origins_world, axes_world, np.array(points_world, dtype=np.float64)

    def solve_tcp(self, joint_positions: tuple[float, ...]) -> Transform:
        """正解：关节值 -> TCP 位姿"""

        _, _, points = self._forward_kinematics(joint_positions)
        x, y, z = points[-1]
        return Transform(translation=Translation(float(x), float(y), float(z)), rotation=Quaternion.Identity())

    def solve_joints(
        self, target_tcp_pose: Transform, reference_joints: tuple[Degree | Radian | float, ...]
    ) -> tuple[Degree | float, ...]:
        """逆解：目标位姿 -> 关节值（CCD 迭代）"""

        raw_reference: list[float] = []
        for value in reference_joints:
            raw_reference.append(_as_float_joint(value))
        joints = list(self._clamp(tuple(raw_reference)))
        target = np.array(target_tcp_pose.translation.to_tuple(), dtype=np.float64)

        for _ in range(120):
            origins, axes, points = self._forward_kinematics(tuple(joints))
            end = points[-1]
            if float(np.linalg.norm(end - target)) < 2e-3:
                return tuple(joints)

            for idx in range(len(joints) - 1, -1, -1):
                origin = origins[idx]
                axis_np = axes[idx]
                axis_vec = Vector.from_array(axis_np).normalized()
                end_vec = end - origin
                target_vec = target - origin

                if self.joint_types[idx] == "revolute":
                    axis = axis_vec.as_array()
                    end_p = end_vec - axis * float(np.dot(end_vec, axis))
                    target_p = target_vec - axis * float(np.dot(target_vec, axis))
                    if float(np.linalg.norm(end_p)) < 1e-12 or float(np.linalg.norm(target_p)) < 1e-12:
                        continue

                    end_u = Vector.from_array(end_p).normalized().as_array()
                    target_u = Vector.from_array(target_p).normalized().as_array()
                    cross = np.cross(end_u, target_u)
                    dot = float(np.clip(np.dot(end_u, target_u), -1.0, 1.0))
                    delta_rad = atan2(float(np.dot(axis, cross)), dot)
                    delta_deg = Degree.from_radians(Radian.from_radians(delta_rad).value).value
                    joints[idx] += delta_deg
                else:
                    axis = axis_vec.as_array()
                    delta_linear = float(np.dot(target_vec - end_vec, axis))
                    joints[idx] += delta_linear

                lim = self.joint_limits[idx]
                joints[idx] = min(lim[1], max(lim[0], joints[idx]))
                _, _, points = self._forward_kinematics(tuple(joints))
                end = points[-1]

        output: list[Degree | float] = []
        for value, joint_type in zip(joints, self.joint_types, strict=True):
            if joint_type == "revolute":
                output.append(Degree(value))
            else:
                output.append(value)
        return tuple(output)

    def forward_points_local(self, joint_positions: tuple[float, ...]) -> tuple[tuple[float, float, float], ...]:
        """返回链节点局部坐标序列（用于渲染）"""

        _, _, points = self._forward_kinematics(joint_positions)
        return tuple((float(x), float(y), float(z)) for x, y, z in points)

    def forward_joint_axes_local(self, joint_positions: tuple[float, ...]) -> tuple[JointAxisGlyph, ...]:
        """返回关节轴可视化信息"""

        origins, axes, _ = self._forward_kinematics(joint_positions)
        output: list[JointAxisGlyph] = []
        for idx, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
            output.append(
                JointAxisGlyph(
                    axis=Axis(
                        origin=Point(float(origin[0]), float(origin[1]), float(origin[2])),
                        z_axis=Vector(float(axis[0]), float(axis[1]), float(axis[2])),
                    ),
                    label=f"j{idx+1}",
                )
            )
        return tuple(output)


# endregion


# region 仿真模型


@dataclass(slots=True)
class ArmSimulationModel:
    """仿真模型容器"""

    bindings: dict[str, ArmSimulationBinding]
    """可交互链绑定字典"""

    static_snapshots: tuple[ChainSnapshot, ...] = ()
    """静态背景快照（如 AGV 底盘轮廓）"""

    def chain_names(self) -> tuple[str, ...]:
        """返回所有链名称"""

        return tuple(self.bindings.keys())

    def get_binding(self, chain_name: str) -> ArmSimulationBinding:
        """按名称获取链绑定"""

        if chain_name not in self.bindings:
            raise KeyError(f"未知链名称：{chain_name}")
        return self.bindings[chain_name]

    def set_joint_positions(self, chain_name: str, joint_positions: tuple[Degree | Radian | float, ...]) -> None:
        """更新指定链的关节值"""

        binding = self.get_binding(chain_name)
        binding.arm_state = ArmMountState(
            lift_end_to_shoulder=binding.arm_state.lift_end_to_shoulder,
            joint_positions=tuple(_as_float_joint(value) for value in joint_positions),
        )

    def _binding_base_transform(self, binding: ArmSimulationBinding) -> Transform:
        """解析链的当前基座位姿"""

        if binding.base_transform_solver is not None:
            return binding.base_transform_solver()
        return binding.base_transform

    def solve_chain_ik(
        self, chain_name: str, target_world_xyz: tuple[float, float, float]
    ) -> tuple[Degree | float, ...]:
        """对指定链执行 IK，并写回最新关节值"""

        binding = self.get_binding(chain_name)
        base = self._binding_base_transform(binding)
        target_world = Transform(translation=Translation(*target_world_xyz), rotation=Quaternion.Identity())
        target_local = Transform.from_SE3(np.linalg.inv(base.as_SE3()) @ target_world.as_SE3())
        result = binding.arm_model.solve_joints(target_local, binding.arm_state.joint_positions)
        self.set_joint_positions(chain_name, result)
        return result

    def _world_points(self, binding: ArmSimulationBinding) -> tuple[Point, ...]:
        """计算链在世界坐标系下的绘制节点"""

        if binding.link_point_solver is not None:
            return binding.link_point_solver(binding.arm_state.joint_positions)

        base = self._binding_base_transform(binding)
        model = binding.arm_model
        local_points: tuple[tuple[float, float, float], ...] = ()
        if hasattr(model, "forward_points_local"):
            local_points = model.forward_points_local(binding.arm_state.joint_positions)  # type: ignore[attr-defined]

        if not local_points:
            local_tcp = model.solve_tcp(binding.arm_state.joint_positions)
            world_tcp = base @ local_tcp
            b = base.translation
            t = world_tcp.translation
            return (Point(b.x, b.y, b.z), Point(t.x, t.y, t.z))

        base_t = base.translation
        return tuple(Point(x + base_t.x, y + base_t.y, z + base_t.z) for x, y, z in local_points)

    def _world_axes(self, binding: ArmSimulationBinding) -> tuple[JointAxisGlyph, ...]:
        """计算关节轴在世界坐标系下的箭头信息"""

        if binding.joint_axis_solver is not None:
            return binding.joint_axis_solver(binding.arm_state.joint_positions)

        model = binding.arm_model
        if not hasattr(model, "forward_joint_axes_local"):
            return ()

        base = self._binding_base_transform(binding)
        base_t = base.translation
        local_axes = model.forward_joint_axes_local(binding.arm_state.joint_positions)  # type: ignore[attr-defined]
        output: list[JointAxisGlyph] = []
        for glyph in local_axes:
            ox, oy, oz = glyph.axis.origin.to_tuple()
            output.append(
                JointAxisGlyph(
                    axis=Axis(
                        origin=Point(ox + base_t.x, oy + base_t.y, oz + base_t.z),
                        z_axis=glyph.axis.z_axis,
                    ),
                    label=glyph.label,
                )
            )
        return tuple(output)

    def snapshots(self) -> tuple[ChainSnapshot, ...]:
        """生成当前帧的动态与静态快照"""

        dynamic: list[ChainSnapshot] = []
        for binding in self.bindings.values():
            dynamic.append(
                ChainSnapshot(
                    chain_name=binding.chain_name,
                    points=self._world_points(binding),
                    color=binding.color,
                    joint_axes=self._world_axes(binding),
                )
            )
        return tuple(dynamic) + self.static_snapshots


# endregion
