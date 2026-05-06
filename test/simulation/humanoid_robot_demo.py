from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from src.robotics.kinematic_models import ArmMountState
from src.simulation import ArmSimulationBinding, ArmSimulationModel, ChainSnapshot, JointUiSpec, KinematicsSimulationWidget, SpatialArmKinematics
from src.simulation.protocols import JointAxisGlyph
from src.utils.datas import Axis, Color, Degree, Point, Quaternion, Transform, Translation, Vector


def _lift_end_transform(lift_binding: ArmSimulationBinding) -> Transform:
    return lift_binding.base_transform @ lift_binding.arm_model.solve_tcp(lift_binding.arm_state.joint_positions)


def _build_arm_chain(chain_name: str, shoulder_offset_x: float, color: Color, side: str, lift_binding: ArmSimulationBinding) -> ArmSimulationBinding:
    sign = 1.0 if side == "left" else -1.0
    kinematics = SpatialArmKinematics(
        name=chain_name,
        link_vectors=(
            (0.26, 0.0, 0.0),
            (0.24, 0.0, 0.0),
            (0.18, 0.0, 0.0),
            (0.10, 0.0, 0.0),
            (0.08, 0.0, 0.0),
            (0.06, 0.0, 0.0),
        ),
        joint_axes_local=(
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        joint_limits=((Degree(-170), Degree(170)), (Degree(-120), Degree(120)), (Degree(-170), Degree(170)), (Degree(-185), Degree(185)), (Degree(-120), Degree(120)), (Degree(-360), Degree(360))),
    )
    joint_positions = (25.0 * sign, -25.0, 35.0, 0.0, 20.0, 0.0)

    return ArmSimulationBinding(
        chain_name=chain_name,
        arm_state=ArmMountState(joint_positions=joint_positions),
        arm_model=kinematics,
        base_transform_solver=lambda: _lift_end_transform(lift_binding) @ Transform(translation=Translation(shoulder_offset_x, 0.0, 0.0)),
        joint_ui=(
            JointUiSpec("j1", Degree(-170), Degree(170), Degree(joint_positions[0])),
            JointUiSpec("j2", Degree(-120), Degree(120), Degree(joint_positions[1])),
            JointUiSpec("j3", Degree(-170), Degree(170), Degree(joint_positions[2])),
            JointUiSpec("j4", Degree(-185), Degree(185), Degree(joint_positions[3])),
            JointUiSpec("j5", Degree(-120), Degree(120), Degree(joint_positions[4])),
            JointUiSpec("j6", Degree(-360), Degree(360), Degree(joint_positions[5])),
        ),
        color=color,
    )


@dataclass(slots=True)
class ParallelPalmKinematics:
    """三指并联手掌运动学（3DOF）。"""

    name: str
    finger_offsets: tuple[tuple[float, float, float], ...]
    finger_length: float
    joint_limits: tuple[tuple[float, float], ...]

    def _clamp(self, joints: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(min(lim[1], max(lim[0], value)) for value, lim in zip(joints, self.joint_limits, strict=True))

    def solve_tcp(self, joint_positions: tuple[float, ...]) -> Transform:
        joints = self._clamp(joint_positions)
        tips = self._finger_tips(joints)
        cx = sum(p[0] for p in tips) / len(tips)
        cy = sum(p[1] for p in tips) / len(tips)
        cz = sum(p[2] for p in tips) / len(tips)
        return Transform(translation=Translation(cx, cy, cz), rotation=Quaternion.Identity())

    def solve_joints(self, target_tcp_pose: Transform, reference_joints: tuple[float, ...]) -> tuple[float, ...]:
        del target_tcp_pose
        return self._clamp(reference_joints)

    def forward_joint_axes_local(self, joint_positions: tuple[float, ...]) -> tuple[JointAxisGlyph, ...]:
        del joint_positions
        return tuple(
            JointAxisGlyph(
                axis=Axis(origin=Point(*offset), z_axis=Vector.ZAxis()),
                label=f"f{i+1}",
            )
            for i, offset in enumerate(self.finger_offsets)
        )

    def _finger_tips(self, joints: tuple[float, ...]) -> tuple[tuple[float, float, float], ...]:
        tips: list[tuple[float, float, float]] = []
        for (ox, oy, oz), angle_deg in zip(self.finger_offsets, joints, strict=True):
            rad = angle_deg * 3.141592653589793 / 180.0
            tips.append((ox + self.finger_length * np.cos(rad), oy + self.finger_length * np.sin(rad), oz))
        return tuple(tips)

    def finger_segments_local(self, joints: tuple[float, ...]) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
        joints_c = self._clamp(joints)
        tips = self._finger_tips(joints_c)
        return tuple((offset, tip) for offset, tip in zip(self.finger_offsets, tips, strict=True))


def _build_palm_chain(chain_name: str, color: Color, arm_binding: ArmSimulationBinding) -> ArmSimulationBinding:
    palm = ParallelPalmKinematics(
        name=chain_name,
        finger_offsets=((0.00, 0.03, 0.00), (0.00, 0.00, 0.00), (0.00, -0.03, 0.00)),
        finger_length=0.08,
        joint_limits=((-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0)),
    )
    joint_positions = (15.0, 0.0, -15.0)

    def palm_base_solver() -> Transform:
        base = arm_binding.base_transform_solver() if arm_binding.base_transform_solver is not None else arm_binding.base_transform
        tcp = arm_binding.arm_model.solve_tcp(arm_binding.arm_state.joint_positions)
        return base @ tcp

    def palm_points_solver(joints: tuple[float, ...]) -> tuple[tuple[float, float, float], ...]:
        base = palm_base_solver().translation
        points: list[tuple[float, float, float]] = [(base.x, base.y, base.z)]
        for start, end in palm.finger_segments_local(joints):
            points.append((base.x + start[0], base.y + start[1], base.z + start[2]))
            points.append((base.x + end[0], base.y + end[1], base.z + end[2]))
        return tuple(points)

    return ArmSimulationBinding(
        chain_name=chain_name,
        arm_state=ArmMountState(joint_positions=joint_positions),
        arm_model=palm,
        base_transform_solver=palm_base_solver,
        joint_ui=(
            JointUiSpec("finger_1", Degree(-45.0), Degree(45.0), Degree(joint_positions[0])),
            JointUiSpec("finger_2", Degree(-45.0), Degree(45.0), Degree(joint_positions[1])),
            JointUiSpec("finger_3", Degree(-45.0), Degree(45.0), Degree(joint_positions[2])),
        ),
        color=color,
        link_point_solver=palm_points_solver,
    )


def _build_lift_chain() -> ArmSimulationBinding:
    lift = SpatialArmKinematics(
        name="lift_pitch",
        link_vectors=((0.0, 0.0, 1.15), (0.0, 0.0, 0.0)),
        joint_axes_local=((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        joint_limits=((Degree(-20.0), Degree(45.0)), (0.0, 0.35)),
        joint_types=("revolute", "prismatic"),
    )
    joint_positions = (8.0, 0.12)

    return ArmSimulationBinding(
        chain_name="lift_pitch",
        arm_state=ArmMountState(joint_positions=joint_positions),
        arm_model=lift,
        base_transform=Transform(translation=Translation(0.0, 0.0, 0.05)),
        joint_ui=(
            JointUiSpec("lift_pitch", Degree(-20.0), Degree(45.0), Degree(joint_positions[0])),
            JointUiSpec("lift_z", 0.0, 0.35, joint_positions[1]),
        ),
        color=Color.from_hex("#8d99ae"),
    )


def _build_agv_outline() -> tuple[ChainSnapshot, ...]:
    agv = ChainSnapshot(
        chain_name="agv_base",
        points=(Point(-0.38, -0.26, 0.0), Point(0.38, -0.26, 0.0), Point(0.38, 0.26, 0.0), Point(-0.38, 0.26, 0.0), Point(-0.38, -0.26, 0.0)),
        color=Color.from_hex("#6c757d"),
    )
    return (agv,)


def build_robot_model() -> ArmSimulationModel:
    lift = _build_lift_chain()
    left_arm = _build_arm_chain("left_arm_6dof", -0.22, Color.from_hex("#1d3557"), "left", lift)
    right_arm = _build_arm_chain("right_arm_6dof", 0.22, Color.from_hex("#457b9d"), "right", lift)
    left_palm = _build_palm_chain("left_palm_3dof", Color.from_hex("#e76f51"), left_arm)
    right_palm = _build_palm_chain("right_palm_3dof", Color.from_hex("#f4a261"), right_arm)

    return ArmSimulationModel(
        bindings={
            lift.chain_name: lift,
            left_arm.chain_name: left_arm,
            right_arm.chain_name: right_arm,
            left_palm.chain_name: left_palm,
            right_palm.chain_name: right_palm,
        },
        static_snapshots=_build_agv_outline(),
    )


def main() -> int:
    pyside6_plugin_path = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6", "plugins", "platforms")
    if os.path.isdir(pyside6_plugin_path):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", pyside6_plugin_path)

    app = QApplication(sys.argv)
    widget = KinematicsSimulationWidget(model=build_robot_model())
    widget.setWindowTitle("AGV + Lift + DualArm + Palm(3DOF) 3D Simulator")
    widget.resize(1460, 920)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
