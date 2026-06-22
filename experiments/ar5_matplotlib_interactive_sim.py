#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_PACKAGE_ROOT = PROJECT_ROOT / "sdk"
SDK_ROOT = SDK_PACKAGE_ROOT / "xcoresdk"
DEFAULT_URDF_PATH = Path(
    r"C:\Project Documents\鼎泰项目\珞石AR5-5LR\AR5-5_07R-W4C4A2-S1_description\urdf\AR5-5_07R-W4C4A2-S1.urdf"
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SDK_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_PACKAGE_ROOT))

from src.robotics.urdf_interface import UrdfConverter, UrdfModel  # noqa: E402

try:
    from sdk.xcoresdk import xCoreSDK_python as IMPORTED_XCORE_SDK  # type: ignore[attr-defined]  # noqa: E402
except Exception:
    IMPORTED_XCORE_SDK = None


# region 数据结构


@dataclass(frozen=True, slots=True)
class JointSpec:
    """单个可动关节的静态定义。"""

    name: str
    origin_xyz_m: np.ndarray
    origin_rpy_rad: np.ndarray
    axis_xyz: np.ndarray
    lower_rad: float
    upper_rad: float


@dataclass(slots=True)
class RobotState:
    """交互式仿真状态。"""

    joint_values_rad: np.ndarray
    target_xyz_m: np.ndarray
    target_rpy_rad: np.ndarray


@dataclass(frozen=True, slots=True)
class IKSolveResult:
    """逆解结果。"""

    success: bool
    method: str
    message: str
    joint_values_rad: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class KeyboardBinding:
    """键盘映射说明。"""

    key: str
    description: str


# endregion


# region URDF 解析与变换


def _build_joint_specs(model: UrdfModel) -> list[JointSpec]:
    """从 URDF 中提取按定义顺序排列的转动关节。"""

    revolute_joints = [
        joint for joint in model.joints if joint.joint_type == "revolute"
    ]
    if len(revolute_joints) != 7:
        raise ValueError(
            f"期望 AR5 具有 7 个 revolute 关节，当前为 {len(revolute_joints)}"
        )

    specs: list[JointSpec] = []
    for joint in revolute_joints:
        if (
            joint.limit is None
            or joint.limit.lower is None
            or joint.limit.upper is None
        ):
            raise ValueError(f"关节 {joint.name} 缺少完整的 limit 上下限")
        specs.append(
            JointSpec(
                name=joint.name,
                origin_xyz_m=np.asarray(joint.origin.xyz, dtype=np.float64),
                origin_rpy_rad=np.asarray(joint.origin.rpy, dtype=np.float64),
                axis_xyz=_normalize_vector(np.asarray(joint.axis, dtype=np.float64)),
                lower_rad=float(joint.limit.lower),
                upper_rad=float(joint.limit.upper),
            )
        )
    return specs


def _find_tcp_offset_m(model: UrdfModel) -> np.ndarray:
    """读取末端固定 TCP 偏移。"""

    for joint in model.joints:
        if joint.name.endswith("_tcp_joint") and joint.joint_type == "fixed":
            return np.asarray(joint.origin.xyz, dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """返回单位化向量。"""

    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("收到零长度轴向向量，无法归一化")
    return vector / norm


def _make_transform(translation_m: np.ndarray, rpy_rad: np.ndarray) -> np.ndarray:
    """构造 4x4 齐次变换矩阵。"""

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_euler("xyz", rpy_rad, degrees=False).as_matrix()
    mat[:3, 3] = translation_m
    return mat


def _make_axis_rotation(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    """构造绕任意轴旋转的 4x4 变换矩阵。"""

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_rotvec(axis_xyz * angle_rad).as_matrix()
    return mat


def _transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    """应用齐次变换到点。"""

    hom = np.ones(4, dtype=np.float64)
    hom[:3] = point_xyz
    return (transform @ hom)[:3]


def _compute_robot_geometry(
    joint_specs: list[JointSpec],
    tcp_offset_m: np.ndarray,
    state: RobotState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算当前所有关节原点、旋转轴方向和折线骨架点。"""

    base_tf = np.eye(4, dtype=np.float64)
    joint_origins_world: list[np.ndarray] = [base_tf[:3, 3].copy()]
    joint_axes_world: list[np.ndarray] = []
    current_tf = base_tf

    for index, spec in enumerate(joint_specs):
        origin_tf = _make_transform(spec.origin_xyz_m, spec.origin_rpy_rad)
        joint_frame_tf = current_tf @ origin_tf
        origin_world = joint_frame_tf[:3, 3].copy()
        axis_world = joint_frame_tf[:3, :3] @ spec.axis_xyz

        joint_origins_world.append(origin_world)
        joint_axes_world.append(axis_world)

        joint_motion_tf = _make_axis_rotation(
            spec.axis_xyz, float(state.joint_values_rad[index])
        )
        current_tf = joint_frame_tf @ joint_motion_tf

    tcp_world = _transform_point(current_tf, tcp_offset_m)
    skeleton_points_world = np.vstack([joint_origins_world, tcp_world[None, :]])
    tcp_tf = current_tf @ _make_transform(tcp_offset_m, np.zeros(3, dtype=np.float64))
    return (
        np.vstack(joint_origins_world[1:]),
        np.vstack(joint_axes_world),
        skeleton_points_world,
        tcp_tf,
    )


def _estimate_plot_radius_m(
    joint_specs: list[JointSpec], tcp_offset_m: np.ndarray
) -> float:
    """估算绘图半径。"""

    reach = float(
        sum(np.linalg.norm(spec.origin_xyz_m) for spec in joint_specs)
        + np.linalg.norm(tcp_offset_m)
    )
    return max(0.5, reach * 1.4)


def _pose_error_vector(target_tf: np.ndarray, current_tf: np.ndarray) -> np.ndarray:
    """返回 6 维位姿误差，[dx, dy, dz, dRx, dRy, dRz]。"""

    position_error = target_tf[:3, 3] - current_tf[:3, 3]
    rotation_error = (
        R.from_matrix(target_tf[:3, :3]) * R.from_matrix(current_tf[:3, :3]).inv()
    ).as_rotvec()
    return np.concatenate([position_error, rotation_error])


def _build_target_transform(state: RobotState) -> np.ndarray:
    """根据目标 xyzrpy 生成目标位姿。"""

    return _make_transform(state.target_xyz_m, state.target_rpy_rad)


def _compute_pose_jacobian(
    joint_specs: list[JointSpec],
    tcp_offset_m: np.ndarray,
    joint_values_rad: np.ndarray,
    epsilon_rad: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """对当前关节值做数值差分，构建 6xN 雅可比。"""

    state = RobotState(
        joint_values_rad=joint_values_rad.copy(),
        target_xyz_m=np.zeros(3, dtype=np.float64),
        target_rpy_rad=np.zeros(3, dtype=np.float64),
    )
    _, _, _, tcp_tf = _compute_robot_geometry(joint_specs, tcp_offset_m, state)
    current_pose = np.concatenate(
        [tcp_tf[:3, 3], R.from_matrix(tcp_tf[:3, :3]).as_rotvec()]
    )

    jacobian = np.zeros((6, len(joint_specs)), dtype=np.float64)
    for index in range(len(joint_specs)):
        perturbed_joints = joint_values_rad.copy()
        perturbed_joints[index] += epsilon_rad
        perturbed_state = RobotState(
            joint_values_rad=perturbed_joints,
            target_xyz_m=np.zeros(3, dtype=np.float64),
            target_rpy_rad=np.zeros(3, dtype=np.float64),
        )
        _, _, _, perturbed_tf = _compute_robot_geometry(
            joint_specs, tcp_offset_m, perturbed_state
        )
        perturbed_pose = np.concatenate(
            [
                perturbed_tf[:3, 3],
                R.from_matrix(perturbed_tf[:3, :3]).as_rotvec(),
            ]
        )
        jacobian[:, index] = (perturbed_pose - current_pose) / epsilon_rad
    return jacobian, tcp_tf


def _make_cartesian_position_from_transform(
    xcore_sdk: Any,
    target_tf: np.ndarray,
) -> Any:
    """构造 SDK `CartesianPosition`。"""

    pose6 = list(target_tf[:3, 3]) + list(
        R.from_matrix(target_tf[:3, :3]).as_euler("xyz", degrees=False)
    )
    try:
        return xcore_sdk.CartesianPosition(pose6)
    except Exception:
        pose = xcore_sdk.CartesianPosition()
        pose.trans = [float(value) for value in pose6[:3]]
        pose.rpy = [float(value) for value in pose6[3:]]
        return pose


def _load_xcore_sdk_module() -> Any | None:
    """按当前 Python 版本尝试加载离线 xCoreSDK 扩展模块。"""

    suffix = f"cp{sys.version_info.major}{sys.version_info.minor}"
    candidates = sorted(SDK_ROOT.glob(f"xCoreSDK_python.{suffix}-win_amd64.pyd"))
    if not candidates:
        return None

    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(SDK_ROOT))
        except OSError:
            pass

    module_path = candidates[0]
    spec = importlib.util.spec_from_file_location("xCoreSDK_python", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules["xCoreSDK_python"] = module
    spec.loader.exec_module(module)
    return module


def _try_solve_with_xcore_sdk(target_tf: np.ndarray) -> IKSolveResult:
    """优先使用 xCoreSDK 的离线逆解。"""

    try:
        xcore_sdk = IMPORTED_XCORE_SDK
        if xcore_sdk is None:
            xcore_sdk = _load_xcore_sdk_module()
    except Exception as exc:
        return IKSolveResult(False, "xcore-sdk", f"SDK load failed: {exc}")
    if xcore_sdk is None:
        return IKSolveResult(
            False, "xcore-sdk", "SDK extension not available for current Python"
        )

    try:
        robot_model = None
        for robot_type_name in ("xMateErProRobot", "ArRobot"):
            robot_type = getattr(xcore_sdk, robot_type_name, None)
            if robot_type is None:
                continue
            try:
                robot_model = robot_type().model()
                break
            except Exception:
                continue
        if robot_model is None:
            return IKSolveResult(
                False, "xcore-sdk", "cannot create offline robot model"
            )
        toolset = xcore_sdk.Toolset()
        posture = _make_cartesian_position_from_transform(xcore_sdk, target_tf)
        ec: dict[str, object] = {}
        joint_values = np.asarray(
            robot_model.calcIk(posture, toolset, ec), dtype=np.float64
        )
        ec_value = ec.get("ec", 0)
        ec_code = int(ec_value) if isinstance(ec_value, int | float | str) else 0
        if ec_code != 0:
            return IKSolveResult(False, "xcore-sdk", f"calcIk ec={ec.get('ec')}")
        return IKSolveResult(True, "xcore-sdk", "calcIk succeeded", joint_values)
    except Exception as exc:
        return IKSolveResult(False, "xcore-sdk", f"calcIk failed: {exc}")


def _solve_with_newton(
    joint_specs: list[JointSpec],
    tcp_offset_m: np.ndarray,
    target_tf: np.ndarray,
    reference_joints_rad: np.ndarray,
) -> IKSolveResult:
    """使用阻尼牛顿法求解逆解。"""

    joints = reference_joints_rad.copy()
    limits_lower = np.array([spec.lower_rad for spec in joint_specs], dtype=np.float64)
    limits_upper = np.array([spec.upper_rad for spec in joint_specs], dtype=np.float64)
    damping = 1e-2
    max_iterations = 80
    position_tol_m = 5e-4
    rotation_tol_rad = math.radians(0.5)

    for iteration in range(max_iterations):
        jacobian, current_tf = _compute_pose_jacobian(joint_specs, tcp_offset_m, joints)
        error = _pose_error_vector(target_tf, current_tf)
        if (
            float(np.linalg.norm(error[:3])) <= position_tol_m
            and float(np.linalg.norm(error[3:])) <= rotation_tol_rad
        ):
            return IKSolveResult(
                True,
                "newton",
                f"converged in {iteration} iterations",
                joints.copy(),
            )

        jjt = jacobian @ jacobian.T
        step = jacobian.T @ np.linalg.solve(
            jjt + (damping**2) * np.eye(6, dtype=np.float64),
            error,
        )
        joints = np.clip(joints + step, limits_lower, limits_upper)

    return IKSolveResult(False, "newton", "failed to converge within 80 iterations")


def _solve_target_ik(
    joint_specs: list[JointSpec],
    tcp_offset_m: np.ndarray,
    state: RobotState,
) -> IKSolveResult:
    """先尝试 SDK 逆解，失败后回退牛顿法。"""

    target_tf = _build_target_transform(state)
    sdk_result = _try_solve_with_xcore_sdk(target_tf)
    if sdk_result.success and sdk_result.joint_values_rad is not None:
        return sdk_result
    newton_result = _solve_with_newton(
        joint_specs=joint_specs,
        tcp_offset_m=tcp_offset_m,
        target_tf=target_tf,
        reference_joints_rad=state.joint_values_rad,
    )
    if newton_result.success:
        return IKSolveResult(
            True,
            "newton",
            f"{newton_result.message}; sdk fallback reason: {sdk_result.message}",
            newton_result.joint_values_rad,
        )
    return IKSolveResult(
        False,
        "newton",
        f"sdk failed: {sdk_result.message}; newton failed: {newton_result.message}",
    )


# endregion


# region 交互绘制


class MatplotlibRobotSimulator:
    """基于 matplotlib 的 AR5 交互式骨架仿真。"""

    def __init__(
        self,
        urdf_path: Path,
        translation_step_mm: float,
        rotation_step_deg: float,
        joint_step_deg: float,
    ) -> None:
        self._urdf_path = urdf_path
        self._translation_step_m = float(translation_step_mm) / 1000.0
        self._rotation_step_rad = math.radians(float(rotation_step_deg))
        self._joint_step_rad = math.radians(float(joint_step_deg))

        model = UrdfConverter().from_file(urdf_path)
        self._joint_specs = _build_joint_specs(model)
        self._tcp_offset_m = _find_tcp_offset_m(model)
        self._plot_radius_m = _estimate_plot_radius_m(
            self._joint_specs, self._tcp_offset_m
        )
        self._axis_draw_length_m = max(0.05, self._plot_radius_m * 0.08)
        self._state = RobotState(
            joint_values_rad=np.array(
                [(spec.lower_rad + spec.upper_rad) * 0.5 for spec in self._joint_specs],
                dtype=np.float64,
            ),
            target_xyz_m=np.zeros(3, dtype=np.float64),
            target_rpy_rad=np.zeros(3, dtype=np.float64),
        )
        _, _, _, tcp_tf = _compute_robot_geometry(
            self._joint_specs, self._tcp_offset_m, self._state
        )
        self._state.target_xyz_m[:] = tcp_tf[:3, 3]
        self._state.target_rpy_rad[:] = R.from_matrix(tcp_tf[:3, :3]).as_euler(
            "xyz", degrees=False
        )
        self._bindings = self._build_bindings()
        self._figure = plt.figure(figsize=(12, 9))
        self._axes = cast(Axes3D, self._figure.add_subplot(111, projection="3d"))
        self._annotation_artists: list[Any] = []
        self._last_ik_result = IKSolveResult(
            success=True,
            method="init",
            message="target synced to current tcp",
        )
        self._figure.canvas.mpl_connect(
            "key_press_event",
            lambda event: self._on_key_press(cast(KeyEvent, event)),
        )
        self._draw()

    def show(self) -> None:
        """打开交互窗口。"""

        plt.show()

    def save_preview(self, output_path: Path) -> None:
        """保存当前视图。"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(output_path, dpi=160, bbox_inches="tight")

    def print_state(self) -> None:
        """打印当前状态，便于无窗口冒烟验证。"""

        target_xyz_mm = self._state.target_xyz_m * 1000.0
        target_rpy_deg = np.degrees(self._state.target_rpy_rad)
        joints_deg = np.degrees(self._state.joint_values_rad)
        print(f"URDF: {self._urdf_path}")
        print(f"target xyz(mm): {target_xyz_mm.round(3).tolist()}")
        print(f"target rpy(deg): {target_rpy_deg.round(3).tolist()}")
        print(f"joints(deg): {joints_deg.round(3).tolist()}")
        print(f"ik: {self._last_ik_result.method} | {self._last_ik_result.message}")

    def _build_bindings(self) -> tuple[KeyboardBinding, ...]:
        return (
            KeyboardBinding("q / a", "target X + / -"),
            KeyboardBinding("w / s", "target Y + / -"),
            KeyboardBinding("e / d", "target Z + / -"),
            KeyboardBinding("r / f", "target Roll + / -"),
            KeyboardBinding("t / g", "target Pitch + / -"),
            KeyboardBinding("y / h", "target Yaw + / -"),
            KeyboardBinding("u / j", "J1 + / -"),
            KeyboardBinding("i / k", "J2 + / -"),
            KeyboardBinding("o / l", "J3 + / -"),
            KeyboardBinding("p / ;", "J4 + / -"),
            KeyboardBinding("z / x", "J5 + / -"),
            KeyboardBinding("c / v", "J6 + / -"),
            KeyboardBinding("b / n", "J7 + / -"),
            KeyboardBinding("m", "solve IK to target"),
            KeyboardBinding(",", "sync target from current tcp"),
            KeyboardBinding("0", "reset joints and target"),
        )

    def _on_key_press(self, event: KeyEvent) -> None:
        key = "" if event.key is None else event.key.lower()
        if key == "":
            return

        handled = True
        if key == "q":
            self._state.target_xyz_m[0] += self._translation_step_m
        elif key == "a":
            self._state.target_xyz_m[0] -= self._translation_step_m
        elif key == "w":
            self._state.target_xyz_m[1] += self._translation_step_m
        elif key == "s":
            self._state.target_xyz_m[1] -= self._translation_step_m
        elif key == "e":
            self._state.target_xyz_m[2] += self._translation_step_m
        elif key == "d":
            self._state.target_xyz_m[2] -= self._translation_step_m
        elif key == "r":
            self._state.target_rpy_rad[0] += self._rotation_step_rad
        elif key == "f":
            self._state.target_rpy_rad[0] -= self._rotation_step_rad
        elif key == "t":
            self._state.target_rpy_rad[1] += self._rotation_step_rad
        elif key == "g":
            self._state.target_rpy_rad[1] -= self._rotation_step_rad
        elif key == "y":
            self._state.target_rpy_rad[2] += self._rotation_step_rad
        elif key == "h":
            self._state.target_rpy_rad[2] -= self._rotation_step_rad
        elif key == "u":
            self._apply_joint_delta(0, self._joint_step_rad)
        elif key == "j":
            self._apply_joint_delta(0, -self._joint_step_rad)
        elif key == "i":
            self._apply_joint_delta(1, self._joint_step_rad)
        elif key == "k":
            self._apply_joint_delta(1, -self._joint_step_rad)
        elif key == "o":
            self._apply_joint_delta(2, self._joint_step_rad)
        elif key == "l":
            self._apply_joint_delta(2, -self._joint_step_rad)
        elif key == "p":
            self._apply_joint_delta(3, self._joint_step_rad)
        elif key == ";":
            self._apply_joint_delta(3, -self._joint_step_rad)
        elif key == "z":
            self._apply_joint_delta(4, self._joint_step_rad)
        elif key == "x":
            self._apply_joint_delta(4, -self._joint_step_rad)
        elif key == "c":
            self._apply_joint_delta(5, self._joint_step_rad)
        elif key == "v":
            self._apply_joint_delta(5, -self._joint_step_rad)
        elif key == "b":
            self._apply_joint_delta(6, self._joint_step_rad)
        elif key == "n":
            self._apply_joint_delta(6, -self._joint_step_rad)
        elif key == "m":
            self._run_ik()
        elif key == ",":
            self._sync_target_to_current_tcp()
        elif key == "0":
            self._reset_state()
        else:
            handled = False

        if handled:
            self._draw()

    def _apply_joint_delta(self, joint_index: int, delta_rad: float) -> None:
        spec = self._joint_specs[joint_index]
        next_value = self._state.joint_values_rad[joint_index] + delta_rad
        self._state.joint_values_rad[joint_index] = float(
            np.clip(next_value, spec.lower_rad, spec.upper_rad)
        )
        self._last_ik_result = IKSolveResult(
            success=True,
            method="manual-joint",
            message=f"updated J{joint_index + 1}",
        )

    def _run_ik(self) -> None:
        result = _solve_target_ik(self._joint_specs, self._tcp_offset_m, self._state)
        self._last_ik_result = result
        if result.success and result.joint_values_rad is not None:
            self._state.joint_values_rad[:] = result.joint_values_rad

    def _sync_target_to_current_tcp(self) -> None:
        _, _, _, tcp_tf = _compute_robot_geometry(
            self._joint_specs, self._tcp_offset_m, self._state
        )
        self._state.target_xyz_m[:] = tcp_tf[:3, 3]
        self._state.target_rpy_rad[:] = R.from_matrix(tcp_tf[:3, :3]).as_euler(
            "xyz", degrees=False
        )
        self._last_ik_result = IKSolveResult(
            success=True,
            method="sync",
            message="target synced from current tcp",
        )

    def _reset_state(self) -> None:
        for index, spec in enumerate(self._joint_specs):
            self._state.joint_values_rad[index] = (
                spec.lower_rad + spec.upper_rad
            ) * 0.5
        self._sync_target_to_current_tcp()

    def _draw(self) -> None:
        self._axes.cla()
        joint_origins_world, joint_axes_world, skeleton_points_world, tcp_tf = (
            _compute_robot_geometry(
                self._joint_specs,
                self._tcp_offset_m,
                self._state,
            )
        )

        self._draw_world_frame(self._axes)
        self._draw_skeleton(self._axes, skeleton_points_world)
        self._draw_joint_axes(self._axes, joint_origins_world, joint_axes_world)
        self._draw_joint_labels(self._axes, joint_origins_world)
        self._draw_target_pose(self._axes, _build_target_transform(self._state))
        self._draw_tcp_frame(self._axes, tcp_tf)
        self._setup_axes(self._axes)
        self._update_annotations()
        self._figure.canvas.draw_idle()

    def _draw_world_frame(self, axes: Axes3D) -> None:
        origin = np.zeros(3, dtype=np.float64)
        unit = self._axis_draw_length_m * 1.2
        basis = np.eye(3, dtype=np.float64)
        plot_axes = cast(Any, axes)
        plot_axes.quiver(
            [origin[0], origin[0], origin[0]],
            [origin[1], origin[1], origin[1]],
            [origin[2], origin[2], origin[2]],
            basis[0, :],
            basis[1, :],
            basis[2, :],
            length=unit,
            normalize=True,
            colors=("r", "g", "b"),
            linewidths=2.0,
        )

    def _draw_skeleton(self, axes: Axes3D, skeleton_points_world: np.ndarray) -> None:
        axes.plot(
            skeleton_points_world[:, 0],
            skeleton_points_world[:, 1],
            skeleton_points_world[:, 2],
            color="#1f77b4",
            linewidth=3.0,
            marker="o",
            markersize=5.5,
        )
        tcp = skeleton_points_world[-1]
        plot_axes = cast(Any, axes)
        plot_axes.scatter(
            [tcp[0]], [tcp[1]], [tcp[2]], color="#ff7f0e", s=70, label="TCP"
        )

    def _draw_tcp_frame(self, axes: Axes3D, tcp_tf: np.ndarray) -> None:
        plot_axes = cast(Any, axes)
        origin = tcp_tf[:3, 3]
        basis = tcp_tf[:3, :3]
        plot_axes.quiver(
            [origin[0], origin[0], origin[0]],
            [origin[1], origin[1], origin[1]],
            [origin[2], origin[2], origin[2]],
            basis[0, :],
            basis[1, :],
            basis[2, :],
            length=self._axis_draw_length_m * 0.8,
            normalize=True,
            colors=("#ff7f0e", "#ffbf00", "#ff4f81"),
            linewidths=1.6,
        )

    def _draw_target_pose(self, axes: Axes3D, target_tf: np.ndarray) -> None:
        plot_axes = cast(Any, axes)
        origin = target_tf[:3, 3]
        basis = target_tf[:3, :3]
        plot_axes.scatter(
            [origin[0]],
            [origin[1]],
            [origin[2]],
            color="#2ca02c",
            s=55,
            label="Target",
        )
        plot_axes.quiver(
            [origin[0], origin[0], origin[0]],
            [origin[1], origin[1], origin[1]],
            [origin[2], origin[2], origin[2]],
            basis[0, :],
            basis[1, :],
            basis[2, :],
            length=self._axis_draw_length_m,
            normalize=True,
            colors=("#2ca02c", "#17becf", "#9467bd"),
            linewidths=1.6,
        )

    def _draw_joint_axes(
        self,
        axes: Axes3D,
        joint_origins_world: np.ndarray,
        joint_axes_world: np.ndarray,
    ) -> None:
        plot_axes = cast(Any, axes)
        plot_axes.quiver(
            joint_origins_world[:, 0],
            joint_origins_world[:, 1],
            joint_origins_world[:, 2],
            joint_axes_world[:, 0],
            joint_axes_world[:, 1],
            joint_axes_world[:, 2],
            length=self._axis_draw_length_m,
            normalize=True,
            color="#d62728",
            linewidths=1.6,
        )

    def _draw_joint_labels(self, axes: Axes3D, joint_origins_world: np.ndarray) -> None:
        label_offset = self._axis_draw_length_m * 0.15
        for index, point in enumerate(joint_origins_world, start=1):
            axes.text(
                float(point[0]),
                float(point[1]),
                float(point[2] + label_offset),
                f"J{index}",
                fontsize=9,
                color="#222222",
            )

    def _setup_axes(self, axes: Axes3D) -> None:
        center = np.zeros(3, dtype=np.float64)
        radius = self._plot_radius_m
        axes.set_xlim(center[0] - radius, center[0] + radius)
        axes.set_ylim(center[1] - radius, center[1] + radius)
        axes.set_zlim(center[2] - radius * 0.2, center[2] + radius * 1.8)
        axes.set_box_aspect((1.0, 1.0, 1.0))
        axes.set_xlabel("X (m)")
        axes.set_ylabel("Y (m)")
        axes.set_zlabel("Z (m)")
        axes.set_title("AR5-5LR Matplotlib Interactive Simulator")
        axes.grid(True)
        axes.legend(loc="upper right")

    def _update_annotations(self) -> None:
        for artist in self._annotation_artists:
            remove = getattr(artist, "remove", None)
            if callable(remove):
                remove()
        self._annotation_artists.clear()

        target_xyz_mm = self._state.target_xyz_m * 1000.0
        target_rpy_deg = np.degrees(self._state.target_rpy_rad)
        joint_deg = np.degrees(self._state.joint_values_rad)
        state_lines = [
            f"URDF: {self._urdf_path.name}",
            (
                "target xyz(mm): "
                f"[{target_xyz_mm[0]:.1f}, {target_xyz_mm[1]:.1f}, {target_xyz_mm[2]:.1f}]"
            ),
            (
                "target rpy(deg): "
                f"[{target_rpy_deg[0]:.1f}, {target_rpy_deg[1]:.1f}, {target_rpy_deg[2]:.1f}]"
            ),
            "joints(deg): "
            + ", ".join(
                f"J{index + 1}={value:.1f}" for index, value in enumerate(joint_deg)
            ),
            f"ik status: {self._last_ik_result.method} | {self._last_ik_result.message}",
            (
                "steps: "
                f"xyz={self._translation_step_m * 1000.0:.1f} mm, "
                f"rpy={math.degrees(self._rotation_step_rad):.1f} deg, "
                f"joint={math.degrees(self._joint_step_rad):.1f} deg"
            ),
        ]
        binding_lines = [
            f"{binding.key}: {binding.description}" for binding in self._bindings
        ]
        self._annotation_artists.append(
            self._figure.text(
                0.02,
                0.98,
                "\n".join(state_lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
        )
        self._annotation_artists.append(
            self._figure.text(
                0.02,
                0.72,
                "\n".join(binding_lines),
                va="top",
                ha="left",
                fontsize=10,
                family="monospace",
            )
        )
        self._figure.subplots_adjust(left=0.34, right=0.98, top=0.96, bottom=0.05)


# endregion


# region CLI


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 matplotlib 的 AR5-5LR 交互式骨架仿真。"
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help="URDF 文件路径，默认指向珞石 AR5-5LR 描述目录。",
    )
    parser.add_argument(
        "--translation-step-mm",
        type=float,
        default=10.0,
        help="xyz 每次按键平移步长，单位 mm。",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=5.0,
        help="rpy 每次按键旋转步长，单位 deg。",
    )
    parser.add_argument(
        "--joint-step-deg",
        type=float,
        default=5.0,
        help="关节每次按键旋转步长，单位 deg。",
    )
    parser.add_argument(
        "--save-preview", type=Path, default=None, help="保存当前窗口图像到指定路径。"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不打开交互窗口，只执行一次构图。适合静态验证或配合 --save-preview 使用。",
    )
    parser.add_argument(
        "--ui-backend",
        choices=("qt", "matplotlib"),
        default="qt",
        help="界面后端。默认使用 qt；仅在需要旧版纯 matplotlib 行为时才选 matplotlib。",
    )
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    if args.ui_backend == "qt":
        from ar5_pyside6_interactive_sim import main as qt_main

        forwarded_args = [
            "--urdf",
            str(args.urdf),
            "--translation-step-mm",
            str(args.translation_step_mm),
            "--rotation-step-deg",
            str(args.rotation_step_deg),
            "--joint-step-deg",
            str(args.joint_step_deg),
        ]
        original_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0], *forwarded_args]
            return qt_main()
        finally:
            sys.argv = original_argv

    simulator = MatplotlibRobotSimulator(
        urdf_path=args.urdf,
        translation_step_mm=args.translation_step_mm,
        rotation_step_deg=args.rotation_step_deg,
        joint_step_deg=args.joint_step_deg,
    )
    if args.save_preview is not None:
        simulator.save_preview(args.save_preview)
    simulator.print_state()
    if not args.no_show:
        simulator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
