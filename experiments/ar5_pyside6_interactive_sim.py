#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, cast

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial.transform import Rotation as R

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[0]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ar5_matplotlib_interactive_sim import (  # noqa: E402
    DEFAULT_URDF_PATH,
    IKSolveResult,
    JointSpec,
    RobotState,
    _build_joint_specs,
    _build_target_transform,
    _compute_robot_geometry,
    _estimate_plot_radius_m,
    _find_tcp_offset_m,
    _solve_target_ik,
)
from src.robotics.urdf_interface import UrdfConverter  # noqa: E402


class Ar5QtSimulatorWidget(QWidget):
    """基于 PySide6 的 AR5 交互式仿真窗口。"""

    def __init__(
        self,
        urdf_path: Path,
        translation_step_mm: float,
        rotation_step_deg: float,
        joint_step_deg: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._urdf_path = urdf_path
        self._translation_step_m = float(translation_step_mm) / 1000.0
        self._rotation_step_rad = math.radians(float(rotation_step_deg))
        self._joint_step_rad = math.radians(float(joint_step_deg))

        model = UrdfConverter().from_file(urdf_path)
        self._joint_specs: list[JointSpec] = _build_joint_specs(model)
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
        self._last_ik_result = IKSolveResult(
            success=True,
            method="init",
            message="target synced to current tcp",
        )

        self._figure = Figure(figsize=(9.0, 7.2))
        self._figure.subplots_adjust(left=0.04, right=0.98, bottom=0.06, top=0.96)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._axes = cast(Axes3D, self._figure.add_subplot(111, projection="3d"))
        self._axes.mouse_init(rotate_btn=[1], zoom_btn=[3])  # type: ignore
        self._status_label = QLabel(self)
        self._hint_text = QTextEdit(self)
        self._control_panel = QWidget(self)

        self._setup_ui()
        self._refresh_scene()

    def _setup_ui(self) -> None:
        self.setWindowTitle("AR5-5LR PySide6 Interactive Simulator")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(1480, 920)

        self._status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._status_label.setWordWrap(True)

        self._hint_text.setReadOnly(True)
        self._hint_text.setMinimumWidth(360)
        self._hint_text.setPlainText(
            "\n".join(
                [
                    "Qt controls:",
                    "q/a: target X + / -",
                    "w/s: target Y + / -",
                    "e/d: target Z + / -",
                    "r/f: target Roll + / -",
                    "t/g: target Pitch + / -",
                    "y/h: target Yaw + / -",
                    "u/j: J1 + / -",
                    "i/k: J2 + / -",
                    "o/l: J3 + / -",
                    "p/;: J4 + / -",
                    "z/x: J5 + / -",
                    "c/v: J6 + / -",
                    "b/n: J7 + / -",
                    "m: solve IK",
                    ",: sync target from current tcp",
                    "0: reset joints and target",
                    "",
                    "Mouse on plot:",
                    "left drag: rotate",
                    "right drag: zoom",
                ]
            )
        )

        right_panel = QVBoxLayout()
        right_panel.addWidget(self._status_label)
        right_panel.addWidget(self._build_control_panel())
        right_panel.addWidget(self._hint_text, stretch=1)

        right_holder = QWidget(self)
        right_holder.setLayout(right_panel)
        right_holder.setMinimumWidth(380)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)
        root.addWidget(self._canvas, stretch=3)
        root.addWidget(right_holder, stretch=1)

    def _build_control_panel(self) -> QWidget:
        layout = QGridLayout(self._control_panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)

        button_specs = [
            ("X+", "q", 0, 0),
            ("X-", "a", 0, 1),
            ("Y+", "w", 0, 2),
            ("Y-", "s", 0, 3),
            ("Z+", "e", 0, 4),
            ("Z-", "d", 0, 5),
            ("R+", "r", 1, 0),
            ("R-", "f", 1, 1),
            ("P+", "t", 1, 2),
            ("P-", "g", 1, 3),
            ("Y+", "y", 1, 4),
            ("Y-", "h", 1, 5),
            ("J1+", "u", 2, 0),
            ("J1-", "j", 2, 1),
            ("J2+", "i", 2, 2),
            ("J2-", "k", 2, 3),
            ("J3+", "o", 2, 4),
            ("J3-", "l", 2, 5),
            ("J4+", "p", 3, 0),
            ("J4-", ";", 3, 1),
            ("J5+", "z", 3, 2),
            ("J5-", "x", 3, 3),
            ("J6+", "c", 3, 4),
            ("J6-", "v", 3, 5),
            ("J7+", "b", 4, 0),
            ("J7-", "n", 4, 1),
            ("Solve IK", "m", 4, 2),
            ("Sync TCP", ",", 4, 3),
            ("Reset", "0", 4, 4),
        ]
        for text, key, row, column in button_specs:
            button = QPushButton(f"{text}\n[{key}]", self._control_panel)
            button.clicked.connect(
                lambda _checked=False, command_key=key: self._on_control_button_clicked(
                    command_key
                )
            )
            layout.addWidget(button, row, column)
        return self._control_panel

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.setFocus()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.text().lower()
        if key == "":
            super().keyPressEvent(event)
            return

        if self._execute_command(key):
            self._refresh_scene()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_control_button_clicked(self, key: str) -> None:
        if self._execute_command(key):
            self._refresh_scene()
        self.setFocus()

    def _execute_command(self, key: str) -> bool:
        if key == "q":
            self._state.target_xyz_m[0] += self._translation_step_m
            self._set_manual_status("target X +")
        elif key == "a":
            self._state.target_xyz_m[0] -= self._translation_step_m
            self._set_manual_status("target X -")
        elif key == "w":
            self._state.target_xyz_m[1] += self._translation_step_m
            self._set_manual_status("target Y +")
        elif key == "s":
            self._state.target_xyz_m[1] -= self._translation_step_m
            self._set_manual_status("target Y -")
        elif key == "e":
            self._state.target_xyz_m[2] += self._translation_step_m
            self._set_manual_status("target Z +")
        elif key == "d":
            self._state.target_xyz_m[2] -= self._translation_step_m
            self._set_manual_status("target Z -")
        elif key == "r":
            self._state.target_rpy_rad[0] += self._rotation_step_rad
            self._set_manual_status("target Roll +")
        elif key == "f":
            self._state.target_rpy_rad[0] -= self._rotation_step_rad
            self._set_manual_status("target Roll -")
        elif key == "t":
            self._state.target_rpy_rad[1] += self._rotation_step_rad
            self._set_manual_status("target Pitch +")
        elif key == "g":
            self._state.target_rpy_rad[1] -= self._rotation_step_rad
            self._set_manual_status("target Pitch -")
        elif key == "y":
            self._state.target_rpy_rad[2] += self._rotation_step_rad
            self._set_manual_status("target Yaw +")
        elif key == "h":
            self._state.target_rpy_rad[2] -= self._rotation_step_rad
            self._set_manual_status("target Yaw -")
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
            return False
        return True

    def _set_manual_status(self, message: str) -> None:
        self._last_ik_result = IKSolveResult(
            success=True,
            method="manual",
            message=message,
        )

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

    def _refresh_scene(self) -> None:
        self._axes.cla()
        joint_origins_world, joint_axes_world, skeleton_points_world, tcp_tf = (
            _compute_robot_geometry(
                self._joint_specs,
                self._tcp_offset_m,
                self._state,
            )
        )
        target_tf = _build_target_transform(self._state)
        self._draw_world_frame()
        self._draw_skeleton(skeleton_points_world)
        self._draw_joint_axes(joint_origins_world, joint_axes_world)
        self._draw_joint_labels(joint_origins_world)
        self._draw_tcp_frame(tcp_tf)
        self._draw_target_pose(target_tf)
        self._setup_axes()
        self._update_status_label(tcp_tf)
        self._canvas.draw_idle()

    def _draw_world_frame(self) -> None:
        plot_axes = cast(Any, self._axes)
        plot_axes.quiver(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            length=self._axis_draw_length_m * 1.2,
            normalize=True,
            colors=("r", "g", "b"),
            linewidths=2.0,
        )

    def _draw_skeleton(self, skeleton_points_world: np.ndarray) -> None:
        self._axes.plot(
            skeleton_points_world[:, 0],
            skeleton_points_world[:, 1],
            skeleton_points_world[:, 2],
            color="#1f77b4",
            linewidth=3.0,
            marker="o",
            markersize=5.5,
        )
        tcp = skeleton_points_world[-1]
        plot_axes = cast(Any, self._axes)
        plot_axes.scatter(
            [tcp[0]], [tcp[1]], [tcp[2]], color="#ff7f0e", s=70, label="TCP"
        )

    def _draw_joint_axes(
        self, joint_origins_world: np.ndarray, joint_axes_world: np.ndarray
    ) -> None:
        plot_axes = cast(Any, self._axes)
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

    def _draw_joint_labels(self, joint_origins_world: np.ndarray) -> None:
        label_offset = self._axis_draw_length_m * 0.15
        for index, point in enumerate(joint_origins_world, start=1):
            self._axes.text(
                float(point[0]),
                float(point[1]),
                float(point[2] + label_offset),
                f"J{index}",
                fontsize=9,
                color="#222222",
            )

    def _draw_tcp_frame(self, tcp_tf: np.ndarray) -> None:
        plot_axes = cast(Any, self._axes)
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

    def _draw_target_pose(self, target_tf: np.ndarray) -> None:
        plot_axes = cast(Any, self._axes)
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

    def _setup_axes(self) -> None:
        radius = self._plot_radius_m
        self._axes.set_xlim(-radius, radius)
        self._axes.set_ylim(-radius, radius)
        self._axes.set_zlim(-radius * 0.2, radius * 1.8)
        self._axes.set_box_aspect((1.0, 1.0, 1.0))
        self._axes.set_xlabel("X (m)")
        self._axes.set_ylabel("Y (m)")
        self._axes.set_zlabel("Z (m)")
        self._axes.set_title("AR5-5LR PySide6 Interactive Simulator")
        self._axes.grid(True)
        self._axes.legend(loc="upper right")

    def _update_status_label(self, tcp_tf: np.ndarray) -> None:
        target_xyz_mm = self._state.target_xyz_m * 1000.0
        target_rpy_deg = np.degrees(self._state.target_rpy_rad)
        current_xyz_mm = tcp_tf[:3, 3] * 1000.0
        current_rpy_deg = R.from_matrix(tcp_tf[:3, :3]).as_euler("xyz", degrees=True)
        joint_deg = np.degrees(self._state.joint_values_rad)
        self._status_label.setText(
            "\n".join(
                [
                    f"URDF: {self._urdf_path.name}",
                    (
                        "Current TCP xyz(mm): "
                        f"[{current_xyz_mm[0]:.1f}, {current_xyz_mm[1]:.1f}, {current_xyz_mm[2]:.1f}]"
                    ),
                    (
                        "Current TCP rpy(deg): "
                        f"[{current_rpy_deg[0]:.1f}, {current_rpy_deg[1]:.1f}, {current_rpy_deg[2]:.1f}]"
                    ),
                    (
                        "Target xyz(mm): "
                        f"[{target_xyz_mm[0]:.1f}, {target_xyz_mm[1]:.1f}, {target_xyz_mm[2]:.1f}]"
                    ),
                    (
                        "Target rpy(deg): "
                        f"[{target_rpy_deg[0]:.1f}, {target_rpy_deg[1]:.1f}, {target_rpy_deg[2]:.1f}]"
                    ),
                    "Joints(deg): "
                    + ", ".join(
                        f"J{index + 1}={value:.1f}"
                        for index, value in enumerate(joint_deg)
                    ),
                    (
                        "IK status: "
                        f"{self._last_ik_result.method} | {self._last_ik_result.message}"
                    ),
                    (
                        "Steps: "
                        f"xyz={self._translation_step_m * 1000.0:.1f} mm, "
                        f"rpy={math.degrees(self._rotation_step_rad):.1f} deg, "
                        f"joint={math.degrees(self._joint_step_rad):.1f} deg"
                    ),
                ]
            )
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 PySide6 + matplotlib 的 AR5-5LR 交互式仿真。"
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help="URDF 文件路径。",
    )
    parser.add_argument(
        "--translation-step-mm",
        type=float,
        default=10.0,
        help="目标 xyz 每次按键平移步长，单位 mm。",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=5.0,
        help="目标 rpy 每次按键旋转步长，单位 deg。",
    )
    parser.add_argument(
        "--joint-step-deg",
        type=float,
        default=5.0,
        help="关节每次按键旋转步长，单位 deg。",
    )
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication(sys.argv)
    widget = Ar5QtSimulatorWidget(
        urdf_path=args.urdf,
        translation_step_mm=args.translation_step_mm,
        rotation_step_deg=args.rotation_step_deg,
        joint_step_deg=args.joint_step_deg,
    )
    widget.show()
    widget.raise_()
    widget.activateWindow()
    widget.setFocus()
    if owns_app:
        return app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
