from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def build_frame_from_three_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """根据三个点构造右手坐标系的 4x4 齐次变换矩阵。

    约定：
    - 点 0 作为原点
    - 点 1 定义 x 轴方向
    - 点 2 提供 y 轴参考方向

    返回值表示“这个坐标系相对于输入点所在基坐标系”的位姿。
    """
    x_axis = p1 - p0
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_ref = p2 - p0
    z_axis = np.cross(x_axis, y_ref)
    z_axis = z_axis / np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)

    mat = np.eye(4, dtype=float)
    mat[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    mat[:3, 3] = p0
    return mat


def build_xyzrpy_matrix(x: float, y: float, z: float, rx: float, ry: float, rz: float) -> np.ndarray:
    """按项目约定的 XYZ 欧拉角构造 4x4 齐次变换矩阵。"""
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = R.from_euler("XYZ", [rx, ry, rz], degrees=True).as_matrix()
    mat[:3, 3] = [x, y, z]
    return mat


def draw_frame(ax: plt.Axes, mat: np.ndarray, name: str, length: float = 40.0) -> None:
    """在 3D 坐标系中绘制一个齐次变换表示的坐标系。"""
    origin = mat[:3, 3]
    axes = mat[:3, :3]
    colors = ["r", "g", "b"]
    labels = ["x", "y", "z"]

    for idx, (color, label) in enumerate(zip(colors, labels, strict=True)):
        vec = axes[:, idx] * length
        end = origin + vec
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color=color, linewidth=2)
        ax.text(end[0], end[1], end[2], f"{name}.{label}", color=color)

    ax.text(origin[0], origin[1], origin[2], name, color="k")


def main() -> None:
    record_path = Path(r"test/wuji/records/charuco_delta_board_jog_20260707_205824.json")
    data = json.loads(record_path.read_text(encoding="utf-8"))

    points = np.array([item["board_delta"]["translation_mm"] for item in data["points"]], dtype=float)
    p0, p1, p2 = points

    print("tcp point distances:")
    print(f"d01={np.linalg.norm(p1 - p0):.6f}")
    print(f"d02={np.linalg.norm(p2 - p0):.6f}")
    print(f"d12={np.linalg.norm(p2 - p1):.6f}")

    frame_from_points = build_frame_from_three_points(p0, p1, p2)

    # 你给的偏移：xyzrpy(534.5/586.93/-131.69/-104.49/-8.49/-51.04)
    # 这里按“先叠加这个偏移，再使用三点建系结果”的左乘方式组合。
    offset = build_xyzrpy_matrix(
        x=534.5,
        y=586.93,
        z=-131.69,
        rx=-104.49,
        ry=-8.49,
        rz=-51.04,
    )

    final_mat = offset @ frame_from_points
    final_rpy = R.from_matrix(final_mat[:3, :3]).as_euler("XYZ", degrees=True)

    print("final xyzrpy:")
    print(
        f"x={final_mat[0, 3]:.6f}, "
        f"y={final_mat[1, 3]:.6f}, "
        f"z={final_mat[2, 3]:.6f}, "
        f"rx={final_rpy[0]:.6f}, "
        f"ry={final_rpy[1]:.6f}, "
        f"rz={final_rpy[2]:.6f}"
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Reference Frame (left-multiplied) and Final Frame")
    draw_frame(ax, offset, "ref", length=40.0)
    draw_frame(ax, final_mat, "final", length=40.0)

    all_points = np.vstack([offset[:3, 3], final_mat[:3, 3], p0, p1, p2])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = np.max(maxs - mins)
    half = max(span / 2.0, 1.0)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=-60)
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
