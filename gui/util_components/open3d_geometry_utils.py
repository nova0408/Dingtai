import numpy as np
import open3d as o3d  # type: ignore[attr-defined]


def create_coordinate_axis_lines(
    size: float,
    origin: np.ndarray | list[float] | tuple[float, float, float] | None = None,
):
    """构造不依赖 TriangleMesh.create_coordinate_frame 的坐标轴线框。

    2026-06-09 在 Windows + PySide6 6.11.x + Open3D Qt 嵌入路径中，
    `TriangleMesh.create_coordinate_frame(...)` 曾稳定触发进程级崩溃。
    这里统一改为 `LineSet` 版本，避免 GUI 初始化阶段再次踩到该问题。
    """
    axis_size = float(size)
    axis_origin = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    points = np.asarray(
        [
            axis_origin,
            axis_origin + np.array([axis_size, 0.0, 0.0], dtype=np.float64),
            axis_origin + np.array([0.0, axis_size, 0.0], dtype=np.float64),
            axis_origin + np.array([0.0, 0.0, axis_size], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    lines = np.asarray([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    colors = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(points)
    axis.lines = o3d.utility.Vector2iVector(lines)
    axis.colors = o3d.utility.Vector3dVector(colors)
    return axis
