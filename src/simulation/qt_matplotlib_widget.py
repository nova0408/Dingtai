from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from src.simulation.protocols import ChainSnapshot


# region 数据结构


@dataclass(frozen=True, slots=True)
class PlotStyle:
    """三维绘图边界配置。"""

    xlim: tuple[float, float] = (-1.5, 1.5)
    """X 轴范围，单位 米。"""

    ylim: tuple[float, float] = (-1.5, 1.5)
    """Y 轴范围，单位 米。"""

    zlim: tuple[float, float] = (0.0, 2.0)
    """Z 轴范围，单位 米。"""


# endregion


# region Qt 组件


class MatplotKinematicsWidget(QWidget):
    """Matplotlib 三维实时绘图 Qt 组件。"""

    def __init__(self, parent: QWidget | None = None, style: PlotStyle | None = None) -> None:
        super().__init__(parent)
        mpl.rcParams["axes3d.mouserotationstyle"] = "arcball"
        self._style = PlotStyle() if style is None else style
        self._view_elev = 24.0
        self._view_azim = 38.0
        self._view_roll = 0.0
        self._interaction_enabled = True

        self._figure = Figure(figsize=(7.0, 6.0))
        self._figure.subplots_adjust(left=0.03, right=0.98, bottom=0.04, top=0.95)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._ax = self._figure.add_subplot(111, projection="3d")
        self._ax.mouse_init(rotate_btn=[1], zoom_btn=[3])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self._init_axis(reset_view=True)

    def set_interaction_enabled(self, enabled: bool) -> None:
        """控制画布鼠标交互；用于拖动滑条时避免误旋转。"""

        self._interaction_enabled = enabled
        if enabled:
            self._ax.mouse_init(rotate_btn=[1], zoom_btn=[3])
        else:
            self._ax.mouse_init(rotate_btn=[], zoom_btn=[])

    def _capture_view(self) -> None:
        """缓存当前视角，避免重绘时重置观察方向。"""

        self._view_elev = float(self._ax.elev)
        self._view_azim = float(self._ax.azim)
        self._view_roll = float(getattr(self._ax, "roll", 0.0))

    def _init_axis(self, reset_view: bool = False) -> None:
        """初始化坐标轴与可视范围。"""

        self._ax.cla()
        self._ax.set_title("3D Kinematics Preview")
        self._ax.set_xlabel("X (m)")
        self._ax.set_ylabel("Y (m)")
        self._ax.set_zlabel("Z (m)")
        self._ax.set_xlim(self._style.xlim)
        self._ax.set_ylim(self._style.ylim)
        self._ax.set_zlim(self._style.zlim)
        if reset_view:
            self._view_elev = 24.0
            self._view_azim = 38.0
            self._view_roll = 0.0
        self._ax.view_init(elev=self._view_elev, azim=self._view_azim, roll=self._view_roll)
        self.set_interaction_enabled(self._interaction_enabled)

    def render_snapshots(
        self,
        snapshots: tuple[ChainSnapshot, ...],
        target_xyz: tuple[float, float, float] | None = None,
        active_chain: str | None = None,
    ) -> None:
        """渲染链快照、关节轴箭头和 IK 目标点。"""

        self._capture_view()
        self._init_axis(reset_view=False)

        for snapshot in snapshots:
            xs = [point.x for point in snapshot.points]
            ys = [point.y for point in snapshot.points]
            zs = [point.z for point in snapshot.points]
            line_width = 3.0 if snapshot.chain_name == active_chain else 2.0
            marker_size = 6 if snapshot.chain_name == active_chain else 4
            plot_color = snapshot.color.to_hex()
            self._ax.plot(
                xs,
                ys,
                zs,
                "-o",
                color=plot_color,
                linewidth=line_width,
                markersize=marker_size,
                label=snapshot.chain_name,
            )

            for axis_glyph in snapshot.joint_axes:
                ox, oy, oz = axis_glyph.axis.origin.to_tuple()
                dx, dy, dz = axis_glyph.axis.z_axis.to_tuple()
                self._ax.quiver(
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    length=0.10,
                    normalize=True,
                    color=plot_color,
                    linewidths=0.8,
                    arrow_length_ratio=0.22,
                )

        if target_xyz is not None:
            self._ax.scatter(
                [target_xyz[0]],
                [target_xyz[1]],
                [target_xyz[2]],
                marker="x",
                c="#d62828",
                s=90,
                label="IK Target",
            )

        self._ax.legend(loc="upper left")
        self._canvas.draw_idle()


# endregion
