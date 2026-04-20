import os
import sys

import numpy as np
import open3d as o3d
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .open3d_widget import O3DViewerWidget


class O3DViewerWindow(QWidget):
    closed = Signal()

    def __init__(self, title="3D Viewer", size=(1024, 768), stay_on_top=False):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(*size)
        if stay_on_top:
            self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

        self.viewer = O3DViewerWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer)

    def closeEvent(self, event):
        self.setVisible(False)
        self.viewer.cleanup()
        self.closed.emit()
        event.accept()


# ---------------------------------------------------------
# 3. 演示示例：带几何体列表的 Demo
# ---------------------------------------------------------
class GeometryListItem(QFrame):
    """列表中的单行组件"""

    toggled = Signal(str, bool)

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.is_visible = True
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setProperty("themeRole", "cloud-item")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        self.label = QLabel(name)
        self.btn_eye = QPushButton("👁")  # 借用 Emoji 作为图标
        self.btn_eye.setFixedWidth(40)
        self.btn_eye.setCheckable(True)
        self.btn_eye.setChecked(True)
        self.btn_eye.setProperty("themeRole", "cloud-eye-button")

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.btn_eye)

        self.btn_eye.clicked.connect(self._on_toggle)

    def _on_toggle(self, checked):
        self.btn_eye.setText("👁" if checked else "❌")
        self.toggled.emit(self.name, checked)


class ListDemoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open3D Component Demo")
        self.resize(1200, 800)

        # 主布局：左右分割
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 左侧：3D 视图 (使用我们的 Widget)
        self.viewer = O3DViewerWidget()
        main_layout.addWidget(self.viewer, stretch=4)

        # 2. 右侧：控制面板
        self.panel = QWidget()
        self.panel.setFixedWidth(250)
        self.panel.setProperty("themeRole", "side-panel")
        panel_layout = QVBoxLayout(self.panel)

        title = QLabel("几何体列表")
        title.setProperty("themeRole", "section-title")
        title.setContentsMargins(10, 10, 10, 10)
        panel_layout.addWidget(title)

        # 滚动区域存放列表
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.addStretch()  # 把项目往上挤
        self._scroll.setWidget(self.list_container)
        panel_layout.addWidget(self._scroll)

        main_layout.addWidget(self.panel)

        # 初始化一些测试数据
        self.setup_demo_data()

    def setup_demo_data(self):
        # 设置背景色
        self.viewer.set_background_color(0.1, 0.1, 0.1)

        # 1. 添加一个球体
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1, 0.7, 0.3])
        self.add_item_to_viewer("Golden_Sphere", sphere)

        # 2. 添加一个坐标轴
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        self.add_item_to_viewer("World_Axes", axes)

        # 3. 添加一个随机点云
        pts = np.random.uniform(-2, 2, (10000, 3))
        cols = np.random.uniform(0.4, 1, (10000, 3))
        self.viewer.add_point_cloud("Random_Cloud", pts, cols)
        self._create_list_widget("Random_Cloud")  # 手动补个列表项

    def add_item_to_viewer(self, name, geom):
        self.viewer.add_point_cloud(name, geom)
        self._create_list_widget(name)

    def _create_list_widget(self, name):
        item_ui = GeometryListItem(name)
        # item_ui.toggled.connect(self.viewer.set_geometry_visible)
        # 插入到最前面
        self.list_layout.insertWidget(self.list_layout.count() - 1, item_ui)


if __name__ == "__main__":
    # 环境路径设置
    pyside6_dir = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6")
    os.environ["QT_PLUGIN_PATH"] = os.path.join(pyside6_dir, "plugins")
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(pyside6_dir, "plugins", "platforms")

    app = QApplication(sys.argv)
    demo = ListDemoWindow()
    demo.show()
    sys.exit(app.exec())
