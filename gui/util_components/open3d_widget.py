import uuid

import numpy as np
import open3d as o3d
import win32api
import win32con
import win32gui
from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QWindow
from PySide6.QtWidgets import QVBoxLayout, QWidget

# ---------------------------------------------------------
# 1. 核心渲染组件：O3DViewerWidget (API 承载体)
# ---------------------------------------------------------


class O3DViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.unique_name = f"O3D_{uuid.uuid4().hex[:8]}"
        # 关闭 o3d 警告
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        self.vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        self.hwnd = 0
        self._is_cleaned = False
        self.default_zoom = 0.7
        # 用于记录鼠标状态，实现“点击”检测（边缘触发）
        self._was_ctrl_lbutton_down = False
        # 几何体管理
        self._geometries: dict[str, o3d.geometry.Geometry] = {}
        self._geometry_transforms: dict[str, np.ndarray] = {}
        self._show_origin_axis = False
        self._helpers_data = {"bboxes": {}, "others": {}, "select_points": {}}
        self._select_points: list[np.ndarray] = []
        self._origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=np.array([0, 0, 0]))
        self._setup_ui()
        self._init_open3d()
        self.destroyed.connect(self.cleanup)

    @property
    def helpers(self):
        """返回分类存储的辅助对象字典"""
        return self._helpers_data

    def _setup_ui(self):
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet("background-color: black;")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def _init_open3d(self):
        # 【方案一：坐标放逐】
        # 在创建时就把它丢到屏幕外（-9999），防止在屏幕中心闪现
        logger.debug(f"创建 Open3D 窗口 {self.unique_name}")
        self.vis.create_window(
            window_name=self.unique_name,
            width=640,
            height=480,
            visible=True,
            left=-9999,
            top=-9999,  # 核心点：开在看不见的地方
        )
        # 获取视图控制器
        view_ctl = self.vis.get_view_control()
        # 设置正交投影
        current_fov = view_ctl.get_field_of_view()
        # logger.debug(f"当前 FOV :{current_fov}")
        fov_step = 5 - current_fov  # for example self.target_fov == 5
        view_ctl.change_field_of_view(step=fov_step)
        # logger.debug(f"修改后 FOV :{view_ctl.get_field_of_view()}")

        self.hwnd = win32gui.FindWindowEx(0, 0, None, self.unique_name)

        if self.hwnd:
            # 【方案二：强制抹除窗口样式】
            # 去掉标题栏、边框、系统菜单。这让它变成一个纯粹的“像素矩形”
            style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_STYLE)
            style &= ~win32con.WS_CAPTION
            win32gui.SetWindowLong(self.hwnd, win32con.GWL_STYLE, style)

            self.o3d_window = QWindow.fromWinId(self.hwnd)
            self.container = QWidget.createWindowContainer(self.o3d_window, self)
            self._layout.addWidget(self.container)

            # 初始时让它在容器内也是隐藏的，直到第一次 Resize 确定大小
            win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)

            self.timer = QTimer(self)
            self.timer.timeout.connect(self._render_loop)
            self.timer.start(16)

    def _render_loop(self):
        """渲染循环：Open3D 所有的 API 都在这里执行，确保线程安全"""
        if self._is_cleaned:
            return

        # 1. Open3D 正常的事件处理和渲染
        self.vis.poll_events()
        self.vis.update_renderer()

        # 2. 安全的输入轮询 (Input Polling)
        self._check_input_picking()

    def _check_input_picking(self):
        """在渲染循环中安全地检查鼠标状态"""
        # 获取 Ctrl 键和鼠标左键的实时状态
        # 0x8000 表示按键当前处于按下状态
        ctrl_down = win32api.GetKeyState(win32con.VK_CONTROL) & 0x8000
        lbutton_down = win32api.GetKeyState(win32con.VK_LBUTTON) & 0x8000

        current_state = bool(ctrl_down and lbutton_down)

        # 边缘触发检测：只有在“刚按下”那一瞬间触发
        if current_state and not self._was_ctrl_lbutton_down:
            cursor_pos = win32api.GetCursorPos()  # 全局屏幕坐标
            try:
                local_pos = win32gui.ScreenToClient(self.hwnd, cursor_pos)
                x, y = local_pos

                rect = win32gui.GetClientRect(self.hwnd)
                w, h = rect[2], rect[3]

                # 如果点击在有效范围内
                if 0 <= x <= w and 0 <= y <= h:
                    # logger.debug(f"检测到安全点选请求：({x}, {y})")
                    # 使用统一的深度拾取逻辑
                    self.pick_point_from_depth(x, y)
            except Exception as e:
                logger.error(f"坐标转换异常：{e}")

        self._was_ctrl_lbutton_down = current_state

    # region Qt 事件重写

    def hideEvent(self, event):
        """当父级 hide 或注销时，在任何动作之前先致盲"""
        if self.hwnd and not self._is_cleaned:
            # 极速隐藏
            win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)
            # 丢出屏幕
            win32gui.SetWindowPos(self.hwnd, 0, -9999, -9999, 0, 0, win32con.SWP_NOZORDER | win32con.SWP_NOSIZE)
        super().hideEvent(event)

    def showEvent(self, event):
        if self.hwnd and not self._is_cleaned:
            win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        super().showEvent(event)

    def resizeEvent(self, event):
        """
        响应 Qt 尺寸变化事件，但不立即同步到底层 Open3D 窗口，
        而是采用防抖方式延迟处理。
        """
        super().resizeEvent(event)

        if self._is_cleaned or not self.hwnd:
            return

        if not self._is_initialized_size():
            return

    # endregion
    def _is_initialized_size(self) -> bool:
        """
        检查 Widget 是否已经获得了由布局管理器分配的有效尺寸。
        通常 Qt 在初始化的几毫秒内会产生 (0,0) 或 (1,1) 的尺寸。
        """
        size = self.size()
        return size.width() > 1 and size.height() > 1

    def cleanup(self):
        if self._is_cleaned:
            return
        self._is_cleaned = True
        logger.debug(f"清理 Open3D 窗口 {self.unique_name}")
        if hasattr(self, "timer") and self.timer:
            self.timer.stop()

        try:
            if self.hwnd:
                # 【方案三：断后逻辑】
                # 1. 先设置透明度为 0（如果系统支持）或直接隐藏
                win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)
                # 2. 再次放逐到远方
                win32gui.SetWindowPos(self.hwnd, 0, -9999, -9999, 0, 0, win32con.SWP_NOZORDER)
                # 3. 解除父子关系
                win32gui.SetParent(self.hwnd, 0)

            if self.vis:
                # 4. 强制销毁
                self.vis.destroy_window()
                self.vis = None
        except Exception as e:
            logger.error(f"清理出错：{e}")

    # region 几何体操作 API (重构版)

    def add_point_cloud(self, name: str, geometry, reset_view=True):
        """添加核心业务几何体（点云等），允许根据需求重置视角"""
        self.remove_geometry(name)
        self._geometries[name] = geometry
        self._geometry_transforms[name] = np.eye(4)
        # 添加主几何体时，根据参数决定是否重置包围盒（即相机视角）
        self.vis.add_geometry(geometry, reset_bounding_box=reset_view)
        self.vis.update_geometry(geometry)
        self._display_axis()

    def update_point_cloud(
        self,
        name: str,
        points: np.ndarray | None = None,
        colors: np.ndarray | None = None,
    ) -> None:
        """原地更新已注册点云的数据。

        Parameters
        ----------
        name : str
            点云名称。
        points : np.ndarray | None, optional
            新点坐标，形状为 ``(N, 3)``。
        colors : np.ndarray | None, optional
            新颜色，形状为 ``(N, 3)``，范围通常为 ``[0, 1]``。

        Raises
        ------
        TypeError
            当目标对象不是点云时抛出。
        """
        geometry = self._geometries.get(name)
        if geometry is None:
            raise KeyError(f"未找到点云：{name}")
        if not isinstance(geometry, o3d.geometry.PointCloud):
            raise TypeError(f"{name} 不是点云对象，当前类型为 {type(geometry).__name__}")

        if points is not None:
            pts = np.asarray(points, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(f"points 维度非法，期望 (N, 3)，当前为 {pts.shape}")
            geometry.points = o3d.utility.Vector3dVector(pts)

        if colors is not None:
            cols = np.asarray(colors, dtype=np.float64)
            if cols.ndim != 2 or cols.shape[1] != 3:
                raise ValueError(f"colors 维度非法，期望 (N, 3)，当前为 {cols.shape}")
            geometry.colors = o3d.utility.Vector3dVector(cols)

        self.vis.update_geometry(geometry)

    def transform_geometry(
        self,
        name: str,
        matrix: np.ndarray,
        *,
        absolute: bool = True,
    ) -> None:
        """对已注册几何体应用变换。

        Parameters
        ----------
        name : str
            几何体名称。
        matrix : np.ndarray
            4x4 齐次变换矩阵。
        absolute : bool, optional
            是否按世界坐标系下的绝对位姿解释该矩阵。

            - True:
            `matrix` 表示该对象相对于世界原点的目标位姿。
            - False:
            `matrix` 表示在当前位姿基础上的相对增量变换。

        Notes
        -----
        Open3D 的 ``geometry.transform(matrix)`` 是原地累乘变换。
        因此当 `absolute=True` 时，这里会自动根据当前记录的世界矩阵
        计算增量变换：

        ``delta = target @ inv(current)``

        然后再调用原生 `transform(delta)`。
        """
        geometry = self._geometries.get(name)
        if geometry is None:
            raise KeyError(f"未找到几何体：{name}")
        if not hasattr(geometry, "transform"):
            raise ValueError(f"{name} 不是可变换几何体")
        current = self._geometry_transforms.get(name, np.eye(4))

        if absolute:
            delta = matrix @ np.linalg.inv(current)
            geometry.transform(delta)
            self._geometry_transforms[name] = matrix
        else:
            geometry.transform(matrix)
            self._geometry_transforms[name] = matrix @ current

        self.vis.update_geometry(geometry)

    def transform_helper_geometry(
        self,
        name: str,
        matrix: np.ndarray,
        *,
        absolute: bool = True,
    ) -> None:
        """对已注册几何体应用变换。

        Parameters
        ----------
        name : str
            几何体名称。
        matrix : np.ndarray
            4x4 齐次变换矩阵。
        absolute : bool, optional
            是否按世界坐标系下的绝对位姿解释该矩阵。

            - True:
            `matrix` 表示该对象相对于世界原点的目标位姿。
            - False:
            `matrix` 表示在当前位姿基础上的相对增量变换。

        Notes
        -----
        Open3D 的 ``geometry.transform(matrix)`` 是原地累乘变换。
        因此当 `absolute=True` 时，这里会自动根据当前记录的世界矩阵
        计算增量变换：

        ``delta = target @ inv(current)``

        然后再调用原生 `transform(delta)`。
        """
        geometry = None
        for d in self.helpers.values():
            geometry = d.get(name)
            if geometry:
                break
        if geometry is None:
            raise KeyError(f"未找到几何体：{name}")
        if not hasattr(geometry, "transform"):
            logger.warning(f"{name} 不是可变换几何体")
            return
        current = self._geometry_transforms.get(name, np.eye(4))

        if absolute:
            delta = matrix @ np.linalg.inv(current)
            geometry.transform(delta)
            self._geometry_transforms[name] = matrix
        else:
            geometry.transform(matrix)
            self._geometry_transforms[name] = matrix @ current

        self.vis.update_geometry(geometry)

    def add_helper_geometry(self, name: str, geometry, helper_type: str = "others"):
        """添加辅助几何体（标记、框、轴），严格禁止相机跳变"""
        self.remove_geometry(name)
        if helper_type not in self._helpers_data:
            helper_type = "others"
        self._helpers_data[helper_type][name] = geometry

        self.vis.add_geometry(geometry, reset_bounding_box=False)
        self._geometry_transforms[name] = np.eye(4)
        self.vis.update_geometry(geometry)

    def remove_geometry(self, name: str):
        """全量搜索并移除几何体"""
        if name in self._geometries:
            self.vis.remove_geometry(self._geometries.pop(name), reset_bounding_box=False)
            self._geometry_transforms.pop(name, None)
            return
        for category in self._helpers_data.values():
            if name in category:
                self.vis.remove_geometry(category.pop(name), reset_bounding_box=False)
                self._geometry_transforms.pop(name, None)
                return

    def remove_point_cloud(self, name: str):
        """移除核心业务几何体（点云等）"""
        if name in self._geometries:
            self.vis.remove_geometry(self._geometries.pop(name), reset_bounding_box=False)
            self._geometry_transforms.pop(name, None)
            self._display_axis()

    def remove_select_point(self):
        for point in self.helpers["select_points"].values():
            self.vis.remove_geometry(point, reset_bounding_box=False)

    def _calculate_origin_axis(self):
        # 1. 获取场景在各维度上的跨度
        extents = self._calculate_combined_bounds()
        max_dim = np.max(extents)

        # 2. 计算坐标轴尺寸
        # 建议比例：最大维度的 20% (0.2)。这样既能看清，又不会遮挡点云。
        size = max_dim * 0.2

        # 3. 动态最小值处理
        # 如果场景为空或物体极小（比如 max_dim 为 0），
        # 我们不能直接设为 1，而应该根据场景是否为空来决定。
        if size <= 0:
            size = 1.0  # 场景为空时的默认大小
        elif size < 0.01:
            # 如果是微观点云（如毫米级以下），坐标轴不应强制为 1
            # 这里可以不设下限，或者设为一个更小的感官阈值
            pass

        # 4. 创建坐标轴
        # origin=[0, 0, 0] 确保坐标轴始终在世界坐标系的原点
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=np.array([0, 0, 0]))

    @property
    def show_origin_axis(self):
        return self._show_origin_axis

    @show_origin_axis.setter
    def show_origin_axis(self, show: bool):
        self._show_origin_axis = show
        # logger.debug(f"修改显示原点坐标轴为 {show}")
        self._display_axis()

    def _display_axis(self):
        """移除坐标轴辅助对象"""
        if self._origin_axis is not None:
            self.vis.remove_geometry(self._origin_axis, reset_bounding_box=False)
            self._origin_axis = None
        if not self._show_origin_axis:
            return
        self._origin_axis = self._calculate_origin_axis()
        # logger.debug(f"{'显示'if self._show_origin_axis else '隐藏'}原点坐标轴")
        self.vis.add_geometry(self._origin_axis, reset_bounding_box=False)

    def remove_bounding_box(self, name: str):
        """移除包围框辅助对象"""
        if name in self.helpers["bboxes"]:
            self.vis.remove_geometry(self.helpers["bboxes"].pop(name), reset_bounding_box=False)
            return

    def add_bounding_box(self, name: str, geometry):
        """添加包围框辅助对象"""
        self.add_helper_geometry(name, geometry, helper_type="bboxes")

    def add_select_point(self, name: str, coord: np.ndarray):
        """添加选中点辅助对象"""
        self._select_points.append(coord)
        dynamic_radius = self._get_dynamic_marker_radius()

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=dynamic_radius)
        sphere.paint_uniform_color(np.array([1, 0, 0]))  # 红色标记
        sphere.translate(coord)
        self.add_helper_geometry(name, sphere, helper_type="select_points")

    # endregion

    def clear_geometries(self):
        """清空核心点云数据，保留辅助对象"""
        names = list(self._geometries.keys())
        for n in names:
            self.remove_geometry(n)

    # region 视角控制
    def _get_all_visible_geometries(self):
        """获取所有已加载的几何体列表"""
        return list(self._geometries.values()) + list(
            [item for sublist in self._helpers_data.values() for item in sublist.values()]
        )

    def _calculate_combined_bounds_center(self):
        """计算所有几何体的合并包围盒中心"""
        all_geos = self._get_all_visible_geometries()
        if not all_geos:
            return np.array([0.0, 0.0, 0.0])

        # 提取所有物体的 min_bound 和 max_bound
        mins = [g.get_min_bound() for g in all_geos]
        maxs = [g.get_max_bound() for g in all_geos]

        global_min = np.min(mins, axis=0)
        global_max = np.max(maxs, axis=0)
        return (global_min + global_max) / 2.0

    def _calculate_combined_bounds(self):
        """计算所有几何体的合并包围盒尺寸"""
        all_geos = self._get_all_visible_geometries()
        if not all_geos:
            return np.array([0.0, 0.0, 0.0])

        # 提取所有物体的 min_bound 和 max_bound
        mins = [g.get_min_bound() for g in all_geos]
        maxs = [g.get_max_bound() for g in all_geos]

        global_min = np.min(mins, axis=0)
        global_max = np.max(maxs, axis=0)
        return global_max - global_min

    def _calculate_combined_center(self):
        """基于所有（包含辅助对象）几何体计算合并中心"""
        all_objs = list(self._geometries.values())
        for cat in self._helpers_data.values():
            all_objs.extend(list(cat.values()))

        if not all_objs:
            return np.array([0, 0, 0])

        mins = [obj.get_min_bound() for obj in all_objs]
        maxs = [obj.get_max_bound() for obj in all_objs]
        return (np.min(mins, axis=0) + np.max(maxs, axis=0)) / 2.0

    def set_standard_view(self, view_name: str, zoom: float = None):
        """设置标准化视角"""
        view_ctl = self.vis.get_view_control()
        if not view_ctl:
            return

        center = self._calculate_combined_center()
        zoom = zoom or self.default_zoom

        views = {
            "front": {"front": [0, 0, -1], "up": [0, 1, 0]},
            "back": {"front": [0, 0, 1], "up": [0, 1, 0]},
            "top": {"front": [0, -1, 0], "up": [0, 0, -1]},
            "bottom": {"front": [0, 1, 0], "up": [0, 0, 1]},
            "left": {"front": [1, 0, 0], "up": [0, 1, 0]},
            "right": {"front": [-1, 0, 0], "up": [0, 1, 0]},
            "iso": {"front": [-1, -1, -1], "up": [0, 1, 0]},
        }

        conf = views.get(view_name.lower(), views["front"])
        view_ctl.set_lookat(center)
        view_ctl.set_front(np.array(conf["front"]))
        view_ctl.set_up(np.array(conf["up"]))
        view_ctl.set_zoom(zoom)

        # 设置一个较大的远裁剪面，防止深度图在远处被截断
        view_ctl.set_constant_z_far(2000.0)
        view_ctl.set_constant_z_near(0.1)

        self.vis.poll_events()
        self.vis.update_renderer()

    def reset_view(self):
        self.set_standard_view("front")

    # endregion

    # region 点选功能实现

    def pick_point_from_depth(self, x, y):
        """利用深度缓冲区实现精准点选"""
        if self._is_cleaned:
            return

        try:
            # 获取深度图
            depth_img = self.vis.capture_depth_float_buffer(do_render=True)
            depth_data = np.asarray(depth_img)

            h, w = depth_data.shape
            if x >= w or y >= h or x < 0 or y < 0:
                return

            z_depth = depth_data[y, x]

            # 【修复点选错误】
            # 这里的 z_depth 是物理深度（如 340.47mm）。
            # 背景的深度通常被 Open3D 设为 0 或者非常大的远裁剪面值。
            # 我们只需要排除 0 或无效值。
            if z_depth <= 0.0:
                # logger.debug(f"点击位置无深度信息 (背景)")
                return

            # 如果 z_depth 特别大（接近远裁剪面），也视为背景
            # 这里取 999 只是一个经验值，取决于 set_constant_z_far 的设置
            if z_depth > 2000.0:
                # logger.debug(f"点击位置在远裁剪面之外 (z={z_depth})")
                return

            view_ctl = self.vis.get_view_control()
            cam_params = view_ctl.convert_to_pinhole_camera_parameters()
            intrinsic = cam_params.intrinsic.intrinsic_matrix
            extrinsic = cam_params.extrinsic

            # 反求 3D 坐标逻辑 (保持数学准确性)
            z = z_depth
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]

            cam_x = (x - cx) * z / fx
            cam_y = (y - cy) * z / fy
            cam_pos = np.array([cam_x, cam_y, z, 1.0])

            world_pos = np.linalg.inv(extrinsic) @ cam_pos
            actual_xyz = world_pos[:3]

            self._add_selection_mark(actual_xyz)
            logger.debug(f"点选成功：{actual_xyz}")

        except Exception as e:
            logger.error(f"深度拾取异常：{e}")

    def _get_dynamic_marker_radius(self):
        """根据当前场景所有物体的包围盒动态计算标记球半径"""
        all_geos = self._get_all_visible_geometries()
        if not all_geos:
            return 0.01

        # 计算合并后的包围盒
        mins = [g.get_min_bound() for g in all_geos]
        maxs = [g.get_max_bound() for g in all_geos]
        global_min = np.min(mins, axis=0)
        global_max = np.max(maxs, axis=0)

        # 取包围盒对角线长度的 0.5% 作为半径
        extent = np.linalg.norm(global_max - global_min)
        return extent * 0.005 if extent > 0 else 0.01

    def _add_selection_mark(self, coord):
        """添加辅助标记球"""
        name = f"mark_{uuid.uuid4().hex[:4]}"
        self.add_select_point(name, coord)

        # 强制刷新界面
        self.vis.poll_events()
        self.vis.update_renderer()

    def clear_select_points(self):
        """清除所有选中标记"""
        for name in list(self._helpers_data["select_points"].keys()):
            self.remove_geometry(name)
        self._select_points.clear()

    def get_select_points(self):
        res = self._select_points.copy()
        self.clear_select_points()
        return res

    # endregion

    def set_background_color(self, r, g, b):
        self.vis.get_render_option().background_color = np.asarray([r, g, b])

    def set_point_size(self, size):
        self.vis.get_render_option().point_size = size
        self.vis.get_render_option().point_size = size
