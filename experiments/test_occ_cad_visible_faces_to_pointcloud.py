from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from OCC.Core.AIS import AIS_Shape
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import topexp
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import TopTools_IndexedMapOfShape

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.occ.urdf_occ_helpers import load_mesh_shape
from src.occ.viewer_widget import qtViewer3dWidget


DEFAULT_CAD_PATH = Path("experiments/real_pallet_cad/物料板-大.STEP")
DEFAULT_POINT_DENSITY = 0.05
DEFAULT_LINEAR_DEFLECTION = 1.0
DEFAULT_ANGULAR_DEFLECTION = 0.5


@dataclass(frozen=True)
class CliArgs:
    cad_path: Path
    point_density: float
    linear_deflection: float
    angular_deflection: float


class FaceSelectorWindow(QWidget):
    def __init__(self, args: CliArgs) -> None:
        super().__init__()
        self.setWindowTitle("CAD 外表面选面、预览并导出点云")
        self.resize(1560, 940)

        self._args = args
        self._shape = load_mesh_shape(args.cad_path)
        self._face_map = TopTools_IndexedMapOfShape()
        _map_shape_faces(self._shape, self._face_map)

        self._base_shape_ais: AIS_Shape | None = None
        self._saved_face_ids: list[int] = []
        self._saved_face_ais: dict[int, AIS_Shape] = {}
        self._preview_points: np.ndarray | None = None
        self._pending_face_ids: list[int] = []
        self._active_saved_face_ids: set[int] = set()
        self._syncing_list_selection = False

        self._viewer = qtViewer3dWidget(view_cube=True, view_trihedron=False, origin_trihedron=False, enable_multiply_select=True)
        self._viewer.signal_AISs_selected.connect(self._on_view_selection_changed)

        self._status_label = QLabel("")
        self._saved_list = QListWidget()
        self._saved_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._saved_list.itemSelectionChanged.connect(self._on_list_selection_changed)

        self._point_density_spin = QDoubleSpinBox()
        self._point_density_spin.setRange(0.000001, 1000.0)
        self._point_density_spin.setDecimals(6)
        self._point_density_spin.setSingleStep(0.005)
        self._point_density_spin.setValue(self._args.point_density)

        self._linear_deflection_spin = QDoubleSpinBox()
        self._linear_deflection_spin.setRange(0.001, 1000.0)
        self._linear_deflection_spin.setDecimals(4)
        self._linear_deflection_spin.setSingleStep(0.1)
        self._linear_deflection_spin.setValue(self._args.linear_deflection)

        self._angular_deflection_spin = QDoubleSpinBox()
        self._angular_deflection_spin.setRange(0.001, 3.14159)
        self._angular_deflection_spin.setDecimals(4)
        self._angular_deflection_spin.setSingleStep(0.05)
        self._angular_deflection_spin.setValue(self._args.angular_deflection)

        self._btn_save_selected = QPushButton("保存当前选中面到列表 (S)")
        self._btn_clear_faces = QPushButton("清空已保存面")
        self._btn_preview = QPushButton("预览点云 (Enter)")
        self._btn_save_cloud = QPushButton("保存点云 (Save)")
        self._btn_quit = QPushButton("退出 (Esc)")

        self._btn_save_selected.clicked.connect(self._save_current_selection_to_list)
        self._btn_clear_faces.clicked.connect(self._clear_saved_faces)
        self._btn_preview.clicked.connect(self._preview_pointcloud)
        self._btn_save_cloud.clicked.connect(self._save_pointcloud)
        self._btn_quit.clicked.connect(self.close)

        self._build_layout()
        self._bind_hotkeys()
        QTimer.singleShot(0, self._init_view)

    def _build_layout(self) -> None:
        side_info = QLabel(
            "流程:\n"
            "1) 鼠标选择面（已切到 FACE 选择模式）。\n"
            "2) 按 S 切换保存状态（已保存则删除）。\n"
            "3) 调参数后按 Enter 重新生成预览。\n"
            "4) 只有按 Save 按钮才会写出点云文件。\n"
            "5) 列表和画布选择会双向联动高亮。"
        )
        side_info.setWordWrap(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(side_info)

        right_layout.addWidget(QLabel("全局点密度 density（点/面积单位）"))
        right_layout.addWidget(self._point_density_spin)

        right_layout.addWidget(QLabel("线性偏差 linear_deflection"))
        right_layout.addWidget(self._linear_deflection_spin)

        right_layout.addWidget(QLabel("角度偏差 angular_deflection"))
        right_layout.addWidget(self._angular_deflection_spin)

        right_layout.addWidget(self._btn_save_selected)
        right_layout.addWidget(self._btn_clear_faces)

        right_layout.addWidget(QLabel("已保存面列表"))
        right_layout.addWidget(self._saved_list, stretch=1)

        right_layout.addWidget(self._btn_preview)
        right_layout.addWidget(self._btn_save_cloud)
        right_layout.addWidget(self._btn_quit)
        right_layout.addWidget(self._status_label)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(380)

        root_layout = QHBoxLayout()
        root_layout.addWidget(self._viewer, stretch=1)
        root_layout.addWidget(right_widget)
        self.setLayout(root_layout)

    def _bind_hotkeys(self) -> None:
        self._viewer.register_key_action(Qt.Key.Key_S, self._save_current_selection_to_list)
        self._viewer.register_key_action(Qt.Key.Key_Return, self._preview_pointcloud)
        self._viewer.register_key_action(Qt.Key.Key_Enter, self._preview_pointcloud)
        self._viewer.register_key_action(Qt.Key.Key_Escape, self.close)

    def _init_view(self) -> None:
        if not self._viewer._inited:
            self._viewer.InitDriver()

        self._viewer.viewer3d.set_selection_mode(TopAbs_FACE)

        self._base_shape_ais = AIS_Shape(self._shape)
        self._base_shape_ais.SetColor(Quantity_Color(0.75, 0.75, 0.75, Quantity_TOC_RGB))
        self._viewer.context.Display(self._base_shape_ais, False)

        self._viewer.viewer3d.FitAll()
        self._viewer.context.UpdateCurrentViewer()
        self._set_status(f"已加载 {self._args.cad_path.name}，总面数: {_indexed_map_size(self._face_map)}，选择模式: FACE")

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)
        logger.info(text)

    def _on_view_selection_changed(self, _selected_ais: list) -> None:
        self._pending_face_ids = self._collect_current_selected_face_ids()
        selected_saved_ids = [face_id for face_id in self._pending_face_ids if face_id in self._saved_face_ids]
        self._sync_list_selection_from_face_ids(selected_saved_ids)
        self._update_active_saved_face_ids()
        if self._pending_face_ids:
            self._set_status(f"当前选中面: {self._pending_face_ids}，按 S 切换保存/删除")
        else:
            self._set_status("当前未选中任何面")

    def _on_list_selection_changed(self) -> None:
        if self._syncing_list_selection:
            return
        self._update_active_saved_face_ids()
        selected_ids = sorted(self._selected_face_ids_from_list())
        if selected_ids:
            self._set_status(f"列表选中面: {selected_ids}")
        else:
            self._set_status("列表未选中任何保存面")

    def _collect_current_selected_face_ids(self) -> list[int]:
        face_ids: list[int] = []
        self._viewer.context.InitSelected()
        while self._viewer.context.MoreSelected():
            selected_shape = self._viewer.context.SelectedShape()
            face_id = int(self._face_map.FindIndex(selected_shape))
            if face_id > 0 and face_id not in face_ids:
                face_ids.append(face_id)
            self._viewer.context.NextSelected()
        face_ids.sort()
        return face_ids

    def _save_current_selection_to_list(self) -> None:
        pending = self._collect_current_selected_face_ids()
        if not pending:
            self._set_status("当前没有选中面，无法切换保存状态")
            return

        added_ids: list[int] = []
        removed_ids: list[int] = []
        for face_id in pending:
            if face_id in self._saved_face_ids:
                self._saved_face_ids.remove(face_id)
                removed_ids.append(face_id)
            else:
                self._saved_face_ids.append(face_id)
                added_ids.append(face_id)

        self._saved_face_ids.sort()
        self._refresh_saved_list()
        remain_selected_ids = [face_id for face_id in pending if face_id in self._saved_face_ids]
        self._sync_list_selection_from_face_ids(remain_selected_ids)
        self._update_active_saved_face_ids()
        self._set_status(
            f"S 切换完成，新增: {sorted(added_ids)}，删除: {sorted(removed_ids)}，当前保存面: {self._saved_face_ids}"
        )

    def _clear_saved_faces(self) -> None:
        self._saved_face_ids.clear()
        self._preview_points = None
        self._active_saved_face_ids.clear()
        self._refresh_saved_list()
        self._refresh_saved_face_visuals()
        self._set_status("已清空保存面列表与预览缓存")

    def _refresh_saved_list(self) -> None:
        selected_ids = self._selected_face_ids_from_list()
        self._saved_list.clear()
        for face_id in self._saved_face_ids:
            item = QListWidgetItem(f"Face #{face_id}")
            item.setData(Qt.ItemDataRole.UserRole, int(face_id))
            self._saved_list.addItem(item)
            if face_id in selected_ids:
                item.setSelected(True)

    def _refresh_saved_face_visuals(self) -> None:
        for _, ais in list(self._saved_face_ais.items()):
            self._viewer.context.Remove(ais, False)
        self._saved_face_ais.clear()

        for face_id in self._saved_face_ids:
            face = self._face_map.FindKey(face_id)
            ais = AIS_Shape(face)
            if face_id in self._active_saved_face_ids:
                ais.SetColor(Quantity_Color(1.0, 0.95, 0.15, Quantity_TOC_RGB))
            else:
                ais.SetColor(Quantity_Color(1.0, 0.35, 0.15, Quantity_TOC_RGB))
            ais.SetTransparency(0.0)
            self._viewer.context.Display(ais, False)
            self._saved_face_ais[face_id] = ais

        self._viewer.context.UpdateCurrentViewer()

    def _selected_face_ids_from_list(self) -> set[int]:
        selected_ids: set[int] = set()
        for item in self._saved_list.selectedItems():
            face_id = item.data(Qt.ItemDataRole.UserRole)
            if face_id is None:
                continue
            selected_ids.add(int(face_id))
        return selected_ids

    def _sync_list_selection_from_face_ids(self, face_ids: list[int]) -> None:
        target = {int(face_id) for face_id in face_ids}
        self._syncing_list_selection = True
        try:
            for i in range(self._saved_list.count()):
                item = self._saved_list.item(i)
                face_id = int(item.data(Qt.ItemDataRole.UserRole))
                item.setSelected(face_id in target)
        finally:
            self._syncing_list_selection = False

    def _update_active_saved_face_ids(self) -> None:
        active_ids = set(self._selected_face_ids_from_list())
        if active_ids == self._active_saved_face_ids:
            return
        self._active_saved_face_ids = active_ids
        self._refresh_saved_face_visuals()

    def _preview_pointcloud(self) -> None:
        if not self._saved_face_ids:
            self._set_status("请先按 S 保存至少一个面")
            return

        point_density = float(self._point_density_spin.value())
        linear_deflection = float(self._linear_deflection_spin.value())
        angular_deflection = float(self._angular_deflection_spin.value())

        points = sample_points_from_selected_faces(
            shape=self._shape,
            face_map=self._face_map,
            selected_face_ids=self._saved_face_ids,
            point_density=point_density,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
        if points.shape[0] == 0:
            self._preview_points = None
            self._set_status("预览失败：没有采样到点")
            return

        self._preview_points = points
        self._set_status(
            "预览已更新: "
            f"faces={len(self._saved_face_ids)}, points={points.shape[0]}, "
            f"density={point_density:.6f}, lin={linear_deflection:.4f}, ang={angular_deflection:.4f}"
        )
        _preview_points_open3d(points)

    def _save_pointcloud(self) -> None:
        if self._preview_points is None or self._preview_points.shape[0] == 0:
            self._set_status("请先按 Enter 生成预览，再保存")
            return

        output_path = self._args.cad_path.with_suffix(".ply")
        write_xyz_ply(output_path, self._preview_points)
        self._set_status(f"已保存点云: {output_path.name}，点数: {self._preview_points.shape[0]}")


def sample_points_from_selected_faces(
    *,
    shape,
    face_map: TopTools_IndexedMapOfShape,
    selected_face_ids: list[int],
    point_density: float,
    linear_deflection: float,
    angular_deflection: float,
) -> np.ndarray:
    mesher = BRepMesh_IncrementalMesh(shape, float(linear_deflection), False, float(angular_deflection), True)
    mesher.Perform()

    tri_pools: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    for face_id in selected_face_ids:
        face = face_map.FindKey(face_id)
        loc = TopLoc_Location()
        tri = _get_face_triangulation(face, loc)
        if tri is None:
            logger.warning(f"Face #{face_id} 没有可用三角网格，跳过")
            continue

        trsf = loc.Transformation()
        verts = np.zeros((tri.NbNodes(), 3), dtype=np.float64)
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i)
            p.Transform(trsf)
            verts[i - 1] = (p.X(), p.Y(), p.Z())

        tris = np.zeros((tri.NbTriangles(), 3), dtype=np.int32)
        for i in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(i).Get()
            tris[i - 1] = (a - 1, b - 1, c - 1)

        a = verts[tris[:, 0]]
        b = verts[tris[:, 1]]
        c = verts[tris[:, 2]]
        cross = np.cross(b - a, c - a)
        tri_area = 0.5 * np.linalg.norm(cross, axis=1)
        valid = tri_area > 1e-12
        if not np.any(valid):
            continue
        tri_pools.append((a[valid], b[valid], c[valid], float(np.sum(tri_area[valid]))))

    if not tri_pools:
        return np.zeros((0, 3), dtype=np.float64)
    total_area = float(sum(item[3] for item in tri_pools))
    total_target = int(round(total_area * float(point_density)))
    if total_target <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    face_areas = np.array([item[3] for item in tri_pools], dtype=np.float64)
    face_counts = _allocate_counts_by_weight(total_target, face_areas)

    out_points: list[np.ndarray] = []
    for idx, face_count in enumerate(face_counts):
        if face_count <= 0:
            continue
        a, b, c, _ = tri_pools[idx]
        pts = _sample_points_from_triangles_quasirandom(a, b, c, int(face_count), seed_offset=idx * 7919)
        if pts.shape[0] > 0:
            out_points.append(pts)

    if not out_points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(out_points, axis=0)


def _sample_points_from_triangles_quasirandom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    target_points: int,
    seed_offset: int,
) -> np.ndarray:
    if a.size == 0 or b.size == 0 or c.size == 0 or target_points <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    cross = np.cross(b - a, c - a)
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(np.sum(tri_area))
    if total_area <= 1e-12:
        return np.zeros((0, 3), dtype=np.float64)

    counts = _allocate_counts_by_weight(int(target_points), tri_area)

    out: list[np.ndarray] = []
    phi_u = 0.6180339887498949
    phi_v = 0.4142135623730950

    for idx, n in enumerate(counts):
        if n <= 0:
            continue
        aa = a[idx]
        bb = b[idx]
        cc = c[idx]

        seq = np.arange(1, n + 1, dtype=np.float64) + float(seed_offset)
        u = np.mod(seq * phi_u, 1.0)
        v = np.mod(seq * phi_v, 1.0)
        sqrt_u = np.sqrt(u)

        pts = (1.0 - sqrt_u)[:, None] * aa + (sqrt_u * (1.0 - v))[:, None] * bb + (sqrt_u * v)[:, None] * cc
        out.append(pts)

    if not out:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(out, axis=0)


def _allocate_counts_by_weight(total_count: int, weights: np.ndarray) -> np.ndarray:
    if total_count <= 0:
        return np.zeros(weights.shape[0], dtype=np.int32)

    w = np.asarray(weights, dtype=np.float64)
    w = np.where(w > 0.0, w, 0.0)
    ws = float(np.sum(w))
    if ws <= 1e-12:
        return np.zeros(weights.shape[0], dtype=np.int32)

    expected = w / ws * float(total_count)
    base = np.floor(expected).astype(np.int32)
    remainder = int(total_count - int(np.sum(base)))
    if remainder > 0:
        frac_order = np.argsort(-(expected - base))
        base[frac_order[:remainder]] += 1
    return base


def write_xyz_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def _map_shape_faces(shape, face_map: TopTools_IndexedMapOfShape) -> None:
    topexp.MapShapes(shape, TopAbs_FACE, face_map)


def _indexed_map_size(shape_map: TopTools_IndexedMapOfShape) -> int:
    return int(shape_map.Size())


def _get_face_triangulation(face, loc: TopLoc_Location):
    return BRep_Tool.Triangulation(face, loc)


def _preview_points_open3d(points: np.ndarray) -> None:
    try:
        import open3d as o3d
    except Exception as exc:
        logger.warning(f"Open3D 预览不可用，仅完成点云数据预览统计: {exc}")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.98, 0.45, 0.12])
    o3d.visualization.draw_geometries([pcd], window_name="点云预览（Enter 重新生成）")


def parse_args() -> CliArgs:
    parser = argparse.ArgumentParser(description="选择 CAD 外表面并导出点云")
    parser.add_argument("--cad", type=Path, default=DEFAULT_CAD_PATH, help="输入 CAD 文件路径（支持 STEP/IGES/STL）")
    parser.add_argument("--point-density", type=float, default=DEFAULT_POINT_DENSITY, help="全局点密度（点/面积单位）")
    parser.add_argument("--linear-deflection", type=float, default=DEFAULT_LINEAR_DEFLECTION, help="OCC 三角化线性偏差")
    parser.add_argument("--angular-deflection", type=float, default=DEFAULT_ANGULAR_DEFLECTION, help="OCC 三角化角度偏差")

    ns = parser.parse_args()
    cad_path = Path(ns.cad)
    if not cad_path.is_absolute():
        cad_path = (PROJECT_ROOT / cad_path).resolve()

    if not cad_path.exists():
        raise FileNotFoundError(f"CAD 文件不存在: {cad_path}")
    if float(ns.point_density) <= 0.0:
        raise ValueError("point-density 必须 > 0")

    return CliArgs(
        cad_path=cad_path,
        point_density=float(ns.point_density),
        linear_deflection=float(ns.linear_deflection),
        angular_deflection=float(ns.angular_deflection),
    )


def main() -> None:
    args = parse_args()
    app = QApplication.instance() or QApplication(sys.argv)
    window = FaceSelectorWindow(args)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
