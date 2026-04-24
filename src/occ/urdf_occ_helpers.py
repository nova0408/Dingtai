from __future__ import annotations

import math
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


MUJOCO_MAX_STL_FACES = 200000


@lru_cache(maxsize=1)
def _pypinyin_lazy_pinyin():
    try:
        from pypinyin import lazy_pinyin
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少依赖 `pypinyin`。请先安装：python -m pip install pypinyin") from exc
    return lazy_pinyin


def is_readable_identifier(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value))


def to_ascii_identifier(raw: str, *, default: str = "mesh") -> str:
    """Convert text to readable ASCII identifier.

    - ASCII letters/digits/underscore are preserved (lower-cased).
    - Chinese characters are converted with pypinyin.
    - Other separators are normalized to underscore boundaries.
    """
    parts: list[str] = []
    token: list[str] = []
    has_non_ascii = any(ord(ch) > 127 for ch in raw)
    lazy_pinyin = _pypinyin_lazy_pinyin() if has_non_ascii else None

    def flush() -> None:
        if token:
            parts.append("".join(token))
            token.clear()

    for ch in raw.strip():
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            token.append(ch.lower())
            continue

        if lazy_pinyin is not None and "\u4e00" <= ch <= "\u9fff":
            flush()
            pinyin = lazy_pinyin(ch, errors="ignore")
            if pinyin:
                parts.extend([item.lower() for item in pinyin if item])
            continue

        if ch in {" ", "-", ".", "/", "\\", "(", ")", "[", "]"}:
            flush()
            continue
        flush()

    flush()
    value = "_".join(filter(None, parts))
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = default
    if value[0].isdigit():
        value = f"n_{value}"
    return value


@dataclass(frozen=True)
class UrdfAssemblyJoint:
    """URDF joint definition."""

    name: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class UrdfAssemblyLink:
    """URDF link definition."""

    name: str
    mesh_ref: str | None


@dataclass(frozen=True)
class UrdfAssemblyData:
    """URDF assembly tree data."""

    robot_name: str
    root_link: str
    links: dict[str, UrdfAssemblyLink]
    joints: list[UrdfAssemblyJoint]
    children_by_parent: dict[str, list[UrdfAssemblyJoint]]


@dataclass(frozen=True)
class OccDisplayNode:
    """A display node for OCC rendering."""

    link_name: str
    mesh_path: Path
    world_trsf: Any


@dataclass(frozen=True)
class OccAssemblySummary:
    """Summary of OCC assembly rendering."""

    robot_name: str
    links_count: int
    joints_count: int
    nodes_count: int


def _parse_float_triplet(raw: str | None) -> tuple[float, float, float]:
    if not raw:
        return 0.0, 0.0, 0.0
    values = [float(x) for x in raw.split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got: {raw}")
    return values[0], values[1], values[2]


def _rpy_to_rotation_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def make_trsf_from_xyz_rpy(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
):
    """Build OCC gp_Trsf from URDF xyz/rpy."""
    from OCC.Core.gp import gp_Trsf

    rotation = _rpy_to_rotation_matrix(rpy)
    trsf = gp_Trsf()
    trsf.SetValues(
        float(rotation[0, 0]),
        float(rotation[0, 1]),
        float(rotation[0, 2]),
        float(xyz[0]),
        float(rotation[1, 0]),
        float(rotation[1, 1]),
        float(rotation[1, 2]),
        float(xyz[1]),
        float(rotation[2, 0]),
        float(rotation[2, 1]),
        float(rotation[2, 2]),
        float(xyz[2]),
    )
    return trsf


def compose_trsf(parent, local):
    """Compose transform with non-mutating API when possible."""
    from OCC.Core.gp import gp_Trsf

    if hasattr(parent, "Multiplied"):
        return parent.Multiplied(local)

    # Compatibility fallback.
    world = gp_Trsf(parent)
    world.Multiply(local)
    return world


def resolve_mesh_path(mesh_ref: str, mesh_dir: Path) -> Path:
    """Resolve URDF mesh reference to local STL path."""
    basename = Path(mesh_ref).name

    direct = mesh_dir / basename
    if direct.exists():
        return direct

    lower_map = {p.name.lower(): p for p in mesh_dir.glob("*.STL")}
    found = lower_map.get(basename.lower())
    if found is not None:
        return found

    raise FileNotFoundError(f"Mesh not found for reference: {mesh_ref}")


def read_urdf_assembly(urdf_path: Path) -> UrdfAssemblyData:
    """Read URDF and build assembly tree data."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"URDF root must be robot, got '{root.tag}'")

    robot_name = root.attrib.get("name", urdf_path.stem)
    links: dict[str, UrdfAssemblyLink] = {}

    for link_elem in root.findall("link"):
        name = link_elem.attrib["name"]
        visual_mesh: str | None = None

        visual = link_elem.find("visual")
        if visual is not None:
            mesh_elem = visual.find("geometry/mesh")
            if mesh_elem is not None:
                visual_mesh = mesh_elem.attrib.get("filename")

        if not visual_mesh:
            collision = link_elem.find("collision")
            if collision is not None:
                mesh_elem = collision.find("geometry/mesh")
                if mesh_elem is not None:
                    visual_mesh = mesh_elem.attrib.get("filename")

        links[name] = UrdfAssemblyLink(name=name, mesh_ref=visual_mesh)

    joints: list[UrdfAssemblyJoint] = []
    children_by_parent: dict[str, list[UrdfAssemblyJoint]] = {}
    child_links: set[str] = set()

    for joint_elem in root.findall("joint"):
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue

        origin_elem = joint_elem.find("origin")
        xyz = _parse_float_triplet(origin_elem.attrib.get("xyz") if origin_elem is not None else None)
        rpy = _parse_float_triplet(origin_elem.attrib.get("rpy") if origin_elem is not None else None)

        joint = UrdfAssemblyJoint(
            name=joint_elem.attrib.get("name", f"joint_{len(joints)}"),
            parent=parent_elem.attrib["link"],
            child=child_elem.attrib["link"],
            xyz=xyz,
            rpy=rpy,
        )
        joints.append(joint)
        children_by_parent.setdefault(joint.parent, []).append(joint)
        child_links.add(joint.child)

    root_candidates = [name for name in links if name not in child_links]
    if not root_candidates:
        raise ValueError("Cannot find root link in URDF")

    return UrdfAssemblyData(
        robot_name=robot_name,
        root_link=root_candidates[0],
        links=links,
        joints=joints,
        children_by_parent=children_by_parent,
    )


def build_occ_display_nodes(
    assembly: UrdfAssemblyData,
    mesh_dir: Path,
) -> list[OccDisplayNode]:
    """Flatten assembly tree into OCC display nodes."""
    from OCC.Core.gp import gp_Trsf

    nodes: list[OccDisplayNode] = []

    def walk(link_name: str, parent_world) -> None:
        link = assembly.links[link_name]
        if link.mesh_ref:
            mesh_path = resolve_mesh_path(link.mesh_ref, mesh_dir)
            nodes.append(OccDisplayNode(link_name=link_name, mesh_path=mesh_path, world_trsf=parent_world))

        for joint in assembly.children_by_parent.get(link_name, []):
            local = make_trsf_from_xyz_rpy(joint.xyz, joint.rpy)
            child_world = compose_trsf(parent_world, local)
            walk(joint.child, child_world)

    walk(assembly.root_link, gp_Trsf())
    return nodes


def load_stl_shape(mesh_path: Path):
    """Load STL as OCC shape."""
    from OCC.Core.StlAPI import StlAPI_Reader
    from OCC.Core.TopoDS import TopoDS_Shape

    shape = TopoDS_Shape()
    reader = StlAPI_Reader()
    ok = reader.Read(shape, str(mesh_path))
    if not ok:
        raise RuntimeError(f"Failed to read STL: {mesh_path}")
    return shape


def load_step_shape(mesh_path: Path):
    """Load STEP as OCC shape."""
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(mesh_path))
    if status != 1:
        raise RuntimeError(f"Failed to read STEP: {mesh_path}")
    ok = reader.TransferRoots()
    if ok <= 0:
        raise RuntimeError(f"Failed to transfer STEP roots: {mesh_path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"STEP produced null shape: {mesh_path}")
    return shape


def load_iges_shape(mesh_path: Path):
    """Load IGES as OCC shape."""
    from OCC.Core.IGESControl import IGESControl_Reader

    reader = IGESControl_Reader()
    status = reader.ReadFile(str(mesh_path))
    if status != 1:
        raise RuntimeError(f"Failed to read IGES: {mesh_path}")
    ok = reader.TransferRoots()
    if ok <= 0:
        raise RuntimeError(f"Failed to transfer IGES roots: {mesh_path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError(f"IGES produced null shape: {mesh_path}")
    return shape


def load_mesh_shape(mesh_path: Path):
    """Load mesh/cad file as OCC shape by suffix."""
    suffix = mesh_path.suffix.lower()
    if suffix == ".stl":
        return load_stl_shape(mesh_path)
    if suffix in {".step", ".stp"}:
        return load_step_shape(mesh_path)
    if suffix in {".iges", ".igs"}:
        return load_iges_shape(mesh_path)
    raise ValueError(f"Unsupported mesh/cad suffix: {mesh_path.suffix}")


def write_shape_to_stl(
    shape,
    output_path: Path,
    linear_deflection: float = 0.5,
    angular_deflection: float = 0.5,
    unit_scale: float = 1.0,
) -> None:
    """Triangulate OCC shape and write STL."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape_to_write = scale_shape(shape, unit_scale)

    mesher = BRepMesh_IncrementalMesh(shape_to_write, float(linear_deflection), False, float(angular_deflection), True)
    mesher.Perform()
    writer = StlAPI_Writer()
    # MuJoCo only supports binary STL reliably; force binary output when OCC supports the toggle.
    if hasattr(writer, "SetASCIIMode"):
        writer.SetASCIIMode(False)
    ok = writer.Write(shape_to_write, str(output_path))
    if not ok:
        raise RuntimeError(f"Failed to write STL: {output_path}")


def count_stl_faces(stl_path: Path) -> tuple[int, str]:
    """Count STL facets and guess format.

    Returns (faces, format) where format in {"binary", "ascii"}.
    """
    raw = stl_path.read_bytes()
    if len(raw) >= 84:
        try:
            nfaces = struct.unpack("<I", raw[80:84])[0]
            expected = 84 + 50 * int(nfaces)
            if expected == len(raw) and nfaces >= 0:
                return int(nfaces), "binary"
        except Exception:
            pass
    # Fallback ASCII counting.
    text = raw.decode("utf-8", errors="ignore").lower()
    faces = text.count("facet normal")
    return int(faces), "ascii"


def _ascii_stl_to_binary(source_path: Path, target_path: Path) -> int:
    """Convert ASCII STL to binary STL without OCC dependency.

    Returns facet count.
    """
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    facets: list[tuple[tuple[float, float, float], list[tuple[float, float, float]]]] = []
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].lower()
        if not ln.startswith("facet normal"):
            i += 1
            continue
        parts = lines[i].split()
        if len(parts) < 5:
            i += 1
            continue
        try:
            normal = (float(parts[-3]), float(parts[-2]), float(parts[-1]))
        except Exception:
            normal = (0.0, 0.0, 0.0)
        verts: list[tuple[float, float, float]] = []
        j = i + 1
        while j < n and len(verts) < 3:
            vln = lines[j].strip().lower()
            if vln.startswith("vertex"):
                vp = lines[j].split()
                if len(vp) >= 4:
                    try:
                        verts.append((float(vp[-3]), float(vp[-2]), float(vp[-1])))
                    except Exception:
                        pass
            j += 1
        if len(verts) == 3:
            facets.append((normal, verts))
        i = j

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as f:
        header = b"Converted by SimplifyModule (ASCII STL -> Binary STL)"
        f.write(header[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(facets)))
        for normal, verts in facets:
            f.write(struct.pack("<3f", *normal))
            for v in verts:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))
    return len(facets)


def normalize_mesh_for_mujoco(
    source_path: Path,
    target_path: Path,
    *,
    unit_scale: float = 1.0,
    max_faces: int = MUJOCO_MAX_STL_FACES,
) -> tuple[int, str]:
    """Normalize source mesh/cad into MuJoCo-friendly binary STL and validate face count.

    For CAD (STEP/IGES), it will try progressive coarser tessellation when face count is too high.
    For STL, it rewrites into binary STL and validates face count.
    Returns (face_count, format) of written target STL.
    """
    suffix = source_path.suffix.lower()
    if suffix in {".step", ".stp", ".iges", ".igs"}:
        shape = load_mesh_shape(source_path)
        # Progressive tessellation to keep detail when possible while respecting MuJoCo limit.
        for lin_def in (0.2, 0.5, 1.0, 2.0, 5.0):
            write_shape_to_stl(
                shape,
                target_path,
                linear_deflection=lin_def,
                angular_deflection=0.5,
                unit_scale=unit_scale,
            )
            faces, fmt = count_stl_faces(target_path)
            if 1 <= faces <= max_faces:
                return faces, fmt
        faces, fmt = count_stl_faces(target_path)
        raise ValueError(
            f"Mesh faces exceed MuJoCo limit after simplification attempts: faces={faces}, limit={max_faces}, file={target_path.name}"
        )

    # STL source: rewrite as binary STL and validate.
    try:
        shape = load_mesh_shape(source_path)
        write_shape_to_stl(
            shape,
            target_path,
            linear_deflection=0.5,
            angular_deflection=0.5,
            unit_scale=unit_scale,
        )
    except ModuleNotFoundError:
        # Fallback path when OCC is unavailable: keep topology, convert ASCII->binary if needed.
        faces0, fmt0 = count_stl_faces(source_path)
        if fmt0 == "binary":
            if source_path.resolve() != target_path.resolve():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
        else:
            _ascii_stl_to_binary(source_path, target_path)

    faces, fmt = count_stl_faces(target_path)
    if fmt != "binary":
        # Defensive fallback: some OCC builds may still emit ASCII STL.
        _ascii_stl_to_binary(target_path, target_path)
        faces, fmt = count_stl_faces(target_path)
    if not (1 <= faces <= max_faces):
        raise ValueError(f"Mesh faces out of MuJoCo range: faces={faces}, limit={max_faces}, file={target_path.name}")
    return faces, fmt


def normalize_mjcf_stl_assets(
    xml_path: Path,
    *,
    max_faces: int = MUJOCO_MAX_STL_FACES,
) -> list[tuple[Path, int, str]]:
    """Normalize STL assets referenced by MJCF to MuJoCo-friendly binary STL.

    Returns a list of normalized files as tuples:
    (mesh_path, face_count_after, format_after).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "").strip() if compiler is not None else ""
    mesh_root = (xml_path.parent / meshdir).resolve() if meshdir else xml_path.parent.resolve()
    asset = root.find("asset")
    if asset is None:
        return []

    normalized: list[tuple[Path, int, str]] = []
    visited: set[Path] = set()
    for mesh in asset.findall("mesh"):
        file_attr = (mesh.get("file") or "").strip()
        if not file_attr:
            continue
        mesh_path = (mesh_root / file_attr).resolve()
        if mesh_path in visited:
            continue
        visited.add(mesh_path)

        if mesh_path.suffix.lower() != ".stl":
            continue
        if not mesh_path.exists():
            raise FileNotFoundError(f"MJCF mesh file not found: {mesh_path}")

        faces_before, fmt_before = count_stl_faces(mesh_path)
        if fmt_before == "binary" and 1 <= faces_before <= max_faces:
            continue

        tmp_path = mesh_path.with_name(f"{mesh_path.name}.tmp_mujoco_norm")
        try:
            faces_after, fmt_after = normalize_mesh_for_mujoco(
                mesh_path,
                tmp_path,
                unit_scale=1.0,
                max_faces=max_faces,
            )
            tmp_path.replace(mesh_path)
            normalized.append((mesh_path, faces_after, fmt_after))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return normalized


def normalize_mjcf_mesh_filenames(xml_path: Path) -> list[tuple[str, str]]:
    """Normalize MJCF asset mesh filenames to ASCII names and rename files on disk.

    Returns list of (old_file_attr, new_file_attr) changes.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "").strip() if compiler is not None else ""
    mesh_root = (xml_path.parent / meshdir).resolve() if meshdir else xml_path.parent.resolve()
    asset = root.find("asset")
    if asset is None:
        return []

    file_entries: list[tuple[ET.Element, str]] = []
    for mesh in asset.findall("mesh"):
        file_attr = (mesh.get("file") or "").strip()
        if file_attr:
            file_entries.append((mesh, file_attr))

    if not file_entries:
        return []

    used: set[str] = {Path(file_attr).name.lower() for _, file_attr in file_entries}
    planned: list[tuple[ET.Element, str, str]] = []
    for mesh, file_attr in file_entries:
        file_name = Path(file_attr).name
        suffix = Path(file_name).suffix or ".stl"
        stem = Path(file_name).stem
        ascii_stem = to_ascii_identifier(stem, default="mesh")
        candidate = f"{ascii_stem}{suffix.lower()}"
        if file_name.isascii() and re.fullmatch(r"[A-Za-z0-9_.-]+", file_name):
            continue

        used.discard(file_name.lower())
        final_name = candidate
        idx = 1
        while final_name.lower() in used:
            final_name = f"{ascii_stem}_x{idx}{suffix.lower()}"
            idx += 1
        used.add(final_name.lower())
        planned.append((mesh, file_attr, final_name))

    if not planned:
        return []

    changed: list[tuple[str, str]] = []
    for mesh, old_attr, new_name in planned:
        old_name = Path(old_attr).name
        old_path = (mesh_root / old_name).resolve()
        new_path = (mesh_root / new_name).resolve()
        if old_path.exists() and old_path != new_path:
            if new_path.exists():
                old_path.unlink()
            else:
                old_path.rename(new_path)
        mesh.set("file", new_name)
        changed.append((old_attr, new_name))

    ET.indent(tree, space="  ")  # type: ignore[attr-defined]
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return changed


def normalize_mjcf_mesh_asset_names(xml_path: Path) -> list[tuple[str, str]]:
    """Normalize MJCF mesh asset names to readable names derived from filenames.

    This updates `<asset><mesh name=...>` and all referencing `geom@mesh`.
    Returns list of (old_name, new_name).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        return []

    mesh_elems = [mesh for mesh in asset.findall("mesh") if (mesh.get("name") or "").strip()]
    if not mesh_elems:
        return []

    mesh_usage_groups: dict[str, set[int]] = {}
    for geom in root.findall(".//geom"):
        mesh_name = (geom.get("mesh") or "").strip()
        if not mesh_name:
            continue
        try:
            group_id = int((geom.get("group") or "0").strip())
        except Exception:
            group_id = 0
        mesh_usage_groups.setdefault(mesh_name, set()).add(group_id)

    def suffix_for_groups(groups: set[int], old_name: str) -> str:
        lowered = old_name.lower()
        if "collision_soft" in lowered:
            return "collision_soft"
        if "collision_hard" in lowered or "collision" in lowered:
            return "collision_hard"
        if "visual" in lowered:
            return "visual"
        if groups == {5}:
            return "collision_soft"
        if groups == {3}:
            return "collision_hard"
        if groups == {0}:
            return "visual"
        return "mesh"

    used: set[str] = {((mesh.get("name") or "").strip()).lower() for mesh in mesh_elems}
    remap: dict[str, str] = {}
    for mesh in mesh_elems:
        old_name = (mesh.get("name") or "").strip()
        file_attr = (mesh.get("file") or "").strip()
        if not old_name or not file_attr:
            continue

        old_is_generic = old_name.startswith("mesh_") or old_name == "mesh" or old_name.endswith("_mesh")
        if is_readable_identifier(old_name) and not old_is_generic:
            continue

        base = to_ascii_identifier(Path(file_attr).stem, default="mesh")
        suffix = suffix_for_groups(mesh_usage_groups.get(old_name, set()), old_name)
        candidate_base = f"{base}_{suffix}"

        used.discard(old_name.lower())
        candidate = candidate_base
        idx = 1
        while candidate.lower() in used:
            candidate = f"{candidate_base}_x{idx}"
            idx += 1
        used.add(candidate.lower())
        remap[old_name] = candidate

    if not remap:
        return []

    for mesh in mesh_elems:
        old_name = (mesh.get("name") or "").strip()
        new_name = remap.get(old_name)
        if new_name:
            mesh.set("name", new_name)

    for geom in root.findall(".//geom"):
        old_mesh = (geom.get("mesh") or "").strip()
        new_mesh = remap.get(old_mesh)
        if new_mesh:
            geom.set("mesh", new_mesh)

    ET.indent(tree, space="  ")  # type: ignore[attr-defined]
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return sorted(remap.items(), key=lambda x: x[0])


def compute_shape_aabb_diag(shape) -> float:
    """Compute diagonal length of shape AABB."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    dx = max(0.0, xmax - xmin)
    dy = max(0.0, ymax - ymin)
    dz = max(0.0, zmax - zmin)
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def compute_shape_aabb_extents(shape) -> tuple[float, float, float]:
    """Compute axis-aligned extents of shape bounding box."""
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        float(max(0.0, xmax - xmin)),
        float(max(0.0, ymax - ymin)),
        float(max(0.0, zmax - zmin)),
    )


def scale_shape(shape, unit_scale: float):
    """Return scaled copy of shape."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Pnt, gp_Trsf

    if abs(float(unit_scale) - 1.0) <= 1e-12:
        return shape
    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), float(unit_scale))
    return BRepBuilderAPI_Transform(shape, scale_trsf, True).Shape()


def transform_shape(shape, trsf):
    """Return transformed copy of shape."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    return BRepBuilderAPI_Transform(shape, trsf, True).Shape()


def trihedron_size_from_shape(shape, minimum: float = 0.001) -> float:
    """Estimate trihedron size by mesh AABB."""
    diag = compute_shape_aabb_diag(shape)
    if diag <= 1e-9:
        return minimum
    return max(minimum, diag * 0.2)


def display_nodes_on_occ_canvas(
    occ_widget,
    nodes: list[OccDisplayNode],
    show_origins: bool = True,
) -> None:
    """Display assembly nodes and origins on OCC canvas."""
    from OCC.Core.AIS import AIS_Shape
    from src.occ.occ_viewer import createTrihedron

    occ_widget.EraseAll()
    context = occ_widget.viewer3d.Context

    for node in nodes:
        shape = load_stl_shape(node.mesh_path)
        ais_shape = AIS_Shape(shape)
        ais_shape.SetLocalTransformation(node.world_trsf)
        context.Display(ais_shape, False)

        if show_origins:
            trihedron_size = trihedron_size_from_shape(shape)
            trihedron = createTrihedron(node.world_trsf, arrow_length=trihedron_size)
            context.Display(trihedron, False)

    occ_widget.viewer3d.FitAll()
    context.UpdateCurrentViewer()


def display_single_node_on_occ_canvas(
    occ_widget,
    node: OccDisplayNode,
    show_origins: bool = True,
    fit_all: bool = False,
    refresh_viewer: bool = False,
) -> object:
    """Display one assembly node on OCC canvas and optionally refresh view."""
    from OCC.Core.AIS import AIS_Shape
    from src.occ.occ_viewer import createTrihedron

    context = occ_widget.viewer3d.Context
    shape = load_stl_shape(node.mesh_path)
    ais_shape = AIS_Shape(shape)
    ais_shape.SetLocalTransformation(node.world_trsf)
    context.Display(ais_shape, False)

    if show_origins:
        trihedron_size = trihedron_size_from_shape(shape)
        trihedron = createTrihedron(node.world_trsf, arrow_length=trihedron_size)
        context.Display(trihedron, False)

    if fit_all:
        occ_widget.viewer3d.FitAll()
    elif refresh_viewer:
        context.UpdateCurrentViewer()
    return ais_shape


def prepare_node_payload(node: OccDisplayNode) -> tuple[OccDisplayNode, object, float]:
    """Preload shape data for a node in background thread."""
    shape = load_stl_shape(node.mesh_path)
    trihedron_size = trihedron_size_from_shape(shape)
    return node, shape, trihedron_size


def display_prepared_node_on_occ_canvas(
    occ_widget,
    payload: tuple[OccDisplayNode, object, float],
    show_origins: bool = True,
    fit_all: bool = False,
    refresh_viewer: bool = False,
) -> object:
    """Display a preloaded node payload on OCC canvas."""
    from OCC.Core.AIS import AIS_Shape
    from src.occ.occ_viewer import createTrihedron

    node, shape, trihedron_size = payload
    context = occ_widget.viewer3d.Context

    ais_shape = AIS_Shape(shape)
    ais_shape.SetLocalTransformation(node.world_trsf)
    context.Display(ais_shape, False)

    if show_origins:
        trihedron = createTrihedron(node.world_trsf, arrow_length=trihedron_size)
        context.Display(trihedron, False)

    if fit_all:
        occ_widget.viewer3d.FitAll()
    elif refresh_viewer:
        context.UpdateCurrentViewer()
    return ais_shape


def make_occ_summary(assembly: UrdfAssemblyData, nodes: list[OccDisplayNode]) -> OccAssemblySummary:
    """Build OCC assembly summary."""
    return OccAssemblySummary(
        robot_name=assembly.robot_name,
        links_count=len(assembly.links),
        joints_count=len(assembly.joints),
        nodes_count=len(nodes),
    )
