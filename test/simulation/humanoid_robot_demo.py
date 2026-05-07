from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R
from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from src.robotics.kinematic_models import ArmMountState
from src.robotics.urdf_interface import UrdfConverter, UrdfJoint, UrdfModel
from src.simulation import (
    ArmSimulationBinding,
    ArmSimulationModel,
    ChainSnapshot,
    JointUiSpec,
    KinematicsSimulationWidget,
    SpatialArmKinematics,
)
from src.simulation.protocols import JointAxisGlyph
from src.utils.datas import Axis, Color, Degree, Point, Quaternion, Transform, Translation, Vector

RESOURCE_DIR = PROJECT_ROOT / "resources"
ASSEMBLY_TEMPLATE_PATH = RESOURCE_DIR / "humanoid_assembly.urdf.xacro"
MOVABLE_TYPES = {"revolute", "prismatic", "continuous"}
XACRO_BACKEND = "unknown"


class ResourceAutoReloader(QObject):
    """监听资源文件变化并热重载模型。"""

    def __init__(self, widget: KinematicsSimulationWidget, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._widget = widget
        self._changed_files: set[str] = set()
        self._watcher = QFileSystemWatcher(self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(180)
        self._debounce.timeout.connect(self._reload)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watch_paths = self._collect_watch_paths()
        self._watcher.addPaths(self._watch_paths)

    @staticmethod
    def _collect_watch_paths() -> list[str]:
        paths = [str(ASSEMBLY_TEMPLATE_PATH.resolve())]
        for p in sorted(RESOURCE_DIR.glob("*.urdf")):
            paths.append(str(p.resolve()))
        for p in sorted(RESOURCE_DIR.glob("*.xacro")):
            rp = str(p.resolve())
            if rp not in paths:
                paths.append(rp)
        profile_path = RESOURCE_DIR / "woosh_agv.toml"
        if profile_path.exists():
            rp = str(profile_path.resolve())
            if rp not in paths:
                paths.append(rp)
        return paths

    def _ensure_paths(self) -> None:
        existing = set(self._watcher.files())
        for path in self._watch_paths:
            if Path(path).exists() and path not in existing:
                self._watcher.addPath(path)

    def _on_file_changed(self, path: str) -> None:
        self._changed_files.add(path)
        self._debounce.start()

    @staticmethod
    def _resolve_changed_chains(changed_files: list[str]) -> tuple[str, ...]:
        """根据变更文件推断受影响链名称。"""

        changed_names = {Path(p).name.lower() for p in changed_files}
        impacted: list[str] = []
        try:
            _, configs = _load_assembly_template()
        except Exception:
            return ()
        for cfg in configs:
            if cfg.urdf.lower() in changed_names:
                impacted.append(cfg.name)
        return tuple(impacted)

    def _reload(self) -> None:
        self._ensure_paths()
        changed = sorted(self._changed_files)
        self._changed_files.clear()
        changed_text = ", ".join(Path(p).name for p in changed) if changed else "unknown file"
        changed_chains = self._resolve_changed_chains(changed)
        chain_text = ", ".join(changed_chains) if changed_chains else "unknown"
        try:
            model = build_robot_model()
        except Exception as exc:
            self._widget.set_status_text(f"Reload failed [chains: {chain_text}] ({changed_text}): {exc}")
            print(f"[hot-reload] failed, chains: {chain_text}, changed: {changed_text}, error: {exc}")
            return
        self._widget.replace_model(model, preserve_joint_values=True)
        self._widget.set_status_text(
            f"Reloaded [chains: {chain_text}] ({changed_text}) [xacro: {XACRO_BACKEND}]"
        )
        print(f"[hot-reload] reloaded, chains: {chain_text}, changed: {changed_text}, xacro={XACRO_BACKEND}")


@dataclass(frozen=True, slots=True)
class ChainConfig:
    name: str
    urdf: str
    color: str
    mount_type: str
    parent: str | None
    parent_link: str | None
    mount_translation: tuple[float, float, float]
    base_translation: tuple[float, float, float]
    init_values: tuple[float, ...] | None


@dataclass(slots=True)
class ChainRuntime:
    config: ChainConfig
    model: UrdfModel
    movable_joints: tuple[UrdfJoint, ...]
    movable_index_by_name: dict[str, int]
    children_by_parent: dict[str, list[UrdfJoint]]
    root_links: tuple[str, ...]
    binding: ArmSimulationBinding


def _urdf_origin_transform(joint: UrdfJoint) -> Transform:
    tx, ty, tz = joint.origin.xyz
    rr, rp, ry = joint.origin.rpy
    rot_mat = R.from_euler("xyz", [rr, rp, ry], degrees=False).as_matrix()
    se3 = np.eye(4, dtype=np.float64)
    se3[:3, :3] = rot_mat
    se3[:3, 3] = [tx, ty, tz]
    return Transform.from_SE3(se3)


def _joint_motion_transform(joint: UrdfJoint, joint_value: float) -> Transform:
    if joint.joint_type in {"revolute", "continuous"}:
        q = Quaternion.from_axis_angle(Vector.from_list(joint.axis), joint_value)
        return Transform(translation=Translation.Zero(), rotation=q)
    if joint.joint_type == "prismatic":
        ax, ay, az = joint.axis
        return Transform(translation=Translation(ax * joint_value, ay * joint_value, az * joint_value))
    return Transform.Identity()


def _load_urdf_model(urdf_name: str) -> UrdfModel:
    urdf_path = RESOURCE_DIR / urdf_name
    if urdf_path.suffix == ".xacro":
        urdf_text = _expand_xacro_to_urdf_text(urdf_path)
        return UrdfConverter().from_xml_text(urdf_text)
    return UrdfConverter().from_file(urdf_path)


def _expand_xacro_to_urdf_text(xacro_path: Path) -> str:
    """调用 ROS 官方 xacro 命令展开为 URDF 文本。"""

    cmd = ["xacro", str(xacro_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(RESOURCE_DIR))
    except FileNotFoundError:
        return _expand_xacro_to_urdf_text_fallback(xacro_path)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(f"xacro expand failed for {xacro_path.name}: {stderr}")
    global XACRO_BACKEND
    XACRO_BACKEND = "ros-xacro"
    return proc.stdout


@dataclass(frozen=True, slots=True)
class _XacroMacro:
    param_names: tuple[str, ...]
    defaults: dict[str, str]
    body: tuple[ET.Element, ...]


def _parse_xacro_params(raw: str) -> tuple[tuple[str, ...], dict[str, str]]:
    names: list[str] = []
    defaults: dict[str, str] = {}
    for token in raw.split():
        if ":=" in token:
            key, default = token.split(":=", 1)
            names.append(key)
            defaults[key] = default.strip("'\"")
        else:
            names.append(token)
    return tuple(names), defaults


def _collect_xacro_macros(root: ET.Element, base_dir: Path, macros: dict[str, _XacroMacro]) -> None:
    for child in list(root):
        local = _tag_local_name(child.tag)
        if local == "include":
            inc = child.attrib.get("filename", "")
            if not inc:
                continue
            inc_path = (base_dir / inc).resolve()
            inc_root = ET.fromstring(inc_path.read_text(encoding="utf-8"))
            _collect_xacro_macros(inc_root, inc_path.parent, macros)
        elif local == "macro":
            name = child.attrib.get("name", "").strip()
            params_raw = child.attrib.get("params", "")
            if not name:
                continue
            param_names, defaults = _parse_xacro_params(params_raw)
            macros[name] = _XacroMacro(
                param_names=param_names,
                defaults=defaults,
                body=tuple(ET.fromstring(ET.tostring(elem, encoding="unicode")) for elem in list(child)),
            )


def _subst_text(value: str, ctx: dict[str, str]) -> str:
    out = value
    for key, val in ctx.items():
        out = out.replace("${" + key + "}", val)
    return out


def _subst_element(elem: ET.Element, ctx: dict[str, str]) -> None:
    for k, v in list(elem.attrib.items()):
        elem.attrib[k] = _subst_text(v, ctx)
    for child in list(elem):
        _subst_element(child, ctx)


def _expand_xacro_to_urdf_text_fallback(xacro_path: Path) -> str:
    """轻量 xacro 回退展开器（支持 include + macro + 参数替换）。"""

    root = ET.fromstring(xacro_path.read_text(encoding="utf-8"))
    global XACRO_BACKEND
    XACRO_BACKEND = "builtin-fallback"
    macros: dict[str, _XacroMacro] = {}
    _collect_xacro_macros(root, xacro_path.parent, macros)

    expanded_children: list[ET.Element] = []
    for child in list(root):
        local = _tag_local_name(child.tag)
        if local in {"include", "macro"}:
            continue
        if local in macros:
            macro = macros[local]
            ctx: dict[str, str] = dict(macro.defaults)
            for name in macro.param_names:
                if name in child.attrib:
                    ctx[name] = child.attrib[name]
                elif name not in ctx:
                    ctx[name] = ""
            for body_elem in macro.body:
                cloned = ET.fromstring(ET.tostring(body_elem, encoding="unicode"))
                _subst_element(cloned, ctx)
                expanded_children.append(cloned)
            continue
        expanded_children.append(ET.fromstring(ET.tostring(child, encoding="unicode")))

    urdf_root = ET.Element("robot", {"name": root.attrib.get("name", "expanded_from_xacro")})
    for elem in expanded_children:
        urdf_root.append(elem)
    return ET.tostring(urdf_root, encoding="unicode")


def _movable_joints(model: UrdfModel) -> tuple[UrdfJoint, ...]:
    return tuple(j for j in model.joints if j.joint_type in MOVABLE_TYPES)


def _build_kinematics_from_urdf(chain_name: str, joints: tuple[UrdfJoint, ...]) -> SpatialArmKinematics:
    link_vectors = tuple(joint.origin.xyz for joint in joints)
    joint_axes_local = tuple(joint.axis for joint in joints)
    joint_types = tuple("prismatic" if joint.joint_type == "prismatic" else "revolute" for joint in joints)

    limits: list[tuple[float, float]] = []
    for joint in joints:
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise ValueError(f"joint '{joint.name}' is missing limit")
        if joint.joint_type == "prismatic":
            limits.append((joint.limit.lower, joint.limit.upper))
        else:
            limits.append((math.degrees(joint.limit.lower), math.degrees(joint.limit.upper)))

    return SpatialArmKinematics(
        name=chain_name,
        link_vectors=link_vectors,
        joint_axes_local=joint_axes_local,
        joint_limits=tuple(limits),
        joint_types=joint_types,
    )


def _joint_ui_from_urdf(joints: tuple[UrdfJoint, ...], positions: tuple[float, ...]) -> tuple[JointUiSpec, ...]:
    if len(joints) != len(positions):
        raise ValueError(f"joint count mismatch: urdf_joints={len(joints)}, provided_positions={len(positions)}")

    specs: list[JointUiSpec] = []
    for index, (joint, current) in enumerate(zip(joints, positions, strict=True), start=1):
        if joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise ValueError(f"joint '{joint.name}' is missing limit")
        label = joint.name if joint.name else f"j{index}"
        if joint.joint_type == "prismatic":
            specs.append(JointUiSpec(label, joint.limit.lower, joint.limit.upper, current))
        else:
            specs.append(
                JointUiSpec(
                    label,
                    Degree(math.degrees(joint.limit.lower)),
                    Degree(math.degrees(joint.limit.upper)),
                    Degree(current),
                )
            )
    return tuple(specs)


def _default_joint_positions(joints: tuple[UrdfJoint, ...], init_values: tuple[float, ...] | None) -> tuple[float, ...]:
    if init_values is not None:
        if len(init_values) != len(joints):
            raise ValueError(f"init_values length mismatch: expected={len(joints)} actual={len(init_values)}")
        return init_values

    defaults: list[float] = []
    for joint in joints:
        raw_default = 0.0
        if joint.limit is not None and joint.limit.lower is not None and joint.limit.upper is not None:
            if joint.limit.lower <= 0.0 <= joint.limit.upper:
                raw_default = 0.0
            else:
                raw_default = (joint.limit.lower + joint.limit.upper) * 0.5
        defaults.append(raw_default if joint.joint_type == "prismatic" else math.degrees(raw_default))
    return tuple(defaults)


def _parse_vec3_attr(raw: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    if raw is None or raw.strip() == "":
        return default
    values = raw.strip().split()
    if len(values) != 3:
        raise ValueError(f"vec3 attr expects 3 values, got: {raw}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _tag_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


@dataclass(frozen=True, slots=True)
class XacroChainInstance:
    module_name: str
    chain_name: str
    prefix: str
    parent_link: str
    xyz: tuple[float, float, float]
    color: str
    urdf_source: str


def _parse_xacro_instances(root: ET.Element) -> tuple[XacroChainInstance, ...]:
    instances: list[XacroChainInstance] = []
    for elem in root:
        module_name = _tag_local_name(elem.tag)
        if not module_name.endswith("_module"):
            continue
        prefix = elem.attrib.get("prefix", "")
        parent_link = elem.attrib.get("parent_link", "")
        chain_name = elem.attrib.get("chain_name", "").strip()
        color = elem.attrib.get("color", "#2a9d8f")
        urdf_source = elem.attrib.get("urdf_source", "").strip()
        xyz = _parse_vec3_attr(elem.attrib.get("xyz"))
        if not prefix or not parent_link or not chain_name or not urdf_source:
            continue
        instances.append(
            XacroChainInstance(
                module_name=module_name,
                chain_name=chain_name,
                prefix=prefix,
                parent_link=parent_link,
                xyz=xyz,
                color=color,
                urdf_source=urdf_source,
            )
        )
    if not instances:
        raise ValueError("no xacro module instances found in assembly file")
    return tuple(instances)


def _load_assembly_template() -> tuple[ET.Element, tuple[ChainConfig, ...]]:
    root = ET.fromstring(ASSEMBLY_TEMPLATE_PATH.read_text(encoding="utf-8"))
    instances = _parse_xacro_instances(root)
    configs: list[ChainConfig] = []
    by_prefix = {inst.prefix: inst for inst in instances}

    for inst in instances:
        if inst.parent_link == "world":
            configs.append(
                ChainConfig(
                    name=inst.chain_name,
                    urdf=inst.urdf_source,
                    color=inst.color,
                    mount_type="world",
                    parent=None,
                    parent_link=None,
                    mount_translation=(0.0, 0.0, 0.0),
                    base_translation=inst.xyz,
                    init_values=None,
                )
            )
            continue

        parent_inst: XacroChainInstance | None = None
        parent_prefix = ""
        for prefix, candidate in by_prefix.items():
            if inst.parent_link.startswith(prefix) and len(prefix) > len(parent_prefix):
                parent_inst = candidate
                parent_prefix = prefix
        if parent_inst is None:
            raise ValueError(f"cannot resolve parent chain for {inst.chain_name}, parent_link={inst.parent_link}")

        local_parent_link = inst.parent_link[len(parent_prefix) :] if parent_prefix else inst.parent_link
        mount_type = "tcp" if local_parent_link in {"link6", "tcp", "tool0"} else "link"
        configs.append(
            ChainConfig(
                name=inst.chain_name,
                urdf=inst.urdf_source,
                color=inst.color,
                mount_type=mount_type,
                parent=parent_inst.chain_name,
                parent_link=None if mount_type == "tcp" else local_parent_link,
                mount_translation=inst.xyz,
                base_translation=(0.0, 0.0, 0.0),
                init_values=None,
            )
        )

    return root, tuple(configs)


def _binding_base_transform(binding: ArmSimulationBinding) -> Transform:
    if binding.base_transform_solver is not None:
        return binding.base_transform_solver()
    return binding.base_transform


def _solve_all_link_transforms(
    children_by_parent: dict[str, list[UrdfJoint]],
    root_links: tuple[str, ...],
    movable_index_by_name: dict[str, int],
    joint_positions: tuple[float, ...],
) -> dict[str, Transform]:
    link_transforms: dict[str, Transform] = {root: Transform.Identity() for root in root_links}
    stack: list[str] = list(root_links)
    while stack:
        parent_link = stack.pop()
        parent_tf = link_transforms[parent_link]
        for joint in children_by_parent.get(parent_link, ()):
            local_origin = _urdf_origin_transform(joint)
            joint_index = movable_index_by_name.get(joint.name)
            joint_value = 0.0 if joint_index is None else joint_positions[joint_index]
            motion = _joint_motion_transform(joint, joint_value)
            child_tf = parent_tf @ local_origin @ motion
            link_transforms[joint.child_link] = child_tf
            stack.append(joint.child_link)

    return link_transforms


def _parent_mount_transform(parent_runtime: ChainRuntime, cfg: ChainConfig) -> Transform:
    if cfg.mount_type == "tcp":
        return parent_runtime.binding.arm_model.solve_tcp(parent_runtime.binding.arm_state.joint_positions)

    if cfg.mount_type == "link":
        if cfg.parent_link is None:
            raise ValueError(f"chain '{cfg.name}' mount_type=link requires parent_link")
        link_transforms = _solve_all_link_transforms(
            parent_runtime.children_by_parent,
            parent_runtime.root_links,
            parent_runtime.movable_index_by_name,
            parent_runtime.binding.arm_state.joint_positions,
        )
        if cfg.parent_link not in link_transforms:
            raise ValueError(
                f"chain '{cfg.name}' parent_link '{cfg.parent_link}' not found in parent URDF '{parent_runtime.config.urdf}'"
            )
        return link_transforms[cfg.parent_link]

    raise ValueError(f"unsupported mount_type: {cfg.mount_type}")


def _build_runtime(cfg: ChainConfig, runtimes: dict[str, ChainRuntime]) -> ChainRuntime:
    urdf_model = _load_urdf_model(cfg.urdf)
    joints = _movable_joints(urdf_model)
    kinematics = _build_kinematics_from_urdf(cfg.name, joints)
    joint_positions = _default_joint_positions(joints, cfg.init_values)
    children_by_parent: dict[str, list[UrdfJoint]] = {}
    for joint in urdf_model.joints:
        children_by_parent.setdefault(joint.parent_link, []).append(joint)
    child_links = {joint.child_link for joint in urdf_model.joints}
    root_links = tuple(link.name for link in urdf_model.links if link.name not in child_links)
    if not root_links:
        raise ValueError(f"URDF '{cfg.urdf}' has no root link")

    if cfg.mount_type == "world":
        binding = ArmSimulationBinding(
            chain_name=cfg.name,
            arm_state=ArmMountState(joint_positions=joint_positions),
            arm_model=kinematics,
            base_transform=Transform(translation=Translation(*cfg.base_translation)),
            joint_ui=_joint_ui_from_urdf(joints, joint_positions),
            color=Color.from_hex(cfg.color),
        )
    else:
        if cfg.parent is None:
            raise ValueError(f"chain '{cfg.name}' requires parent for mount_type={cfg.mount_type}")
        if cfg.parent not in runtimes:
            raise ValueError(f"chain '{cfg.name}' parent '{cfg.parent}' not built yet")
        parent_runtime = runtimes[cfg.parent]

        def base_solver() -> Transform:
            parent_base = _binding_base_transform(parent_runtime.binding)
            mount = _parent_mount_transform(parent_runtime, cfg)
            assembly = Transform(translation=Translation(*cfg.mount_translation))
            return parent_base @ mount @ assembly

        binding = ArmSimulationBinding(
            chain_name=cfg.name,
            arm_state=ArmMountState(joint_positions=joint_positions),
            arm_model=kinematics,
            base_transform_solver=base_solver,
            joint_ui=_joint_ui_from_urdf(joints, joint_positions),
            color=Color.from_hex(cfg.color),
        )

    def link_point_solver(current_positions: tuple[float, ...]) -> tuple[Point, ...]:
        local_transforms = _get_cached_local_transforms(current_positions)
        base = _binding_base_transform(binding)
        points: list[Point] = []

        def append_tree(parent_link: str) -> None:
            parent_local = local_transforms[parent_link]
            parent_world = base @ parent_local
            parent_point = parent_world.translation
            for child_joint in children_by_parent.get(parent_link, ()):
                child_local = local_transforms[child_joint.child_link]
                child_world = base @ child_local
                child_point = child_world.translation
                points.append(Point(parent_point.x, parent_point.y, parent_point.z))
                points.append(Point(child_point.x, child_point.y, child_point.z))
                points.append(Point(float("nan"), float("nan"), float("nan")))
                append_tree(child_joint.child_link)

        for root in root_links:
            append_tree(root)
        if not points:
            b = base.translation
            points.append(Point(b.x, b.y, b.z))
        return tuple(points)

    cached_positions: tuple[float, ...] | None = None
    cached_local_transforms: dict[str, Transform] | None = None

    def _get_cached_local_transforms(current_positions: tuple[float, ...]) -> dict[str, Transform]:
        nonlocal cached_positions, cached_local_transforms
        if cached_positions == current_positions and cached_local_transforms is not None:
            return cached_local_transforms
        cached_local_transforms = _solve_all_link_transforms(
            children_by_parent,
            root_links,
            {joint.name: idx for idx, joint in enumerate(joints)},
            current_positions,
        )
        cached_positions = current_positions
        return cached_local_transforms

    def joint_axis_solver(current_positions: tuple[float, ...]) -> tuple[JointAxisGlyph, ...]:
        local_transforms = _get_cached_local_transforms(current_positions)
        base = _binding_base_transform(binding)
        glyphs: list[JointAxisGlyph] = []
        for joint in joints:
            parent_local = local_transforms.get(joint.parent_link)
            if parent_local is None:
                continue
            joint_frame = base @ parent_local @ _urdf_origin_transform(joint)
            axis_world = Vector.from_list(joint.axis).transformed(joint_frame.rotation).normalized()
            t = joint_frame.translation
            glyphs.append(
                JointAxisGlyph(
                    axis=Axis(origin=Point(t.x, t.y, t.z), z_axis=axis_world),
                    label=joint.name,
                )
            )
        return tuple(glyphs)

    binding.link_point_solver = link_point_solver
    binding.joint_axis_solver = joint_axis_solver

    return ChainRuntime(
        config=cfg,
        model=urdf_model,
        movable_joints=joints,
        movable_index_by_name={joint.name: idx for idx, joint in enumerate(joints)},
        children_by_parent=children_by_parent,
        root_links=root_links,
        binding=binding,
    )


def _build_agv_outline(template_root: ET.Element) -> tuple[ChainSnapshot, ...]:
    agv_node = template_root.find("agv")
    if agv_node is None:
        profile_path = RESOURCE_DIR / "woosh_agv.toml"
        agv_color = "#6c757d"
    else:
        profile_path = RESOURCE_DIR / agv_node.attrib.get("profile", "woosh_agv.toml")
        agv_color = agv_node.attrib.get("color", "#6c757d")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib  # type: ignore[import-not-found]
    profile = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    length = float(profile["size"]["length"])
    width = float(profile["size"]["width"])
    install_height = float(profile["mount"]["install_height"])
    color = Color.from_hex(agv_color)

    lx = length * 0.5
    wy = width * 0.5
    return (
        ChainSnapshot(
            chain_name="agv_base",
            points=(
                Point(-lx, -wy, install_height),
                Point(lx, -wy, install_height),
                Point(lx, wy, install_height),
                Point(-lx, wy, install_height),
                Point(-lx, -wy, install_height),
            ),
            color=color,
        ),
    )


def build_robot_model() -> ArmSimulationModel:
    template_root, chain_cfgs = _load_assembly_template()
    runtimes: dict[str, ChainRuntime] = {}

    pending = list(chain_cfgs)
    while pending:
        progressed = False
        next_pending: list[ChainConfig] = []
        for cfg in pending:
            if cfg.mount_type == "world" or (cfg.parent is not None and cfg.parent in runtimes):
                runtime = _build_runtime(cfg, runtimes)
                runtimes[cfg.name] = runtime
                progressed = True
            else:
                next_pending.append(cfg)
        if not progressed:
            unresolved = ", ".join(cfg.name for cfg in next_pending)
            raise ValueError(f"cannot resolve chain dependency order: {unresolved}")
        pending = next_pending

    bindings = {name: runtime.binding for name, runtime in runtimes.items()}
    return ArmSimulationModel(bindings=bindings, static_snapshots=_build_agv_outline(template_root))


def main() -> int:
    pyside6_plugin_path = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6", "plugins", "platforms")
    if os.path.isdir(pyside6_plugin_path):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", pyside6_plugin_path)

    template_root = ET.fromstring(ASSEMBLY_TEMPLATE_PATH.read_text(encoding="utf-8"))
    viewer_node = template_root.find("viewer")
    window_title = "Kinematics Simulator"
    window_width = 1460
    window_height = 920
    if viewer_node is not None:
        window_title = viewer_node.attrib.get("window_title", window_title)
        window_width = int(viewer_node.attrib.get("window_width", str(window_width)))
        window_height = int(viewer_node.attrib.get("window_height", str(window_height)))

    app = QApplication(sys.argv)
    widget = KinematicsSimulationWidget(model=build_robot_model())
    auto_reloader = ResourceAutoReloader(widget, parent=widget)
    widget._auto_reloader = auto_reloader  # type: ignore[attr-defined]
    widget.set_status_text(f"Ready [xacro: {XACRO_BACKEND}]")
    widget.setWindowTitle(window_title)
    widget.resize(window_width, window_height)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
