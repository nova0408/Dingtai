from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R
from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

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
DEMO_CONFIG_PATH = RESOURCE_DIR / "humanoid_robot_demo.toml"
MOVABLE_TYPES = {"revolute", "prismatic", "continuous"}


class ResourceAutoReloader(QObject):
    """监听资源文件变化并热重载模型。"""

    def __init__(self, widget: KinematicsSimulationWidget, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._widget = widget
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
        paths = [str(DEMO_CONFIG_PATH.resolve())]
        for p in sorted(RESOURCE_DIR.glob("*.urdf")):
            paths.append(str(p.resolve()))
        for p in sorted(RESOURCE_DIR.glob("*.toml")):
            rp = str(p.resolve())
            if rp not in paths:
                paths.append(rp)
        return paths

    def _ensure_paths(self) -> None:
        existing = set(self._watcher.files())
        for path in self._watch_paths:
            if Path(path).exists() and path not in existing:
                self._watcher.addPath(path)

    def _on_file_changed(self, _: str) -> None:
        self._debounce.start()

    def _reload(self) -> None:
        self._ensure_paths()
        try:
            model = build_robot_model()
        except Exception as exc:
            self._widget.set_status_text(f"Reload failed: {exc}")
            return
        self._widget.replace_model(model, preserve_joint_values=True)
        self._widget.set_status_text("Reloaded from updated URDF/TOML")


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
    return UrdfConverter().from_file(RESOURCE_DIR / urdf_name)


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


def _load_demo_config() -> tuple[dict[str, object], tuple[ChainConfig, ...]]:
    raw = tomllib.loads(DEMO_CONFIG_PATH.read_text(encoding="utf-8"))
    raw_chains = raw.get("chains")
    if not isinstance(raw_chains, list) or not raw_chains:
        raise ValueError("humanoid_robot_demo.toml must contain non-empty [[chains]]")

    configs: list[ChainConfig] = []
    for item in raw_chains:
        if not isinstance(item, dict):
            raise ValueError("each [[chains]] entry must be a table")

        mount_type = str(item.get("mount_type", "tcp"))
        parent_raw = item.get("parent")
        parent = str(parent_raw) if parent_raw is not None else None
        parent_link_raw = item.get("parent_link")
        parent_link = str(parent_link_raw) if parent_link_raw is not None else None
        init_values_raw = item.get("init_values")
        init_values: tuple[float, ...] | None = None
        if init_values_raw is not None:
            if not isinstance(init_values_raw, list):
                raise ValueError("init_values must be an array when provided")
            init_values = tuple(float(v) for v in init_values_raw)

        mount_translation_raw = item.get("mount_translation", [0.0, 0.0, 0.0])
        base_translation_raw = item.get("base_translation", [0.0, 0.0, 0.0])
        if not isinstance(mount_translation_raw, list) or len(mount_translation_raw) != 3:
            raise ValueError("mount_translation must be [x, y, z]")
        if not isinstance(base_translation_raw, list) or len(base_translation_raw) != 3:
            raise ValueError("base_translation must be [x, y, z]")

        configs.append(
            ChainConfig(
                name=str(item["name"]),
                urdf=str(item["urdf"]),
                color=str(item["color"]),
                mount_type=mount_type,
                parent=parent,
                parent_link=parent_link,
                mount_translation=tuple(float(v) for v in mount_translation_raw),
                base_translation=tuple(float(v) for v in base_translation_raw),
                init_values=init_values,
            )
        )

    return raw, tuple(configs)


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


def _build_agv_outline(raw_cfg: dict[str, object]) -> tuple[ChainSnapshot, ...]:
    agv_cfg = raw_cfg.get("agv")
    if not isinstance(agv_cfg, dict):
        raise ValueError("humanoid_robot_demo.toml missing [agv]")

    profile = tomllib.loads((RESOURCE_DIR / str(agv_cfg.get("profile", "woosh_agv.toml"))).read_text(encoding="utf-8"))
    length = float(profile["size"]["length"])
    width = float(profile["size"]["width"])
    install_height = float(profile["mount"]["install_height"])
    color = Color.from_hex(str(agv_cfg.get("color", "#6c757d")))

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
    raw_cfg, chain_cfgs = _load_demo_config()
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
    return ArmSimulationModel(bindings=bindings, static_snapshots=_build_agv_outline(raw_cfg))


def main() -> int:
    pyside6_plugin_path = os.path.join(sys.prefix, "Lib", "site-packages", "PySide6", "plugins", "platforms")
    if os.path.isdir(pyside6_plugin_path):
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", pyside6_plugin_path)

    cfg = tomllib.loads(DEMO_CONFIG_PATH.read_text(encoding="utf-8"))
    window_title = str(cfg.get("window_title", "Kinematics Simulator"))
    window_width = int(cfg.get("window_width", 1460))
    window_height = int(cfg.get("window_height", 920))

    app = QApplication(sys.argv)
    widget = KinematicsSimulationWidget(model=build_robot_model())
    auto_reloader = ResourceAutoReloader(widget, parent=widget)
    widget._auto_reloader = auto_reloader  # type: ignore[attr-defined]
    widget.setWindowTitle(window_title)
    widget.resize(window_width, window_height)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
