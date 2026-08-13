"""RecordReplay 命名动作顺序、动作参数和 CSV 资产校验。

本模块只负责读取不可变的 JSON 顺序文件并把动作项绑定到唯一 CSV。
它不连接设备、不创建线程，也不执行任何运动；因此可以在 ``start`` 前作为
纯离线前置检查使用。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

from .contracts import ReplayRow
from .csv_repository import discover_csv_paths, load_replay_rows
from .settings import OffsetConfig, ReplayOffsetSettings

ACTION_SEQUENCE_FILE_NAME: Final = "action_sequence.json"
ACTION_SEQUENCE_SCHEMA_VERSION: Final = 4
MIN_MOTION_SPEED_MM_S: Final = 5.0
MAX_MOTION_SPEED_MM_S: Final = 4000.0
MAX_ZONE_MM: Final = 200.0
ACTION_TYPES: Final = frozenset({"capture", "fast", "precise"})
INDEXED_ACTIONS: Final = frozenset(
    {"get_tray", "put_tray", "get_new_tray", "put_new_tray"}
)
KNOWN_ACTION_TYPES: Final = {
    "go_out": "fast",
    "open_door": "fast",
    "before_calibration": "fast",
    "calibration": "capture",
    "get_tray": "precise",
    "after_get_tray": "fast",
    "put_tray": "precise",
    "before_get_new_tray": "fast",
    "get_new_tray": "precise",
    "before_put_new_tray": "fast",
    "put_new_tray": "precise",
    "calibration_new_tray": "capture",
    "after_put_new_tray": "fast",
    "close_door": "fast",
    "go_home": "fast",
}
SYNC_ACTION_ORDER: Final = ("open_door", "close_door")
"双臂只允许按该名称顺序建立起点同步；不建立终点同步。"
_TIMESTAMP_PATTERN: Final = re.compile(r"\d{8}_\d{6}\Z")
_RECORDING_PREFIX_PATTERN: Final = re.compile(r"[0-9]+\Z")


@dataclass(frozen=True, slots=True)
class CsvAsset:
    """从新命名规则解析出的一个 CSV 资产。"""

    path: Path
    "CSV 绝对或服务内路径。"
    action_name: str
    "record_left/record_right 中的动作名。"
    index: int | None
    "多目标动作的位置序号；普通动作为空。"
    arm_side: str
    "机械臂侧别。"
    timestamp: str
    "录制时间；历史录制可以为空，新命名 CSV 使用 YYYYMMDD_HHMMSS。"


@dataclass(frozen=True, slots=True)
class ReplayDeploymentConfig:
    """统一动作 JSON 中的 offset 行为和先验文件入口。"""

    offset_settings: ReplayOffsetSettings
    """本轮 offset 触发、应用和检测参数。"""
    offset_config: OffsetConfig
    """本轮三球、ChArUco 和手眼先验文件位置。"""


@dataclass(frozen=True, slots=True)
class ActionItem:
    """JSON 中的一项命名动作及其独立运动参数。"""

    function_name: str
    "封闭白名单中的显式动作名。"
    action_type: str
    "capture、fast 或 precise。"
    speed: float
    "普通 arm 点的 MoveAbsJ 末端线速度，单位 mm/s。"
    zone: float
    "普通 arm 点的 zone，单位 mm。"
    index: int | None = None
    "多目标动作选用的 CSV index。"
    final_speed: float | None = None
    "拍摄动作最后一个 arm 点的慢速，单位 mm/s。"
    settle_delay: float = 0.0
    "到位后调用拍摄/算法前的稳定等待时间，单位 s。"


@dataclass(frozen=True, slots=True)
class CaptureAction:
    """拍摄动作：末点慢速到位后调用显式算法。"""

    item: ActionItem


@dataclass(frozen=True, slots=True)
class FastAction:
    """快速动作：用于开门、转移、收尾等非精确动作。"""

    item: ActionItem


@dataclass(frozen=True, slots=True)
class PreciseAction:
    """精确动作：执行层强制使用 zone=0。"""

    item: ActionItem


MotionAction = CaptureAction | FastAction | PreciseAction
"""三类底层动作的封闭联合类型。"""


@dataclass(frozen=True, slots=True)
class NamedActionPlan:
    """已绑定唯一 CSV 的可执行动作计划项。"""

    arm_side: str
    "机械臂侧别。"
    action: MotionAction
    "JSON 中冻结并分类后的底层动作。"
    csv_asset: CsvAsset
    "start 前唯一解析到的 CSV 资产。"

    @property
    def item(self) -> ActionItem:
        """返回动作的公共 JSON 参数。"""

        return self.action.item


@dataclass(frozen=True, slots=True)
class ActionSequencePlan:
    """单次 start 使用的不可变命名动作计划。"""

    schema_version: int
    "JSON schema 版本。"
    left_actions: tuple[NamedActionPlan, ...]
    "左臂按 JSON 顺序执行的动作。"
    right_actions: tuple[NamedActionPlan, ...]
    "右臂按 JSON 顺序执行的动作。"
    source_path: Path
    "顺序文件路径。"
    source_sha256: str
    "冻结的 JSON 原始字节 SHA-256。"
    deployment: ReplayDeploymentConfig
    "start 前从同一 JSON 读取的 offset 与先验配置。"
    preloaded_rows_by_path: tuple[tuple[Path, tuple[ReplayRow, ...]], ...] = ()
    "start 前冻结的被引用 CSV 行；执行期不重新读取磁盘。"

    def all_actions(self) -> tuple[NamedActionPlan, ...]:
        """按侧别返回全部动作，供预加载和静态摘要使用。"""

        return self.left_actions + self.right_actions


class ActionSequenceValidationError(ValueError):
    """动作顺序或 CSV 资产校验失败，并保留完整错误列表。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("动作顺序校验失败：" + "；".join(self.errors))


def parse_csv_filename(path_or_name: Path | str) -> CsvAsset:
    """解析命名 CSV，并兼容录制时附加的纯数字首段前缀。

    数字前缀只用于动作名匹配；返回的 ``CsvAsset.path`` 始终保留实际文件路径，
    因此执行、状态展示和 CSV 行记录仍按磁盘上的原始文件名处理。
    """

    path = Path(path_or_name)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"不是 CSV 文件：{path.name}")
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"CSV 文件名不符合命名规则：{path.name}")
    timestamp = ""
    if len(parts) >= 3 and _TIMESTAMP_PATTERN.fullmatch("_".join(parts[-2:])) is not None:
        arm_side = parts[-3]
        timestamp = "_".join(parts[-2:])
        action_parts = parts[:-3]
    else:
        arm_side = parts[-1]
        action_parts = parts[:-1]
    if arm_side not in {"left", "right"}:
        raise ValueError(f"CSV 文件名缺少有效机械臂侧别：{path.name}")
    if action_parts and _RECORDING_PREFIX_PATTERN.fullmatch(action_parts[0]) is not None:
        action_parts = action_parts[1:]
    index: int | None = None
    if len(action_parts) >= 2 and action_parts[-1].isdigit():
        index = int(action_parts.pop())
        if index <= 0:
            raise ValueError(f"CSV index 必须大于 0：{path.name}")
    action_name = "_".join(action_parts)
    if not action_name or action_name in INDEXED_ACTIONS and index is None:
        raise ValueError(f"CSV 多目标动作必须带 index：{path.name}")
    if action_name not in KNOWN_ACTION_TYPES:
        raise ValueError(f"CSV 动作名不在白名单中：{path.name}")
    if action_name not in INDEXED_ACTIONS and index is not None:
        raise ValueError(f"非多目标动作不能带 index：{path.name}")
    return CsvAsset(path, action_name, index, arm_side, timestamp)


def load_action_sequence(
    config_path: Path,
    left_record_dir: Path,
    right_record_dir: Path,
    *,
    old_tray_current_index: int,
    old_tray_put_index: int,
    new_tray_current_index: int,
    new_tray_put_index: int,
) -> ActionSequencePlan:
    """读取 JSON、校验 schema 和 CSV 唯一映射，返回单次冻结计划。

    四个托盘 index 只允许由本次 start 或 plan 请求传入。动作 JSON 不保存 index，
    避免配置默认值与请求参数形成两个数据源。
    """

    errors: list[str] = []
    payload, raw_bytes = _read_action_sequence_payload(config_path)
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    allowed_root = {"schema_version", "left", "right", "deployment"}
    errors.extend(_unknown_fields(payload, allowed_root, "根节点"))
    schema_version = _read_int(payload.get("schema_version"), "schema_version", errors)
    if schema_version != ACTION_SEQUENCE_SCHEMA_VERSION:
        errors.append(
            f"schema_version 必须为 {ACTION_SEQUENCE_SCHEMA_VERSION}，实际为 {schema_version!r}"
        )

    deployment = _parse_deployment_config(payload.get("deployment"), config_path, errors)

    left_items = _parse_items(payload.get("left"), "left", errors)
    right_items = _parse_items(payload.get("right"), "right", errors)
    if not left_items:
        errors.append("left 动作列表不能为空")
    if not right_items:
        errors.append("right 动作列表不能为空")
    _validate_sync_pairing(left_items, right_items, errors)
    left_items = _apply_runtime_tray_index(
        left_items,
        old_tray_current_index,
        old_tray_put_index,
        new_tray_current_index,
        new_tray_put_index,
        errors,
    )
    right_items = _apply_runtime_tray_index(
        right_items,
        old_tray_current_index,
        old_tray_put_index,
        new_tray_current_index,
        new_tray_put_index,
        errors,
    )

    required_keys_by_side = {
        "left": {(item.function_name, item.index) for item in left_items},
        "right": {(item.function_name, item.index) for item in right_items},
    }
    assets_by_side = {
        "left": _discover_assets(
            left_record_dir,
            "left",
            required_keys_by_side["left"],
            errors,
        ),
        "right": _discover_assets(
            right_record_dir,
            "right",
            required_keys_by_side["right"],
            errors,
        ),
    }
    left_actions = _bind_items(left_items, assets_by_side["left"], "left", errors)
    right_actions = _bind_items(right_items, assets_by_side["right"], "right", errors)
    _validate_capture_dependencies(left_actions, right_actions, errors)
    offset_settings = deployment.offset_settings
    _validate_offset_precompile_order(left_actions, right_actions, offset_settings, errors)
    trigger = offset_settings.calculate_after_action_name
    if trigger is not None and not any(
        action.item.function_name == trigger for action in left_actions
    ):
        errors.append(
            "deployment.offset.calculate_after_action_name 必须对应 left 动作列表中的已实现动作："
            f"{trigger}"
        )
    preloaded_rows_by_path = _freeze_action_rows(left_actions + right_actions, errors)
    if errors:
        raise ActionSequenceValidationError(errors)
    if schema_version is None:
        raise ActionSequenceValidationError(["顺序文件的 schema_version 缺失"])
    return ActionSequencePlan(
        schema_version,
        tuple(left_actions),
        tuple(right_actions),
        config_path,
        source_sha256,
        deployment,
        tuple(preloaded_rows_by_path),
    )


def _validate_offset_precompile_order(
    left_actions: list[NamedActionPlan],
    right_actions: list[NamedActionPlan],
    settings: ReplayOffsetSettings,
    errors: list[str],
) -> None:
    """在 start 冻结阶段验证 offset 目标归属和预编译先后关系。"""

    overlap = sorted(
        settings.target_action_names
        & (settings.left_charuco_target_action_names | settings.right_charuco_target_action_names)
    )
    if overlap:
        errors.append("命名动作不能同时配置 ChArUco 与三球 offset：" + ", ".join(overlap))
    right_targets = sorted(
        {
            action.item.function_name
            for action in right_actions
            if action.item.function_name in settings.target_action_names
        }
    )
    if right_targets:
        errors.append("三球 offset 目标动作只能配置在左臂：" + ", ".join(right_targets))
    left_names = [action.item.function_name for action in left_actions]
    target_positions = [
        index
        for index, action_name in enumerate(left_names)
        if action_name in settings.target_action_names
    ]
    if not target_positions:
        return
    trigger = settings.calculate_after_action_name
    if trigger is None or trigger not in left_names:
        return
    trigger_position = left_names.index(trigger)
    first_target_position = min(target_positions)
    if trigger_position >= first_target_position:
        errors.append(
            "三球 offset 必须在首个目标动作前完成："
            f"trigger={trigger} first_target={left_names[first_target_position]}"
        )


def load_replay_deployment_config(config_path: Path) -> ReplayDeploymentConfig:
    """读取统一 JSON 的部署配置，不构建需要运行时 index 的动作计划。"""

    errors: list[str] = []
    payload, _ = _read_action_sequence_payload(config_path)
    allowed_root = {"schema_version", "left", "right", "deployment"}
    errors.extend(_unknown_fields(payload, allowed_root, "根节点"))
    schema_version = _read_int(payload.get("schema_version"), "schema_version", errors)
    if schema_version != ACTION_SEQUENCE_SCHEMA_VERSION:
        errors.append(
            f"schema_version 必须为 {ACTION_SEQUENCE_SCHEMA_VERSION}，实际为 {schema_version!r}"
        )
    deployment = _parse_deployment_config(payload.get("deployment"), config_path, errors)
    left_items = _parse_items(payload.get("left"), "left", errors)
    right_items = _parse_items(payload.get("right"), "right", errors)
    if not left_items:
        errors.append("left 动作列表不能为空")
    if not right_items:
        errors.append("right 动作列表不能为空")
    _validate_sync_pairing(left_items, right_items, errors)
    if errors:
        raise ActionSequenceValidationError(errors)
    return deployment


def _read_action_sequence_payload(config_path: Path) -> tuple[dict[object, object], bytes]:
    """读取 UTF-8 JSON object，并保留原始字节供完整计划计算摘要。"""

    try:
        raw_bytes = config_path.read_bytes()
    except OSError as error:
        raise ActionSequenceValidationError([f"顺序文件不可读：{config_path}: {error}"]) from error
    try:
        payload: object = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActionSequenceValidationError([f"顺序文件不是有效 UTF-8 JSON：{error}"]) from error
    if not isinstance(payload, dict):
        raise ActionSequenceValidationError(["顺序文件根节点必须是 JSON object"])
    return payload, raw_bytes


def _apply_runtime_tray_index(
    items: list[ActionItem],
    old_tray_current_index: int,
    old_tray_put_index: int,
    new_tray_current_index: int,
    new_tray_put_index: int,
    errors: list[str],
) -> list[ActionItem]:
    """把 start 的四个托盘位置 index 应用到四个托盘动作。"""

    names_and_values = (
        ("old_tray_current_index", old_tray_current_index),
        ("old_tray_put_index", old_tray_put_index),
        ("new_tray_current_index", new_tray_current_index),
        ("new_tray_put_index", new_tray_put_index),
    )
    for name, value in names_and_values:
        if not _is_positive_int(value):
            errors.append(f"{name} 必须是大于 0 的整数")
    if not all(_is_positive_int(value) for _, value in names_and_values):
        return items
    index_by_action = {
        "get_tray": old_tray_current_index,
        "put_tray": old_tray_put_index,
        "get_new_tray": new_tray_current_index,
        "put_new_tray": new_tray_put_index,
    }
    return [
        replace(
            item,
            index=index_by_action.get(item.function_name, item.index),
        )
        for item in items
    ]


def _is_positive_int(value: int | None) -> bool:
    """判断 start 传入的 index 是否为正整数。"""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_deployment_config(
    value: object,
    config_path: Path,
    errors: list[str],
) -> ReplayDeploymentConfig:
    """从统一 JSON 读取 offset 策略和固定先验文件入口。"""

    service_root = config_path.parent.resolve()
    default_config = ReplayDeploymentConfig(
        ReplayOffsetSettings(),
        OffsetConfig(
            service_root / "prior_data" / "ball_pose_prior.json",
            service_root / "prior_data" / "hand_eye_result.txt",
        ),
    )
    if not isinstance(value, dict):
        errors.append("deployment 必须是 JSON object")
        return default_config
    errors.extend(_unknown_fields(value, {"offset", "prior_files"}, "deployment"))
    prior_value = value.get("prior_files")
    offset_value = value.get("offset")
    if not isinstance(prior_value, dict):
        errors.append("deployment.prior_files 必须是 JSON object")
        return default_config
    if not isinstance(offset_value, dict):
        errors.append("deployment.offset 必须是 JSON object")
        return default_config

    prior_keys = {
        "ball_pose",
        "hand_eye_result",
        "charuco_board",
        "charuco_history",
        "left_head_base_camera",
        "right_head_base_camera",
    }
    errors.extend(_unknown_fields(prior_value, prior_keys, "deployment.prior_files"))
    prior_paths = {
        key: _resolve_deployment_path(
            prior_value.get(key),
            key,
            service_root,
            errors,
        )
        for key in prior_keys
    }
    camera_name = offset_value.get("camera_name")
    if not isinstance(camera_name, str) or not camera_name.strip():
        errors.append("deployment.offset.camera_name 必须是非空字符串")
        camera_name = "left_hand_camera"
    offset_settings = _parse_offset_settings(offset_value, errors)
    return ReplayDeploymentConfig(
        offset_settings,
        OffsetConfig(
            prior_paths["ball_pose"],
            prior_paths["hand_eye_result"],
            camera_name.strip(),
            prior_paths["charuco_board"],
            prior_paths["charuco_history"],
            prior_paths["left_head_base_camera"],
            prior_paths["right_head_base_camera"],
        ),
    )


def _resolve_deployment_path(
    value: object,
    label: str,
    service_root: Path,
    errors: list[str],
) -> Path:
    """解析统一 JSON 的相对先验路径，并禁止越出服务目录。"""

    if not isinstance(value, str) or not value.strip():
        errors.append(f"deployment.prior_files.{label} 必须是非空相对路径")
        return service_root / "prior_data" / label
    candidate = Path(value)
    if candidate.is_absolute():
        errors.append(f"deployment.prior_files.{label} 不允许绝对路径")
        return service_root / candidate.name
    resolved = (service_root / candidate).resolve()
    try:
        resolved.relative_to(service_root)
    except ValueError:
        errors.append(f"deployment.prior_files.{label} 不得越出服务目录")
    return resolved


def _parse_offset_settings(
    value: dict[object, object],
    errors: list[str],
) -> ReplayOffsetSettings:
    """显式解析 JSON offset 字段，数值字段使用当前安全默认值。"""

    defaults = ReplayOffsetSettings()
    allowed = {
        "camera_name",
        "calculate_after_action_name",
        "target_action_names",
        "sample_count",
        "detection_timeout_ms",
        "detection_attempts_per_sample",
        "capture_settle_delay_s",
        "mad_scale",
        "min_outlier_threshold_mm",
        "narrow_consistency_tolerance_mm",
        "left_charuco_target_action_names",
        "right_charuco_target_action_names",
        "charuco_head_yaw_deg",
        "charuco_head_pitch_deg",
        "charuco_head_settle_s",
        "charuco_camera_timeout_s",
        "charuco_max_frame_count",
        "charuco_min_corners",
        "charuco_rpc_timeout_s",
        "charuco_timeout_retry_count",
        "charuco_timeout_retry_delay_s",
        "charuco_safety_attempt_count",
        "charuco_safety_retry_delay_s",
        "charuco_history_min_accepted_samples",
        "charuco_sigma_limit",
        "charuco_max_translation_norm_mm",
        "charuco_max_rotation_norm_deg",
    }
    errors.extend(_unknown_fields(value, allowed, "deployment.offset"))
    target_action_names = _read_string_set(
        value.get("target_action_names"),
        "deployment.offset.target_action_names",
        defaults.target_action_names,
        errors,
    )
    left_charuco = _read_string_set(
        value.get("left_charuco_target_action_names"),
        "deployment.offset.left_charuco_target_action_names",
        defaults.left_charuco_target_action_names,
        errors,
    )
    right_charuco = _read_string_set(
        value.get("right_charuco_target_action_names"),
        "deployment.offset.right_charuco_target_action_names",
        defaults.right_charuco_target_action_names,
        errors,
    )
    trigger = value.get("calculate_after_action_name", defaults.calculate_after_action_name)
    if trigger is not None and not isinstance(trigger, str):
        errors.append("deployment.offset.calculate_after_action_name 必须是字符串或 null")
        trigger = defaults.calculate_after_action_name
    numeric_defaults: dict[str, int | float] = {
        "sample_count": defaults.sample_count,
        "detection_timeout_ms": defaults.detection_timeout_ms,
        "detection_attempts_per_sample": defaults.detection_attempts_per_sample,
        "capture_settle_delay_s": defaults.capture_settle_delay_s,
        "mad_scale": defaults.mad_scale,
        "min_outlier_threshold_mm": defaults.min_outlier_threshold_mm,
        "narrow_consistency_tolerance_mm": defaults.narrow_consistency_tolerance_mm,
        "charuco_head_yaw_deg": defaults.charuco_head_yaw_deg,
        "charuco_head_pitch_deg": defaults.charuco_head_pitch_deg,
        "charuco_head_settle_s": defaults.charuco_head_settle_s,
        "charuco_camera_timeout_s": defaults.charuco_camera_timeout_s,
        "charuco_max_frame_count": defaults.charuco_max_frame_count,
        "charuco_min_corners": defaults.charuco_min_corners,
        "charuco_rpc_timeout_s": defaults.charuco_rpc_timeout_s,
        "charuco_timeout_retry_count": defaults.charuco_timeout_retry_count,
        "charuco_timeout_retry_delay_s": defaults.charuco_timeout_retry_delay_s,
        "charuco_safety_attempt_count": defaults.charuco_safety_attempt_count,
        "charuco_safety_retry_delay_s": defaults.charuco_safety_retry_delay_s,
        "charuco_history_min_accepted_samples": defaults.charuco_history_min_accepted_samples,
        "charuco_sigma_limit": defaults.charuco_sigma_limit,
        "charuco_max_translation_norm_mm": defaults.charuco_max_translation_norm_mm,
        "charuco_max_rotation_norm_deg": defaults.charuco_max_rotation_norm_deg,
    }
    numeric_values: dict[str, int | float] = {}
    for key, default in numeric_defaults.items():
        raw = value.get(key, default)
        if isinstance(default, int):
            if not isinstance(raw, int) or isinstance(raw, bool):
                errors.append(f"deployment.offset.{key} 必须是整数")
                raw = default
        elif not isinstance(raw, int | float) or isinstance(raw, bool):
            errors.append(f"deployment.offset.{key} 必须是数字")
            raw = default
        numeric_values[key] = raw
    return ReplayOffsetSettings(
        calculate_after_action_name=trigger,
        target_action_names=target_action_names,
        sample_count=int(numeric_values["sample_count"]),
        detection_timeout_ms=int(numeric_values["detection_timeout_ms"]),
        detection_attempts_per_sample=int(numeric_values["detection_attempts_per_sample"]),
        capture_settle_delay_s=float(numeric_values["capture_settle_delay_s"]),
        mad_scale=float(numeric_values["mad_scale"]),
        min_outlier_threshold_mm=float(numeric_values["min_outlier_threshold_mm"]),
        narrow_consistency_tolerance_mm=float(numeric_values["narrow_consistency_tolerance_mm"]),
        left_charuco_target_action_names=left_charuco,
        right_charuco_target_action_names=right_charuco,
        charuco_head_yaw_deg=float(numeric_values["charuco_head_yaw_deg"]),
        charuco_head_pitch_deg=float(numeric_values["charuco_head_pitch_deg"]),
        charuco_head_settle_s=float(numeric_values["charuco_head_settle_s"]),
        charuco_camera_timeout_s=float(numeric_values["charuco_camera_timeout_s"]),
        charuco_max_frame_count=int(numeric_values["charuco_max_frame_count"]),
        charuco_min_corners=int(numeric_values["charuco_min_corners"]),
        charuco_rpc_timeout_s=float(numeric_values["charuco_rpc_timeout_s"]),
        charuco_timeout_retry_count=int(numeric_values["charuco_timeout_retry_count"]),
        charuco_timeout_retry_delay_s=float(numeric_values["charuco_timeout_retry_delay_s"]),
        charuco_safety_attempt_count=int(numeric_values["charuco_safety_attempt_count"]),
        charuco_safety_retry_delay_s=float(numeric_values["charuco_safety_retry_delay_s"]),
        charuco_history_min_accepted_samples=int(numeric_values["charuco_history_min_accepted_samples"]),
        charuco_sigma_limit=float(numeric_values["charuco_sigma_limit"]),
        charuco_max_translation_norm_mm=float(numeric_values["charuco_max_translation_norm_mm"]),
        charuco_max_rotation_norm_deg=float(numeric_values["charuco_max_rotation_norm_deg"]),
    )


def _read_string_set(
    value: object,
    label: str,
    default: frozenset[str],
    errors: list[str],
) -> frozenset[str]:
    """读取动作名称数组并转换为不可变集合。"""

    if value is None:
        return default
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{label} 必须是字符串数组")
        return default
    return frozenset(value)


def _parse_items(
    value: object,
    arm_side: str,
    errors: list[str],
) -> list[ActionItem]:
    """解析一侧动作列表，并收集该列表全部字段错误。"""

    if not isinstance(value, list):
        errors.append(f"{arm_side} 必须是 JSON array")
        return []
    result: list[ActionItem] = []
    for position, raw_item in enumerate(value, start=1):
        label = f"{arm_side}[{position}]"
        if not isinstance(raw_item, dict):
            errors.append(f"{label} 必须是 JSON object")
            continue
        allowed = {"function_name", "type", "speed", "zone", "final_speed", "settle_delay"}
        errors.extend(_unknown_fields(raw_item, allowed, label))
        function_name = raw_item.get("function_name")
        action_type = raw_item.get("type")
        if not isinstance(function_name, str) or function_name not in KNOWN_ACTION_TYPES:
            errors.append(f"{label}.function_name 不在动作白名单中：{function_name!r}")
            continue
        if not isinstance(action_type, str) or action_type not in ACTION_TYPES:
            errors.append(f"{label}.type 必须是 capture/fast/precise：{action_type!r}")
            continue
        expected_type = KNOWN_ACTION_TYPES[function_name]
        if action_type != expected_type:
            errors.append(f"{label} 动作类型错误：{function_name} 应为 {expected_type}")
        speed = _read_float(raw_item.get("speed"), f"{label}.speed", errors)
        zone = _read_float(raw_item.get("zone"), f"{label}.zone", errors)
        has_final_speed = "final_speed" in raw_item
        final_speed = _read_optional_float(raw_item.get("final_speed"), f"{label}.final_speed", errors)
        has_settle_delay = "settle_delay" in raw_item
        settle_delay = (
            _read_optional_float(raw_item.get("settle_delay"), f"{label}.settle_delay", errors)
            if has_settle_delay
            else None
        )
        if speed is not None and not MIN_MOTION_SPEED_MM_S <= speed <= MAX_MOTION_SPEED_MM_S:
            errors.append(f"{label}.speed 必须在 {MIN_MOTION_SPEED_MM_S} 到 {MAX_MOTION_SPEED_MM_S} 之间")
        if zone is not None and not 0.0 <= zone <= MAX_ZONE_MM:
            errors.append(f"{label}.zone 必须在 0 到 {MAX_ZONE_MM} 之间")
        if action_type == "precise" and zone != 0.0:
            errors.append(f"{label} precise 动作的 zone 必须为 0")
        if action_type == "fast" and zone == 0.0:
            errors.append(f"{label} fast 动作的 zone 必须大于 0")
        if action_type == "capture":
            if final_speed is None:
                errors.append(f"{label} capture 动作必须提供 final_speed")
            elif not MIN_MOTION_SPEED_MM_S <= final_speed <= MAX_MOTION_SPEED_MM_S:
                errors.append(f"{label}.final_speed 超出有效速度范围")
            elif speed is not None and final_speed > speed:
                errors.append(f"{label}.final_speed 必须不大于 speed")
            if settle_delay is None:
                errors.append(f"{label} capture 动作必须提供 settle_delay")
        elif has_final_speed:
            errors.append(f"{label} 非 capture 动作不能提供 final_speed")
        if action_type != "capture" and has_settle_delay:
            errors.append(f"{label} 非 capture 动作不能提供 settle_delay")
        if settle_delay is not None and not 0.0 <= settle_delay <= 60.0:
            errors.append(f"{label}.settle_delay 必须在 0 到 60 秒之间")
        if speed is None or zone is None:
            continue
        result.append(
            ActionItem(
                function_name,
                action_type,
                speed,
                zone,
                None,
                final_speed,
                0.0 if settle_delay is None else settle_delay,
            )
        )
    return result


def _bind_items(
    items: list[ActionItem],
    assets: list[CsvAsset],
    arm_side: str,
    errors: list[str],
) -> list[NamedActionPlan]:
    """将动作项绑定到唯一的同侧 CSV。"""

    result: list[NamedActionPlan] = []
    for position, item in enumerate(items, start=1):
        candidates = [
            asset
            for asset in assets
            if asset.action_name == item.function_name and asset.index == item.index
        ]
        label = f"{arm_side}[{position}]/{item.function_name}"
        if not candidates:
            errors.append(f"{label} 没有匹配 CSV index={item.index!r}")
            continue
        if len(candidates) != 1:
            names = ", ".join(asset.path.name for asset in candidates)
            errors.append(f"{label} 匹配到重复 CSV：{names}")
            continue
        asset = candidates[0]
        try:
            rows = load_replay_rows(asset.path)
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{label} CSV 无法解析：{asset.path.name}: {error}")
            continue
        if not rows and item.function_name != "calibration_new_tray":
            errors.append(f"{label} CSV 不能为空：{asset.path.name}")
            continue
        if rows and item.function_name == "calibration_new_tray":
            errors.append(f"{label} calibration_new_tray CSV 必须保持为空：{asset.path.name}")
            continue
        if rows and item.action_type == "capture" and not any(row.action_type == "arm" for row in rows):
            errors.append(f"{label} capture CSV 至少需要一条 arm 记录：{asset.path.name}")
            continue
        result.append(NamedActionPlan(arm_side, _make_motion_action(item), asset))
    return result


def _make_motion_action(item: ActionItem) -> MotionAction:
    """按显式 type 构造三类动作，不使用反射或动态注册表。"""

    if item.action_type == "capture":
        return CaptureAction(item)
    if item.action_type == "fast":
        return FastAction(item)
    if item.action_type == "precise":
        return PreciseAction(item)
    raise ValueError(f"未知动作类型：{item.action_type}")


def _discover_assets(
    record_dir: Path,
    arm_side: str,
    required_keys: set[tuple[str, int | None]],
    errors: list[str],
) -> list[CsvAsset]:
    """只解析当前 JSON 引用动作的候选 CSV，保留同目标重复检查。"""

    try:
        paths = discover_csv_paths(record_dir)
    except OSError as error:
        errors.append(f"{arm_side} CSV 目录不可读：{record_dir}: {error}")
        return []
    assets: list[CsvAsset] = []
    for path in paths:
        if not _is_required_asset_name(path, arm_side, required_keys):
            continue
        try:
            asset = parse_csv_filename(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        if asset.arm_side != arm_side:
            errors.append(f"CSV 侧别与目录不一致：{path.name} 目录={arm_side}")
            continue
        assets.append(asset)
    return assets


def _is_required_asset_name(
    path: Path,
    arm_side: str,
    required_keys: set[tuple[str, int | None]],
) -> bool:
    """按动作名、index 和侧别筛选候选文件，不解析无关资产。"""

    stem = path.stem
    stem_parts = stem.split("_", maxsplit=1)
    if len(stem_parts) == 2 and _RECORDING_PREFIX_PATTERN.fullmatch(stem_parts[0]) is not None:
        stem = stem_parts[1]
    for function_name, index in required_keys:
        index_suffix = "" if index is None else f"_{index}"
        prefix = f"{function_name}{index_suffix}_{arm_side}"
        if stem == prefix or stem.startswith(f"{prefix}_"):
            return True
    return False


def _validate_sync_pairing(
    left_items: list[ActionItem],
    right_items: list[ActionItem],
    errors: list[str],
) -> None:
    """离线断言双臂同步动作名称和相对顺序一致。"""

    left_sync_order = [
        item.function_name
        for item in left_items
        if item.function_name in SYNC_ACTION_ORDER
    ]
    right_sync_order = [
        item.function_name
        for item in right_items
        if item.function_name in SYNC_ACTION_ORDER
    ]
    if left_sync_order != right_sync_order:
        errors.append(
            "双臂同步动作顺序不一致："
            f"left={left_sync_order!r}, right={right_sync_order!r}"
        )
    for function_name in SYNC_ACTION_ORDER:
        left_count = sum(item.function_name == function_name for item in left_items)
        right_count = sum(item.function_name == function_name for item in right_items)
        if left_count != right_count:
            errors.append(
                f"{function_name} 左右出现次数不一致：left={left_count}, right={right_count}"
            )


def _validate_capture_dependencies(
    left_actions: list[NamedActionPlan],
    right_actions: list[NamedActionPlan],
    errors: list[str],
) -> None:
    """验证当前动作列表中拍摄动作确实有可执行 CSV。"""

    for action in left_actions + right_actions:
        if action.item.action_type == "capture" and not action.csv_asset.path.is_file():
            errors.append(f"capture 依赖 CSV 不存在：{action.csv_asset.path}")


def _freeze_action_rows(
    actions: list[NamedActionPlan],
    errors: list[str],
) -> list[tuple[Path, tuple[ReplayRow, ...]]]:
    """在 start 前冻结所有被引用 CSV 的行数据。"""

    result: dict[Path, tuple[ReplayRow, ...]] = {}
    for action in actions:
        path = action.csv_asset.path
        if path in result:
            continue
        try:
            rows = tuple(load_replay_rows(path))
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"CSV 冻结读取失败：{path.name}: {error}")
            continue
        if not rows and action.item.function_name != "calibration_new_tray":
            errors.append(f"CSV 在计划冻结时为空：{path.name}")
            continue
        if rows and action.item.function_name == "calibration_new_tray":
            errors.append(f"calibration_new_tray CSV 在计划冻结时必须为空：{path.name}")
            continue
        if rows and action.item.action_type == "capture" and not any(row.action_type == "arm" for row in rows):
            errors.append(f"capture CSV 在计划冻结时缺少 arm 记录：{path.name}")
            continue
        result[path] = rows
    return list(result.items())


def _unknown_fields(payload: dict[object, object], allowed: set[str], label: str) -> list[str]:
    """报告 JSON object 中未声明的字段。"""

    return [f"{label} 存在未知字段：{key!r}" for key in payload if key not in allowed]


def _read_int(value: object, label: str, errors: list[str]) -> int | None:
    """读取非 bool 整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{label} 必须是整数：{value!r}")
        return None
    return value


def _read_float(value: object, label: str, errors: list[str]) -> float | None:
    """读取非 bool 的有限浮点数。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        errors.append(f"{label} 必须是数字：{value!r}")
        return None
    number = float(value)
    if not number == number or number in {float("inf"), float("-inf")}:
        errors.append(f"{label} 必须是有限数字")
        return None
    return number


def _read_optional_float(value: object, label: str, errors: list[str]) -> float | None:
    """读取可选浮点字段。"""

    if value is None:
        return None
    return _read_float(value, label, errors)
