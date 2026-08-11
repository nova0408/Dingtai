"""RecordReplay 先验文件校验、替换和服务端备份。"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..charuco_offset import _load_history, _load_matrix, _load_prior_board
from ..offset_detector_gateway import load_three_ball_priors
from ..offset_math import load_tool_camera_transform
from ..settings import OffsetConfig

PriorKind = Literal["ball_pose", "charuco"]
"可通过 HTTP 替换的两类 JSON 先验。"

CHARUCO_HISTORY_MIN_ACCEPTED_SAMPLES = 6
"全局 ChArUco 历史要求的最少有效样本数。"


@dataclass(frozen=True, slots=True)
class PriorValidationResult:
    """一次完整先验检查的结果。"""

    errors: tuple[str, ...]
    "按先验类别归属的缺失或格式错误摘要。"

    @property
    def valid(self) -> bool:
        """返回所有先验是否均已通过检查。"""

        return not self.errors

    def error_text(self) -> str:
        """构造可直接返回给 HTTP 客户端的错误文本。"""

        if self.valid:
            return ""
        return "先验检查失败：" + "；".join(self.errors)


@dataclass(frozen=True, slots=True)
class PriorReplacement:
    """一次 JSON 先验替换的结果。"""

    file_name: str
    "被替换的先验文件名；目标路径来自统一动作 JSON。"

    backup_file: str | None
    "旧文件在服务端 `.archive` 下的相对路径；首次创建时为 None。"


class RecordReplayPriorStore:
    """管理固定先验目录中的 JSON 文件和服务端备份。"""

    def __init__(self, prior_data_dir: Path) -> None:
        """创建先验存储边界，不连接设备或启动回放线程。"""

        self._prior_data_dir = prior_data_dir
        self._service_root = prior_data_dir.parent
        self._archive_dir = self._service_root / ".archive" / "prior_data"

    def validate_all(self, offset_config: OffsetConfig | None = None) -> PriorValidationResult:
        """完整检查本轮统一 JSON 指定的全部先验文件。"""

        config = offset_config if offset_config is not None else OffsetConfig(
            self._path("ball_pose_prior.json"),
            self._path("hand_eye_result.txt"),
            charuco_prior_path=self._path("charuco_board_prior.json"),
            charuco_history_path=self._path("charuco_offset_history.csv"),
            left_head_base_camera_path=self._path("left_head_base_camera.npy"),
            right_head_base_camera_path=self._path("right_head_base_camera.npy"),
        )
        if (
            config.charuco_prior_path is None
            or config.charuco_history_path is None
            or config.left_head_base_camera_path is None
            or config.right_head_base_camera_path is None
        ):
            return PriorValidationResult(("统一 JSON 缺少完整 ChArUco 先验路径",))

        checks: tuple[tuple[str, Path, Callable[[Path], None]], ...] = (
            ("ball_pose_prior.json", config.prior_capture_path, self._validate_ball_pose),
            ("hand_eye_result.txt", config.hand_eye_result_path, self._validate_hand_eye),
            ("charuco_board_prior.json", config.charuco_prior_path, self._validate_charuco_board),
            ("charuco_offset_history.csv", config.charuco_history_path, self._validate_charuco_history),
            (
                "left_head_base_camera.npy",
                config.left_head_base_camera_path,
                self._validate_left_head_base_camera,
            ),
            (
                "right_head_base_camera.npy",
                config.right_head_base_camera_path,
                self._validate_right_head_base_camera,
            ),
        )
        errors: list[str] = []
        for file_name, path, validator in checks:
            if not path.is_file():
                errors.append(f"{file_name}: 文件不存在 path={path}")
                continue
            try:
                validator(path)
            except Exception as exc:
                errors.append(f"{file_name}: {exc}")
        return PriorValidationResult(tuple(errors))

    def replace_json(
        self,
        kind: PriorKind,
        payload: object,
        target_path: Path | None = None,
    ) -> PriorReplacement:
        """校验并原子替换统一 JSON 指定的先验，同时备份旧文件。"""

        if not isinstance(payload, dict):
            raise ValueError("先验请求 body 必须是 JSON object")
        file_name = self._file_name(kind)
        target = self._resolve_target(target_path if target_path is not None else self._path(file_name))
        temporary = target.with_name(f".{target.name}.upload.tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._validate_json(kind, temporary)
            backup_file = self._backup_existing(target)
            os.replace(temporary, target)
            return PriorReplacement(file_name, backup_file)
        finally:
            temporary.unlink(missing_ok=True)

    def _resolve_target(self, target_path: Path) -> Path:
        """校验先验写回路径仍位于服务目录内。"""

        target = target_path.resolve()
        try:
            target.relative_to(self._service_root.resolve())
        except ValueError as error:
            raise ValueError(f"先验写回路径不得越出服务目录：{target_path}") from error
        return target

    def _path(self, file_name: str) -> Path:
        """返回未提供统一配置时使用的默认先验文件路径。"""

        return self._prior_data_dir / file_name

    @staticmethod
    def _file_name(kind: PriorKind) -> str:
        """把 API 先验类别映射到固定文件名。"""

        if kind == "ball_pose":
            return "ball_pose_prior.json"
        if kind == "charuco":
            return "charuco_board_prior.json"
        raise ValueError(f"不支持的先验类型：{kind}")

    def _validate_json(self, kind: PriorKind, path: Path) -> None:
        """校验单个可上传 JSON 先验。"""

        if kind == "ball_pose":
            self._validate_ball_pose(path)
            return
        if kind == "charuco":
            self._validate_charuco_board(path)
            return
        raise ValueError(f"不支持的先验类型：{kind}")

    @staticmethod
    def _validate_ball_pose(path: Path) -> None:
        """校验三球 JSON，不要求调试 overlay 图片。"""

        load_three_ball_priors(path)

    @staticmethod
    def _validate_hand_eye(path: Path) -> None:
        """校验手眼标定文本中的工具到相机矩阵。"""

        load_tool_camera_transform(path)

    @staticmethod
    def _validate_charuco_board(path: Path) -> None:
        """校验 ChArUco 相机到板矩阵。"""

        _load_prior_board(path)

    @staticmethod
    def _validate_charuco_history(path: Path) -> None:
        """校验 ChArUco 历史格式和全局有效样本数量。"""

        values = _load_history(path)
        sample_count = values.shape[0]
        if sample_count < CHARUCO_HISTORY_MIN_ACCEPTED_SAMPLES:
            raise ValueError(
                f"全局有效历史样本不足：{sample_count} < "
                f"{CHARUCO_HISTORY_MIN_ACCEPTED_SAMPLES}"
            )

    @staticmethod
    def _validate_left_head_base_camera(path: Path) -> None:
        """校验左臂头部相机外参。"""

        _load_matrix(path, "左臂 T_base_camera")

    @staticmethod
    def _validate_right_head_base_camera(path: Path) -> None:
        """校验右臂头部相机外参。"""

        _load_matrix(path, "右臂 T_base_camera")

    def _backup_existing(self, target: Path) -> str | None:
        """把旧文件复制到时间戳目录并返回相对路径。"""

        if not target.is_file():
            return None
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_path = self._archive_dir / timestamp / target.name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_path)
        return str(backup_path.relative_to(self._service_root)).replace("\\", "/")
