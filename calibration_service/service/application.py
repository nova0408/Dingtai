"""手眼标定与先验记录业务门面。

本模块只读取 RobotControl 的设备状态，只请求 CameraPipeline 拍摄和检测，
不导入 qmlinker、xCoreSDK，也不发送任何设备控制请求。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from camera_pipeline.client import CameraName
from camera_pipeline.service.http_client import CameraPipelineHttpClient
from camera_pipeline.service.protocol import CharucoDetectionRequest, CharucoDetectionResponse
from src.calibration.hand_eye import calibrate_hand_eye_from_pose_sequences
from src.wuji.prior_calibration import (
    PriorCalibrationConfig,
    PriorCalibrationRecorder,
)

from .protocol import CalibrationResponse

SERVICE_ROOT = Path(__file__).resolve().parents[1]
"服务根目录。"

RECORD_REPLAY_PRIOR_DIR = SERVICE_ROOT.parent / "record_replay" / "prior_data"
"RecordReplay 使用的固定先验结果目录。"

ROBOT_CONTROL_STATUS_URL = "http://127.0.0.1:6500/api/v1/status"
"Orin 本机 RobotControl 只读状态接口。"

BALL_CAMERA_NAME = CameraName("left_hand_camera")
"RecordReplay 当前使用的手部相机。"

HEAD_CAMERA_NAME = CameraName("head_camera")
"RecordReplay 当前使用的头部相机。"

CalibrationKind = Literal["left_eye_in_hand", "head_eye_to_hand"]
"两类标定会话名称。"

HandEyeMethod = Literal["closed_form", "multi_method"]
"左手眼求解方法。"

HandEyePairMode = Literal["all", "adjacent"]
"左手眼样本配对方式。"


@dataclass(frozen=True, slots=True)
class ArmSnapshot:
    """从 RobotControl 只读状态转换出的 AR5 快照。

    所有姿态矩阵内部使用米，角度字段使用度；实例不持有设备连接。
    """

    joint_deg: tuple[float, ...]
    "七个关节角，单位 deg。"

    pose_matrix_m: tuple[tuple[float, ...], ...]
    "TCP 齐次矩阵，平移单位 m。"

    xyz_mm: tuple[float, float, float]
    "TCP 平移，单位 mm。"

    rpy_deg: tuple[float, float, float]
    "TCP 欧拉角，单位 deg，SciPy 小写外禀 xyz。"

    elbow_deg: float
    "AR5 臂角，单位 deg。"


@dataclass(frozen=True, slots=True)
class HandEyeSample:
    """一组手眼标定同步样本。"""

    robot_pose_matrix_m: tuple[tuple[float, ...], ...]
    "T_base_tool，平移单位 m。"

    camera_board_matrix_m: tuple[tuple[float, ...], ...]
    "T_camera_board，平移单位 m。"


@dataclass(frozen=True, slots=True)
class PendingReplacement:
    """等待前端二次确认的结果替换。"""

    replacement_id: str
    "替换确认标识。"

    staging_dir: Path
    "缓存结果所在的临时目录。"

    files: tuple[tuple[Path, Path], ...]
    "临时结果文件与 RecordReplay 目标文件的映射。"

    data: dict[str, object]
    "计算结果及前端展示信息。"


class RobotControlReadClient:
    """通过 RobotControl HTTP GET 读取 AR5 状态。

    该客户端没有任何 POST 方法，避免把设备控制职责带入本服务。
    """

    def __init__(self, status_url: str = ROBOT_CONTROL_STATUS_URL) -> None:
        self._status_url = status_url
        self._opener = build_opener(ProxyHandler({}))

    def read_ar5(self, side: str) -> ArmSnapshot:
        """读取指定侧 AR5 的当前只读状态。"""

        if side not in {"left", "right"}:
            raise ValueError("arm_side 必须是 left 或 right")
        request = Request(self._status_url, method="GET")
        try:
            with self._opener.open(request, timeout=30.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"RobotControl 状态读取失败：{type(error).__name__}: {error}") from error
        return _parse_ar5_status(payload, side)


class CalibrationApplication:
    """串行管理拍摄、先验计算和手眼求解任务。"""

    def __init__(
        self,
        camera_client_factory: Callable[[], CameraPipelineHttpClient],
        prior_dir: Path = RECORD_REPLAY_PRIOR_DIR,
        robot_client: RobotControlReadClient | None = None,
    ) -> None:
        """创建应用门面，不连接相机或机器人。"""

        self._camera_client_factory = camera_client_factory
        self._prior_dir = prior_dir
        self._robot_client = robot_client or RobotControlReadClient()
        self._lock = threading.Lock()
        self._busy = False
        self._hand_eye_samples: list[HandEyeSample] = []
        self._head_eye_samples: dict[str, list[HandEyeSample]] = {
            "left": [],
            "right": [],
        }
        self._active_kind: CalibrationKind | None = None
        self._active_arm_side = "left"
        self._collecting = False
        self._started_at: str | None = None
        self._ended_at: str | None = None
        self._last_error: str | None = None
        self._pending_replacement: PendingReplacement | None = None

    def status(self) -> CalibrationResponse:
        """返回服务任务状态，不读取设备。"""

        with self._lock:
            return CalibrationResponse(
                state="busy" if self._busy else "idle",
                data={
                    "active_calibration_kind": self._active_kind,
                    "active_arm_side": self._active_arm_side,
                    "collecting": self._collecting,
                    "started_at": self._started_at,
                    "ended_at": self._ended_at,
                    "hand_eye_sample_count": len(self._hand_eye_samples),
                    "head_eye_sample_count": {
                        side: len(samples)
                        for side, samples in self._head_eye_samples.items()
                    },
                    "pending_replacement": self._pending_replacement_data(),
                    "prior_dir": str(self._prior_dir),
                },
                error=self._last_error,
            )

    def cancel(self) -> CalibrationResponse:
        """取消当前采样或计算结果，清理缓存但不替换 RecordReplay 文件。"""

        with self._lock:
            if self._busy:
                return CalibrationResponse(
                    state="busy",
                    accepted=False,
                    error="当前任务仍在执行，请等待接口返回后再取消",
                )
            pending = self._pending_replacement
            sample_count = len(self._hand_eye_samples) + sum(
                len(samples) for samples in self._head_eye_samples.values()
            )
            self._hand_eye_samples.clear()
            for samples in self._head_eye_samples.values():
                samples.clear()
            self._pending_replacement = None
            self._active_kind = None
            self._active_arm_side = "left"
            self._collecting = False
            self._started_at = None
            self._ended_at = datetime.now().isoformat(timespec="milliseconds")
            self._last_error = None

        if pending is not None:
            _remove_staging_dir(pending.staging_dir)
        return CalibrationResponse(
            data={
                "cancelled": True,
                "replacement_discarded": pending is not None,
                "discarded_replacement_id": (
                    pending.replacement_id if pending is not None else None
                ),
                "discarded_sample_count": sample_count,
                "replacement_performed": False,
            }
        )

    def record_head_prior(self) -> CalibrationResponse:
        """触发头部 ChArUco 拍摄和计算，等待二次确认后替换头部先验。"""

        def operation() -> dict[str, object]:
            staging_dir = _create_staging_dir()
            try:
                client = self._camera_client_factory()
                try:
                    recorder = PriorCalibrationRecorder(
                        client,
                        PriorCalibrationConfig(output_dir=staging_dir),
                    )
                    result = recorder.record_head_prior()
                finally:
                    client.close()
                return self._cache_replacement(
                    staging_dir,
                    {
                        "calibration_kind": "head",
                        "result_path": _relative_service_path(
                            self._prior_dir / "charuco_board_prior.json"
                        ),
                        "message": result.message,
                    },
                )
            except Exception:
                _remove_staging_dir(staging_dir)
                raise

        return self._run(operation)

    def start_calibration(
        self,
        calibration_kind: CalibrationKind,
        arm_side: str = "left",
    ) -> CalibrationResponse:
        """提示并开始一次标定会话，不连接设备或触发拍摄。"""

        with self._lock:
            if calibration_kind not in {"left_eye_in_hand", "head_eye_to_hand"}:
                return CalibrationResponse(
                    accepted=False,
                    error="calibration_kind 必须是 left_eye_in_hand 或 head_eye_to_hand",
                )
            if arm_side not in {"left", "right"}:
                return CalibrationResponse(accepted=False, error="arm_side 必须是 left 或 right")
            if self._busy:
                return CalibrationResponse(state="busy", accepted=False, error="已有任务正在执行")
            if self._pending_replacement is not None:
                return CalibrationResponse(
                    accepted=False,
                    error="已有待确认的结果替换，请先确认或取消",
                )
            if calibration_kind == "left_eye_in_hand":
                if arm_side != "left":
                    return CalibrationResponse(
                        accepted=False,
                        error="左手眼在手上标定只支持 arm_side=left",
                    )
                self._hand_eye_samples.clear()
            else:
                self._head_eye_samples[arm_side] = []
            self._active_kind = calibration_kind
            self._active_arm_side = arm_side
            self._collecting = True
            self._started_at = datetime.now().isoformat(timespec="milliseconds")
            self._ended_at = None
            self._last_error = None
            return CalibrationResponse(
                data={
                    "calibration_kind": calibration_kind,
                    "arm_side": arm_side,
                    "started_at": self._started_at,
                }
            )

    def end_calibration(
        self,
        calibration_kind: CalibrationKind,
        arm_side: str = "left",
    ) -> CalibrationResponse:
        """提示并结束采样会话，不自动求解、不写入设备。"""

        with self._lock:
            if self._busy:
                return CalibrationResponse(
                    state="busy",
                    accepted=False,
                    error="已有拍摄或计算任务正在执行",
                )
            if self._active_kind != calibration_kind or self._active_arm_side != arm_side:
                return CalibrationResponse(
                    accepted=False,
                    error="当前活动标定会话与请求不匹配",
                )
            self._collecting = False
            self._ended_at = datetime.now().isoformat(timespec="milliseconds")
            sample_count = self._sample_count(calibration_kind, arm_side)
            return CalibrationResponse(
                data={
                    "calibration_kind": calibration_kind,
                    "arm_side": arm_side,
                    "ended_at": self._ended_at,
                    "sample_count": sample_count,
                    "next_action": "调用对应 solve 接口计算并保存结果",
                }
            )

    def record_hand_prior(self, arm_side: str = "left") -> CalibrationResponse:
        """读取当前 AR5 状态并计算手部先验，等待二次确认后替换文件。"""

        def operation() -> dict[str, object]:
            staging_dir = _create_staging_dir()
            try:
                snapshot = self._robot_client.read_ar5(arm_side)
                client = self._camera_client_factory()
                try:
                    recorder = PriorCalibrationRecorder(
                        client,
                        PriorCalibrationConfig(output_dir=staging_dir),
                    )
                    result = recorder.record_ball_prior(snapshot)
                finally:
                    client.close()
                return self._cache_replacement(
                    staging_dir,
                    {
                        "calibration_kind": "hand",
                        "arm_side": arm_side,
                        "result_path": _relative_service_path(
                            self._prior_dir / "ball_pose_prior.json"
                        ),
                        "message": result.message,
                    },
                )
            except Exception:
                _remove_staging_dir(staging_dir)
                raise

        return self._run(operation)

    def capture_hand_eye_sample(self, arm_side: str = "left") -> CalibrationResponse:
        """读取当前 AR5 状态并按需拍摄一组 ChArUco 手眼样本。"""

        def operation() -> dict[str, object]:
            self._require_collecting("left_eye_in_hand", arm_side)
            snapshot = self._robot_client.read_ar5(arm_side)
            client = self._camera_client_factory()
            try:
                response = client.detect_charuco(
                    CharucoDetectionRequest(
                        camera_name=CameraName("left_hand_camera" if arm_side == "left" else "right_hand_camera"),
                        dictionary_name="DICT_APRILTAG_16H5",
                        squares_x=4,
                        squares_y=4,
                        square_length_mm=20.0,
                        marker_length_mm=14.0,
                        min_charuco_corners=6,
                        max_frames=300,
                        stable_timeout_s=10.0,
                        enable_debug=False,
                    )
                )
            finally:
                client.close()
            sample = _build_hand_eye_sample(snapshot, response, "左手眼在手上")
            with self._lock:
                self._hand_eye_samples.append(sample)
                sample_index = len(self._hand_eye_samples)
            return {
                "sample_index": sample_index,
                "arm_side": arm_side,
                "camera_name": response.camera_name.value,
                "marker_count": response.marker_num,
                "charuco_count": response.charuco_num,
                "reprojection_error_px": response.error_px,
            }

        return self._run(operation)

    def capture_head_eye_sample(self, arm_side: str = "left") -> CalibrationResponse:
        """读取当前 AR5 状态并按需拍摄一组头部眼在手外样本。"""

        def operation() -> dict[str, object]:
            self._require_collecting("head_eye_to_hand", arm_side)
            snapshot = self._robot_client.read_ar5(arm_side)
            client = self._camera_client_factory()
            try:
                response = client.detect_charuco(
                    CharucoDetectionRequest(
                        camera_name=HEAD_CAMERA_NAME,
                        dictionary_name="DICT_APRILTAG_16H5",
                        squares_x=4,
                        squares_y=4,
                        square_length_mm=20.0,
                        marker_length_mm=14.0,
                        min_charuco_corners=6,
                        max_frames=300,
                        stable_timeout_s=10.0,
                        enable_debug=False,
                    )
                )
            finally:
                client.close()
            sample = _build_hand_eye_sample(snapshot, response, "头部眼在手外")
            with self._lock:
                samples = self._head_eye_samples[arm_side]
                samples.append(sample)
                sample_index = len(samples)
            return {
                "sample_index": sample_index,
                "arm_side": arm_side,
                "camera_name": response.camera_name.value,
                "marker_count": response.marker_num,
                "charuco_count": response.charuco_num,
                "reprojection_error_px": response.error_px,
            }

        return self._run(operation)

    def solve_hand_eye(
        self,
        payload: Mapping[str, object],
    ) -> CalibrationResponse:
        """计算手眼矩阵并缓存，等待二次确认后替换 `hand_eye_result.txt`。"""

        def operation() -> dict[str, object]:
            self._require_collecting("left_eye_in_hand", "left", allow_ended=True)
            samples = _parse_samples(payload.get("samples")) if "samples" in payload else self._snapshot_hand_eye_samples()
            if len(samples) < 3:
                raise ValueError(f"手眼标定至少需要 3 组样本，实际 {len(samples)}")
            method = _parse_method(payload.get("method", "closed_form"))
            pair_mode = _parse_pair_mode(payload.get("pair_mode", "all"))
            result = calibrate_hand_eye_from_pose_sequences(
                [np.asarray(sample.robot_pose_matrix_m, dtype=np.float64) for sample in samples],
                [np.asarray(sample.camera_board_matrix_m, dtype=np.float64) for sample in samples],
                pair_mode=pair_mode,
                method=method,
            )
            matrix = np.asarray(result.transform.as_SE3(), dtype=np.float64).reshape(4, 4)
            staging_dir = _create_staging_dir()
            try:
                _write_hand_eye(
                    staging_dir / "hand_eye_result.txt",
                    matrix,
                    result.residual.sample_count,
                    method,
                    pair_mode,
                    result.residual.rotation_rmse_deg,
                    result.residual.rotation_max_deg,
                    result.residual.translation_rmse,
                    result.residual.translation_max,
                )
                return self._cache_replacement(
                    staging_dir,
                    {
                        "calibration_kind": "hand_eye",
                        "result_path": _relative_service_path(
                            self._prior_dir / "hand_eye_result.txt"
                        ),
                        "sample_count": result.residual.sample_count,
                        "method": method,
                        "pair_mode": pair_mode,
                        "rotation_rmse_deg": result.residual.rotation_rmse_deg,
                        "rotation_max_deg": result.residual.rotation_max_deg,
                        "translation_rmse_m": result.residual.translation_rmse,
                        "translation_max_m": result.residual.translation_max,
                        "transform": matrix.tolist(),
                    },
                )
            except Exception:
                _remove_staging_dir(staging_dir)
                raise

        return self._run(operation)

    def solve_head_eye(
        self,
        arm_side: str,
        payload: Mapping[str, object],
    ) -> CalibrationResponse:
        """计算头部眼在手外矩阵并缓存，等待二次确认后替换对应 npy。"""

        def operation() -> dict[str, object]:
            self._require_collecting("head_eye_to_hand", arm_side, allow_ended=True)
            samples = (
                _parse_samples(payload.get("samples"))
                if "samples" in payload
                else self._snapshot_head_eye_samples(arm_side)
            )
            if len(samples) < 3:
                raise ValueError(f"头部眼在手外标定至少需要 3 组样本，实际 {len(samples)}")
            matrix = _solve_eye_to_hand(samples)
            staging_dir = _create_staging_dir()
            try:
                _write_matrix_npy(
                    staging_dir / f"{arm_side}_head_base_camera.npy",
                    matrix,
                )
                return self._cache_replacement(
                    staging_dir,
                    {
                        "calibration_kind": "head_eye_to_hand",
                        "arm_side": arm_side,
                        "result_path": _relative_service_path(
                            self._prior_dir / f"{arm_side}_head_base_camera.npy"
                        ),
                        "sample_count": len(samples),
                        "transform_semantics": "T_base_camera",
                        "transform": matrix.tolist(),
                    },
                )
            except Exception:
                _remove_staging_dir(staging_dir)
                raise

        return self._run(operation)

    def get_result(self, result_kind: str, arm_side: str = "left") -> CalibrationResponse:
        """读取已保存的标定或先验结果，不访问设备。"""

        if result_kind == "hand_eye":
            path = self._prior_dir / "hand_eye_result.txt"
            if not path.is_file():
                return CalibrationResponse(accepted=False, error=f"结果文件不存在：{path}")
            matrix = _read_hand_eye_matrix(path)
            return CalibrationResponse(
                data={
                    "result_kind": result_kind,
                    "result_path": _relative_service_path(path),
                    "transform_semantics": "T_tool_cam",
                    "transform": matrix.tolist(),
                    "metrics": _read_hand_eye_metrics(path),
                }
            )
        if result_kind == "head_eye":
            path = self._prior_dir / f"{arm_side}_head_base_camera.npy"
            if not path.is_file():
                return CalibrationResponse(accepted=False, error=f"结果文件不存在：{path}")
            matrix = _matrix4(np.load(path, allow_pickle=False), str(path))
            return CalibrationResponse(
                data={
                    "result_kind": result_kind,
                    "arm_side": arm_side,
                    "result_path": _relative_service_path(path),
                    "transform_semantics": "T_base_camera",
                    "transform": matrix.tolist(),
                }
            )
        if result_kind in {"head_prior", "hand_prior"}:
            file_name = "charuco_board_prior.json" if result_kind == "head_prior" else "ball_pose_prior.json"
            path = self._prior_dir / file_name
            if not path.is_file():
                return CalibrationResponse(accepted=False, error=f"结果文件不存在：{path}")
            value = json.loads(path.read_text(encoding="utf-8"))
            return CalibrationResponse(
                data={
                    "result_kind": result_kind,
                    "result_path": _relative_service_path(path),
                    "result": value,
                }
            )
        raise ValueError(f"不支持的 result_kind：{result_kind}")

    def confirm_replacement(
        self,
        replacement_id: str,
        confirmed: bool,
    ) -> CalibrationResponse:
        """在前端二次确认后替换结果，并将旧文件改名保留。"""

        if not confirmed:
            return CalibrationResponse(
                accepted=False,
                error="必须显式确认 replacement_id 后才能替换文件",
            )
        with self._lock:
            if self._busy:
                return CalibrationResponse(
                    state="busy",
                    accepted=False,
                    error="已有拍摄或计算任务正在执行",
                )
            pending = self._pending_replacement
            if pending is None:
                return CalibrationResponse(
                    accepted=False,
                    error="当前没有待确认的结果替换",
                )
            if pending.replacement_id != replacement_id:
                return CalibrationResponse(
                    accepted=False,
                    error="replacement_id 已失效或不匹配",
                )
            self._busy = True
            self._last_error = None
        try:
            renamed_old_files = _replace_pending(pending)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            with self._lock:
                self._busy = False
                self._last_error = message
            return CalibrationResponse(state="idle", accepted=False, error=message)

        with self._lock:
            self._busy = False
            self._pending_replacement = None
            self._last_error = None
        _remove_staging_dir(pending.staging_dir)
        data = dict(pending.data)
        data.update(
            {
                "replacement_state": "replaced",
                "requires_confirmation": False,
                "replacement_performed": True,
                "renamed_old_files": renamed_old_files,
            }
        )
        return CalibrationResponse(state="idle", data=data)

    def _snapshot_hand_eye_samples(self) -> tuple[HandEyeSample, ...]:
        with self._lock:
            return tuple(self._hand_eye_samples)

    def _snapshot_head_eye_samples(self, arm_side: str) -> tuple[HandEyeSample, ...]:
        with self._lock:
            return tuple(self._head_eye_samples[arm_side])

    def _sample_count(self, calibration_kind: CalibrationKind, arm_side: str) -> int:
        if calibration_kind == "left_eye_in_hand":
            return len(self._hand_eye_samples)
        return len(self._head_eye_samples[arm_side])

    def _require_collecting(
        self,
        calibration_kind: CalibrationKind,
        arm_side: str,
        *,
        allow_ended: bool = False,
    ) -> None:
        with self._lock:
            if self._active_kind != calibration_kind or self._active_arm_side != arm_side:
                raise RuntimeError("请先调用匹配的 calibration/start API")
            if not self._collecting and not allow_ended:
                raise RuntimeError("标定会话已结束，请重新调用 calibration/start API")

    def _pending_replacement_data(self) -> dict[str, object] | None:
        pending = self._pending_replacement
        if pending is None:
            return None
        data = dict(pending.data)
        data.update(
            {
                "replacement_id": pending.replacement_id,
                "replacement_state": "pending_confirmation",
                "requires_confirmation": True,
                "replacement_performed": False,
            }
        )
        return data

    def _cache_replacement(
        self,
        staging_dir: Path,
        data: dict[str, object],
    ) -> dict[str, object]:
        files = _staged_files(staging_dir, self._prior_dir)
        replacement_id = uuid.uuid4().hex
        pending_data = dict(data)
        pending_data.update(
            {
                "replacement_state": "pending_confirmation",
                "requires_confirmation": True,
                "replacement_performed": False,
            }
        )
        pending = PendingReplacement(
            replacement_id=replacement_id,
            staging_dir=staging_dir,
            files=files,
            data=pending_data,
        )
        with self._lock:
            if self._pending_replacement is not None:
                raise RuntimeError("已有待确认的结果替换，请先确认或取消")
            self._pending_replacement = pending
        return {
            **pending_data,
            "replacement_id": replacement_id,
        }

    def _run(self, operation: Callable[[], dict[str, object]]) -> CalibrationResponse:
        with self._lock:
            if self._busy:
                return CalibrationResponse(
                    state="busy",
                    accepted=False,
                    error="已有拍摄或计算任务正在执行",
                )
            if self._pending_replacement is not None:
                return CalibrationResponse(
                    accepted=False,
                    error="已有待确认的结果替换，请先确认或取消",
                )
            self._busy = True
            self._last_error = None
        try:
            data = operation()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            with self._lock:
                self._last_error = message
            return CalibrationResponse(state="idle", accepted=False, error=message)
        finally:
            with self._lock:
                self._busy = False
        return CalibrationResponse(state="idle", data=data)


def _parse_ar5_status(payload: object, side: str) -> ArmSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("RobotControl 状态响应根节点必须是 object")
    devices = payload.get("devices")
    if not isinstance(devices, list):
        raise ValueError("RobotControl 状态缺少 devices 列表")
    name = f"ar5_{side}"
    device = next((item for item in devices if isinstance(item, dict) and item.get("name") == name), None)
    if not isinstance(device, dict):
        raise ValueError(f"RobotControl 状态缺少 {name}")
    data = device.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"RobotControl {name} 缺少 data")
    joints = data.get("joints")
    tcp = data.get("tcp")
    elbow = data.get("elbow")
    if not isinstance(joints, dict) or not isinstance(tcp, dict) or not isinstance(elbow, dict):
        raise ValueError(f"RobotControl {name} 状态字段不完整")
    pose = _matrix_tuple(_matrix4(tcp.get("pose_matrix_m"), "pose_matrix_m"))
    return ArmSnapshot(
        joint_deg=_vector(joints.get("angle_deg"), 7, "angle_deg"),
        pose_matrix_m=pose,
        xyz_mm=_vector3(tcp.get("xyz_mm"), "xyz_mm"),
        rpy_deg=_vector3(tcp.get("rpy_deg"), "rpy_deg"),
        elbow_deg=_number(elbow.get("angle_deg"), "elbow.angle_deg"),
    )


def _parse_samples(value: object) -> tuple[HandEyeSample, ...]:
    if not isinstance(value, list):
        raise ValueError("samples 必须是数组")
    samples: list[HandEyeSample] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"samples[{index}] 必须是 object")
        samples.append(
            HandEyeSample(
                robot_pose_matrix_m=_matrix_tuple(_matrix4(item.get("robot_pose_matrix_m"), f"samples[{index}].robot_pose_matrix_m")),
                camera_board_matrix_m=_matrix_tuple(_matrix4(item.get("camera_board_matrix_m"), f"samples[{index}].camera_board_matrix_m")),
            )
        )
    return tuple(samples)


def _build_hand_eye_sample(
    snapshot: ArmSnapshot,
    response: CharucoDetectionResponse,
    context: str,
) -> HandEyeSample:
    """把 CameraPipeline ChArUco 响应转换为米制手眼样本。"""

    if response.status != "detected" or len(response.t_cam_board_mm) != 4:
        raise RuntimeError(
            f"{context} ChArUco 拍摄失败："
            f"status={response.status} markers={response.marker_num} "
            f"charuco={response.charuco_num}"
        )
    matrix_m = np.asarray(response.t_cam_board_mm, dtype=np.float64).reshape(4, 4)
    matrix_m[:3, 3] *= 0.001
    return HandEyeSample(
        robot_pose_matrix_m=snapshot.pose_matrix_m,
        camera_board_matrix_m=_matrix_tuple(_matrix4(matrix_m, "T_camera_board")),
    )


def _solve_eye_to_hand(samples: tuple[HandEyeSample, ...]) -> np.ndarray:
    """按固定标定板随工具运动的语义求解 `T_base_camera`。"""

    base_tool = [
        _matrix4(sample.robot_pose_matrix_m, "T_base_tool")
        for sample in samples
    ]
    camera_board = [
        _matrix4(sample.camera_board_matrix_m, "T_camera_board")
        for sample in samples
    ]
    tool_base = [np.linalg.inv(matrix) for matrix in base_tool]
    rotation_camera_to_base, translation_camera_to_base = cv2.calibrateHandEye(  # pyright: ignore[reportAttributeAccessIssue]
        R_gripper2base=[matrix[:3, :3] for matrix in tool_base],
        t_gripper2base=[matrix[:3, 3].reshape(3, 1) for matrix in tool_base],
        R_target2cam=[matrix[:3, :3] for matrix in camera_board],
        t_target2cam=[matrix[:3, 3].reshape(3, 1) for matrix in camera_board],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    park_base_camera = np.eye(4, dtype=np.float64)
    park_base_camera[:3, :3] = np.asarray(rotation_camera_to_base, dtype=np.float64).reshape(3, 3)
    park_base_camera[:3, 3] = np.asarray(translation_camera_to_base, dtype=np.float64).reshape(3)
    gripper_board = tuple(
        np.linalg.inv(base_pose) @ park_base_camera @ board_pose
        for base_pose, board_pose in zip(base_tool, camera_board, strict=True)
    )
    mean_gripper_board = _mean_transform(gripper_board)
    base_camera = tuple(
        base_pose @ mean_gripper_board @ np.linalg.inv(board_pose)
        for base_pose, board_pose in zip(base_tool, camera_board, strict=True)
    )
    return _mean_transform(base_camera)


def _mean_transform(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """对一组 SE(3) 做平移均值和旋转几何均值。"""

    values = tuple(_matrix4(matrix, "SE(3)") for matrix in matrices)
    if not values:
        raise ValueError("SE(3) 样本不能为空")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean(np.asarray([matrix[:3, 3] for matrix in values]), axis=0)
    result[:3, :3] = Rotation.from_matrix(
        np.asarray([matrix[:3, :3] for matrix in values])
    ).mean().as_matrix()
    return result


def _parse_method(value: object) -> HandEyeMethod:
    if value not in {"closed_form", "multi_method"}:
        raise ValueError("method 必须是 closed_form 或 multi_method")
    return cast(HandEyeMethod, value)


def _parse_pair_mode(value: object) -> HandEyePairMode:
    if value not in {"all", "adjacent"}:
        raise ValueError("pair_mode 必须是 all 或 adjacent")
    return cast(HandEyePairMode, value)


def _matrix4(value: object, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} 必须是有限的 4x4 矩阵")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{field_name} 最后一行必须为 [0, 0, 0, 1]")
    return matrix


def _matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _vector(value: object, length: int, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{field_name} 必须包含 {length} 个数值")
    values = tuple(float(item) for item in value)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} 必须是有限数值")
    return values


def _vector3(value: object, field_name: str) -> tuple[float, float, float]:
    """读取固定长度为三的向量，保留精确的 tuple 类型。"""

    values = _vector(value, 3, field_name)
    return values[0], values[1], values[2]


def _number(value: object, field_name: str) -> float:
    if not isinstance(value, int | float | str):
        raise ValueError(f"{field_name} 必须是数值")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} 必须是有限数值")
    return result


def _create_staging_dir() -> Path:
    """创建只保存当前进程缓存结果的临时目录。"""

    return Path(tempfile.mkdtemp(prefix="dingtai-calibration-"))


def _remove_staging_dir(staging_dir: Path) -> None:
    """删除已知的计算缓存目录，不触碰 RecordReplay 正式结果。"""

    if staging_dir.exists():
        shutil.rmtree(staging_dir)


def _staged_files(
    staging_dir: Path,
    target_dir: Path,
) -> tuple[tuple[Path, Path], ...]:
    """构造临时结果到固定先验目录的文件映射。"""

    files = tuple(
        (source, target_dir / source.relative_to(staging_dir))
        for source in sorted(staging_dir.rglob("*"))
        if source.is_file()
    )
    if not files:
        raise RuntimeError("计算没有生成可替换的结果文件")
    return files


def _replace_pending(pending: PendingReplacement) -> list[str]:
    """确认替换结果，并将每个旧文件改名保留而非删除。"""

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    renamed: list[tuple[Path, Path]] = []
    moved: list[tuple[Path, Path]] = []
    backup_paths: list[tuple[Path, Path]] = []
    for source, target in pending.files:
        if not source.is_file():
            raise FileNotFoundError(f"待确认结果文件不存在：{source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not target.is_file():
            raise ValueError(f"正式结果目标不是文件：{target}")
        if target.is_file():
            backup = target.with_name(f"{target.stem}_{timestamp}{target.suffix}")
            if backup.exists():
                raise FileExistsError(f"时间戳备份文件已存在：{backup}")
            backup_paths.append((target, backup))

    try:
        for target, backup in backup_paths:
            target.rename(backup)
            renamed.append((target, backup))
        for source, target in pending.files:
            source.rename(target)
            moved.append((source, target))
    except Exception:
        for source, target in reversed(moved):
            if target.is_file() and not source.exists():
                target.rename(source)
        for target, backup in reversed(renamed):
            if backup.is_file() and not target.exists():
                backup.rename(target)
        raise

    return [
        _relative_service_path(backup)
        for _, backup in renamed
    ]


def _write_hand_eye(
    target: Path,
    matrix: np.ndarray,
    sample_count: int,
    method: HandEyeMethod,
    pair_mode: HandEyePairMode,
    rotation_rmse_deg: float,
    rotation_max_deg: float,
    translation_rmse_m: float,
    translation_max_m: float,
) -> None:
    """以 RecordReplay 可读取的格式原子写入 T_tool_cam。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    lines = [
        "Hand-eye calibration result",
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        "Formula: T_base_board = T_base_tool @ T_tool_cam @ T_cam_board",
        f"method={method}",
        f"pair_mode={pair_mode}",
        f"sample_count={sample_count}",
        f"rotation_rmse_deg={rotation_rmse_deg}",
        f"rotation_max_deg={rotation_max_deg}",
        f"translation_rmse_m={translation_rmse_m}",
        f"translation_max_m={translation_max_m}",
        "",
        "T_tool_cam:",
        np.array2string(matrix, precision=10, suppress_small=False),
        "",
    ]
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(target)


def _relative_service_path(path: Path) -> str:
    return str(path.relative_to(SERVICE_ROOT.parent)).replace("\\", "/")


def _write_matrix_npy(target: Path, matrix: np.ndarray) -> None:
    """原子写入 RecordReplay 读取的 4x4 `.npy` 矩阵。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as file:
        np.save(file, _matrix4(matrix, str(target)), allow_pickle=False)
    temporary.replace(target)


def _read_hand_eye_matrix(path: Path) -> np.ndarray:
    """读取手眼结果文本中的 `T_tool_cam` 矩阵。"""

    rows: list[list[float]] = []
    collecting = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "T_tool_cam:":
            collecting = True
            continue
        if collecting and not line.strip():
            break
        if collecting:
            values = [
                float(token)
                for token in line.replace("[", " ").replace("]", " ").split()
            ]
            if len(values) != 4:
                raise ValueError(f"手眼矩阵行格式错误：{line}")
            rows.append(values)
            if len(rows) == 4:
                break
    return _matrix4(rows, str(path))


def _read_hand_eye_metrics(path: Path) -> dict[str, object]:
    """读取手眼结果文本中供前端展示的求解元数据。"""

    metrics: dict[str, object] = {}
    allowed_keys = {
        "method",
        "pair_mode",
        "sample_count",
        "rotation_rmse_deg",
        "rotation_max_deg",
        "translation_rmse_m",
        "translation_max_m",
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key not in allowed_keys:
            continue
        if key in {"method", "pair_mode"}:
            metrics[key] = value
        elif key == "sample_count":
            metrics[key] = int(value)
        else:
            metrics[key] = float(value)
    return metrics
