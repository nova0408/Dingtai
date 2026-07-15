"""触发 CSV 后更新全局笛卡尔纠偏矩阵。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from sdk.xcoresdk import xCoreSDK_python

from .arm_gateway import apply_named_toolset, retry_non_motion_call
from .offset_detection import camera_ball_transform_m
from .offset_detector_gateway import ThreeBallDetector
from .offset_math import calculate_global_offset, load_prior_base_ball_transform, load_tool_camera_transform
from .runtime import ReplayRuntime
from .settings import OffsetConfig


class GlobalOffsetUpdater:
    """在指定左臂 CSV 完成后采样三球并更新运行时 T_off。"""

    def __init__(self, config: OffsetConfig, detector: ThreeBallDetector) -> None:
        self._config = config
        self._detector = detector

    def should_update_after(self, runtime: ReplayRuntime, csv_name: str) -> bool:
        """判断当前文件是否为左臂 offset 计算触发文件。"""

        return (
            runtime.connected_arm.arm_side == "left"
            and _extract_sequence(csv_name) == runtime.settings.offset.calculate_at_sequence
        )

    def update(self, runtime: ReplayRuntime) -> None:
        """读取当前 TCP 与三球检测结果，计算并写入全局 T_off。"""

        robot = runtime.connected_arm.robot
        ec = runtime.connected_arm.ec
        apply_named_toolset(runtime.connected_arm, runtime.settings)
        tcp_pose = retry_non_motion_call(
            f"cartPosture(endInRef, offset-calc:{runtime.connected_arm.arm_side})",
            lambda: robot.cartPosture(xCoreSDK_python.endInRef, ec),
            runtime.settings.non_motion_retry_count,
            runtime.settings.non_motion_retry_delay_s,
        )
        if ec.get("ec", 0) != 0:
            raise RuntimeError("读取当前 TCP 位姿失败，无法计算全局 offset")
        tcp_matrix = _cartesian_pose_to_matrix(tcp_pose)
        prior_base_ball = load_prior_base_ball_transform(self._config.prior_capture_path, self._config.hand_eye_result_path)
        samples_mm = [
            np.asarray(sample, dtype=np.float64)
            for sample in self._detector.capture_samples(runtime.settings.offset.sample_count)
        ]
        camera_ball = camera_ball_transform_m(samples_mm, runtime.settings.offset)
        runtime.global_cartesian_offset = calculate_global_offset(
            tcp_matrix,
            load_tool_camera_transform(self._config.hand_eye_result_path),
            camera_ball,
            prior_base_ball,
        )


def _cartesian_pose_to_matrix(pose: xCoreSDK_python.CartesianPosition) -> np.ndarray:
    """将 SDK 的 trans(m)+rpy(rad) 位姿转换为 4x4 齐次矩阵。"""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("XYZ", pose.rpy, degrees=False).as_matrix()
    matrix[:3, 3] = np.asarray(pose.trans, dtype=np.float64)
    return matrix


def _extract_sequence(csv_name: str) -> int:
    """解析 CSV 文件名前缀阶段序号。"""

    return int(csv_name.split("_", maxsplit=1)[0])
