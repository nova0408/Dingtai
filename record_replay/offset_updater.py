"""触发 CSV 后更新全局笛卡尔纠偏矩阵。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .arm_gateway import CartesianPose, apply_named_toolset, read_cart_posture, retry_non_motion_call
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

    def update(self, runtime: ReplayRuntime) -> None:
        """读取当前 TCP 与三球检测结果，计算并写入全局 T_off。"""

        _raise_if_stopped(runtime)
        apply_named_toolset(
            runtime.connected_arm,
            runtime.settings,
            runtime.stop_event,
        )
        tcp_pose = retry_non_motion_call(
            f"cartPosture(endInRef, offset-calc:{runtime.connected_arm.arm_side})",
            lambda: read_cart_posture(runtime.connected_arm),
            runtime.settings.non_motion_retry_count,
            runtime.settings.non_motion_retry_delay_s,
            runtime.stop_event,
        )
        tcp_matrix = _cartesian_pose_to_matrix(tcp_pose)
        _raise_if_stopped(runtime)
        prior_base_ball = load_prior_base_ball_transform(self._config.prior_capture_path, self._config.hand_eye_result_path)
        if runtime.stop_event.wait(timeout=max(0.0, runtime.settings.offset.capture_settle_delay_s)):
            raise RuntimeError("检测到停止请求，终止三球采样前稳定等待")
        samples_mm = [
            np.asarray(sample, dtype=np.float64)
            for sample in self._detector.capture_samples(runtime.settings.offset.sample_count)
        ]
        _raise_if_stopped(runtime)
        camera_ball = camera_ball_transform_m(samples_mm, runtime.settings.offset)
        runtime.global_cartesian_offset = calculate_global_offset(
            tcp_matrix,
            load_tool_camera_transform(self._config.hand_eye_result_path),
            camera_ball,
            prior_base_ball,
        )


def _raise_if_stopped(runtime: ReplayRuntime) -> None:
    """阻止停止锁存后的拍摄算法继续请求设备。"""

    if runtime.stop_event.is_set():
        raise RuntimeError("检测到停止请求，禁止继续执行三球拍摄")


def _cartesian_pose_to_matrix(pose: CartesianPose) -> np.ndarray:
    """将 SDK 的 trans(m)+rpy(rad) 位姿转换为 4x4 齐次矩阵。"""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("xyz", pose.rpy_rad, degrees=False).as_matrix()
    matrix[:3, 3] = np.asarray(pose.trans_m, dtype=np.float64)
    return matrix
