"""手眼 ChArUco 默认参数接口的离线验证。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import cast

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration_service.service.application import (
    ArmSnapshot,
    CalibrationApplication,
    RobotControlReadClient,
)
from calibration_service.service.camera_client import (
    CharucoDetectionResponse,
    CharucoDetectionRequest,
    CameraPipelineHttpClient,
)
from calibration_service.service.charuco_config import available_aruco_dictionary_names


class _FakeRobotClient:
    """提供固定 AR5 快照，不连接现场设备。"""

    def read_ar5(self, side: str) -> ArmSnapshot:
        assert side == "left"
        return ArmSnapshot(
            joint_deg=(0.0,) * 7,
            pose_matrix_m=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            xyz_mm=(0.0, 0.0, 0.0),
            rpy_deg=(0.0, 0.0, 0.0),
            elbow_deg=0.0,
        )


class _FakeCameraClient:
    """记录检测请求并返回固定的检测成功结果。"""

    def __init__(self) -> None:
        self.requests: list[CharucoDetectionRequest] = []

    def detect_charuco(self, request: CharucoDetectionRequest) -> CharucoDetectionResponse:
        self.requests.append(request)
        return CharucoDetectionResponse(
            status="detected",
            camera_name=request.camera_name,
            t_cam_board_mm=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            error_px=0.1,
            marker_num=8,
            charuco_num=9,
            overlay_bgr=np.empty((0, 0, 3), dtype=np.uint8),
        )

    def close(self) -> None:
        """模拟客户端关闭。"""


def test_config_lists_opencv_dictionaries_and_updates_capture_request() -> None:
    camera = _FakeCameraClient()
    application = CalibrationApplication(
        camera_client_factory=cast(Callable[[], CameraPipelineHttpClient], lambda: camera),
        robot_client=cast(RobotControlReadClient, _FakeRobotClient()),
    )

    config_response = application.get_hand_eye_config()
    config_data = config_response.data
    available_names = config_data["available_dictionary_names"]
    assert isinstance(available_names, list)
    assert "DICT_5X5_1000" in available_names

    update_response = application.update_hand_eye_config(
        {
            "dictionary_name": "DICT_5X5_1000",
            "squares_x": 12,
            "squares_y": 9,
            "square_length_mm": 15.0,
            "marker_length_mm": 11.0,
            "min_charuco_corners": 8,
            "max_frames": 20,
            "stable_timeout_s": 2.5,
            "enable_debug": True,
        }
    )
    assert update_response.accepted
    updated_config = update_response.data["config"]
    assert isinstance(updated_config, dict)
    assert updated_config["dictionary_name"] == "DICT_5X5_1000"

    assert application.start_calibration("left_eye_in_hand").accepted
    sample_response = application.capture_hand_eye_sample()
    assert sample_response.accepted
    request = camera.requests[0]
    assert request.dictionary_name == "DICT_5X5_1000"
    assert request.squares_x == 12
    assert request.squares_y == 9
    assert request.square_length_mm == 15.0
    assert request.marker_length_mm == 11.0
    assert request.min_charuco_corners == 8
    assert request.max_frames == 20
    assert request.stable_timeout_s == 2.5
    assert request.enable_debug


def main() -> None:
    """IDE 直跑入口；仅使用伪造客户端，不连接机器人或相机。"""

    assert available_aruco_dictionary_names()
    test_config_lists_opencv_dictionaries_and_updates_capture_request()
    logger.success("手眼 ChArUco 默认参数和 OpenCV 字典离线验证通过")
    logger.warning("本测试未连接真实 RobotControl、CameraPipeline 或硬件")


if __name__ == "__main__":
    main()
