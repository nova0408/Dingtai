"""CameraPipeline OpenCV ChArUco 字典解析离线验证。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.charuco_detection import (
    available_aruco_dictionary_names,
    get_predefined_aruco_dictionary,
)


def test_dictionary_names_match_opencv_and_build_board_dictionary() -> None:
    names = available_aruco_dictionary_names()
    assert names == tuple(sorted(set(names)))
    assert "DICT_5X5_1000" in names
    assert get_predefined_aruco_dictionary("DICT_5X5_1000") is not None


def main() -> None:
    """IDE 直跑入口；不连接相机。"""

    test_dictionary_names_match_opencv_and_build_board_dictionary()
    logger.success("CameraPipeline OpenCV ChArUco 字典离线验证通过")
    logger.warning("本测试未连接真实相机")


if __name__ == "__main__":
    main()
