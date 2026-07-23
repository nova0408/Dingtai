from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "camera_pipeline" / "client.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraName, CameraPipelineClient
from camera_pipeline.service.protocol import CharucoDetectionRequest, CharucoDetectionResponse

# region 默认配置

DEFAULT_SERVICE_ADDR = "tcp://192.168.1.128:6200" if sys.platform == "win32" else "tcp://127.0.0.1:6200"
"CameraPipeline 服务地址；Windows 访问 Orin 管理网，Orin 平铺脚本访问本机服务。"

DEFAULT_CAMERA_NAME = "head_camera"
"ChArUco 检测使用的逻辑相机名称。"

DEFAULT_TIMEOUT_MS = 60_000
"RPC 收发超时时间，单位 ms。"

DEFAULT_STABLE_TIMEOUT_S = 10.0
"等待单帧稳定图像的超时时间，单位 s。"

DEFAULT_MAX_FRAMES = 60
"单次检测最多检查的稳定帧数。"

DEFAULT_DICTIONARY_NAME = "DICT_APRILTAG_16H5"
"目标 ChArUco 板使用的 ArUco 字典名称。"

DEFAULT_SQUARES_X = 4
"目标 ChArUco 板横向方格数量。"

DEFAULT_SQUARES_Y = 4
"目标 ChArUco 板纵向方格数量。"

DEFAULT_SQUARE_LENGTH_MM = 20.0
"目标 ChArUco 板方格边长，单位 mm。"

DEFAULT_MARKER_LENGTH_MM = 14.0
"目标 ChArUco 板 marker 边长，单位 mm。"

DEFAULT_MIN_CHARUCO_CORNERS = 6
"判定目标板有效所需的最少 ChArUco 角点数量。"

# endregion


# region 主流程


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    stable_timeout_s: float = DEFAULT_STABLE_TIMEOUT_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> int:
    """通过 CameraPipeline 服务执行一次真实 ChArUco 检测。"""

    selected_camera = CameraName(camera_name)
    logger.info(
        "ChArUco 服务测试开始 service_addr={} camera_name={} max_frames={}",
        service_addr,
        selected_camera,
        max_frames,
    )
    client = CameraPipelineClient(service_addr=service_addr, timeout_ms=timeout_ms)
    try:
        camera_status = client.get_camera_status(selected_camera, timeout_s=stable_timeout_s)
        if not camera_status.online:
            raise RuntimeError(f"相机当前不在线：camera_name={selected_camera}")
        response = client.detect_charuco(
            CharucoDetectionRequest(
                camera_name=selected_camera,
                dictionary_name=DEFAULT_DICTIONARY_NAME,
                squares_x=DEFAULT_SQUARES_X,
                squares_y=DEFAULT_SQUARES_Y,
                square_length_mm=DEFAULT_SQUARE_LENGTH_MM,
                marker_length_mm=DEFAULT_MARKER_LENGTH_MM,
                min_charuco_corners=DEFAULT_MIN_CHARUCO_CORNERS,
                max_frames=max_frames,
                stable_timeout_s=stable_timeout_s,
            )
        )
    finally:
        client.close()

    _validate_response(response)
    print(json.dumps(_build_summary(response), ensure_ascii=False, indent=2))
    logger.success(
        "ChArUco 服务测试通过 status={} marker_num={} charuco_num={} error_px={}",
        response.status,
        response.marker_num,
        response.charuco_num,
        response.error_px,
    )
    return 0


# endregion


# region 校验与输出


def _validate_response(response: CharucoDetectionResponse) -> None:
    """验证服务响应包含可用于实验的完整目标板位姿。"""

    if response.status != "detected":
        raise RuntimeError(
            "ChArUco 目标板未检测成功："
            f"status={response.status} marker_num={response.marker_num} charuco_num={response.charuco_num}"
        )
    if response.charuco_num < DEFAULT_MIN_CHARUCO_CORNERS:
        raise RuntimeError(
            "ChArUco 有效角点不足："
            f"expected>={DEFAULT_MIN_CHARUCO_CORNERS} actual={response.charuco_num}"
        )
    if not math.isfinite(response.error_px):
        raise RuntimeError(f"ChArUco 重投影误差不是有限值：{response.error_px}")
    if len(response.t_cam_board_mm) != 4 or any(len(row) != 4 for row in response.t_cam_board_mm):
        raise RuntimeError("ChArUco 响应缺少 4x4 T_camera_board 矩阵")
    if not all(math.isfinite(value) for row in response.t_cam_board_mm for value in row):
        raise RuntimeError("ChArUco T_camera_board 矩阵包含非有限值")


def _build_summary(response: CharucoDetectionResponse) -> dict[str, str | int | float | list[list[float]]]:
    """构造适合终端查看的检测摘要。"""

    return {
        "status": response.status,
        "marker_num": response.marker_num,
        "charuco_num": response.charuco_num,
        "error_px": response.error_px,
        "t_cam_board_mm": [list(row) for row in response.t_cam_board_mm],
    }


# endregion


# region CLI


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """解析批量测试所需的薄 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="CameraPipeline ChArUco 真实相机测试")
    parser.add_argument("--service-addr", default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--stable-timeout-s", type=float, default=DEFAULT_STABLE_TIMEOUT_S)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(main())
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=args.service_addr,
            camera_name=args.camera_name,
            timeout_ms=args.timeout_ms,
            stable_timeout_s=args.stable_timeout_s,
            max_frames=args.max_frames,
        )
    )


# endregion
