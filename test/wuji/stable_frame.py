from __future__ import annotations

import argparse
from pathlib import Path
import sys

from loguru import logger

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "camera_pipeline" / "client.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraName, CameraPipelineClient

# region 默认常量

DEFAULT_CAMERA_NAME = "left_hand_camera"
# Windows 开发机访问 Orin 管理网 IP；Orin 平铺部署脚本仅访问 localhost。
DEFAULT_SERVICE_ADDR = (
    "tcp://192.168.1.128:6200"
    if sys.platform == "win32"
    else "tcp://127.0.0.1:6200"
)
# RPC socket 收发超时时间，单位 ms。
DEFAULT_TIMEOUT_MS = 60_000
# 服务端等待稳定帧的最长时间，单位 s。
DEFAULT_STABLE_TIMEOUT_S = 15.0

# endregion


# region 主流程


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    stable_timeout_s: float = DEFAULT_STABLE_TIMEOUT_S,
) -> int:
    """验证指定相机在线，并通过正式 API 获取有效稳定帧。"""

    selected_camera = CameraName(camera_name)
    client = CameraPipelineClient(service_addr=service_addr, timeout_ms=timeout_ms)
    logger.info(
        "稳定帧 API 测试开始：service_addr={} camera_name={} timeout={} s",
        service_addr,
        camera_name,
        stable_timeout_s,
    )
    try:
        status = client.get_camera_status(selected_camera, timeout_s=stable_timeout_s)
        if status.camera_name != camera_name or not status.online:
            raise RuntimeError(f"相机状态无效：{status!r}")

        stable_frame = client.get_stable_frame(
            selected_camera,
            timeout_s=stable_timeout_s,
        )
        if stable_frame.camera_name != camera_name:
            raise RuntimeError(
                "稳定帧相机不匹配："
                f"expected={camera_name} actual={stable_frame.camera_name}"
            )
        if stable_frame.frame_id <= 0 or stable_frame.timestamp_ms <= 0.0:
            raise RuntimeError(f"稳定帧响应无效：{stable_frame!r}")

        logger.success(
            "稳定帧 API 验证通过：camera_name={} frame_id={} timestamp_ms={} ms",
            stable_frame.camera_name,
            stable_frame.frame_id,
            stable_frame.timestamp_ms,
        )
    finally:
        client.close()
    return 0


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """解析命令行覆盖参数，默认值同时支持 IDE 直接运行。"""

    parser = argparse.ArgumentParser(description="camera stable frame API smoke test")
    parser.add_argument("--service-addr", default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument(
        "--stable-timeout-s",
        type=float,
        default=DEFAULT_STABLE_TIMEOUT_S,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=args.service_addr,
            camera_name=args.camera_name,
            timeout_ms=args.timeout_ms,
            stable_timeout_s=args.stable_timeout_s,
        )
    )


# endregion
