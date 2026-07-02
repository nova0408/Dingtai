from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from camera_pipeline.client import CameraPipelineClient

DEFAULT_ORIN_SERVICE_ADDR = "tcp://192.168.1.118:6200"
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_FRAMES = 5


def main() -> int:
    """验证 Orin 侧相机状态、内参与订阅帧流是否可用。"""

    args = _parse_cli()
    client = CameraPipelineClient(service_addr=args.service_addr, timeout_ms=int(args.timeout_s * 1000.0))
    try:
        status = client.get_camera_status(timeout_s=float(args.timeout_s))
        intrinsics = client.get_camera_intrinsics(timeout_s=float(args.timeout_s))
        logger.success("相机状态与内参获取成功")
        print({
            "camera_name": status.camera_name,
            "camera_id": status.camera_id,
            "camera_model": status.camera_model,
            "width": status.width,
            "height": status.height,
            "online": status.online,
            "color_enabled": status.color_enabled,
            "depth_enabled": status.depth_enabled,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "distortion": intrinsics.distortion,
        })

        frame_count = 0
        for frame in client.subscribe_camera_frames(args.camera_name):
            frame_count += 1
            color_bgr = np.asarray(frame.color_bgr, dtype=np.uint8)
            logger.info(
                "frame {}: id={} shape={}x{} dtype={} ts_ms={} source_meta={}",
                frame_count,
                int(frame.frame_id),
                int(color_bgr.shape[1]),
                int(color_bgr.shape[0]),
                color_bgr.dtype,
                float(frame.timestamp_ms),
                frame.source_meta,
            )
            if args.preview:
                cv2.imshow("Orin Camera Stream", color_bgr)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    logger.warning("收到退出指令，结束测试。")
                    break
                if cv2.getWindowProperty("Orin Camera Stream", cv2.WND_PROP_VISIBLE) < 1:
                    logger.warning("预览窗口关闭，结束测试。")
                    break
            if frame_count >= int(args.max_frames):
                break

        if args.preview:
            cv2.destroyAllWindows()

        if frame_count == 0:
            raise RuntimeError("未收到任何订阅帧")
        logger.success("Orin 相机流测试完成，共收到 {} 帧", frame_count)
        return 0
    finally:
        client.close()


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 Orin 相机状态、内参和订阅帧流")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_ORIN_SERVICE_ADDR, help="camera_pipeline_service 地址")
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME, help="逻辑相机名")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="状态/内参请求超时")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="最多消费多少帧")
    parser.add_argument("--preview", action="store_true", help="显示实时预览窗口")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
