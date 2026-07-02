from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from camera_pipeline.client import CameraPipelineClient  
DEFAULT_READY_TIMEOUT_S = 10.0
DEFAULT_ORIN_SERVICE_ADDR = "tcp://192.168.1.118:6200"


def main() -> int:
    """测试本机到 Orin 统一相机服务是否能获取到第一帧摘要。"""

    logger.info("硬件测试脚本：需要连通 Orin 统一 camera_pipeline_service，未连硬件时会失败")
    client = CameraPipelineClient(service_addr=DEFAULT_ORIN_SERVICE_ADDR, timeout_ms=30_000)
    try:
        summary_response = client.get_camera_summary(timeout_s=float(DEFAULT_READY_TIMEOUT_S))
        summary = {
            "frame_id": int(summary_response.frame_id),
            "camera_name": str(summary_response.camera_name),
            "timestamp_ms": float(summary_response.timestamp_ms),
            "color_shape": [int(value) for value in summary_response.color_shape],
            "depth_shape": [int(value) for value in summary_response.depth_shape],
            "fx": float(summary_response.fx),
            "fy": float(summary_response.fy),
            "cx": float(summary_response.cx),
            "cy": float(summary_response.cy),
            "source_meta": dict(summary_response.source_meta),
        }
        logger.success("Orin 相机首帧摘要获取成功")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
