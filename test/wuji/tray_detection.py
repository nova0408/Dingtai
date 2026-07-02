from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "camera_pipeline").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.118:6210"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "tray_detection_capture"

from camera_pipeline.tray_detection.protocol import OrinTrayDetectionRequest
from camera_pipeline.tray_detection.transport import OrinTrayDetectionRpcClient, ZmqSocketOptions


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("tray_detection smoke test start")
    client = OrinTrayDetectionRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=30_000, send_timeout_ms=30_000),
    )
    try:
        response = client.call(
            OrinTrayDetectionRequest(
                request_id=1,
                camera_name=str(camera_name),
                frame_id=-1,
                enable_debug=True,
            )
        )
    finally:
        client.close()
    if response.error is not None:
        raise RuntimeError(response.error)
    if response.tray_count <= 0 or len(response.tray_results) == 0:
        raise RuntimeError("tray detection returned no results")
    _save_capture(output_dir, response)
    print(
        json.dumps(
            {
                "frame_id": response.frame_id,
                "camera_name": response.camera_name,
                "tray_count": response.tray_count,
                "elapsed_ms": response.elapsed_ms,
                "error": response.error,
                "tray_ids": [item.tray_id for item in response.tray_results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _save_capture(output_dir: Path, response: Any) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "frame_id": response.frame_id,
                "camera_name": response.camera_name,
                "tray_count": response.tray_count,
                "elapsed_ms": response.elapsed_ms,
                "error": response.error,
                "tray_ids": [item.tray_id for item in response.tray_results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if response.debug is not None and response.debug.overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "overlay.jpg"), np.asarray(response.debug.overlay_bgr, dtype=np.uint8))
    if response.debug is not None and response.debug.mask_bgr is not None:
        cv2.imwrite(str(output_dir / "mask.jpg"), np.asarray(response.debug.mask_bgr, dtype=np.uint8))


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tray detection smoke test")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(main(service_addr=str(args.service_addr), camera_name=str(args.camera_name), output_dir=Path(args.output_dir)))
