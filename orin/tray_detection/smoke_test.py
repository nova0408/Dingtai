from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .protocol import OrinTrayDetectionRequest
from .transport import OrinTrayDetectionRpcClient, ZmqSocketOptions


DEFAULT_SERVICE_ADDR = "tcp://127.0.0.1:6210"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "smoke_test_local"


def main(service_addr: str = DEFAULT_SERVICE_ADDR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = OrinTrayDetectionRpcClient(
        connect_addr=str(service_addr),
        options=ZmqSocketOptions(recv_timeout_ms=30_000, send_timeout_ms=30_000),
    )
    try:
        response = client.call(OrinTrayDetectionRequest(request_id=1, frame_id=-1, enable_debug=True))
    finally:
        client.close()
    if response.debug is not None and response.debug.overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "overlay.jpg"), np.asarray(response.debug.overlay_bgr, dtype=np.uint8))
    if response.debug is not None and response.debug.mask_bgr is not None:
        cv2.imwrite(str(output_dir / "mask_preview.jpg"), np.asarray(response.debug.mask_bgr, dtype=np.uint8))
    if response.debug is not None and len(response.debug.tray_masks) > 0:
        cv2.imwrite(str(output_dir / "tray_mask.png"), np.asarray(response.debug.tray_masks[0], dtype=np.uint8))
    (output_dir / "summary.txt").write_text(
        "frame_id={0}\ntray_count={1}\nerror={2}\n".format(response.frame_id, response.tray_count, response.error),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orin tray detection local smoke test")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    raise SystemExit(main(service_addr=str(args.service_addr), output_dir=Path(args.output_dir)))
