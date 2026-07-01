from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_SERVICE_ADDR = "tcp://127.0.0.1:6210"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "tray_detection_capture"

if (PROJECT_ROOT / "orin").is_dir():
    from orin.tray_detection.protocol import OrinTrayDetectionRequest
    from orin.tray_detection.transport import OrinTrayDetectionRpcClient, ZmqSocketOptions
else:
    from camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig
    from tray_detection.engine import OrinTrayDetectionExecutor, OrinTrayDetectionExecutorConfig
    from tray_detection.protocol import OrinTrayDetectionRequest


def main(service_addr: str = DEFAULT_SERVICE_ADDR, camera_name: str = DEFAULT_CAMERA_NAME, output_dir: Path = DEFAULT_OUTPUT_DIR) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (PROJECT_ROOT / "orin").is_dir():
        return _run_service_mode(service_addr=service_addr, camera_name=camera_name, output_dir=output_dir)
    return _run_library_mode(camera_name=camera_name, output_dir=output_dir)


def _run_service_mode(service_addr: str, camera_name: str, output_dir: Path) -> int:
    logger.info("tray_detection service mode start")
    client = OrinTrayDetectionRpcClient(connect_addr=str(service_addr), options=ZmqSocketOptions(recv_timeout_ms=30_000, send_timeout_ms=30_000))
    try:
        response = client.call(OrinTrayDetectionRequest(request_id=1, camera_name=str(camera_name), frame_id=-1, enable_debug=True))
    finally:
        client.close()
    print(_emit_debug(response, output_dir))
    return 0


def _run_library_mode(camera_name: str, output_dir: Path) -> int:
    logger.info("tray_detection library mode start")
    runtime = CameraStreamRuntime(
        CameraStreamRuntimeConfig(
            camera_name=str(camera_name),
        )
    )
    runtime.start()
    try:
        if not runtime.wait_until_ready(timeout_s=8.0):
            raise RuntimeError("camera stream not ready")
        executor = OrinTrayDetectionExecutor(frame_runtime=runtime, config=OrinTrayDetectionExecutorConfig())
        print(f"[tray] camera ready camera={camera_name}")
        response = executor.process_request(OrinTrayDetectionRequest(request_id=1, camera_name=str(camera_name), frame_id=-1, enable_debug=True))
        print(f"[tray] frame_id={response.frame_id} tray_count={response.tray_count} elapsed_ms={response.elapsed_ms:.1f} error={response.error}")
        for tray in response.tray_results:
            print(f"[tray] tray_id={tray.tray_id} bbox={tray.bbox_xywh} center_uv={tray.center_uv} conf={tray.confidence_2d:.3f} source={tray.source}")
        _save_tray_capture(output_dir, response)
    finally:
        runtime.stop()
    return 0


def _emit_debug(response: Any, output_dir: Path) -> str:
    _save_tray_capture(output_dir, response)
    summary = {
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "tray_count": response.tray_count,
        "selected_tray_index": None,
        "elapsed_ms": response.elapsed_ms,
        "error": response.error,
        "all_tray_count": len(response.tray_results),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    return text


def _save_tray_capture(output_dir: Path, response: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "frame_id": response.frame_id,
                "camera_name": response.camera_name,
                "tray_count": response.tray_count,
                "selected_tray_index": None,
                "elapsed_ms": response.elapsed_ms,
                "error": response.error,
                "all_tray_count": len(response.tray_results),
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
