from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.wuji.pose_context import WujiPoseExecutionContext  # noqa: E402


DEFAULT_SERVICE_ADDR = "tcp://192.168.1.116:6200"
DEFAULT_CAMERA_NAME = "left_hand_camera"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / "artifacts" / "grasp_pose_rpc_smoke"


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_tray_index: int = 0,
    enable_debug: bool = True,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = WujiPoseExecutionContext(
        service_addr=str(service_addr),
        camera_id=str(camera_name),
    )
    result = context.run_once(target_tray_index=int(target_tray_index), enable_debug=bool(enable_debug), frame_id=1)
    _save_response(output_dir, result.response)
    print(json.dumps(_summary(result), ensure_ascii=False, indent=2))
    return 0


def _save_response(output_dir: Path, response: Any) -> None:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_summary_from_response(response), ensure_ascii=False, indent=2), encoding="utf-8")
    if response is None or response.debug is None:
        return
    if response.debug.overlay_bgr is not None:
        cv2.imwrite(str(output_dir / "overlay.jpg"), response.debug.overlay_bgr)
    if response.debug.contrast_bgr is not None:
        cv2.imwrite(str(output_dir / "contrast.jpg"), response.debug.contrast_bgr)
    if response.debug.selected_tray_mask is not None:
        cv2.imwrite(str(output_dir / "selected_tray_mask.png"), response.debug.selected_tray_mask)
    if response.debug.near_plane_mask is not None:
        cv2.imwrite(str(output_dir / "near_plane_mask.png"), response.debug.near_plane_mask)
    if response.debug.no_hole_mask is not None:
        cv2.imwrite(str(output_dir / "no_hole_mask.png"), response.debug.no_hole_mask)


def _summary(result: Any) -> Dict[str, Any]:
    return _summary_from_response(result.response, fallback_error=result.error, frame_id=result.frame_id)


def _summary_from_response(response: Any, fallback_error: Optional[str] = None, frame_id: Optional[int] = None) -> Dict[str, Any]:
    selected = None
    if response is not None and response.selected_result is not None:
        selected = {
            "tray_index": response.selected_result.tray_index,
            "pose": None
            if response.selected_result.pose is None
            else {
                "grasp_point_mm": list(response.selected_result.pose.grasp_point_mm),
                "pre_grasp_point_mm": list(response.selected_result.pose.pre_grasp_point_mm),
                "rpy_deg": list(response.selected_result.pose.rpy_deg),
            },
        }
    return {
        "frame_id": frame_id if frame_id is not None else (None if response is None else response.frame_id),
        "camera_name": None if response is None else response.camera_name,
        "tray_count": 0 if response is None else response.tray_count,
        "selected_tray_index": 0 if response is None else response.selected_tray_index,
        "elapsed_ms": None if response is None else response.elapsed_ms,
        "error": fallback_error if response is None else response.error,
        "selected_result": selected,
        "all_tray_count": 0 if response is None else len(response.all_tray_results),
        "has_debug": bool(response is not None and response.debug is not None),
    }


def _parse_cli(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test GraspPose RPC using wuyou LEFT RGBD stream")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-tray-index", type=int, default=0)
    parser.add_argument("--enable-debug", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(args.service_addr),
            camera_name=str(args.camera_name),
            output_dir=Path(args.output_dir),
            target_tray_index=int(args.target_tray_index),
            enable_debug=bool(args.enable_debug),
        )
    )

