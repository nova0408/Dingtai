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

from camera_pipeline.ball_pose_detection.protocol import (
    BallDetectionInfo,
    BallPoseDetectionDebugArtifacts,
    BallPoseDetectionRequest,
    BallPoseDetectionResponse,
    BallPosePriorInfo,
)
from camera_pipeline.client import CameraName, CameraPipelineClient

# region 默认常量

DEFAULT_CAMERA_NAME = "left_hand_camera"
# 远端统一 camera_pipeline 服务地址；在 Orin 本机执行时建议覆盖为 tcp://127.0.0.1:6200
DEFAULT_SERVICE_ADDR = "tcp://192.168.1.128:6200"
# RPC 超时时间，单位 ms
DEFAULT_TIMEOUT_MS = 60_000
# 等待稳定帧超时时间，单位 s
DEFAULT_STABLE_TIMEOUT_S = 15.0
# 是否请求 debug 产物
DEFAULT_ENABLE_DEBUG = True
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture"
DEFAULT_PRIORS = (
    BallPosePriorInfo(color_hex="#ffff00", radius_mm=20.0, model_center_mm=(0.0, 0.0, 0.0)),
    BallPosePriorInfo(color_hex="#ff0000", radius_mm=20.0, model_center_mm=(1.0, 0.0, 0.0)),
    BallPosePriorInfo(color_hex="#ff00ff", radius_mm=20.0, model_center_mm=(0.0, 1.0, 0.0)),
)

# endregion


# region 主流程


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    stable_timeout_s: float = DEFAULT_STABLE_TIMEOUT_S,
    enable_debug: bool = DEFAULT_ENABLE_DEBUG,
) -> int:
    """执行一次当前 ball_pose_detection 服务冒烟测试。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ball_pose_detection 冒烟开始：service_addr={} camera_name={}", service_addr, camera_name)
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=int(timeout_ms))
    try:
        selected_camera = CameraName(camera_name)
        status = client.get_camera_status(selected_camera, timeout_s=float(stable_timeout_s))
        logger.info(
            "相机状态：camera_name={} online={} width={} px height={} px",
            status.camera_name,
            status.online,
            status.width,
            status.height,
        )
        target_frame_id = _resolve_target_frame_id(
            client=client,
            camera_name=selected_camera,
            stable_timeout_s=float(stable_timeout_s),
        )
        response = client.detect_ball(
            BallPoseDetectionRequest(
                request_id=1,
                camera_name=selected_camera,
                frame_id=int(target_frame_id),
                enable_debug=bool(enable_debug),
                priors=DEFAULT_PRIORS,
            )
        )
    finally:
        client.close()

    _validate_response(response)
    _save_capture(output_dir=output_dir, response=response)
    _print_summary(response=response, service_addr=service_addr)
    logger.success(
        "ball_pose_detection 冒烟通过：matched_count={} frame_id={}",
        response.matched_count,
        response.frame_id,
    )
    return 0


# endregion


# region 校验与输出


def _resolve_target_frame_id(
    client: CameraPipelineClient,
    camera_name: CameraName,
    stable_timeout_s: float,
) -> int:
    stable_frame = client.get_stable_frame(camera_name, timeout_s=float(stable_timeout_s))
    logger.info(
        "稳定帧获取成功：frame_id={} timestamp_ms={} ms",
        stable_frame.frame_id,
        stable_frame.timestamp_ms,
    )
    return int(stable_frame.frame_id)


def _validate_response(response: BallPoseDetectionResponse) -> None:
    if len(response.detections) != len(DEFAULT_PRIORS):
        raise RuntimeError(
            "ball pose detection 返回数量异常：" f"expected={len(DEFAULT_PRIORS)} actual={len(response.detections)}"
        )
    if response.matched_count <= 0:
        raise RuntimeError("ball pose detection 未返回任何有效三维球心")
    detected_count = sum(1 for item in response.detections if item.detected)
    if detected_count != response.matched_count:
        raise RuntimeError(
            "matched_count 与 detected 数量不一致："
            f"matched_count={response.matched_count} detected_count={detected_count}"
        )


def _save_capture(output_dir: Path, response: BallPoseDetectionResponse) -> None:
    summary = {
        "request_id": response.request_id,
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "timestamp_ms": response.timestamp_ms,
        "elapsed_ms": response.elapsed_ms,
        "matched_count": response.matched_count,
        "detections": [_serialize_detection(item) for item in response.detections],
        "debug_artifacts_count": len(response.debug_artifacts),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not response.debug_artifacts:
        logger.warning("当前请求未返回 debug_artifacts，跳过图像落盘")
        return
    debug_artifact = response.debug_artifacts[0]
    _save_debug_artifacts(output_dir=output_dir, debug_artifact=debug_artifact)


def _save_debug_artifacts(output_dir: Path, debug_artifact: BallPoseDetectionDebugArtifacts) -> None:
    cv2.imwrite(str(output_dir / "color_bgr.jpg"), np.asarray(debug_artifact.color_bgr, dtype=np.uint8))
    cv2.imwrite(str(output_dir / "depth.jpg"), _build_depth_view(np.asarray(debug_artifact.depth_mm)))
    cv2.imwrite(str(output_dir / "overlay.jpg"), np.asarray(debug_artifact.overlay_bgr, dtype=np.uint8))
    cv2.imwrite(
        str(output_dir / "detection_overlay.jpg"),
        np.asarray(debug_artifact.detection_overlay_bgr, dtype=np.uint8),
    )


def _print_summary(response: BallPoseDetectionResponse, service_addr: str) -> None:
    payload = {
        "service_addr": service_addr,
        "frame_id": response.frame_id,
        "camera_name": response.camera_name,
        "timestamp_ms": response.timestamp_ms,
        "elapsed_ms": response.elapsed_ms,
        "matched_count": response.matched_count,
        "detections": [_serialize_detection(item) for item in response.detections],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _serialize_detection(item: BallDetectionInfo) -> dict[str, Any]:
    return {
        "color_hex": item.color_hex,
        "detected": item.detected,
        "center_px": list(item.center_px),
        "center_mm": list(item.center_mm),
        "radius_mm": item.radius_mm,
        "radius_px": item.radius_px,
        "center_norm": list(item.center_norm),
        "radius_norm": item.radius_norm,
        "point_count": item.point_count,
        "status": item.status,
    }


def _build_depth_view(depth_mm: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1.0)
    hsv = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    if np.any(valid):
        z_min = float(np.percentile(depth[valid], 2))
        z_max = float(np.percentile(depth[valid], 98))
        norm = np.clip((depth - z_min) / max(1e-6, z_max - z_min), 0.0, 1.0)
        hsv[..., 0] = np.where(valid, np.rint((1.0 - norm) * 120.0), 0).astype(np.uint8)
        hsv[..., 1] = np.where(valid, 255, 0).astype(np.uint8)
        hsv[..., 2] = np.where(valid, 255, 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# endregion


# region CLI


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ball pose detection smoke test")
    parser.add_argument("--service-addr", type=str, default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", type=str, default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--stable-timeout-s", type=float, default=DEFAULT_STABLE_TIMEOUT_S)
    parser.add_argument("--disable-debug", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=str(cli_args.service_addr),
            camera_name=str(cli_args.camera_name),
            output_dir=Path(cli_args.output_dir),
            timeout_ms=int(cli_args.timeout_ms),
            stable_timeout_s=float(cli_args.stable_timeout_s),
            enable_debug=not bool(cli_args.disable_debug),
        )
    )


# endregion
