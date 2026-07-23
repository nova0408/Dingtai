from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "camera_pipeline" / "client.py").is_file()
)
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
from camera_pipeline.service.protocol import CameraStatusResponse

# region 默认常量

DEFAULT_CAMERA_NAME = "left_hand_camera"
# Windows 开发机访问 Orin 管理网 IP；Orin 平铺部署脚本仅访问 localhost。
DEFAULT_SERVICE_ADDR = (
    "tcp://192.168.1.128:6200"
    if sys.platform == "win32"
    else "tcp://127.0.0.1:6200"
)
# RPC 超时时间，单位 ms
DEFAULT_TIMEOUT_MS = 60_000
# 等待稳定帧超时时间，单位 s
DEFAULT_STABLE_TIMEOUT_S = 15.0
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / ".archive" / "ball_pose_detection_capture"
# 首次检测仅输入参考颜色和物理直径，占位中心不包含有效相对位置关系
DEFAULT_REFERENCE_PRIORS = (
    BallPosePriorInfo("#ffff00", 20.0, (0.0, 0.0, 0.0)),
    BallPosePriorInfo("#ff0000", 20.0, (1.0, 0.0, 0.0)),
    BallPosePriorInfo("#ff00ff", 20.0, (0.0, 1.0, 0.0)),
)

# endregion


# region 主流程


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    stable_timeout_s: float = DEFAULT_STABLE_TIMEOUT_S,
) -> int:
    """依次执行连通性、首次三球检测和相对位置先验复检。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ball_pose_detection 冒烟开始：service_addr={} camera_name={}", service_addr, camera_name)
    client = CameraPipelineClient(service_addr=str(service_addr), timeout_ms=int(timeout_ms))
    try:
        selected_camera = CameraName(camera_name)
        logger.info("阶段 1/3：检查服务连通性、相机状态和功能版本")
        expected_service_version = client.expected_service_version
        status = client.get_camera_status(selected_camera, timeout_s=float(stable_timeout_s))
        _validate_connectivity(
            status=status,
            expected_service_version=expected_service_version,
        )
        logger.success(
            "阶段 1/3 通过：service_version={} camera_name={} width={} px height={} px",
            status.service_version,
            status.camera_name,
            status.width,
            status.height,
        )

        logger.info("阶段 2/3：使用参考颜色、物理直径和占位中心执行首次三球检测")
        reference_frame_id = _resolve_target_frame_id(
            client=client,
            camera_name=selected_camera,
            stable_timeout_s=float(stable_timeout_s),
        )
        reference_response = client.detect_ball(
            BallPoseDetectionRequest(
                request_id=2,
                camera_name=selected_camera,
                frame_id=reference_frame_id,
                enable_debug=True,
                priors=DEFAULT_REFERENCE_PRIORS,
            )
        )
        _save_capture(
            output_dir=output_dir / "02_reference_detection",
            response=reference_response,
            stage_name="reference_detection",
        )
        _validate_detection_response(
            response=reference_response,
            target_frame_id=reference_frame_id,
            stage_name="首次三球检测",
        )
        logger.success(
            "阶段 2/3 通过：frame_id={} matched_count={}",
            reference_response.frame_id,
            reference_response.matched_count,
        )

        metric_priors = _build_metric_priors(reference_response)
        logger.info("阶段 3/3：使用首次检测得到的三球相对位置先验执行复检")
        metric_frame_id = _resolve_target_frame_id(
            client=client,
            camera_name=selected_camera,
            stable_timeout_s=float(stable_timeout_s),
        )
        metric_response = client.detect_ball(
            BallPoseDetectionRequest(
                request_id=3,
                camera_name=selected_camera,
                frame_id=metric_frame_id,
                enable_debug=True,
                priors=metric_priors,
            )
        )
        _save_capture(
            output_dir=output_dir / "03_metric_prior_detection",
            response=metric_response,
            stage_name="metric_prior_detection",
        )
        _validate_detection_response(
            response=metric_response,
            target_frame_id=metric_frame_id,
            stage_name="相对位置先验复检",
        )
    finally:
        client.close()

    _print_summary(
        service_addr=service_addr,
        expected_service_version=expected_service_version,
        status=status,
        reference_response=reference_response,
        metric_response=metric_response,
    )
    logger.success(
        "阶段 3/3 通过，ball_pose_detection 三阶段冒烟完成："
        "reference_frame_id={} metric_frame_id={}",
        reference_response.frame_id,
        metric_response.frame_id,
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


def _validate_connectivity(
    status: CameraStatusResponse,
    expected_service_version: str,
) -> None:
    """验证相机在线，并确认远端功能版本与本地客户端一致。"""

    if not status.online:
        raise RuntimeError(f"远端相机不在线：camera_name={status.camera_name}")
    if status.service_version != expected_service_version:
        raise RuntimeError(
            "CameraPipeline 功能版本不一致："
            f"client={expected_service_version} service={status.service_version}"
        )


def _validate_detection_response(
    response: BallPoseDetectionResponse,
    target_frame_id: int,
    stage_name: str,
) -> None:
    if response.frame_id != target_frame_id:
        raise RuntimeError(
            f"{stage_name}返回帧与请求稳定帧不一致："
            f"expected={target_frame_id} actual={response.frame_id}"
        )
    if len(response.detections) != len(DEFAULT_REFERENCE_PRIORS):
        raise RuntimeError(
            f"{stage_name}返回数量异常："
            f"expected={len(DEFAULT_REFERENCE_PRIORS)} "
            f"actual={len(response.detections)}"
        )
    if response.matched_count != len(DEFAULT_REFERENCE_PRIORS):
        raise RuntimeError(
            f"{stage_name}未完整检出三球：matched_count={response.matched_count}"
        )
    expected_colors = {prior.color_hex for prior in DEFAULT_REFERENCE_PRIORS}
    actual_colors = {item.color_hex for item in response.detections}
    if actual_colors != expected_colors:
        raise RuntimeError(
            f"{stage_name}颜色集合异常：expected={expected_colors} actual={actual_colors}"
        )
    for detection in response.detections:
        if (
            not detection.detected
            or len(detection.center_mm) != 3
            or detection.diameter_mm <= 0.0
            or len(detection.observed_hsv) != 3
        ):
            raise RuntimeError(
                f"{stage_name}包含无效检测："
                f"color={detection.color_hex} status={detection.status}"
            )
    if len(response.debug_artifacts) != 1:
        raise RuntimeError(
            f"{stage_name}必须返回一份 debug 产物："
            f"actual={len(response.debug_artifacts)}"
        )


def _build_metric_priors(
    reference_response: BallPoseDetectionResponse,
) -> tuple[BallPosePriorInfo, ...]:
    """使用首次检测球心构造带实际毫米尺度相对位置的三球先验。"""

    reference_diameters = {
        prior.color_hex: prior.diameter_mm for prior in DEFAULT_REFERENCE_PRIORS
    }
    return tuple(
        BallPosePriorInfo(
            color_hex=item.color_hex,
            diameter_mm=reference_diameters[item.color_hex],
            model_center_mm=(
                float(item.center_mm[0]),
                float(item.center_mm[1]),
                float(item.center_mm[2]),
            ),
        )
        for item in reference_response.detections
    )


def _save_capture(
    output_dir: Path,
    response: BallPoseDetectionResponse,
    stage_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_artifact = response.debug_artifacts[0]
    _save_debug_artifacts(output_dir=output_dir, debug_artifact=debug_artifact)
    summary = {
        "stage": stage_name,
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
    logger.success("冒烟 debug 图像已保存：{}", output_dir)


def _save_debug_artifacts(
    output_dir: Path,
    debug_artifact: BallPoseDetectionDebugArtifacts,
) -> None:
    _save_required_image(
        output_dir / "last_frame_bgr.jpg",
        np.asarray(debug_artifact.color_bgr, dtype=np.uint8),
    )
    _save_required_image(
        output_dir / "last_frame_depth.jpg",
        _build_depth_view(np.asarray(debug_artifact.depth_mm)),
    )
    _save_required_image(
        output_dir / "overlay.jpg",
        np.asarray(debug_artifact.overlay_bgr, dtype=np.uint8),
    )
    _save_required_image(
        output_dir / "detection_overlay.jpg",
        np.asarray(debug_artifact.detection_overlay_bgr, dtype=np.uint8),
    )


def _save_required_image(path: Path, image: np.ndarray) -> None:
    """保存冒烟核验图，写入失败时直接判定测试失败。"""

    if image.size == 0 or not cv2.imwrite(str(path), image):
        raise RuntimeError(f"冒烟核验图片保存失败：{path}")


def _print_summary(
    service_addr: str,
    expected_service_version: str,
    status: CameraStatusResponse,
    reference_response: BallPoseDetectionResponse,
    metric_response: BallPoseDetectionResponse,
) -> None:
    payload = {
        "service_addr": service_addr,
        "client_service_version": expected_service_version,
        "remote_service_version": status.service_version,
        "camera_name": status.camera_name,
        "reference_detection": _serialize_response(reference_response),
        "metric_prior_detection": _serialize_response(metric_response),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _serialize_response(response: BallPoseDetectionResponse) -> dict[str, Any]:
    return {
        "frame_id": response.frame_id,
        "timestamp_ms": response.timestamp_ms,
        "elapsed_ms": response.elapsed_ms,
        "matched_count": response.matched_count,
        "detections": [_serialize_detection(item) for item in response.detections],
    }


def _serialize_detection(item: BallDetectionInfo) -> dict[str, Any]:
    return {
        "color_hex": item.color_hex,
        "detected": item.detected,
        "center_px": list(item.center_px),
        "center_mm": list(item.center_mm),
        "diameter_mm": item.diameter_mm,
        "radius_px": item.radius_px,
        "center_norm": list(item.center_norm),
        "radius_norm": item.radius_norm,
        "point_count": item.point_count,
        "status": item.status,
        "observed_hsv": list(item.observed_hsv),
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
        )
    )


# endregion
