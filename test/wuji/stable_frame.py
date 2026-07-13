from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_pipeline.client import CameraPipelineClient
from camera_pipeline.protocol import CameraFramePacket
from camera_pipeline.stable_frame import StableFrameConfig, StableFrameDetector

# region 默认常量

DEFAULT_CAMERA_NAME = "left_hand_camera"
# 远端统一 camera_pipeline 服务地址；Orin 本机执行时覆盖为 tcp://127.0.0.1:6200
DEFAULT_SERVICE_ADDR = "tcp://wujibrain-desktop.local:6200"
# RPC 收发超时时间，单位 ms
DEFAULT_TIMEOUT_MS = 60_000
# 测试要求的连续稳定时长，单位 s；生产服务仍使用 StableFrameConfig 的 1 s 默认值
DEFAULT_TEST_STABLE_DURATION_S = 5.0
# 订阅测试的最长等待时间，单位 s
DEFAULT_TEST_TIMEOUT_S = 30.0
# 生产服务稳定帧请求的最长等待时间，单位 s
DEFAULT_SERVICE_STABLE_TIMEOUT_S = 15.0

# endregion


# region 主流程


def main(
    service_addr: str = DEFAULT_SERVICE_ADDR,
    camera_name: str = DEFAULT_CAMERA_NAME,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    test_stable_duration_s: float = DEFAULT_TEST_STABLE_DURATION_S,
    test_timeout_s: float = DEFAULT_TEST_TIMEOUT_S,
    service_stable_timeout_s: float = DEFAULT_SERVICE_STABLE_TIMEOUT_S,
) -> int:
    """验证当前阈值可连续稳定 5 秒，并检查生产服务的 1 秒稳定帧请求。"""

    config = StableFrameConfig(stable_duration_s=test_stable_duration_s)
    detector = StableFrameDetector(config=config)
    metrics: list[dict[str, float | bool]] = []
    client = CameraPipelineClient(service_addr=service_addr, timeout_ms=timeout_ms)
    logger.info(
        "稳定帧测试开始: service_addr={} camera_name={} 连续稳定要求 {} s",
        service_addr,
        camera_name,
        test_stable_duration_s,
    )
    started_at = time.monotonic()
    previous: CameraFramePacket | None = None
    stable_frame_id: int | None = None
    try:
        for frame in client.subscribe_camera_frames(camera_name=camera_name):
            if previous is not None:
                metrics.append(_measure_pair(previous=previous, current=frame, config=config))
            previous = frame
            stable_frame_id = detector.update(frame)
            if stable_frame_id is not None:
                break
            if time.monotonic() - started_at >= test_timeout_s:
                break

        _print_metrics(metrics=metrics, config=config)
        if stable_frame_id is None:
            raise RuntimeError(
                f"画面未在 {test_timeout_s:.1f} s 内形成连续 {test_stable_duration_s:.1f} s 稳定窗口"
            )
        logger.success(
            "本地同算法稳定窗口通过: 连续稳定 {} s midpoint_frame_id={}",
            test_stable_duration_s,
            stable_frame_id,
        )

        service_result = client.get_stable_frame(timeout_s=service_stable_timeout_s)
        logger.success(
            "远端生产稳定帧请求通过: 生产连续稳定要求 {} s frame_id={} timestamp_ms={} ms",
            StableFrameConfig().stable_duration_s,
            service_result.frame_id,
            service_result.timestamp_ms,
        )
    finally:
        client.close()
    return 0


# endregion


# region 指标


def _measure_pair(
    previous: CameraFramePacket,
    current: CameraFramePacket,
    config: StableFrameConfig,
) -> dict[str, float | bool]:
    previous_gray, previous_depth = _extract_features(previous, config)
    current_gray, current_depth = _extract_features(current, config)

    gray_delta = current_gray - previous_gray
    gray_delta -= float(np.median(gray_delta))
    absolute_gray_delta = np.abs(gray_delta)
    color_mean_delta = float(np.mean(absolute_gray_delta))
    color_changed_ratio = float(
        np.mean(absolute_gray_delta > config.color_pixel_delta_threshold)
    )

    valid_depth = (previous_depth > 0.0) & (current_depth > 0.0)
    valid_depth_ratio = float(np.mean(valid_depth))
    if np.any(valid_depth):
        depth_delta = np.abs(current_depth[valid_depth] - previous_depth[valid_depth])
        depth_median_delta_mm = float(np.median(depth_delta))
        depth_p75_delta_mm = float(np.percentile(depth_delta, 75.0))
        depth_p80_delta_mm = float(np.percentile(depth_delta, 80.0))
        depth_p85_delta_mm = float(np.percentile(depth_delta, 85.0))
        depth_percentile_delta_mm = float(
            np.percentile(depth_delta, config.depth_percentile)
        )
    else:
        depth_median_delta_mm = float("inf")
        depth_p75_delta_mm = float("inf")
        depth_p80_delta_mm = float("inf")
        depth_p85_delta_mm = float("inf")
        depth_percentile_delta_mm = float("inf")

    frame_gap_ms = current.timestamp_ms - previous.timestamp_ms
    passed = bool(
        0.0 < frame_gap_ms <= config.max_frame_gap_ms
        and color_mean_delta <= config.color_mean_delta_threshold
        and color_changed_ratio <= config.color_changed_ratio_threshold
        and valid_depth_ratio >= config.min_valid_depth_ratio
        and depth_median_delta_mm <= config.depth_median_delta_threshold_mm
        and depth_percentile_delta_mm
        <= config.depth_percentile_delta_threshold_mm
    )
    return {
        "frame_gap_ms": frame_gap_ms,
        "color_mean_delta": color_mean_delta,
        "color_changed_ratio": color_changed_ratio,
        "valid_depth_ratio": valid_depth_ratio,
        "depth_median_delta_mm": depth_median_delta_mm,
        "depth_p75_delta_mm": depth_p75_delta_mm,
        "depth_p80_delta_mm": depth_p80_delta_mm,
        "depth_p85_delta_mm": depth_p85_delta_mm,
        "depth_percentile_delta_mm": depth_percentile_delta_mm,
        "passed": passed,
    }


def _extract_features(
    frame: CameraFramePacket,
    config: StableFrameConfig,
) -> tuple[np.ndarray, np.ndarray]:
    interpolation = cv2.INTER_AREA if config.image_scale < 1.0 else cv2.INTER_LINEAR
    color_small = cv2.resize(
        frame.color_bgr,
        dsize=None,
        fx=config.image_scale,
        fy=config.image_scale,
        interpolation=interpolation,
    )
    depth_small = cv2.resize(
        frame.depth_mm,
        dsize=(color_small.shape[1], color_small.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    gray = cv2.cvtColor(color_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, depth_small.astype(np.float32)


def _print_metrics(
    metrics: list[dict[str, float | bool]],
    config: StableFrameConfig,
) -> None:
    if not metrics:
        logger.warning("未采集到可比较的相邻帧")
        return
    metric_names = (
        "frame_gap_ms",
        "color_mean_delta",
        "color_changed_ratio",
        "valid_depth_ratio",
        "depth_median_delta_mm",
        "depth_p75_delta_mm",
        "depth_p80_delta_mm",
        "depth_p85_delta_mm",
        "depth_percentile_delta_mm",
    )
    summary: dict[str, object] = {
        "sample_count": len(metrics),
        "passed_count": sum(1 for item in metrics if item["passed"]),
        "config": {
            "stable_duration_s": config.stable_duration_s,
            "image_scale": config.image_scale,
            "max_frame_gap_ms": config.max_frame_gap_ms,
            "color_mean_delta_threshold": config.color_mean_delta_threshold,
            "color_changed_ratio_threshold": config.color_changed_ratio_threshold,
            "color_pixel_delta_threshold": config.color_pixel_delta_threshold,
            "min_valid_depth_ratio": config.min_valid_depth_ratio,
            "depth_median_delta_threshold_mm": config.depth_median_delta_threshold_mm,
            "depth_percentile": config.depth_percentile,
            "depth_percentile_delta_threshold_mm": config.depth_percentile_delta_threshold_mm,
        },
    }
    for name in metric_names:
        values = np.asarray([float(item[name]) for item in metrics], dtype=np.float64)
        summary[name] = {
            "p50": float(np.percentile(values, 50.0)),
            "p95": float(np.percentile(values, 95.0)),
            "p99": float(np.percentile(values, 99.0)),
            "max": float(np.max(values)),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# endregion


# region CLI


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="camera stable frame smoke test")
    parser.add_argument("--service-addr", default=DEFAULT_SERVICE_ADDR)
    parser.add_argument("--camera-name", default=DEFAULT_CAMERA_NAME)
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument(
        "--test-stable-duration-s",
        type=float,
        default=DEFAULT_TEST_STABLE_DURATION_S,
    )
    parser.add_argument("--test-timeout-s", type=float, default=DEFAULT_TEST_TIMEOUT_S)
    parser.add_argument(
        "--service-stable-timeout-s",
        type=float,
        default=DEFAULT_SERVICE_STABLE_TIMEOUT_S,
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    raise SystemExit(
        main(
            service_addr=args.service_addr,
            camera_name=args.camera_name,
            timeout_ms=args.timeout_ms,
            test_stable_duration_s=args.test_stable_duration_s,
            test_timeout_s=args.test_timeout_s,
            service_stable_timeout_s=args.service_stable_timeout_s,
        )
    )


# endregion
