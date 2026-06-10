from __future__ import annotations

# region 依赖导入
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wuji import (  # noqa: E402
    SUPPORTED_WUJI_CAMERAS,
    WujiCameraFrame,
    WujiCameraName,
    WujiZmqCameraClient,
    WujiZmqCameraConfig,
    load_wuji_robot_network_config,
)

# endregion


# region 默认参数
DEFAULT_CAMERA_NAME = "left_hand_camera"  # 默认逻辑相机名
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / "artifacts" / "zmq_rgbd_record"  # 输出目录
DEFAULT_HOST = load_wuji_robot_network_config().base_control_ip  # wuyou ZMQ 相机服务主机
DEFAULT_REQUEST_TIMEOUT_MS = 3000  # 控制命令超时，单位 ms
DEFAULT_STREAM_TIMEOUT_MS = 5000  # 数据流接收超时，单位 ms
DEFAULT_MAX_FRAMES = 30  # 默认保存帧数
DEFAULT_SAVE_DEPTH_VIS = True  # 是否保存深度伪彩图
DEFAULT_WINDOW_NAME = "Wuyou ZMQ RGBD Record"  # 预览窗口名
# endregion


# region 主入口
def main(
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    host: str = DEFAULT_HOST,
    request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    stream_timeout_ms: int = DEFAULT_STREAM_TIMEOUT_MS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    save_depth_vis: bool = DEFAULT_SAVE_DEPTH_VIS,
) -> None:
    """从 wuyou ZMQ 相机信息流持续获取 RGBD 并持久化到本地文件。"""

    logger.info("硬件测试脚本：需要连通 wuyou 上 ZMQ 相机服务，未连硬件时会失败。")
    logger.info("record camera {} host {} output {}", camera_name, host, output_dir)
    logger.warning("当前脚本不直接访问本机 Orbbec，相机画面来自 wuyou ZMQ 信息流。")
    if int(max_frames) <= 0:
        raise ValueError("max_frames must be > 0")

    rgb_dir = Path(output_dir) / "rgb"
    depth_dir = Path(output_dir) / "depth_u16"
    depth_vis_dir = Path(output_dir) / "depth_vis"
    meta_dir = Path(output_dir) / "meta"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    if bool(save_depth_vis):
        depth_vis_dir.mkdir(parents=True, exist_ok=True)

    client = WujiZmqCameraClient(
        WujiZmqCameraConfig(
            host=str(host),
            request_timeout_ms=int(request_timeout_ms),
            stream_timeout_ms=int(stream_timeout_ms),
        )
    )
    metadata_path = Path(output_dir) / "frames.jsonl"
    summary_path = Path(output_dir) / "summary.json"
    metadata_records: list[dict[str, Any]] = []

    try:
        intrinsics = client.get_camera_intrinsics(_to_camera_name(camera_name))
        intrinsics_payload = {
            "camera_name": str(camera_name),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "distortion": [float(v) for v in intrinsics.distortion],
        }
        (meta_dir / "intrinsics.json").write_text(json.dumps(intrinsics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        saved_count = 0
        for frame_index, frame in enumerate(client.stream_camera_rgbd_frames(_to_camera_name(camera_name)), start=1):
            stem = f"{frame_index:06d}"
            color_path = rgb_dir / f"{stem}.jpg"
            depth_path = depth_dir / f"{stem}.png"
            depth_vis_path = depth_vis_dir / f"{stem}.png"

            if not cv2.imwrite(str(color_path), np.asarray(frame.color_bgr, dtype=np.uint8)):
                raise RuntimeError(f"failed to write rgb image: {color_path}")
            depth_u16 = _require_depth_u16(frame)
            if not cv2.imwrite(str(depth_path), depth_u16):
                raise RuntimeError(f"failed to write depth image: {depth_path}")
            preview_depth = _build_depth_vis(depth_u16)
            if bool(save_depth_vis):
                if not cv2.imwrite(str(depth_vis_path), preview_depth):
                    raise RuntimeError(f"failed to write depth vis image: {depth_vis_path}")

            preview = np.hstack([np.asarray(frame.color_bgr, dtype=np.uint8), preview_depth])
            cv2.putText(preview, f"frame {frame_index}/{max_frames}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(DEFAULT_WINDOW_NAME, preview)
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q") or key == ord("Q"):
                logger.warning("用户提前结束录制。")
                break

            record = {
                "frame_index": int(frame_index),
                "camera_name": str(camera_name),
                "timestamp_ms": float(time.time() * 1000.0),
                "color_path": str(color_path.relative_to(output_dir)),
                "depth_u16_path": str(depth_path.relative_to(output_dir)),
                "depth_vis_path": None if not bool(save_depth_vis) else str(depth_vis_path.relative_to(output_dir)),
                "image_width": int(frame.color_bgr.shape[1]),
                "image_height": int(frame.color_bgr.shape[0]),
            }
            metadata_records.append(record)
            saved_count += 1
            if saved_count >= int(max_frames):
                break
    finally:
        client.close()
        _safe_destroy_cv_window(DEFAULT_WINDOW_NAME)

    metadata_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in metadata_records), encoding="utf-8")
    summary_payload = {
        "camera_name": str(camera_name),
        "host": str(host),
        "saved_frame_count": len(metadata_records),
        "request_timeout_ms": int(request_timeout_ms),
        "stream_timeout_ms": int(stream_timeout_ms),
        "max_frames": int(max_frames),
        "save_depth_vis": bool(save_depth_vis),
        "intrinsics_path": "meta/intrinsics.json",
        "metadata_path": "frames.jsonl",
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("RGBD 录制完成 saved_frame_count {} output {}", len(metadata_records), output_dir)


def _parse_cli(argv: list[str]) -> tuple[str, Path, str, int, int, int, bool]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="从 wuyou ZMQ 信息流录制 RGBD 数据")
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA_NAME, help="逻辑相机名")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--request-timeout-ms", type=int, default=DEFAULT_REQUEST_TIMEOUT_MS)
    parser.add_argument("--stream-timeout-ms", type=int, default=DEFAULT_STREAM_TIMEOUT_MS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--save-depth-vis", action=argparse.BooleanOptionalAction, default=DEFAULT_SAVE_DEPTH_VIS)
    args = parser.parse_args(argv)
    return (
        str(args.camera),
        Path(args.output_dir),
        str(args.host),
        int(args.request_timeout_ms),
        int(args.stream_timeout_ms),
        int(args.max_frames),
        bool(args.save_depth_vis),
    )


# endregion


# region 工具函数
def _require_depth_u16(frame: WujiCameraFrame) -> np.ndarray:
    """从 RGBD 帧中提取深度图。"""

    if frame.depth is None:
        raise RuntimeError("rgbd frame depth is None")
    return np.asarray(frame.depth, dtype=np.uint16)


def _build_depth_vis(depth_u16: np.ndarray) -> np.ndarray:
    """把深度图转换为 HSV 着色预览。"""

    depth = np.asarray(depth_u16, dtype=np.float32)
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


def _to_camera_name(camera_name: str) -> WujiCameraName:
    supported_names = {item.name for item in SUPPORTED_WUJI_CAMERAS}
    if camera_name not in supported_names:
        raise ValueError(f"unsupported camera name: {camera_name}")
    return camera_name  # type: ignore[return-value]


def _safe_destroy_cv_window(window_name: str) -> None:
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            cv2.destroyWindow(window_name)
    except cv2.error:
        pass


# endregion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        camera_arg, output_arg, host_arg, request_timeout_arg, stream_timeout_arg, max_frames_arg, save_depth_vis_arg = _parse_cli(sys.argv[1:])
        main(camera_arg, output_arg, host_arg, request_timeout_arg, stream_timeout_arg, max_frames_arg, save_depth_vis_arg)
    else:
        main()
