from __future__ import annotations

# region 依赖导入
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

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

DEFAULT_CAMERA_NAME = "all"  # 默认探测全部逻辑相机；可改为单个逻辑相机名
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / "artifacts" / "zmq_camera_probe"  # 结果输出目录
DEFAULT_HOST = load_wuji_robot_network_config().base_control_ip  # ZMQ 相机服务主机
DEFAULT_CAPTURE_RGB = True  # 是否采集 RGB 首帧
DEFAULT_CAPTURE_RGBD = True  # 是否采集 RGBD 首帧
DEFAULT_REQUEST_TIMEOUT_MS = 3000  # 控制命令超时，单位 ms
DEFAULT_STREAM_TIMEOUT_MS = 5000  # 首帧等待超时，单位 ms

# endregion


# region 数据结构


@dataclass(slots=True)
class ZmqCameraProbeArtifacts:
    """单路 ZMQ 相机探测生成的文件路径集合。"""

    rgb_path: str | None = None
    rgbd_color_path: str | None = None
    depth_u16_path: str | None = None
    depth_vis_path: str | None = None


@dataclass(slots=True)
class ZmqCameraProbeResult:
    """单路 ZMQ 相机探测摘要。"""

    camera_name: str
    online: bool
    color_enabled: bool
    depth_enabled: bool
    intrinsics_ok: bool
    intrinsics_summary: str
    rgb_ok: bool
    rgb_summary: str
    rgbd_ok: bool
    rgbd_summary: str
    artifacts: ZmqCameraProbeArtifacts


# endregion


# region 主入口


def main(
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    host: str = DEFAULT_HOST,
    capture_rgb: bool = DEFAULT_CAPTURE_RGB,
    capture_rgbd: bool = DEFAULT_CAPTURE_RGBD,
    request_timeout_ms: int = DEFAULT_REQUEST_TIMEOUT_MS,
    stream_timeout_ms: int = DEFAULT_STREAM_TIMEOUT_MS,
) -> None:
    """探测 `wuyou` 上 ZMQ 相机控制口与数据口是否真实可用。

    Parameters
    ----------
    camera_name:
        待探测相机名，支持 `all` 或单个逻辑相机名。
    output_dir:
        结果输出目录。
    host:
        ZMQ 相机服务主机地址。
    capture_rgb:
        是否采集 RGB 首帧。
    capture_rgbd:
        是否采集 RGBD 首帧。
    request_timeout_ms:
        控制命令超时，单位 ms。
    stream_timeout_ms:
        数据流首帧等待超时，单位 ms。

    Notes
    -----
    当前 `wuyou` 现场相机服务不是 qmlinker CameraService，而是 `sensors_depthcamera_ob_zmq_v2`。
    本脚本直接验证这条真实链路。
    """

    logger.info("硬件测试脚本：需要连通 wuyou 上 ZMQ 相机服务，未连硬件时会失败。")
    logger.info("探测目标 host {}", host)
    logger.info("输出目录 {}", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = WujiZmqCameraClient(
        WujiZmqCameraConfig(
            host=host,
            request_timeout_ms=int(request_timeout_ms),
            stream_timeout_ms=int(stream_timeout_ms),
        )
    )
    try:
        results = [
            _probe_one_camera(
                client=client,
                camera_name=current_camera,
                output_dir=output_dir / current_camera,
                capture_rgb=capture_rgb,
                capture_rgbd=capture_rgbd,
            )
            for current_camera in _resolve_camera_names(camera_name)
        ]
    finally:
        client.close()

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "host": host,
        "request_timeout_ms": int(request_timeout_ms),
        "stream_timeout_ms": int(stream_timeout_ms),
        "results": [asdict(item) for item in results],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("探测摘要已写入 {}", summary_path)


# endregion


# region 探测流程


def _probe_one_camera(
    client: WujiZmqCameraClient,
    camera_name: WujiCameraName,
    output_dir: Path,
    capture_rgb: bool,
    capture_rgbd: bool,
) -> ZmqCameraProbeResult:
    """探测单路相机状态、内参与首帧数据。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    status = client.get_camera_status(camera_name)
    logger.info(
        "相机 {} 状态 online {} color_enabled {} depth_enabled {}",
        camera_name,
        status.online,
        status.color_enabled,
        status.depth_enabled,
    )

    intrinsics_ok = False
    intrinsics_summary = "未读取"
    rgb_ok = False
    rgb_summary = "未采集"
    rgbd_ok = False
    rgbd_summary = "未采集"
    artifacts = ZmqCameraProbeArtifacts()

    try:
        intrinsics = client.get_camera_intrinsics(camera_name)
        intrinsics_ok = True
        intrinsics_summary = (
            f"fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
            f"cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}, "
            f"size={intrinsics.width}x{intrinsics.height}, dist={list(intrinsics.distortion)}"
        )
        logger.success("相机 {} 内参读取成功 {}", camera_name, intrinsics_summary)
    except Exception as exc:  # noqa: BLE001
        intrinsics_summary = f"{type(exc).__name__}: {exc}"
        logger.warning("相机 {} 内参读取失败 {}", camera_name, intrinsics_summary)

    if capture_rgb and status.online:
        try:
            frame = next(client.stream_camera_rgb_frames(camera_name))
            artifacts.rgb_path = str(_save_rgb_frame(output_dir, frame, "rgb"))
            rgb_ok = True
            rgb_summary = f"RGB 首帧保存成功: {artifacts.rgb_path}"
            logger.success("相机 {} {}", camera_name, rgb_summary)
        except Exception as exc:  # noqa: BLE001
            rgb_summary = f"{type(exc).__name__}: {exc}"
            logger.warning("相机 {} RGB 首帧失败 {}", camera_name, rgb_summary)

    if capture_rgbd and status.online:
        try:
            frame = next(client.stream_camera_rgbd_frames(camera_name))
            artifacts.rgbd_color_path = str(_save_rgb_frame(output_dir, frame, "rgbd_color"))
            depth_u16_path, depth_vis_path = _save_depth_frame(output_dir, frame)
            artifacts.depth_u16_path = str(depth_u16_path)
            artifacts.depth_vis_path = str(depth_vis_path)
            rgbd_ok = True
            rgbd_summary = (
                f"RGBD 首帧保存成功: color={artifacts.rgbd_color_path}, "
                f"depth={artifacts.depth_u16_path}, depth_vis={artifacts.depth_vis_path}"
            )
            logger.success("相机 {} {}", camera_name, rgbd_summary)
        except Exception as exc:  # noqa: BLE001
            rgbd_summary = f"{type(exc).__name__}: {exc}"
            logger.warning("相机 {} RGBD 首帧失败 {}", camera_name, rgbd_summary)
        finally:
            try:
                client.stop_camera_depth_stream(camera_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("相机 {} 关闭深度流失败 {}", camera_name, exc)

    result = ZmqCameraProbeResult(
        camera_name=camera_name,
        online=status.online,
        color_enabled=status.color_enabled,
        depth_enabled=status.depth_enabled,
        intrinsics_ok=intrinsics_ok,
        intrinsics_summary=intrinsics_summary,
        rgb_ok=rgb_ok,
        rgb_summary=rgb_summary,
        rgbd_ok=rgbd_ok,
        rgbd_summary=rgbd_summary,
        artifacts=artifacts,
    )
    (output_dir / "summary.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# endregion


# region 文件输出


def _save_rgb_frame(output_dir: Path, frame: WujiCameraFrame, stem: str) -> Path:
    """保存单帧 RGB 图像。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / f"{stem}.jpg"
    if not cv2.imwrite(str(rgb_path), frame.color_bgr):
        raise RuntimeError(f"rgb image write failed: {rgb_path}")
    return rgb_path


def _save_depth_frame(output_dir: Path, frame: WujiCameraFrame) -> tuple[Path, Path]:
    """保存单帧深度图与伪彩图。"""

    if frame.depth is None:
        raise RuntimeError("rgbd frame depth is None")
    depth_u16 = np.asarray(frame.depth, dtype=np.uint16)
    depth_u16_path = output_dir / "depth_u16.png"
    if not cv2.imwrite(str(depth_u16_path), depth_u16):
        raise RuntimeError(f"depth image write failed: {depth_u16_path}")

    valid_mask = depth_u16 > 0
    gray = np.zeros(depth_u16.shape, dtype=np.uint8)
    if np.any(valid_mask):
        valid_values = depth_u16[valid_mask].astype(np.float32, copy=False)
        min_value = float(np.min(valid_values))
        max_value = float(np.max(valid_values))
        if max_value > min_value:
            gray[valid_mask] = np.clip(
                (valid_values - min_value) * 255.0 / (max_value - min_value),
                0.0,
                255.0,
            ).astype(np.uint8)
    depth_vis = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    depth_vis_path = output_dir / "depth_vis.png"
    if not cv2.imwrite(str(depth_vis_path), depth_vis):
        raise RuntimeError(f"depth vis write failed: {depth_vis_path}")
    return depth_u16_path, depth_vis_path


# endregion


# region 基础工具


def _resolve_camera_names(camera_name: str) -> list[WujiCameraName]:
    """解析待探测相机列表。"""

    if camera_name == "all":
        return [spec.name for spec in SUPPORTED_WUJI_CAMERAS]
    supported_names = {spec.name for spec in SUPPORTED_WUJI_CAMERAS}
    if camera_name not in supported_names:
        raise ValueError(f"unsupported camera name: {camera_name}")
    return [camera_name]  # type: ignore[list-item]


def _parse_cli(argv: list[str]) -> tuple[str, Path, str, bool, bool, int, int]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="探测 wuyou 上 ZMQ 相机控制口与数据口")
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA_NAME, help="all 或单个逻辑相机名")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="ZMQ 相机服务主机地址")
    parser.add_argument("--capture-rgb", action=argparse.BooleanOptionalAction, default=DEFAULT_CAPTURE_RGB, help="是否采集 RGB 首帧")
    parser.add_argument("--capture-rgbd", action=argparse.BooleanOptionalAction, default=DEFAULT_CAPTURE_RGBD, help="是否采集 RGBD 首帧")
    parser.add_argument("--request-timeout-ms", type=int, default=DEFAULT_REQUEST_TIMEOUT_MS, help="控制命令超时，单位 ms")
    parser.add_argument("--stream-timeout-ms", type=int, default=DEFAULT_STREAM_TIMEOUT_MS, help="首帧等待超时，单位 ms")
    args = parser.parse_args(argv)
    return (
        str(args.camera),
        Path(args.output_dir),
        str(args.host),
        bool(args.capture_rgb),
        bool(args.capture_rgbd),
        int(args.request_timeout_ms),
        int(args.stream_timeout_ms),
    )


# endregion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        camera_arg, output_arg, host_arg, rgb_arg, rgbd_arg, request_timeout_arg, stream_timeout_arg = _parse_cli(
            sys.argv[1:]
        )
        main(
            camera_name=camera_arg,
            output_dir=output_arg,
            host=host_arg,
            capture_rgb=rgb_arg,
            capture_rgbd=rgbd_arg,
            request_timeout_ms=request_timeout_arg,
            stream_timeout_ms=stream_timeout_arg,
        )
    else:
        main()
