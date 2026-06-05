from __future__ import annotations

# region 依赖导入
import argparse
import json
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Literal

import cv2
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.wuji import (  # noqa: E402
    SUPPORTED_WUJI_CAMERAS,
    WujiCameraEnableState,
    WujiCameraFrame,
    WujiCameraIntrinsicsInfo,
    WujiCameraName,
    WujiQmlinkerClient,
    WujiQmlinkerConfig,
    load_wuji_robot_network_config,
)

# endregion


# region 默认参数

DEFAULT_CAMERA_NAME = "all"  # 默认探测全部逻辑相机；可改为 head_camera/chest_camera/left_hand_camera/right_hand_camera
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test" / "wuji" / "artifacts" / "camera_probe"  # 探测结果输出目录
DEFAULT_HOST = load_wuji_robot_network_config().qmlinker.host  # qmlinker 目标主机
DEFAULT_PORT = load_wuji_robot_network_config().qmlinker.port  # qmlinker 目标端口
DEFAULT_REQUEST_TIMEOUT_S = 3.0  # unary RPC 超时，单位 s
DEFAULT_STREAM_TIMEOUT_S = 8.0  # 首帧等待超时，单位 s
DEFAULT_CAPTURE_RGB = True  # 是否采集 RGB 首帧
DEFAULT_CAPTURE_RGBD = True  # 是否采集 RGBD 首帧

# endregion


# region 数据结构


@dataclass(slots=True)
class CameraProbeArtifacts:
    """单路相机探测生成的文件路径集合。"""

    rgb_path: str | None = None
    depth_u16_path: str | None = None
    depth_vis_path: str | None = None


@dataclass(slots=True)
class CameraProbeResult:
    """单路相机探测摘要。"""

    camera_name: str
    enable_api_available: bool
    enable_state: bool
    enable_message: str
    intrinsics_ok: bool
    intrinsics_summary: str
    rgb_ok: bool
    rgb_summary: str
    rgbd_ok: bool
    rgbd_summary: str
    artifacts: CameraProbeArtifacts


# endregion


# region 主入口


def main(
    camera_name: str = DEFAULT_CAMERA_NAME,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    stream_timeout_s: float = DEFAULT_STREAM_TIMEOUT_S,
    capture_rgb: bool = DEFAULT_CAPTURE_RGB,
    capture_rgbd: bool = DEFAULT_CAPTURE_RGBD,
) -> None:
    """探测 qmlinker 相机接口是否可用，并保存首帧样张用于逻辑相机映射核验。

    Parameters
    ----------
    camera_name:
        待探测相机名，支持 `all` 或单个逻辑相机名。
    output_dir:
        输出目录。每路相机会生成独立子目录与 `summary.json`。
    host:
        qmlinker 主机地址。
    port:
        qmlinker 端口号。
    request_timeout_s:
        unary RPC 超时，单位 s。
    stream_timeout_s:
        RGB/RGBD 首帧等待超时，单位 s。
    capture_rgb:
        是否采集 RGB 首帧。
    capture_rgbd:
        是否采集 RGBD 首帧。

    Notes
    -----
    该脚本的目标不是自动判定“头部/胸部/左右手”物理安装位，而是把每路逻辑相机的真实
    首帧落盘，供现场人工对照安装位与画面内容确认映射关系。
    """

    logger.info("硬件测试脚本：需要连通 wuyou 上的 qmlinker 相机服务，未连硬件时会失败。")
    logger.info("探测目标 host {} port {}", host, port)
    logger.info("输出目录 {}", output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    camera_names = _resolve_camera_names(camera_name)
    results: list[CameraProbeResult] = []

    for current_camera in camera_names:
        logger.info("开始探测相机 {}", current_camera)
        result = _probe_one_camera(
            camera_name=current_camera,
            output_dir=output_dir / current_camera,
            host=host,
            port=port,
            request_timeout_s=request_timeout_s,
            stream_timeout_s=stream_timeout_s,
            capture_rgb=capture_rgb,
            capture_rgbd=capture_rgbd,
        )
        results.append(result)
        logger.info(
            "相机 {} 探测完成 enable_api={} intrinsics_ok={} rgb_ok={} rgbd_ok={}",
            result.camera_name,
            result.enable_api_available,
            result.intrinsics_ok,
            result.rgb_ok,
            result.rgbd_ok,
        )

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "host": host,
        "port": int(port),
        "request_timeout_s": float(request_timeout_s),
        "stream_timeout_s": float(stream_timeout_s),
        "mapping_note": "请人工核对每路逻辑相机样张是否与真实安装位一致；当前仓库不能自动证明逻辑名与物理设备一一对应。",
        "results": [
            {
                **asdict(result),
            }
            for result in results
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success("探测摘要已写入 {}", summary_path)


# endregion


# region 探测流程


def _probe_one_camera(
    camera_name: WujiCameraName,
    output_dir: Path,
    host: str,
    port: int,
    request_timeout_s: float,
    stream_timeout_s: float,
    capture_rgb: bool,
    capture_rgbd: bool,
) -> CameraProbeResult:
    """探测单路相机的使能、内参、RGB 与 RGBD 首帧。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = WujiQmlinkerConfig(
        host=host,
        port=int(port),
        request_timeout_s=float(request_timeout_s),
    )

    enable_state = _read_enable_state(config=config, camera_name=camera_name)
    intrinsics_ok = False
    intrinsics_summary = "未读取"
    rgb_ok = False
    rgb_summary = "未采集"
    rgbd_ok = False
    rgbd_summary = "未采集"
    artifacts = CameraProbeArtifacts()

    try:
        with _client_context(config) as client:
            intrinsics = client.get_camera_intrinsics(camera_name)
            intrinsics_ok = True
            intrinsics_summary = _format_intrinsics(intrinsics)
            logger.success("相机 {} 内参读取成功 {}", camera_name, intrinsics_summary)
    except Exception as exc:  # noqa: BLE001
        intrinsics_summary = f"{type(exc).__name__}: {exc}"
        logger.warning("相机 {} 内参读取失败 {}", camera_name, intrinsics_summary)

    if capture_rgb:
        rgb_result = _capture_stream_probe(
            config=config,
            camera_name=camera_name,
            stream_mode="rgb",
            output_dir=output_dir,
            timeout_s=stream_timeout_s,
        )
        rgb_ok = rgb_result[0]
        rgb_summary = rgb_result[1]
        artifacts.rgb_path = rgb_result[2]

    if capture_rgbd:
        rgbd_result = _capture_stream_probe(
            config=config,
            camera_name=camera_name,
            stream_mode="rgbd",
            output_dir=output_dir,
            timeout_s=stream_timeout_s,
        )
        rgbd_ok = rgbd_result[0]
        rgbd_summary = rgbd_result[1]
        artifacts.depth_u16_path = rgbd_result[2]
        artifacts.depth_vis_path = rgbd_result[3]

    result = CameraProbeResult(
        camera_name=camera_name,
        enable_api_available=enable_state.api_available,
        enable_state=enable_state.enabled,
        enable_message=enable_state.message,
        intrinsics_ok=intrinsics_ok,
        intrinsics_summary=intrinsics_summary,
        rgb_ok=rgb_ok,
        rgb_summary=rgb_summary,
        rgbd_ok=rgbd_ok,
        rgbd_summary=rgbd_summary,
        artifacts=artifacts,
    )
    camera_summary_path = output_dir / "summary.json"
    camera_summary_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _read_enable_state(config: WujiQmlinkerConfig, camera_name: WujiCameraName) -> WujiCameraEnableState:
    """读取相机使能状态。"""

    try:
        with _client_context(config) as client:
            state = client.get_camera_enable_state(camera_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("相机 {} 使能状态读取失败 {}", camera_name, exc)
        return WujiCameraEnableState(
            camera_name=camera_name,
            enabled=False,
            api_available=False,
            message=f"{type(exc).__name__}: {exc}",
        )
    logger.info(
        "相机 {} 使能状态 enabled {} api_available {} message {}",
        camera_name,
        state.enabled,
        state.api_available,
        state.message,
    )
    return state


def _capture_stream_probe(
    config: WujiQmlinkerConfig,
    camera_name: WujiCameraName,
    stream_mode: Literal["rgb", "rgbd"],
    output_dir: Path,
    timeout_s: float,
) -> tuple[bool, str, str | None, str | None]:
    """在超时保护下采集一帧图像并落盘。"""

    queue: Queue[tuple[str, Any]] = Queue(maxsize=1)

    def _worker() -> None:
        try:
            with _client_context(config) as client:
                if stream_mode == "rgb":
                    frame = next(client.stream_camera_rgb_frames(camera_name))
                    rgb_path = _save_rgb_frame(output_dir=output_dir, frame=frame)
                    queue.put(("ok", (f"RGB 首帧保存成功: {rgb_path}", str(rgb_path), None)))
                    return

                frame = next(client.stream_camera_rgbd_frames(camera_name))
                rgb_path = _save_rgb_frame(output_dir=output_dir, frame=frame, stem="rgbd_color")
                depth_u16_path, depth_vis_path = _save_depth_frame(output_dir=output_dir, frame=frame)
                queue.put(
                    (
                        "ok",
                        (
                            f"RGBD 首帧保存成功: color={rgb_path}, depth={depth_u16_path}, depth_vis={depth_vis_path}",
                            str(depth_u16_path),
                            str(depth_vis_path),
                        ),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            queue.put(("error", f"{type(exc).__name__}: {exc}"))

    worker = threading.Thread(target=_worker, name=f"camera-probe-{camera_name}-{stream_mode}", daemon=True)
    worker.start()
    worker.join(timeout=max(0.1, float(timeout_s)))

    if worker.is_alive():
        summary = f"{stream_mode} 首帧超时 timeout_s={timeout_s}"
        logger.warning("相机 {} {}", camera_name, summary)
        return False, summary, None, None

    status, payload = queue.get_nowait()
    if status == "error":
        logger.warning("相机 {} {} 失败 {}", camera_name, stream_mode, payload)
        return False, str(payload), None, None

    summary, first_path, second_path = payload
    logger.success("相机 {} {}", camera_name, summary)
    return True, str(summary), first_path, second_path


# endregion


# region 文件输出


def _save_rgb_frame(output_dir: Path, frame: WujiCameraFrame, stem: str = "rgb") -> Path:
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

    depth_vis = _make_depth_vis(depth_u16)
    depth_vis_path = output_dir / "depth_vis.png"
    if not cv2.imwrite(str(depth_vis_path), depth_vis):
        raise RuntimeError(f"depth vis write failed: {depth_vis_path}")
    return depth_u16_path, depth_vis_path


def _make_depth_vis(depth_u16: np.ndarray) -> np.ndarray:
    """将 uint16 深度图转为便于人工核对的伪彩色图。"""

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
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


# endregion


# region 基础工具


class _ClientContext:
    """为脚本封装 qmlinker client 的显式关闭。"""

    def __init__(self, config: WujiQmlinkerConfig) -> None:
        self._config = config
        self._client: WujiQmlinkerClient | None = None

    def __enter__(self) -> WujiQmlinkerClient:
        self._client = WujiQmlinkerClient(self._config)
        self._client.check_ready()
        return self._client

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if self._client is not None:
            self._client.close()
            self._client = None


def _client_context(config: WujiQmlinkerConfig) -> _ClientContext:
    """返回脚本内部使用的 client 上下文。"""

    return _ClientContext(config)


def _resolve_camera_names(camera_name: str) -> list[WujiCameraName]:
    """解析待探测相机列表。"""

    if camera_name == "all":
        return [spec.name for spec in SUPPORTED_WUJI_CAMERAS]
    supported_names = {spec.name for spec in SUPPORTED_WUJI_CAMERAS}
    if camera_name not in supported_names:
        raise ValueError(f"unsupported camera name: {camera_name}")
    return [camera_name]  # type: ignore[list-item]


def _format_intrinsics(intrinsics: WujiCameraIntrinsicsInfo) -> str:
    """格式化相机内参摘要。"""

    return (
        f"fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
        f"cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}, "
        f"size={intrinsics.width}x{intrinsics.height}, dist={list(intrinsics.distortion)}"
    )


def _parse_cli(argv: list[str]) -> tuple[str, Path, str, int, float, float, bool, bool]:
    """解析 CLI 覆盖参数。"""

    parser = argparse.ArgumentParser(description="独立探测 qmlinker 相机使能、内参与 RGB/RGBD 首帧")
    parser.add_argument("--camera", type=str, default=DEFAULT_CAMERA_NAME, help="all 或单个逻辑相机名")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="qmlinker 主机地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="qmlinker 端口")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S, help="unary RPC 超时，单位 s")
    parser.add_argument("--stream-timeout-s", type=float, default=DEFAULT_STREAM_TIMEOUT_S, help="首帧等待超时，单位 s")
    parser.add_argument("--capture-rgb", action=argparse.BooleanOptionalAction, default=DEFAULT_CAPTURE_RGB, help="是否采集 RGB 首帧")
    parser.add_argument("--capture-rgbd", action=argparse.BooleanOptionalAction, default=DEFAULT_CAPTURE_RGBD, help="是否采集 RGBD 首帧")
    args = parser.parse_args(argv)
    return (
        str(args.camera),
        Path(args.output_dir),
        str(args.host),
        int(args.port),
        float(args.request_timeout_s),
        float(args.stream_timeout_s),
        bool(args.capture_rgb),
        bool(args.capture_rgbd),
    )


# endregion


if __name__ == "__main__":
    if len(sys.argv) > 1:
        camera_arg, output_arg, host_arg, port_arg, request_timeout_arg, stream_timeout_arg, rgb_arg, rgbd_arg = _parse_cli(
            sys.argv[1:]
        )
        main(
            camera_name=camera_arg,
            output_dir=output_arg,
            host=host_arg,
            port=port_arg,
            request_timeout_s=request_timeout_arg,
            stream_timeout_s=stream_timeout_arg,
            capture_rgb=rgb_arg,
            capture_rgbd=rgbd_arg,
        )
    else:
        main()
