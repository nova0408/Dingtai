from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("需要安装 opencv-python 才能运行图像采样脚本。") from exc

from pyorbbecsdk import OBFormat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import Gemini305, SessionOptions


# region 默认参数（优先在这里直接改）
DEFAULT_TIMEOUT_MS = 120  # 等待帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 4300.0  # 深度可视化上限，单位 mm
DEFAULT_OUTPUT_DIR = Path("experiments/sampled_images")  # 采样输出目录
DEFAULT_WINDOW_NAME = "Orbbec 图像采样（空格保存）"  # 预览窗口名
DEFAULT_WINDOW_WIDTH = 1400  # 预览窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 预览窗口高度，单位 像素
DEFAULT_SAVE_DEPTH_VIS = True  # 是否额外保存深度伪彩图
# endregion


# region 主流程
def main(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
    save_depth_vis: bool = DEFAULT_SAVE_DEPTH_VIS,
) -> None:
    if max_depth_mm <= 0:
        raise ValueError("max_depth_mm must be > 0")

    output_dir = Path(output_dir)
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth_u16"
    depth_vis_dir = output_dir / "depth_vis"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    if save_depth_vis:
        depth_vis_dir.mkdir(parents=True, exist_ok=True)

    options = SessionOptions(
        timeout_ms=int(timeout_ms),
        preferred_capture_fps=max(1, int(capture_fps)),
    )

    with Gemini305(options=options) as session:
        logger.info(
            f"图像采样开始：输出目录 {output_dir}，请求帧率 {capture_fps} fps，深度可视化上限 {max_depth_mm:.1f} mm"
        )
        logger.info("按键说明：空格 保存当前帧；Q 或 ESC 退出。")

        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        sample_index = _resolve_next_index(color_dir=color_dir)
        frames_seen = 0

        try:
            while True:
                frames = session.wait_for_frames()
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if depth_frame is None:
                    continue

                color_bgr = _decode_color_frame_bgr(color_frame)
                depth_u16 = _decode_depth_u16(depth_frame)
                if depth_u16 is None:
                    continue

                depth_vis = _depth_to_colormap_bgr(depth_u16=depth_u16, max_depth_mm=max_depth_mm)
                preview = _compose_preview(color_bgr=color_bgr, depth_vis=depth_vis)

                cv2.imshow(DEFAULT_WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF

                frames_seen += 1
                if frames_seen % 120 == 0:
                    logger.info(f"预览运行中：已处理帧数 {frames_seen} 帧")

                if key in (27, ord("q"), ord("Q")):
                    logger.warning("收到退出指令，结束采样")
                    break

                if key == 32:
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    stem = f"sample_{sample_index:04d}_{stamp}"

                    color_path = color_dir / f"{stem}.png"
                    depth_path = depth_dir / f"{stem}.png"
                    cv2.imwrite(str(depth_path), depth_u16)

                    if color_bgr is not None:
                        cv2.imwrite(str(color_path), color_bgr)
                    else:
                        color_path = Path("<无彩色帧>")

                    vis_path = None
                    if save_depth_vis:
                        vis_path = depth_vis_dir / f"{stem}.png"
                        cv2.imwrite(str(vis_path), depth_vis)

                    logger.success(
                        f"采样完成：index {sample_index}，color {color_path}，depth_u16 {depth_path}"
                        + (f"，depth_vis {vis_path}" if vis_path is not None else "")
                    )
                    sample_index += 1

                if cv2.getWindowProperty(DEFAULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.warning("预览窗口关闭，结束采样")
                    break
        finally:
            cv2.destroyWindow(DEFAULT_WINDOW_NAME)


# endregion


# region 解码与预览工具
def _decode_color_frame_bgr(color_frame) -> np.ndarray | None:
    if color_frame is None:
        return None

    width = int(color_frame.get_width())
    height = int(color_frame.get_height())
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if width <= 0 or height <= 0 or data.size == 0:
        return None

    if color_format == OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3)).copy()
    if color_format in (OBFormat.YUYV, OBFormat.YUY2):
        yuy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuy, cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    if color_format == OBFormat.I420:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    logger.warning(f"当前 color format 暂未支持直接预览：{color_format}")
    return None


def _decode_depth_u16(depth_frame) -> np.ndarray | None:
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    if data.size != width * height:
        return None
    return data.reshape(height, width)


def _depth_to_colormap_bgr(depth_u16: np.ndarray, max_depth_mm: float) -> np.ndarray:
    depth = depth_u16.astype(np.float32)
    valid = (depth > 0.0) & (depth <= float(max_depth_mm))

    normalized = np.zeros_like(depth, dtype=np.uint8)
    if np.any(valid):
        normalized[valid] = np.clip((depth[valid] / float(max_depth_mm)) * 255.0, 0, 255).astype(np.uint8)

    vis = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    vis[~valid] = 0
    return vis


def _compose_preview(color_bgr: np.ndarray | None, depth_vis: np.ndarray) -> np.ndarray:
    if color_bgr is None:
        color_bgr = np.zeros_like(depth_vis)
        cv2.putText(
            color_bgr,
            "No Color Frame",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    h = max(color_bgr.shape[0], depth_vis.shape[0])
    w = max(color_bgr.shape[1], depth_vis.shape[1])
    color_r = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    canvas = np.hstack([color_r, depth_r])
    cv2.putText(canvas, "RGB", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Depth", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _resolve_next_index(color_dir: Path) -> int:
    exist = sorted(color_dir.glob("sample_*.png"))
    if not exist:
        return 1

    last = exist[-1].stem
    parts = last.split("_")
    if len(parts) < 2:
        return len(exist) + 1
    try:
        return int(parts[1]) + 1
    except Exception:
        return len(exist) + 1


# endregion


# region CLI（仅用于覆盖默认参数）
def _parse_cli() -> tuple[Path, int, int, float, bool]:
    parser = argparse.ArgumentParser(description="Orbbec 图像采样：实时预览并按空格保存 RGB/Depth")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="采样输出目录")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames 超时时间（ms）")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="期望采样帧率（fps）")
    parser.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM, help="深度伪彩显示上限（mm）")
    parser.add_argument(
        "--save-depth-vis",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_DEPTH_VIS,
        help="是否保存深度伪彩图",
    )
    args = parser.parse_args()
    return Path(args.output_dir), int(args.timeout_ms), int(args.capture_fps), float(args.max_depth_mm), bool(args.save_depth_vis)


# endregion


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            out_arg, timeout_arg, fps_arg, depth_arg, save_vis_arg = _parse_cli()
            main(
                output_dir=out_arg,
                timeout_ms=timeout_arg,
                capture_fps=fps_arg,
                max_depth_mm=depth_arg,
                save_depth_vis=save_vis_arg,
            )
        else:
            main()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise

