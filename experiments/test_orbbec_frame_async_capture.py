from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
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

from src.rgbd_camera import (
    OrbbecSession,
    SessionOptions,
    filter_valid_points,
    normalize_points,
    set_point_cloud_filter_format,
)

# region 默认参数（优先在这里直接改）
DEFAULT_TIMEOUT_MS = 120  # 等待帧超时，单位 ms
DEFAULT_CAPTURE_FPS = 30  # 请求采集帧率，单位 fps
DEFAULT_MAX_DEPTH_MM = 4300.0  # 深度灰度映射上限，单位 mm
DEFAULT_OUTPUT_ROOT = Path(".")  # 输出根目录（会在其下自动创建 yymmdd 目录）
DEFAULT_QUEUE_SIZE = 512  # 异步写盘队列上限，单位 帧
DEFAULT_WINDOW_NAME = "Orbbec image frame sampler"  # 预览窗口名（ASCII）
DEFAULT_WINDOW_WIDTH = 1400  # 预览窗口宽度，单位 像素
DEFAULT_WINDOW_HEIGHT = 900  # 预览窗口高度，单位 像素
# endregion


@dataclass(frozen=True)
class FrameSaveTask:
    frame_index: int
    second_tag: str
    color_bgr: np.ndarray
    depth_gray: np.ndarray
    points: np.ndarray


class AsyncFrameWriter:
    def __init__(self, output_dir: Path, queue_size: int) -> None:
        self.output_dir = Path(output_dir)
        self.queue: queue.Queue[FrameSaveTask | None] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread = threading.Thread(target=self._run, name="orbbec-frame-writer", daemon=True)
        self._started = False
        self._saved_count = 0
        self._failed_count = 0

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    def enqueue(self, task: FrameSaveTask) -> bool:
        if not self._started:
            raise RuntimeError("AsyncFrameWriter 未启动")
        try:
            self.queue.put_nowait(task)
            return True
        except queue.Full:
            return False

    def stop(self, log_interval_s: float = 2.0) -> tuple[int, int]:
        if not self._started:
            return self._saved_count, self._failed_count
        self.queue.put(None)
        last_log = time.monotonic()
        while self._thread.is_alive():
            self._thread.join(timeout=0.2)
            now = time.monotonic()
            if now - last_log >= max(0.2, float(log_interval_s)):
                logger.info(
                    f"后台写盘进行中：待写队列 {self.queue.qsize()} 帧，"
                    f"已写入 {self._saved_count} 帧，失败 {self._failed_count} 帧"
                )
                last_log = now
        self._started = False
        return self._saved_count, self._failed_count

    def _run(self) -> None:
        while True:
            task = self.queue.get()
            if task is None:
                self.queue.task_done()
                break
            try:
                self._save_task(task)
                self._saved_count += 1
            except Exception as exc:
                self._failed_count += 1
                logger.warning(f"后台写盘失败：frame {task.frame_index}，异常 {exc}")
            finally:
                self.queue.task_done()

    def _save_task(self, task: FrameSaveTask) -> None:
        rgb_path = self.output_dir / f"frame_{task.frame_index}_rgb_{task.second_tag}.png"
        depth_path = self.output_dir / f"frame_{task.frame_index}_depth_{task.second_tag}.png"
        pcd_path = self.output_dir / f"frame_{task.frame_index}_pcd_{task.second_tag}.pcd"

        cv2.imwrite(str(rgb_path), task.color_bgr)
        cv2.imwrite(str(depth_path), task.depth_gray)
        _write_pcd_ascii(pcd_path, task.points)


# region 主流程
def main(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    capture_fps: int = DEFAULT_CAPTURE_FPS,
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM,
    queue_size: int = DEFAULT_QUEUE_SIZE,
) -> None:
    if max_depth_mm <= 0:
        raise ValueError("max_depth_mm must be > 0")

    run_dir = Path(output_root) / time.strftime("%y%m%d")
    run_dir.mkdir(parents=True, exist_ok=True)

    options = SessionOptions(
        timeout_ms=int(timeout_ms),
        preferred_capture_fps=max(1, int(capture_fps)),
    )

    logger.info("正在构造 OrbbecSession ...")
    session = OrbbecSession(options=options)
    logger.info("OrbbecSession 构造完成，正在启动流 ...")
    session.start()
    logger.success("OrbbecSession 启动完成")

    writer: AsyncFrameWriter | None = AsyncFrameWriter(output_dir=run_dir, queue_size=queue_size)
    writer.start()
    dropped_frames = 0

    try:
        logger.info(f"实时预览已启动：输出目录 {run_dir}")
        logger.info("按键说明：空格 开始连续逐帧保存；Q 或 ESC 退出。")

        cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        point_filter = session.create_point_cloud_filter(camera_param=session.get_camera_param())
        capture_started = False
        frame_index = 0
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

                if color_bgr is None:
                    color_bgr = _make_fallback_color(depth_u16.shape)

                depth_vis = _depth_to_colormap_bgr(depth_u16=depth_u16, max_depth_mm=max_depth_mm)
                preview = _compose_preview(color_bgr=color_bgr, depth_vis=depth_vis)

                cv2.imshow(DEFAULT_WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q"), ord("Q")):
                    logger.warning("收到退出指令，结束采样")
                    break

                if key == 32 and not capture_started:
                    capture_started = True
                    frame_index = 0
                    logger.success("收到空格指令，开始连续逐帧保存，frame 从 0 计数")

                if capture_started:
                    depth_gray = _depth_to_grayscale(depth_u16=depth_u16, max_depth_mm=max_depth_mm)
                    points = _compute_frame_points(
                        session=session,
                        frames=frames,
                        depth_scale=float(depth_frame.get_depth_scale()),
                        point_filter=point_filter,
                        max_depth_mm=max_depth_mm,
                    )
                    second_tag = time.strftime("%S")

                    task = FrameSaveTask(
                        frame_index=frame_index,
                        second_tag=second_tag,
                        color_bgr=color_bgr.copy(),
                        depth_gray=depth_gray.copy(),
                        points=points.copy(),
                    )
                    ok = writer.enqueue(task)
                    if not ok:
                        dropped_frames += 1
                        logger.warning(f"写盘队列已满：frame {frame_index} 被丢弃，累计丢弃 {dropped_frames} 帧")
                    frame_index += 1

                frames_seen += 1
                if frames_seen % 120 == 0:
                    logger.info(f"预览运行中：已处理帧数 {frames_seen} 帧，当前写盘队列 {writer.queue.qsize()} 帧")

                if cv2.getWindowProperty(DEFAULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.warning("预览窗口关闭，结束采样")
                    break
        finally:
            _safe_destroy_window(DEFAULT_WINDOW_NAME)
    finally:
        try:
            session.stop()
        except Exception as exc:
            logger.warning(f"停止 OrbbecSession 时出现异常：{exc}")
        if writer is not None:
            logger.info("等待异步写盘队列清空...")
            saved_count, failed_count = writer.stop()
            logger.success(
                f"写盘结束：成功写入 {saved_count} 帧，失败 {failed_count} 帧，"
                f"丢弃 {dropped_frames} 帧，目录 {run_dir}"
            )


# endregion


# region 点云与图像工具
def _compute_frame_points(
    session: OrbbecSession,
    frames,
    depth_scale: float,
    point_filter,
    max_depth_mm: float,
) -> np.ndarray:
    point_frames, use_color = session.prepare_frame_for_point_cloud(frames)
    set_point_cloud_filter_format(
        point_filter,
        depth_scale=float(depth_scale),
        use_color=use_color,
    )
    cloud_frame = point_filter.process(point_frames)
    if cloud_frame is None:
        return np.empty((0, 3), dtype=np.float32)

    raw_points = np.asarray(point_filter.calculate(cloud_frame), dtype=np.float32)
    normalized = normalize_points(raw_points)
    valid_points, _ = filter_valid_points(normalized, max_depth_mm=max_depth_mm)
    if len(valid_points) == 0:
        return np.empty((0, normalized.shape[1] if normalized.ndim == 2 else 3), dtype=np.float32)
    return np.ascontiguousarray(valid_points, dtype=np.float32)


def _write_pcd_ascii(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] not in (3, 6):
        points = np.empty((0, 3), dtype=np.float32)

    n = int(points.shape[0])
    has_rgb = points.shape[1] >= 6

    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z r g b" if has_rgb else "FIELDS x y z",
        "SIZE 4 4 4 1 1 1" if has_rgb else "SIZE 4 4 4",
        "TYPE F F F U U U" if has_rgb else "TYPE F F F",
        "COUNT 1 1 1 1 1 1" if has_rgb else "COUNT 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        f"POINTS {n}",
        "DATA ascii",
    ]

    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header_lines))
        f.write("\n")
        if n <= 0:
            return

        xyz = points[:, :3]
        if not has_rgb:
            for i in range(n):
                f.write(f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f}\n")
            return

        rgb = points[:, 3:6]
        if rgb.size > 0 and float(np.max(rgb)) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)

        for i in range(n):
            f.write(
                f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
                f"{int(rgb[i, 0])} {int(rgb[i, 1])} {int(rgb[i, 2])}\n"
            )


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


def _depth_to_grayscale(depth_u16: np.ndarray, max_depth_mm: float) -> np.ndarray:
    depth = depth_u16.astype(np.float32)
    valid = (depth > 0.0) & (depth <= float(max_depth_mm))
    gray = np.zeros_like(depth_u16, dtype=np.uint8)
    if np.any(valid):
        gray[valid] = np.clip((depth[valid] / float(max_depth_mm)) * 255.0, 0, 255).astype(np.uint8)
    return gray


def _depth_to_colormap_bgr(depth_u16: np.ndarray, max_depth_mm: float) -> np.ndarray:
    gray = _depth_to_grayscale(depth_u16=depth_u16, max_depth_mm=max_depth_mm)
    vis = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    vis[gray == 0] = 0
    return vis


def _compose_preview(color_bgr: np.ndarray, depth_vis: np.ndarray) -> np.ndarray:
    h = max(color_bgr.shape[0], depth_vis.shape[0])
    w = max(color_bgr.shape[1], depth_vis.shape[1])
    color_r = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_r = cv2.resize(depth_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    canvas = np.hstack([color_r, depth_r])
    cv2.putText(canvas, "RGB", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Depth", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _make_fallback_color(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "No Color Frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return img


def _safe_destroy_window(window_name: str) -> None:
    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0:
            cv2.destroyWindow(window_name)
    except Exception:
        # 用户手动关窗后 destroyWindow 可能抛出 NULL window，直接忽略。
        return


# endregion


# region CLI（仅用于覆盖默认参数）
def _parse_cli(argv: list[str] | None = None) -> tuple[Path, int, int, float, int]:
    parser = argparse.ArgumentParser(description="Orbbec 空格触发连续逐帧异步保存（pcd/rgb/depth-gray）")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录（其下会创建 yymmdd）")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_TIMEOUT_MS, help="wait_for_frames 超时时间（ms）")
    parser.add_argument("--capture-fps", type=int, default=DEFAULT_CAPTURE_FPS, help="期望采样帧率（fps）")
    parser.add_argument("--max-depth-mm", type=float, default=DEFAULT_MAX_DEPTH_MM, help="深度灰度映射上限（mm）")
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE, help="异步写盘队列上限（帧）")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning(f"检测到未识别参数，已忽略：{unknown}")
    return (
        Path(args.output_root),
        int(args.timeout_ms),
        int(args.capture_fps),
        float(args.max_depth_mm),
        int(args.queue_size),
    )


# endregion


if __name__ == "__main__":
    try:
        out_root_arg, timeout_arg, fps_arg, depth_arg, queue_arg = _parse_cli(sys.argv[1:])
        main(
            output_root=out_root_arg,
            timeout_ms=timeout_arg,
            capture_fps=fps_arg,
            max_depth_mm=depth_arg,
            queue_size=queue_arg,
        )
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
