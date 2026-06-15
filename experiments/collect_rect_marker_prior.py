from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "需要安装 opencv-python 才能运行矩形标记件先验采集脚本。"
    ) from exc

from pyorbbecsdk import OBFormat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rgbd_camera import Gemini305, SessionOptions

# region 默认参数（优先在这里直接改）
DEFAULT_TIMEOUT_MS = 120
DEFAULT_CAPTURE_FPS = 30
DEFAULT_OUTPUT_DIR = Path("experiments/rect_marker_prior_sessions")
DEFAULT_WINDOW_NAME = "矩形标记件先验采集"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 820
DEFAULT_SAMPLE_SHRINK_RATIO = 0.72
DEFAULT_MAX_DEPTH_MM = 4300.0
# endregion


# region 数据结构
@dataclass(frozen=True)
class CameraCalibrationJson:
    """先验采集时保存的彩色相机标定参数。"""

    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float]
    distortion_model: str
    coordinate_space: str


@dataclass(frozen=True)
class RectMarkerPriorJson:
    """单个矩形标记件先验数据。"""

    marker_id: str
    corners_px: list[list[float]]
    corner_order: str
    rgb_prior: list[int]
    rgb_median: list[int]
    rgb_mad: list[float]
    expected_area_px: float
    expected_angle_deg: float
    expected_aspect_ratio: float
    max_center_shift_px: float
    max_angle_delta_deg: float
    min_area_ratio: float
    max_area_ratio: float
    roi_expand_px: int
    roi_expand_ratio: float
    color_sample_shrink_ratio: float


@dataclass(frozen=True)
class RectMarkerSetPriorJson:
    """矩形标记件先验集合。"""

    schema_version: str
    created_at: str
    source_image: str
    camera: CameraCalibrationJson
    markers: list[RectMarkerPriorJson]


@dataclass
class MarkerDraft:
    """交互采集中的单个标记件草稿。"""

    marker_id: str
    points: list[tuple[float, float]]
    completed: RectMarkerPriorJson | None = None


# endregion


# region 交互采集应用
class RectMarkerPriorCollector:
    """矩形标记件先验交互采集器。"""

    def __init__(
        self,
        output_dir: Path,
        timeout_ms: int,
        capture_fps: int,
        sample_shrink_ratio: float,
        max_depth_mm: float,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._timeout_ms = int(timeout_ms)
        self._capture_fps = max(1, int(capture_fps))
        self._sample_shrink_ratio = float(sample_shrink_ratio)
        self._max_depth_mm = float(max_depth_mm)

        self._frozen = False
        self._current_bgr: np.ndarray | None = None
        self._current_depth_u16: np.ndarray | None = None
        self._frozen_bgr: np.ndarray | None = None
        self._frozen_depth_u16: np.ndarray | None = None
        self._camera: CameraCalibrationJson | None = None
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None

        self._markers: list[RectMarkerPriorJson] = []
        self._draft = MarkerDraft(marker_id="marker_1", points=[])

    def run(self) -> None:
        """启动相机预览并进入鼠标采集循环。"""

        if not 0.1 <= self._sample_shrink_ratio <= 0.95:
            raise ValueError("sample_shrink_ratio must be in [0.1, 0.95]")
        if self._max_depth_mm <= 0:
            raise ValueError("max_depth_mm must be > 0")

        options = SessionOptions(
            timeout=int(self._timeout_ms),
            preferred_capture_fps=int(self._capture_fps),
        )

        with Gemini305(options=options) as session:
            camera_param = session.get_camera_param()
            self._set_camera_param(camera_param)

            cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                DEFAULT_WINDOW_NAME, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
            )
            cv2.setMouseCallback(DEFAULT_WINDOW_NAME, self._handle_mouse)

            logger.info(
                "按键：空格 冻结/恢复；鼠标左键 点四角；N 新标记；Backspace 撤销点；S 保存；Q/ESC 退出。"
            )
            try:
                while True:
                    if not self._frozen:
                        frames = session.wait_for_frames()
                        if frames is not None:
                            self._update_live_frame(frames)

                    canvas = self._build_canvas()
                    cv2.imshow(DEFAULT_WINDOW_NAME, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if self._handle_key(key):
                        break

                    if (
                        cv2.getWindowProperty(DEFAULT_WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                        < 1
                    ):
                        logger.warning("预览窗口关闭，结束采集。")
                        break
            finally:
                cv2.destroyWindow(DEFAULT_WINDOW_NAME)

    def _set_camera_param(self, camera_param) -> None:
        color_intrinsic = camera_param.rgb_intrinsic
        color_distortion = camera_param.rgb_distortion

        self._camera_matrix = np.array(
            [
                [float(color_intrinsic.fx), 0.0, float(color_intrinsic.cx)],
                [0.0, float(color_intrinsic.fy), float(color_intrinsic.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = [
            float(color_distortion.k1),
            float(color_distortion.k2),
            float(color_distortion.p1),
            float(color_distortion.p2),
            float(color_distortion.k3),
            float(color_distortion.k4),
            float(color_distortion.k5),
            float(color_distortion.k6),
        ]
        self._dist_coeffs = np.asarray(distortion, dtype=np.float64)
        self._camera = CameraCalibrationJson(
            image_width=int(color_intrinsic.width),
            image_height=int(color_intrinsic.height),
            fx=float(color_intrinsic.fx),
            fy=float(color_intrinsic.fy),
            cx=float(color_intrinsic.cx),
            cy=float(color_intrinsic.cy),
            distortion=distortion,
            distortion_model="opencv_rational",
            coordinate_space="undistorted_pixel",
        )

    def _update_live_frame(self, frames) -> None:
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_bgr = _decode_color_frame_bgr(color_frame)
        if color_bgr is None:
            return

        self._current_bgr = self._undistort_color(color_bgr)
        self._current_depth_u16 = (
            _decode_depth_u16(depth_frame) if depth_frame is not None else None
        )

    def _undistort_color(self, color_bgr: np.ndarray) -> np.ndarray:
        if self._camera_matrix is None or self._dist_coeffs is None:
            return color_bgr
        return cv2.undistort(color_bgr, self._camera_matrix, self._dist_coeffs)

    def _handle_key(self, key: int) -> bool:
        if key in (255,):
            return False
        if key in (27, ord("q"), ord("Q")):
            logger.warning("收到退出指令，结束采集。")
            return True
        if key == 32:
            self._toggle_freeze()
            return False
        if key in (ord("n"), ord("N")):
            self._start_next_marker()
            return False
        if key in (8, 127):
            self._undo_last_point()
            return False
        if key in (ord("s"), ord("S")):
            self._save_session()
            return False
        return False

    def _toggle_freeze(self) -> None:
        if self._frozen:
            self._frozen = False
            self._frozen_bgr = None
            self._frozen_depth_u16 = None
            self._draft.points.clear()
            logger.info("恢复实时预览，当前未完成点位已清空。")
            return

        if self._current_bgr is None:
            logger.warning("还没有有效彩色帧，无法冻结。")
            return
        self._frozen = True
        self._frozen_bgr = self._current_bgr.copy()
        self._frozen_depth_u16 = (
            None if self._current_depth_u16 is None else self._current_depth_u16.copy()
        )
        logger.info(f"已冻结当前帧，开始采集 {self._draft.marker_id} 的四个角点。")

    def _start_next_marker(self) -> None:
        next_index = len(self._markers) + 1
        if self._draft.points:
            logger.warning("当前标记件存在未完成角点，已清空草稿。")
        self._draft = MarkerDraft(marker_id=f"marker_{next_index + 1}", points=[])
        logger.info(f"切换到 {self._draft.marker_id}。")

    def _undo_last_point(self) -> None:
        if not self._draft.points:
            logger.info("当前没有可撤销角点。")
            return
        removed = self._draft.points.pop()
        logger.info(f"撤销角点：({removed[0]:.1f}, {removed[1]:.1f})。")

    def _handle_mouse(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self._frozen or self._frozen_bgr is None:
            logger.warning("请先按空格冻结图像，再点击角点。")
            return
        if len(self._draft.points) >= 4:
            logger.warning("当前标记件已完成四角点；按 N 切换到下一个标记件。")
            return

        self._draft.points.append((float(x), float(y)))
        logger.info(
            f"{self._draft.marker_id} 角点 {len(self._draft.points)}/4：({x}, {y})"
        )
        if len(self._draft.points) == 4:
            self._complete_current_marker()

    def _complete_current_marker(self) -> None:
        if self._frozen_bgr is None:
            return
        ordered = _order_quad_points(self._draft.points)
        marker = _build_marker_prior(
            marker_id=self._draft.marker_id,
            corners=ordered,
            image_bgr=self._frozen_bgr,
            sample_shrink_ratio=self._sample_shrink_ratio,
        )
        self._markers.append(marker)
        logger.success(
            f"{marker.marker_id} 采集完成：中心 {np.mean(np.asarray(marker.corners_px), axis=0).round(1).tolist()}，"
            f"RGB {marker.rgb_median}"
        )
        self._draft = MarkerDraft(
            marker_id=f"marker_{len(self._markers) + 1}", points=[]
        )

    def _build_canvas(self) -> np.ndarray:
        image = (
            self._frozen_bgr
            if self._frozen and self._frozen_bgr is not None
            else self._current_bgr
        )
        if image is None:
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                "Waiting for Orbbec color frame...",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            return canvas

        canvas = image.copy()
        for marker in self._markers:
            _draw_marker(canvas, marker, completed=True)
        if self._draft.points:
            _draw_draft(canvas, self._draft.points)

        status = "FROZEN" if self._frozen else "LIVE"
        cv2.putText(
            canvas,
            status,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"markers={len(self._markers)} current={self._draft.marker_id} points={len(self._draft.points)}/4",
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "SPACE freeze/live | LMB corners | N next | Backspace undo | S save | Q exit",
            (20, canvas.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _save_session(self) -> None:
        if self._camera is None:
            logger.warning("相机内参尚未初始化，无法保存。")
            return
        if self._frozen_bgr is None:
            logger.warning("请先冻结一帧图像后再保存。")
            return
        if not self._markers:
            logger.warning("至少需要完成一个矩形标记件后才能保存。")
            return

        stamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = self._output_dir / stamp
        session_dir.mkdir(parents=True, exist_ok=True)

        color_path = session_dir / "color_undistorted.png"
        annotated_path = session_dir / "annotated.png"
        prior_path = session_dir / "prior.json"
        depth_path = session_dir / "depth_u16.png"
        depth_vis_path = session_dir / "depth_vis.png"

        annotated = self._build_canvas()
        cv2.imwrite(str(color_path), self._frozen_bgr)
        cv2.imwrite(str(annotated_path), annotated)

        if self._frozen_depth_u16 is not None:
            cv2.imwrite(str(depth_path), self._frozen_depth_u16)
            cv2.imwrite(
                str(depth_vis_path),
                _depth_to_colormap_bgr(self._frozen_depth_u16, self._max_depth_mm),
            )

        prior = RectMarkerSetPriorJson(
            schema_version="rect_marker_prior.v1",
            created_at=stamp,
            source_image=color_path.name,
            camera=self._camera,
            markers=list(self._markers),
        )
        prior_path.write_text(
            json.dumps(asdict(prior), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.success(f"先验采集结果已保存：{session_dir}")


# endregion


# region 标记件几何与颜色统计
def _order_quad_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    sums = ordered[:, 0] + ordered[:, 1]
    start = int(np.argmin(sums))
    ordered = np.roll(ordered, -start, axis=0)

    # 屏幕坐标 y 向下，正面积表示顺时针；这里统一输出 top_left -> top_right -> bottom_right -> bottom_left。
    if _polygon_area(ordered) < 0:
        ordered = np.array(
            [ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float64
        )
    return [(float(x), float(y)) for x, y in ordered]


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _build_marker_prior(
    marker_id: str,
    corners: list[tuple[float, float]],
    image_bgr: np.ndarray,
    sample_shrink_ratio: float,
) -> RectMarkerPriorJson:
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    rgb_pixels = _sample_marker_rgb(
        image_bgr=image_bgr, corners=pts, shrink_ratio=sample_shrink_ratio
    )
    rgb_median = np.median(rgb_pixels, axis=0)
    rgb_mad = np.median(np.abs(rgb_pixels.astype(np.float64) - rgb_median), axis=0)

    edge_lengths = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    width = float(0.5 * (edge_lengths[0] + edge_lengths[2]))
    height = float(0.5 * (edge_lengths[1] + edge_lengths[3]))
    angle = float(np.degrees(np.arctan2(pts[1, 1] - pts[0, 1], pts[1, 0] - pts[0, 0])))
    area = abs(_polygon_area(pts))
    aspect = width / max(1e-6, height)

    return RectMarkerPriorJson(
        marker_id=str(marker_id),
        corners_px=[[float(x), float(y)] for x, y in corners],
        corner_order="top_left,top_right,bottom_right,bottom_left",
        rgb_prior=[int(round(v)) for v in rgb_median],
        rgb_median=[int(round(v)) for v in rgb_median],
        rgb_mad=[float(round(v, 3)) for v in rgb_mad],
        expected_area_px=float(round(area, 3)),
        expected_angle_deg=float(round(angle, 3)),
        expected_aspect_ratio=float(round(aspect, 6)),
        max_center_shift_px=float(round(max(12.0, min(width, height) * 0.25), 3)),
        max_angle_delta_deg=12.0,
        min_area_ratio=0.55,
        max_area_ratio=1.65,
        roi_expand_px=int(round(max(20.0, min(width, height) * 0.25))),
        roi_expand_ratio=0.25,
        color_sample_shrink_ratio=float(sample_shrink_ratio),
    )


def _sample_marker_rgb(
    image_bgr: np.ndarray, corners: np.ndarray, shrink_ratio: float
) -> np.ndarray:
    center = np.mean(corners, axis=0)
    sample_corners = center + (corners - center) * float(shrink_ratio)
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(sample_corners).astype(np.int32), 255)
    pixels_bgr = image_bgr[mask > 0]
    if pixels_bgr.size == 0:
        raise ValueError("颜色采样区域为空，请重新采集角点。")
    return pixels_bgr[:, ::-1].astype(np.uint8)


# endregion


# region 绘制工具
def _draw_marker(
    canvas: np.ndarray, marker: RectMarkerPriorJson, completed: bool
) -> None:
    pts = (
        np.round(np.asarray(marker.corners_px, dtype=np.float64))
        .astype(np.int32)
        .reshape(-1, 1, 2)
    )
    color = (0, 255, 0) if completed else (0, 255, 255)
    cv2.polylines(
        canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA
    )
    center = np.mean(pts.reshape(-1, 2), axis=0).astype(int)
    cv2.circle(canvas, tuple(center), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        marker.marker_id,
        tuple(center + np.array([8, -8])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_draft(canvas: np.ndarray, points: list[tuple[float, float]]) -> None:
    pts = np.round(np.asarray(points, dtype=np.float64)).astype(np.int32)
    for idx, pt in enumerate(pts):
        cv2.circle(canvas, tuple(pt), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(idx + 1),
            tuple(pt + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
    if len(pts) >= 2:
        cv2.polylines(
            canvas,
            [pts.reshape(-1, 1, 2)],
            isClosed=False,
            color=(0, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )


# endregion


# region Orbbec 帧解码
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
    if depth_frame is None:
        return None
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
        normalized[valid] = np.clip(
            (depth[valid] / float(max_depth_mm)) * 255.0, 0, 255
        ).astype(np.uint8)
    vis = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    vis[~valid] = 0
    return vis


# endregion


# region CLI
def _parse_cli() -> tuple[Path, int, int, float, float]:
    parser = argparse.ArgumentParser(
        description="矩形标记件先验采集：冻结图像、点击四角并保存 prior.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="先验采集输出目录"
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="wait_for_frames 超时时间（ms）",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=DEFAULT_CAPTURE_FPS,
        help="期望采样帧率（fps）",
    )
    parser.add_argument(
        "--sample-shrink-ratio",
        type=float,
        default=DEFAULT_SAMPLE_SHRINK_RATIO,
        help="颜色采样多边形向中心收缩比例",
    )
    parser.add_argument(
        "--max-depth-mm",
        type=float,
        default=DEFAULT_MAX_DEPTH_MM,
        help="深度伪彩显示上限（mm）",
    )
    args = parser.parse_args()
    return (
        Path(args.output_dir),
        int(args.timeout_ms),
        int(args.capture_fps),
        float(args.sample_shrink_ratio),
        float(args.max_depth_mm),
    )


# endregion


if __name__ == "__main__":
    try:
        output_arg, timeout_arg, fps_arg, shrink_arg, depth_arg = _parse_cli()
        RectMarkerPriorCollector(
            output_dir=output_arg,
            timeout_ms=timeout_arg,
            capture_fps=fps_arg,
            sample_shrink_ratio=shrink_arg,
            max_depth_mm=depth_arg,
        ).run()
    except KeyboardInterrupt:
        logger.warning("用户中断，程序退出。")
    except Exception as exc:
        logger.warning(f"程序异常退出：{exc}")
        raise
