from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..motion_shift import estimate_phase_shift, prepare_tracking_gray, warp_mask
from .detector import TrayPointExcluder
from .types import TrayDetectionConfig


# region 配置与状态数据结构
@dataclass(frozen=True)
class TrayPipelineConfig:
    """托盘检测流程配置。

    该配置用于 `TrayDetectionPipeline`，集中管理跨帧缓存更新频率、运动估计参数
    和快速分割策略参数，避免在类内部散落大量常量。

    设计思想：
    - 使用不可变 dataclass 承载可调参数，便于测试和调用层显式覆盖。
    - 将高频实时参数与模型参数解耦。模型参数由 `TrayDetectionConfig` 管理。
    - 默认值保持与重构前逻辑一致，不引入行为兼容分支。
    """

    detect_every_n: int = 6
    """每隔多少帧触发一次高置信异步检测刷新，单位 帧。"""
    motion_downsample: float = 0.25
    """运动估计灰度图下采样比例，范围 0-1。"""
    motion_smooth_alpha: float = 0.60
    """运动先验指数平滑系数，范围 0-1，越大越平滑。"""
    motion_max_shift_px: float = 36.0
    """单帧运动估计平移限幅，单位 像素。"""
    fast_gray_percentile: float = 48.0
    """快速分割灰度阈值分位数，范围 0-100。"""
    fast_top_crop_ratio: float = 0.36
    """快速分割时屏蔽图像上部比例，范围 0-1。"""


@dataclass
class TrayRuntimeState:
    """托盘检测运行态缓存。

    该结构用于实时循环中跨帧共享临时状态，供主线程和异步检测线程协作。

    设计思想：
    - 与模型对象分离，仅持有轻量运行态数据。
    - 使用 `threading.Lock` 保证共享状态读写一致性。
    - 将运动估计与缓存掩码统一收敛到一个状态对象，降低参数透传复杂度。
    """

    cached_mask: np.ndarray | None = None
    """缓存托盘掩码，形状 `(H, W)`，dtype `uint8`。"""
    cached_ok: bool = False
    """缓存掩码是否来自高置信检测器。"""
    compute_count: int = 0
    """累计计算帧数。"""
    detect_inflight: bool = False
    """是否已有异步检测任务。"""
    lock: threading.Lock = field(default_factory=threading.Lock)
    """跨线程读写缓存互斥锁。"""
    prev_motion_gray_small: np.ndarray | None = None
    """上一帧缩小灰度图，形状 `(Hs, Ws)`，dtype `uint8`。"""
    motion_dx_smooth: float = 0.0
    """平滑 X 平移先验，单位 像素。"""
    motion_dy_smooth: float = 0.0
    """平滑 Y 平移先验，单位 像素。"""


# endregion


# region 托盘检测流程
class TrayDetectionPipeline:
    """托盘检测与跨帧掩码更新流程。

    职责边界：
    - 负责快速托盘分割、异步高置信检测刷新和跨帧掩码运动补偿。
    - 不负责点云投影、抓取位姿估计、相机采集或 GUI 渲染。

    设计思想：
    - 快速分割提供低延迟兜底掩码，高置信检测异步更新缓存掩码。
    - 运动先验用于将缓存掩码平移到当前帧，降低刷新周期内的误差。
    - 配置通过 `TrayPipelineConfig` 注入，避免参数散落在类内字段。
    """

    def __init__(self, tray_detector: TrayPointExcluder | None, config: TrayPipelineConfig | None = None) -> None:
        """初始化托盘检测流程。

        Parameters
        ----------
        tray_detector:
            高置信托盘检测器。为 None 时仅使用快速分割兜底。
        config:
            托盘流程配置。为 None 时使用默认配置。
        """
        self._tray_detector = tray_detector
        self._config = config if config is not None else TrayPipelineConfig()

    @staticmethod
    def build_default_detector() -> TrayPointExcluder | None:
        """构建默认托盘检测器。"""
        try:
            cfg = TrayDetectionConfig(
                strict_target_filter=True,
                max_targets=1,
                use_sam=False,
                min_confidence=0.20,
                topk_objects=2,
                sam_max_boxes=1,
                sam_primary_only=True,
                combine_prompts_forward=True,
                detect_max_side=384,
            )
            return TrayPointExcluder(cfg)
        except Exception:
            return None

    def segment_tray(self, rgb_bgr: np.ndarray, state: TrayRuntimeState) -> tuple[np.ndarray, bool]:
        """输出当前帧托盘掩码。

        Parameters
        ----------
        rgb_bgr:
            输入 BGR 图像，形状 `(H, W, 3)`，dtype 通常为 `uint8`。
        state:
            托盘运行态缓存对象。

        Returns
        -------
        mask:
            当前帧托盘掩码，形状 `(H, W)`，dtype `uint8`。
        from_detector:
            True 表示掩码来自高置信检测缓存，False 表示来自快速分割。
        """
        motion_dx, motion_dy = self._estimate_motion(rgb_bgr, state)
        state.compute_count += 1
        fast_mask = self._segment_tray_fast(rgb_bgr)
        should_refresh = self._tray_detector is not None and (
            state.cached_mask is None
            or (state.compute_count % max(1, int(self._config.detect_every_n)) == 1)
            or (not state.cached_ok)
        )
        if should_refresh:
            self._start_async_refine(rgb_bgr, state)

        with state.lock:
            cached = None if state.cached_mask is None else np.asarray(state.cached_mask, dtype=np.uint8).copy()
            cached_ok = bool(state.cached_ok)
        if cached is None:
            return fast_mask, False
        return self._warp_mask(cached, motion_dx, motion_dy), cached_ok

    def _start_async_refine(self, rgb_bgr: np.ndarray, state: TrayRuntimeState) -> None:
        """启动异步高置信检测刷新任务。"""
        if self._tray_detector is None:
            return
        detector = self._tray_detector
        with state.lock:
            if state.detect_inflight:
                return
            state.detect_inflight = True
        frame = rgb_bgr.copy()

        def _task() -> None:
            try:
                dets = detector.detect(frame)
                if len(dets) > 0:
                    det = max(dets, key=lambda d: int(np.count_nonzero(np.asarray(d.mask))))
                    mask = cv2.morphologyEx(
                        np.asarray(det.mask, dtype=np.uint8),
                        cv2.MORPH_CLOSE,
                        np.ones((9, 9), dtype=np.uint8),
                        iterations=2,
                    )
                    with state.lock:
                        state.cached_mask = mask
                        state.cached_ok = True
            finally:
                with state.lock:
                    state.detect_inflight = False

        threading.Thread(target=_task, name="tray_refine_async", daemon=True).start()

    def _segment_tray_fast(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """基于灰度阈值和连通域的快速托盘分割。"""
        h, w = rgb_bgr.shape[:2]
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        # gray: (H, W) uint8；分位阈值分割黑色托盘候选区域。
        base = (gray <= np.percentile(gray, float(self._config.fast_gray_percentile))).astype(np.uint8) * 255
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=1)
        base[: int(float(self._config.fast_top_crop_ratio) * h), :] = 0

        num, cc, stats, _ = cv2.connectedComponentsWithStats(base, connectivity=8)
        if num <= 1:
            return base

        best = 1
        best_score = -1e18
        tgt = np.array([0.5 * w, 0.78 * h], dtype=np.float64)
        for idx in range(1, num):
            area = float(stats[idx, cv2.CC_STAT_AREA])
            cx = float(stats[idx, cv2.CC_STAT_LEFT] + 0.5 * stats[idx, cv2.CC_STAT_WIDTH])
            cy = float(stats[idx, cv2.CC_STAT_TOP] + 0.5 * stats[idx, cv2.CC_STAT_HEIGHT])
            dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float64) - tgt))
            score = area - 1.2 * dist
            if score > best_score:
                best_score = score
                best = idx

        out = np.zeros((h, w), dtype=np.uint8)
        out[cc == best] = 255
        return out

    def _estimate_motion(self, rgb_bgr: np.ndarray, state: TrayRuntimeState) -> tuple[float, float]:
        """估计当前帧相对上一帧的平移并做平滑。"""
        max_side = int(round(max(rgb_bgr.shape[0], rgb_bgr.shape[1]) * float(self._config.motion_downsample)))
        small, scale = prepare_tracking_gray(rgb_bgr, max(32, max_side))
        small_u8 = np.asarray(np.clip(small, 0.0, 255.0), dtype=np.uint8)
        with state.lock:
            prev = None if state.prev_motion_gray_small is None else state.prev_motion_gray_small.copy()
            state.prev_motion_gray_small = small_u8
            dx_s = float(state.motion_dx_smooth)
            dy_s = float(state.motion_dy_smooth)
        if prev is None or prev.shape != small_u8.shape:
            return dx_s, dy_s

        shift = estimate_phase_shift(
            ref_gray=np.asarray(prev, dtype=np.float32),
            cur_gray=np.asarray(small_u8, dtype=np.float32),
            scale=float(scale),
            min_response=0.02,
            max_shift_px=float(self._config.motion_max_shift_px),
        )
        if not shift.valid:
            return dx_s, dy_s

        dx_raw = float(shift.dx_px)
        dy_raw = float(shift.dy_px)

        a = float(np.clip(float(self._config.motion_smooth_alpha), 0.05, 0.98))
        dx_new = a * dx_s + (1.0 - a) * dx_raw
        dy_new = a * dy_s + (1.0 - a) * dy_raw
        with state.lock:
            state.motion_dx_smooth = dx_new
            state.motion_dy_smooth = dy_new
        return dx_new, dy_new

    @staticmethod
    def _warp_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """对托盘掩码执行平移补偿。"""
        return warp_mask(np.asarray(mask, dtype=np.uint8), dx_px=float(dx), dy_px=float(dy))


# endregion
