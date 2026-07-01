from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger

from camera_pipeline.opening_detection_pipeline.protocol import GraspPosePipelineRequest
from camera_pipeline.opening_detection_pipeline.transport import GraspPosePipelineRpcClient, ZmqSocketOptions
from src.wuji.camera_protocol import WujiCameraFrame, WujiCameraIntrinsicsInfo, WujiCameraName
from src.wuji.zmq_camera_catalog import SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL
from src.wuji.zmq_camera_client import WujiZmqCameraClient

# region 模块常量
LEFT_CAMERA_NAME = "left_hand_camera"
DEFAULT_WUJI_CAMERA_HOST = "192.168.100.60"
DEFAULT_RPC_TIMEOUT_MS = 60_000
DEFAULT_ORIN_PIPELINE_SERVICE_PORT = 6220
# endregion


# region 数据结构
@dataclass(frozen=True)
class WujiPoseRequestBundle:
    """一次抓取位姿计算所需的原始输入。"""

    frame_id: int
    frame: WujiCameraFrame
    intrinsics: WujiCameraIntrinsicsInfo
    request: GraspPosePipelineRequest


@dataclass(frozen=True)
class WujiPoseExecutionResult:
    """一次抓取位姿计算的完整输出。"""

    frame_id: int
    frame: WujiCameraFrame
    intrinsics: WujiCameraIntrinsicsInfo
    request: GraspPosePipelineRequest
    response: Any | None
    error: str | None


@dataclass(frozen=True)
class WujiPoseRuntimeStatus:
    """持续执行上下文的当前状态。"""

    pending_frame_id: int | None
    completed_frame_id: int | None
    busy: bool
    enabled: bool
    submitted_count: int
    completed_count: int
    dropped_count: int
    last_error: str | None
    compute_interval_ms: int


@dataclass(frozen=True)
class WujiPosePreviewSnapshot:
    """持续执行模式下缓存的最新相机帧。"""

    frame_id: int
    frame: WujiCameraFrame
    intrinsics: WujiCameraIntrinsicsInfo
# endregion


# region 执行上下文
class WujiPoseExecutionContext:
    """无际抓取位姿联调用执行上下文。

    统一负责抓取真实相机帧、构造请求、调用 Orin、保存结果队列。
    GUI 与冒烟测试都只能通过这个上下文获取计算结果。
    """

    def __init__(
        self,
        service_addr: str,
        camera_host: Optional[str] = None,
        camera_control_port: int = 5570,
        camera_stream_port: int = 5562,
        camera_id: WujiCameraName = "left_hand_camera",
    ) -> None:
        self._service_addr = str(service_addr)
        self._camera_host = str(DEFAULT_WUJI_CAMERA_HOST if camera_host is None else camera_host)
        self._camera_control_port = int(camera_control_port)
        self._camera_stream_port = int(camera_stream_port)
        self._camera_id: WujiCameraName = camera_id
        self._latest_result: Optional[WujiPoseExecutionResult] = None
        self._latest_preview: Optional[WujiPosePreviewSnapshot] = None

        self._job_queue: queue.Queue[WujiPoseRequestBundle | None] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[WujiPoseExecutionResult] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._enabled = False
        self._busy = False
        self._pending_frame_id: int | None = None
        self._completed_frame_id: int | None = None
        self._target_tray_index = 0
        self._enable_debug = True
        self._submitted_count = 0
        self._completed_count = 0
        self._dropped_count = 0
        self._last_error: str | None = None
        self._compute_interval_ms = 1000
        self._last_submit_monotonic = 0.0
        self._worker_thread: Optional[threading.Thread] = None
        self._producer_thread: Optional[threading.Thread] = None

    # region 对外接口
    def run_once(self, target_tray_index: int = 0, enable_debug: bool = True, frame_id: int = 1) -> WujiPoseExecutionResult:
        """执行一次同步抓取位姿计算。"""

        logger.debug(
            "pose context run_once start: frame_id={} target_tray_index={} enable_debug={}",
            int(frame_id),
            int(target_tray_index),
            bool(enable_debug),
        )
        result = self._execute_frame(
            frame_id=int(frame_id),
            target_tray_index=int(target_tray_index),
            enable_debug=bool(enable_debug),
        )
        with self._state_lock:
            self._latest_result = result
            self._completed_frame_id = int(frame_id)
            self._last_error = result.error
        logger.debug(
            "pose context run_once finish: frame_id={} error={}",
            int(frame_id),
            result.error,
        )
        return result

    def start_continuous(
        self,
        target_tray_index: int = 0,
        enable_debug: bool = True,
        compute_interval_ms: int = 1000,
    ) -> None:
        """启动持续抓取与后台 RPC 计算流程。"""

        with self._state_lock:
            self._target_tray_index = int(target_tray_index)
            self._enable_debug = bool(enable_debug)
            self._compute_interval_ms = max(100, int(compute_interval_ms))
            if self._enabled:
                logger.debug(
                    "pose context start_continuous skipped: already enabled target_tray_index={} compute_interval_ms={}",
                    self._target_tray_index,
                    self._compute_interval_ms,
                )
                return
            self._enabled = True
            self._busy = False
            self._pending_frame_id = None
            self._completed_frame_id = None
            self._last_error = None
            self._submitted_count = 0
            self._completed_count = 0
            self._dropped_count = 0
            self._last_submit_monotonic = 0.0
            self._stop_event.clear()
        logger.debug(
            "pose context continuous start: camera_id={} target_tray_index={} enable_debug={} compute_interval_ms={} service_addr={}",
            self._camera_id,
            int(target_tray_index),
            bool(enable_debug),
            self._compute_interval_ms,
            self._service_addr,
        )
        self._clear_job_queue()
        self._clear_result_queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="wuji-pose-worker", daemon=True)
        self._worker_thread.start()
        self._producer_thread = threading.Thread(target=self._capture_loop, name="wuji-pose-capture", daemon=True)
        self._producer_thread.start()

    def stop_continuous(self) -> None:
        """停止持续执行流程，并尽量完成线程与队列清理。"""

        logger.debug("pose context continuous stop requested")
        with self._state_lock:
            self._enabled = False
            self._busy = False
            self._pending_frame_id = None
        self._stop_event.set()
        self._offer_stop_job()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=1.0)
            self._producer_thread = None
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None
        self._clear_job_queue()
        logger.debug("pose context continuous stop finished")

    def update_target(self, target_tray_index: int, enable_debug: bool = True, compute_interval_ms: int | None = None) -> None:
        """更新持续模式下的目标托盘与计算节拍。"""

        with self._state_lock:
            self._target_tray_index = int(target_tray_index)
            self._enable_debug = bool(enable_debug)
            if compute_interval_ms is not None:
                self._compute_interval_ms = max(100, int(compute_interval_ms))
        # logger.debug(
        #     "pose context target updated: target_tray_index={} enable_debug={} compute_interval_ms={}",
        #     int(target_tray_index),
        #     bool(enable_debug),
        #     self._compute_interval_ms,
        # )

    def poll_result(self) -> Optional[WujiPoseExecutionResult]:
        latest: Optional[WujiPoseExecutionResult] = None
        while True:
            try:
                latest = self._result_queue.get_nowait()
            except queue.Empty:
                return latest

    def snapshot_status(self) -> WujiPoseRuntimeStatus:
        with self._state_lock:
            return WujiPoseRuntimeStatus(
                pending_frame_id=self._pending_frame_id,
                completed_frame_id=self._completed_frame_id,
                busy=self._busy,
                enabled=self._enabled,
                submitted_count=self._submitted_count,
                completed_count=self._completed_count,
                dropped_count=self._dropped_count,
                last_error=self._last_error,
                compute_interval_ms=self._compute_interval_ms,
            )

    def get_latest_preview(self) -> Optional[WujiPosePreviewSnapshot]:
        return self._latest_preview
    # endregion

    # region 请求构造与 RPC 调用
    def capture_live_request_bundle(self, frame_id: int, target_tray_index: int, enable_debug: bool) -> WujiPoseRequestBundle:
        """主动抓取一帧 RGBD 数据，并构造一次 RPC 请求包。"""

        client = WujiZmqCameraClient(
            host=self._camera_host,
            control_port=self._camera_control_port,
            request_timeout_ms=3000,
            stream_timeout_ms=8000,
            camera_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
        )
        try:
            intrinsics = client.get_camera_intrinsics(self._camera_id)
            frame = next(client.stream_camera_rgbd_frames(self._camera_id))
        finally:
            client.close()
        resolved_frame_id = int(frame_id)
        if resolved_frame_id <= 0:
            resolved_frame_id = self._resolve_stream_frame_id(frame)
        request = self.build_request(
            frame_id=resolved_frame_id,
            frame=frame,
            intrinsics=intrinsics,
            target_tray_index=target_tray_index,
            enable_debug=enable_debug,
        )
        return WujiPoseRequestBundle(
            frame_id=int(resolved_frame_id),
            frame=frame,
            intrinsics=self._to_gui_intrinsics(intrinsics),
            request=request,
        )

    def build_request(
        self,
        frame_id: int,
        frame: WujiCameraFrame,
        intrinsics: Any,
        target_tray_index: int,
        enable_debug: bool,
    ) -> GraspPosePipelineRequest:
        """将相机帧元信息转换为统一 pipeline RPC 请求对象。

        Notes
        -----
        当前 GUI 只负责从本地相机链路抓取预览帧并取得 `frame_id`，真正的 RGBD 数据读取、
        托盘检测与抓取位姿计算全部由 Orin 侧 `opening_detection_pipeline` 主服务完成。
        """

        return GraspPosePipelineRequest(
            request_id=int(frame_id),
            frame_id=int(frame_id),
            camera_name=str(frame.camera_name),
            target_tray_index=int(target_tray_index),
            enable_debug=bool(enable_debug),
        )

    def call_orin(self, bundle: WujiPoseRequestBundle) -> WujiPoseExecutionResult:
        """发送一次 RPC 请求，并将原始输入与响应打包成统一结果。"""

        client = GraspPosePipelineRpcClient(
            self._service_addr,
            options=ZmqSocketOptions(recv_timeout_ms=DEFAULT_RPC_TIMEOUT_MS, send_timeout_ms=DEFAULT_RPC_TIMEOUT_MS),
        )
        try:
            logger.debug(
                "pose context rpc send: frame_id={} camera_name={} target_tray_index={} enable_debug={}",
                int(bundle.request.frame_id),
                bundle.request.camera_name,
                int(bundle.request.target_tray_index),
                bool(bundle.request.enable_debug),
            )
            response = client.call(bundle.request)
            logger.debug(
                "pose context rpc recv: frame_id={} tray_count={} selected_tray_index={} error={}",
                int(bundle.frame_id),
                0 if response is None else int(response.tray_count),
                -1 if response is None else int(response.selected_tray_index),
                None if response is None else response.error,
            )
            return WujiPoseExecutionResult(
                frame_id=bundle.frame_id,
                frame=bundle.frame,
                intrinsics=bundle.intrinsics,
                request=bundle.request,
                response=response,
                error=response.error,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("pose rpc call failed: {}", exc)
            return WujiPoseExecutionResult(
                frame_id=bundle.frame_id,
                frame=bundle.frame,
                intrinsics=bundle.intrinsics,
                request=bundle.request,
                response=None,
                error="{0}: {1}".format(type(exc).__name__, exc),
            )
        finally:
            client.close()

    def get_latest_result(self) -> Optional[WujiPoseExecutionResult]:
        return self._latest_result
    # endregion

    # region 持续模式内部线程
    def _capture_loop(self) -> None:
        """持续读取相机流，并按节拍把最新帧投递给后台 RPC 线程。"""

        client = WujiZmqCameraClient(
            host=self._camera_host,
            control_port=self._camera_control_port,
            request_timeout_ms=3000,
            stream_timeout_ms=8000,
            camera_endpoints=SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL,
        )
        try:
            intrinsics = self._to_gui_intrinsics(client.get_camera_intrinsics(self._camera_id))
            for frame in client.stream_camera_rgbd_frames(self._camera_id):
                if self._stop_event.is_set():
                    break
                frame_id = self._resolve_stream_frame_id(frame)
                self._latest_preview = WujiPosePreviewSnapshot(
                    frame_id=int(frame_id),
                    frame=frame,
                    intrinsics=intrinsics,
                )
                with self._state_lock:
                    if not self._enabled:
                        break
                    target_tray_index = int(self._target_tray_index)
                    enable_debug = bool(self._enable_debug)
                    interval_s = max(0.1, float(self._compute_interval_ms) / 1000.0)
                    last_submit = float(self._last_submit_monotonic)
                    busy = bool(self._busy)
                now = time.perf_counter()
                if busy or (now - last_submit < interval_s):
                    continue
                try:
                    request = self.build_request(
                        frame_id=int(frame_id),
                        frame=frame,
                        intrinsics=intrinsics,
                        target_tray_index=int(target_tray_index),
                        enable_debug=bool(enable_debug),
                    )
                    bundle = WujiPoseRequestBundle(
                        frame_id=int(frame_id),
                        frame=frame,
                        intrinsics=intrinsics,
                        request=request,
                    )
                    logger.debug(
                        "pose context queue put before: frame_id={} busy={} queue_size={} submitted_count={}",
                        int(frame_id),
                        bool(busy),
                        self._job_queue.qsize(),
                        self._submitted_count,
                    )
                    self._job_queue.put_nowait(bundle)
                    with self._state_lock:
                        self._submitted_count += 1
                        self._pending_frame_id = int(frame_id)
                        self._last_submit_monotonic = time.perf_counter()
                        self._last_error = None
                    logger.debug(
                        "pose context queue put after: frame_id={} queue_size={} submitted_count={} pending_frame_id={}",
                        int(frame_id),
                        self._job_queue.qsize(),
                        self._submitted_count,
                        self._pending_frame_id,
                    )
                except queue.Full:
                    with self._state_lock:
                        self._dropped_count += 1
                    logger.debug(
                        "pose context queue put dropped: frame_id={} dropped_count={} queue_size={}",
                        int(frame_id),
                        self._dropped_count,
                        self._job_queue.qsize(),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("pose context capture/build request failed")
                    self._push_failed_result(
                        frame_id=int(frame_id),
                        target_tray_index=int(target_tray_index),
                        enable_debug=bool(enable_debug),
                        error="{0}: {1}".format(type(exc).__name__, exc),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("pose context camera stream failed")
            self._push_failed_result(
                frame_id=-1,
                target_tray_index=int(self._target_tray_index),
                enable_debug=bool(self._enable_debug),
                error="{0}: {1}".format(type(exc).__name__, exc),
            )
        finally:
            client.close()

    def _worker_loop(self) -> None:
        """从请求队列取出任务，串行执行 RPC，并维护结果缓存状态。"""

        while not self._stop_event.is_set():
            try:
                job = self._job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if job is None:
                self._job_queue.task_done()
                break
            with self._state_lock:
                self._busy = True
            logger.debug(
                "pose context worker dequeued: frame_id={} job_queue_size={} result_queue_size={}",
                int(job.frame_id),
                self._job_queue.qsize(),
                self._result_queue.qsize(),
            )
            result = self.call_orin(job)
            logger.debug(
                "pose context result enqueue before: frame_id={} result_queue_size={} error={}",
                int(result.frame_id),
                self._result_queue.qsize(),
                result.error,
            )
            self._put_latest_result(result)
            with self._state_lock:
                self._latest_result = result
                self._completed_frame_id = int(result.frame_id)
                self._pending_frame_id = None
                self._completed_count += 1
                self._last_error = result.error
                self._busy = False
            logger.debug(
                "pose context result enqueue after: frame_id={} result_queue_size={} completed_count={} last_error={}",
                int(result.frame_id),
                self._result_queue.qsize(),
                self._completed_count,
                self._last_error,
            )
            self._job_queue.task_done()

    def _execute_frame(self, frame_id: int, target_tray_index: int, enable_debug: bool) -> WujiPoseExecutionResult:
        """同步抓取一帧并完成一次 RPC 调用，供单次执行模式使用。"""

        try:
            bundle = self.capture_live_request_bundle(
                frame_id=int(frame_id),
                target_tray_index=int(target_tray_index),
                enable_debug=bool(enable_debug),
            )
            return self.call_orin(bundle)
        except Exception as exc:  # noqa: BLE001
            logger.error("pose context execution failed: frame={} target={}", frame_id, target_tray_index)
            return WujiPoseExecutionResult(
                frame_id=int(frame_id),
                frame=self._empty_frame(),
                intrinsics=self._empty_intrinsics(),
                request=self._empty_request(frame_id=int(frame_id), target_tray_index=int(target_tray_index), enable_debug=bool(enable_debug)),
                response=None,
                error="{0}: {1}".format(type(exc).__name__, exc),
            )
    # endregion

    # region 队列与兜底对象
    def _put_latest_result(self, result: WujiPoseExecutionResult) -> None:
        """保留最新结果；当结果队列已满时主动丢弃更旧结果。"""

        while True:
            try:
                self._result_queue.put_nowait(result)
                return
            except queue.Full:
                logger.debug(
                    "pose context result queue full: drop oldest before enqueue frame_id={} queue_size={}",
                    int(result.frame_id),
                    self._result_queue.qsize(),
                )
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    return

    def _push_failed_result(self, frame_id: int, target_tray_index: int, enable_debug: bool, error: str) -> None:
        """当采集或构包失败时，向结果流中推送结构化失败结果。"""

        failed_result = WujiPoseExecutionResult(
            frame_id=int(frame_id),
            frame=self._empty_frame(),
            intrinsics=self._empty_intrinsics(),
            request=self._empty_request(
                frame_id=int(frame_id),
                target_tray_index=int(target_tray_index),
                enable_debug=bool(enable_debug),
            ),
            response=None,
            error=str(error),
        )
        self._put_latest_result(failed_result)
        with self._state_lock:
            self._completed_count += 1
            self._last_error = failed_result.error
        logger.debug(
            "pose context failed result pushed: frame_id={} completed_count={} error={}",
            int(frame_id),
            self._completed_count,
            failed_result.error,
        )

    def _offer_stop_job(self) -> None:
        """向任务队列投递停止哨兵；队列满时优先移除旧任务再投递。"""

        try:
            self._job_queue.put_nowait(None)
        except queue.Full:
            try:
                self._job_queue.get_nowait()
                self._job_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._job_queue.put_nowait(None)
            except queue.Full:
                return

    def _clear_job_queue(self) -> None:
        """清空后台任务队列，避免旧任务串入新一轮持续执行。"""

        while True:
            try:
                self._job_queue.get_nowait()
                self._job_queue.task_done()
            except queue.Empty:
                return

    def _clear_result_queue(self) -> None:
        """清空结果队列，保证调用方读取到的是新一轮执行结果。"""

        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                return

    def _empty_request(self, frame_id: int, target_tray_index: int, enable_debug: bool) -> GraspPosePipelineRequest:
        """构造最小空请求，用于异常兜底结果。"""

        return GraspPosePipelineRequest(
            request_id=int(frame_id),
            frame_id=int(frame_id),
            camera_name=str(self._camera_id),
            target_tray_index=int(target_tray_index),
            enable_debug=bool(enable_debug),
        )

    def _empty_frame(self) -> WujiCameraFrame:
        """构造最小空帧，确保失败结果仍具备统一字段结构。"""

        return WujiCameraFrame(
            camera_name=self._camera_id,
            color_bgr=np.zeros((1, 1, 3), dtype=np.uint8),
            timestamp=None,
            sequence_id=None,
            depth=np.zeros((1, 1), dtype=np.uint16),
        )

    def _empty_intrinsics(self) -> WujiCameraIntrinsicsInfo:
        """构造最小空内参，供异常路径复用。"""

        return WujiCameraIntrinsicsInfo(
            camera_name=self._camera_id,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            distortion=tuple(),
            width=1,
            height=1,
        )
    # endregion

    # region 静态辅助方法
    @staticmethod
    def _to_gui_intrinsics(intrinsics: Any) -> WujiCameraIntrinsicsInfo:
        """将底层相机内参对象复制为 GUI / 测试层统一结构。"""

        return WujiCameraIntrinsicsInfo(
            camera_name=LEFT_CAMERA_NAME,
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            cx=float(intrinsics.cx),
            cy=float(intrinsics.cy),
            distortion=tuple(float(v) for v in getattr(intrinsics, "distortion", tuple())),
            width=int(getattr(intrinsics, "width", 0)),
            height=int(getattr(intrinsics, "height", 0)),
        )

    @staticmethod
    def _resolve_stream_frame_id(frame: WujiCameraFrame) -> int:
        """优先使用 sequence_id，其次回退到时间戳，最后使用当前毫秒时间。"""

        if frame.sequence_id is not None:
            return int(frame.sequence_id)
        timestamp = frame.timestamp
        if isinstance(timestamp, (int, np.integer)):
            return int(timestamp)
        return int(time.time() * 1000.0)
    # endregion
# endregion
