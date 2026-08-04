from __future__ import annotations

from threading import Event, Thread

from PySide6.QtCore import QObject, Signal

from camera_pipeline.protocol import CameraColorFramePacket, CameraFramePacket, CameraName
from camera_pipeline.service.http_client import CameraPipelineHttpClient
from camera_pipeline.service.protocol import CameraIntrinsicsResponse, CameraStatusResponse
from src.wuji.camera_protocol import (
    WujiCameraConnectionState,
    WujiCameraFrame,
    WujiCameraIntrinsicsInfo,
    WujiCameraName,
    WujiCameraRuntimeInfo,
    parse_wuji_camera_name,
)


class CameraBridge(QObject):
    inventoryReady = Signal(object)
    connectionStateReady = Signal(object)
    intrinsicsReady = Signal(object)
    frameReady = Signal(object, int)
    errorRaised = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._client: CameraPipelineHttpClient | None = None
        self._stream_stop = Event()
        self._stream_thread: Thread | None = None
        self._stream_run_id = 0

    def set_client(self, client: CameraPipelineHttpClient | None) -> None:
        self.stop_stream()
        self._client = client

    def activate(self) -> None:
        self.refresh_inventory()

    def refresh_inventory(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            runtime_infos = tuple(
                self._to_runtime_info(status)
                for status in client.get_camera_inventory()
            )
            self.inventoryReady.emit(runtime_infos)
        except Exception as exc:  # noqa: BLE001
            self.errorRaised.emit(f"相机清单刷新失败: {exc}")

    def refresh_camera(self, camera_name: str) -> None:
        client = self._client
        if client is None:
            return
        typed_name = self._parse_camera_name(camera_name)
        if typed_name is None:
            self.errorRaised.emit(f"未知相机名: {camera_name}")
            return
        try:
            pipeline_name = CameraName(typed_name)
            self.connectionStateReady.emit(
                self._to_connection_state(client.get_camera_status(pipeline_name))
            )
            intrinsics = client.get_camera_intrinsics(pipeline_name)
            self.intrinsicsReady.emit(
                self._to_wuji_intrinsics(typed_name, intrinsics)
            )
        except Exception as exc:  # noqa: BLE001
            self.errorRaised.emit(f"相机状态刷新失败: {exc}")

    def start_rgb_stream(self, camera_name: str) -> None:
        self._start_stream(camera_name, rgbd=False)

    def start_rgbd_stream(self, camera_name: str) -> None:
        self._start_stream(camera_name, rgbd=True)

    def stop_stream(self) -> None:
        self._stream_stop.set()
        self._stream_run_id += 1

    def _start_stream(self, camera_name: str, *, rgbd: bool) -> None:
        client = self._client
        if client is None:
            return
        typed_name = self._parse_camera_name(camera_name)
        if typed_name is None:
            self.errorRaised.emit(f"未知相机名: {camera_name}")
            return
        self.stop_stream()
        self._stream_stop = Event()
        self._stream_run_id += 1
        run_id = self._stream_run_id
        pipeline_name = CameraName(typed_name)
        target = client.subscribe_camera_frames if rgbd else client.subscribe_camera_color_frames

        def _run() -> None:
            try:
                for frame in target(pipeline_name):
                    if self._stream_stop.is_set() or run_id != self._stream_run_id:
                        break
                    self.frameReady.emit(self._to_wuji_frame(typed_name, frame), run_id)
            except Exception as exc:  # noqa: BLE001
                if not self._stream_stop.is_set() and run_id == self._stream_run_id:
                    self.errorRaised.emit(f"相机流失败: {exc}")

        self._stream_thread = Thread(target=_run, name=f"camera-stream-{camera_name}", daemon=True)
        self._stream_thread.start()

    @staticmethod
    def _parse_camera_name(camera_name: str) -> WujiCameraName | None:
        return parse_wuji_camera_name(camera_name)

    @staticmethod
    def _to_runtime_info(status: CameraStatusResponse) -> WujiCameraRuntimeInfo:
        camera_name = parse_wuji_camera_name(status.camera_name)
        if camera_name is None:
            raise ValueError(f"unsupported camera name: {status.camera_name}")
        return WujiCameraRuntimeInfo(
            camera_name=camera_name,
            camera_id=status.camera_id,
            serial_number="",
            display_name=status.camera_id,
            online=status.online,
            color_enabled=status.color_enabled,
            depth_enabled=status.depth_enabled,
        )

    @staticmethod
    def _to_connection_state(
        status: CameraStatusResponse,
    ) -> WujiCameraConnectionState:
        camera_name = parse_wuji_camera_name(status.camera_name)
        if camera_name is None:
            raise ValueError(f"unsupported camera name: {status.camera_name}")
        return WujiCameraConnectionState(
            camera_name=camera_name,
            connected=bool(status.online),
            message=(
                f"camera_id={status.camera_id}, "
                f"color_enabled={bool(status.color_enabled)}, "
                f"depth_enabled={bool(status.depth_enabled)}"
            ),
        )

    @staticmethod
    def _to_wuji_intrinsics(
        camera_name: WujiCameraName,
        intrinsics: CameraIntrinsicsResponse,
    ) -> WujiCameraIntrinsicsInfo:
        return WujiCameraIntrinsicsInfo(
            camera_name=camera_name,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            distortion=intrinsics.distortion,
            width=intrinsics.width,
            height=intrinsics.height,
        )

    @staticmethod
    def _to_wuji_frame(
        camera_name: WujiCameraName,
        frame: CameraColorFramePacket | CameraFramePacket,
    ) -> WujiCameraFrame:
        depth = frame.depth_mm if isinstance(frame, CameraFramePacket) else None
        return WujiCameraFrame(
            camera_name=camera_name,
            color_bgr=frame.color_bgr,
            timestamp=frame.timestamp_ms,
            sequence_id=frame.frame_id,
            depth=depth,
        )
