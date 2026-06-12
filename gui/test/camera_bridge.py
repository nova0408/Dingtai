from __future__ import annotations

from threading import Event, Thread

from PySide6.QtCore import QObject, Signal

from src.wuji.camera_protocol import WujiCameraEnableState, WujiCameraName, WujiCameraRuntimeInfo, parse_wuji_camera_name
from src.wuji.zmq_camera_client import WujiZmqCameraClient


class CameraBridge(QObject):
    inventoryReady = Signal(object)
    enableStateReady = Signal(object)
    intrinsicsReady = Signal(object)
    frameReady = Signal(object, int)
    errorRaised = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._client: WujiZmqCameraClient | None = None
        self._stream_stop = Event()
        self._stream_thread: Thread | None = None
        self._stream_run_id = 0

    def set_client(self, client: WujiZmqCameraClient | None) -> None:
        self.stop_stream()
        self._client = client

    def activate(self) -> None:
        self.refresh_inventory()

    def refresh_inventory(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            runtime_infos: tuple[WujiCameraRuntimeInfo, ...] = client.list_camera_runtime_infos(online_only=False)
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
            self.enableStateReady.emit(client.get_camera_enable_state(typed_name))
            self.intrinsicsReady.emit(client.get_camera_intrinsics(typed_name))
        except Exception as exc:  # noqa: BLE001
            self.errorRaised.emit(f"相机状态刷新失败: {exc}")

    def set_enable(self, camera_name: str, enabled: bool) -> None:
        client = self._client
        if client is None:
            return
        typed_name = self._parse_camera_name(camera_name)
        if typed_name is None:
            self.errorRaised.emit(f"未知相机名: {camera_name}")
            return
        try:
            state: WujiCameraEnableState = client.set_camera_color_enabled(typed_name, enabled)
            self.enableStateReady.emit(state)
        except Exception as exc:  # noqa: BLE001
            self.errorRaised.emit(f"相机使能切换失败: {exc}")

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
        target = client.stream_camera_rgbd_frames if rgbd else client.stream_camera_rgb_frames

        def _run() -> None:
            try:
                for frame in target(typed_name):
                    if self._stream_stop.is_set() or run_id != self._stream_run_id:
                        break
                    self.frameReady.emit(frame, run_id)
            except Exception as exc:  # noqa: BLE001
                if not self._stream_stop.is_set() and run_id == self._stream_run_id:
                    self.errorRaised.emit(f"相机流失败: {exc}")

        self._stream_thread = Thread(target=_run, name=f"camera-stream-{camera_name}", daemon=True)
        self._stream_thread.start()

    @staticmethod
    def _parse_camera_name(camera_name: str) -> WujiCameraName | None:
        return parse_wuji_camera_name(camera_name)
