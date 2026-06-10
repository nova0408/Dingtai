from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from src.wuji.dahuan_gripper_client import DahuanGripperClient, DahuanGripperInfo
from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.protocol import WujiQmlinkerConfig

# region 数据结构


@dataclass(frozen=True, slots=True)
class _GripperRequest:
    """寰宇夹爪后台请求描述。

    职责边界：
    - 只描述一次夹爪后台任务的动作和去重键。
    - 不持有 GUI 控件，不保存 SSH 连接或线程资源。

    设计思想：
    - GUI 层需要区分“可用性探测”和“状态/控制回读”两类结果，因此使用显式动作字段。
    - 线程池任务完成后通过稳定键值回收 pending 状态，避免重复发起同类硬件请求。

    生命周期：
    - 由 `DahuanGripperBackend` 在发起任务时创建。
    - worker 完成后即从 pending 集合中移除。

    继承关系：
    - 不继承业务基类，作为后端内部调度数据结构使用。
    """

    action: str
    "业务动作名，例如 `probe`、`refresh`、`set_position`。"

    key: str
    "请求去重键。"


class _GripperWorkerSignals(QObject):
    """夹爪 worker 回传信号。"""

    finished = Signal(str, object)
    failed = Signal(str, str)


class _GripperWorker(QRunnable):
    """在线程池中执行一次同步夹爪请求。"""

    def __init__(self, key: str, task: Callable[[], object]) -> None:
        super().__init__()
        self.key = key
        self.task = task
        self.signals = _GripperWorkerSignals()

    @Slot()
    def run(self) -> None:
        """执行同步夹爪请求并通过信号返回结果。"""

        try:
            result = self.task()
        except Exception as exc:  # noqa: BLE001
            try:
                self.signals.failed.emit(self.key, f"{type(exc).__name__}: {exc}")
            except RuntimeError:
                return
            return
        try:
            self.signals.finished.emit(self.key, result)
        except RuntimeError:
            return


# endregion


# region 主入口


class DahuanGripperBackend(QObject):
    """寰宇夹爪 GUI 后端。

    职责边界：
    - 负责将 GUI 请求转换为经 `ssh orin` 的夹爪读写调用。
    - 不创建 GUI 控件，不直接修改 Qt 视图。
    - 不复用 qmlinker `QMHand` 语义，不把夹爪伪装成手部执行器。

    设计思想：
    - 夹爪链路与 qmlinker 主服务链路不同，单独维护后端可避免 `WujiQmlinkerClient`
      混入具体末端工具协议细节。
    - 所有控制操作在后台线程执行，成功后统一回读最新 `DahuanGripperInfo`，
      让 GUI 只消费状态快照。

    生命周期：
    - 随主窗口创建和销毁。
    - 内部只持有轻量 `DahuanGripperClient` 单例与 Qt 线程池，不持有常驻 SSH 会话。

    继承关系：
    - 继承 `QObject` 以提供 Qt 信号，不继承业务基类。

    线程/异步语义：
    - 所有硬件读写在 Qt 线程池中执行。
    - `_pending` 只在 GUI 线程读写，worker 通过 Qt queued signal 回传结果。
    """

    availabilityResolved = Signal(bool, str)
    gripperInfoReceived = Signal(object)
    requestFailed = Signal(str)

    def __init__(self, parent: QObject | None = None, request_timeout_s: float = 10.0) -> None:
        super().__init__(parent)
        self._request_timeout_s = float(request_timeout_s)
        self._thread_pool = QThreadPool.globalInstance()
        self._base_client: WujiQmlinkerBaseClient | None = None
        self._client: DahuanGripperClient | None = None
        self._client_lock = Lock()
        self._pending: dict[str, _GripperRequest] = {}
        self._workers: dict[str, _GripperWorker] = {}

    def probe_gripper(self) -> None:
        """探测夹爪链路可用性。"""

        self._start_worker(
            _GripperRequest(action="probe", key="probe"),
            lambda: self._with_client(lambda client: client.probe()),
        )

    def refresh_gripper_info(self) -> None:
        """读取夹爪当前状态。"""

        self._start_worker(
            _GripperRequest(action="refresh", key="refresh"),
            lambda: self._with_client(lambda client: client.get_gripper_info()),
        )

    def set_gripper_enable(self, enabled: bool) -> None:
        """设置夹爪使能并回读。"""

        self._start_worker(
            _GripperRequest(action="set_enable", key="set_enable"),
            lambda: self._with_client(lambda client: self._set_enable_and_readback(client, enabled)),
        )

    def set_gripper_position(self, position: int) -> None:
        """设置夹爪目标位置并回读。"""

        self._start_worker(
            _GripperRequest(action="set_position", key="set_position"),
            lambda: self._with_client(lambda client: self._set_position_and_readback(client, position)),
        )

    def set_gripper_speed(self, speed: int) -> None:
        """设置夹爪速度并回读。"""

        self._start_worker(
            _GripperRequest(action="set_speed", key="set_speed"),
            lambda: self._with_client(lambda client: self._set_speed_and_readback(client, speed)),
        )

    def set_gripper_force(self, force: int) -> None:
        """设置夹爪力并回读。"""

        self._start_worker(
            _GripperRequest(action="set_force", key="set_force"),
            lambda: self._with_client(lambda client: self._set_force_and_readback(client, force)),
        )

    def calibrate_gripper(self) -> None:
        """执行夹爪校准并回读。"""

        self._start_worker(
            _GripperRequest(action="calibrate", key="calibrate"),
            lambda: self._with_client(lambda client: self._calibrate_and_readback(client)),
        )

    def _with_client(self, func: Callable[[DahuanGripperClient], object]) -> object:
        """获取后端持有的夹爪客户端并执行同步请求。"""

        client = self._client
        if client is None:
            with self._client_lock:
                client = self._client
                if client is None:
                    base_client = self._base_client
                    if base_client is None:
                        base_client = WujiQmlinkerBaseClient(WujiQmlinkerConfig(request_timeout_s=self._request_timeout_s))
                        self._base_client = base_client
                    client = DahuanGripperClient(base_client)
                    self._client = client
        return func(client)

    def _start_worker(self, request: _GripperRequest, task: Callable[[], object]) -> None:
        """启动一个线程池夹爪请求。"""

        if request.key in self._pending:
            return
        worker = _GripperWorker(request.key, task)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.failed.connect(self._on_worker_failed)
        self._pending[request.key] = request
        self._workers[request.key] = worker
        self._thread_pool.start(worker)

    def _set_enable_and_readback(self, client: DahuanGripperClient, enabled: bool) -> DahuanGripperInfo:
        client.set_enable(enabled)
        return client.get_gripper_info()

    def _set_position_and_readback(self, client: DahuanGripperClient, position: int) -> DahuanGripperInfo:
        client.move_gripper_position(position)
        return client.get_gripper_info()

    def _set_speed_and_readback(self, client: DahuanGripperClient, speed: int) -> DahuanGripperInfo:
        client.set_speed(speed)
        return client.get_gripper_info()

    def _set_force_and_readback(self, client: DahuanGripperClient, force: int) -> DahuanGripperInfo:
        client.set_force(force)
        return client.get_gripper_info()

    def _calibrate_and_readback(self, client: DahuanGripperClient) -> DahuanGripperInfo:
        client.calibrate()
        return client.get_gripper_info()

    @Slot(str, object)
    def _on_worker_finished(self, key: str, payload: object) -> None:
        """处理夹爪 worker 成功结果。"""

        request = self._pending.pop(key, None)
        self._workers.pop(key, None)
        if request is None:
            return
        if not isinstance(payload, DahuanGripperInfo):
            self.requestFailed.emit("dahuan gripper response format error")
            return
        if request.action == "probe":
            self.availabilityResolved.emit(True, "dahuan gripper available via orin")
        self.gripperInfoReceived.emit(payload)

    @Slot(str, str)
    def _on_worker_failed(self, key: str, message: str) -> None:
        """处理夹爪 worker 失败结果。"""

        request = self._pending.pop(key, None)
        self._workers.pop(key, None)
        if request is not None and request.action == "probe":
            self.availabilityResolved.emit(False, message)
            return
        self.requestFailed.emit(message)

    def close(self) -> None:
        """关闭后端持有的基础客户端。"""

        base_client = self._base_client
        if base_client is not None:
            base_client.close()
            self._base_client = None


# endregion
