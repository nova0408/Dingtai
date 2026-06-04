from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from threading import Lock
from typing import Any

from loguru import logger
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from src.arm import (
    SUPPORTED_ARM_DEVICES,
    WujiArmQmlinkerClient,
    WujiArmQmlinkerConfig,
    axis_names_for_device,
    load_wuji_robot_network_config,
    parse_arm_axis_name,
)
from src.arm.wuji_arm_protocol import ArmDeviceName

# region 数据结构


@dataclass(frozen=True, slots=True)
class _SdkRequest:
    """qmlinker 工作线程关联请求。

    职责边界：
    - 只记录一次异步 SDK 请求的业务动作与可选设备信息。
    - 不持有 GUI 控件。

    设计思想：
    - Qt 线程池完成时需要恢复请求语义，便于将结果分发到对应 GUI 信号。

    生命周期：
    - 由 `WujiArmGrpcBackend` 创建，在 worker 完成后从 pending 集合中删除。

    继承关系：
    - 不继承业务基类，作为后端内部数据结构使用。
    """

    action: str
    "业务动作名称，例如 `sdk_probe`、`get_joints`。"

    key: str
    "请求去重键，通常由动作与设备名组成。"

    axis_name: str | None = None
    "GUI 轴名，仅关节相关请求使用。"

    device_name: ArmDeviceName | None = None
    "机械臂设备名，仅机械臂请求使用。"


class _SdkWorkerSignals(QObject):
    """qmlinker worker 回传信号。"""

    finished = Signal(str, object)
    failed = Signal(str, str)


class _SdkWorker(QRunnable):
    """在线程池中执行一次同步 qmlinker 调用。"""

    def __init__(self, key: str, task: Callable[[], object]) -> None:
        super().__init__()
        self.key = key
        self.task = task
        self.signals = _SdkWorkerSignals()

    @Slot()
    def run(self) -> None:
        """执行同步 qmlinker 调用并通过信号返回结果。"""

        try:
            result = self.task()
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.key, f"{type(exc).__name__}: {exc}")
            return
        self.signals.finished.emit(self.key, result)


# endregion


# region 主入口


class WujiArmGrpcBackend(QObject):
    """无际机械臂 GUI 后端。

    职责边界：
    - 负责将 GUI 请求转换为本机 qmlinker SDK 调用并分发结果。
    - 不通过 SSH 执行远端 Python，不修改远端环境。
    - 不创建或修改 GUI 控件。

    设计思想：
    - 保留原 GUI 使用的信号名，降低调试页改动范围。
    - 底层使用接口文档提供的 qmlinker wheel，避免重复实现协议细节。
    - 后端持有一个 qmlinker 客户端，复用 SDK 内部状态更新线程。

    生命周期：
    - 随 GUI 主窗口创建和销毁。
    - 每个请求进入 Qt 全局线程池，完成后回到 GUI 线程发信号。

    继承关系：
    - 继承 `QObject` 以提供 Qt 信号，不继承业务基类。

    线程/异步语义：
    - qmlinker 请求在 Qt 线程池中执行。
    - `_pending` 只在 GUI 线程读写，worker 通过 Qt queued signal 回传结果。
    """

    sshStateChanged = Signal(bool, str)
    enableStateReceived = Signal(str, bool)
    dofValuesReceived = Signal(dict)
    requestFailed = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        ssh_host_alias: str = "base_control",
        config: WujiArmQmlinkerConfig | None = None,
    ) -> None:
        super().__init__(parent)
        self._legacy_host_alias = ssh_host_alias
        self._config = load_wuji_robot_network_config().qmlinker if config is None else config
        self._thread_pool = QThreadPool.globalInstance()
        self._pending: dict[str, _SdkRequest] = {}
        self._workers: dict[str, _SdkWorker] = {}
        self._client: WujiArmQmlinkerClient | None = None
        self._client_lock = Lock()

    def configure_endpoint(self, host: str, port: int | None = None) -> None:
        """从 GUI 输入更新 qmlinker 目标地址。"""

        clean_host = host.strip()
        if not clean_host:
            clean_host = self._config.host
        previous_target = self._config.target()
        self._config = WujiArmQmlinkerConfig(
            host=clean_host,
            port=self._config.port if port is None else int(port),
            default_speed_ratio=self._config.default_speed_ratio,
            request_timeout_s=self._config.request_timeout_s,
            stream_first_timeout_s=self._config.stream_first_timeout_s,
        )
        if previous_target != self._config.target():
            self._close_client()
            logger.info("Configured qmlinker endpoint: target={}", self._config.target())

    def connect_ssh(self) -> None:
        """检查 qmlinker 目标是否可读取机械臂状态。"""

        logger.info(
            "qmlinker connect requested: alias={} target={}",
            self._legacy_host_alias,
            self._config.target(),
        )
        self._start_worker(
            _SdkRequest(action="sdk_probe", key="sdk_probe"),
            lambda: self._with_client(lambda client: client.check_ready()),
        )

    def disconnect_ssh(self) -> None:
        """断开 GUI 侧连接状态。"""

        logger.info("qmlinker disconnect requested: target={}", self._config.target())
        self._close_client()
        self.sshStateChanged.emit(False, "qmlinker disconnected")

    def refresh_ssh_state(self) -> None:
        """刷新 qmlinker 目标可达性。"""

        self.connect_ssh()

    def refresh_enable_state(self, device_name: str) -> None:
        """读取机械臂真实使能状态。"""

        if device_name not in SUPPORTED_ARM_DEVICES:
            logger.warning("Skip unsupported enable refresh: device_name={}", device_name)
            return
        typed_device = self._typed_device(device_name)
        self._start_worker(
            _SdkRequest(
                action="get_enable",
                key=f"get_enable:{typed_device}",
                device_name=typed_device,
            ),
            lambda: self._with_client(
                lambda client: {
                    "device_name": typed_device,
                    "enabled": client.get_enable(typed_device),
                }
            ),
        )

    def set_enable_state(self, device_name: str, enabled: bool) -> None:
        """设置机械臂使能并读取真实回写状态。"""

        if device_name not in SUPPORTED_ARM_DEVICES:
            logger.warning("Reject unsupported enable request: device_name={}", device_name)
            self.requestFailed.emit(f"当前接口文档未提供 {device_name} 使能接口")
            return
        typed_device = self._typed_device(device_name)
        self._start_worker(
            _SdkRequest(
                action="set_enable",
                key=f"set_enable:{typed_device}",
                device_name=typed_device,
            ),
            lambda: self._with_client(
                lambda client: self._set_enable_and_readback(client, typed_device, enabled)
            ),
        )

    def refresh_dof_value(self, axis_name: str) -> None:
        """读取某个机械臂轴所在整臂的真实关节角度。"""

        parsed = parse_arm_axis_name(axis_name)
        if parsed is None:
            logger.warning("Skip unsupported DoF refresh: axis_name={}", axis_name)
            return
        device_name, _ = parsed
        self._start_joint_refresh(device_name, axis_name, action="get_joints")

    def set_dof_target(self, axis_name: str, target_angle_deg: float) -> None:
        """设置单个机械臂关节目标角度。"""

        parsed = parse_arm_axis_name(axis_name)
        if parsed is None:
            logger.warning("Reject unsupported DoF target: axis_name={}", axis_name)
            self.requestFailed.emit(f"当前接口文档未提供 {axis_name} 控制接口")
            return
        device_name, joint_index = parsed
        self._start_worker(
            _SdkRequest(
                action="set_joint",
                key=f"set_joint:{device_name}:{joint_index}",
                axis_name=axis_name,
                device_name=device_name,
            ),
            lambda: self._with_client(
                lambda client: self._set_joint_and_readback(
                    client,
                    device_name,
                    joint_index,
                    target_angle_deg,
                )
            ),
        )

    def get_joint_states(self, device_name: ArmDeviceName) -> object:
        """同步读取机械臂关节状态，供非 GUI 自动化测试使用。"""

        return self._with_client(lambda client: client.get_joint_states(device_name))

    def set_joints(
        self,
        device_name: ArmDeviceName,
        joint_commands: Iterable[dict[str, float | int]],
        sync_threshold: float = 0.0,
    ) -> object:
        """同步批量设置机械臂关节角度，供非 GUI 自动化测试使用。"""

        return self._with_client(
            lambda client: client.set_joints(device_name, joint_commands, sync_threshold)
        )

    def stream_get_joint_states(
        self,
        device_name: ArmDeviceName,
        duration_s: float,
    ) -> Iterator[object]:
        """同步流式读取机械臂关节状态，供非 GUI 自动化测试使用。"""

        client = self._with_client(lambda value: value)
        if not isinstance(client, WujiArmQmlinkerClient):
            raise TypeError("qmlinker client type error")
        return client.stream_get_joint_states(device_name, duration_s)

    def stream_set_joint_states(
        self,
        device_name: ArmDeviceName,
        command_frames: Iterable[Iterable[dict[str, float | int]]],
    ) -> object:
        """同步流式设置机械臂关节角度，供非 GUI 自动化测试使用。"""

        return self._with_client(lambda client: client.stream_set_joint_states(device_name, command_frames))

    def fk(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> object:
        """同步执行 qmlinker FK，供非 GUI 自动化测试使用。"""

        return self._with_client(lambda client: client.fk(device_name, joint_angles_rad))

    def fk_fast(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> object:
        """同步执行 qmlinker 快速 FK，供非 GUI 自动化测试使用。"""

        return self._with_client(lambda client: client.fk_fast(device_name, joint_angles_rad))

    def ik(
        self,
        device_name: ArmDeviceName,
        target_pose: Any,
        reference_joint_angles_rad: Iterable[float],
    ) -> object:
        """同步执行 qmlinker IK，供非 GUI 自动化测试使用。"""

        return self._with_client(
            lambda client: client.ik(device_name, target_pose, reference_joint_angles_rad)
        )

    def current_fk_fast(self, device_name: ArmDeviceName) -> object:
        """同步读取当前关节角并计算末端位姿矩阵。"""

        return self._with_client(lambda client: client.current_fk_fast(device_name))

    def _start_joint_refresh(
        self,
        device_name: ArmDeviceName,
        axis_name: str | None,
        action: str,
    ) -> None:
        """启动读取整臂关节状态的异步请求。"""

        self._start_worker(
            _SdkRequest(
                action=action,
                key=f"{action}:{device_name}",
                axis_name=axis_name,
                device_name=device_name,
            ),
            lambda: self._with_client(lambda client: self._read_joint_values(client, device_name)),
        )

    def _start_worker(self, request: _SdkRequest, task: Callable[[], object]) -> None:
        """启动一个线程池 qmlinker 请求。"""

        if request.key in self._pending:
            logger.debug("Skip duplicated qmlinker request: key={}", request.key)
            return
        worker = _SdkWorker(request.key, task)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.failed.connect(self._on_worker_failed)
        self._pending[request.key] = request
        self._workers[request.key] = worker
        logger.info("Start qmlinker request: action={} key={}", request.action, request.key)
        self._thread_pool.start(worker)

    def _with_client(self, func: Callable[[WujiArmQmlinkerClient], object]) -> object:
        """获取后端持久 qmlinker 客户端并执行同步请求。"""

        client = self._client
        if client is None:
            with self._client_lock:
                client = self._client
                if client is None:
                    client = WujiArmQmlinkerClient(self._config)
                    self._client = client
        return func(client)

    def _close_client(self) -> None:
        """关闭后端持有的 qmlinker 客户端。"""

        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()

    def _set_enable_and_readback(
        self,
        client: WujiArmQmlinkerClient,
        device_name: ArmDeviceName,
        enabled: bool,
    ) -> dict[str, object]:
        """设置使能并读取回写状态。"""

        if not client.set_enable(device_name, enabled):
            raise RuntimeError(f"qmlinker set enable failed: {device_name}")
        return {"device_name": device_name, "enabled": client.get_enable(device_name)}

    def _read_joint_values(
        self,
        client: WujiArmQmlinkerClient,
        device_name: ArmDeviceName,
    ) -> dict[str, object]:
        """读取整臂关节状态并转换为 GUI 轴名字典。"""

        joints = client.get_joint_states(device_name)
        axis_names = axis_names_for_device(device_name)
        values = {
            axis_name: float(joint.angle_deg)
            for axis_name, joint in zip(
                axis_names,
                sorted(joints, key=lambda item: item.joint_id),
                strict=False,
            )
        }
        return {"device_name": device_name, "values": values}

    def _set_joint_and_readback(
        self,
        client: WujiArmQmlinkerClient,
        device_name: ArmDeviceName,
        joint_index: int,
        target_angle_deg: float,
    ) -> dict[str, object]:
        """设置单关节目标并读取整臂回写状态。"""

        if not client.set_joint(device_name, joint_index, target_angle_deg):
            raise RuntimeError(f"qmlinker set joint failed: {device_name} j{joint_index}")
        return self._read_joint_values(client, device_name)

    @Slot(str, object)
    def _on_worker_finished(self, key: str, payload: object) -> None:
        """处理 qmlinker worker 成功结果。"""

        request = self._pending.pop(key, None)
        self._workers.pop(key, None)
        if request is None:
            logger.warning("qmlinker worker finished without pending request: key={}", key)
            return

        logger.info("qmlinker request finished: action={} key={}", request.action, key)
        if request.action == "sdk_probe":
            self.sshStateChanged.emit(True, "qmlinker connected")
            return

        if request.action in {"get_enable", "set_enable"}:
            if not isinstance(payload, dict):
                self.requestFailed.emit("qmlinker enable response format error")
                return
            device_name = str(payload.get("device_name") or request.device_name)
            self.enableStateReceived.emit(device_name, bool(payload.get("enabled", False)))
            return

        if request.action in {"get_joints", "set_joint"}:
            if not isinstance(payload, dict) or not isinstance(payload.get("values"), dict):
                self.requestFailed.emit("qmlinker joint response format error")
                return
            self.dofValuesReceived.emit(payload["values"])

    @Slot(str, str)
    def _on_worker_failed(self, key: str, message: str) -> None:
        """处理 qmlinker worker 失败结果。"""

        request = self._pending.pop(key, None)
        self._workers.pop(key, None)
        action = request.action if request is not None else "<unknown>"
        logger.error("qmlinker request failed: action={} key={} message={}", action, key, message)
        if action == "sdk_probe":
            self.sshStateChanged.emit(False, message)
            return
        self.requestFailed.emit(message)

    def _typed_device(self, device_name: str) -> ArmDeviceName:
        """将字符串设备名收窄为机械臂设备字面量。"""

        if device_name == "left_arm":
            return "left_arm"
        if device_name == "right_arm":
            return "right_arm"
        raise ValueError(f"unsupported arm device: {device_name}")


# endregion
