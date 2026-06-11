from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
import time
from threading import Event, Lock, Thread
from typing import Any, cast
from loguru import logger
from PySide6.QtCore import QObject, QThreadPool, Signal
from qmlinker import QMMoveBase

from src.arm.wuji_arm_protocol import (
    SUPPORTED_ARM_DEVICES,
    ArmDeviceName,
    axis_names_for_device,
    parse_arm_axis_name,
    parse_body_axis_name,
    parse_head_axis_name,
)
from src.agv import parse_agv_axis_name
from src.hand import parse_hand_axis_name
from src.wuji.camera_protocol import WujiCameraName, parse_wuji_camera_name
from src.wuji.backend_tasks import _SdkRequest, _SdkWorker
from src.wuji.client_base import WujiQmlinkerBaseClient
from src.wuji.device_clients import WujiQmlinkerClientSet
from src.wuji.protocol import (
    SUPPORTED_WUJI_QMLINKER_ENABLE_MODULES,
    WujiQmlinkerConfig,
    WujiQmlinkerEnableModuleName,
    load_wuji_robot_network_config,
)
from src.wuji.subscription_context import WujiQmlinkerSubscriptionContext
from src.wuji.zmq_camera_client import WujiZmqCameraClient, WujiZmqCameraConfig


class WujiQmlinkerBackend(QObject):
    serviceStateChanged = Signal(bool, str)
    enableStateReceived = Signal(str, bool)
    dofValuesReceived = Signal(dict)
    cameraInventoryReceived = Signal(object)
    cameraEnableStateReceived = Signal(object)
    cameraIntrinsicsReceived = Signal(object)
    cameraFrameReceived = Signal(object)
    requestFailed = Signal(str)

    def __init__(self, parent: QObject | None = None, service_host_alias: str = "base_control", config: WujiQmlinkerConfig | None = None) -> None:
        super().__init__(parent)
        self._service_host_alias = service_host_alias
        self._config = load_wuji_robot_network_config().qmlinker if config is None else config
        self._thread_pool = QThreadPool.globalInstance()
        self._pending: dict[str, _SdkRequest] = {}
        self._workers: dict[str, _SdkWorker] = {}
        self._client: WujiQmlinkerClientSet | None = None
        self._camera_client: WujiZmqCameraClient | None = None
        self._client_lock = Lock()
        self._camera_client_lock = Lock()
        self._subscription_context: WujiQmlinkerSubscriptionContext | None = None
        self._camera_stream_stop_event: Event | None = None
        self._camera_stream_thread: Thread | None = None
        self._camera_stream_name: WujiCameraName | None = None

    def configure_endpoint(self, host: str, port: int | None = None) -> None:
        clean_host = host.strip() or self._config.host
        previous_target = self._config.target()
        self._config = WujiQmlinkerConfig(host=clean_host, port=self._config.port if port is None else int(port), default_speed_ratio=self._config.default_speed_ratio, request_timeout_s=self._config.request_timeout_s, stream_first_timeout_s=self._config.stream_first_timeout_s)
        if previous_target != self._config.target():
            self._close_client()
            logger.info("Configured qmlinker endpoint: target={}", self._config.target())

    def connect_service(self) -> None:
        self._start_worker(_SdkRequest(action="sdk_probe", key="sdk_probe"), lambda: self._with_client(lambda client: client.check_ready()))

    def disconnect_service(self) -> None:
        self._close_client()
        self.serviceStateChanged.emit(False, "qmlinker disconnected")

    def refresh_service_state(self) -> None:
        self.connect_service()

    def snapshot_state_values(self) -> dict[str, float]:
        return {} if self._subscription_context is None else self._subscription_context.snapshot_values()

    def snapshot_enable_states(self) -> dict[str, bool]:
        return {} if self._subscription_context is None else self._subscription_context.snapshot_enable_states()

    def refresh_enable_state(self, device_name: str) -> None:
        if device_name in {"left_arm", "right_arm"}:
            typed_device = self._typed_arm_device(device_name)
            self._start_worker(
                _SdkRequest(action="get_enable", key=f"get_enable:{typed_device}", device_name=typed_device),
                lambda: self._with_client(
                    lambda client: {
                        "device_name": typed_device,
                        "enabled": client.get_enable(typed_device),
                    }
                ),
            )
            return
        if device_name in {"right_hand", "agv"}:
            self._start_worker(
                _SdkRequest(action="get_enable", key=f"get_enable:{device_name}", device_name=device_name),
                lambda: self._with_client(
                    lambda client: {
                        "device_name": device_name,
                        "enabled": client.get_right_hand_enable()
                        if device_name == "right_hand"
                        else client.get_agv_enable(),
                    }
                ),
            )
            return
        if device_name not in SUPPORTED_ARM_DEVICES and device_name not in SUPPORTED_WUJI_QMLINKER_ENABLE_MODULES:
            return
        if device_name in SUPPORTED_ARM_DEVICES:
            typed_device = self._typed_arm_device(device_name)
            self._start_worker(
                _SdkRequest(action="get_enable", key=f"get_enable:{typed_device}", device_name=typed_device),
                lambda: self._with_client(lambda client: {"device_name": typed_device, "enabled": client.get_enable(typed_device)}),
            )
            return
        typed_device = self._typed_module(device_name)
        self._start_worker(
            _SdkRequest(action="get_enable", key=f"get_enable:{typed_device}", device_name=typed_device),
            lambda: self._with_client(lambda client: {"device_name": typed_device, "enabled": client.get_module_enable(typed_device)}),
        )

    def set_enable_state(self, device_name: str, enabled: bool) -> None:
        if device_name == "right_hand":
            self.set_right_hand_enable_state(enabled)
            return
        if device_name == "agv":
            self.set_agv_enable_state(enabled)
            return
        if device_name in {"left_arm", "right_arm"}:
            typed_device = self._typed_arm_device(device_name)
            self._start_worker(
                _SdkRequest(action="set_enable", key=f"set_enable:{typed_device}", device_name=typed_device),
                lambda: self._with_client(
                    lambda client: self._set_arm_enable_and_readback(client, typed_device, enabled)
                ),
            )
            return
        typed_module = self._typed_module(device_name)
        self._start_worker(
            _SdkRequest(action="set_enable", key=f"set_enable:{typed_module}", device_name=typed_module),
            lambda: self._with_client(lambda client: self._set_module_enable_and_readback(client, typed_module, enabled)),
        )

    def refresh_dof_value(self, axis_name: str) -> None:
        parsed_arm = parse_arm_axis_name(axis_name)
        if parsed_arm is not None:
            device_name, _ = parsed_arm
            self._start_joint_refresh(device_name, axis_name, action="get_joints")
            return
        if parse_body_axis_name(axis_name) is not None:
            self._start_worker(_SdkRequest(action="get_body_axis", key=f"get_body_axis:{axis_name}", axis_name=axis_name), lambda: self._with_client(lambda client: self._read_body_axis(client, axis_name)))
            return
        if parse_head_axis_name(axis_name) is not None:
            self._start_worker(_SdkRequest(action="get_head_axis", key=f"get_head_axis:{axis_name}", axis_name=axis_name), lambda: self._with_client(lambda client: self._read_head_axis(client, axis_name)))
            return
        parsed_hand = parse_hand_axis_name(axis_name)
        if parsed_hand is not None:
            device_name, _ = parsed_hand
            self._start_worker(_SdkRequest(action="get_hand_axis", key=f"get_hand_axis:{device_name}", axis_name=axis_name), lambda: self._with_client(lambda client: {"values": client.get_right_hand_values()}))
            return
        if parse_agv_axis_name(axis_name) is not None:
            self._start_worker(_SdkRequest(action="get_agv_status", key="get_agv_status", axis_name=axis_name), lambda: self._with_client(lambda client: {"values": client.get_agv_status_values()}))

    def set_dof_target(self, axis_name: str, target_value: float) -> None:
        parsed_arm = parse_arm_axis_name(axis_name)
        if parsed_arm is not None:
            device_name, joint_index = parsed_arm
            self._start_worker(_SdkRequest(action="set_joint", key=f"set_joint:{device_name}:{joint_index}", axis_name=axis_name, device_name=device_name), lambda: self._with_client(lambda client: self._set_joint_and_readback(client, device_name, joint_index, target_value)))
            return
        if parse_body_axis_name(axis_name) is not None:
            self._start_worker(_SdkRequest(action="set_body_axis", key=f"set_body_axis:{axis_name}", axis_name=axis_name), lambda: self._with_client(lambda client: self._set_body_axis_and_readback(client, axis_name, target_value)))
            return
        if parse_head_axis_name(axis_name) is not None:
            self._start_worker(_SdkRequest(action="set_head_axis", key=f"set_head_axis:{axis_name}", axis_name=axis_name), lambda: self._with_client(lambda client: self._set_head_axis_and_readback(client, axis_name, target_value)))
            return
        parsed_hand = parse_hand_axis_name(axis_name)
        if parsed_hand is not None:
            device_name, actuator_id = parsed_hand
            self._start_worker(_SdkRequest(action="set_hand_axis", key=f"set_hand_axis:{device_name}:{actuator_id}", axis_name=axis_name), lambda: self._with_client(lambda client: self._set_right_hand_axis_and_readback(client, actuator_id, target_value)))
            return
        self.requestFailed.emit(f"当前接口文档未提供 {axis_name} 控制接口")

    def set_right_hand_enable_state(self, enabled: bool) -> None:
        self._start_worker(_SdkRequest(action="set_hand_enable", key="set_hand_enable:right_hand"), lambda: self._with_client(lambda client: {"device_name": "right_hand", "enabled": client.set_right_hand_enable(enabled)}))

    def set_right_hand_demo_pose(self) -> None:
        self._start_worker(
            _SdkRequest(action="set_hand_state", key="set_hand_state:right_hand"),
            lambda: self._with_client(
                lambda client: {
                    "device_name": "right_hand",
                    "enabled": client.set_right_hand_state([0.0 for _ in client.get_right_hand_instance_specs()]),
                }
            ),
        )

    def set_agv_enable_state(self, enabled: bool) -> None:
        self._start_worker(_SdkRequest(action="set_agv_enable", key="set_agv_enable"), lambda: self._with_client(lambda client: {"device_name": "agv", "enabled": client.set_agv_enable(enabled)}))

    def refresh_camera_inventory(self) -> None:
        self._start_worker(
            _SdkRequest(action="get_camera_inventory", key="get_camera_inventory"),
            lambda: self._with_camera_client(lambda client: client.list_camera_runtime_infos(online_only=False)),
        )

    def refresh_camera_intrinsics(self, camera_name: str) -> None:
        typed_camera = self._typed_camera(camera_name)
        self._start_worker(
            _SdkRequest(action="get_camera_intrinsics", key=f"get_camera_intrinsics:{typed_camera}", axis_name=camera_name),
            lambda: self._with_camera_client(lambda client: client.get_camera_intrinsics(typed_camera)),
        )

    def refresh_camera_enable_state(self, camera_name: str) -> None:
        typed_camera = self._typed_camera(camera_name)
        self._start_worker(
            _SdkRequest(action="get_camera_enable", key=f"get_camera_enable:{typed_camera}", axis_name=camera_name),
            lambda: self._with_camera_client(lambda client: client.get_camera_enable_state(typed_camera)),
        )

    def set_camera_enable_state(self, camera_name: str, enabled: bool) -> None:
        typed_camera = self._typed_camera(camera_name)
        self._start_worker(
            _SdkRequest(action="set_camera_enable", key=f"set_camera_enable:{typed_camera}", axis_name=camera_name),
            lambda: self._with_camera_client(lambda client: self._set_camera_enable_and_readback(client, typed_camera, enabled)),
        )

    def start_camera_rgb_stream(self, camera_name: str) -> None:
        self._start_camera_stream(camera_name, "rgb")

    def start_camera_rgbd_stream(self, camera_name: str) -> None:
        self._start_camera_stream(camera_name, "rgbd")

    def agv_translate(self, distance: float, angle_deg: float, speed_ratio: float) -> None:
        self._start_worker(_SdkRequest(action="agv_translate", key="agv_translate"), lambda: self._with_client(lambda client: {"device_name": "agv", "enabled": client.move_agv_translate(float(speed_ratio), int(angle_deg), float(distance))}))

    def agv_rotate(self, angle_deg: float, direction: str, speed_ratio: float) -> None:
        self._start_worker(_SdkRequest(action="agv_rotate", key="agv_rotate"), lambda: self._with_client(lambda client: {"device_name": "agv", "enabled": client.move_agv_rotate(float(angle_deg), QMMoveBase.RotationDirection.LEFT if direction.upper() == "LEFT" else QMMoveBase.RotationDirection.RIGHT, float(speed_ratio))}))

    def agv_stop(self) -> None:
        self._start_worker(_SdkRequest(action="agv_stop", key="agv_stop"), lambda: self._with_client(lambda client: {"device_name": "agv", "enabled": client.stop_agv()}))

    def agv_move_direction(self, direction: str) -> None:
        """按方向执行 AGV 平移动作。"""

        direction_map = {
            "forward": 0,
            "backward": 180,
            "left": 90,
            "right": 270,
        }
        if direction not in direction_map:
            self.requestFailed.emit(f"不支持的 AGV 方向: {direction}")
            return
        self._start_worker(
            _SdkRequest(action="agv_move_direction", key=f"agv_move_direction:{direction}"),
            lambda: self._with_client(
                lambda client: {
                    "device_name": "agv",
                    "direction": direction,
                    "enabled": client.move_agv_real_time_translate(0.3, direction_map[direction]),
                }
            ),
        )

    def agv_navigate_to(self, target_name: str) -> None:
        """发送 AGV 导航目标。"""

        clean_target = target_name.strip()
        if not clean_target:
            self.requestFailed.emit("AGV 导航目标不能为空")
            return
        self._start_worker(
            _SdkRequest(action="agv_navigate_to", key=f"agv_navigate_to:{clean_target}"),
            lambda: self._with_client(lambda client: {"device_name": "agv", "target": clean_target, "result": client.agv_navigate_to(clean_target)}),
        )

    def agv_navigate_to_charge(self) -> None:
        """发送 AGV 去充电导航目标。"""

        self._start_worker(
            _SdkRequest(action="agv_navigate_to_charge", key="agv_navigate_to_charge"),
            lambda: self._with_client(lambda client: {"device_name": "agv", "target": "charge", "result": client.agv_navigate_to_charge()}),
        )

    def get_joint_states(self, device_name: ArmDeviceName) -> object:
        return self._with_client(lambda client: client.get_joint_states(device_name))

    def stop_arm(self, device_name: ArmDeviceName) -> None:
        self._with_client(lambda client: client.stop_arm(device_name))

    def set_joints(self, device_name: ArmDeviceName, joint_angles_deg: Iterable[float], sync_threshold: float = 0.0) -> object:
        joint_angle_list = [float(angle) for angle in joint_angles_deg]
        return self._with_client(lambda client: client.set_joints(device_name, joint_angle_list, int(round(sync_threshold))))

    def stream_get_joint_states(self, device_name: ArmDeviceName, duration_s: float) -> Iterator[object]:
        client = self._client_or_create()
        return client.stream_get_joint_states(device_name, duration_s)

    def stream_set_joint_states(self, device_name: ArmDeviceName, command_frames: Iterable[Iterable[dict[str, float | int]]]) -> object:
        command_frame_list = [[dict(command) for command in frame] for frame in command_frames]
        return self._with_client(lambda client: client.stream_set_joint_states(device_name, command_frame_list))

    def fk(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> object:
        return self._with_client(lambda client: client.fk(device_name, list(joint_angles_rad)))

    def fk_fast(self, device_name: ArmDeviceName, joint_angles_rad: Iterable[float]) -> object:
        return self._with_client(lambda client: client.fk_fast(device_name, list(joint_angles_rad)))

    def ik(self, device_name: ArmDeviceName, target_pose: Any, reference_joint_angles_rad: Iterable[float]) -> object:
        return self._with_client(lambda client: client.ik(device_name, target_pose, list(reference_joint_angles_rad)))

    def current_fk_fast(self, device_name: ArmDeviceName) -> object:
        return self._with_client(lambda client: client.current_fk_fast(device_name))

    def _start_joint_refresh(self, device_name: ArmDeviceName, axis_name: str | None, action: str) -> None:
        self._start_worker(_SdkRequest(action=action, key=f"{action}:{device_name}", axis_name=axis_name, device_name=device_name), lambda: self._with_client(lambda client: self._read_joint_values(client, device_name)))

    def _start_worker(self, request: _SdkRequest, task: Callable[[], object]) -> None:
        if request.key in self._pending:
            return
        worker = _SdkWorker(request.key, task)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.failed.connect(self._on_worker_failed)
        self._pending[request.key] = request
        self._workers[request.key] = worker
        self._thread_pool.start(worker)

    def _with_client(self, func: Callable[[WujiQmlinkerClientSet], object]) -> object:
        client = self._client
        if client is None:
            with self._client_lock:
                client = self._client
                if client is None:
                    client = WujiQmlinkerClientSet(WujiQmlinkerBaseClient(self._config))
                    self._client = client
        return func(client)

    def _client_or_create(self) -> WujiQmlinkerClientSet:
        return cast(WujiQmlinkerClientSet, self._with_client(lambda client: client))

    def _close_client(self) -> None:
        self.stop_camera_stream()
        self._stop_subscription_context()
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()

    def _client_for_subscription(self) -> WujiQmlinkerClientSet:
        return self._client_or_create()

    def _start_subscription_context(self) -> None:
        if self._subscription_context is None:
            self._subscription_context = WujiQmlinkerSubscriptionContext(self._client_for_subscription)
        self._subscription_context.start()

    def _stop_subscription_context(self) -> None:
        context = self._subscription_context
        self._subscription_context = None
        if context is not None:
            context.stop()

    def _set_module_enable_and_readback(self, client: WujiQmlinkerClientSet, module_name: WujiQmlinkerEnableModuleName, enabled: bool) -> dict[str, object]:
        if not client.set_module_enable(module_name, enabled):
            raise RuntimeError(f"qmlinker set enable failed: {module_name}")
        return {"device_name": module_name, "enabled": client.get_module_enable(module_name)}

    def _set_arm_enable_and_readback(
        self,
        client: WujiQmlinkerClientSet,
        device_name: ArmDeviceName,
        enabled: bool,
    ) -> dict[str, object]:
        if not client.set_enable(device_name, enabled):
            raise RuntimeError(f"qmlinker set enable failed: {device_name}")

        last_error: Exception | None = None
        for _ in range(5):
            try:
                actual_enabled = bool(client.get_enable(device_name))
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.05)
                continue
            return {"device_name": device_name, "enabled": actual_enabled}

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"qmlinker get enable failed after set: {device_name}")

    def _set_camera_enable_and_readback(self, client: WujiZmqCameraClient, camera_name: WujiCameraName, enabled: bool) -> object:
        return client.set_camera_color_enabled(camera_name, enabled)

    def _get_camera_enable_payload(self, client: WujiZmqCameraClient, camera_name: WujiCameraName) -> object:
        return client.get_camera_enable_state(camera_name)

    def _start_camera_stream(self, camera_name: str, stream_mode: str) -> None:
        typed_camera = self._typed_camera(camera_name)
        self.stop_camera_stream()
        stop_event = Event()
        self._camera_stream_stop_event = stop_event
        self._camera_stream_name = typed_camera
        thread = Thread(target=self._run_camera_stream, args=(typed_camera, stream_mode, stop_event), name=f"wuji-camera-{typed_camera}-{stream_mode}", daemon=True)
        self._camera_stream_thread = thread
        thread.start()

    def stop_camera_stream(self) -> None:
        event = self._camera_stream_stop_event
        self._camera_stream_stop_event = None
        if event is not None:
            event.set()

    def _run_camera_stream(self, camera_name: WujiCameraName, stream_mode: str, stop_event: Event) -> None:
        try:
            client = self._camera_client_for_stream()
            frames = client.stream_camera_rgb_frames(camera_name) if stream_mode == "rgb" else client.stream_camera_rgbd_frames(camera_name)
            for frame in frames:
                if stop_event.is_set():
                    break
                self.cameraFrameReceived.emit(frame)
        except Exception as exc:  # noqa: BLE001
            if not stop_event.is_set():
                self.requestFailed.emit(f"qmlinker camera stream failed: {exc}")
        finally:
            if stream_mode == "rgbd":
                try:
                    self._with_camera_client(lambda client: client.stop_camera_depth_stream(camera_name))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("qmlinker camera depth stream stop ignored: {}", exc)

    def _read_joint_values(self, client: WujiQmlinkerClientSet, device_name: ArmDeviceName) -> dict[str, object]:
        joints = client.get_joint_states(device_name)
        axis_names = axis_names_for_device(device_name, len(joints))
        values = {axis_name: float(joint.angle_deg) for axis_name, joint in zip(axis_names, sorted(joints, key=lambda item: item.joint_id), strict=False)}
        return {"device_name": device_name, "values": values}

    def _set_joint_and_readback(self, client: WujiQmlinkerClientSet, device_name: ArmDeviceName, joint_index: int, target_angle_deg: float) -> dict[str, object]:
        if not client.set_joint(device_name, joint_index, target_angle_deg):
            raise RuntimeError(f"qmlinker set joint failed: {device_name} j{joint_index}")
        return self._read_joint_values(client, device_name)

    def _read_body_axis(self, client: WujiQmlinkerClientSet, axis_name: str) -> dict[str, object]:
        return {"values": {"body_z": client.get_body_z()}} if axis_name == "body_z" else {"values": {"body_ry": client.get_body_ry()}}

    def _set_body_axis_and_readback(self, client: WujiQmlinkerClientSet, axis_name: str, target_value: float) -> dict[str, object]:
        if axis_name == "body_z":
            if not client.set_body_z(target_value):
                raise RuntimeError("qmlinker set body_z failed")
            return self._read_body_axis(client, axis_name)
        if axis_name == "body_ry":
            if not client.set_body_ry(target_value):
                raise RuntimeError("qmlinker set body_ry failed")
            return self._read_body_axis(client, axis_name)
        raise ValueError(f"unsupported body axis: {axis_name}")

    def _read_head_axis(self, client: WujiQmlinkerClientSet, axis_name: str) -> dict[str, object]:
        return {"values": {"head_yaw": client.get_head_yaw()}}

    def _set_head_axis_and_readback(self, client: WujiQmlinkerClientSet, axis_name: str, target_value: float) -> dict[str, object]:
        if not client.set_head_yaw(target_value):
            raise RuntimeError("qmlinker set head_yaw failed")
        return self._read_head_axis(client, axis_name)

    def _set_right_hand_axis_and_readback(self, client: WujiQmlinkerClientSet, actuator_id: int, target_value: float) -> dict[str, object]:
        if not client.set_right_hand_axis(actuator_id, target_value):
            raise RuntimeError(f"qmlinker set right hand axis failed: right_hand a{actuator_id}")
        return {"values": client.get_right_hand_values()}

    def _on_worker_finished(self, key: str, payload: object) -> None:
        request = self._pending.pop(key, None)
        self._workers.pop(key, None)
        if request is None:
            return
        if request.action == "sdk_probe":
            self._start_subscription_context()
            self.serviceStateChanged.emit(True, "qmlinker connected")
            return
        if request.action in {"get_enable", "set_enable", "set_agv_enable"} and isinstance(payload, dict):
            self.enableStateReceived.emit(str(payload.get("device_name") or request.device_name), bool(payload.get("enabled", False)))
            return
        if request.action in {"get_camera_enable", "set_camera_enable"}:
            self.cameraEnableStateReceived.emit(payload)
            return
        if request.action == "get_camera_inventory":
            self.cameraInventoryReceived.emit(payload)
            return
        if request.action == "get_camera_intrinsics":
            self.cameraIntrinsicsReceived.emit(payload)
            return
        if request.action in {"get_joints", "set_joint", "get_body_axis", "set_body_axis", "get_head_axis", "set_head_axis", "get_hand_axis", "set_hand_axis", "get_agv_status"} and isinstance(payload, dict) and isinstance(payload.get("values"), dict):
            self.dofValuesReceived.emit(payload["values"])

    def _on_worker_failed(self, key: str, message: str) -> None:
        request = self._pending.pop(key, None)
        self._workers.pop(key, None)
        if request is not None and request.action == "sdk_probe":
            self.serviceStateChanged.emit(False, message)
            return
        if request is not None and self._should_suppress_worker_error(request, message):
            logger.debug("Suppress qmlinker worker error: action={} key={} message={}", request.action, request.key, message)
            return
        self.requestFailed.emit(message)

    def _should_suppress_worker_error(self, request: _SdkRequest, message: str) -> bool:
        if "StatusCode.CANCELLED" not in message:
            return False
        return request.action in {"get_enable", "set_enable", "agv_stop"}

    def _typed_arm_device(self, device_name: str) -> ArmDeviceName:
        if device_name == "left_arm":
            return "left_arm"
        if device_name == "right_arm":
            return "right_arm"
        raise ValueError(f"unsupported arm device: {device_name}")

    def _typed_module(self, device_name: str) -> WujiQmlinkerEnableModuleName:
        if device_name in {"body", "head", "left_arm", "right_arm"}:
            return cast(WujiQmlinkerEnableModuleName, device_name)
        raise ValueError(f"unsupported module: {device_name}")

    def _typed_camera(self, camera_name: str) -> WujiCameraName:
        parsed = parse_wuji_camera_name(camera_name)
        if parsed is None:
            raise ValueError(f"unsupported camera: {camera_name}")
        return parsed

    def _with_camera_client(self, func: Callable[[WujiZmqCameraClient], object]) -> object:
        client = self._camera_client
        if client is None:
            with self._camera_client_lock:
                client = self._camera_client
                if client is None:
                    client = WujiZmqCameraClient(WujiZmqCameraConfig(host=self._config.host, request_timeout_ms=max(500, int(self._config.request_timeout_s * 1000.0)), stream_timeout_ms=max(500, int(self._config.stream_first_timeout_s * 1000.0))))
                    self._camera_client = client
        return func(client)

    def _camera_client_for_stream(self) -> WujiZmqCameraClient:
        client = self._with_camera_client(lambda client: client)
        if not isinstance(client, WujiZmqCameraClient):
            raise TypeError("ZMQ camera client type error")
        return client


__all__ = ["WujiQmlinkerBackend", "WujiQmlinkerSubscriptionContext"]
