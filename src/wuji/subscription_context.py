from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import cast

import grpc
from loguru import logger

from src.arm.wuji_arm_protocol import ArmDeviceName, axis_names_for_device
from src.wuji.device_clients import WujiQmlinkerClientSet


class WujiQmlinkerSubscriptionContext:
    _UNARY_REFRESH_INTERVAL_S = 0.05
    _HAND_REFRESH_INTERVAL_S = 0.5
    _UNARY_RPC_RETRY_INTERVAL_S = 5.0
    _SKIPPABLE_RPC_CODES = {
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNIMPLEMENTED,
        grpc.StatusCode.UNAVAILABLE,
    }

    def __init__(self, client_getter: Callable[[], WujiQmlinkerClientSet]) -> None:
        self._client_getter = client_getter
        self._stop_event = Event()
        self._lock = Lock()
        self._threads: list[Thread] = []
        self._values: dict[str, float] = {}
        self._enable_states: dict[str, bool] = {}
        self._unary_rpc_retry_at: dict[str, float] = {}

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        self._start_thread("qmlinker-arm-refresh", self._run_arm_refresh)
        self._start_thread("qmlinker-right-hand-refresh", self._run_right_hand_refresh)
        self._start_thread("qmlinker-unary-state-refresh", self._run_unary_refresh)

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=0.5)
        self._threads.clear()

    def snapshot_values(self) -> dict[str, float]:
        with self._lock:
            return dict(self._values)

    def snapshot_enable_states(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._enable_states)

    def _start_thread(self, name: str, target: Callable[[], None]) -> None:
        thread = Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _update_values(self, values: dict[str, float]) -> None:
        with self._lock:
            self._values.update(values)

    def _update_enable_states(self, states: dict[str, bool]) -> None:
        with self._lock:
            self._enable_states.update(states)

    def _run_arm_refresh(self) -> None:
        while not self._stop_event.is_set():
            try:
                client = self._client_getter()
                for device_name in ("left_arm", "right_arm"):
                    joints = client.get_joint_states(device_name)
                    axis_names = axis_names_for_device(device_name, len(joints))
                    values = {
                        axis_name: float(joint.angle_deg)
                        for axis_name, joint in zip(
                            axis_names,
                            sorted(joints, key=lambda item: getattr(item, "joint_id", 0)),
                            strict=False,
                        )
                    }
                    self._update_values(values)
            except Exception as exc:  # noqa: BLE001
                logger.error("qmlinker arm refresh failed: {}", exc)
                self._stop_event.wait(self._UNARY_REFRESH_INTERVAL_S)

    def _run_right_hand_refresh(self) -> None:
        while not self._stop_event.is_set():
            try:
                client = self._client_getter()
                self._update_values(client.get_right_hand_values())
            except Exception as exc:  # noqa: BLE001
                logger.error("qmlinker right hand refresh failed: {}", exc)
            self._stop_event.wait(self._HAND_REFRESH_INTERVAL_S)

    def _run_unary_refresh(self) -> None:
        while not self._stop_event.is_set():
            client = self._client_getter()
            self._refresh_unary_values(client)
            self._refresh_unary_enable_states(client)
            self._stop_event.wait(self._UNARY_REFRESH_INTERVAL_S)

    def _refresh_unary_values(self, client: WujiQmlinkerClientSet) -> None:
        values: dict[str, float] = {}
        for key, reader in (("body_z", client.get_body_z), ("body_ry", client.get_body_ry), ("head_yaw", client.get_head_yaw)):
            value = self._read_unary_item(f"value.{key}", reader)
            if value is not None:
                values[key] = cast(float, value)
        agv_values = self._read_unary_item("value.agv_status", client.get_agv_status_values)
        if isinstance(agv_values, dict):
            for axis_name, axis_value in agv_values.items():
                if isinstance(axis_value, (int, float)):
                    axis_value_f = cast(float, axis_value)
                    values[axis_name] = axis_value_f
        if values:
            self._update_values(values)

    def _refresh_unary_enable_states(self, client: WujiQmlinkerClientSet) -> None:
        states: dict[str, bool] = {}
        for key, reader in (
            ("body", lambda: client.get_module_enable("body")),
            ("head", lambda: client.get_module_enable("head")),
            ("right_hand", lambda: client.get_right_hand_enable()),
            ("agv", lambda: client.get_agv_enable()),
        ):
            value = self._read_unary_item(f"enable.{key}", reader)
            if value is not None:
                states[key] = bool(value)
        if states:
            self._update_enable_states(states)

    def _read_unary_item(self, key: str, reader: Callable[[], object]) -> object | None:
        now_s = __import__("time").monotonic()
        retry_at_s = self._unary_rpc_retry_at.get(key, 0.0)
        if now_s < retry_at_s:
            return None
        try:
            return reader()
        except grpc.RpcError as exc:
            code = exc.code()
            if code in self._SKIPPABLE_RPC_CODES:
                self._unary_rpc_retry_at[key] = now_s + self._UNARY_RPC_RETRY_INTERVAL_S
                logger.warning("qmlinker unary state item unavailable: key={} code={}", key, code.name)
                return None
            logger.error("qmlinker unary state item failed: key={} error={}", key, exc)
            return None
        except Exception as exc:  # noqa: BLE001
            self._unary_rpc_retry_at[key] = now_s + self._UNARY_RPC_RETRY_INTERVAL_S
            logger.warning("qmlinker unary state item unavailable: key={} error={}", key, exc)
            return None
