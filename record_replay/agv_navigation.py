"""AGV 导航命令与 runtime info 到位确认。"""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
from typing import Protocol, cast

from loguru import logger


class AgvClient(Protocol):
    """循环服务所需的最小 AGV 客户端接口。"""

    def navigate_to(self, target_name: str, /) -> object:
        """下发站点导航命令。"""

        ...

    def get_base_status(self) -> object:
        """读取 qmlinker AGV 底层状态。"""

        ...

    def stop(self) -> object:
        """停止当前 AGV 导航。"""

        ...


class _AgvBaseStatusProtocol(Protocol):
    """qmlinker GetBaseStatus 响应的最小字段。"""

    navi_status: str


def wait_until_arrived(
    client: AgvClient,
    target: str,
    timeout_s: float,
    poll_s: float,
    stop_event: threading.Event | None = None,
    command_lock: threading.Lock | None = None,
) -> None:
    """导航到目标站点并等待 AGV 从运动状态恢复为空闲状态。

    Parameters
    ----------
    client:
        提供导航命令与运行状态查询的 AGV 客户端。
    target:
        AGV 地图导航点名称。
    timeout_s:
        等待本次导航结束的最长时间，单位 s。
    poll_s:
        查询运行状态的间隔，单位 s。

    Raises
    ------
    TimeoutError
        超时前未依次观察到 ``busy`` 和 ``idel`` 时抛出。

    Notes
    -----
    AGV 协议中 ``busy`` 表示正在导航，``idel`` 表示其余状态。必须先观察到
    ``busy`` 才接受后续的 ``idel``，避免把命令刚下发时尚未切换的空闲状态误判为到位。
    """

    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("检测到停止请求，禁止下发 AGV 导航")
    if command_lock is None:
        client.navigate_to(target)
    else:
        with command_lock:
            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("检测到停止请求，禁止下发 AGV 导航")
            client.navigate_to(target)
    deadline = time.monotonic() + timeout_s
    observed_busy = False
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("检测到停止请求，终止 AGV 到位等待")
        payload = client.get_base_status()
        raw_status = _read_navigation_status(payload)
        logger.info("AGV 导航状态 target={} raw_status={!r}", target, raw_status)
        if raw_status == "busy":
            observed_busy = True
        elif observed_busy and raw_status == "idel":
            logger.success("AGV 导航结束 target={} raw_status={!r}", target, raw_status)
            return
        if stop_event is not None and stop_event.wait(timeout=poll_s):
            raise RuntimeError("检测到停止请求，终止 AGV 到位等待")
        if stop_event is None:
            time.sleep(poll_s)
    raise TimeoutError(f"AGV 导航到位超时：target={target}, observed_busy={observed_busy}")


def stop_navigation(
    client: AgvClient,
    timeout_s: float,
    command_lock: threading.Lock | None = None,
) -> None:
    """显式调用 AGV Stop RPC，在固定超时内检查返回值。"""

    def stop_command() -> object:
        if command_lock is None:
            return client.stop()
        with command_lock:
            return client.stop()

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="record-replay-agv-stop")
    future = executor.submit(stop_command)
    try:
        result = future.result(timeout=timeout_s)
    except TimeoutError as error:
        raise TimeoutError(f"AGV Stop RPC 超时：{timeout_s:.1f}s") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    if result is not True:
        raise RuntimeError(f"AGV Stop RPC 未确认成功：{result!r}")


def _read_navigation_status(payload: object) -> str:
    """从 qmlinker AGV 状态中读取导航状态。"""

    if isinstance(payload, Mapping):
        return str(payload.get("navi_status", "")).strip().lower()
    status = cast(_AgvBaseStatusProtocol, payload)
    return str(status.navi_status).strip().lower()
