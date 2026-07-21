"""双臂自动回放的总运行上下文。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .contracts import ReplayServiceState, ReplayStatusSnapshot
from .settings import ReplayCycleConfig
from .settings import ReplayServiceSettings

if TYPE_CHECKING:
    from .runtime import ReplayRuntime


@dataclass(slots=True)
class ReplayRuntimeResources:
    """一轮回放创建的设备资源。"""

    left_runtime: ReplayRuntime | None = None
    "左臂运行时资源，由设备网关创建。"
    right_runtime: ReplayRuntime | None = None
    "右臂运行时资源，仅同步 CSV 存在时创建。"


class ReplayContext:
    """自动回放的唯一跨模块资源与状态边界。

    该类管理配置、左右运行时资源、共享停止事件及状态快照；不解析 CSV、不做
    运动计算、不发 AGV 指令。执行器负责写入资源和状态，服务及外部查询方只读
    快照，从而避免 CSV、设备和服务模块相互穿透。
    """

    # region 初始化

    def __init__(self, config: ReplayCycleConfig) -> None:
        """创建一轮或多轮可复用的执行上下文。"""

        self.config = config
        self.resources = ReplayRuntimeResources()
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = ReplayStatusSnapshot(ReplayServiceState.WAITING, None, None, None)

    # endregion

    # region 状态

    def snapshot(self) -> ReplayStatusSnapshot:
        """返回当前不可变状态快照。"""

        with self._lock:
            return self._snapshot

    def set_state(
        self,
        state: ReplayServiceState,
        *,
        left_csv_state: str | None = None,
        plan_index: int | None = None,
        error_text: str | None = None,
    ) -> None:
        """原子更新服务阶段与当前左臂执行状态。"""

        with self._lock:
            self._snapshot = ReplayStatusSnapshot(state, left_csv_state, plan_index, error_text)

    def reset_for_next_cycle(self) -> None:
        """清除停止信号和上一轮状态，准备接收下一轮指令。"""

        self.stop_event.clear()
        self.set_state(ReplayServiceState.WAITING)

    def update_settings(self, settings: ReplayServiceSettings) -> None:
        """仅在未执行硬件任务时替换后续轮次使用的运行参数。"""

        with self._lock:
            if self._snapshot.state not in (ReplayServiceState.WAITING, ReplayServiceState.FAILED):
                raise RuntimeError("回放正在执行，不能修改运行参数")
            self.config = replace(self.config, settings=settings)

    def attach_runtimes(self, left_runtime: ReplayRuntime, right_runtime: ReplayRuntime | None) -> None:
        """登记当前轮次已创建的左右运行时资源。"""

        self.resources.left_runtime = left_runtime
        self.resources.right_runtime = right_runtime

    def detach_runtimes(self) -> tuple[ReplayRuntime | None, ReplayRuntime | None]:
        """取走当前 runtime 引用，避免下一轮复用已经关闭的资源。"""

        left_runtime = self.resources.left_runtime
        right_runtime = self.resources.right_runtime
        self.resources.left_runtime = None
        self.resources.right_runtime = None
        return left_runtime, right_runtime

    # endregion
