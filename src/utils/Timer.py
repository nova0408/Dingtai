import asyncio
import atexit
import ctypes
import heapq
import inspect
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from loguru import logger


class TimerState(Enum):
    """计时器状态"""

    IDLE = "idle"  # 空闲
    RUNNING = "running"  # 运行中
    PAUSED = "paused"  # 暂停
    STOPPED = "stopped"  # 已停止


# 定义 Windows API
winmm = ctypes.WinDLL("winmm")


class HighPrecisionTimer:
    """
    高精度计时器，支持启动/停止、暂停/继续、定时任务、多线程
    使用 time.perf_counter() 实现纳秒级精度，使用 Condition 优化并发调度
    """

    # 全局引用计数控制 Windows 时钟精度
    _active_timers_count = 0
    _global_lock = threading.Lock()
    _executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="TimerWorker")

    def __init__(self, name: str = "Timer", auto_start: bool = False):
        self.name = name
        self._state = TimerState.IDLE

        # --- 核心并发控制 ---
        # Condition 内部包含一把 RLock，替代原有的 self._lock
        # 所有涉及共享状态读写的地方都必须使用 with self._condition
        # 在 Windows 默认状态下，系统的**时间片（Clock Interrupt Rate）**通常是 15.625 ms
        self._condition = threading.Condition(threading.RLock())

        # 计时相关属性
        self._start_time: float | None = None
        self._pause_time: float | None = None
        self._anchor_time: float = -99999.0
        self._total_elapsed: float = 0.0
        self._lap_times: list[float] = []

        # 定时任务相关
        # 堆元素结构：(exec_time, task_id, func, args, kwargs)
        self._scheduled_tasks: list[tuple[float, int, Callable, tuple, dict]] = []
        self._task_counter: int = 0

        # [新增] 任务代数：每次 reset 自增，用于废弃旧周期的任务
        self._task_generation: int = 0
        # 追踪被取消的任务 ID，用于处理周期性任务和延迟删除
        self._cancelled_ids: set[int] = set()

        self._scheduler_thread: threading.Thread | None = None
        self._stop_scheduler_flag = False  # 替换 Event，配合 Condition 使用更简单

        self._timer_resolution_active = False

        # 自动清理
        atexit.register(self.stop_all)

        if auto_start:
            self.start()

    def __del__(self):
        # 手动析构，确保如果对象在程序运行期间被销毁，时钟频率能及时降回去
        try:
            self.stop_all()
        except:
            pass

    # ==========================
    # 基础计时功能 (使用 Condition 锁)
    # ==========================

    def _set_high_precision(self, enable: bool):
        if os.name != "nt":
            return

        with HighPrecisionTimer._global_lock:
            if enable:
                if HighPrecisionTimer._active_timers_count == 0:
                    winmm.timeBeginPeriod(1)
                HighPrecisionTimer._active_timers_count += 1
                self._timer_resolution_active = True
            else:
                if self._timer_resolution_active:
                    HighPrecisionTimer._active_timers_count -= 1
                    if HighPrecisionTimer._active_timers_count == 0:
                        winmm.timeEndPeriod(1)
                    self._timer_resolution_active = False

    def start(self) -> bool:
        """开始计时"""
        with self._condition:
            if self._state in [TimerState.RUNNING, TimerState.PAUSED]:
                return False
            self._set_high_precision(True)  # 启动时提升精度

            # 1. 锁定基准时间
            now = time.perf_counter()
            self._start_time = now  # 用于 elapsed 计算
            self._anchor_time = now  # 用于任务调度基准 (关键!)

            # 2. 修正堆中所有预先添加的任务的绝对执行时间
            #    注意：这里需要重新构建堆
            new_heap = []
            for offset, task_id, func, args, kwargs in self._scheduled_tasks:
                abs_exec_time = self._anchor_time + offset
                heapq.heappush(new_heap, (abs_exec_time, task_id, func, args, kwargs))
            self._scheduled_tasks = new_heap

            # 3. 启动调度线程
            self._start_scheduler()
            self._condition.notify_all()
            return True

    def stop(self) -> float:
        """停止计时并返回总耗时"""
        if self._start_time is None:
            return 0.0

        with self._condition:
            self._set_high_precision(False)  # 停止时务必恢复，否则增加系统功耗
            if self._state == TimerState.STOPPED:
                return self._total_elapsed

            if self._state == TimerState.RUNNING:
                self._total_elapsed += time.perf_counter() - self._start_time

            self._state = TimerState.STOPPED
            self._start_time = None
            return self._total_elapsed

    def pause(self) -> float:
        """暂停计时"""
        if self._start_time is None:
            return 0.0

        with self._condition:
            if self._state != TimerState.RUNNING:
                return self._total_elapsed

            self._pause_time = time.perf_counter()
            self._total_elapsed += self._pause_time - self._start_time
            self._state = TimerState.PAUSED
            return self._total_elapsed

    def resume(self) -> bool:
        """继续计时"""
        with self._condition:
            if self._state != TimerState.PAUSED or self._pause_time is None or self._start_time is None:
                return False

            now = time.perf_counter()
            pause_duration = now - self._pause_time

            # 同时修正开始时间和锚点时间
            self._start_time += pause_duration
            self._anchor_time += pause_duration

            # 同时需要修正堆中所有任务的绝对执行时间
            new_heap = []
            for exec_time, task_id, func, args, kwargs in self._scheduled_tasks:
                new_heap.append((exec_time + pause_duration, task_id, func, args, kwargs))
            heapq.heapify(new_heap)
            self._scheduled_tasks = new_heap

            self._pause_time = None
            self._state = TimerState.RUNNING
            self._condition.notify_all()
            return True

    def reset(self) -> None:
        """重置计时器：清除状态、取消所有任务、废弃正在运行的周期任务"""
        with self._condition:
            # 1. 基础计时状态重置
            self._state = TimerState.IDLE
            self._start_time = None
            self._pause_time = None
            self._total_elapsed = 0.0
            self._lap_times = []
            self._anchor_time = -9999.0

            # 2. 清除所有等待中的任务
            self._scheduled_tasks.clear()

            # 3. 清除取消列表（因为任务都没了，没必要记录谁被取消了）
            self._cancelled_ids.clear()

            # 4. 更新任务代数
            self._task_generation += 1

            # 5. 唤醒调度器
            # 调度器线程会被唤醒，发现 _scheduled_tasks 为空，于是进入无限 wait() 状态
            self._condition.notify_all()

    def lap(self) -> float | None:
        """记录一个分段计时点"""
        with self._condition:
            if self._state != TimerState.RUNNING:
                return None

            current = self.elapsed
            if self._lap_times:
                lap_time = current - self._lap_times[-1]
            else:
                lap_time = current

            self._lap_times.append(current)
            return lap_time

    @property
    def elapsed(self) -> float:
        """获取当前已耗时"""
        if self._start_time is None:
            return 0.0

        with self._condition:
            if self._state == TimerState.IDLE:
                return 0.0
            elif self._state == TimerState.STOPPED:
                return self._total_elapsed
            elif self._state == TimerState.PAUSED:
                return self._total_elapsed
            else:  # RUNNING
                return self._total_elapsed + (time.perf_counter() - self._start_time)

    @property
    def state(self) -> TimerState:
        return self._state

    def get_lap_times(self) -> list[float]:
        with self._condition:
            return self._lap_times.copy()

    def format_time(self, seconds: float | None) -> str:
        """静态工具方法，不需要锁"""
        if seconds is None:
            return "N/A"
        if seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"
        elif seconds < 1e-3:
            return f"{seconds * 1e6:.2f} µs"
        elif seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        else:
            return f"{seconds:.6f} s"

    # ==========================
    # 定时任务功能 (重构重点)
    # ==========================

    def _add_task_to_heap(self, exec_time: float, task_id: int, func: Callable, args: tuple, kwargs: dict):
        """内部辅助方法：将任务推入堆并唤醒调度器"""
        # 注意：此方法必须在 with self._condition 块中调用
        heapq.heappush(self._scheduled_tasks, (exec_time, task_id, func, args, kwargs))
        self._condition.notify()  # 关键：唤醒可能正在 wait 的调度器

    def schedule(self, offset: float, func: Callable, *args, **kwargs) -> int:
        """
        安排任务在 start() 之后的第 offset 秒执行
        如果 timer 已经在运行，则相对于当前 anchor_time 计算
        """
        with self._condition:
            self._task_counter += 1
            task_id = self._task_counter

            if self._state == TimerState.RUNNING:
                # 如果已经在运行，直接计算绝对时间推入堆
                exec_time = self._anchor_time + offset
                self._add_task_to_heap(exec_time, task_id, func, args, kwargs)
            else:
                # 如果还没启动，先暂存 offset，等 start() 时再修正
                # 复用 _scheduled_tasks 列表，暂时存 (offset, ...)
                # 注意：这里存的是 offset，不是绝对时间！
                heapq.heappush(self._scheduled_tasks, (offset, task_id, func, args, kwargs))

            return task_id

    def schedule_interval(self, interval: float, func: Callable, *args, **kwargs) -> int:
        """安排周期性任务（支持 reset 自动废弃）"""
        with self._condition:
            self._task_counter += 1
            task_id = self._task_counter

            # [新增] 捕获当前代数
            current_gen = self._task_generation

            next_exec_time = time.perf_counter() + interval

            def wrapper(scheduled_time: float):
                # 1. 检查有效性
                with self._condition:
                    # 如果已被取消，或者代数不匹配（说明发生过 reset），则停止
                    if task_id in self._cancelled_ids or self._task_generation != current_gen:
                        return

                # 2. 执行用户函数 (捕获异常防止线程退出)
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"[{self.name}] 周期任务 {task_id} 执行出错：{e}")

                # 3. 安排下一次执行
                with self._condition:
                    # 再次检查：执行期间是否发生了 reset？
                    if task_id in self._cancelled_ids or self._task_generation != current_gen:
                        return

                    new_scheduled_time = scheduled_time + interval
                    now = time.perf_counter()
                    if new_scheduled_time < now:
                        new_scheduled_time = now + interval

                    self._add_task_to_heap(new_scheduled_time, task_id, lambda: wrapper(new_scheduled_time), (), {})

            # 启动第一次
            self._add_task_to_heap(next_exec_time, task_id, lambda: wrapper(next_exec_time), (), {})

            if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
                self._start_scheduler()

            return task_id

    def cancel_task(self, task_id: int) -> bool:
        """取消任务"""
        with self._condition:
            self._cancelled_ids.add(task_id)

            # 尝试从堆中移除（可选，主要是为了立即释放堆空间）
            # 对于一次性任务，这里移除有效；对于周期任务，wrapper 执行时会检查 _cancelled_ids
            original_len = len(self._scheduled_tasks)
            self._scheduled_tasks = [t for t in self._scheduled_tasks if t[1] != task_id]

            if len(self._scheduled_tasks) < original_len:
                heapq.heapify(self._scheduled_tasks)
                self._condition.notify()  # 任务变动，唤醒调度器（可能不需要睡那么久了）
                return True

            # 如果不在堆中（可能正在运行或等待重排的周期任务），标记 ID 即可
            return True

    def _start_scheduler(self):
        """启动调度器线程"""
        self._stop_scheduler_flag = False
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True, name=f"{self.name}_Scheduler"
        )
        self._scheduler_thread.start()

    def _scheduler_loop(self):
        """
        重构后的调度循环：使用 Condition.wait()
        """
        while True:
            with self._condition:
                # ### 改进 3: 如果暂停，则无限等待直到 resume 调用 notify
                while self._state == TimerState.PAUSED and not self._stop_scheduler_flag:
                    self._condition.wait()

                if self._stop_scheduler_flag:
                    break

                if not self._scheduled_tasks:
                    # 没有任务，释放锁并无限等待，直到被 notify()
                    self._condition.wait()
                    continue

                # 查看堆顶任务（不取出）
                exec_time, task_id, func, args, kwargs = self._scheduled_tasks[0]
                now = time.perf_counter()

                if task_id in self._cancelled_ids:
                    # 延迟清理：如果是已取消的任务，直接弹出丢弃
                    heapq.heappop(self._scheduled_tasks)
                    # 从取消集合中移除以防止集合无限增长（假设 ID 不复用）
                    self._cancelled_ids.discard(task_id)
                    continue

                if exec_time <= now:
                    # 时间到！弹出任务
                    heapq.heappop(self._scheduled_tasks)
                    # 使用线程池执行，而不是新建线程
                    # 使用线程执行以防止长任务阻塞后续短任务
                    HighPrecisionTimer._executor.submit(self._execute_task, func, args, kwargs)
                else:
                    # 时间未到，释放锁并等待指定时长
                    # wait() 会在超时或收到 notify() 时返回
                    # 这允许我们在等待期间随时插入更早的任务
                    sleep_time = exec_time - now
                    # 只有在 RUNNING 状态下才限时等待
                    self._condition.wait(timeout=sleep_time)

    def _execute_task(self, func: Callable, args: tuple, kwargs: dict):
        """执行任务（支持同步和异步函数）"""
        try:
            # 如果是异步函数，获取或创建事件循环
            if inspect.iscoroutinefunction(func):
                # 如果是异步函数，获取或创建事件循环
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.run_coroutine_threadsafe(func(*args, **kwargs), loop)
                except RuntimeError:
                    asyncio.run(func(*args, **kwargs))
            else:
                # 执行普通函数
                res = func(*args, **kwargs)
                # 如果普通函数返回了一个协程对象 (例如被 partial 包裹的异步函数)
                if inspect.iscoroutine(res):
                    asyncio.run(res)

        except Exception as e:
            logger.error(f"[{self.name}] 任务执行失败：{e}")

    def stop_all(self):
        """停止所有任务和计时"""
        self.stop()
        with self._condition:
            self._stop_scheduler_flag = True
            self._scheduled_tasks.clear()
            self._condition.notify_all()  # 唤醒调度线程让其退出

        if self._scheduler_thread and self._scheduler_thread.is_alive():
            try:
                # 避免在 atexit 中无限等待（如果线程卡死）
                self._scheduler_thread.join(timeout=1.0)
            except RuntimeError:
                pass  # 防止主线程已退出时 join 报错

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self.stop()
        # 注意：如果是多线程共享同一个 Timer 实例，这里会导致其他线程的计时也被停止
        # 但符合上下文管理器的常规语义
        logger.debug(f"[{self.name}] 总耗时：{self.format_time(elapsed)}")
        return False

    def __repr__(self) -> str:
        return f"<HighPrecisionTimer {self.name}, state={self.state.value}, elapsed={self.format_time(self.elapsed)}>"


# ==========================
# 测试代码
# ==========================
def demo_usage_fixed():
    print("=== 1. 基本计时与 Condition 测试 ===")
    timer = HighPrecisionTimer("DemoTimer")
    timer.start()
    time.sleep(0.1)
    print(f"耗时：{timer.format_time(timer.elapsed)}")
    timer.lap()
    time.sleep(0.1)
    print(f"LAP: {timer.format_time(timer.lap())}")
    timer.stop()

    print("\n=== 2. 定时任务取消测试 (修复版) ===")
    timer2 = HighPrecisionTimer("TaskTimer")

    def print_msg(msg):
        print(f"[{time.perf_counter():.3f}] {msg}")

    # 安排周期任务：每 0.2 秒一次
    print(f"[{time.perf_counter():.3f}] 开始安排任务")
    task_id = timer2.schedule_interval(0.2, print_msg, "周期任务 (应执行 3 次)")

    # 让它运行 0.7 秒 (应该执行 T+0.2, T+0.4, T+0.6)
    time.sleep(0.7)

    print(f"[{time.perf_counter():.3f}] 取消任务 {task_id}")
    timer2.cancel_task(task_id)

    # 再等待一段时间，验证是否真的停止了
    time.sleep(0.5)
    print(f"[{time.perf_counter():.3f}] 结束等待，如果不出现新的打印则修复成功")

    timer2.stop_all()

    print("\n=== 3. 并发插入测试 (验证 notify) ===")
    timer3 = HighPrecisionTimer("NotifyTimer")

    # 场景：先安排一个很久以后的任务，调度器会进入长睡眠
    timer3.schedule(10.0, print_msg, "这个任务很晚才执行")

    time.sleep(0.1)  # 确保调度器已经进入 wait(10.0)

    print(f"[{time.perf_counter():.3f}] 插入一个立即执行的任务")
    # 插入一个只需 0.5 秒的任务，Condition.notify 应该唤醒调度器重新计算 wait 时间
    timer3.schedule(0.5, print_msg, "我是插队的任务！")

    time.sleep(1.0)
    timer3.stop_all()


if __name__ == "__main__":
    demo_usage_fixed()
