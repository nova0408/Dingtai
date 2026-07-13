from __future__ import annotations

import time
from dataclasses import dataclass, field

from .camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig
from .protocol import RgbdFrameProtocol
from .stable_frame import StableFrameConfig, StableFrameDetector

# region 数据结构


@dataclass(frozen=True, slots=True)
class PipelineContextConfig:
    """总流程上下文配置。

    职责边界：
    - 只保存相机流运行时所需配置。
    - 不负责建立连接，不负责缓存帧，不负责执行算法。

    设计思想：
    - 将相机访问参数收口到上下文层，由上下文统一组装底层 runtime。
    - 让服务入口只处理命令行与配置映射，不直接拼接相机访问细节。

    生命周期：
    - 作为纯配置对象可跨线程传递。
    - 不持有硬件、socket 或队列资源。

    继承关系：
    - 不继承业务基类，仅作为上下文配置数据结构。
    """

    camera_control_port: int = 5570
    "相机控制口端口号，单位 端口号。"

    camera_stream_port: int = 5562
    "相机数据流端口号，单位 端口号。"

    camera_id: str = "LEFT"
    "远端相机控制标识。"

    camera_name: str = "left_hand_camera"
    "项目内逻辑相机名。"

    camera_request_timeout_ms: int = 3000
    "相机控制命令超时，单位 ms。"

    camera_stream_timeout_ms: int = 8000
    "相机数据流接收超时，单位 ms。"

    camera_frame_cache_size: int = 64
    "最近相机帧缓存数量，需要覆盖稳定时间窗中点帧的回取。"

    stable_frame_config: StableFrameConfig = field(default_factory=StableFrameConfig)
    "稳定帧算法配置。"


class PipelineContext:
    """统一管理相机流和帧数据输入输出的上下文。

    职责边界：
    - 负责根据配置组装相机运行时，并提供统一的帧访问入口。
    - 不负责算法本身，不负责服务协议细节，不负责 GUI 展示。

    设计思想：
    - 把相机访问参数固定在上下文层，避免上层脚本直接拼 runtime 配置。
    - 由上下文统一协调启动、等待就绪、关闭资源，减少调用方重复代码。

    生命周期：
    - 可随服务进程长期存在。
    - 持有相机运行时，必须显式调用 `close()` 释放。

    继承关系：
    - 不继承业务基类。
    """

    def __init__(self, config: PipelineContextConfig) -> None:
        self._config = config
        self._frame_runtime = CameraStreamRuntime(
            CameraStreamRuntimeConfig(
                control_port=int(config.camera_control_port),
                stream_port=int(config.camera_stream_port),
                camera_id=str(config.camera_id),
                camera_name=str(config.camera_name),
                request_timeout_ms=int(config.camera_request_timeout_ms),
                stream_timeout_ms=int(config.camera_stream_timeout_ms),
                cache_size=int(config.camera_frame_cache_size),
            )
        )

    def start(self) -> None:
        """启动相机流运行时。"""

        self._frame_runtime.start()

    def close(self) -> None:
        """关闭相机流运行时。"""

        self._frame_runtime.stop()

    def wait_until_ready(self, timeout_s: float = 8.0) -> bool:
        """等待相机首帧就绪。"""

        return self._frame_runtime.wait_until_ready(timeout_s=timeout_s)

    def get_camera_runtime(self) -> CameraStreamRuntime:
        """返回当前相机运行时。"""

        return self._frame_runtime

    def get_camera_id(self) -> str:
        """返回当前相机控制标识。"""

        return str(self._config.camera_id)

    def get_camera_name(self) -> str:
        """返回当前相机逻辑名称。"""

        return str(self._config.camera_name)

    def get_latest_frame(self):
        """返回最新缓存帧。"""

        return self._frame_runtime.get_latest_frame()

    def get_frame_by_id(self, frame_id: int):
        """按帧号查询缓存帧。"""

        return self._frame_runtime.get_frame_by_id(frame_id)

    def resolve_frame(self, frame_id: int):
        """按请求帧号选择相机帧，未指定帧号时默认等待稳定帧。"""

        if frame_id > 0:
            frame = self.get_frame_by_id(frame_id)
            if frame is not None:
                return frame
            raise RuntimeError(f"camera frame {frame_id} is not available")
        return self.wait_for_stable_frame()

    def wait_for_stable_frame(self, timeout_s: float = 10.0) -> RgbdFrameProtocol:
        """等待画面连续稳定，并返回稳定时间窗中点的相机帧。

        Parameters
        ----------
        timeout_s:
            等待稳定帧的最长时间，单位 s。

        Returns
        -------
        RgbdFrameProtocol
            稳定时间窗中点附近的缓存相机帧。

        Raises
        ------
        ValueError
            `timeout_s` 不大于零。
        RuntimeError
            超时前没有形成稳定时间窗，或中点帧已无法从缓存取回。
        """

        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be greater than zero")

        detector = StableFrameDetector(config=self._config.stable_frame_config)
        deadline = time.monotonic() + timeout_s
        last_frame_id = -1
        missing_stable_frame_id: int | None = None
        while time.monotonic() < deadline:
            frame = self.get_latest_frame()
            if frame is None or frame.frame_id == last_frame_id:
                time.sleep(0.01)
                continue

            last_frame_id = frame.frame_id
            stable_frame_id = detector.update(frame)
            if stable_frame_id is None:
                continue
            stable_frame = self.get_frame_by_id(stable_frame_id)
            if stable_frame is not None:
                return stable_frame
            missing_stable_frame_id = stable_frame_id

        if missing_stable_frame_id is not None:
            raise RuntimeError(
                f"stable frame {missing_stable_frame_id} is no longer available in camera frame cache"
            )
        raise RuntimeError(f"camera did not become stable within {timeout_s:.1f}s")


# endregion
