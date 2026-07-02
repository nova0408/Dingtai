from __future__ import annotations

from dataclasses import dataclass

from .camera_stream import CameraStreamRuntime, CameraStreamRuntimeConfig
from .ports import DEFAULT_CAMERA_ID, DEFAULT_CAMERA_NAME

# region 数据结构


@dataclass(frozen=True)
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

    camera_id: str = DEFAULT_CAMERA_ID
    "远端相机控制标识。"

    camera_name: str = DEFAULT_CAMERA_NAME
    "项目内逻辑相机名。"

    camera_request_timeout_ms: int = 3000
    "相机控制命令超时，单位 ms。"

    camera_stream_timeout_ms: int = 8000
    "相机数据流接收超时，单位 ms。"


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

    def get_latest_frame(self):
        """返回最新缓存帧。"""

        return self._frame_runtime.get_latest_frame()

    def get_frame_by_id(self, frame_id: int):
        """按帧号查询缓存帧。"""

        return self._frame_runtime.get_frame_by_id(frame_id)

    def resolve_frame(self, frame_id: int):
        """按请求帧号选择可用相机帧。"""

        if int(frame_id) > 0:
            frame = self.get_frame_by_id(int(frame_id))
            if frame is not None:
                return frame
        frame = self.get_latest_frame()
        if frame is None:
            raise RuntimeError("camera frame not ready")
        return frame

# endregion
