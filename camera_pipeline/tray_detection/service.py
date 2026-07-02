from __future__ import annotations

from typing import Optional

from .engine import OrinTrayDetectionExecutor, OrinTrayDetectionExecutorConfig
from .protocol import OrinTrayDetectionRequest, OrinTrayDetectionResponse


class OrinTrayDetectionService:
    """托盘检测纯计算执行器包装。

    职责边界：
    - 只接收已经解析好的单帧图像和请求对象。
    - 不负责相机流、PipelineContext、RPC 监听或信号处理。
    - 只负责托盘检测结果的组装。

    设计思想：
    - 将算法对象与 IO 对象分离，避免服务层持有上下文。
    - 让上层编排器决定 frame 从哪里来，减少子模块耦合。

    生命周期：
    - 不持有硬件资源。
    - 可跨线程复用，但默认仅作为单次请求处理器使用。

    继承关系：
    - 不继承业务基类。
    """

    def __init__(self, executor_config: Optional[OrinTrayDetectionExecutorConfig] = None) -> None:
        self._executor = OrinTrayDetectionExecutor(config=executor_config)

    def compute(self, frame, request: OrinTrayDetectionRequest) -> OrinTrayDetectionResponse:
        """基于输入帧和请求计算托盘检测结果。"""

        return self._executor.compute(frame, request)
