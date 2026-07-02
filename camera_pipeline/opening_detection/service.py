from __future__ import annotations

from typing import Optional

from .engine import OpeningDetectionPipelineExecutor, OpeningDetectionPipelineExecutorConfig


class OpeningDetectionPipelineService:
    """开口检测纯计算执行器包装。

    职责边界：
    - 只接收单帧 RGBD、托盘掩码和请求字段。
    - 不负责相机流、PipelineContext、RPC 监听或托盘请求。
    - 只负责开口检测与抓取位姿求解。

    设计思想：
    - 保持算法层只关心输入数据和输出数据。
    - 把请求编排留给上层主循环，减少子模块职责外溢。

    生命周期：
    - 不持有硬件资源。
    - 可跨线程复用，但默认仅作为单次请求处理器使用。

    继承关系：
    - 不继承业务基类。
    """

    def __init__(self, executor_config: Optional[OpeningDetectionPipelineExecutorConfig] = None) -> None:
        self._executor = OpeningDetectionPipelineExecutor(config=executor_config)

    def compute(
        self,
        frame,
        tray_mask,
        request_id: int,
        target_tray_index: int,
        enable_debug: bool = True,
    ) -> tuple[object, object | None]:
        """基于输入帧和托盘掩码计算开口与位姿结果。"""

        return self._executor.compute(
            frame=frame,
            tray_mask=tray_mask,
            request_id=int(request_id),
            target_tray_index=int(target_tray_index),
            enable_debug=bool(enable_debug),
        )
