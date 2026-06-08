from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# region 数据结构

WujiCameraName = Literal["head_camera", "chest_camera", "left_hand_camera", "right_hand_camera"]


@dataclass(frozen=True, slots=True)
class WujiCameraSpec:
    """无际相机实例规格。

    职责边界：
    - 只描述 GUI 与 qmlinker 之间共享的相机名称和显示标题。
    - 不持有 SDK 对象、图像帧、网络连接或 GUI 控件。

    设计思想：
    - 使用固定字面量名称，避免 GUI 下拉框直接依赖 qmlinker proto 枚举值。
    - 保留 title 字段，便于后续把不同规格或安装位相机显示为用户可读文本。

    生命周期：
    - 模块加载时构造，可长期跨线程只读复用。

    继承关系：
    - 不继承业务基类，作为相机配置数据使用。
    """

    name: WujiCameraName
    "相机逻辑名称，供 GUI 与后端传递。"

    title: str
    "相机显示名称，供 GUI 下拉框展示。"


@dataclass(frozen=True, slots=True)
class WujiCameraRuntimeInfo:
    """无际现场相机运行时清单项。

    职责边界：
    - 只表达远端相机服务当前暴露的一路在线相机槽位信息。
    - 不负责 SSH 读取、ZMQ 通信、图像拉流或 GUI 控件更新。

    设计思想：
    - 将逻辑相机名、远端槽位名与序列号放在同一结构中，避免 GUI 再做物理位置猜测。
    - 保留在线与彩色/深度开关状态，便于界面直接展示“当前这一路是否真实可用”。
    - 使用不可变 dataclass，确保后台线程查询结果可安全发回 GUI 线程。

    生命周期：
    - 每次远端清单刷新时构造；不持有外部资源。

    继承关系：
    - 不继承业务基类，作为相机运行时元数据使用。
    """

    camera_name: WujiCameraName
    "项目内逻辑相机名，例如 `head_camera`。"

    camera_id: str
    "远端 ZMQ 控制槽位名，例如 `HEAD`、`CHEST`、`LEFT`、`RIGHT`。"

    serial_number: str
    "远端配置文件给出的可复制序列号。"

    display_name: str
    "供 GUI 展示的组合名称，通常包含槽位名与序列号。"

    online: bool
    "远端相机服务报告的在线状态。"

    color_enabled: bool
    "远端彩色流开关状态。"

    depth_enabled: bool
    "远端深度流开关状态。"


@dataclass(frozen=True, slots=True)
class WujiCameraIntrinsicsInfo:
    """无际 qmlinker 相机内参与基准分辨率。

    职责边界：
    - 只保存 `GetCameraIntrinsics` 返回的相机参数。
    - 不负责畸变校正、深度图解码、图像展示或 SDK 请求。

    设计思想：
    - 把内参、畸变和分辨率作为一个整体传递，避免 GUI 层散收无语义浮点数。
    - 使用不可变 dataclass，便于从 worker 线程传回 GUI 线程。

    生命周期：
    - 每次切换相机或手动刷新时构造，不持有外部资源。

    继承关系：
    - 不继承业务基类，作为 qmlinker 相机协议结果数据使用。
    """

    camera_name: WujiCameraName
    "相机逻辑名称。"

    fx: float
    "X 方向焦距，单位 像素。"

    fy: float
    "Y 方向焦距，单位 像素。"

    cx: float
    "主点 X 坐标，单位 像素。"

    cy: float
    "主点 Y 坐标，单位 像素。"

    distortion: tuple[float, ...]
    "畸变系数序列，通常为 k1、k2、p1、p2、k3。"

    width: int
    "基准图像宽度，单位 像素。"

    height: int
    "基准图像高度，单位 像素。"


@dataclass(frozen=True, slots=True)
class WujiCameraEnableState:
    """无际 qmlinker 相机使能状态结果。

    职责边界：
    - 只表达一次相机使能状态读取或设置回写的结果。
    - 不负责实际 RPC 调用、GUI 渲染或错误弹窗。

    设计思想：
    - 将“当前使能值”和“接口是否可用”拆开表达，避免把 `UNIMPLEMENTED` 误显示成已禁用。
    - 保留 message 文本，便于 GUI 和测试脚本给出可读诊断。

    生命周期：
    - 由后端 worker 在一次请求完成后构造，不持有外部资源。

    继承关系：
    - 不继承业务基类，作为相机状态协议结果数据使用。
    """

    camera_name: WujiCameraName
    "相机逻辑名称。"

    enabled: bool
    "当前使能值；当接口不可用时仅作占位值使用。"

    api_available: bool
    "相机使能接口是否由当前 qmlinker 服务实现。"

    message: str = ""
    "附加诊断信息，例如未实现原因或回写摘要。"


@dataclass(frozen=True, slots=True)
class WujiCameraFrame:
    """无际 qmlinker 相机单帧图像数据。

    职责边界：
    - 只保存 qmlinker 相机流返回的单帧 RGB 或 RGBD 数据。
    - 不负责启动流、停止流、图像控件渲染或文件保存。

    设计思想：
    - BGR 图像保持 OpenCV/qmlinker wrapper 的原始通道顺序，由 GUI 渲染层按 Qt 格式转换。
    - depth 允许为 None，用同一结构表达纯 2D 流和 RGBD 流。

    生命周期：
    - 由相机流线程构造，经 Qt signal 传回 GUI 线程后即可丢弃。

    继承关系：
    - 不继承业务基类，作为跨线程帧数据使用。

    线程/异步语义：
    - frame 内的数组在构造时复制，避免 SDK 缓冲区复用导致 GUI 读到变化数据。
    """

    camera_name: WujiCameraName
    "相机逻辑名称。"

    color_bgr: np.ndarray
    "彩色图像，形状为 `(H, W, 3)`，dtype 为 `uint8`，通道顺序为 BGR。"

    timestamp: object | None
    "qmlinker 返回的帧时间戳，类型由 proto wrapper 决定。"

    sequence_id: int | None = None
    "流式相机帧序号。ZMQ 相机流可提供单调递增 sequence；qmlinker 无该字段时为 `None`。"

    depth: np.ndarray | None = None
    "深度图，形状为 `(H, W)`，dtype 通常为 `uint16`，单位由 qmlinker 服务定义。"


# endregion


# region 配置

SUPPORTED_WUJI_CAMERAS: tuple[WujiCameraSpec, ...] = (
    WujiCameraSpec("head_camera", "头部全景相机"),
    WujiCameraSpec("chest_camera", "胸部相机"),
    WujiCameraSpec("left_hand_camera", "左手相机（Gemini 336/336L）"),
    WujiCameraSpec("right_hand_camera", "右手相机（离线）"),
)
"当前 qmlinker 相机服务文档定义的相机安装位。"

DEFAULT_WUJI_CAMERA: WujiCameraName = "head_camera"
"打开相机 tab 时默认选择的相机。"


# endregion


# region 主入口

def parse_wuji_camera_name(camera_name: str) -> WujiCameraName | None:
    """解析 GUI 传入的无际相机名称。

    Parameters
    ----------
    camera_name:
        GUI 下拉框或配置文件中的相机逻辑名称。

    Returns
    -------
    parsed:
        支持的相机名称；未知名称返回 `None`。
    """

    for spec in SUPPORTED_WUJI_CAMERAS:
        if camera_name == spec.name:
            return spec.name
    return None


# endregion
