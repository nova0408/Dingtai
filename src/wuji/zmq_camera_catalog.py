from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.wuji.camera_protocol import WujiCameraName, WujiCameraRuntimeInfo

if TYPE_CHECKING:
    from src.wuji.zmq_camera_client import WujiZmqCameraClient

# region 数据结构


@dataclass(frozen=True, slots=True)
class WujiZmqCameraEndpoint:
    """无际 ZMQ 相机端点定义。

    职责边界：
    - 只描述逻辑相机名、控制标识、数据端口与默认分辨率。
    - 不负责连接建立、状态轮询、帧解码或 GUI 展示。

    设计思想：
    - 将多相机静态端点表独立成纯数据，供 GUI 和测试脚本共享。
    - 端点表是部署约定，不与 client 的网络生命周期绑定。

    生命周期：
    - 模块加载时构造，可长期只读复用。

    继承关系：
    - 不继承业务基类，作为 ZMQ 相机端点数据使用。
    """

    camera_name: WujiCameraName
    "项目内逻辑相机名。"

    camera_id: str
    "ZMQ 控制口使用的相机标识，例如 `HEAD`。"

    stream_port: int
    "对应相机数据流 ZMQ PUB 端口，单位 TCP 端口号。"

    width: int = 1280
    "默认图像宽度，单位 像素。"

    height: int = 720
    "默认图像高度，单位 像素。"


@dataclass(frozen=True, slots=True)
class WujiZmqCameraStatus:
    """无际 ZMQ 相机状态结果。

    职责边界：
    - 只保存控制口 `get_status` 返回的在线、彩色和深度状态。
    - 不负责控制命令发送、帧拉流或 UI 显示逻辑。

    设计思想：
    - 把在线状态与彩色/深度开关一起返回，便于上层区分“离线”和“被关闭”。
    - 保持字段扁平，避免上层继续解析无结构 JSON。

    生命周期：
    - 每次控制请求完成后构造，不持有外部资源。

    继承关系：
    - 不继承业务基类，作为相机状态数据使用。
    """

    camera_name: WujiCameraName
    "项目内逻辑相机名。"

    online: bool
    "相机服务端报告的在线状态。"

    color_enabled: bool
    "彩色流开关状态。"

    depth_enabled: bool
    "深度流开关状态。"


# endregion


# region 配置


SUPPORTED_WUJI_ZMQ_CAMERAS: tuple[WujiZmqCameraEndpoint, ...] = (
    WujiZmqCameraEndpoint("head_camera", "HEAD", 5560),
    WujiZmqCameraEndpoint("chest_camera", "CHEST", 5561),
    WujiZmqCameraEndpoint("left_hand_camera", "LEFT", 5562),
    WujiZmqCameraEndpoint("right_hand_camera", "RIGHT", 5563),
)
"项目当前默认使用的逻辑相机与 ZMQ 数据口映射。"

SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL: tuple[WujiZmqCameraEndpoint, ...] = (
    WujiZmqCameraEndpoint("head_camera", "HEAD", 5559),
    WujiZmqCameraEndpoint("chest_camera", "CHEST", 5560),
    WujiZmqCameraEndpoint("left_hand_camera", "LEFT", 5561),
    WujiZmqCameraEndpoint("right_hand_camera", "RIGHT", 5562),
)
"仅用于本机而非 Orin 的远程调试端口映射。"


# endregion


# region 基础工具


def get_wuji_zmq_camera_endpoint(
    camera_name: WujiCameraName,
    *,
    supported_endpoints: tuple[WujiZmqCameraEndpoint, ...] = SUPPORTED_WUJI_ZMQ_CAMERAS,
) -> WujiZmqCameraEndpoint:
    """按逻辑相机名返回对应端点。

    Parameters
    ----------
    camera_name:
        待查询逻辑相机名。
    supported_endpoints:
        当前调用方采用的端点表。默认使用现场部署端点；本机 SSH 调试时可显式传入
        `SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL`。

    Returns
    -------
    endpoint:
        与逻辑相机名匹配的静态端点配置。
    """

    for endpoint in supported_endpoints:
        if endpoint.camera_name == camera_name:
            return endpoint
    raise ValueError(f"unsupported ZMQ camera: {camera_name}")


def build_wuji_zmq_camera_runtime_info(
    endpoint: WujiZmqCameraEndpoint,
    status: WujiZmqCameraStatus,
) -> WujiCameraRuntimeInfo:
    """按静态端点和实时状态构造多相机清单项。"""

    return WujiCameraRuntimeInfo(
        camera_name=endpoint.camera_name,
        camera_id=endpoint.camera_id,
        serial_number="",
        display_name=endpoint.camera_id,
        online=status.online,
        color_enabled=status.color_enabled,
        depth_enabled=status.depth_enabled,
    )


def list_wuji_zmq_camera_runtime_infos(
    client: "WujiZmqCameraClient",
    *,
    online_only: bool = True,
    supported_endpoints: tuple[WujiZmqCameraEndpoint, ...] = SUPPORTED_WUJI_ZMQ_CAMERAS,
) -> tuple[WujiCameraRuntimeInfo, ...]:
    """列出静态多相机清单及其实时状态。

    Parameters
    ----------
    client:
        ZMQ 相机客户端。
    online_only:
        是否只返回在线相机。
    supported_endpoints:
        当前调用方采用的端点表。默认使用现场部署端点；本机 SSH 调试时可显式传入
        `SUPPORTED_WUJI_ZMQ_CAMERAS_LOCAL`。

    Returns
    -------
    runtime_infos:
        端点表中每一路相机对应的运行时状态列表。
    """

    runtime_infos: list[WujiCameraRuntimeInfo] = []
    for endpoint in supported_endpoints:
        status = client.get_camera_status(endpoint.camera_name)
        if online_only and not status.online:
            continue
        runtime_infos.append(build_wuji_zmq_camera_runtime_info(endpoint, status))
    return tuple(runtime_infos)


# endregion
