from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

# region 数据结构

ArmSide = Literal["left", "right"]
ArmDeviceName = Literal["left_arm", "right_arm"]


@dataclass(frozen=True, slots=True)
class QmlinkerArmRemoteConfig:
    """无际 qmlinker 机械臂远端连接配置。

    职责边界：
    - 只描述远端 Python 脚本访问 qmlinker gRPC 服务所需的连接参数。
    - 不负责 SSH 认证、Qt 信号、GUI 控件或本地进程生命周期。

    设计思想：
    - 与接口示例文件保持一致，默认 `grpc_uri` 沿用示例中的 `192.168.20.133`。
    - 独立配置便于后续把 gRPC 地址改为 Orin 本机地址或配置文件读取。

    生命周期：
    - 不持有网络连接，可在线程间安全传递。

    继承关系：
    - 不继承业务基类，保持数据契约单一。
    """

    grpc_uri: str = "192.168.20.133"
    "qmlinker gRPC 服务地址，单位为主机或 host:port 字符串。"

    default_speed_ratio: float = 0.3
    "关节命令默认速度比例，范围通常为 0.0 到 1.0。"


@dataclass(frozen=True, slots=True)
class QmlinkerArmCommand:
    """发送给 Orin 远端脚本的机械臂命令。

    职责边界：
    - 表达 GUI 调试页需要的最小命令集合。
    - 不直接导入 qmlinker，避免本地开发环境必须安装远端 SDK。

    设计思想：
    - 使用 JSON 可序列化字段，便于通过 `ssh orin python3 -c ...` 传递。
    - 每条命令独立执行，第一版优先保证真实可用与容易诊断。

    生命周期：
    - 命令对象只在本地组装阶段存在，不持有外部资源。

    继承关系：
    - 不继承业务基类，避免隐式动态分发。
    """

    action: Literal["ping", "get_enable", "set_enable", "get_joints", "set_joint"]
    "命令动作名称。"

    device_name: ArmDeviceName | None = None
    "机械臂设备名，取值为 `left_arm` 或 `right_arm`。"

    enabled: bool | None = None
    "目标使能状态，仅 `set_enable` 使用。"

    joint_index: int | None = None
    "关节索引，范围 1 到 6，仅 `set_joint` 使用。"

    target_angle_deg: float | None = None
    "目标关节角度，单位 deg，仅 `set_joint` 使用。"

    speed_ratio: float | None = None
    "速度比例，范围通常为 0.0 到 1.0，仅 `set_joint` 使用。"


# endregion


# region 轴与设备映射

SUPPORTED_ARM_DEVICES: tuple[ArmDeviceName, ...] = ("left_arm", "right_arm")
"当前 arm 示例文档可真实控制的机械臂设备。"


def parse_arm_axis_name(axis_name: str) -> tuple[ArmDeviceName, int] | None:
    """解析 GUI DoF 轴名为 qmlinker 机械臂设备与关节索引。

    Parameters
    ----------
    axis_name:
        GUI 轴名，例如 `left_j1` 或 `right_j6`。

    Returns
    -------
    tuple[ArmDeviceName, int] | None
        成功时返回设备名与 1 基关节索引；非机械臂轴返回 `None`。
    """

    if axis_name.startswith("left_j"):
        index_text = axis_name.removeprefix("left_j")
        if index_text.isdigit() and 1 <= int(index_text) <= 6:
            return "left_arm", int(index_text)
        return None
    if axis_name.startswith("right_j"):
        index_text = axis_name.removeprefix("right_j")
        if index_text.isdigit() and 1 <= int(index_text) <= 6:
            return "right_arm", int(index_text)
        return None
    return None


def axis_names_for_device(device_name: ArmDeviceName) -> tuple[str, ...]:
    """返回指定机械臂对应的 GUI 轴名序列。

    Parameters
    ----------
    device_name:
        机械臂设备名，取值为 `left_arm` 或 `right_arm`。

    Returns
    -------
    tuple[str, ...]
        GUI 轴名序列，长度为 6，单位语义为 deg。
    """

    prefix = "left" if device_name == "left_arm" else "right"
    return tuple(f"{prefix}_j{idx}" for idx in range(1, 7))


# endregion


# region 远端脚本

REMOTE_QMLINKER_ARM_SCRIPT = r'''
from __future__ import annotations

import json
import sys
import time

from qmlinker import QMArm, create_channel


def _arm_type(QMArm, device_name: str):
    if device_name == "left_arm":
        return QMArm.ARM_LEFT
    if device_name == "right_arm":
        return QMArm.ARM_RIGHT
    raise ValueError(f"unsupported arm device: {device_name}")


def _response(ok: bool, **kwargs):
    payload = {"ok": ok}
    payload.update(kwargs)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _joint_angles(arm) -> list[float]:
    time.sleep(0.2)
    return [float(joint.angle_deg) for joint in arm.arm_info.joint_states]


def main() -> None:
    request = json.loads(sys.argv[1])
    channel = create_channel(request.get("grpc_uri", "192.168.20.133"))
    action = request["action"]

    if action == "ping":
        _response(True, connected=True)
        return

    device_name = request["device_name"]
    arm = QMArm(channel, _arm_type(QMArm, device_name))

    if action == "get_enable":
        _response(True, device_name=device_name, enabled=bool(arm.get_enable()))
        return

    if action == "set_enable":
        target = bool(request["enabled"])
        result = arm.set_enable(target)
        _response(True, device_name=device_name, enabled=bool(arm.get_enable()), raw=str(result))
        return

    if action == "get_joints":
        _response(True, device_name=device_name, joints_deg=_joint_angles(arm))
        return

    if action == "set_joint":
        angles = _joint_angles(arm)
        joint_index = int(request["joint_index"])
        angles[joint_index - 1] = float(request["target_angle_deg"])
        speed_ratio = float(request.get("speed_ratio", 0.3))
        commands = [
            {"joint_id": idx, "target_angle_deg": angle, "speed_ratio": speed_ratio}
            for idx, angle in enumerate(angles, start=1)
        ]
        result = arm.set_joints(commands)
        _response(True, device_name=device_name, joints_deg=_joint_angles(arm), raw=str(result))
        return

    raise ValueError(f"unsupported action: {action}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _response(False, error=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
'''
"在 Orin 端通过 `python3 -c` 执行的 qmlinker 机械臂 JSON 命令脚本。"


def build_remote_payload(config: QmlinkerArmRemoteConfig, command: QmlinkerArmCommand) -> str:
    """构建远端脚本入参 JSON。

    Parameters
    ----------
    config:
        qmlinker 远端连接配置。
    command:
        机械臂命令对象。

    Returns
    -------
    str
        JSON 字符串，可作为远端 Python 脚本的第一个参数。
    """

    payload: dict[str, Any] = {
        "grpc_uri": config.grpc_uri,
        "action": command.action,
    }
    if command.device_name is not None:
        payload["device_name"] = command.device_name
    if command.enabled is not None:
        payload["enabled"] = command.enabled
    if command.joint_index is not None:
        payload["joint_index"] = command.joint_index
    if command.target_angle_deg is not None:
        payload["target_angle_deg"] = command.target_angle_deg
    payload["speed_ratio"] = config.default_speed_ratio if command.speed_ratio is None else command.speed_ratio
    return json.dumps(payload, ensure_ascii=False)


# endregion
