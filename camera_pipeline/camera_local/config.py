from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalStreamProfileConfig:
    """单路本机 USB 视频流配置。

    配置值必须与目标相机实际公开的 profile 完全匹配。运行时不会静默回退到
    默认 profile，避免实际分辨率、帧率或内参与配置不一致。
    """

    width: int
    "图像宽度，单位 pixel。"
    height: int
    "图像高度，单位 pixel。"
    fps: int
    "采集帧率，单位 frame/s。"
    format_name: str
    "pyorbbecsdk `OBFormat` 名称，例如 `MJPG`、`Y16`。"


@dataclass(frozen=True, slots=True)
class LocalCameraRuntimeConfig:
    """单台本机 USB Orbbec 相机的长期运行配置。

    该配置只保存设备选择、流 profile 和重试策略，不持有 SDK 资源。SN 为空时
    运行时保持离线并持续重试，便于先部署配置再填写现场设备编号。
    """

    camera_name: str
    "项目内逻辑相机名。"
    camera_id: str
    "固定安装位标识，只允许 LEFT、RIGHT、HEAD、CHEST。"
    serial_number: str
    "Orbbec 设备序列号；空字符串表示尚未配置。"
    color: LocalStreamProfileConfig
    "彩色流 profile。"
    depth: LocalStreamProfileConfig
    "深度流 profile。"
    frame_timeout_ms: int = 2000
    "单次等待 RGBD 帧组超时，单位 ms。"
    reconnect_initial_interval_s: float = 5.0
    "首次连接失败或连接中断后的重试间隔，单位 s。"
    reconnect_max_interval_s: float = 60.0
    "连续连接失败时指数退避的最大间隔，单位 s。"
