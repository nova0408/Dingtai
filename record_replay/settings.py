"""双臂自动回放服务的唯一配置定义页。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReplayDeviceConnection:
    """一轮回放所需设备的现场连接参数。"""

    left_arm_ip: str
    "左臂控制器 IPv4 地址。"
    right_arm_ip: str
    "右臂控制器 IPv4 地址。"
    qmlinker_host: str = "192.168.100.60"
    "Orin 部署时直连的 qmlinker 地址；本机不直接访问设备网。"
    qmlinker_port: int = 50062
    "手部与 body 使用的 qmlinker 端口。"
    gripper_port: int = 50066
    "左夹爪使用的 qmlinker 端口。"


# region 设备动作配置


@dataclass(frozen=True, slots=True)
class ReplayArmSettings:
    """机械臂连接、NRT 准备与 MoveAbsJ 时序参数。"""

    tool_name: str = "g_tool_0"
    "回放固定使用的工具坐标系名称。"
    wobj_name: str = "g_wobj_0"
    "回放固定使用的工件坐标系名称。"
    left_arm_type: str = "AR5-5_0.8L-W4C1C9-ZY2"
    "左臂控制器应上报的机型名称。"
    right_arm_type: str = "AR5-5_0.8R-W4C1C9-ZY2"
    "右臂控制器应上报的机型名称。"
    default_cartesian_speed_mm_s: float = 50.0
    "NRT 准备默认笛卡尔速度，单位 mm/s。"
    default_cartesian_zone_mm: float = 1.0
    "NRT 准备默认笛卡尔转弯区，单位 mm。"
    power_on_timeout_s: float = 3.0
    "等待控制器确认上电的超时，单位 s。"
    power_on_poll_interval_s: float = 0.1
    "等待控制器确认上电的状态轮询周期，单位 s。"
    motion_state_poll_interval_s: float = 0.1
    "等待机械臂运动完成的轮询周期，单位 s。"
    reset_ready_timeout_s: float = 2.0
    "调用 moveReset 前的就绪等待超时，单位 s。"
    reset_ready_stable_idle_checks: int = 2
    "调用 moveReset 前连续确认 idle 的次数。"
    reset_ready_poll_interval_s: float = 0.2
    "调用 moveReset 前的状态轮询周期，单位 s。"


@dataclass(frozen=True, slots=True)
class ReplayHandSettings:
    """夹爪、M11 与升降动作参数。"""

    m11_state_read_timeout_s: float = 10.0
    "读取至少 11 个有效右手执行器状态的总超时时间，单位 s。"
    m11_state_read_poll_interval_s: float = 0.1
    "右手状态内容无效时的重新读取间隔，单位 s。"
    lift_enable_state_timeout_s: float = 10.0
    "下发 lift enable 后等待 get_enable() 状态为 True 的总超时时间，单位 s。"
    lift_enable_retry_interval_s: float = 0.2
    "lift enable 状态尚未生效时的重新下发间隔，单位 s。"
    lift_target_reissue_interval_s: float = 1.0
    "lift 实际高度尚未到位时重新下发目标高度的间隔，单位 s。"
    lift_motion_timeout_s: float = 30.0
    "等待升降机构到位的总超时时间，单位 s。"
    lift_poll_interval_s: float = 0.1
    "有效高度尚未到位时的轮询间隔，单位 s；负数通信无效值立即重读。"
    lift_height_tolerance_mm: float = 4.0
    "升降机构到位误差容忍，单位 mm。"
    m11_root_actuator_ids: tuple[int, ...] = (3, 5, 7, 9)
    "M11 根部执行器索引。"
    m11_tip_actuator_ids: tuple[int, ...] = (4, 6, 8, 10)
    "M11 指尖执行器索引。"
    gripper_calibration_wait_s: float = 3.0
    "夹爪校准命令后的固定等待时间，单位 s。"
    gripper_zero_position: int = 0
    "回放开始前夹爪归零位置。"
    gripper_zero_poll_interval_s: float = 0.2
    "夹爪归零状态轮询间隔，单位 s。"


@dataclass(frozen=True, slots=True)
class ReplayOffsetSettings:
    """三球全局纠偏的触发、采样与鲁棒聚合参数。"""

    calculate_after_action_name: str | None = "calibration"
    """完成该命名动作后执行三球采样；为空表示本轮不触发。"""
    target_action_names: frozenset[str] = frozenset({"get_tray", "put_new_tray"})
    "需要应用全局笛卡尔纠偏的命名动作。"
    sample_count: int = 2
    "单次全局纠偏连续采样次数。"
    detection_timeout_ms: int = 30_000
    "单次三球检测 RPC 超时，单位 ms。"
    detection_attempts_per_sample: int = 3
    "每个宽 HSV 或窄 HSV 阶段获得完整三球结果的最大尝试次数。"
    capture_settle_delay_s: float = 0.0
    "纠偏触发 CSV 完成后、采集三球前的等待时间，单位 s。"
    mad_scale: float = 3.5
    "三球样本 MAD 异常剔除倍数。"
    min_outlier_threshold_mm: float = 2.0
    "MAD 过小时采用的最小异常距离阈值，单位 mm。"
    narrow_consistency_tolerance_mm: float = 8.0
    "窄 HSV 与宽 HSV 同色球心允许的最大差异，单位 mm。"
    left_charuco_target_action_names: frozenset[str] = frozenset({"open_door", "close_door"})
    "左臂应用 ChArUco offset 的命名动作。"
    right_charuco_target_action_names: frozenset[str] = frozenset({"open_door", "close_door"})
    "右臂应用 ChArUco offset 的命名动作。"
    charuco_head_yaw_deg: float = 60.0
    "ChArUco 检测前头部 yaw 目标角度，单位 deg。"
    charuco_head_pitch_deg: float = 45.0
    "ChArUco 检测前头部 pitch 目标角度，单位 deg。"
    charuco_head_settle_s: float = 1.0
    "头部到达 ChArUco 检测姿态后的稳定等待时间，单位 s。"
    charuco_camera_timeout_s: float = 10.0
    "ChArUco 每次稳定帧请求的超时时间，单位 s。"
    charuco_max_frame_count: int = 5
    "单次 ChArUco 检测允许检查的稳定帧数量。"
    charuco_min_corners: int = 6
    "进入 ChArUco PnP 的最少角点数量。"
    charuco_rpc_timeout_s: float = 55.0
    "ChArUco RPC 单次接收超时时间，单位 s。"
    charuco_timeout_retry_count: int = 3
    "ChArUco RPC 超时重试次数。"
    charuco_timeout_retry_delay_s: float = 1.0
    "ChArUco RPC 超时重试间隔，单位 s。"
    charuco_safety_attempt_count: int = 3
    "ChArUco offset 历史安全检查失败后的重新检测次数。"
    charuco_safety_retry_delay_s: float = 1.0
    "ChArUco offset 安全检查失败后的重新检测间隔，单位 s。"
    charuco_history_min_accepted_samples: int = 6
    "允许使用 ChArUco offset 的同侧有效历史最少条数。"
    charuco_sigma_limit: float = 4.0
    "ChArUco 历史分量与模长统计的标准差倍数。"
    charuco_max_translation_norm_mm: float = 60.0
    "ChArUco offset 平移模长绝对上限，单位 mm。"
    charuco_max_rotation_norm_deg: float = 5.0
    "ChArUco offset 旋转模长绝对上限，单位 deg。"


@dataclass(frozen=True, slots=True)
class OffsetConfig:
    """三球检测服务和标定文件的现场输入配置。"""

    prior_capture_path: Path
    "先验三球采集结果路径。"
    hand_eye_result_path: Path
    "手眼标定结果路径。"
    camera_name: str = "left_hand_camera"
    "球位姿检测相机名称。"
    charuco_prior_path: Path | None = None
    "ChArUco T_camera_board 先验路径。"
    charuco_history_path: Path | None = None
    "人工确认的 ChArUco offset 历史 CSV 路径。"
    left_head_base_camera_path: Path | None = None
    "左臂基坐标系下的 T_base_camera 路径。"
    right_head_base_camera_path: Path | None = None
    "右臂基坐标系下的 T_base_camera 路径。"


@dataclass(frozen=True, slots=True)
class ReplayServiceSettings:
    """自动回放服务运行期间可调的全部策略参数。"""

    arm: ReplayArmSettings = field(default_factory=ReplayArmSettings)
    "机械臂控制参数。"
    hand: ReplayHandSettings = field(default_factory=ReplayHandSettings)
    "手部与升降动作参数。"
    offset: ReplayOffsetSettings = field(default_factory=ReplayOffsetSettings)
    "三球 offset 参数。"
    agv_navigation_timeout_s: float = 600.0
    "AGV 单次导航到位等待超时，单位 s。"
    agv_navigation_poll_interval_s: float = 1.0
    "AGV runtime info 轮询周期，单位 s。"
    agv_stop_timeout_s: float = 5.0
    "AGV Stop RPC 超时时间，单位 s。"
    non_motion_retry_count: int = 3
    "非运动设备调用最大尝试次数。"
    non_motion_retry_delay_s: float = 0.5
    "非运动设备调用重试间隔，单位 s。"


# endregion


# region 一轮服务配置


@dataclass(frozen=True, slots=True)
class ReplayCycleConfig:
    """一轮自动回放服务的静态目录、站点和运行策略配置。"""

    left_record_dir: Path
    "左臂 CSV 目录。"
    right_record_dir: Path
    "右臂 CSV 目录。"
    action_sequence_path: Path
    "固定命名动作顺序 JSON 路径。"
    device_connection: ReplayDeviceConnection
    "双臂、手部和升降设备的现场连接参数。"
    settings: ReplayServiceSettings = field(default_factory=ReplayServiceSettings)
    "服务运行参数的唯一配置对象。"


# endregion
