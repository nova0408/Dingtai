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
    move_abs_j_end_linear_speed_mm_s: float = 1000.0
    "普通连续 MoveAbsJ 段末端线速度，单位 mm/s。"
    move_abs_j_zone_mm: float = 10.0
    "连续 MoveAbsJ 中间点转弯区半径，单位 mm。"
    motion_state_poll_interval_s: float = 0.1
    "等待机械臂运动完成的轮询周期，单位 s。"
    reset_ready_timeout_s: float = 2.0
    "调用 moveReset 前的就绪等待超时，单位 s。"
    reset_ready_stable_idle_checks: int = 2
    "调用 moveReset 前连续确认 idle 的次数。"
    reset_ready_poll_interval_s: float = 0.2
    "调用 moveReset 前的状态轮询周期，单位 s。"
    left_zero_zone_sequences: frozenset[int] = frozenset()
    "左臂 CSV 末尾强制 zone=0 的阶段序号。"


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


@dataclass(frozen=True, slots=True)
class ReplayOffsetSettings:
    """三球全局纠偏的触发、采样与鲁棒聚合参数。"""

    target_sequences: frozenset[int] = frozenset({4, 6})
    "需要应用全局笛卡尔纠偏的 CSV 阶段序号。"
    calculate_at_sequence: int = 3
    "完成该左臂 CSV 后计算新的全局纠偏。"
    sample_count: int = 2
    "单次全局纠偏连续采样次数。"
    detection_timeout_ms: int = 30_000
    "单次三球检测 RPC 超时，单位 ms。"
    trigger_move_abs_j_end_linear_speed_mm_s: float = 700.0
    "纠偏触发 CSV 临时使用的末端线速度，单位 mm/s。"
    capture_settle_delay_s: float = 0.0
    "纠偏触发 CSV 完成后、采集三球前的等待时间，单位 s。"
    mad_scale: float = 3.5
    "三球样本 MAD 异常剔除倍数。"
    min_outlier_threshold_mm: float = 2.0
    "MAD 过小时采用的最小异常距离阈值，单位 mm。"


@dataclass(frozen=True, slots=True)
class OffsetConfig:
    """三球检测服务和标定文件的现场输入配置。"""

    prior_capture_path: Path
    "先验三球采集结果路径。"
    hand_eye_result_path: Path
    "手眼标定结果路径。"
    camera_name: str = "left_hand_camera"
    "球位姿检测相机名称。"


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
    trigger_poll_interval_s: float = 1.0
    "触发文件等待轮询周期，单位 s。"
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
    device_connection: ReplayDeviceConnection
    "双臂、手部和升降设备的现场连接参数。"
    settings: ReplayServiceSettings = field(default_factory=ReplayServiceSettings)
    "服务运行参数的唯一配置对象。"
    start_station: str = "3"
    "执行前 AGV 目标站点。"
    finish_station: str = "1"
    "执行完成后 AGV 目标站点。"
    state_prefix: str = "left_"
    "左臂 CSV 状态名需要删除的文件名前缀。"
    trigger_file: Path | None = None
    "循环服务的触发文件路径。"


# endregion
