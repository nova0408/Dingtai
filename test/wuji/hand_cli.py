from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import grpc
from loguru import logger
from qmlinker.grpc_py import common_pb2, hand_pb2


# region 路径初始化


def _append_repo_root_to_sys_path() -> None:
    """允许直接 F5 运行 test 下的脚本。"""

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "src").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


_append_repo_root_to_sys_path()


from src.wuji.client_base import WujiQmlinkerBaseClient  # noqa: E402
from src.wuji.protocol import WujiQmlinkerConfig  # noqa: E402
from src.wuji.right_hand_client import WujiRightHandClient  # noqa: E402
from src.wuji.right_hand_specs import RIGHT_HAND_ACTUATOR_SPECS, WujiRightHandActuatorSpec  # noqa: E402


# endregion


DEFAULT_HOST = "192.168.100.60"
DEFAULT_PORT = 50062

REQUEST_TIMEOUT_S = 3.0

# 轮询周期
POLL_INTERVAL_S = 0.05

# 目标轴读数变化小于该值，认为本次读数没有明显变化
STABLE_EPS = 1e-3

# 目标轴读数连续 1s 没有明显变化后，认为本次动作结束
UNCHANGED_TIMEOUT_S = 1.0

# 防止读数一直抖动导致死循环。正常情况下不会用到。
HARD_POLL_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True)
class HandCliConfig:
    """右手交互式 CLI 配置。"""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    request_timeout_s: float = REQUEST_TIMEOUT_S

    def target(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class AxisState:
    """单个 actuator 当前状态。"""

    actuator_id: int
    axis_name: str
    position: float
    current: float
    temperature: float
    is_stalled: bool
    raw_pwm: int
    raw_current: int


@dataclass(frozen=True, slots=True)
class HandSnapshot:
    """一次右手状态快照。"""

    axes: tuple[AxisState, ...]
    timestamp_ms: float

    def axis_by_id(self) -> dict[int, AxisState]:
        return {axis.actuator_id: axis for axis in self.axes}

    def position_of(self, actuator_id: int) -> float:
        for axis in self.axes:
            if axis.actuator_id == int(actuator_id):
                return axis.position
        raise KeyError(f"状态中没有 actuator_id={actuator_id}")


@dataclass(frozen=True, slots=True)
class AxisMotionResult:
    """单轴动作轮询结果。"""

    target_value: float
    final_value: float
    error: float
    elapsed_s: float
    sample_count: int
    stopped_reason: str
    final_snapshot: HandSnapshot


# region 基础 RPC


def _format_rpc_error(error: grpc.RpcError) -> str:
    """格式化 gRPC 异常信息。"""

    try:
        code = error.code()
    except Exception:
        code = "UNKNOWN"

    try:
        details = error.details()
    except Exception:
        details = str(error)

    return f"{code}: {details}"


def _create_base_client(config: HandCliConfig) -> WujiQmlinkerBaseClient:
    """创建 qmlinker base client。"""

    try:
        qmlinker_config = WujiQmlinkerConfig(
            host=config.host,
            port=config.port,
            request_timeout_s=config.request_timeout_s,
        )
    except TypeError:
        qmlinker_config = WujiQmlinkerConfig(host=config.host, port=config.port)

    return WujiQmlinkerBaseClient(qmlinker_config)


def _read_enable(hand: WujiRightHandClient, timeout_s: float) -> bool:
    """读取右手使能状态。"""

    request = hand_pb2.GetHandInfoRequest()
    request.hand_id = cast(Any, hand.hand_id)

    try:
        response = hand.stub.GetEnabled(request, timeout=float(timeout_s))
    except grpc.RpcError as error:
        raise RuntimeError(f"GetEnabled failed: {_format_rpc_error(error)}") from error

    return bool(response.status.success and response.current_state == common_pb2.MODULE_ENABLED)


def _ensure_enabled(hand: WujiRightHandClient, config: HandCliConfig) -> None:
    """确保右手已使能，避免下发后被 ROS 端直接拒绝。

    Parameters
    ----------
    hand:
        右手客户端。
    config:
        CLI 配置，用于请求超时。

    Notes
    -----
    `wuyou` 上的 ROS 手节点会先检查使能状态，未使能时会直接拒绝执行。
    这里在测试入口显式补上使能动作，避免把“未使能拒绝”误判成“指令无动作”。
    """

    enabled = _read_enable(hand, timeout_s=config.request_timeout_s)
    if enabled:
        return

    print("右手当前未使能，先执行 enable 再下发动作。")
    if not _set_enable(hand, enable=True, timeout_s=config.request_timeout_s):
        raise RuntimeError("右手使能下发失败，拒绝继续控制。")

    enabled_after = _read_enable(hand, timeout_s=config.request_timeout_s)
    if not enabled_after:
        raise RuntimeError("右手使能后仍未进入 enabled 状态。")


def _set_enable(hand: WujiRightHandClient, enable: bool, timeout_s: float) -> bool:
    """设置右手使能状态。"""

    request = hand_pb2.HandEnableRequest()
    request.hand_id = cast(Any, hand.hand_id)
    request.enable = bool(enable)

    try:
        response = hand.stub.SetEnabled(request, timeout=float(timeout_s))
    except grpc.RpcError as error:
        raise RuntimeError(f"SetEnabled failed: {_format_rpc_error(error)}") from error

    return bool(response.status.success)


def _read_hand_info(hand: WujiRightHandClient, timeout_s: float) -> dict[str, Any]:
    """读取右手基础信息。"""

    request = hand_pb2.GetHandInfoRequest()
    request.hand_id = cast(Any, hand.hand_id)

    try:
        response = hand.stub.GetHandInfo(request, timeout=float(timeout_s))
    except grpc.RpcError as error:
        raise RuntimeError(f"GetHandInfo failed: {_format_rpc_error(error)}") from error

    return {
        "hand_id": int(response.hand_id),
        "model_name": str(response.model_name),
        "actuator_count": int(response.actuator_count),
        "has_tactile": bool(response.has_tactile),
        "actuator_names": list(response.actuator_names),
    }


def _read_snapshot(hand: WujiRightHandClient, timeout_s: float) -> HandSnapshot:
    """真实读取一次右手 actuator 状态，不使用缓存。"""

    request = hand_pb2.GetHandStateRequest()
    request.hand_id = cast(Any, hand.hand_id)
    request.include_tactile = False

    try:
        response = hand.stub.GetHandState(request, timeout=float(timeout_s))
    except grpc.RpcError as error:
        raise RuntimeError(f"GetHandState failed: {_format_rpc_error(error)}") from error

    axes: list[AxisState] = []
    for actuator in response.actuators:
        actuator_id = int(actuator.actuator_id)
        axes.append(
            AxisState(
                actuator_id=actuator_id,
                axis_name=f"right_hand_a{actuator_id}",
                position=float(actuator.position),
                current=float(actuator.current),
                temperature=float(actuator.temperature),
                is_stalled=bool(actuator.is_stalled),
                raw_pwm=int(actuator.raw_pwm),
                raw_current=int(actuator.raw_current),
            )
        )

    timestamp_ms = float(
        response.timestamp.seconds * 1000
        + response.timestamp.nanos / 1_000_000
    )

    return HandSnapshot(
        axes=tuple(sorted(axes, key=lambda item: item.actuator_id)),
        timestamp_ms=timestamp_ms,
    )


# endregion


# region 轴规格与打印


def _axis_name_from_spec(spec: WujiRightHandActuatorSpec) -> str:
    """读取项目侧轴名。"""

    axis_name = getattr(spec, "axis_name", None)
    if isinstance(axis_name, str) and axis_name:
        return axis_name
    return f"right_hand_a{int(spec.actuator_id)}"


def _minimum_from_spec(spec: WujiRightHandActuatorSpec) -> float | None:
    value = getattr(spec, "minimum", None)
    return None if value is None else float(value)


def _maximum_from_spec(spec: WujiRightHandActuatorSpec) -> float | None:
    value = getattr(spec, "maximum", None)
    return None if value is None else float(value)


def _format_range(spec: WujiRightHandActuatorSpec) -> str:
    minimum = _minimum_from_spec(spec)
    maximum = _maximum_from_spec(spec)

    if minimum is None and maximum is None:
        return "-"

    left = "-inf" if minimum is None else f"{minimum:.3f}"
    right = "+inf" if maximum is None else f"{maximum:.3f}"
    return f"[{left}, {right}]"


def _resolve_axis_by_user_index(user_text: str) -> WujiRightHandActuatorSpec:
    """把用户输入的 1、2、3 映射为 right_hand_a1、right_hand_a2、right_hand_a3。"""

    text = user_text.strip()

    if not text.isdigit():
        raise ValueError("轴号必须是数字，例如 1、2、3。")

    actuator_id = int(text)

    for spec in RIGHT_HAND_ACTUATOR_SPECS:
        if int(spec.actuator_id) == actuator_id:
            return spec

    valid_ids = ", ".join(str(int(spec.actuator_id)) for spec in RIGHT_HAND_ACTUATOR_SPECS)
    raise ValueError(f"未知轴号：{actuator_id}。可选轴号：{valid_ids}")


def _validate_target(spec: WujiRightHandActuatorSpec, target_value: float) -> None:
    """检查右手控制目标必须是 0 到 1 的归一化值。"""

    axis_name = _axis_name_from_spec(spec)
    value = float(target_value)

    if not math.isfinite(value):
        raise ValueError(f"{axis_name} 目标值必须是有限数，当前为 {target_value!r}")

    if value < 0.0 or value > 1.0:
        raise ValueError(f"{axis_name} 目标值必须在 0-1 之间，当前为 {value:.6f}")


def _print_hand_info(hand: WujiRightHandClient, config: HandCliConfig) -> None:
    """打印右手信息。"""

    hand_info = _read_hand_info(hand, timeout_s=config.request_timeout_s)
    enabled = _read_enable(hand, timeout_s=config.request_timeout_s)

    print("")
    print("右手信息：")
    print(f"  target         : {config.target()}")
    print(f"  hand_id        : {hand_info['hand_id']}")
    print(f"  model_name     : {hand_info['model_name']}")
    print(f"  actuator_count : {hand_info['actuator_count']}")
    print(f"  has_tactile    : {hand_info['has_tactile']}")
    print(f"  enabled        : {enabled}")


def _print_axis_specs() -> None:
    """打印可选轴。"""

    print("")
    print("可选轴：")
    print("  输入号  actuator_id  axis_name         range")
    print("  ------  -----------  ----------------  ----------------")
    for spec in RIGHT_HAND_ACTUATOR_SPECS:
        actuator_id = int(spec.actuator_id)
        print(
            f"  {actuator_id:<6}  "
            f"{actuator_id:<11}  "
            f"{_axis_name_from_spec(spec):<16}  "
            f"{_format_range(spec)}"
        )


def _print_snapshot(snapshot: HandSnapshot) -> None:
    """打印当前右手真实状态。"""

    state_by_id = snapshot.axis_by_id()

    print("")
    print("当前右手各轴状态：")
    print("  id      axis_name         position       current      temp       stalled")
    print("  ------  ----------------  -------------  -----------  ---------  -------")

    for spec in RIGHT_HAND_ACTUATOR_SPECS:
        actuator_id = int(spec.actuator_id)
        axis_name = _axis_name_from_spec(spec)
        axis_state = state_by_id.get(actuator_id)

        if axis_state is None:
            print(f"  {actuator_id:<6}  {axis_name:<16}  <missing>")
            continue

        print(
            f"  {actuator_id:<6}  "
            f"{axis_name:<16}  "
            f"{axis_state.position:>13.6f}  "
            f"{axis_state.current:>11.6f}  "
            f"{axis_state.temperature:>9.3f}  "
            f"{str(axis_state.is_stalled):>7}"
        )


def _print_startup_status(hand: WujiRightHandClient, config: HandCliConfig) -> None:
    """启动时打印使能状态和各轴状态。"""

    _print_hand_info(hand, config)
    snapshot = _read_snapshot(hand, timeout_s=config.request_timeout_s)
    _print_snapshot(snapshot)


# endregion


# region 动作轮询


def _poll_axis_until_unchanged_for_1s(
    hand: WujiRightHandClient,
    config: HandCliConfig,
    actuator_id: int,
    target_value: float,
) -> AxisMotionResult:
    """下发后立即轮询，直到目标轴数值连续 1s 不变。"""

    start_time = time.monotonic()
    hard_deadline = start_time + HARD_POLL_TIMEOUT_S

    sample_count = 0
    last_value: float | None = None
    last_changed_time = start_time
    final_snapshot = _read_snapshot(hand, timeout_s=config.request_timeout_s)

    print("")
    print("开始轮询目标轴状态：")

    while True:
        now = time.monotonic()

        final_snapshot = _read_snapshot(hand, timeout_s=config.request_timeout_s)
        current_value = final_snapshot.position_of(actuator_id)
        sample_count += 1

        if last_value is None:
            last_value = current_value
            last_changed_time = now
            print(
                f"  t={now - start_time:>6.3f}s  "
                f"actual={current_value:>13.6f}  "
                f"target={target_value:>13.6f}  "
                f"error={current_value - target_value:>13.6f}"
            )
        else:
            changed = abs(current_value - last_value) > STABLE_EPS

            if changed:
                last_value = current_value
                last_changed_time = now
                print(
                    f"  t={now - start_time:>6.3f}s  "
                    f"actual={current_value:>13.6f}  "
                    f"target={target_value:>13.6f}  "
                    f"error={current_value - target_value:>13.6f}"
                )

        unchanged_duration = now - last_changed_time
        if unchanged_duration >= UNCHANGED_TIMEOUT_S:
            final_value = final_snapshot.position_of(actuator_id)
            return AxisMotionResult(
                target_value=target_value,
                final_value=final_value,
                error=final_value - target_value,
                elapsed_s=now - start_time,
                sample_count=sample_count,
                stopped_reason=f"目标轴读数连续 {UNCHANGED_TIMEOUT_S:.3f}s 无明显变化",
                final_snapshot=final_snapshot,
            )

        if now >= hard_deadline:
            final_value = final_snapshot.position_of(actuator_id)
            return AxisMotionResult(
                target_value=target_value,
                final_value=final_value,
                error=final_value - target_value,
                elapsed_s=now - start_time,
                sample_count=sample_count,
                stopped_reason=f"达到硬超时 {HARD_POLL_TIMEOUT_S:.3f}s",
                final_snapshot=final_snapshot,
            )

        time.sleep(POLL_INTERVAL_S)


def _run_single_axis_motion(
    hand: WujiRightHandClient,
    config: HandCliConfig,
    spec: WujiRightHandActuatorSpec,
    target_value: float,
) -> AxisMotionResult:
    """执行一次单轴目标值设置。"""

    actuator_id = int(spec.actuator_id)
    axis_name = _axis_name_from_spec(spec)

    _ensure_enabled(hand, config)

    before_snapshot = _read_snapshot(hand, timeout_s=config.request_timeout_s)
    before_value = before_snapshot.position_of(actuator_id)

    _validate_target(spec, target_value)

    print("")
    print(
        f"准备设置：{axis_name} "
        f"before={before_value:.6f}, target={target_value:.6f}"
    )

    positions = [
        before_snapshot.position_of(spec_item.actuator_id)
        for spec_item in RIGHT_HAND_ACTUATOR_SPECS
    ]
    positions[int(spec.actuator_id)] = float(target_value)

    ok = hand.set_hand_state(positions)

    print(f"下发结果：{ok}")

    if not ok:
        raise RuntimeError(f"{axis_name} 单轴目标下发失败。")

    result = _poll_axis_until_unchanged_for_1s(
        hand=hand,
        config=config,
        actuator_id=actuator_id,
        target_value=target_value,
    )

    print("")
    print("单轴动作结果：")
    print(f"  axis           : {axis_name}")
    print(f"  target         : {result.target_value:.6f}")
    print(f"  actual         : {result.final_value:.6f}")
    print(f"  error          : {result.error:.6f}")
    print(f"  elapsed        : {result.elapsed_s:.3f}s")
    print(f"  samples        : {result.sample_count}")
    print(f"  stopped_reason : {result.stopped_reason}")

    _print_snapshot(result.final_snapshot)

    return result


# endregion


# region 交互菜单


def _prompt(text: str) -> str:
    """统一 input 包装，便于后续替换。"""

    return input(text).strip()


def _axis_mode(hand: WujiRightHandClient, config: HandCliConfig) -> None:
    """选轴模式。执行完一次动作后仍停留在 axis 模式。"""

    while True:
        print("")
        print("========== 右手 axis 模式 ==========")
        _print_axis_specs()

        snapshot = _read_snapshot(hand, timeout_s=config.request_timeout_s)
        _print_snapshot(snapshot)

        print("")
        print("输入轴号选择 actuator，例如：1")
        print("输入 state 重新读取状态")
        print("输入 back 返回主菜单")
        print("输入 q 退出程序")

        user_axis = _prompt("axis> ")

        if user_axis in {"back", "b"}:
            return

        if user_axis in {"q", "quit", "exit"}:
            raise KeyboardInterrupt

        if user_axis == "state":
            continue

        try:
            spec = _resolve_axis_by_user_index(user_axis)
        except ValueError as error:
            print(f"输入错误：{error}")
            continue

        axis_name = _axis_name_from_spec(spec)

        while True:
            target_text = _prompt(f"{axis_name} target> ")

            if target_text in {"back", "b"}:
                break

            if target_text in {"q", "quit", "exit"}:
                raise KeyboardInterrupt

            if target_text == "":
                print("目标值不能为空。")
                continue

            try:
                target_value = float(target_text)
            except ValueError:
                print(f"目标值不是有效数字：{target_text}")
                continue

            try:
                _run_single_axis_motion(
                    hand=hand,
                    config=config,
                    spec=spec,
                    target_value=target_value,
                )
            except Exception as error:
                logger.exception("单轴控制失败")
                print(f"单轴控制失败：{error}")

            # 执行完后回到 axis 模式，而不是继续问同一个轴 target。
            break


def _main_menu(hand: WujiRightHandClient, config: HandCliConfig) -> None:
    """主菜单。"""

    while True:
        print("")
        print("========== 右手主菜单 ==========")
        print("enable  : 开启右手使能，然后回到主菜单")
        print("disable : 关闭右手使能，然后回到主菜单")
        print("state   : 打印当前使能状态和各轴状态")
        print("axis    : 进入选轴模式")
        print("q       : 退出")

        command = _prompt("main> ")

        if command in {"q", "quit", "exit"}:
            return

        if command == "enable":
            ok = _set_enable(hand, enable=True, timeout_s=config.request_timeout_s)
            enabled = _read_enable(hand, timeout_s=config.request_timeout_s)
            print(f"enable 下发结果：{ok}")
            print(f"当前使能状态：{enabled}")
            continue

        if command == "disable":
            ok = _set_enable(hand, enable=False, timeout_s=config.request_timeout_s)
            enabled = _read_enable(hand, timeout_s=config.request_timeout_s)
            print(f"disable 下发结果：{ok}")
            print(f"当前使能状态：{enabled}")
            continue

        if command == "state":
            _print_startup_status(hand, config)
            continue

        if command == "axis":
            _axis_mode(hand, config)
            continue

        if command == "":
            continue

        print(f"未知命令：{command}")


# endregion


# region 主入口


def main() -> None:
    """F5 直接运行入口。"""

    config = HandCliConfig()

    logger.info("connect qmlinker: {}", config.target())

    base_client = _create_base_client(config)
    hand = WujiRightHandClient(base_client)

    try:
        _print_startup_status(hand, config)
        _main_menu(hand, config)
    except KeyboardInterrupt:
        print("")
        print("退出。")
    finally:
        base_client.close()


if __name__ == "__main__":
    main()


# endregion
