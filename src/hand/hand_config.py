from __future__ import annotations

from pathlib import Path
import tomllib

from src.hand.wuji_hand_protocol import (
    DEFAULT_WUJI_HAND_INSTANCES,
    WUJI_HAND_SPECS,
    HandSpecName,
    WujiHandInstanceSpec,
)

# region 配置

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_NETWORK_CONFIG_PATH = PROJECT_ROOT / "config" / "robot_network.toml"


# endregion


# region 主入口

def load_wuji_hand_instances(path: Path | None = None) -> tuple[WujiHandInstanceSpec, ...]:
    """读取无际左右手实例配置。

    Parameters
    ----------
    path:
        可选配置路径，为 `None` 时读取项目根目录下 `config/robot_network.toml`。

    Returns
    -------
    tuple[WujiHandInstanceSpec, ...]
        左右手实例配置。配置文件缺失或手型未知时使用默认 `hand_3`。

    Notes
    -----
    该函数只读取配置和校验手型名称，不创建 qmlinker `QMHand`，也不连接硬件。
    """

    config_path = ROBOT_NETWORK_CONFIG_PATH if path is None else path
    if not config_path.exists():
        return DEFAULT_WUJI_HAND_INSTANCES

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    hand_data = data.get("hand", {})
    left_spec = _coerce_hand_spec(hand_data.get("left", {}).get("spec", "hand_3"))
    right_spec = _coerce_hand_spec(hand_data.get("right", {}).get("spec", "hand_3"))
    return (
        WujiHandInstanceSpec("left_hand", "left hand", left_spec),
        WujiHandInstanceSpec("right_hand", "right hand", right_spec),
    )


# endregion


# region 基础工具

def _coerce_hand_spec(value: object) -> HandSpecName:
    """将配置值收窄为项目已知手型。

    Parameters
    ----------
    value:
        TOML 中读取到的手型配置值。

    Returns
    -------
    HandSpecName
        已知手型名称。未知值回退为 `hand_3`。
    """

    spec = str(value)
    if spec in WUJI_HAND_SPECS:
        return spec  # type: ignore[return-value]
    return "hand_3"


# endregion
