from __future__ import annotations

from typing import TYPE_CHECKING

from src.hand.wuji_hand_protocol import WujiHandInstanceSpec

if TYPE_CHECKING:
    from src.wuji.qmlinker_client import WujiQmlinkerClient

# region 主入口


def load_wuji_hand_instances(client: WujiQmlinkerClient | None = None) -> tuple[WujiHandInstanceSpec, ...]:
    """从 qmlinker 运行时读取左右手执行器规格。

    Parameters
    ----------
    client:
        可选 qmlinker 客户端。为 `None` 时会临时创建客户端并读取当前手部状态。

    Returns
    -------
    tuple[WujiHandInstanceSpec, ...]
        左右手实例规格。若 qmlinker 不可用，则返回空元组。

    Notes
    -----
    该函数只读取当前连接的 qmlinker 设备状态，不再依赖本地 TOML 中的手型模板。
    """

    from src.wuji.qmlinker_client import WujiQmlinkerClient

    owns_client = client is None
    runtime_client = WujiQmlinkerClient() if owns_client else client
    if runtime_client is None:
        return ()
    try:
        return (
            WujiHandInstanceSpec("left_hand", "left hand", len(runtime_client.get_hand_values("left_hand"))),
            WujiHandInstanceSpec("right_hand", "right hand", len(runtime_client.get_hand_values("right_hand"))),
        )
    except Exception:
        return ()
    finally:
        if owns_client:
            runtime_client.close()


# endregion
