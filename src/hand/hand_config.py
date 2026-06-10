from __future__ import annotations

from typing import TYPE_CHECKING

from src.hand.wuji_hand_protocol import WujiHandInstanceSpec

if TYPE_CHECKING:
    from src.wuji.client_base import WujiQmlinkerBaseClient
    from src.wuji.right_hand_client import WujiRightHandClient

# region 主入口


def load_wuji_hand_instances(client: WujiRightHandClient | None = None) -> tuple[WujiHandInstanceSpec, ...]:
    """从 qmlinker 运行时读取右手执行器规格。

    Parameters
    ----------
    client:
        可选 qmlinker 客户端。为 `None` 时会临时创建客户端并读取当前手部状态。

    Returns
    -------
    tuple[WujiHandInstanceSpec, ...]
        右手实例规格。若 qmlinker 不可用，则返回空元组。

    Notes
    -----
    该函数只读取当前连接的 qmlinker 设备状态，不再依赖本地 TOML 中的手型模板。
    """

    from src.wuji.client_base import WujiQmlinkerBaseClient
    from src.wuji.right_hand_client import WujiRightHandClient

    owns_client = client is None
    if owns_client:
        runtime_base = WujiQmlinkerBaseClient()
        runtime_client = WujiRightHandClient(runtime_base)
    else:
        runtime_client = client
    try:
        return runtime_client.get_hand_instance_specs()
    except Exception:
        return ()
    finally:
        if owns_client:
            runtime_client._base.close()


# endregion
