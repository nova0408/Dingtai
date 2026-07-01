"""远端 Orin 统一服务入口。

该文件作为仓库根目录 shim，兼容远端继续使用
`python -m service <service_name> ...` 的启动方式，
实际逻辑统一转发到 `orin.service`。
"""

from __future__ import annotations

from orin.service import main as orin_service_main


def main(argv=None) -> int:
    """转发到 `orin.service.main`。"""

    return int(orin_service_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
