from __future__ import annotations

# region 依赖导入

import json
from pathlib import Path

# endregion


# region 配置

DEFAULT_SSH_CONFIG_PATH = Path.home() / ".ssh" / "config"
"当前用户 SSH 配置文件路径。"

DEFAULT_CACHE_PATH = Path(__file__).resolve().with_name("network_discovery_cache.json")
"本地测试主机发现缓存文件路径。"

DEFAULT_ORIN_SSH_ALIAS = "orin"
"Orin 的本机 SSH Host 别名。"

DEFAULT_WUYOU_SSH_ALIAS = "wuyou"
"wuyou 的本机 SSH Host 别名。"

DEFAULT_ORIN_FALLBACKS = (
    "192.168.1.118",
    "wujibrain-desktop.local",
    "192.168.100.60",
)
"Orin 的候选主机列表。"

DEFAULT_WUYOU_FALLBACKS = (
    "192.168.1.119",
    "wuyou-X1-SBC.local",
    "192.168.1.113",
    "192.168.100.60",
)
"wuyou 的候选主机列表。"

_HOST_CACHE: dict[str, str] | None = None

# endregion


# region 主入口


def get_cached_orin_host() -> str:
    """返回 Orin 当前默认主机。

    Notes
    -----
    该函数只读取缓存和静态候选，不做网络探测，避免在脚本启动阶段卡顿。
    真正的候选轮询只在连接失败后由调用方触发。
    """

    return get_cached_host(DEFAULT_ORIN_SSH_ALIAS, DEFAULT_ORIN_FALLBACKS)


def get_cached_wuyou_host() -> str:
    """返回 wuyou 当前默认主机。"""

    return get_cached_host(DEFAULT_WUYOU_SSH_ALIAS, DEFAULT_WUYOU_FALLBACKS)


def get_cached_host(alias: str, fallbacks: tuple[str, ...]) -> str:
    """返回指定设备当前缓存主机。"""

    cache = _load_cache()
    cached_host = _normalize_host(cache.get(alias))
    if cached_host is not None:
        return cached_host

    ssh_host = _read_ssh_alias_host(alias)
    if ssh_host is not None:
        return ssh_host

    for host in fallbacks:
        normalized_host = _normalize_host(host)
        if normalized_host is not None:
            return normalized_host
    raise RuntimeError(f"no cached host for alias {alias}")


def iter_candidate_hosts(alias: str, fallbacks: tuple[str, ...], preferred_host: str | None = None) -> tuple[str, ...]:
    """返回失败后用于重试的候选主机列表。"""

    ordered_hosts: list[str] = []
    for item in (
        preferred_host,
        _load_cache().get(alias),
        _read_ssh_alias_host(alias),
        *fallbacks,
    ):
        normalized_host = _normalize_host(item)
        if normalized_host is None:
            continue
        if normalized_host in ordered_hosts:
            continue
        ordered_hosts.append(normalized_host)
    return tuple(ordered_hosts)


def remember_host(alias: str, host: str) -> None:
    """写回本地成功主机缓存。"""

    normalized_host = _normalize_host(host)
    if normalized_host is None:
        return
    cache = _load_cache()
    cache[alias] = normalized_host
    DEFAULT_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# endregion


# region 缓存与 SSH 配置


def _load_cache() -> dict[str, str]:
    """读取本地缓存。"""

    global _HOST_CACHE
    if _HOST_CACHE is not None:
        return _HOST_CACHE

    if not DEFAULT_CACHE_PATH.is_file():
        _HOST_CACHE = {}
        return _HOST_CACHE

    try:
        payload = json.loads(DEFAULT_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _HOST_CACHE = {}
        return _HOST_CACHE

    if not isinstance(payload, dict):
        _HOST_CACHE = {}
        return _HOST_CACHE

    cache: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        normalized_host = _normalize_host(value)
        if normalized_host is None:
            continue
        cache[key] = normalized_host
    _HOST_CACHE = cache
    return _HOST_CACHE


def _read_ssh_alias_host(alias: str) -> str | None:
    """读取本机 SSH 配置中指定 Host 的 HostName。"""

    if not DEFAULT_SSH_CONFIG_PATH.is_file():
        return None

    current_aliases: tuple[str, ...] = ()
    for raw_line in DEFAULT_SSH_CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

        key, _, value = line.partition(" ")
        if not value:
            continue
        normalized_key = key.strip().lower()
        normalized_value = value.strip()

        if normalized_key == "host":
            current_aliases = tuple(item.lower() for item in normalized_value.split() if item)
            continue
        if normalized_key == "hostname" and alias.lower() in current_aliases:
            return _normalize_host(normalized_value)
    return None


# endregion


# region 基础工具


def _normalize_host(host: object) -> str | None:
    """清洗主机字符串。"""

    if host is None:
        return None
    stripped_host = str(host).strip()
    if not stripped_host:
        return None
    return stripped_host


# endregion
