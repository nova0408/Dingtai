"""Orin 侧边缘计算相关模块。"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any


def _patch_dataclass_slots_for_py38() -> None:
    """为 Python 3.8 提供 `dataclass(slots=...)` 兼容层。"""

    if sys.version_info >= (3, 10):
        return
    if getattr(dataclasses.dataclass, "__name__", "") == "_compat_dataclass":
        return

    original_dataclass = dataclasses.dataclass

    def _compat_dataclass(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("slots", None)
        return original_dataclass(*args, **kwargs)

    dataclasses.dataclass = _compat_dataclass


_patch_dataclass_slots_for_py38()
