---
name: test-script-dual-run-mode
description: 统一本仓库测试脚本为 CLI 可调用与 IDE 直跑双模式。适用于 `test/` 与实验性测试入口，要求默认常量驱动、参数可覆盖、无可视化批量可跑，并明确测试逻辑仅以最新实现为准，不保留旧行为兼容分支。
---

# Test Script Dual Run Mode

## 目标

统一测试脚本设计为：

1. `Codex` 可通过 CLI 参数直接调用运行。
2. 人类用户可在 IDE 中直接运行 `main()` 并通过修改默认常量调试。
3. 测试逻辑随实现演进，不为旧测试行为做兼容分支。

## 适用范围

1. 本仓库 `test/` 下脚本。
2. 实验性测试入口脚本（如 `experiments/` 中用于测试验证的脚本）。

## 强制规则

1. 必须同时支持两种运行方式：
   - IDE 直跑：`if __name__ == "__main__":` 时优先使用文件顶部默认常量。
   - CLI 参数运行：仅用于覆盖默认常量做调参测试（典型用于 Codex 批量试参）。
2. 默认参数必须在文件顶部以常量声明，如 `DEFAULT_SRC`、`DEFAULT_TGT`、`DEFAULT_VIS`，并允许用户直接修改这些常量。
3. CLI 参数默认值必须绑定默认常量；无 CLI 入参时不应强依赖参数解析。
4. 测试脚本更新时，不允许为了兼容旧测试行为添加兼容逻辑；一切以最新实现为准。
5. 测试脚本中的 debug 信息应该使用 loguru，通常使用 info 级别。需要包括 success 和 warning 级别。并且 debug 信息应该尽量使用中文，涉及到单位的应该是`参数名 数值 单位`格式。
6. 脚本中尽量使用`#region`注释做好分区

## 推荐实现模板

```python
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SRC = Path("experiments/pcd2.pcd")
DEFAULT_TGT = Path("experiments/pcd1.pcd")
DEFAULT_VIS = False


def main(src_path: Path = DEFAULT_SRC, tgt_path: Path = DEFAULT_TGT, vis: bool = DEFAULT_VIS) -> None:
    ...


def _parse_cli() -> tuple[Path, Path, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--tgt", type=Path, default=DEFAULT_TGT)
    parser.add_argument("--vis", action=argparse.BooleanOptionalAction, default=DEFAULT_VIS)
    args = parser.parse_args()
    return args.src, args.tgt, bool(args.vis)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        src_arg, tgt_arg, vis_arg = _parse_cli()
        main(src_path=src_arg, tgt_path=tgt_arg, vis=vis_arg)
    else:
        main()
```

## 输出与验证要求

1. 输出应包含关键指标与最终变换信息，便于参数对比。
2. 可视化应始终可关闭，便于无界面批量测试。
3. 在硬件或 GUI 未实测时，日志中应明确标注验证边界。
4. `Codex`在验证时应该设置一个超时时间，若实际脚本运行超过 1min，则应该终止，由人类用户自行运行验证。
