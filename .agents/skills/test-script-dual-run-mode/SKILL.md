---
name: test-script-dual-run-mode
description: 统一本仓库测试脚本为 CLI 可调用与 IDE 直跑双模式。适用于 `test/` 与实验性测试入口，要求默认常量驱动、参数可覆盖、无可视化批量可跑，并明确测试逻辑仅以最新实现为准，不保留旧行为兼容分支。
---

# Test Script Dual Run Mode

## 目标

统一测试脚本设计为：

1. 人类用户可在 IDE 中直接运行 `main()` 并通过修改文件顶部默认常量调试。
2. `Codex` 优先修改文件顶部默认常量或 `main()` 默认入参，而不是为测试脚本堆复杂 CLI。
3. 测试逻辑随实现演进，不为旧测试行为做兼容分支。

## 适用范围

1. 本仓库 `test/` 下脚本。
2. 实验性测试入口脚本（如 `experiments/` 中用于测试验证的脚本）。

## 强制规则

1. IDE 直跑是测试脚本的第一入口：`if __name__ == "__main__":` 时默认直接调用 `main()`，不强依赖参数解析。
2. 默认参数必须在文件顶部以常量声明，如 `DEFAULT_HOST`、`DEFAULT_PORT`、`DEFAULT_VIS`，并允许用户直接修改这些常量。
3. 默认参数常量应添加中文注释，至少说明用途与单位（例如 `# 超时时间，单位 ms`），避免仅保留英文命名。
4. 非必要不写复杂 CLI；若脚本没有明确的批量试参需求，则不新增 `argparse`、不维护 `--host/--port/...` 覆盖层。
5. 若确有批量试参需求，CLI 只能作为薄覆盖层，且默认值必须绑定顶部常量；无 CLI 入参时不应强依赖参数解析。
6. 测试脚本更新时，不允许为了兼容旧测试行为添加兼容逻辑；一切以最新实现为准。
7. 测试脚本中的 debug 信息应该使用 loguru，通常使用 info 级别。需要包括 success 和 warning 级别。并且 debug 信息应该尽量使用中文，涉及到单位的应该是`参数名 数值 单位`格式。
8. 脚本中尽量使用`#region`注释做好分区
9. 执行检查与验证时，优先使用当前项目对应的 conda 环境，不要默认落到 `base`。当 `conda` 命令在当前 shell 不可用时，优先使用该环境的绝对路径 Python（例如 `C:\\Users\\ICO\\anaconda3\\envs\\DingTai\\python.exe`）执行脚本与依赖检查。
10. 任何删除、移动、重命名测试脚本前，必须先在 `.archive/` 下建立快照，并保留从项目根目录开始的相对路径；不能先删再补快照。

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
5. 若涉及 Python 包检查或安装建议，应明确说明检查是基于哪个 conda 环境（例如 `DingTai`），避免与 `base` 环境混淆。
