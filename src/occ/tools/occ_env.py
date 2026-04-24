from __future__ import annotations

import os
import sys


# region OCC 环境初始化
def ensure_occ_casroot() -> None:
    """在 Windows 下确保 OCC 运行所需的 CASROOT 已就绪。"""
    if sys.platform != "win32":
        return

    import OCC

    if "CASROOT" in os.environ:
        casroot_path = os.environ["CASROOT"]
        if not os.path.isdir(casroot_path):
            raise AssertionError(
                f"Please set the CASROOT env variable ({casroot_path} is not ok)"
            )
        return

    occ_package_path = os.path.dirname(OCC.__file__)
    casroot_path = os.path.join(
        occ_package_path, "..", "..", "..", "Library", "share", "oce"
    )
    shaders_dict_found = os.path.isdir(os.path.join(casroot_path, "src", "Shaders"))
    unitlexicon_found = os.path.isfile(
        os.path.join(casroot_path, "src", "UnitsAPI", "Lexi_Expr.dat")
    )
    unitsdefinition_found = os.path.isfile(
        os.path.join(casroot_path, "src", "UnitsAPI", "Units.dat")
    )
    if shaders_dict_found and unitlexicon_found and unitsdefinition_found:
        os.environ["CASROOT"] = casroot_path


ensure_occ_casroot()
# endregion
