from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CameraIntrinsicsProtocol(Protocol):
    """相机针孔内参协议。

    该协议仅约束点云投影与抓取估计所需字段，不依赖具体相机实现类。
    任意对象只要提供以下属性即可作为内参输入：
    - `width`、`height`：图像宽高，单位 像素。
    - `fx`、`fy`：焦距，单位 像素。
    - `cx`、`cy`：主点坐标，单位 像素。
    """

    @property
    def width(self) -> int:
        """图像宽度，单位 像素。"""
        ...

    @property
    def height(self) -> int:
        """图像高度，单位 像素。"""
        ...

    @property
    def fx(self) -> float:
        """X 方向焦距，单位 像素。"""
        ...

    @property
    def fy(self) -> float:
        """Y 方向焦距，单位 像素。"""
        ...

    @property
    def cx(self) -> float:
        """主点 X 坐标，单位 像素。"""
        ...

    @property
    def cy(self) -> float:
        """主点 Y 坐标，单位 像素。"""
        ...
