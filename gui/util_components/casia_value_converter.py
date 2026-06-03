from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class CasiaValueConverter(Protocol):
    """
    滑块值转换器协议。

    该协议用于隔离滑块内部整数值与界面展示/输入文本之间的转换关系。
    """

    def convert(self, value: int) -> str:
        """
        将滑块整数值转换为显示文本。

        Parameters
        ----------
        value:
            QSlider 当前整数值。

        Returns
        -------
        str
            显示在滑块中心的文本。
        """
        raise NotImplementedError

    def convert_back(self, text: str) -> int:
        """
        将用户输入文本转换为滑块整数值。

        Parameters
        ----------
        text:
            用户在编辑框中输入的文本。

        Returns
        -------
        int
            可传递给 QSlider.setValue 的整数值。
        """
        raise NotImplementedError


@runtime_checkable
class CasiaEditableValueConverter(Protocol):
    """
    可选的编辑文本转换协议。

    实现该协议后，控件进入点击编辑模式时会使用 ``convert_edit`` 的返回值作为
    输入框文本，避免单位、提示语等展示性内容被选中和编辑。
    """

    def convert_edit(self, value: int) -> str:
        """
        将滑块整数值转换为可编辑文本。

        Parameters
        ----------
        value:
            QSlider 当前整数值。

        Returns
        -------
        str
            仅包含用户应编辑内容的文本。
        """
        raise NotImplementedError


class IntValueConverter:
    """
    默认整数转换器。

    该转换器直接显示和解析 QSlider 的原始整数值。
    """

    def convert(self, value: int) -> str:
        return str(value)

    def convert_edit(self, value: int) -> str:
        return str(value)

    def convert_back(self, text: str) -> int:
        return int(text.strip())


class CallableValueConverter:
    """
    兼容旧接口的只读转换器。

    旧代码中传入的 ``Callable[[int], str]`` 只定义了显示转换，因此输入解析仍使用
    默认整数解析逻辑。
    """

    def __init__(self, converter: Callable[[int], str]):
        self._converter = converter
        self._fallback = IntValueConverter()

    def convert(self, value: int) -> str:
        return self._converter(value)

    def convert_edit(self, value: int) -> str:
        return self._fallback.convert_edit(value)

    def convert_back(self, text: str) -> int:
        return self._fallback.convert_back(text)
