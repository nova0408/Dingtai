"""OpenCV 预定义 ArUco 字典解析。"""

from __future__ import annotations

import cv2


def available_aruco_dictionary_names() -> tuple[str, ...]:
    """返回当前 OpenCV 暴露的去重、规范化预定义字典名称。"""

    names = {
        name.upper()
        for name, value in vars(cv2.aruco).items()
        if name.startswith("DICT_") and isinstance(value, int)
    }
    return tuple(sorted(names))


def get_predefined_aruco_dictionary(dictionary_name: str):
    """按 OpenCV 当前可用名称构造原生预定义字典。"""

    normalized_name = dictionary_name.upper()
    dictionary_ids = {
        name.upper(): int(value)
        for name, value in vars(cv2.aruco).items()
        if name.startswith("DICT_") and isinstance(value, int)
    }
    try:
        dictionary_id = dictionary_ids[normalized_name]
    except KeyError as error:
        raise ValueError(
            f"unsupported dictionary: {dictionary_name}; available={available_aruco_dictionary_names()}"
        ) from error
    return cv2.aruco.getPredefinedDictionary(dictionary_id)

