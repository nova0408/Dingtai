from __future__ import annotations

import numpy as np
from pyorbbecsdk import OBCameraDistortion, OBCameraIntrinsic, OBCameraParam, OBExtrinsic

from .orbbec_models import CameraParamPatch, DistortionPatch, IntrinsicPatch


def clone_camera_param(source: OBCameraParam) -> OBCameraParam:
    """
    深拷贝相机参数对象。

    Parameters
    ----------
    source : OBCameraParam
        源参数对象。

    Returns
    -------
    OBCameraParam
        拷贝后的新对象。
    """
    cloned = OBCameraParam()
    cloned.depth_intrinsic = _clone_intrinsic(source.depth_intrinsic)
    cloned.rgb_intrinsic = _clone_intrinsic(source.rgb_intrinsic)
    cloned.depth_distortion = _clone_distortion(source.depth_distortion)
    cloned.rgb_distortion = _clone_distortion(source.rgb_distortion)
    cloned.transform = _clone_extrinsic(source.transform)
    return cloned


def apply_camera_param_patch(camera_param: OBCameraParam, patch: CameraParamPatch) -> OBCameraParam:
    """
    将补丁应用到相机参数对象。

    Parameters
    ----------
    camera_param : OBCameraParam
        待修改参数对象。
    patch : CameraParamPatch
        参数补丁。

    Returns
    -------
    OBCameraParam
        修改后的同一对象（原地更新）。
    """
    depth_intrinsic = _apply_intrinsic_patch(camera_param.depth_intrinsic, patch.depth)
    color_intrinsic = _apply_intrinsic_patch(camera_param.rgb_intrinsic, patch.color)
    depth_distortion = _apply_distortion_patch(camera_param.depth_distortion, patch.depth_dist)
    color_distortion = _apply_distortion_patch(camera_param.rgb_distortion, patch.color_dist)
    transform = _apply_extrinsic_patch(camera_param.transform, patch.d2c_translation_offset_mm)

    camera_param.depth_intrinsic = depth_intrinsic
    camera_param.rgb_intrinsic = color_intrinsic
    camera_param.depth_distortion = depth_distortion
    camera_param.rgb_distortion = color_distortion
    camera_param.transform = transform
    return camera_param


def camera_param_summary(name: str, camera_param: OBCameraParam) -> str:
    """
    生成人类可读的相机参数摘要文本。

    Parameters
    ----------
    name : str
        摘要标签名。
    camera_param : OBCameraParam
        相机参数对象。

    Returns
    -------
    str
        单行摘要字符串。
    """
    di = camera_param.depth_intrinsic
    ci = camera_param.rgb_intrinsic
    dd = camera_param.depth_distortion
    d2c_t = np.asarray(camera_param.transform.transform, dtype=np.float32).reshape(3)
    return (
        f"[{name}] depth(fx={di.fx:.2f}, fy={di.fy:.2f}, cx={di.cx:.2f}, cy={di.cy:.2f}) "
        f"color(fx={ci.fx:.2f}, fy={ci.fy:.2f}, cx={ci.cx:.2f}, cy={ci.cy:.2f}) "
        f"depth_dist(k1={dd.k1:.5f}, k2={dd.k2:.5f}) "
        f"d2c_t=({d2c_t[0]:.2f}, {d2c_t[1]:.2f}, {d2c_t[2]:.2f})mm"
    )


def _clone_intrinsic(source: OBCameraIntrinsic) -> OBCameraIntrinsic:
    """
    深拷贝相机内参。

    Parameters
    ----------
    source : OBCameraIntrinsic
        源内参对象。

    Returns
    -------
    OBCameraIntrinsic
        拷贝结果。
    """
    cloned = OBCameraIntrinsic()
    cloned.fx = float(source.fx)
    cloned.fy = float(source.fy)
    cloned.cx = float(source.cx)
    cloned.cy = float(source.cy)
    cloned.width = int(source.width)
    cloned.height = int(source.height)
    return cloned


def _clone_distortion(source: OBCameraDistortion) -> OBCameraDistortion:
    """
    深拷贝畸变参数。

    Parameters
    ----------
    source : OBCameraDistortion
        源畸变参数对象。

    Returns
    -------
    OBCameraDistortion
        拷贝结果。
    """
    cloned = OBCameraDistortion()
    cloned.k1 = float(source.k1)
    cloned.k2 = float(source.k2)
    cloned.k3 = float(source.k3)
    cloned.k4 = float(source.k4)
    cloned.k5 = float(source.k5)
    cloned.k6 = float(source.k6)
    cloned.p1 = float(source.p1)
    cloned.p2 = float(source.p2)
    return cloned


def _clone_extrinsic(source: OBExtrinsic) -> OBExtrinsic:
    """
    深拷贝外参参数。

    Parameters
    ----------
    source : OBExtrinsic
        源外参对象。

    Returns
    -------
    OBExtrinsic
        拷贝结果。
    """
    cloned = OBExtrinsic()
    cloned.rot = np.asarray(source.rot, dtype=np.float32).copy().reshape(3, 3)
    cloned.transform = np.asarray(source.transform, dtype=np.float32).copy().reshape(3)
    return cloned


def _apply_intrinsic_patch(intrinsic: OBCameraIntrinsic, patch: IntrinsicPatch) -> OBCameraIntrinsic:
    """
    将内参补丁应用到对象（原地修改）。

    Parameters
    ----------
    intrinsic : OBCameraIntrinsic
        待修改内参对象。
    patch : IntrinsicPatch
        内参补丁。

    Returns
    -------
    OBCameraIntrinsic
        修改后的同一对象。
    """
    intrinsic.fx = float(intrinsic.fx * patch.fx_scale)
    intrinsic.fy = float(intrinsic.fy * patch.fy_scale)
    intrinsic.cx = float(intrinsic.cx + patch.cx_offset)
    intrinsic.cy = float(intrinsic.cy + patch.cy_offset)
    return intrinsic


def _apply_distortion_patch(distortion: OBCameraDistortion, patch: DistortionPatch) -> OBCameraDistortion:
    """
    将畸变补丁应用到对象（原地修改）。

    Parameters
    ----------
    distortion : OBCameraDistortion
        待修改畸变对象。
    patch : DistortionPatch
        畸变补丁。

    Returns
    -------
    OBCameraDistortion
        修改后的同一对象。
    """
    distortion.k1 = float(distortion.k1 + patch.k1_offset)
    distortion.k2 = float(distortion.k2 + patch.k2_offset)
    distortion.p1 = float(distortion.p1 + patch.p1_offset)
    distortion.p2 = float(distortion.p2 + patch.p2_offset)
    return distortion


def _apply_extrinsic_patch(extrinsic: OBExtrinsic, translation_offset_mm: tuple[float, float, float]) -> OBExtrinsic:
    """
    对外参平移分量施加偏移（原地修改）。

    Parameters
    ----------
    extrinsic : OBExtrinsic
        待修改外参对象。
    translation_offset_mm : tuple[float, float, float]
        X/Y/Z 平移偏移量（毫米）。

    Returns
    -------
    OBExtrinsic
        修改后的同一对象。
    """
    translation = np.asarray(extrinsic.transform, dtype=np.float32).reshape(3)
    translation += np.asarray(translation_offset_mm, dtype=np.float32).reshape(3)
    extrinsic.transform = translation
    return extrinsic
