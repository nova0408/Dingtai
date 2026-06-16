from __future__ import annotations
import pybind11_stubgen.typing_ext
import typing
import xCoreSDK_python
__all__: list[str] = ['Model_0_3', 'Model_0_4', 'Model_0_6', 'Model_1_5', 'Model_1_6', 'Model_1_7', 'SegmentFrame', 'TorqueType']
class Model_0_3:
    """
    用于在python中使用model类
    """
    def __init__(self, arg0: xCoreSDK_python.Robot_T_Industrial_3) -> None:
        ...
    def calcAllIkSolutions(self, posture: xCoreSDK_python.CartesianPosition, confs: list, ec: dict) -> list[list[float]]:
        """
        计算笛卡尔位姿所有逆解结果。支持除xMateSR(XMS)之外的所有机型
        
        Args:
            posture (CartesianPosition): 笛卡尔位姿，法兰相对与基座标系。其它坐标系需自行转换。
            confs (list[list[int]]): 对应的confdata，错误码为0时有效
            ec (dict): 错误码，含逆解计算失败错误：-50102奇异点 | -50114 超限位 | -50519 超范围 | -50002 其它逆解错误
        
        Returns:
            逆解结果 (list[list[float]]): 单位弧度，错误码为0时有效
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)], toolset: xCoreSDK_python.Toolset, ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, toolset: xCoreSDK_python.Toolset, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
class Model_0_4:
    """
    用于在python中使用model类
    """
    def __init__(self, arg0: xCoreSDK_python.Robot_T_Industrial_4) -> None:
        ...
    def calcAllIkSolutions(self, posture: xCoreSDK_python.CartesianPosition, confs: list, ec: dict) -> list[list[float]]:
        """
        计算笛卡尔位姿所有逆解结果。支持除xMateSR(XMS)之外的所有机型
        
        Args:
            posture (CartesianPosition): 笛卡尔位姿，法兰相对与基座标系。其它坐标系需自行转换。
            confs (list[list[int]]): 对应的confdata，错误码为0时有效
            ec (dict): 错误码，含逆解计算失败错误：-50102奇异点 | -50114 超限位 | -50519 超范围 | -50002 其它逆解错误
        
        Returns:
            逆解结果 (list[list[float]]): 单位弧度，错误码为0时有效
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(4)], ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(4)], toolset: xCoreSDK_python.Toolset, ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(4)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, toolset: xCoreSDK_python.Toolset, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(4)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
class Model_0_6:
    """
    用于在python中使用model类
    """
    def __init__(self, arg0: xCoreSDK_python.Robot_T_Industrial_6) -> None:
        ...
    def calcAllIkSolutions(self, posture: xCoreSDK_python.CartesianPosition, confs: list, ec: dict) -> list[list[float]]:
        """
        计算笛卡尔位姿所有逆解结果。支持除xMateSR(XMS)之外的所有机型
        
        Args:
            posture (CartesianPosition): 笛卡尔位姿，法兰相对与基座标系。其它坐标系需自行转换。
            confs (list[list[int]]): 对应的confdata，错误码为0时有效
            ec (dict): 错误码，含逆解计算失败错误：-50102奇异点 | -50114 超限位 | -50519 超范围 | -50002 其它逆解错误
        
        Returns:
            逆解结果 (list[list[float]]): 单位弧度，错误码为0时有效
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)], ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)], toolset: xCoreSDK_python.Toolset, ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, toolset: xCoreSDK_python.Toolset, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
class Model_1_5:
    """
    用于在python中使用model类
    """
    def __init__(self, arg0: xCoreSDK_python.Robot_T_Collaborative_5) -> None:
        ...
    def calcAllIkSolutions(self, posture: xCoreSDK_python.CartesianPosition, confs: list, ec: dict) -> list[list[float]]:
        """
        计算笛卡尔位姿所有逆解结果。支持除xMateSR(XMS)之外的所有机型
        
        Args:
            posture (CartesianPosition): 笛卡尔位姿，法兰相对与基座标系。其它坐标系需自行转换。
            confs (list[list[int]]): 对应的confdata，错误码为0时有效
            ec (dict): 错误码，含逆解计算失败错误：-50102奇异点 | -50114 超限位 | -50519 超范围 | -50002 其它逆解错误
        
        Returns:
            逆解结果 (list[list[float]]): 单位弧度，错误码为0时有效
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(5)], ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(5)], toolset: xCoreSDK_python.Toolset, ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(5)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, toolset: xCoreSDK_python.Toolset, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(5)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
class Model_1_6:
    """
    用于在python中使用model类
    """
    def __init__(self, arg0: xCoreSDK_python.Robot_T_Collaborative_6) -> None:
        ...
    def calcAllIkSolutions(self, posture: xCoreSDK_python.CartesianPosition, confs: list, ec: dict) -> list[list[float]]:
        """
        计算笛卡尔位姿所有逆解结果。支持除xMateSR(XMS)之外的所有机型
        
        Args:
            posture (CartesianPosition): 笛卡尔位姿，法兰相对与基座标系。其它坐标系需自行转换。
            confs (list[list[int]]): 对应的confdata，错误码为0时有效
            ec (dict): 错误码，含逆解计算失败错误：-50102奇异点 | -50114 超限位 | -50519 超范围 | -50002 其它逆解错误
        
        Returns:
            逆解结果 (list[list[float]]): 单位弧度，错误码为0时有效
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)], ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)], toolset: xCoreSDK_python.Toolset, ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, toolset: xCoreSDK_python.Toolset, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(6)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
class Model_1_7:
    """
    用于在python中使用model类
    """
    def __init__(self, arg0: xCoreSDK_python.Robot_T_Collaborative_7) -> None:
        ...
    def calcAllIkSolutions(self, posture: xCoreSDK_python.CartesianPosition, confs: list, ec: dict) -> list[list[float]]:
        """
        计算笛卡尔位姿所有逆解结果。支持除xMateSR(XMS)之外的所有机型
        
        Args:
            posture (CartesianPosition): 笛卡尔位姿，法兰相对与基座标系。其它坐标系需自行转换。
            confs (list[list[int]]): 对应的confdata，错误码为0时有效
            ec (dict): 错误码，含逆解计算失败错误：-50102奇异点 | -50114 超限位 | -50519 超范围 | -50002 其它逆解错误
        
        Returns:
            逆解结果 (list[list[float]]): 单位弧度，错误码为0时有效
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(7)], ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcFk(self, joints: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(7)], toolset: xCoreSDK_python.Toolset, ec: dict) -> xCoreSDK_python.CartesianPosition:
        """
        根据轴角度计算正解
        
        Args:
            joints (list[float]): 轴角度列表，单位：弧度
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            CartesianPosition: 机器人末端位姿，相对于外部参考坐标系
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(7)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
    @typing.overload
    def calcIk(self, posture: xCoreSDK_python.CartesianPosition, toolset: xCoreSDK_python.Toolset, ec: dict) -> typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(7)]:
        """
        根据位姿计算逆解
        
        Args:
            posture (CartesianPosition): 机器人末端位姿，相对于外部参考坐标系
            toolset (Toolset): 工具工件坐标系设置
            ec (dict): 错误码
        
        Returns:
            list[float]: 轴角度，单位弧度
        """
class SegmentFrame:
    """
    连杆标号
    
    Members:
    
      joint1
    
      joint2
    
      joint3
    
      joint4
    
      joint5
    
      joint6
    
      joint7
    
      flange
    
      endEffector
    
      stiffness
    """
    __members__: typing.ClassVar[dict[str, SegmentFrame]]  # value = {'joint1': <SegmentFrame.joint1: 1>, 'joint2': <SegmentFrame.joint2: 2>, 'joint3': <SegmentFrame.joint3: 3>, 'joint4': <SegmentFrame.joint4: 4>, 'joint5': <SegmentFrame.joint5: 5>, 'joint6': <SegmentFrame.joint6: 6>, 'joint7': <SegmentFrame.joint7: 7>, 'flange': <SegmentFrame.flange: 8>, 'endEffector': <SegmentFrame.endEffector: 9>, 'stiffness': <SegmentFrame.stiffness: 10>}
    endEffector: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.endEffector: 9>
    flange: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.flange: 8>
    joint1: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.joint1: 1>
    joint2: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.joint2: 2>
    joint3: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.joint3: 3>
    joint4: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.joint4: 4>
    joint5: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.joint5: 5>
    joint6: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.joint6: 6>
    joint7: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.joint7: 7>
    stiffness: typing.ClassVar[SegmentFrame]  # value = <SegmentFrame.stiffness: 10>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class TorqueType:
    """
    力矩类型
    
    Members:
    
      full : 关节力矩，由动力学模型计算得到
    
      inertia : 惯性力
    
      coriolis :  科氏力
    
      friction : 摩擦力
    
      gravity : 重力
    """
    __members__: typing.ClassVar[dict[str, TorqueType]]  # value = {'full': <TorqueType.full: 0>, 'inertia': <TorqueType.inertia: 1>, 'coriolis': <TorqueType.coriolis: 2>, 'friction': <TorqueType.friction: 3>, 'gravity': <TorqueType.gravity: 4>}
    coriolis: typing.ClassVar[TorqueType]  # value = <TorqueType.coriolis: 2>
    friction: typing.ClassVar[TorqueType]  # value = <TorqueType.friction: 3>
    full: typing.ClassVar[TorqueType]  # value = <TorqueType.full: 0>
    gravity: typing.ClassVar[TorqueType]  # value = <TorqueType.gravity: 4>
    inertia: typing.ClassVar[TorqueType]  # value = <TorqueType.inertia: 1>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
