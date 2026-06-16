from __future__ import annotations
import xCoreSDK_python
__all__: list[str] = ['BaseEthercat', 'SDOData', 'SlaveInfo']
class BaseEthercat:
    def GetSlaveCount(self, ec: dict) -> int:
        """
        获取从站数量
        
        Args:
            ec (dict): 错误码输出
        
        Returns:
            int: 从站数量
        """
    def GetSlaveInfo(self, slave_addr: int, ec: dict) -> ...:
        """
        获取从站信息
        
        Args:
            slave_addr: 从站地址
            ec (dict): 错误码输出
        
        Returns:
            从站信息
        """
    def GetSlaveState(self, slave_addr: int, ec: dict) -> int:
        """
        获取某个从站状态
        
        Args:
            slave_addr: 从站地址
            ec (dict): 错误码输出
        
        Returns:
            从站状态
        """
    def GetSlavesInfo(self, ec: dict) -> list[...]:
        """
        获取所有从站信息
        
        Args:
            ec (dict): 错误码输出
        
        Returns:
            所有从站信息
        """
    def ReadPDO(self, slave_addr: int, offset: int, size: int, data: list, ec: dict) -> bool:
        """
        读PDO
        
        Args:
            slave_addr: 从站地址
            offset: pdo偏移
            size: pdo长度
            data: 数据
            ec (dict): 错误码输出
        
        Returns:
            bool: 是否成功
        """
    def ReadSDO(self, slave_addr: int, index: int, sub_index: int, length: int, data: list, over_time: int, ec: dict) -> bool:
        """
        读SDO
                            
        Args:
            slave_addr: 从站地址
            index: 索引
            sub_index: 子索引
            length: 长度
            data: 数据
            over_time: 超时时间
            ec (dict): 错误码输出
        
        Returns:
            bool: 是否成功
        """
    def SetSlavesState(self, state: int, ec: dict) -> bool:
        """
        从站全部切状态
        
        Args:
            state: 从站状态
            ec (dict): 错误码输出
        
        Returns:
            bool: 是否成功
        """
    def WriteMultiSDO(self, slave_addr: int, SDO_data: list, ec: dict) -> bool:
        """
        写多个SDO
        
        Args:
            slave_addr: 从站地址
            SDO_data: 数据
            ec (dict): 错误码输出
        
        Returns:
            bool: 是否成功
        """
    def WritePDO(self, slave_addr: int, offset: int, size: int, data: list, ec: dict) -> bool:
        """
        写PDO
        
        Args:
            slave_addr: 从站地址
            offset: pdo偏移
            size: pdo长度
            data: 数据
            ec (dict): 错误码输出
        
        Returns:
            bool: 是否成功
        """
    def WriteSDO(self, slave_addr: int, index: int, sub_index: int, length: int, data: list, over_time: int, ec: dict) -> bool:
        """
        写SDO
        
        Args:
            slave_addr: 从站地址
            index: 索引
            sub_index: 子索引
            length: 长度
            data: 数据
            over_time: 超时时间
            ec (dict): 错误码输出
        
        Returns:
            bool: 是否成功
        """
    def __init__(self, arg0: xCoreSDK_python.BaseRobot) -> None:
        ...
class SDOData:
    """
    SDO数据信息
    """
    data: list
    index: int
    length: int
    over_time: int
    print_data: int
    sub_index: int
    wait_time: int
    def __init__(self) -> None:
        ...
class SlaveInfo:
    """
    机器人从站信息
    """
    alStatus: int
    productCode: int
    reversionNumber: int
    slaveAddr: int
    slaveId: int
    slaveName: str
    vendorId: int
    def __init__(self) -> None:
        ...
