from __future__ import annotations

from qmlinker import QMLift, QMWaist
from qmlinker.grpc_py import lift_pb2_grpc, waist_pb2_grpc

# region 无后台线程的执行器适配


class WujiLiftClient(QMLift):
    """不启动后台轮询线程的升降执行器客户端。

    职责边界：
    - 复用 ``QMLift`` 已有的同步 RPC 方法和返回格式。
    - 仅替换第三方构造过程，不负责 channel 的创建、关闭或重连。

    设计思想：
    - ``QMLift.__init__`` 会启动无法停止的非守护轮询线程。这里显式创建同一
      gRPC stub，避免程序结束时线程继续访问已关闭的 channel。
    - 继承第三方 SDK 类是为了保留现有调用接口，不引入动态代理或兼容分支。

    生命周期：
    - 本类不创建线程，也不拥有外部 channel。调用方关闭 channel 后即可正常释放。
    - 同一实例的线程安全性由 gRPC channel 和第三方同步方法保证。

    继承关系：
    - 继承 ``QMLift`` 以复用升降 RPC 方法，但刻意不调用其有副作用的构造函数。
    """

    def __init__(self, channel: object) -> None:
        """创建无后台轮询线程的升降客户端。

        Parameters
        ----------
        channel:
            qmlinker ``create_channel()`` 返回的基础 channel，或包含 ``DEFAULT``
            channel 的字典。

        Notes
        -----
        这里只初始化父类 RPC 方法所需的 ``channel`` 和 ``stub`` 字段，不启动
        qmlinker 内部的无限轮询线程。
        """

        self.channel = channel["DEFAULT"] if isinstance(channel, dict) else channel
        self.stub = lift_pb2_grpc.LiftServiceStub(self.channel)


class WujiWaistClient(QMWaist):
    """不启动后台轮询线程的腰部执行器客户端。

    职责边界：
    - 复用 ``QMWaist`` 已有的同步 RPC 方法和返回格式。
    - 仅替换第三方构造过程，不管理 channel、SSH 隧道或设备使能时序。

    设计思想：
    - ``QMWaist.__init__`` 与升降客户端一样会启动不可停止的非守护线程。
      本适配类只构造相同的 gRPC stub，从源头消除退出阶段的重复 RPC。
    - 保留明确继承关系，使现有腰部控制代码无需额外包装或动态转发。

    生命周期：
    - 本类不创建线程，也不拥有外部 channel。channel 生命周期由调用方统一管理。
    - 同一实例的线程安全性由 gRPC channel 和第三方同步方法保证。

    继承关系：
    - 继承 ``QMWaist`` 以复用腰部 RPC 方法，但刻意绕过其有副作用的构造函数。
    """

    def __init__(self, channel: object) -> None:
        """创建无后台轮询线程的腰部客户端。

        Parameters
        ----------
        channel:
            qmlinker ``create_channel()`` 返回的基础 channel，或包含 ``DEFAULT``
            channel 的字典。

        Notes
        -----
        这里只初始化父类 RPC 方法所需的 ``channel`` 和 ``stub`` 字段，不启动
        qmlinker 内部的无限轮询线程。
        """

        self.channel = channel["DEFAULT"] if isinstance(channel, dict) else channel
        self.stub = waist_pb2_grpc.WaistServiceStub(self.channel)


# endregion

# region body 客户端


class WujiBodyClient:
    """无际 body 客户端。

    职责边界：
    - 封装无后台线程的升降与腰部 SDK 适配对象，负责对应执行器的读写。
    - 不负责 GUI 状态同步、订阅调度或其他设备的使能逻辑。

    设计思想：
    - body 在项目语义里由两个独立执行器组成，因此这里显式持有两个 SDK 对象。
    - 只把项目侧常用的毫米/角度接口整理出来，避免上层重复处理 proto 细节。

    生命周期：
    - 依赖外部传入的 qmlinker channel 生命周期。
    - 不创建或持有线程，调用方关闭 channel 后即可正常释放。

    继承关系：
    - 不继承业务基类，避免把 lift 和 waist 强行捆成动态多态对象。
    """

    def __init__(self, channel: object) -> None:
        """创建 body 客户端。

        Parameters
        ----------
        channel:
            qmlinker `create_channel()` 返回的基础 channel 或 channel dict。

        Notes
        -----
        本类不拥有 channel 生命周期，关闭连接仍由调用方负责。
        """

        self._lift = WujiLiftClient(channel)
        self._waist = WujiWaistClient(channel)

    @property
    def lift(self) -> WujiLiftClient:
        """返回底层升降 SDK 对象。

        Returns
        -------
        WujiLiftClient
            无后台线程的升降执行器对象，暴露 SDK 原始同步接口。
        """

        return self._lift

    @property
    def waist(self) -> WujiWaistClient:
        """返回底层腰部 SDK 对象。

        Returns
        -------
        WujiWaistClient
            无后台线程的腰部执行器对象，暴露 SDK 原始同步接口。
        """

        return self._waist

# endregion
