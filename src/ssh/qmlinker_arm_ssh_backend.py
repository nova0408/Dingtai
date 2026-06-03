from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QProcess, Signal

from src.simulation.qmlinker_arm_remote import (
    QmlinkerArmCommand,
    QmlinkerArmRemoteConfig,
    REMOTE_QMLINKER_ARM_SCRIPT,
    SUPPORTED_ARM_DEVICES,
    axis_names_for_device,
    build_remote_payload,
    parse_arm_axis_name,
)

# region 数据结构


@dataclass(frozen=True, slots=True)
class _PendingRequest:
    """SSH 子进程关联请求。

    职责边界：
    - 只记录一个本地 QProcess 对应的命令与业务动作。
    - 不持有 GUI 控件、qmlinker 对象或远端连接句柄。

    设计思想：
    - QProcess 完成时只给出进程对象，需要通过该结构恢复请求语义。

    生命周期：
    - 由 `QmlinkerArmSshBackend` 创建并在进程结束后删除。

    继承关系：
    - 不继承业务基类，作为后端内部数据结构使用。
    """

    process: QProcess
    "Qt 子进程对象。"

    action: str
    "业务动作名称。"

    axis_name: str | None = None
    "GUI 轴名，仅关节相关请求使用。"

    device_name: str | None = None
    "设备名，仅机械臂使能或状态请求使用。"


# endregion


# region 主入口


class QmlinkerArmSshBackend(QObject):
    """通过系统 `ssh orin` 访问 Orin 端 qmlinker 机械臂接口。

    职责边界：
    - 负责 SSH 子进程生命周期、远端 JSON 命令执行和结果解析。
    - 不创建或修改 GUI 控件，不在本地导入 qmlinker。
    - 第一版只覆盖 `arm_example.py` 中明确存在的左右机械臂 enable 与 6 轴角度接口。

    设计思想：
    - 使用系统 SSH 配置中的 `orin` 主机，避免 GUI 保存密码或重复维护认证参数。
    - 每次请求使用独立 `QProcess`，实现简单且不会阻塞 Qt 事件循环。
    - 远端脚本沿用 qmlinker 示例中的 `QMArm` 调用方式，便于和文档对照。

    生命周期：
    - 随 GUI 主窗口创建和销毁。
    - 每个请求持有一个短生命周期 QProcess，结束后释放。

    继承关系：
    - 继承 `QObject` 以提供 Qt 信号，不继承业务基类。

    线程/异步语义：
    - 所有 SSH 请求都在 QProcess 子进程中运行。
    - 结果通过 Qt 信号回到 GUI 线程，GUI 只做状态回写。
    """

    sshStateChanged = Signal(bool, str)
    enableStateReceived = Signal(str, bool)
    dofValuesReceived = Signal(dict)
    requestFailed = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        ssh_host_alias: str = "orin",
        config: QmlinkerArmRemoteConfig | None = None,
    ) -> None:
        """初始化 qmlinker 机械臂 SSH 后端。

        Parameters
        ----------
        parent:
            Qt 父对象。
        ssh_host_alias:
            系统 SSH 配置中的主机别名，例如 `orin`。
        config:
            qmlinker 远端连接配置，为 `None` 时使用默认配置。
        """

        super().__init__(parent)
        self._ssh_host_alias = ssh_host_alias
        self._config = QmlinkerArmRemoteConfig() if config is None else config
        self._pending: dict[QProcess, _PendingRequest] = {}

    def connect_ssh(self) -> None:
        """请求检查 SSH 与远端 qmlinker 可用性。"""

        self._start_request(QmlinkerArmCommand(action="ping"), action="ping")

    def disconnect_ssh(self) -> None:
        """断开 GUI 侧连接状态。

        Notes
        -----
        当前实现每次请求使用短生命周期 SSH 子进程，没有常驻连接需要关闭。
        """

        self.sshStateChanged.emit(False, "SSH disconnected")

    def refresh_ssh_state(self) -> None:
        """刷新 SSH 与远端 qmlinker 可用性。"""

        self._start_request(QmlinkerArmCommand(action="ping"), action="ping")

    def refresh_enable_state(self, device_name: str) -> None:
        """读取机械臂真实使能状态。

        Parameters
        ----------
        device_name:
            设备名，仅支持 `left_arm` 或 `right_arm`。
        """

        if device_name not in SUPPORTED_ARM_DEVICES:
            return
        self._start_request(
            QmlinkerArmCommand(action="get_enable", device_name=device_name), action="get_enable", device_name=device_name
        )

    def set_enable_state(self, device_name: str, enabled: bool) -> None:
        """设置机械臂使能并读取真实回写状态。

        Parameters
        ----------
        device_name:
            设备名，仅支持 `left_arm` 或 `right_arm`。
        enabled:
            目标使能状态。
        """

        if device_name not in SUPPORTED_ARM_DEVICES:
            self.requestFailed.emit(f"当前接口文档未提供 {device_name} 使能接口")
            return
        self._start_request(
            QmlinkerArmCommand(action="set_enable", device_name=device_name, enabled=enabled),
            action="set_enable",
            device_name=device_name,
        )

    def refresh_dof_value(self, axis_name: str) -> None:
        """读取某个机械臂轴所在整臂的真实关节角度。

        Parameters
        ----------
        axis_name:
            GUI 轴名，例如 `left_j1`。非机械臂轴会被忽略。
        """

        parsed = parse_arm_axis_name(axis_name)
        if parsed is None:
            return
        device_name, _ = parsed
        self._start_request(
            QmlinkerArmCommand(action="get_joints", device_name=device_name),
            action="get_joints",
            axis_name=axis_name,
            device_name=device_name,
        )

    def set_dof_target(self, axis_name: str, target_angle_deg: float) -> None:
        """设置单个机械臂关节目标角度。

        Parameters
        ----------
        axis_name:
            GUI 轴名，例如 `right_j3`。
        target_angle_deg:
            目标关节角度，单位 deg。
        """

        parsed = parse_arm_axis_name(axis_name)
        if parsed is None:
            self.requestFailed.emit(f"当前接口文档未提供 {axis_name} 控制接口")
            return
        device_name, joint_index = parsed
        self._start_request(
            QmlinkerArmCommand(
                action="set_joint",
                device_name=device_name,
                joint_index=joint_index,
                target_angle_deg=target_angle_deg,
            ),
            action="set_joint",
            axis_name=axis_name,
            device_name=device_name,
        )

    def _start_request(
        self,
        command: QmlinkerArmCommand,
        action: str,
        axis_name: str | None = None,
        device_name: str | None = None,
    ) -> None:
        """启动一次 SSH JSON 请求。

        Parameters
        ----------
        command:
            机械臂远端命令。
        action:
            本地业务动作名称。
        axis_name:
            可选 GUI 轴名。
        device_name:
            可选设备名。
        """

        process = QProcess(self)
        process.setProgram("ssh")
        process.setArguments(
            [
                self._ssh_host_alias,
                "python3",
                "-c",
                REMOTE_QMLINKER_ARM_SCRIPT,
                build_remote_payload(self._config, command),
            ]
        )
        request = _PendingRequest(process=process, action=action, axis_name=axis_name, device_name=device_name)
        self._pending[process] = request
        process.finished.connect(lambda exit_code, _exit_status, proc=process: self._on_process_finished(proc, exit_code))
        process.errorOccurred.connect(lambda _error, proc=process: self._on_process_error(proc))
        process.start()

    def _on_process_error(self, process: QProcess) -> None:
        """处理 SSH 子进程启动或运行错误。"""

        request = self._pending.pop(process, None)
        message = process.errorString()
        if request is not None and request.action == "ping":
            self.sshStateChanged.emit(False, message)
        else:
            self.requestFailed.emit(message)
        process.deleteLater()

    def _on_process_finished(self, process: QProcess, exit_code: int) -> None:
        """解析 SSH 子进程输出并发出业务信号。"""

        request = self._pending.pop(process, None)
        stdout = bytes(process.readAllStandardOutput().data()).decode("utf-8", errors="replace").strip()
        stderr = bytes(process.readAllStandardError().data()).decode("utf-8", errors="replace").strip()
        process.deleteLater()
        if request is None:
            return

        payload = self._decode_payload(stdout)
        if exit_code != 0 or not payload.get("ok", False):
            message = str(payload.get("error") or stderr or f"ssh request failed: {request.action}")
            if request.action == "ping":
                self.sshStateChanged.emit(False, message)
            else:
                self.requestFailed.emit(message)
            return

        self._dispatch_success(request, payload)

    def _decode_payload(self, stdout: str) -> dict[str, Any]:
        """从远端 stdout 中解析最后一行 JSON。

        Parameters
        ----------
        stdout:
            SSH 标准输出文本。

        Returns
        -------
        dict[str, Any]
            解析出的 JSON 字典。解析失败时返回空字典。
        """

        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return {}

    def _dispatch_success(self, request: _PendingRequest, payload: dict[str, Any]) -> None:
        """根据请求类型分发成功结果。"""

        if request.action == "ping":
            self.sshStateChanged.emit(True, "SSH connected")
            return

        if request.action in {"get_enable", "set_enable"}:
            device_name = str(payload.get("device_name") or request.device_name)
            self.enableStateReceived.emit(device_name, bool(payload.get("enabled", False)))
            return

        if request.action in {"get_joints", "set_joint"}:
            device_name = str(payload.get("device_name") or request.device_name)
            axis_names = axis_names_for_device(device_name)  # type: ignore[arg-type]
            joints = payload.get("joints_deg", [])
            if not isinstance(joints, list):
                self.requestFailed.emit(f"关节状态响应格式错误：{device_name}")
                return
            values = {axis_name: float(value) for axis_name, value in zip(axis_names, joints, strict=False)}
            self.dofValuesReceived.emit(values)


# endregion
