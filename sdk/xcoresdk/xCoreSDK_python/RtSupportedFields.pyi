"""
机器人实时数据支持的字段

说明：数据名后为数据类型
  ArrayXD = std::array<double, DoF>, DoF 为轴数
  Array6D = std::array<double, 6>
  Array16D = std::array<double, 16>

--- 无需打开实时模式即可获取 ---

jointPos_m: 关节角度 [rad] - ArrayXD
jointVel_m: 关节速度 [rad/s] - ArrayXD
tcpPose_m: 末端位姿, 相对于基坐标系, 行优先齐次变换矩阵 - Array16D
tcpPoseAbc_m: 末端位姿, 相对于基坐标系 [X,Y,Z,Rx,Ry,Rz] - Array6D
exJointPos_m: 外部轴数值 [rad] 导轨[m] - Array6D, 实际有效数据个数为外部轴数
exJointVel_m: 外部轴速度 [rad/s] 导轨[m/s] - Array6D, 实际有效数据个数为外部轴数
exMotor_m: 外部轴电机位置 - Array6D, 实际有效数据个数为外部轴数
elbow_m: 臂角 [rad] - double
tau_m: 关节力矩 [Nm] - ArrayXD
theta_m: 电机位置 - ArrayXD
thetaVel_m: 电机位置微分 - ArrayXD
motorTau: 电机转矩 - ArrayXD
keypads: 末端按键状态 - ArrayXD

--- 仅在打开实时模式控制之后数据有效 ---

jointPos_c: 指令关节角度 [rad] - ArrayXD
jointVel_c: 指令关节速度 [rad/s] - ArrayXD
jointAcc_m: 关节加速度 [rad/s^2] - ArrayXD
jointAcc_c: 指令关节加速度 [rad/s^2] - ArrayXD
tcpPose_c: 发送的末端位姿指令, 相对于基坐标系, 行优先齐次变换矩阵 - Array16D
tcpVel_m: 机器人末端速度 - Array6D
tcpVel_c: 指令机器人末端速度 - Array6D
tcpAcc_m: 机器人末端加速度 - Array6D
tcpAcc_c: 指令机器人末端加速度 - Array6D
elbow_c: 指令臂角 [rad] - double
elbowVel_c: 指令臂角速度 [rad/s] - double
elbowAcc_c: 指令臂角加速度 [rad/s] - double
tau_c: 指令关节力矩 [Nm] - ArrayXD
tauFiltered_m: 滤波后关节力矩 [Nm] - ArrayXD
tauVel_c: 指令力矩微分 [Nm/s] - ArrayXD
tauExt_inBase: 基坐标系中外部力矩 [Nm] - Array6D
tauExt_inStiff: 力控坐标系中外部力矩 [Nm] - Array6D
motorTauFiltered: 滤波后电机转矩 - ArrayXD 
"""
from __future__ import annotations
__all__: list[str] = ['elbowAcc_c', 'elbowVel_c', 'elbow_c', 'elbow_m', 'exJointPos_m', 'exJointVel_m', 'exMotor_m', 'jointAcc_c', 'jointAcc_m', 'jointPos_c', 'jointPos_m', 'jointVel_c', 'jointVel_m', 'keypads', 'motorTau', 'motorTauFiltered', 'tauExt_inBase', 'tauExt_inStiff', 'tauFiltered_m', 'tauVel_c', 'tau_c', 'tau_m', 'tcpAcc_c', 'tcpAcc_m', 'tcpPoseAbc_m', 'tcpPose_c', 'tcpPose_m', 'tcpVel_c', 'tcpVel_m', 'thetaVel_m', 'theta_m']
elbowAcc_c: str = 'psi_acc_c'
elbowVel_c: str = 'psi_vel_c'
elbow_c: str = 'psi_c'
elbow_m: str = 'psi_m'
exJointPos_m: str = 'ex_q_m'
exJointVel_m: str = 'ex_dq_m'
exMotor_m: str = 'ex_motor_m'
jointAcc_c: str = 'ddq_c'
jointAcc_m: str = 'ddq_m'
jointPos_c: str = 'q_c'
jointPos_m: str = 'q_m'
jointVel_c: str = 'dq_c'
jointVel_m: str = 'dq_m'
keypads: str = 'io_keypad'
motorTau: str = 'motor_tau'
motorTauFiltered: str = 'motor_tau_filtered'
tauExt_inBase: str = 'tau_ext_base'
tauExt_inStiff: str = 'tau_ext_stiff'
tauFiltered_m: str = 'tau_filtered_m'
tauVel_c: str = 'tau_vel_c'
tau_c: str = 'tau_c'
tau_m: str = 'tau_m'
tcpAcc_c: str = 'pos_acc_c'
tcpAcc_m: str = 'pos_acc_m'
tcpPoseAbc_m: str = 'pos_abc_m'
tcpPose_c: str = 'pos_c'
tcpPose_m: str = 'pos_m'
tcpVel_c: str = 'pos_vel_c'
tcpVel_m: str = 'pos_vel_m'
thetaVel_m: str = 'theta_vel_m'
theta_m: str = 'theta_m'
