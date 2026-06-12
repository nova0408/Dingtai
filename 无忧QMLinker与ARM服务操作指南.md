# 无忧 QMLinker 与 ARM 服务操作指南

## 1. 适用范围

本文用于排查 `wuyou` 机器上的以下服务链路：

- `qmlinker` / gRPC 控制链路
- `grpc_bridge_v2`
- `actuators_arm`
- `sensors_depthcamera_ob_zmq_v2`
- `ros2_autostart.service`

本文结论基于当前现场实际检查结果，不是仅根据仓库代码推测。

---

## 2. 当前服务架构

### 2.1 关键结论

- `qmlinker` 控制服务端在 `wuyou` 上，不在 `orin` 上。
- `qmlinker` 对外入口实际是 `grpc_bridge_v2`，监听端口固定为 `50062`。
- 机械臂控制不是 `grpc_bridge_v2` 单独完成，而是依赖：
  - `actuators_arm`
  - 更下层 EtherCAT / 总线通信
- 相机真实服务不是 qmlinker CameraService，而是：
  - `sensors_depthcamera_ob_zmq_v2`

### 2.2 当前启动链

实际启动顺序如下：

1. `systemd` 启动 `ros2_autostart.service`
2. `ros2_autostart.service` 执行 `/home/wuyou/Desktop/sw495/workspace/ros2_scripts/run.sh`
3. `run.sh` 完成以下准备：
   - `sleep 9`
   - 启动 EtherCAT
   - 放开 `/dev/EtherCAT0` 权限
   - `source` ROS 与项目环境
4. `run.sh` 执行：
   - `ros2 launch /home/wuyou/Desktop/sw495/workspace/ros2_scripts/launch.py`
5. `launch.py` 拉起整组 ROS2 节点

### 2.3 与当前链路直接相关的关键节点

- `comm_rs485`
- `comm_can`
- `actuators_arm`
- `actuators_hand`
- `actuators_head`
- `actuators_waist`
- `actuators_lift`
- `actuators_gripper`
- `grpc_bridge_v2`
- `sensors_depthcamera_ob_zmq_v2`

---

## 3. 登录方式

### 3.1 先登录跳板机

```bash
ssh orin
```

### 3.2 直接跳转到 wuyou

```bash
ssh -J orin wuyou
```

如果本机 SSH 配置已经生效，也可以直接：

```bash
ssh wuyou
```

---

## 4. 常用检查命令

### 4.1 检查自启动服务状态

```bash
systemctl status ros2_autostart.service --no-pager
```

期望结果：

- `Loaded: loaded`
- `Active: active (running)`

### 4.2 检查关键进程是否在线

```bash
ps -ef | grep -E 'run.sh|ros2 launch|grpc_bridge_v2|actuators_arm|sensors_depthcamera_ob_zmq_v2' | grep -v grep
```

期望至少看到：

- `run.sh`
- `ros2 launch ... launch.py`
- `grpc_bridge_v2`
- `actuators_arm`
- `sensors_depthcamera_ob_zmq_v2`

### 4.3 检查 50062 端口是否监听

```bash
ss -ltnp | grep 50062
```

期望看到：

```text
LISTEN ... *:50062 ... grpc_bridge_v2
```

### 4.4 检查 ROS2 节点

注意必须先进入同一 ROS 环境：

```bash
export ROS_DOMAIN_ID=28
source /opt/ros/humble/setup.bash
source /home/wuyou/Desktop/sw495/workspace/QMFramework/install/setup.bash
ros2 node list
```

### 4.5 检查日志

```bash
journalctl -u ros2_autostart.service --no-pager -n 200
```

```bash
tail -n 200 /home/wuyou/.ros/log/*/launch.log
```

---

## 5. 日志位置说明

### 5.1 systemd 日志

查看：

```bash
journalctl -u ros2_autostart.service --no-pager
```

特点：

- 最适合看整条启动链是否成功
- 最适合看当前活跃运行过程中的 stdout / stderr

### 5.2 ROS2 launch 总日志

目录：

```text
/home/wuyou/.ros/log/
```

其中每次启动都会生成一个批次目录，例如：

```text
/home/wuyou/.ros/log/2026-05-26-16-20-12-010722-wuyou-X1-SBC-1993/launch.log
```

特点：

- 适合看某次启动中哪些节点拉起了
- 适合看 `process has died` 这类退出记录

### 5.3 节点级日志

同目录下还会生成单节点日志，例如：

- `grpc_bridge_v2_*.log`
- `actuators_arm_*.log`
- `sensors_depthcamera_ob_zmq_v2_*.log`

注意：

- 当前 `launch.py` 中 `grpc_bridge_v2` 和 `sensors_depthcamera_ob_zmq_v2` 使用了 `output="screen"`
- 因此很多关键输出会进入 `journalctl` 和 `launch.log`
- 单独的 `grpc_bridge_v2_*.log` 可能为空，这是正常现象

### 5.4 手工启动日志

如果曾手工后台启动过 `grpc_bridge_v2`，还可能看到：

```text
/home/wuyou/Log/grpc_bridge_v2_manual_test.log
```

---

## 6. 故障判断方法

### 6.1 GUI 连不上 qmlinker

优先检查：

1. `wuyou:50062` 是否监听
2. `grpc_bridge_v2` 是否存在
3. `ros2_autostart.service` 是否仍在运行

如果 `50062` 没监听，基本可以判断 `grpc_bridge_v2` 已掉，或者整条 ROS2 启动链没完整拉起。

### 6.2 qmlinker 连得上，但机械臂不能动

优先检查：

1. `actuators_arm` 是否存在
2. `journalctl -u ros2_autostart.service -n 200`
3. 是否出现以下底层错误：

```text
Failed to execute SDO upload: Input/output error
```

说明：

- `grpc_bridge_v2` 只是桥接层
- 如果 `actuators_arm` 或底层 EtherCAT 异常，`50062` 即使还在监听，机械臂也可能不可用

### 6.3 相机正常但 arm 不正常

这并不矛盾，因为：

- 相机走 `sensors_depthcamera_ob_zmq_v2`
- 机械臂走 `grpc_bridge_v2 + actuators_arm`

两者是不同链路。

---

## 7. ARM 服务掉了以后能不能再起来

### 7.1 结论

可以手动再拉起，但当前配置不具备可靠的自动自愈能力。

### 7.2 原因

当前 `launch.py` 中各个 `Node(...)` 没看到 `respawn=True`。

这意味着：

- `actuators_arm` 如果单独崩掉，`ros2 launch` 不会自动把它重新拉起
- `grpc_bridge_v2` 如果单独崩掉，也不会自动重启

同时 `/etc/systemd/system/ros2_autostart.service` 中配置为：

```text
Restart=no
```

这意味着：

- 如果整条 `run.sh -> ros2 launch` 主链退出，`systemd` 也不会自动重启

### 7.3 当前可恢复方式

分两种情况处理：

1. 只有 `grpc_bridge_v2` 掉了
2. `actuators_arm` 或更底层链路掉了

第一种恢复相对简单，第二种通常需要重拉整组 ROS2 服务。

---

## 8. 恢复操作

### 8.1 先确认当前状态

```bash
systemctl status ros2_autostart.service --no-pager
ps -ef | grep -E 'ros2 launch|grpc_bridge_v2|actuators_arm' | grep -v grep
ss -ltnp | grep 50062
```

### 8.2 仅 `grpc_bridge_v2` 掉了时的手工恢复

先登录：

```bash
ssh -J orin wuyou
```

然后执行：

```bash
export ROS_DOMAIN_ID=28
source /opt/ros/humble/setup.bash
source /home/wuyou/Desktop/sw495/workspace/QMFramework/install/setup.bash
/home/wuyou/Desktop/sw495/workspace/QMFramework/install/lib/grpc_bridge_v2/grpc_bridge_v2
```

如果需要后台运行：

```bash
nohup bash -lc '
export ROS_DOMAIN_ID=28
source /opt/ros/humble/setup.bash
source /home/wuyou/Desktop/sw495/workspace/QMFramework/install/setup.bash
exec /home/wuyou/Desktop/sw495/workspace/QMFramework/install/lib/grpc_bridge_v2/grpc_bridge_v2
' > /home/wuyou/Log/grpc_bridge_v2_manual.log 2>&1 &
```

恢复后检查：

```bash
ss -ltnp | grep 50062
```

### 8.3 `actuators_arm` 掉了时的恢复

如果掉的是 `actuators_arm`，只拉 `grpc_bridge_v2` 不够，建议重启整条 ROS2 自启动链。

先停：

```bash
sudo systemctl stop ros2_autostart.service
```

如怀疑残留进程未清理，可执行：

```bash
/home/wuyou/Desktop/sw495/workspace/ros2_scripts/kill_ros.sh
```

该脚本内容比较激进，会直接：

```bash
ps aux | grep -i ros | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

因此只建议在明确要重拉整组 ROS2 节点时使用。

然后重新启动：

```bash
sudo systemctl start ros2_autostart.service
```

启动后检查：

```bash
systemctl status ros2_autostart.service --no-pager
ps -ef | grep -E 'ros2 launch|grpc_bridge_v2|actuators_arm' | grep -v grep
ss -ltnp | grep 50062
```

---

## 9. 启动失败时的重点排查项

### 9.1 残余 ROS 进程

`launch.py` 启动前会检查：

- `ros2 node list` 是否为空
- 系统里是否还存在带 `ros-args` 的残余进程

如果有残余，它会直接退出，不会继续启动。

### 9.2 EtherCAT 未正常起来

重点检查：

- `/opt/etherlab/etc/init.d/ethercat start` 是否成功
- `/dev/EtherCAT0` 是否存在
- `/dev/EtherCAT0` 权限是否正常

### 9.3 底层通信异常

如果日志里看到：

- `Failed to execute SDO upload: Input/output error`
- `bytes_read = 0`
- `process has died`

说明问题可能已经下沉到底层通信，而不只是 `qmlinker` 表层服务。

### 9.4 进程在，但服务不可用

这类情况要区分：

- `grpc_bridge_v2` 是否在线
- `actuators_arm` 是否在线
- 底层总线是否在线

不要只看 `50062` 端口是否监听，就认定 arm 链路健康。

---

## 10. 推荐排障顺序

现场建议固定按以下顺序执行：

1. `ssh wuyou`
2. `systemctl status ros2_autostart.service --no-pager`
3. `ps -ef | grep -E 'ros2 launch|grpc_bridge_v2|actuators_arm|sensors_depthcamera_ob_zmq_v2' | grep -v grep`
4. `ss -ltnp | grep 50062`
5. `journalctl -u ros2_autostart.service --no-pager -n 200`
6. `tail -n 200 /home/wuyou/.ros/log/*/launch.log`
7. 如果仅 `grpc_bridge_v2` 掉了，先单独手工拉起
8. 如果 `actuators_arm` 也掉了，直接重启 `ros2_autostart.service`

---

## 11. 当前现场结论摘要

- `wuyou` 当前确实存在 `ros2_autostart.service`
- 它当前为 `enabled`
- 它负责拉起 `run.sh -> ros2 launch -> 全套节点`
- 当前 `grpc_bridge_v2`、`actuators_arm`、`sensors_depthcamera_ob_zmq_v2` 在线时，服务可正常暴露
- 当前配置没有为单节点崩溃提供自动 respawn
- 当前 `systemd` 配置也没有为整条主链提供自动重启
- 因此 ARM 服务掉线后，可以手工恢复，但不应假设它会自动恢复

