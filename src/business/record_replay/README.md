# 双臂记录回放服务

本包从 `test/wuji/record_left_replay_cli.py` 拆分而来，但业务代码不导入
`test` 目录。它只负责自动双臂回放：AGV 到执行点、CSV 回放、AGV 返回等待点。

## 循环状态

```text
waiting
  -> navigating_to_start (AGV navigate_to("3") + get_runtime_info 到位确认)
  -> replaying          (按左/右 CSV 执行计划回放)
  -> navigating_to_finish (AGV navigate_to("1") + get_runtime_info 到位确认)
  -> waiting
```

任一阶段失败时，状态进入 `failed`，错误文本写入 `ReplayContext.snapshot()`。
下一轮必须重新发出触发指令。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `contracts.py` | 不可变 CSV、计划和服务状态数据契约。 |
| `settings.py` | 唯一配置页：现场连接、机械臂、手部、offset、AGV 与重试策略 dataclass。 |
| `context.py` | 运行资源、停止事件和可查询状态快照的唯一跨模块边界。 |
| `csv_repository.py` / `execution_plan.py` | CSV 文件发现、解析和左右臂阶段计划构建。 |
| `arm_gateway.py` / `hand_gateway.py` | SDK/qmlinker 设备连接、准备和释放。 |
| `arm_actions.py` / `hand_actions.py` | 单一设备动作语义与旧回放时序。 |
| `dual_arm_executor.py` | 双臂阶段、并行、flush、offset 触发边界。 |
| `offset_*.py` | 三球检测、手眼变换及全局纠偏数据流。 |
| `agv_navigation.py` / `cycle_service.py` | AGV 到位确认与常态化循环。 |

## 状态查询

调用方通过 `ReplayContext.snapshot()` 获得 `ReplayStatusSnapshot`：

- `state`：当前服务阶段；
- `left_csv_state`：当前左臂 CSV 去除 `state_prefix` 后的文件名；
- `plan_index`：当前左臂计划下标；
- `error_text`：失败原因。

## Linux 常态化启动

本机/部署入口是 `test/wuji/record__replay_local_test.py`。IP、CSV 目录、
三球服务与手眼结果路径均集中在该入口顶部的常量中。创建触发文件后服务执行一轮：

```bash
touch /path/to/Dingtai/record__replay_next_cycle.trigger
```

可由 systemd 以项目 Python 环境启动：

```ini
[Service]
WorkingDirectory=/path/to/Dingtai
ExecStart=/path/to/python test/wuji/record__replay_local_test.py
Restart=on-failure
```

入口将 `SIGINT` 与 `SIGTERM` 转为 `KeyboardInterrupt`，确保 AGV session 和已创建的
双臂运行资源走 `finally` 清理。部署前必须在现场验证 AGV 导航状态字符串、机械臂型号、
手部/升降 qmlinker 与三球服务；静态检查和无硬件冒烟不能证明真实硬件动作。

### 设备连接参数

`ReplayDeviceConnection` 是唯一进入业务包的现场设备连接数据：

- `left_arm_ip`、`right_arm_ip`；
- `qmlinker_host`、`qmlinker_port`；
- `gripper_port`。

业务模块不包含以上 IP 或端口常量。部署时仅修改本机入口的对应常量，并重新构造
`ReplayCycleConfig`；不要把现场地址回填到网关或动作模块。

Orin 部署默认直接访问 qmlinker `192.168.100.60:50062`、ZMQ
`tcp://192.168.100.60:6200` 与 AGV `192.168.100.70`。本机 service/SSH 隧道绑定地址
固定为 `127.0.0.1`。service 不创建、不管理 SSH 隧道；本机入口的
`LocalServiceTunnelGroup` 以**一个 SSH 进程**统一持有三条转发：

- `127.0.0.1:50061 -> 192.168.100.60:50062`，hand/body；
- `127.0.0.1:50065 -> 192.168.100.60:50066`，left gripper；
- `127.0.0.1:50063 -> 192.168.100.70:50062`，AGV。

AGV 与 hand/body 虽使用相同远端端口，但远端主机不同，因此必须映射为不同本地端口。
隧道组构造完成后才向业务服务提供 `ReplayDeviceConnection` 与 AGV 导航接口，并在入口
`finally` 中统一关闭。

### 运行策略参数

所有会影响执行时序、速度、重试、容差、三球采样和 AGV 轮询的参数只定义在
`settings.py`：

- `ReplayArmSettings`：NRT、tool/wobj、MoveAbsJ、reset 与机械臂型号；
- `ReplayHandSettings`：夹爪/M11/升降动作与容差；
- `ReplayOffsetSettings`：offset 触发、采样、速度和三球鲁棒聚合；
- `OffsetConfig`：三球服务地址、相机名、先验捕获与手眼结果路径；
- `ReplayServiceSettings`：AGV、触发文件及非运动调用重试。

业务模块通过 `ReplayContext.config.settings` 或 `ReplayRuntime.settings` 读取这些数据，
禁止重新声明模块级调试常量。若需现场调参，请在本机入口构造
`ReplayServiceSettings` 的定制实例，再传入 `ReplayCycleConfig`。
