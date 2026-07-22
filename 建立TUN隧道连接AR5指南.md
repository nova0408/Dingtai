# 建立 TUN 隧道连接 AR5 指南

## 1. 目标与网络拓扑

本文说明如何在一台全新的 NVIDIA Jetson Orin 和 Windows 电脑之间建立 WireGuard 三层 TUN 隧道，使 Windows 上的 AR5 官方 GUI 能通过 Orin 访问以下两台设备：

- AR5 地址一：`192.168.100.160`
- AR5 地址二：`192.168.100.161`

推荐拓扑：

```text
Windows
  LAN 地址：与 Orin wlan0 同网段
  WireGuard：10.66.66.2/24
        |
        | UDP 51820 / WireGuard
        v
Jetson Orin
  wlan0：例如 192.168.1.128/24
  wg0：10.66.66.1/24
  eth0：例如 192.168.100.70/24
        |
        +---- 192.168.100.160
        +---- 192.168.100.161
```

Orin 在 `wg0` 与 `eth0` 之间执行 IPv4 转发，并对发往 AR5 网段的流量执行 MASQUERADE。这样 AR5 看到的访问来源是 Orin 的 `eth0` 地址，不要求 AR5 配置返回 `10.66.66.0/24` 的静态路由。

> WireGuard 是三层 TUN 隧道，不转发以太网广播。AR5 GUI 如果支持手动填写地址，应直接填写 `192.168.100.160` 或 `192.168.100.161`。仅依赖二层广播自动发现的功能不能直接跨越本隧道。

## 2. 已验证的软件与系统版本

本指南已在以下 Orin 环境实际验证：

| 项目 | 已验证版本 |
| --- | --- |
| Orin 操作系统 | Ubuntu 20.04.6 LTS（Focal） |
| NVIDIA JetPack | 5.1.3-b29 |
| Jetson Linux / L4T | R35.5.0，`nvidia-l4t-core 35.5.0-20240219203809` |
| Jetson 内核 | `5.10.192-tegra`，aarch64 |
| `nvidia-l4t-kernel` | `5.10.192-tegra-35.5.0-20240613202628` |
| WireGuard 后端 | `wireguard-go v0.0.20220316` |
| WireGuard 工具 | `wireguard-tools v1.0.20260223` |
| Windows WireGuard GUI | 1.1 |
| Windows `wg.exe` | 1.0.20260223 |

检查 Orin 版本：

```bash
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg -l | grep -E '^ii[[:space:]]+(nvidia-jetpack|nvidia-l4t-core|nvidia-l4t-kernel)[[:space:]]'
uname -a
```

### 为什么需要 `wireguard-go`

上述 Jetson 内核没有提供可用的 WireGuard 内核模块。直接启动系统自带的 `wg-quick` 会出现：

```text
Error: Unknown device type.
Unable to access interface: Protocol not supported
```

因此使用官方用户态实现 `wireguard-go`。JetPack 5.1.3 自带的旧版 `wireguard-tools v1.0.20200513` 与用户态 UAPI 配合时可能永久卡在 `wg setconf`，所以本指南同时安装新版 `wg` 和 `wg-quick` 到 `/usr/local/bin`，并让 systemd 显式调用新版 `wg-quick`。

## 3. 开始前的信息与安全要求

准备以下信息：

1. Orin 的 Windows 可达地址，例如 `192.168.1.128`。
2. Orin 连接 AR5 网段的接口名。本文假设为 `eth0`。
3. AR5 地址为 `192.168.100.160` 和 `192.168.100.161`。
4. Windows 隧道地址使用 `10.66.66.2/24`。
5. Orin 隧道地址使用 `10.66.66.1/24`。
6. WireGuard UDP 监听端口使用 `51820`。

安全要求：

- Orin 和 Windows 各自生成私钥，私钥只保存在生成它的设备上。
- 两端只交换公钥。
- 不要把私钥发到聊天、邮件、Git 或截图中。
- 如果私钥曾经泄露，必须重新生成密钥对，并同步更新对端公钥。

## 4. 在全新 Orin 上安装依赖

```bash
sudo apt update
sudo apt install -y \
    wireguard-tools \
    git \
    curl \
    build-essential \
    pkg-config \
    libmnl-dev
```

系统包中的 `wireguard-tools` 用于提供基础目录和依赖，但后续会在 `/usr/local/bin` 安装已验证的新版本。不会覆盖 `/usr/bin/wg` 和 `/usr/bin/wg-quick`。

## 5. 构建并安装 `wireguard-go 0.0.20220316`

该版本使用 Go 1.18.10 构建。以下命令把所有构建文件放入 `/tmp`，只把最终二进制安装到 `/usr/local/bin`。

```bash
WG_GO_BUILD_ROOT="$(mktemp -d /tmp/wireguard-go-build.XXXXXX)"

curl --fail --location --retry 3 --connect-timeout 15 \
    https://dl.google.com/go/go1.18.10.linux-arm64.tar.gz \
    --output "${WG_GO_BUILD_ROOT}/go1.18.10.linux-arm64.tar.gz"

printf '%s  %s\n' \
    '160497c583d4c7cbc1661230e68b758d01f741cf4bece67e48edc4fdd40ed92d' \
    "${WG_GO_BUILD_ROOT}/go1.18.10.linux-arm64.tar.gz" \
    | sha256sum --check

mkdir "${WG_GO_BUILD_ROOT}/toolchain"
tar -C "${WG_GO_BUILD_ROOT}/toolchain" \
    -xzf "${WG_GO_BUILD_ROOT}/go1.18.10.linux-arm64.tar.gz"

git clone --branch 0.0.20220316 --depth 1 \
    https://git.zx2c4.com/wireguard-go \
    "${WG_GO_BUILD_ROOT}/wireguard-go"

export PATH="${WG_GO_BUILD_ROOT}/toolchain/go/bin:${PATH}"
export GOPROXY="https://proxy.golang.org,direct"

make -C "${WG_GO_BUILD_ROOT}/wireguard-go" wireguard-go
sudo install -m 0755 \
    "${WG_GO_BUILD_ROOT}/wireguard-go/wireguard-go" \
    /usr/local/bin/wireguard-go

/usr/local/bin/wireguard-go --version
```

如果 Orin 无法访问 `proxy.golang.org`，可在重新执行 `make` 前切换国内 Go 模块代理：

```bash
export GOPROXY="https://goproxy.cn,direct"
make -C "${WG_GO_BUILD_ROOT}/wireguard-go" wireguard-go
```

版本输出应包含：

```text
wireguard-go v0.0.20220316
```

## 6. 构建并安装新版 `wg` 与 `wg-quick`

本指南固定到已验证的官方 `wireguard-tools` 提交：

```text
a998407747005ea7e4e0258d96f105c97241e1d3
```

执行：

```bash
WG_TOOLS_BUILD_ROOT="$(mktemp -d /tmp/wireguard-tools-build.XXXXXX)"

git clone https://git.zx2c4.com/wireguard-tools \
    "${WG_TOOLS_BUILD_ROOT}/wireguard-tools"

git -C "${WG_TOOLS_BUILD_ROOT}/wireguard-tools" \
    checkout --detach a998407747005ea7e4e0258d96f105c97241e1d3

make -C "${WG_TOOLS_BUILD_ROOT}/wireguard-tools/src" wg

sudo install -m 0755 \
    "${WG_TOOLS_BUILD_ROOT}/wireguard-tools/src/wg" \
    /usr/local/bin/wg

sudo install -m 0755 \
    "${WG_TOOLS_BUILD_ROOT}/wireguard-tools/src/wg-quick/linux.bash" \
    /usr/local/bin/wg-quick

/usr/local/bin/wg --version
bash -n /usr/local/bin/wg-quick
```

`wg --version` 应输出：

```text
wireguard-tools v1.0.20260223
```

## 7. 生成 Orin 密钥对

```bash
sudo install -d -m 0700 /etc/wireguard

sudo sh -c 'umask 077; /usr/local/bin/wg genkey > /etc/wireguard/server_private.key'
sudo sh -c '/usr/local/bin/wg pubkey < /etc/wireguard/server_private.key > /etc/wireguard/server_public.key'

sudo chmod 0600 /etc/wireguard/server_private.key
sudo chmod 0644 /etc/wireguard/server_public.key
sudo cat /etc/wireguard/server_public.key
```

记录输出的 Orin 公钥，稍后填入 Windows 配置。不要复制或显示 `server_private.key` 的内容，除非正在本机写入 `wg0.conf`。

## 8. 在 Windows 上安装并创建隧道

1. 从 WireGuard 官方网站安装 Windows 客户端：<https://www.wireguard.com/install/>。
2. 打开 WireGuard，选择“添加隧道”→“添加空隧道”。
3. GUI 会自动生成 Windows 私钥和公钥。
4. 记录 GUI 显示的 Windows 公钥，稍后写入 Orin 的 peer 配置。
5. 使用下列配置，将占位符替换为实际值：

```ini
[Interface]
PrivateKey = <WINDOWS_PRIVATE_KEY，由 Windows GUI 自动生成并保留在本机>
Address = 10.66.66.2/24

[Peer]
PublicKey = <ORIN_PUBLIC_KEY>
AllowedIPs = 192.168.100.160/32, 192.168.100.161/32, 10.66.66.1/32
Endpoint = <ORIN_在Windows侧可达的IP>:51820
PersistentKeepalive = 25
```

示例 Endpoint：

```ini
Endpoint = 192.168.1.128:51820
```

暂时不要依赖 AR5 自动发现；隧道建立后先通过 ping 和手动 IP 验证。

## 9. 创建 Orin 的 `/etc/wireguard/wg0.conf`

先读取 Orin 私钥，只在 Orin 当前终端中用于填写配置：

```bash
sudo cat /etc/wireguard/server_private.key
```

编辑配置：

```bash
sudo nano /etc/wireguard/wg0.conf
```

写入以下内容，并替换两个占位符：

```ini
[Interface]
Address = 10.66.66.1/24
ListenPort = 51820
PrivateKey = <ORIN_PRIVATE_KEY>
PostUp = iptables -A FORWARD -i %i -o eth0 -d 192.168.100.160/32 -j ACCEPT; iptables -A FORWARD -i %i -o eth0 -d 192.168.100.161/32 -j ACCEPT; iptables -A FORWARD -i eth0 -o %i -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT; iptables -t nat -A POSTROUTING -s 10.66.66.0/24 -o eth0 -d 192.168.100.0/24 -j MASQUERADE
PreDown = iptables -D FORWARD -i %i -o eth0 -d 192.168.100.160/32 -j ACCEPT; iptables -D FORWARD -i %i -o eth0 -d 192.168.100.161/32 -j ACCEPT; iptables -D FORWARD -i eth0 -o %i -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT; iptables -t nat -D POSTROUTING -s 10.66.66.0/24 -o eth0 -d 192.168.100.0/24 -j MASQUERADE

[Peer]
PublicKey = <WINDOWS_PUBLIC_KEY>
AllowedIPs = 10.66.66.2/32
```

如果 Orin 连接 AR5 网段的接口不是 `eth0`，必须把四处 `eth0` 全部替换为实际接口名。可用以下命令确认：

```bash
ip -brief address
ip route get 192.168.100.160
```

设置权限：

```bash
sudo chown root:root /etc/wireguard/wg0.conf
sudo chmod 0600 /etc/wireguard/wg0.conf
```

## 10. 开启并持久化 IPv4 转发

```bash
printf 'net.ipv4.ip_forward = 1\n' \
    | sudo tee /etc/sysctl.d/99-wireguard-forward.conf >/dev/null

sudo chmod 0644 /etc/sysctl.d/99-wireguard-forward.conf
sudo sysctl -w net.ipv4.ip_forward=1
sysctl net.ipv4.ip_forward
```

应输出：

```text
net.ipv4.ip_forward = 1
```

不要为了这一个参数执行 `sysctl --system`。JetPack 中可能存在其他与当前内核不兼容的 sysctl 项，导致出现无关错误并中断安装流程。

## 11. 配置 systemd 使用新版 `wg-quick`

系统自带 `/usr/bin/wg-quick` 会把 `/usr/bin` 放到 PATH 最前，从而再次调用旧版 `/usr/bin/wg`。必须创建 drop-in，让服务直接执行 `/usr/local/bin/wg-quick`：

```bash
sudo install -d -m 0755 /etc/systemd/system/wg-quick@.service.d

sudo tee /etc/systemd/system/wg-quick@.service.d/override.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/local/bin/wg-quick up %i
ExecStop=
ExecStop=/usr/local/bin/wg-quick down %i
EOF

sudo chmod 0644 /etc/systemd/system/wg-quick@.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl enable wg-quick@wg0
sudo systemctl start wg-quick@wg0
```

如果启用了 UFW，还需允许 WireGuard UDP 端口：

```bash
sudo ufw status
sudo ufw allow 51820/udp
```

仅在 UFW 处于 active 时需要执行 `ufw allow`。

## 12. 启动后的验证

### 12.1 Orin 本机验证

```bash
systemctl is-enabled wg-quick@wg0
systemctl is-active wg-quick@wg0
systemctl status wg-quick@wg0 --no-pager -l
ip -brief address show wg0
sudo /usr/local/bin/wg show wg0
ss -lun | grep ':51820'
ping -c 3 192.168.100.160
ping -c 3 192.168.100.161
```

预期结果：

- `wg-quick@wg0` 为 `enabled` 和 `active`。
- `wg0` 包含 `10.66.66.1/24`，状态为 `UP`。
- UDP `51820` 正在监听。
- Orin 能访问两个 AR5 地址。

### 12.2 Windows 验证

在 WireGuard GUI 中启用隧道，然后在 PowerShell 执行：

```powershell
ping 10.66.66.1
ping 192.168.100.160
ping 192.168.100.161
```

也可以确认 Windows 路由：

```powershell
Get-NetRoute -AddressFamily IPv4 |
    Where-Object {
        $_.DestinationPrefix -in @(
            '10.66.66.1/32',
            '192.168.100.160/32',
            '192.168.100.161/32'
        )
    } |
    Format-Table DestinationPrefix, NextHop, InterfaceAlias, RouteMetric
```

三条路由都应指向 WireGuard 隧道接口。

### 12.3 AR5 GUI 验证

在 AR5 官方 GUI 中分别手动填写：

```text
192.168.100.160
192.168.100.161
```

ping 成功只证明三层网络和 ICMP 正常。最终仍需用 AR5 GUI 实际连接，以验证设备所需的 TCP/UDP 业务端口。

### 12.4 重启验证

首次部署完成后，在允许重启 Orin 的维护窗口执行：

```bash
sudo reboot
```

重启后再次检查：

```bash
systemctl is-active wg-quick@wg0
ip -brief address show wg0
sudo /usr/local/bin/wg show wg0
```

## 13. 常见故障排查

### 13.1 `Unknown device type` / `Protocol not supported`

含义：Jetson 内核没有可用的 WireGuard 内核模块。

检查：

```bash
find "/lib/modules/$(uname -r)" -iname 'wireguard.ko*'
command -v wireguard-go
```

如果 `wireguard-go` 已正确安装，`wg-quick` 会先显示内核接口创建失败，然后回退到用户态实现。最终服务仍应成功进入 active。

### 13.2 永久卡在 `wg setconf wg0 /dev/fd/63`

通常是旧版 `wg` 被调用。检查：

```bash
/usr/bin/wg --version
/usr/local/bin/wg --version
systemctl cat wg-quick@wg0
```

必须满足：

- `/usr/local/bin/wg` 为本指南的新版本。
- systemd 的 `ExecStart` 为 `/usr/local/bin/wg-quick up %i`。
- 日志中的配置动作可能显示为新版工具使用的 `wg addconf`，且应立即返回。

查看日志：

```bash
journalctl -u wg-quick@wg0 --no-pager -n 100
```

### 13.3 Windows 没有握手

依次检查：

1. Windows Peer 公钥是否为 Orin 公钥。
2. Orin Peer 公钥是否为 Windows 公钥。
3. Windows Endpoint 是否为 Windows 可达的 Orin 地址与 `51820` 端口。
4. Windows 到 Orin 的基础网络是否可达。
5. 防火墙是否允许 UDP `51820`。
6. 两端是否错误地填入了对方私钥。

命令：

```bash
sudo /usr/local/bin/wg show wg0
ss -lun | grep ':51820'
```

### 13.4 能访问 `10.66.66.1`，不能访问 AR5

检查 Orin 到 AR5：

```bash
ping -c 3 192.168.100.160
ping -c 3 192.168.100.161
ip route get 192.168.100.160
ip neigh show dev eth0
```

检查转发和 NAT：

```bash
sysctl net.ipv4.ip_forward
sudo iptables -S FORWARD
sudo iptables -t nat -S POSTROUTING
```

应存在：

- `wg0 -> eth0` 到两个 `/32` 地址的 ACCEPT 规则。
- `eth0 -> wg0` 的 `RELATED,ESTABLISHED` 规则。
- `10.66.66.0/24 -> eth0 -> 192.168.100.0/24` 的 MASQUERADE 规则。

### 13.5 Windows 本地也存在 `192.168.100.0/24`

本文配置会为 `.160` 和 `.161` 建立更具体的 `/32` 路由，通常优先于本地 `/24`。仍应使用 `Get-NetRoute` 确认这两个 `/32` 确实指向 WireGuard 接口。

### 13.6 ping 成功但 AR5 GUI 找不到设备

如果手动 IP 能连接而自动发现失败，通常是 GUI 使用了广播或组播发现。WireGuard TUN 不提供二层广播透传。优先使用 GUI 的手动 IP 连接功能；只有确认 GUI 完全不支持手动地址后，才考虑额外部署广播中继或改用二层隧道。

## 14. 密钥轮换

Windows 私钥泄露后：

1. 在 Windows WireGuard GUI 中新建密钥对。
2. 保持 Windows 地址仍为 `10.66.66.2/24`。
3. 把新的 Windows 公钥写入 Orin `/etc/wireguard/wg0.conf` 的 `[Peer] PublicKey`。
4. 重启服务：

```bash
sudo systemctl restart wg-quick@wg0
```

Orin 私钥泄露后：

1. 在 Orin 重新生成服务端密钥对。
2. 更新 `/etc/wireguard/wg0.conf` 的 `[Interface] PrivateKey`。
3. 把新的 Orin 公钥写入 Windows 配置的 `[Peer] PublicKey`。
4. 重启两端隧道。

## 15. 当前验证结论

在本文开头列出的已验证环境中，实际结果为：

- Windows 到 `10.66.66.1`：可达。
- Windows 经 Orin 到 `192.168.100.160`：可达。
- Windows 经 Orin 到 `192.168.100.161`：可达。
- Orin 服务状态：`wg-quick@wg0 active (exited)`。
- Orin `wg0`：`10.66.66.1/24`，`UP`。
- IPv4 转发、FORWARD 规则和 MASQUERADE：已生效。

## 16. 参考链接

- WireGuard 官方安装说明：<https://www.wireguard.com/install/>
- WireGuard 官方快速入门：<https://www.wireguard.com/quickstart/>
- `wireguard-go` 官方仓库：<https://git.zx2c4.com/wireguard-go/>
- `wireguard-tools` 官方仓库：<https://git.zx2c4.com/wireguard-tools/>
- Go 官方下载页面：<https://go.dev/dl/>
