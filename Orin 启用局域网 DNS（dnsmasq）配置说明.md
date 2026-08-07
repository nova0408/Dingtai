# Orin 局域网 DNS（dnsmasq）配置指南

本文记录在 Jetson AGX Orin 上启用 `dnsmasq`，并让 Orin 为 `192.168.100.0/24` 局域网提供普通 DNS 解析的方法。

本文使用的实际网络参数如下：

| 项目 | 值 |
| --- | --- |
| Orin 主机名 | `wujibrain-desktop` |
| Orin 局域网网卡 | `eth0` |
| Orin 局域网 IPv4 | `192.168.100.70` |
| 局域网网段 | `192.168.100.0/24` |
| 对外 DNS 地址 | `192.168.100.70:53` |
| 要发布的主机名 | `wujibrain-desktop` |
| 主机名对应地址 | `192.168.100.70` |

最终希望客户端能够通过普通 DNS 得到：

```text
wujibrain-desktop -> 192.168.100.70
```

> 注意：`wujibrain-desktop.local` 属于 mDNS/Avahi 范畴。本文配置的是普通 DNS 名称 `wujibrain-desktop`，两者可以同时保留。

---

## 1. 最终服务结构

Orin 原本已经存在：

- `systemd-resolved`：监听 `127.0.0.53:53`
- Avahi/mDNS：使用 UDP `5353`

新增 `dnsmasq` 后采用以下分工：

```text
127.0.0.53:53        -> systemd-resolved
192.168.100.70:53    -> dnsmasq
UDP 5353             -> Avahi / mDNS
```

因此无需关闭 `systemd-resolved`，也无需关闭 Avahi。

整体结构为：

```text
Windows / Android / Linux 客户端
              |
              | DNS 查询
              v
       192.168.100.70:53
              |
           dnsmasq
              |
       +------+------------------+
       |                         |
       | 本地静态记录             | 其他域名
       v                         v
wujibrain-desktop          127.0.0.53
       |                   systemd-resolved
       v                         |
192.168.100.70                   v
                              上游 DNS
```

---

## 2. 检查 Orin 当前网络

确认 `eth0` 地址：

```bash
ip -br addr show eth0
```

应看到类似：

```text
eth0    UP    192.168.100.70/24
```

检查当前 DNS 和 mDNS 端口：

```bash
sudo ss -lntup | grep -E '(:53 |:5353 )'
```

在配置 `dnsmasq` 之前，通常会看到：

```text
127.0.0.53:53
UDP 5353
```

此时还不会有：

```text
192.168.100.70:53
```

---

## 3. 安装 dnsmasq

执行：

```bash
sudo apt update
sudo apt install -y dnsmasq dnsutils
```

安装完成后，先停止 `dnsmasq`：

```bash
sudo systemctl stop dnsmasq
```

这样可以避免默认配置尚未调整时产生监听冲突。

---

## 4. 创建专用 dnsmasq 配置

不要直接大范围修改 `/etc/dnsmasq.conf`。

新建独立配置文件：

```text
/etc/dnsmasq.d/wuji-lan.conf
```

直接执行以下命令：

```bash
sudo tee /etc/dnsmasq.d/wuji-lan.conf > /dev/null <<'EOF'
# Wuji LAN DNS

# 仅在工业局域网 eth0 上提供 DNS
interface=eth0

# 只监听 Orin 的局域网地址。
# systemd-resolved 继续监听 127.0.0.53:53。
listen-address=192.168.100.70
bind-interfaces

# 不读取 /etc/hosts。
#
# Ubuntu 常见配置中可能存在：
# 127.0.1.1 wujibrain-desktop
#
# 如果让 dnsmasq 自动读取 /etc/hosts，
# 客户端可能同时得到 127.0.1.1 和 192.168.100.70。
# 因此这里关闭 hosts 自动导入。
no-hosts

# 显式发布局域网 DNS A 记录
host-record=wujibrain-desktop,192.168.100.70

# 不把不存在的单标签主机名转发给上游 DNS
domain-needed

# 不把私有地址的反向查询转发到公网
bogus-priv

# DNS 缓存
cache-size=1000

# 不直接读取 /etc/resolv.conf
no-resolv

# 非本地查询继续交给 systemd-resolved
server=127.0.0.53

# 调试阶段记录 DNS 查询
log-queries
EOF
```

---

## 5. 检查配置语法

执行：

```bash
sudo dnsmasq --test
```

正确结果应为：

```text
dnsmasq: syntax check OK.
```

如果不是 `syntax check OK`，先修正配置，不要继续启动服务。

如需查看配置文件：

```bash
cat /etc/dnsmasq.d/wuji-lan.conf
```

---

## 6. 启动并设置开机自启

启用开机启动：

```bash
sudo systemctl enable dnsmasq
```

重新启动：

```bash
sudo systemctl restart dnsmasq
```

检查状态：

```bash
systemctl status dnsmasq --no-pager
```

应看到：

```text
Active: active (running)
```

如果启动失败，查看：

```bash
sudo systemctl status dnsmasq --no-pager -l
sudo journalctl -u dnsmasq -n 100 --no-pager
```

---

## 7. 检查 53 端口监听

执行：

```bash
sudo ss -lntup | grep ':53 '
```

正确情况下应同时存在：

```text
127.0.0.53:53
192.168.100.70:53
```

其含义为：

```text
127.0.0.53:53
    -> systemd-resolved

192.168.100.70:53
    -> dnsmasq
```

不要因为使用 `dnsmasq` 而关闭 `systemd-resolved`。

---

## 8. 在 Orin 本机验证 DNS

先执行：

```bash
dig @192.168.100.70 wujibrain-desktop A +short
```

正确结果必须只有：

```text
192.168.100.70
```

不应该出现：

```text
127.0.1.1
```

如果仍然出现 `127.0.1.1`，确认配置中存在：

```text
no-hosts
```

然后重新启动：

```bash
sudo systemctl restart dnsmasq
```

进一步查看完整响应：

```bash
dig @192.168.100.70 wujibrain-desktop A
```

正确结果应包含：

```text
status: NOERROR
```

以及：

```text
;; ANSWER SECTION:
wujibrain-desktop.    0    IN    A    192.168.100.70
```

实际验证成功时的结果类似：

```text
;; ->>HEADER<<- opcode: QUERY, status: NOERROR
;; flags: qr aa rd ra

;; QUESTION SECTION:
;wujibrain-desktop.             IN      A

;; ANSWER SECTION:
wujibrain-desktop.      0       IN      A       192.168.100.70
```

---

## 9. 测试公网 DNS 转发

执行：

```bash
dig @192.168.100.70 www.google.com A
```

或者：

```bash
dig @192.168.100.70 www.google.com A +short
```

如果可以正常得到公网地址，则说明：

```text
客户端
  |
  v
dnsmasq 192.168.100.70:53
  |
  v
systemd-resolved 127.0.0.53:53
  |
  v
上游 DNS
```

转发链工作正常。

如需检查 Orin 当前上游 DNS：

```bash
resolvectl status
```

也可以直接测试 `systemd-resolved`：

```bash
dig @127.0.0.53 www.google.com A
```

---

## 10. 如果 Orin 启用了 UFW

查看防火墙：

```bash
sudo ufw status
```

如果结果是：

```text
Status: inactive
```

无需额外配置。

如果结果是：

```text
Status: active
```

放行局域网访问 TCP/UDP 53：

```bash
sudo ufw allow in on eth0 from 192.168.100.0/24 to any port 53 proto udp
sudo ufw allow in on eth0 from 192.168.100.0/24 to any port 53 proto tcp
```

再次检查：

```bash
sudo ufw status
```

---

## 11. 从 Windows 验证 Orin DNS

在 Windows PowerShell 中执行：

```powershell
Resolve-DnsName wujibrain-desktop `
    -Server 192.168.100.70 `
    -Type A `
    -DnsOnly `
    -NoHostsFile
```

正确结果应类似：

```text
Name                 Type TTL Section IPAddress
----                 ---- --- ------- ---------
wujibrain-desktop    A    0   Answer  192.168.100.70
```

这一步成功说明：

```text
Windows
    |
    | 普通 DNS 查询
    v
192.168.100.70:53
    |
    v
dnsmasq
    |
    v
wujibrain-desktop -> 192.168.100.70
```

已经正常。

> Windows 的 `nslookup` 对单标签主机名可能表现特殊。判断本方案是否正常时，以 `Resolve-DnsName -DnsOnly`、dnsmasq 日志和抓包结果为准。

---

## 12. 查看 dnsmasq 实时查询日志

在 Orin 执行：

```bash
sudo journalctl -u dnsmasq -f
```

然后在 Windows 再查询：

```powershell
Resolve-DnsName wujibrain-desktop `
    -Server 192.168.100.70 `
    -Type A `
    -DnsOnly `
    -NoHostsFile
```

正常情况下 Orin 会出现：

```text
query[A] wujibrain-desktop from 192.168.100.xxx
config wujibrain-desktop is 192.168.100.70
```

实际已经验证过的日志示例：

```text
query[A] wujibrain-desktop from 192.168.100.36
config wujibrain-desktop is 192.168.100.70
```

按 `Ctrl+C` 退出实时日志。

---

## 13. 使用 tcpdump 验证 DNS 数据包

如果需要从网络层确认请求和响应，在 Orin 执行：

```bash
sudo tcpdump -ni eth0 port 53
```

然后从 Windows 发起 DNS 查询。

实际验证成功的抓包结果为：

```text
192.168.100.36.56537 > 192.168.100.70.53:
    A? wujibrain-desktop.

192.168.100.70.53 > 192.168.100.36.56537:
    A 192.168.100.70
```

这直接证明：

```text
Windows -> Orin DNS 查询        正常
Orin -> Windows DNS A 记录响应  正常
```

按 `Ctrl+C` 停止抓包。

---

## 14. Android Emulator 使用 Orin DNS

先关闭已有 Emulator：

```powershell
adb emu kill
```

检查：

```powershell
adb devices
```

列出可用 AVD：

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\emulator\emulator.exe" -list-avds
```

当前 AVD 名称：

```text
Medium_Tablet
```

使用 Orin 作为 DNS 启动 Emulator：

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\emulator\emulator.exe" -avd Medium_Tablet -dns-server 192.168.100.70 -no-snapshot-load
```

等待启动后检查：

```powershell
adb devices
```

应看到：

```text
emulator-5554    device
```

---

## 15. Android Emulator 验证 DNS

先确认 IP 可达：

```powershell
adb shell ping -c 3 192.168.100.70
```

然后确认 hostname 可以解析：

```powershell
adb shell ping -c 3 wujibrain-desktop
```

正确结果应类似：

```text
PING wujibrain-desktop (192.168.100.70)
64 bytes from 192.168.100.70 ...
```

如果得到：

```text
ping: unknown host wujibrain-desktop
```

在 Orin 打开实时日志：

```bash
sudo journalctl -u dnsmasq -f
```

然后再次：

```powershell
adb shell ping -c 3 wujibrain-desktop
```

通过日志判断 Emulator 的 DNS 查询是否到达 Orin。

---

## 16. 验证 HTTPS

DNS 成功后，Android/Flutter 可以统一使用：

```text
https://wujibrain-desktop
```

可以从 PowerShell 让 Emulator Chrome 打开：

```powershell
adb shell am start `
    -a android.intent.action.VIEW `
    -d "https://wujibrain-desktop"
```

如果 Chrome 能够连接服务器，但提示：

```text
Your connection is not private
```

说明：

| 层级 | 状态 |
| --- | --- |
| Emulator 到 Orin IP | 正常 |
| DNS 解析 | 正常 |
| TCP 443 | 正常 |
| HTTPS 服务 | 正常 |
| Android 对私有 CA 的信任 | 尚未完成 |

此时已经不是 DNS 问题，后续应处理 Android/Flutter 对 CasiaHand Root CA 的信任。

---

## 17. 为什么不依赖 `.local`

原来的：

```text
wujibrain-desktop.local
```

由 Avahi/mDNS 提供。

Windows 在默认名称解析模式下可能可以解析：

```text
wujibrain-desktop.local -> 192.168.100.70
```

但这不代表普通 DNS 服务器中存在该记录。

例如强制只使用普通 DNS：

```powershell
Resolve-DnsName wujibrain-desktop.local `
    -Server 192.168.100.1 `
    -Type A `
    -DnsOnly `
    -NoHostsFile
```

可能返回 DNS 名称不存在。

Android Emulator 位于虚拟 NAT 网络中，也不适合依赖物理 LAN 上的 mDNS multicast。

因此正式 Flutter 客户端推荐使用：

```text
https://wujibrain-desktop
```

而不是：

```text
https://wujibrain-desktop.local
```

---

## 18. 增加更多局域网设备

如果以后需要增加其他静态主机，例如：

```text
wuyou -> 192.168.100.50
woosh -> 192.168.100.80
```

编辑配置：

```bash
sudo nano /etc/dnsmasq.d/wuji-lan.conf
```

加入：

```ini
host-record=wujibrain-desktop,192.168.100.70
host-record=wuyou,192.168.100.50
host-record=woosh,192.168.100.80
```

检查：

```bash
sudo dnsmasq --test
```

重启：

```bash
sudo systemctl restart dnsmasq
```

验证：

```bash
dig @192.168.100.70 wuyou A +short
dig @192.168.100.70 woosh A +short
```

---

## 19. 常用维护命令

### 查看服务状态

```bash
systemctl status dnsmasq --no-pager
```

### 查看是否开机自启

```bash
systemctl is-enabled dnsmasq
```

### 启动

```bash
sudo systemctl start dnsmasq
```

### 停止

```bash
sudo systemctl stop dnsmasq
```

### 重启

```bash
sudo systemctl restart dnsmasq
```

### 检查配置

```bash
sudo dnsmasq --test
```

### 查看最近日志

```bash
sudo journalctl -u dnsmasq -n 100 --no-pager
```

### 实时查看 DNS 查询

```bash
sudo journalctl -u dnsmasq -f
```

### 查看 53 端口

```bash
sudo ss -lntup | grep ':53 '
```

### 本机测试

```bash
dig @192.168.100.70 wujibrain-desktop A
```

### 抓取 DNS 数据包

```bash
sudo tcpdump -ni eth0 port 53
```

---

## 20. 最终验收

### Orin

检查服务：

```bash
systemctl is-active dnsmasq
```

应返回：

```text
active
```

检查 DNS：

```bash
dig @192.168.100.70 wujibrain-desktop A +short
```

应只返回：

```text
192.168.100.70
```

检查端口：

```bash
sudo ss -lntup | grep ':53 '
```

应存在：

```text
192.168.100.70:53
```

### Windows

执行：

```powershell
Resolve-DnsName wujibrain-desktop `
    -Server 192.168.100.70 `
    -Type A `
    -DnsOnly `
    -NoHostsFile
```

应解析为：

```text
192.168.100.70
```

### Android Emulator

使用 Orin DNS 启动：

```powershell
& "$env:LOCALAPPDATA\Android\Sdk\emulator\emulator.exe" -avd Medium_Tablet -dns-server 192.168.100.70 -no-snapshot-load
```

测试：

```powershell
adb shell ping -c 3 wujibrain-desktop
```

应解析为：

```text
wujibrain-desktop -> 192.168.100.70
```

---

## 21. 重要注意事项

1. **不要关闭 `systemd-resolved`。**

   当前设计是：

   ```text
   systemd-resolved -> 127.0.0.53:53
   dnsmasq          -> 192.168.100.70:53
   ```

2. **不要关闭 Avahi。**

   Avahi 可以继续提供：

   ```text
   wujibrain-desktop.local
   ```

   dnsmasq 则提供：

   ```text
   wujibrain-desktop
   ```

3. **建议保留 `no-hosts`。**

   这样可以防止 `/etc/hosts` 中的：

   ```text
   127.0.1.1 wujibrain-desktop
   ```

   被 dnsmasq 发布给局域网客户端。

4. **Flutter 正式通信地址建议统一使用：**

   ```text
   https://wujibrain-desktop
   ```

5. **DNS 成功不代表 Android 已经信任 HTTPS 私有 CA。**

   如果 Chrome 出现：

   ```text
   Your connection is not private
   ```

   说明 DNS、路由和 HTTPS 服务本身已经基本打通，下一阶段应处理 Android/Flutter 的 CA 信任配置。
