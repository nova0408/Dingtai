# CasiaHand 自签名 CA 与 API Gateway 证书

API Gateway 在 aiohttp 进程内直接终止 TLS，并正式监听 `443`。没有额外的 Nginx、Caddy
或其它反向代理。所有正式客户端在首次访问 `https://<orin-host>` 前，必须先安装并信任
`CasiaHand Root CA`；禁止用 `verify=False`、`curl -k` 或忽略浏览器告警代替证书安装。

`generated/` 和所有私钥都被排除在 Git 与服务同步清单之外。CasiaHand CA 私钥必须保存在
离线受控设备，不得复制到 Orin。Orin 只需要服务器私钥、服务器证书、完整证书链和 CA 公钥。
如果必须在新 Orin 上执行快速注册，注册结束后应将加密的 CA 私钥复制到离线受控位置，确认
备份可用后再从 Orin 的工作目录移走；客户端只安装 CA 公钥。

## 1. 正式生成

新 Orin 只需要执行注册脚本，不需要填写 hostname、SAN 或任何路径参数。脚本自动读取当前
`hostname`，并固定使用 `/home/wuji-brain/casiahand-pki`：

```bash
bash /home/wuji-brain/workspace/api_gateway/certificates/scripts/register_api_gateway.sh
```

脚本会要求设置 CA 私钥口令，自动创建 CA、签发当前 hostname 的服务器证书、安装服务器证书
并执行安装后校验。它不需要参数，也不会覆盖已有 PKI 目录。

CA 是签发机构，本身不验证主机名或 IP。服务器证书 SAN 只包含当前 `hostname` 的 DNS 名，
不包含任何 IP 地址；签发脚本和安装脚本只用当前 hostname 做证书链校验，不执行 IP 校验。
客户端必须使用该 hostname 访问，不能使用 IP 地址替代。服务器证书有效期为 825 天，CA 有效期
为 10 年，应在到期前重新签发并部署。

### 新 Orin 快速注册

注册脚本会安装服务器证书，但不会把 CA 公钥安装到客户端，也不会替换 Gateway 代码。脚本结束后，
先把输出目录中的 `casiahand-root-ca.cer` 复制到 Windows/Linux/Android 客户端并安装信任，
再在项目根目录运行 `scripts/sync_and_restart_services.ps1 -ApiGatewayOnly` 更新并重启 Gateway。
重复注册前必须先归档现有
`/home/wuji-brain/casiahand-pki/ca` 和 `orin` 目录；脚本不会覆盖已有 CA 或服务器私钥。

## 2. 安装到 Orin

证书已经生成在固定目录后，在 Orin 执行：

```bash
sudo bash /home/wuji-brain/workspace/api_gateway/certificates/scripts/install_api_gateway_certificate.sh
```

证书安装目录固定为 `/etc/dingtai/api-gateway/tls`。若已有证书，安装脚本会先备份到
`/etc/dingtai/api-gateway/.archive/tls-<时间戳>/`。服务器私钥权限为 `0640 root:wuji-brain`；
systemd 只赋予 Gateway 进程绑定低端口所需的 `CAP_NET_BIND_SERVICE`，服务不以 root 运行。
部署或重启后可做不访问硬件的只读检查：

```bash
gateway_hostname=$(hostname)
curl --cacert /etc/dingtai/api-gateway/tls/casiahand-root-ca.crt.pem \
  --resolve "${gateway_hostname}:443:127.0.0.1" \
  "https://${gateway_hostname}/api/v1/gateway/health"
```

## 3. Windows 客户端安装

在 Windows 本机仓库根目录执行一键下载与安装脚本。脚本会先检查 `ssh orin` 是否可用，
再从 Orin 的注册目录下载 CA 公共证书，并使用脚本自身的相对路径完成安装：

```powershell
powershell -ExecutionPolicy Bypass `
  -File .\api_gateway\certificates\scripts\install_casiahand_ca_windows.ps1
```

脚本固定安装到当前 Windows 用户信任库。SSH 不通时脚本会停止，并提示先配置 `ssh orin`；
不会继续下载或安装。
完全退出并重新打开浏览器或 GUI 后再访问。Firefox 若配置为独立证书库，需要在其证书设置中
另行导入 CA，或启用系统根证书信任。

## 4. Linux 客户端安装

```bash
sudo bash api_gateway/certificates/scripts/install_ca_linux.sh
```

脚本固定读取 `api_gateway/certificates/client/casiahand-root-ca.crt.pem`，支持使用
`update-ca-certificates` 的 Debian/Ubuntu 和使用 `update-ca-trust` 的
RHEL/Fedora。安装后重启浏览器或 GUI。使用独立证书库的应用仍需按应用要求导入 CA。

## 5. Android 客户端安装

Android 禁止脚本静默建立用户 CA 信任。可先通过 adb 复制证书并打开安全设置：

```powershell
pwsh -NoProfile -File .\api_gateway\certificates\scripts\stage_ca_android.ps1
```

然后在设备上选择“安装证书/CA 证书”，从 Download 安装 `CasiaHand-Root-CA.cer`。不同厂商
菜单名称可能不同。Android 7 及以后，应用默认不一定信任用户安装的 CA；使用 Dio/Flutter
或原生 Android GUI 时，应在应用的 Network Security Config 中显式允许用户 CA：

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <base-config cleartextTrafficPermitted="false">
        <trust-anchors>
            <certificates src="system" />
            <certificates src="user" />
        </trust-anchors>
    </base-config>
</network-security-config>
```

并在应用清单的 `<application>` 上设置
`android:networkSecurityConfig="@xml/network_security_config"`。正式发布时应评估把 CasiaHand
CA 公钥随应用打包并只信任该 CA，避免扩大到全部用户 CA；不得在 Dio 中关闭证书校验。

## 6. 更新与撤销

更换服务器证书时先备份 `/etc/dingtai/api-gateway/tls`，再运行安装脚本并重启 Gateway。
如果 CA 私钥泄露，必须生成新的 CA、重新签发服务器证书，并在所有平台删除旧 CA 后安装
新 CA；仅重新签发服务器证书不能解决 CA 私钥泄露。
