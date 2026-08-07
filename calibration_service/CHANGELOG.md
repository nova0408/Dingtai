# Calibration Service 版本日志

当前版本：`1.1.1`

## 1.1.1 - 2026-08-07

### 修复

- 收回手眼算法、先验记录算法和 CameraPipeline HTTP 协议 DTO；服务运行时不再导入仓库 `src/` 或 CameraPipeline Python 包。

## 1.1.0 - 2026-08-07

### 新增

- 计算结果先进入待确认缓存，新增取消接口和替换二次确认接口。
- 正式替换时保留旧文件，并按 `yymmdd_hhmmss` 后缀重命名，不直接删除。
