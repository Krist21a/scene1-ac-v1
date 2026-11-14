# Scene1 最新开发代码备份

2026-11-15 上传工作站 `/home/krist/scene1-ac-speed` 的最新代码，包括 11 月 9 日传统策略 V37 和 11 月 10 日模型控制桥 v7。固定标签：`scene1-latest-20261115`。这是开发状态备份，未通过完整任务验收。

| 内容 | 实际状态 |
| --- | --- |
| 传统策略 V37 | 最后一轮 seed 1476199888 完成 3/4，949.170 秒，第四件观察称重垫超时；六 seed 一致性及 650 秒目标未通过 |
| 模型控制桥 v7 | 已完成抓取、抬升、搬至称重区；用户介入直接松爪后结束，未完成正常自动称重与入箱 |
| 局部 ACT 工具 | 采集、训练、部署入口已保存；未产出已验收的新局部 ACT 模型 |

完整说明：[V37 技术报告](ARM_JITTER_TECHNICAL_REPORT_20261109.txt)、[v7 结束报告](model_control/results/one-parcel-v7-20261110/REPORT.txt)、[模型控制桥接口](model_control/README.txt)。旧交接文件及其“运行中”字样是历史记录，以这两份结束报告为准。历史成绩和冻结清单保持原样；[旧版首页](recovery/originals/README.md)及旧 Git 标签仍可追溯。

## 从 GitHub 取回

```bash
git clone https://github.com/Krist21a/scene1-ac-v1.git
cd scene1-ac-v1
git checkout scene1-latest-20261115
python3 recovery/verify.py
# 安装 GitHub CLI gh 后，下载控制器库、SDK、旧 ACT 权重及准备 bag。
python3 recovery/download_assets.py --runtime
# 可选：恢复官方仿真源码 Git bundle（不是 Docker 镜像）。
python3 recovery/download_assets.py --simulator
```

源码直接保存在 Git；大文件在[本次 Release](https://github.com/Krist21a/scene1-ac-v1/releases/tag/scene1-latest-20261115)，逐文件和附件 SHA-256 清单位于 `recovery/`。不依赖原本地目录就能取回；当前备份没有使用私有 Scene2 仓库作为下载依赖。

[完整恢复说明](docs/RECOVERY.md)列出宿主/容器路径、依赖、旧生产基线和启动前提。Docker 镜像按用户要求没有上传，需要另行获取；Python/ROS/GPU 环境需重新安装。此次只核验代码与附件恢复，不启动仿真、不把文件校验当作策略验收，也未执行整机删除后的完整重装抓取。

已移除当前候选中的旧代码测试/夹具和调试日志，保留运行代码、控制器、网格、配置、历史验收材料及最新必要报告。全量历史录像与原始传感器日志没有在本次重复上传，不能用该仓库替代全部研究数据归档。
