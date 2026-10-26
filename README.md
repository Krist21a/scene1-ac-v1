# Scene1 A+C v1：四包任务提速版

基于 [`ac-v1-20261025`](https://github.com/Krist21a/scene1-ac-v1/releases/tag/ac-v1-20261025) 的 A+C 控制方案，本版将四包任务从约 **34 分 31 秒**推进到 **14 分 13.8 秒–14 分 41.8 秒**。同一冻结版本已在 seed 0、3、4 完成四包，**643 项测试通过**；用户已确认本轮目标初步完成。

当前配置为 [`configs/speed_accessible.yaml`](configs/speed_accessible.yaml)，发布标签为 `ac-v1-speed-20261026`。原始 v1 标签和回退资产继续保留。

> 当前结果是三轮实际执行和原有视觉、安全检查通过，完整多种子验收尚未完成；所有 `task_success=null`，不代表官方裁判已判定成功。

## 实际改善了多少

计时采用正式策略开始至第四包最终完成的**实际墙钟时间**，包含感知、计算、控制、等待及在线诊断；不含场景复位、初始化、事后分析和归档。不是用仿真时间代替实际耗时。

| 版本 / 轮次 | 四包实际耗时 | 控制步数 | 验证结果 |
| --- | ---: | ---: | --- |
| 原 A+C v1，seed 0/2/4/5 均值 | 2070.609 秒（34分30.6秒） | — | 四轮视觉通过 |
| 本轮同机 A+C 基线复测，seed 0 | 2030.285 秒（33分50.3秒） | 12663 | 四包完成 |
| 本版，seed 0 | **853.772 秒（14分13.8秒）** | 8886 | 四包视觉、安全、计时审计通过 |
| 本版，seed 3 | **875.514 秒（14分35.5秒）** | 9114 | 四包视觉、安全、计时审计通过 |
| 本版，seed 4 | **881.845 秒（14分41.8秒）** | 9135 | 四包视觉、安全、计时审计通过 |

本版三轮均值 **870.377 秒（14分30.4秒）**。相较原 A+C 四轮均值，观察到耗时减少约 **58.0%**、约 **2.38 倍**执行速度。seed 0 相较本轮同机基线减少约 **58.0%**。均值的种子组合不同，且本版同时调整了计算、控制节奏和路径，这些数字是整套方案的观察结果，不能当作 A、C 单因素的因果收益。

证据见 [`evidence/speed-v1-20261026/results.json`](evidence/speed-v1-20261026/results.json)、同目录逐轮 `metrics-full.json` / `audit.json`、测试日志和流水线回执。三个成功回合使用同一冻结策略清单；seed 0/4 复用 seed 3 来源的测试结果，不是每轮重新运行 643 项测试。

## A+C 做了什么，带来了什么提升

### A：修正 MPC 正常控制入口的手臂参考点

在 `direct_to_wbc=false` 的正常 MPC→WBC 链路中，把首参考点手臂关节读取从错位的 `segment(24,14)` 改为 `tail(14)`，保证传入的 14 个手臂关节对应正确状态。A 解决的是参考点正确性问题，没有绕过原控制链。

### C：在 WBC 输出后补偿腕部跟踪误差

对六个腕部关节施加有界 PD 补偿：

```text
额外力矩 = clip(1.0 × (qWBC − qactual) − 0.05 × qdotactual, ±0.5 N·m)
总力矩限制 = ±14.1 N·m
```

C 作用于 WBC 输出之后，**不是修改 WBC 内部 QP**；原任务权重、dynamic QR、物理边界保持。没有叠加积分或重力补偿。

原 v1 相同路径的对照中，A+C 在 seed 0、2、4、5 均完成四包，原控制器对应四轮均提前停止。首包释放准备的共同阶段，腕部 WBC 参考→反馈 RMS 为：

| seed | 原控制器 | A+C | 腕部误差下降 |
| --- | ---: | ---: | ---: |
| 0 | 0.393751 rad | 0.041265 rad | 约89.5% |
| 4 | 0.382018 rad | 0.042430 rad | 约88.9% |
| 5 | 0.387302 rad | 0.036528 rad | 约90.6% |

这些证据支持 A+C 改善腕部跟踪和该路径下的任务完成性，为后续提高动作效率提供基础。但共同阶段的近端误差有所增加，腕部静差和动态延迟仍存在。原控制器没有完整成功回合，因此不能计算合格的“原控制器→A+C”四包配对提速率，也不能把本版约58%的耗时下降全部归因于 A+C。详细对照和失败记录见原版 [`ACCEPTANCE_REPORT.txt`](ACCEPTANCE_REPORT.txt)。

## 如何把约34分钟降到15分钟以内

本版保留原 A+C 控制器二进制，主要减少上层计算阻塞、低效控制周期和可避免的观察/到位等待。

| 改进 | 实现方式 | 为什么能减少耗时 |
| --- | --- | --- |
| 感知计算加速 | 原生 C 连通分组、RGB 转换、点云/投影和屋顶掩码；按目标局部区域检测；限制计算库线程数量 | 减少每个循环的 Python 开销、整图重复操作与线程争用 |
| 运动学与退让求解加速 | 相同关节状态的 FK 缓存、解析几何雅可比、SLSQP 内部原生 FK/SAT 约束 | 减少反复运动学和数值差分计算；代表性离线退让投影由99.25ms降至32.82ms，此数字不是整任务收益 |
| 当前帧感知并行 | 重叠 RGB-D 读取与独立图像检测，控制前等待本轮结果 | 减少串行等待，不使用积压旧帧替代当前观测 |
| 分阶段控制节奏 | 适合快速移动的阶段最高20Hz，抬升15Hz，接触/部分持物阶段维持10Hz；分阶段调整伺服增益 | 更快推进移动和收敛，避免所有阶段沿用同一低效节奏；仍受原步长、速度和到位门槛约束 |
| 近回位稳定 | 远处快速回位，接近时锁定原10Hz及原增益 | 减少近目标振荡造成的长时间等待 |
| 遮挡与采样修正 | 托盘被遮挡时，在满足原前提后一次平移8cm重新观察；接近抓取固定使用更密集的候选采样 | 避免手臂遮挡导致观察超时，以及稀疏采样中心偏移导致误停止；恢复后仍须通过原视觉确认 |
| 四个投放位置调整 | 横向偏移为[-115,-53,9,71]mm；第四位置深度偏移为-10mm；第四次到达误差收紧到10mm | 让第四包远离近侧箱壁遮挡并更易到达，减少释放后不可见问题；位置间距62mm，保留原箱壁余量和分槽检查 |
| 释放动作稳定 | 张爪目标每步变化由0.08减为0.04，保留保持补偿和原预算 | 抑制卸载瞬间的不稳定，避免提速后释放失败；这项用于可靠性，并非单独追求最快张爪 |

没有加快仿真物理时间步、增大执行器极限或关闭检查。碰撞最终检查仍使用独立原始运动学和扫掠检查，保持3mm余量；原生求解器不能自行判定最终安全。目标移动10mm检查、接触/称重条件、入箱新增点及稳定帧检查继续保留。新布局对第四次到位采用更严格的10mm门槛，前三次保持25mm。

## 验证范围与已知边界

- 三轮 seed 0/3/4 均为4/4完成、连续步骤、完整计时覆盖，正式阶段目标裁剪和已采样 PD 裁剪为0。
- seed 2 在592.763秒、完成两包时因用户要求的“额度剩余10%停止”规则中止，**不计成功**；seed 1/5及保留种子6–9尚未完成当前版本验证。6–9未参与本轮调参。
- 原有失败、中断和中间候选保留在 [`SPEED_PROGRESS.txt`](SPEED_PROGRESS.txt)。`configs/speed_*.yaml` 中其他配置为研究历史，当前选择仅为 `speed_accessible.yaml`。
- 当前是初步达标版本，尚不能保证任意种子都在15分钟内完成；不是官方成功率证明。
- 全量 RGB-D、ROS 原件保存在工作站外盘，仓库提供审计与指标，未包含 Docker 镜像、官方场景或运行依赖模型。

## 下载、核验与运行

依赖与原 v1 相同：ROS Noetic、Kuavo SDK、匹配的仿真/编译工作区、`kls_dev` Python环境、URDF和相机标定。原控制器 ABI 不匹配时不能直接使用已有二进制。

```bash
git clone https://github.com/Krist21a/scene1-ac-v1.git
cd scene1-ac-v1
git checkout ac-v1-speed-20261026

# 本次发布附带实际测试使用的原生计算库及校验和。
gh release download ac-v1-speed-20261026 --repo Krist21a/scene1-ac-v1
sha256sum -c SHA256SUMS.speed
tar -xzf scene1-ac-v1-speed-native-20261026.tar.gz

# A/C及原版回退二进制继续使用原始v1资产，无需重复下载已有文件。
gh release download ac-v1-20261025 --repo Krist21a/scene1-ac-v1
sha256sum -c SHA256SUMS.release
tar -xzf scene1-ac-v1-control-runtime-20261025.tar.gz
python3 tools/verify_release.py
```

原 `v63-runtime-assets-20261017.tar.gz` 按 [`evidence/inherited-runtime-assets.json`](evidence/inherited-runtime-assets.json) 恢复到匹配环境。先核验冻结文件和控制器，再部署至独立目录；不要覆盖原生产目录。

需要在其他兼容环境重编译计算库时，可执行 `python3 tools/build_components.py`。新编译二进制可能有不同哈希，不能冒充已测试发布资产，应重新测试。

在原工作站，用已适配的驱动启动完整复位、保护监控与四包运行：

```bash
sudo -n python3 tools/speed_trial.py speed-v1-new-run --seed 0 --parcels 4 \
  --speed-profile configs/speed_accessible.yaml
```

运行标签必须唯一。该驱动依赖原工作站目录、容器、外盘和额度日志，结束后会恢复原控制器；新机器须先适配。`usage_guard.py` 绑定原会话日志，新会话需更新为真实当前日志并保留停止规则，不能直接照搬旧额度状态。

如已按原工作站流程完成官方复位、SDK/相机就绪、A+C控制选择及保护监控，在算法容器的独立部署目录中，策略入口为：

```bash
bash run.sh --config configs/visual_cycle_four_parcels.yaml \
  --speed-profile configs/speed_accessible.yaml \
  --task go_run --mode optimized --stop-after-parcels 4 --max-steps 16800
```

必须显式传入 `--speed-profile` 才启用本版提速组合。单独执行策略入口不等于完成环境和监控准备。643项测试的对应命令为：

```bash
/root/miniforge3/envs/kls_dev/bin/python -m unittest discover -s tests
```

控制选择、回退及旧环境准备详见 [`README.txt`](README.txt)、[`WORKSTATION_RUNBOOK.txt`](WORKSTATION_RUNBOOK.txt)；这些是原版历史文档，其529项测试和34分钟结论属于旧标签。本版以本 README、提速配置及新冻结清单为准。

## 版本与证据

- 当前代码/配置/测试及原生库：[`241文件冻结清单`](evidence/speed-v1-20261026/policy-manifest.json)。发布文档和打包工具更新不改变该策略清单。
- 当前三轮证据、643项测试记录、中止回执：[`evidence/speed-v1-20261026/`](evidence/speed-v1-20261026/)。
- 实施过程及待验收种子：[`SPEED_PROGRESS.txt`](SPEED_PROGRESS.txt)、[`SPEED_RESULT.txt`](SPEED_RESULT.txt)。
- A/C源码、补丁和构建材料：[`controllers/`](controllers/)。
- 原版回退：[原始 v1 Release](https://github.com/Krist21a/scene1-ac-v1/releases/tag/ac-v1-20261025)，标签保持不变。

本次发布更新 GitHub 代码与文档，本机活动控制器保持已恢复的原版状态。
