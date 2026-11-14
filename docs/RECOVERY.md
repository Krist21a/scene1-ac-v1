# Scene1 从 GitHub 恢复

本次标签 `scene1-latest-20261115` 保存当前开发状态，包含传统策略 V37、模型控制桥 v7、局部 ACT 工具、配置与控制器源码。备份校验不等于完整策略验收；最新失败和用户介入情况见首页链接的结束报告。

## 1. 代码与附件

在有 Python 3.10+、Git 和 GitHub CLI `gh` 的 Linux 主机运行：

```bash
git clone https://github.com/Krist21a/scene1-ac-v1.git
cd scene1-ac-v1
git checkout scene1-latest-20261115
python3 recovery/verify.py
python3 recovery/download_assets.py --runtime --simulator
```

仓库为公开仓库，网页也可直接下载 Release 附件；CLI 若要求认证，可先 `gh auth login`。手动下载时把附件放到仓库 `downloads/`，再运行同一脚本，它会检查固定大小/SHA-256 后解包，不重新下载。

`source-files.json` 记录原工作区保留文件的字节身份；因发布而重写的首页、忽略规则，其原件保存在 `recovery/originals/`。`runtime-files.json` 检查非 Docker 附件中的全部文件。解包后的目录分工：

- `controllers/`：A/C、原版回退、WBC reduced/warmstart 的实际二进制；对应源码/构建身份在 Git。
- `scene1/_components.so`：原生感知计算库；源码同样保留。
- `runtime/LeTools-Learning/`：当前部署代码、配置、原 ACT 权重、归一化参数与 `rosbag/go.bag`。
- `runtime/kuavo_humanoid_sdk*`：实际使用的 SDK 1.3.3 与元数据。
- `runtime/scene1_act_optimization/`：旧生产代码、公开 URDF、标定、原控制器及编排必需的历史基线。
- `downloads/scene2-simulator-51b3def.bundle`：官方仿真源码 Git bundle。文件沿用共同备份名称，但本次附件完全存放在 Scene1 仓库，不需要访问 Scene2。

旧 ACT 权重用于旧混合模式，是原有模型，不是新训练的局部 ACT 模型。`model_control/` 本身不加载 ACT。当前代码、旧基线和历史冻结清单分别保留，不能用新代码替换旧清单来宣称旧成绩成立。

## 2. Docker 和系统条件

按用户要求未上传任何 Docker 镜像。原仿真镜像为 `kuavo_challenge_cup_2026:latest`，镜像 ID `sha256:8eafa6a83f9edeb93e28da7cabe6b930fcccd88f02b4c3b57668549b6bae7e50`；原算法镜像为本机自建 `kuavo-ros-noetic:s1-s2-tested`，不能假设公网有同名镜像可拉取。

官方镜像下载地址来自已归档官方源码的 `readme.md`：

```bash
wget https://kuavo.lejurobot.com/challenge_cup_2026/kuavo_challenge_cup_2026_latest.tar.gz
docker load -i kuavo_challenge_cup_2026_latest.tar.gz
```

2026-11-15 本机网络检查该地址出现 TLS 握手错误，未验证镜像下载成功；若仍不可用，应向赛事方获取镜像。镜像/驱动重新安装和上游下载源可用性不在本次 GitHub 字节恢复验证范围。

宿主需 NVIDIA 驱动、NVIDIA Container Toolkit、Docker、X11、`python3`、`gh`、`git`、`ffmpeg`、`xdotool`、`xauth`。编排中的默认用户名为 `krist`、显示器为 `:0`、容器名如下；换机器时应按此路径恢复或明确修改路径后重新核验。

| 用途 | 原容器名 | 关键条件 |
| --- | --- | --- |
| 仿真 | `kuavo_challenge_container_GPU_046e4301` | host 网络/IPC、GPU、X11、工作区挂载 `/root/kuavo_ws` |
| 算法 | `kuavo-ros-noetic-s12-ready` | host 网络、GPU、ROS Noetic、Python 环境 `/root/miniforge3/envs/kls_dev` |

## 3. 官方工作区与算法依赖

官方上游为 `https://gitcode.com/OpenLET/leju-kuavo-challenge-cup-2026.git`，固定提交 `51b3defaf8c032957647c7aa193d1fa20daef1f3`。从本仓库附件恢复，不依赖该上游在线：

```bash
mkdir -p /home/krist/kuavo_course/workspace
git clone downloads/scene2-simulator-51b3def.bundle /home/krist/kuavo_course/workspace/leju-kuavo-challenge-cup-2026
git -C /home/krist/kuavo_course/workspace/leju-kuavo-challenge-cup-2026 checkout 51b3defaf8c032957647c7aa193d1fa20daef1f3
```

根据该工作区官方 Docker 启动脚本创建仿真容器，并确认其名称与上表一致。在容器工作区按官方说明编译：

```bash
cd /root/kuavo_ws
source installed/setup.zsh
catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release
catkin build kuavo_msgs challenge_cup_simulator challenge_cup_task_template humanoid_controllers humanoid_interface_ros
source devel/setup.zsh
```

在全新算法容器安装 Miniforge 至 `/root/miniforge3`（来源 `https://github.com/conda-forge/miniforge`），确认 `/opt/ros/noetic/setup.bash` 存在。把已解包的整个 Scene1 仓库复制到 `/root/scene1-ac-speed`，然后在容器内执行：

```bash
cd /root/scene1-ac-speed
bash recovery/install_algorithm_runtime.sh
```

脚本会拒绝覆盖已有算法环境；它用保存的 conda 显式包清单、pip 固定版本重建依赖，再恢复 LeTools、SDK 和旧生产基线。conda/PyPI 仍需联网，所有 Python 包并未做离线镜像。LeTools 源提交 `caf1892a3937c3bbbb075b6d88de8d8be6010116`，附件还包含当前实际修改。

记录的原环境为 Python 3.12.12、NumPy 1.26.4、SciPy 1.15.3、PyYAML 6.0.3、OpenCV headless 4.13.0.92。原环境 OpenCV 声明 NumPy>=2，却实际装了 NumPy 1.26；这项既有依赖声明冲突如实保留，安装脚本的 `--no-deps` 只用于重建已记录版本，不代表环境依赖无冲突。

## 4. 传统策略编排的旧路径依赖

`tools/speed_trial.py` 的生产一致性保护仍依赖旧项目；仅下载最新 Python 源码不足以满足其检查。本次附带其必要运行子集。在全新宿主从仓库根执行：

```bash
# 仅在目标不存在时创建；不要覆盖原来正在使用的项目。
test ! -e /home/krist/kuavo_course/scene1_act_optimization
cp -a runtime/scene1_act_optimization /home/krist/kuavo_course/scene1_act_optimization
export SCENE1_EVIDENCE_ROOT=/media/krist/Data/scene1-evidence-restored
mkdir -p "$SCENE1_EVIDENCE_ROOT"
```

`recovery/install_algorithm_runtime.sh` 已将同一旧基线复制到算法容器 `/root/scene1_act_optimization`。容器源码、宿主源码和原始生产清单必须一致；不能删除这些保护来跳过缺失依赖。

传统编排还使用宿主 `/home/krist/.local/bin/codex` 的已登录账号读取实时额度，未包含账号凭据；恢复后需自行安装、登录。`SCENE1_RESET_CREDIT_RECEIPT` 默认不要设置，它是历史按用户授权使用一次重置卡的显式入口。新机器没有旧会话/账户配置时，传统自动编排可能在额度预检处拒绝启动；模型控制桥入口不依赖该额度编排。

在两容器空闲且官方工作区 ABI 相符时，当前传统 V37 使用 `wbc_reduced`，其库 SHA-256 为 `2a8fda4f2c3f74dee8661db4c3235a74dda1464a77d38a5a7398d049c4ffae8c`。旧 `origin_pd`、`baseline` 和 `wbc_warmstart` 也保留必要工件。旧研究列表中其他已删除变体不承诺恢复。

新完整回合入口（此备份工作没有执行）：

```bash
sudo -n env SCENE1_EVIDENCE_ROOT=/media/krist/Data/scene1-evidence-restored \
  python3 tools/speed_trial.py restored-v37-new-run --seed 1476199888 --parcels 4 \
  --speed-profile configs/speed_traditional_smooth_profile.yaml --controller wbc_reduced \
  --record-video --cpu-layout p2-algo-p4e8-sim-critical-pcores
```

必须创建新回合标签和清单，不用缺失历史运行数据的 `--frozen-from` 冒充历史复验。CPU 布局和最低磁盘空间需符合工具要求；不同 CPU 拓扑需要重新配置与验证。

## 5. 模型控制桥

以 [model_control/RUNTIME.txt](../model_control/RUNTIME.txt)、[接口说明](../model_control/README.txt)和 [v7 结束记录](../model_control/results/one-parcel-v7-20261110/REPORT.txt)为准。旧文档中的运行中 socket、PID、Xauthority 和临时挂载不在重启后继续有效。

桥使用 `model_control/`、`scene1/`、`run_scene1.py`、`configs/` 的同一根目录；可将容器工程根设置为 `/root/scene1-ac-speed`，相应设置 `PYTHONPATH=/root/LeTools-Learning:/root/scene1-ac-speed`。`host_client.py` 用 `--root /root/scene1-ac-speed` 指向该根。必须重新生成显示认证、绑定 DATA 存储，并通过官方 Scene1 reset 建立全新回合。

11 月 10 日模型实验实际控制器为 C，库 SHA-256 `ab20325136810cc785bfdb4abd80c3ecc2b2e73627ebebea42b904bd7bb3f0ba`，与传统 V37 的 WBC reduced 不同。不能把两种控制身份混用后沿用旧成绩。官方源码、对应 A/C 和 baseline 二进制均已备份；已有工具在未知工件/非空闲状态下应拒绝切换。

## 6. 保存与验证边界

本次只备份/下载核验，不启动机器人、仿真、训练或重跑旧批次；不新建或恢复代码测试阶段。当前任务策略未验收，不承诺任意 seed 成功。Git 历史保留已发布旧版本；当前树不恢复已删除的单元测试和临时夹具。

全部历史 RGB-D、控制链、录像、旧实验快照不在本次附件中；Git 中保留最新报告、必要指标、失败情况与历史成绩来源。要保留全量研究数据，应另外保留 DATA 盘原始归档。Docker 镜像、操作系统、GPU 驱动、显示授权和账号凭据均需另行恢复；本次未执行整机删除后的完整重装抓取。
