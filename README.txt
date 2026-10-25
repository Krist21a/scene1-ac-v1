Scene1 A+C 第一代完整版本 — ac-v1-20261025

冻结实现：executor_pad_dense + A+C。这是独立版本归档；原v63仓库保持不变。
A修复正常MPC入口的手臂首参考点索引，C在WBC输出后提供有界腕部PD补偿。
根目录187个策略、配置和测试文件与完成验收的冻结清单逐字一致。
没有新增视觉模型、训练任务或WBC内部QP改动。

验收与耗时
  seed0：四包12679步，2072.117秒
  seed2：四包12727步，2060.058秒
  seed4：四包12621步，2083.284秒（冻结后新seed）
  seed5：四包12644步，2066.977秒（冻结后新seed）
平均正式执行2070.609秒，即34分31秒；不含复位、初始化、分析归档。
相同冻结路径下原控制器四轮均失败，全部失败记录保留。
历史v63完整回合为38.40–46.01分钟，本版观察到的完整任务耗时更短。
这些历史运行条件并非严格配对，当前对照也没有原版完整成功回合，故不能把跨版本耗时差全部归因于A+C。
所有task_success=null，只能称原有视觉检查通过，不能称官方裁判成功。
腕部误差明显改善，近端部分阶段退化；静差/延迟仍存在。
seed4初始化有3个短暂附加力矩裁剪采样（约41ms），正式任务阶段四轮均无采样裁剪。
详见ACCEPTANCE_REPORT.txt及diagnostics/path-tracking-continuation-20261023/MULTISEED-PAD-RESULTS-20261025.json。

内容
  根目录：最终策略、四包配置、529项回归测试及真实RGB-D夹具。
  controllers/：A、C精确源码、补丁、编译/链接命令、原task/QR。
  evidence/analysis-full/：保留历史成功与失败的分析摘要及逐阶段指标。
  diagnostics/：原研究报告、清单、消融、失败记录和工作站脚本快照。
  WORKSTATION_RUNBOOK.txt：原工作站构建、自动复位运行、日志和恢复命令。
  tools/：本归档的哈希核验、隔离重建、空闲状态控制版本切换工具。
Release附带精确A/C二进制和原版回退二进制，以及从v63继承的运行依赖包。
不包含Docker镜像、官方仿真场景、完整周期RGB-D/ROS原始数据；其原件保留在工作站外盘，逐文件证据清单随归档保存。
诊断脚本包含原工作站绝对路径，并非任意机器一键运行工具；不要将历史状态文件误作当前状态。

下载与核验
  git clone https://github.com/Krist21a/scene1-ac-v1.git
  cd scene1-ac-v1
  git checkout ac-v1-20261025
  gh release download ac-v1-20261025 --repo Krist21a/scene1-ac-v1
  sha256sum -c SHA256SUMS.release
  tar -xzf scene1-ac-v1-control-runtime-20261025.tar.gz
  python3 tools/verify_release.py
原依赖包v63-runtime-assets-20261017.tar.gz根目录对应/root/LeTools-Learning；先解压至暂存目录，按evidence/inherited-runtime-assets.json检查，再部署到匹配的官方环境。
依赖ROS Noetic、Kuavo SDK、原Kuavo仿真/构建工作区、kls_dev Python/CUDA环境、公共URDF与既有相机标定。与原库ABI不兼容的环境不得直接使用这些二进制。

控制选择与恢复（宿主执行；默认只打印，--apply才切换；要求两容器完全空闲）
  python3 tools/select_controller.py ac
  sudo -n python3 tools/select_controller.py ac --apply
  sudo -n python3 tools/select_controller.py baseline --apply
切换逐项检查哈希与已知原件，分别原子替换库/VR/task/QR；四个文件不是跨文件事务，如中断应在启动前重新执行并核验。
本次发布仅上传归档，没有切换本机活动控制器；发布时仍为原版。

测试与运行
同步本仓库到算法容器独立目录/root/scene1_ac_v1；不要覆盖原生产或v63。
  docker cp . kuavo-ros-noetic-s12-ready:/root/scene1_ac_v1
  docker exec -w /root/scene1_ac_v1 kuavo-ros-noetic-s12-ready /root/miniforge3/envs/kls_dev/bin/python -m unittest discover -s tests
在仿真容器中加载原ROS工作区环境，经官方入口独立复位：
  rosrun challenge_cup_task_template challenge_task.py --scene scene1 --seed 0 --no-timer-gui
匹配原相机/SDK就绪、初始化、robot_guard和target_guard保护监控后，在算法容器执行：
  bash /root/scene1_ac_v1/run.sh --config configs/visual_cycle_four_parcels.yaml --task go_run --mode optimized --stop-after-parcels 4 --max-steps 16800
默认config.yaml是历史录制配置，不能代替明确的四包配置。
完整官方复位、就绪等待和监控顺序见原full_experiment.py；推荐原工作站使用WORKSTATION_RUNBOOK.txt中已经验证的run_one.py流程。新环境需要先适配目录和容器标识，不能直接照搬原工作站路径假设。

隔离重建（仿真容器内，已有原编译工作区；输出目录必须不存在）
  python3 tools/rebuild_controller.py A --output /tmp/scene1-ac-v1-build-A
  python3 tools/rebuild_controller.py C --output /tmp/scene1-ac-v1-build-C
复用实际记录的编译/链接命令和原依赖，不修改活动控制库。重新构建产物可能因环境不同产生新哈希，不能冒充已验证二进制。

版本边界
本次为用户指定的一代完整版本归档，不代表已证明最优控制或已自动晋升生产。
冻结后新seed4/5均一次运行通过；有限样本不能保证任意seed稳定。
原生产与v63保持原字节；全部旧失败和中断均保留，没有筛选成功覆盖失败。
