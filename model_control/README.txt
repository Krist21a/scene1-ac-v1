Scene1：对话视觉模型控制实验

当前状态（2026-11-10）
控制桥、会话动作协议、异步终态与观测事件、自体点过滤、受约束末端伺服
及夹爪适配已集成。官方seed 0实际回合已完成抓取、抬升和搬运至称重区。
当前one-parcel-v6-20261110继续验证称重放置，一包抓取→称重→入箱尚未验收；
原速录像和桥事件持续记录，不能把持包到达称重区视为完成称重。
单次动作返回reached、视频文件存在或文件名含full，都不等于任务成功。

控制方式
当前对话中的模型根据相机、深度和机器人反馈选择末端目标、夹爪及头部动作。
桥不加载ACT，不执行旧VisualCycle.update或ResidualAssist.apply任务状态机，
不回放go.bag。复用的VisualCycle._servo与几何方法只负责执行模型指定动作。
控制链仍为正常MPC→WBC；模型不直接写力矩，不读取仿真物体真实坐标。
模型推理期间仿真继续运行，唯一SDK发送线程按10Hz墙钟重复保持最后参考。

文件
  bridge.py           Unix socket服务、观测、命令去重、日志与动作调度。
  runtime_support.py  仿真SDK初始化、唯一发送线程、反馈缓存和停止保持。
  motion_adapter.py   模型指定末端/夹爪动作的有界伺服、接触与持物几何检查。
  client.py           容器内客户端，从stdin读一条JSON。
  host_client.py      宿主客户端，调用容器客户端并保存请求/结果/相机原件。
  watch.py            只读等待简短进度/终态，不等待最终观测保存。
  record_video.py     宿主机实时MuJoCo窗口录制，封装后保存时长与哈希收据。
  RUNTIME.txt         本机容器、显示认证、官方场景启动、NVMe映射及运行身份。

启动
先按RUNTIME.txt核对容器、显示认证、NVMe挂载、官方seed及控制器身份。
官方launch包含底层控制器。不要并行运行旧ACT/传统任务入口或其他动作发布者。
将model_control、scene1、run_scene1.py及configs保留在同一工程根目录。
当前算法容器工程根为/tmp/scene1-model-experiment，Python为
/root/miniforge3/envs/kls_dev/bin/python。先source /opt/ros/noetic/setup.bash，
设置PYTHONPATH=/root/LeTools-Learning:/tmp/scene1-model-experiment。

在算法容器工程根启动（output必须是不存在的新目录）：
  /root/miniforge3/envs/kls_dev/bin/python model_control/bridge.py \
    --config configs/visual_cycle_four_parcels.yaml \
    --output /root/scene1_model_evidence/<新回合>/bridge-run
默认socket为/tmp/scene1-model.sock，可用--socket覆盖；已有socket不会自动删除。
启动会检查仿真/双臂/关节模式、普通MPC路由、外部手臂控制模式和命令发布者。
它不执行准备姿态，也不重置头部或夹爪。

宿主调用
在/home/krist/scene1-ac-speed下调用；--run-dir用于保存本轮模型决策和观测：
  echo '{"op":"observe"}' | python3 model_control/host_client.py \
    --run-dir /media/krist/Data/scene1-model-control-20261110/<新回合>/model-decisions
  echo '{"op":"observe","pixel":[320,240]}' | python3 model_control/host_client.py \
    --run-dir /media/krist/Data/scene1-model-control-20261110/<新回合>/model-decisions
  echo '{"op":"status"}' | python3 model_control/host_client.py \
    --run-dir /media/krist/Data/scene1-model-control-20261110/<新回合>/model-decisions
  echo '{"op":"stop","session_id":"从status取得的当前会话ID"}' | python3 model_control/host_client.py \
    --run-dir /media/krist/Data/scene1-model-control-20261110/<新回合>/model-decisions
host_client默认容器为kuavo-ros-noetic-s12-ready，可用--container和--root覆盖。
--summary仅缩短终端输出，每次调用以独立UUID保存完整请求、响应和往返时间，
同时更新latest.json；轮询不会覆盖同一命令的先前结果。
容器内也可直接把JSON传给python3 model_control/client.py。

接口与单位
observe：返回observation_id、RGB图片、深度NPZ、相机标定与外参、关节/夹爪状态、
左右末端位姿、头部角度、视觉候选物/支撑面/箱沿和持物描述。pixel可查询有效
深度反投影到base_link的点；无有效深度则报错，不猜测位置。

move_ee：必填op、command_id、observation_id、side（left/right）、position_m
（三元数组）、quaternion_xyzw（单位四元数）；可填context、target_index、
surface_index、timeout_s。坐标系base_link，位置米，旋转弧度；TCP沿用原
UrdfArmKinematics定义。单次位置目标不超过当前位置65cm，逐步执行仍受原关节
增量与跟踪边界限制。这是分段闭环原语，不能自动生成整套任务。

set_gripper：必填op、command_id、observation_id、side、value；value是归一化
开合命令0..1，不是毫米或原始0..255。当前使用open<0.15、close>0.85，拒绝
模糊中间目标；SDK再将其转换为rq2f85命令。闭合需grasp或handover上下文，
grasp还需当前观测target_index。闭合确认由实测受阻开度、稳定状态和视觉帧
共同判断；保持命令不会因受阻实测开度而自动松开。

context：free、grasp、transport、place、bin或handover。transport要求已确认
持物几何；place要求完整可观测支撑面，首次通过surface_index选定；bin要求
已观察到箱沿；handover要求另一手的持物与接触证据。上下文不代替几何检查。
target_index/surface_index都属于所引用observation_id的候选列表，不跨观测复用。

set_head：必填op、command_id、angles_rad=[yaw,pitch]。桥与SDK均用弧度，
限位yaw±70°、pitch±25°。返回commanded仅表示已发送，随后重新observe确认。

move_ee/set_gripper的timeout_s默认30，范围(0,60]秒。超过60秒的引用观测或
关节较引用观测偏移超过0.03rad时拒绝执行；执行中持续读取新的RGB-D和反馈。
命令串行，忙时返回busy。host_client会为缺少command_id的动作生成UUID；
网络结果不明时重试必须显式使用原ID和原参数，不能再次自动生成新ID。
同一进程内重复ID返回原结果，改参数重用ID被拒绝。

停止与错误行为
stop先设置取消标志，阻止动作适配器继续积分/提交，再由唯一发送线程保持
最后已通过检查的有界关节参考和夹爪命令，等待该参考再次发布确认。不会用
实测关节覆盖参考，因为这会取消原稳态跟踪/重力补偿并导致末端下沉。
停止不会自动张爪、回零、结束录像或关闭仿真；机械臂仍可能向最后参考收敛。
正常情况下发送线程10Hz检查；停止等待一次保持发布，内部确认等待预算2秒。
这不是硬实时急停承诺：SDK调用/线程阻塞会影响响应。传感器过期、ROS时间
回退/停滞、SDK发送错误会锁存故障并停止继续发新命令；底层已有参考可能仍
存在。需检查现场状态和故障原因，不能把报错当作机器人已经物理静止。
动作失败/取消/超时停止推进新参考并保持最后已检查参考；正常reached同样
保留最后到达参考。这是参考保持，不是瞬时制动或物理静止保证。

真实限制
检查包括部署/URDF限位、原关节步长与跟踪补偿边界、目标10mm漂移限制、
开爪夹指扫掠、桌面与边界、持物包围盒、其他可见包裹和箱沿余量、支撑释放
间隙/倾斜/足印及接触稳定性。几何来自RGB-D，采用采样检查；不是全机器人
连续碰撞、动力学或任意遮挡环境的安全证明。接触与持物路径仍需真实回合验证。
目标身份缺失时拒绝继续接近；静止闭合仅有受限遮挡窗口。支撑/箱沿/桌边
证据有时效要求，部分支撑面不可用于放置，保持几何不能无限替代新观测。

进程重启不能恢复持物上下文：命令缓存、观测ID、附着几何与接触确认均为
进程内状态。重启后中间开度且未登记的夹爪会被拒绝；不能通过重新observe
就宣称重新确认抓取。需要显式恢复设计或在无持物的初始场景重新开始。
不要在抓着包裹时为了热更新代码直接重启桥。

原速录像与验收
宿主机保持当前DISPLAY/XAUTHORITY，且只有一个MuJoCo窗口：
  python3 model_control/record_video.py \
    --output /media/krist/Data/scene1-model-control-20261110/<新回合>/full.mp4
默认15fps，X11窗口抓取、h264_nvenc编码，按实际墙钟记录，不加速播放。
SIGINT或SIGTERM发送给record_video.py进程后，它会让ffmpeg正常封装MP4，并
保存同名.json收据（实际时长、帧流、sha256）及.log。强杀进程可能留下未封装
视频。录像只读取显示像素，不给模型提供仿真物体真值；收据task_success为null。

本轮证据目录：
/media/krist/Data/scene1-model-control-20261110/model-seed0-20261110
内含runtime.json、bridge-run*/events.jsonl、model-decisions/观测与请求结果、
桥日志及录像。当前既有four-parcel-full.mp4文件名不代表已经完成四包。
失败/拒绝动作保留，计时包括模型等待。最终需分开报告抓取、称重支撑、稳定
入箱的观测证据与官方裁判结果。未取得实际成功证据前，不标为成功视频。

异步接口（已完成原语实回合验证，完整任务仍待验收）
动作可增加 "async":true，立即返回accepted；accepted不代表动作成功。
{"op":"result","command_id":"原ID"} 查询running/终态/unknown。
仍只允许一个动作，不排队；busy后必须检查原命令结果，不能盲目换ID重发。
{"op":"events","after":0} 获取最近512条command/result事件和cursor，
下次以cursor作为after。seq包含控制采样，允许不连续；oldest_seq用于识别
历史窗口边界，完整证据仍以events.jsonl为准。执行中observe仍返回busy。
终态附execution_s（执行与停止保持时间，不含随后观测保存时间）。
关闭时等待异步动作退出后再关闭运行资源。模型继续使用简短JSON动作。

延迟实验待办：在相同seed、配置、模型low设置下记录模型请求开始/结束、
动作提交/终态及任务完成墙钟，并录制原速视频。分别统计模型推理、通信、
控制执行、观测与空闲等待；观测至动作间隔不能直接称为模型推理延迟。
比较零附加等待和固定附加等待的完整回合，报告成功率与总耗时差；
未成功回合单独报告截止时间与失败原因，不混入成功耗时。
Full Access后已取得实际回合和录像，自体过滤已接入运行桥。
仍无模型推理延迟对完整成功任务耗时的实测因果结论，不把开发等待算成推理。

2026-11-10 Full Access 后续实际运行：
官方 seed 0 已重启；异步动作 accepted/running/timeout 与并行查询已实际观察。
自体过滤已接入 fresh_snapshot；初始四个候选保留、删除零点，尚未验证
夹爪进入视野后的过滤有效性。15cm右臂抬升30.760s超时，实际约3.3cm；
43步平均间隔0.715s，动作提交往返0.107s。保持线程报告10Hz且无故障。
该回合失败，不代表一包任务通过；无模型推理延迟因果结论。
证据：model_control/results/async-seed0-20261110/metrics.json、bridge-events.jsonl；
原速视频：/media/krist/Data/scene1-model-control-20261110/async-seed0-20261110/full.mp4。
结束时发出stop并保持参考，仿真与桥仍运行，无持物。
host_client新增--socket、请求往返计时及每次调用独立UUID归档，避免轮询覆盖。

后续耗时分解：3cm抬升旧实现23.543s reached；省去零剔除时重复分割后
另一初始姿态3cm抬升15.689s reached，不能作为同起点因果比较。
bridge step新增timing_s：snapshot包含rgbd与self_filter，adapter包括几何检查，
publish为等待唯一发送线程确认。详见results/async-profile-20261110/metrics.json。
当前桥socket=/tmp/scene1-profile.sock；空手、stop后保持参考，仿真继续。
一包任务与模型延迟实验尚未完成，不将原语reached认定为任务成功。

空间预筛选候选（2026-11-10）：self_geometry先建立当前采样点KD树，
用含2mm边界的网格包围球预筛选，再执行原局部包围盒/网格距离检查。
同一真实零剔除观测5次中位数110ms→32ms，候选和过滤记录一致；
非零剔除路径仍需实回合验证。实际3cm抬升11.838s reached，误差10.27mm，
30步；不同初始姿态不得当作同条件任务提速。
运行中重复ID返回running，参数冲突拒绝，本次仅一个command事件。
证据results/async-spatial-20261110，原速录像位于NVMe对应目录full.mp4。
当前socket=/tmp/scene1-spatial.sock，空手stop保持，仿真继续。
一包任务和模型延迟对完整任务耗时的影响仍未取得成功回合证据。

协议v2（替代上文动作结果内嵌观测的说明）
先status或observe取得session_id。其他所有操作均必须显式携带该ID，
包括动作、stop、result、events、command_observation。缺失或旧会话被拒绝；
客户端不得自动把旧请求换绑新会话。观测ID仍只接受当前桥保存的最新观测。
例：{"op":"result","session_id":"当前会话","command_id":"原命令"}

命令接收、取消标志清除、stop及关闭由同一admission锁串行化。stop确认
之前已接收的动作不能清除该取消标志；stop之后新接收的动作允许重新开始。
stop是当前动作的取消及参考保持，不是锁存急停。重复ID不会开启新动作。

动作终态不再包含observation；execution_s不含拍照保存时间。终态中的
observation_status=pending是发布时状态，不会被后续查询修改。
用{"op":"command_observation","session_id":"当前会话","command_id":"原命令"}
获取pending、ready（内含observation）、failed（含error）或unknown。
事件顺序为command→result→observation_ready/observation_failed，后两者
仅含命令ID和观测ID或错误，不携带图像/完整观测。
async=true立即返回accepted；省略或false等待动作终态，但不等观测保存。
保存期间继续占用单动作锁，新动作与observe返回busy；stop/status/result
和事件查询保持可用。拿到终态不等于新观测就绪，更不等于任务完成。

释放事务（v5候选，需完整回合验证）
张爪前保持原支撑/箱沿/交接、间隙、倾斜、足印和稳定帧要求，检查当前
持包扫掠后记录授权证据。之后包裹可以接触支撑或离手，不再按旧刚性附着
计算随腕扫掠；这阶段只能保持授权时的双臂参考、单调张开当前夹爪。
关节限位、跟踪补偿界、步长及新鲜传感器检查仍有效。停止/失败/超时保留
观测中的releases状态，只允许同手、同context的set_gripper开爪继续；禁止
新末端运动、闭合或替换目标/支撑。实测开度和参考均满足完成条件后才清除
持物状态。进程重启仍不恢复此事务，不能用重启跳过部分释放。

桌边跟踪（v5空手阶段更新，仍需实际动作验证）
锚定至少三条具有局部直线方向证据的桌边，必须包含靠近机器人的一边。
碰撞检查使用这些边的内侧半空间，未观测侧按桌面继续延伸处理，不以裁剪
端点构造可通行边界。之后复用table_motion的非平行深度边配准，从原始
锚点更新参考，不逐帧累积漂移；原配准内点、残差、3cm/2°门槛保持不变。
v5/v6支撑刷新要求与选定时桌框角点差≤1cm、平面差≤5mm；v6实跑中
接触姿态变化反复触发失效，未完成称重释放。v7改为与上次通过验证的
支撑参考比较，仍保持每次≤1cm、≤5mm；支撑几何始终从最初选定观测
通过当前相对变换重算，桌边配准总界仍由原始桌框锚定。超界不推进支撑
参考、不刷新年龄，必须重新取得合格证据或重新选取完整支撑面。观测附table_tracking和桌边/支撑证据年龄，不能将旧观测当新帧。

有界观测等待（v6候选）
持物组件碰撞歧义、桌边证据过期、目标身份暂不可见时，适配器回滚本步
未通过检查的伺服提案，只保持最后已检查参考，并读取下一帧重新检查。
持续阻塞5秒即失败；总动作timeout仍包含全部等待。限位、跟踪补偿越界、
实际桌面/箱壁扫掠、传感器故障等仍立即失败，不转为自动重试。
每次继续都必须重新通过全部原检查，没有放宽关联阈值或碰撞余量。
result在running时提供phase；events增加progress，仅在executing与
waiting_for_clear_observation切换时发送简短原因。详细失败几何留在step
日志中，终态仍先于最终观测保存发布。stop可随时取消等待并保持参考。

事件等待客户端（在桥所在容器内）
  python3 model_control/watch.py --socket /tmp/scene1-model.sock \
    --session 当前会话 --command 当前命令 --output 新建的回执.jsonl
只在开始、进度切换或终态时输出JSON行；本地轮询不调用模型，不发动作，
不自动取消命令，也不重绑会话。等待超时不等于动作已停止。终态可先返回
observation_status=pending；下一动作仍需遵守单动作锁和新鲜观测检查。
watch_received_wall_s记录桥到容器内等待客户端的交付时刻，不包含模型
推理或工具结果送达模型的时间。--output使用独占新建，防止覆盖旧回执。

桌面绑定目标（v7候选，需完整回合验收）
move_ee可选reference="table"，默认"base"保持原行为。position_m和
quaternion_xyzw仍是绑定observation_id时的base_link位姿；桥将其转换为
不可变桌面参考目标，每个新帧用经验证的桌边变换重投影到base_link。
不是物体真值坐标，也不运行任务状态机。桌边过期或目标参考单次变化
超过10mm XY/5mm高度时进入有界观测保持；5秒持续不恢复则失败。
全部关节、补偿、步长、扫掠、到达与释放检查照常执行。
observe现在刷新持物视觉证据；同一RGB时间戳最多融合一次，释放授权后
不再更新旧刚性附着。图像新鲜不再掩盖旧payload视觉证据。

本次结束：v7按用户明确指令停止并直接开手，非标准place检查成功；未入箱。
详见results/one-parcel-v7-20261110/REPORT.txt、metrics.json及最终画面。
