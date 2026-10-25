"""Generate the final evidence report from retained outcomes, without reruns."""
import datetime
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def number(value, digits=4):
    return 'null' if value is None else f'{value:.{digits}f}'


def main():
    comparison = read(HERE / 'comparison.json')
    verification = read(HERE / 'final-verification.json')
    assert verification['errors'] == []
    assert all(r['status'] not in ['running', 'initializing'] for r in comparison['tasks'])
    task = {r['label']: r for r in comparison['tasks']}
    original = task['feasible-original-p4-s0']['per_parcel']['1']
    candidate = task['feasible-c-p4-s0']['per_parcel']['1']
    assert not task['feasible-c-p4-s0']['four_parcel_visual_complete']
    lines = [
        'scene1 路径修正与 MPC/WBC 跟踪研究：实施、验证及回退报告',
        '生成时间：' + datetime.datetime.now().astimezone().isoformat(),
        '',
        '结论',
        '本轮实现并验证了轴向退让、完整待执行参考/限速后命令扫掠检查，以及独立的可达性求解修正；完成5次空手、6次持物/任务独立官方复位试验。',
        'C + 最终路径通过两包，完整四包在第三包退让预检失败；同路径原控制器在第二包退让预检失败。完整任务可靠性和提速目标尚未达成。',
        '按预先门槛拒绝生产替换，保留研究补丁和全部失败；恢复原控制库及动态QR。开发 seed 0 未通过，因此未启动 seed 1、2，未用留出种子调参。',
        '所有任务回合的 task_success 均为 null，均不能称为官方成功。视觉完成数是保留的抓取、称重、交接、入箱检查结果。',
        '本报告中的“原控制器”也使用同一研究路径，不能误读为原生产上层的全任务性能。未取得完整成功配对，不计算完整任务提速。',
        '',
        '1. 路径实现及边界',
        '父版本：已归档 candidate-policy-reobserve。生产上层和 v63 未覆盖。',
        'axis.patch：用释放后实测右工具 +Z 轴的水平投影生成60mm水平退让，保留70mm抬升。',
        '投影范数不足1e-3、当前RGB-D观察不唯一/无效、方向不远离包裹、目标未通过原关节/纠偏边界可达性预检时拒绝动作。方向不自动翻转。',
        '原位置到位容差14mm、原30度纠偏预算、关节和力矩边界、阶段步数/超时、包络、3mm碰撞余量及20步扫掠阻断停止保留。',
        '特别说明：8度姿态容差是本次轴向候选新增的保守预检条件，不是原生产执行控制器的硬限制。其必要性和过度保守性仍是限制，不能把预检失败称为原任务物理不可达。',
        '可达性求解只认证笛卡尔目标，不把求得的关节解发送，也不重置原参考/增加预算。执行仍使用原逐步控制器。',
        'axis-sweep.patch：在原增量检查之外，补查实测姿态至完整待执行参考，以及 MotionFilter 后实际拟发送命令；后者不通过时在 env.step 前拒绝。',
        '扫掠采样保留端点/中点及最大0.012rad关节采样间隔，作用于已有再抓阶段；这是采样内侧手指几何检查，不是全机器人连续碰撞证明。',
        '先诊断后修改的证据：历史 release-hold 的261行中有12行旧增量检查通过而实际限速命令扫掠被阻断；reobserve 的289行没有此遗漏。',
        '七例离线几何均支持方向，但六例缺少原释放参考，仅是方向依据；只有记录完整的一例支持原预算下可达性回放。',
        'preflight-feasibility.patch：原位置/姿态加权最小二乘可能误拒绝，增加同边界显式约束姿态的SLSQP回退，并最终独立核查所有条件。',
        '真实E失败快照：原求解15.408mm/4.590度被拒；相同边界内存在12.518mm/5.986度解，新增回退复现8.218mm/7.999943度。没有放宽验收条件。',
        '最终候选：candidate-policy-axis-feasible；清单SHA256 a32fefc8cd398e7a544495f0b2a57483c0c3779dcd6632cf50e049d6d007dd37。',
        '390项路径/原检查测试通过；原/候选路径补丁分别保存，未做方向与扫掠改动的完整动态因子消融，不单独归因两包收益。',
        '',
        '2. 控制链定位与候选',
        '正常 direct_to_wbc=false 链：上层发布 → 带时间手臂轨迹 → 简化MPC八个近端关节 → WBC十四个手臂关节 → 力矩。未切换直接控制分支。',
        '运行时配置 armAccel 的 Kp=80、Kd=10.2、accArm权重9；未重复此前已表现不佳的高增益组合。',
        '原代码近端位置/速度来自MPC；六个腕部位置来自带时间轨迹，但正常分支腕部速度保持零。位置/速度仍分别经过现有10Hz滤波。',
        '分析按MPC相对时钟计算轨迹当前值，未把未来端点当作当前目标。未带时间戳的向量按回调到达近似对齐，不能宣称精确执行器延迟。',
        'C：WBC力矩输出后的腕部有界补偿，Kp=1 Nm/rad，测量速度阻尼0.05 Nm/(rad/s)，附加±0.5Nm，总力矩±14.1Nm。无新重力项、积分或近端直接补偿。',
        '未取得QP内部源码；C不称作WBC内部优化，保留为研究对照。',
        'D：只修正常分支腕部速度参考，按传感器时间差分已过滤的位置，使用已有 arm_interpolation_max_velocity=1rad/s 上限；不改位置、近端、增益、权重。',
        'D修复了移动腕部仍为零速度的现象，位置/速度不一致RMS约0.0496→0.00552rad/s，但单独使用时静态腕部偏差仍大。',
        'D+C：在C上只加入D。同空手动作中到位时间和超调退化，未进入接触验证。',
        'E：证据显示C称重时近端 timed-arm→MPC RMS约0.0455rad，而MPC→WBC约0.0020rad，因此单独测试已启用动态QR中的近端位置代价。',
        'E保持C库，只将 dynamic_qr.info 的 Q_base_stance 八个近端位置对角项200→400；R、其余模式、腿/腰/基座及WBC参数不变。',
        'E空手近端偏差改善，但持物第一包近端WBC RMS为0.005807rad，高于原C对应0.005105rad；称重阶段也退化，不保留。',
        '近端MPC当前状态序列与输入速度的差异还受滚动重规划影响，不能只凭跨周期差分不一致直接把MPC输入换成差分速度。',
        '',
        '3. 五次固定空手动作（每个候选一次新官方 seed 0 复位）',
        '每次35秒：相同温和进入动作，近端伸出/返回和腕部伸出/返回，各2秒斜坡+5秒保持。无抓取指令。所有结果均保留。',
        '指标依次为：上层命令动态RMSE(rad) / 末段平均绝对静态偏差(rad) / 进入1度误差带的墙钟秒 / 超调(rad)。null表示观察窗内未到位，不是0。',
    ]
    empty_order = ['original-empty-s0', 'c-empty-s0', 'd-empty-s0', 'dc-empty-s0', 'e-empty-s0']
    for r in sorted(comparison['empty'], key=lambda row: empty_order.index(row['label'])):
        lines.append('  ' + r['label'] + ': ' + r['status'])
        for s in r['segments']:
            lines.append('    ' + s['name'] + ': ' + ' / '.join(number(s[k], 5) for k in
                         ['ramp_command_rmse_rad', 'static_abs_bias_mean_rad', 'settle_within_1deg_s', 'overshoot_rad']))
    lines += [
        'C腕部伸出/返回到位约0.515/0.312秒；D+C约1.366/2.595秒，超调0.01480/0.00959→0.02195/0.01779rad，故拒绝D+C。',
        'C腕部动态延迟估计约0.22–0.24仿真秒；原控制器估计触及400ms搜索边界，结果不确定，不把400ms当成精确延迟。',
        '静态偏差、速度P95、静态波动及逐关节值见 empty-metrics.csv、各回合 metrics.json/reference-analysis.json。低频采样不能证明不存在高频振荡。',
        '',
        '4. 全部六次持物/任务试验（每次独立官方复位、seed 0）',
        '格式：标签 | 请求包数 | 视觉完成包数 | 已发送步数 | 控制墙钟秒 | 官方 task_success | 结果',
    ]
    for r in comparison['tasks']:
        lines.append('  ' + ' | '.join([r['label'], str(r['requested_parcels']), str(r['visual_confirmed_parcels']),
                      str(r['steps']), number(r['policy_duration_wall_s'], 3),
                      json.dumps(r['official_task_success']), r['error'] or '完成请求的视觉检查']))
    lines += [
        'axis-sweep-pd-p2-s0 与 feasible-c-p2-s0 是不同已登记路径版本；E失败后先用真实快照确认求解缺陷，再建立新候选，不是原条件反复重跑筛成功。',
        '最终路径两包：5429步，803.392秒；第二包释放、退让、初始调平和后续称重/交接/入箱链通过。',
        '两包回合未触发新求解回退；其回退由真实快照回放、回归测试以及随后四包中的实际拒绝分支覆盖。',
        'C四包第三包失败：位置14.900246mm、姿态7.999943度，未满足14mm条件；原控制器四包第二包失败：位置50.164410mm、姿态7.999943度。拒绝目标均未发送。',
        'multistart-audit.json：每个失败快照16个固定初值、同一目标/边界/容差，均0个认证解；只是数值复核，不是全局不可达证明。',
        '原控制器第二包加权解位置12.033mm但姿态13.318度；拒绝受候选新增姿态预检影响，应与原执行约束区别解释。',
        '',
        '5. 同冻结路径四包回合中的第一包对照',
        '两回合均完成第一包，但整回合失败；以下仅是相应阶段描述，不能外推整任务提速或成功率。',
        '指标 | 原控制器 | C',
    ]
    for title, key in [('步数', 'steps'), ('墙钟秒', 'elapsed_s'), ('近端WBC跟踪RMS(rad)', 'proximal_wbc_rmse_rad'),
                       ('腕部WBC跟踪RMS(rad)', 'wrist_wbc_rmse_rad')]:
        lines.append(f'  {title} | {number(original[key], 6)} | {number(candidate[key], 6)}')
    lines.append('  WBC参考对实测关节FK末端距离均值(mm) | ' + number(original['wbc_fk_error_m']['mean']*1000, 3) +
                 ' | ' + number(candidate['wbc_fk_error_m']['mean']*1000, 3))
    lines.append('  腕部RMS下降' + number((1-candidate['wrist_wbc_rmse_rad']/original['wrist_wbc_rmse_rad'])*100, 1) +
                 '%，近端RMS增加' + number((candidate['proximal_wbc_rmse_rad']/original['proximal_wbc_rmse_rad']-1)*100, 1) +
                 '%；近端退化构成独立验收失败。')
    lines += [
        'FK误差由同一URDF和关节参考/反馈计算，不能当作独立光学TCP测量。完整逐关节误差、上层/定时轨迹/MPC/WBC分解在 analysis-full/<标签>/。',
        '',
        '6. 纠偏、等待、约束和阶段耗时',
        '下列“冻结策略步数”包含视觉控制接管ACT的有效动作，不能解释为全部空等；显式hold和前进门控另列。',
        '外层residual变化量混合正常目标变化与纠偏，不是纯纠偏行程，不能单独当作纠偏收益。',
    ]
    for r in comparison['tasks']:
        m = read(HERE / 'analysis-full' / r['label'] / 'metrics-full.json')
        phases = m['phases'].values()
        bounds = r['joint_bound_audit']
        lines.append('  ' + r['label'] + ': 冻结策略=' + str(sum(x.get('frozen_policy_steps', 0) for x in phases)) +
                     '，显式hold=' + str(sum(x.get('explicit_hold_steps', 0) for x in phases)) +
                     '，前进门控hold=' + str(r.get('regrasp_forward_gate_hold_steps')) +
                     '，目标裁剪=' + str(r.get('target_clips')) +
                     '，扫掠阻断=' + str(r.get('regrasp_sweep_block_steps')) +
                     '，速度限速步=' + str(r.get('velocity_limited_steps')) +
                     '，residual变化(rad)=' + number(sum(x.get('within_phase_outer_residual_variation_rad', 0) for x in phases), 4) +
                     '，反馈最大越界(rad)=' + number(max(bounds['max_excess_rad']), 6) +
                     '，超过0.0005rad最长可见秒=' + number(max(bounds['longest_observed_over_threshold_s']), 4))
        if (r.get('pd_limits_50hz') or {}).get('formal_policy', {}).get('available'):
            pd = r['pd_limits_50hz']
            lines.append('    C附加力矩裁剪50Hz采样：初始化=' + str(pd['initialization']['raw_clipped_samples']) +
                         '，正式策略=' + str(pd['formal_policy']['raw_clipped_samples']) +
                         '；初始化最长可见秒=' + number(pd['initialization']['longest_observed_clip_s'], 4))
    lines += [
        '反馈有短时越URDF边界的记录，不能声称实际关节从未越界；命令/物理边界未放宽。不同已执行包数的最大值不能直接作等轨迹优劣比较。',
        '运行中保留原保护，并监测持续力矩裁剪、倾斜、参考裁剪和原保护停止；未触发这些停止，不等于内部QP约束从未活跃。',
        'phase-metrics.csv 包含全部阶段墙钟/仿真耗时、关节/末端误差、等待/限速/纠偏变化量。每包阶段及静态偏差在 metrics-full.json；无有效静态平台则为null。',
        '',
        '7. 生效证据、工件和恢复',
        '每次回合记录独立官方 run_id，启动时核对上下层/配置哈希，并核对控制进程实际映射库；runtime.json、experiment.json、candidate-hash-check.json 是入口。',
        '较早回合未单独记录动态QR文件哈希，E及以后的回合才包含该项；不能将早期原配置推断追补为同等运行时证据。',
        '原控制库 SHA256: e46e9a800011f8c5bf43e269cb2ba0ab7d543c27e09ac10deaea6ae06e1179e4',
        'C控制库 SHA256: ab20325136810cc785bfdb4abd80c3ecc2b2e73627ebebea42b904bd7bb3f0ba',
        'D控制库 SHA256: e2603ec0a306e8433eab849a8deb36fd132c9824650b5ec2c9ea16c6338ebae7',
        'D+C控制库 SHA256: dafa16a8946e90af6727ffe6937f0075955fbb74f217351b365049f29dff0634',
        '原动态QR SHA256: 225d217d47741f088830d0a7c038e8a76c465d9948f4a56f194b3524f58677de',
        'E动态QR SHA256: af1b24419d8b65e5c1d086e7efadf97bb1bd66053a6633e17f283621347f4e08',
        '路径补丁：axis.patch、axis-sweep.patch、preflight-feasibility.patch；对应候选目录及清单分别保留。',
        '控制补丁：control-c.patch、candidate-velocity/change.patch、candidate-velocity-pd/change.patch、mpc-arm-q.patch。后两种速度补丁分别相对原库/C，不混称一种独立收益。',
        '构建记录：各候选 build.py/build-manifest.json；速度差分C++边界测试通过，参考时间对齐2项回归测试通过。',
        '原始回合/大库：/media/krist/KINGSTON/scene1-path-tracking-20261023；本目录保留符号链接。磁盘必须挂载。',
        '每次任务的失败、日志、原始链、输出图像/JSON及运行时证据均SHA核对归档；只删除容器中的已验证重复副本。',
        'SAM2：专属源码、权重、分发包及容器editable安装已删除，研究与kls_dev环境 find_spec 均为空；旧结果/脚本作为历史证据保留。没有新增YOLO或其他视觉模型/训练。',
        '最终现场核验时间：' + verification['at'] + '；errors=[]。',
        '生产宿主/容器/快照各124文件、v63备份115文件一致；官方git HEAD与工作区、原VR、原控制库及原dynamic QR一致；两个容器无控制/策略工作进程。',
        '构建、同步、官方复位、日志查看、回放和恢复命令见 RUNBOOK.txt。恢复必须用本目录 switch_wrist_pd.py，旧脚本不处理动态QR。',
        '',
        '8. 验收决定与未完成的性能目标',
        '路径两包门槛通过，完整四包门槛未通过；C近端退化，D/DC/E均不满足相应阶段验收。保留研究候选，生产维持原版。',
        '没有完整成功配对、留出种子结果或官方成功判定，因此不存在已验证的全任务可靠性/速度收益。',
        '仍需解决后续包释放姿态与原纠偏可达范围的矛盾，以及C带来的近端跟踪退化；新增8度预检是否过度保守也应独立审视，不能通过隐藏或直接放宽阈值宣布成功。',
        '本轮按计划的失败保留/回退分支交付。所有失败均保留，不继续相同条件重跑筛选，不把较短失败耗时记作收益。',
    ]
    (HERE / 'REPORT.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(HERE / 'REPORT.txt'), lines=len(lines), empty=len(comparison['empty']), tasks=len(comparison['tasks']))))


if __name__ == '__main__':
    main()
