"""Render the final retained acceptance evidence, including limitations."""
import csv
import datetime
import json
from pathlib import Path

P=Path(__file__).resolve().parent
read=lambda path:json.loads(path.read_text())
d=read(P/'MULTISEED-PAD-RESULTS-20261025.json')
selected=[r for r in d['runs'] if r['version']=='origin_pd']
assert len(selected)==4 and all(r['visual_full4'] for r in selected)
restored=read(P/'final-verification.json')
assert restored['errors']==[]
lines=[]
add=lines.append
add('scene1 路径、控制与多 seed 四包验收报告 — 2026-10-25')
add('生成时间：'+datetime.datetime.now().astimezone().isoformat())
add('')
add('本次冻结候选 executor_pad_dense + A+C 在 seed 0、2、4、5 均完成四包视觉验收；其中 0、2 为开发/回归，4、5 为冻结后未参与调参的新 seed，各运行一次。对应原控制器四轮均失败，失败完整保留。生产与 v63 未替换，运行库/VR/task/QR 已恢复原版并核验。')
add('这不等于官方成功：所有 task_success=null。原环境没有提供 scene1 官方成功裁决；本报告采用独立的原抓取、接触、称重、交接、入箱和回位检查。四个 seed 的有限样本也不是长期可靠率保证。')
add('任务完成性改善已得到本次验收支持；整任务提速未证明，近端跟踪存在退化，腕部静差和动态延迟仍存在。因此不建议据此自动替换生产，也不宣称控制性能已全面优化。')
add('')
add('逐条件结果（相同冻结路径；耗时为正式 policy wall time，不含复位、初始化、分析归档）：')
add('seed | 控制链 | 完成包数 | 步数 | 耗时 s | 结果')
for r in d['runs']:
    result='四包视觉通过' if r['visual_full4'] else r['failure']
    add(f"{r['seed']} | {'原版' if r['version']=='baseline' else 'A+C'} | {r['parcels']}/4 | {r['steps']} | {r['duration_s']:.3f} | {result}")
add('原版与组合四组均为相同 seed/包数/路径，但没有一组原版也完成四包，所有 full_task_timing_eligible=false。失败回合的短耗时不计收益。')
add('')
add('路径修正与本轮新增证据：')
add('1. 保留释放后实测工具轴水平投影向远离包裹方向退让60mm、抬升70mm；先检查投影、方向、可达与完整扫掠。保留物理关节、力矩、限速、包裹包络、3mm余量及原超时。移除仅为接近旧ACT动作而设置的额外邻域限制。')
add('2. 扫掠覆盖未跟踪完的参考差、限速后实际拟发送命令和发送前新鲜反馈；修正PCA尺寸/旋转与世界轴中心混用。首包主要流程已转为现有RGB-D视觉执行，保留必要初始化；未增加SAM2、YOLO或训练任务。')
add('3. 原bin-dense冻结版新seed2在第二包接近时停止。真实RGB-D回放显示四边支持充分，stride3凸包使短边假缩短8.38mm，越过原8mm尺寸门槛；stride1同帧仅变化0.98mm，原门槛全部通过。独立table-dense补丁仅对参考锚定/跟踪密集采样，未改变目标选择采样和门槛。seed2两包6349步通过，525测试、42层补丁重建通过。')
add('4. 原bin-dense冻结版新seed3第二包observe_pad超时。原图确有手和包裹部分遮挡：stride4只有59点、39.31%可见面积，未达到60点/40%；stride2为230点/43.64%，同时满足原两条非平行边要求。独立pad-dense补丁仅在observe_pad对同帧表面重采样stride2，保持原完整托盘边界。seed3两包6126步通过；最终529测试、43层补丁重建通过。')
add('5. 旧冻结seed2/3失败没有删除，也没有作为新seed成功重跑筛选。诊断后将0–3列为已用研究seed，重新冻结后才运行4、5。')
add('')
add('控制修正与边界：')
add('A修正正常direct_to_wbc=false控制链的首参考点，从错位segment(24,14)改为tail(14)。C是WBC输出后的有界腕部补偿，不是QP内部修改：六腕关节 clip(1.0*(qWBC-qactual)-0.05*qdotactual, ±0.5Nm)，总力矩±14.1Nm。原task、dynamic QR、原任务权重和增益保持；没有叠加重力补偿、积分或切换直接控制分支。QP内部实现仍为预编译库，不能声称已优化内部QP。')
add('高增益、增大阻尼、任务权重和MPC近端代价等历史候选均保留消融和失败记录；I/J补偿放大因初始化持续裁剪拒绝，K近端MPC Q翻倍空手结果混合且持物失败，均未用于最终候选。空手相同动作证据见extended-empty-comparison.json、extended-ac-empty-comparison.json及早期mpc-wbc-tracking-20261022/comparison.json。')
add('')
add('同包同阶段比较（首包释放准备；WBC参考→反馈RMS，rad；两侧终止条件不同，阶段耗时不是完整任务提速）：')
add('seed | 原版近端→A+C近端 | 原版腕部→A+C腕部')
for seed,c in d['comparisons'].items():
    phase=c['per_parcel_phase_comparisons']['1']['phases'].get('preparing_release_reference')
    if phase:
        prox,wrist=phase['proximal_wbc_rmse_rad'],phase['wrist_wbc_rmse_rad']
        add(f"{seed} | {prox['reference']:.6f}→{prox['candidate']:.6f} | {wrist['reference']:.6f}→{wrist['candidate']:.6f}")
    else:add(seed+' | 原版在抓取接近已停止，不能比较释放阶段 | 不适用')
add('腕部误差明显下降，但近端在这些共同阶段增加约0.0016–0.0031rad，不应只用腕部平均值验收。全程RMS因原版提前失败、阶段分布不同，只作描述，不直接归因。')
add('')
add('A+C完整回合指标：')
add('seed | 近端/腕部全程RMS rad | 上层目标FK/WBC目标FK平均误差 mm | 准备片段步数 | 纠偏事件数/累计L2 rad')
measurements=[]
for r in selected:
    fk={}
    for field in ['command_fk_error_m','wbc_fk_error_m']:
        vals=[x[field] for x in r['phases'].values() if isinstance(x.get(field),dict) and x[field].get('n')]
        fk[field]=dict(mean_m=sum(x['n']*x['mean'] for x in vals)/sum(x['n'] for x in vals),max_m=max(x['max'] for x in vals))
    adj=read(P/'analysis-full'/r['label']/'release-adjustment-audit.json')
    dense=read(P/'analysis-full'/r['label']/'dense-sampling-audit.json')
    n=sum(x['correction_events'] for x in adj['blocks']); amount=sum(x['summed_l2_correction_rad'] for x in adj['blocks'])
    steps='/'.join(str(x['preparation_steps']) for x in adj['blocks'])
    add(f"{r['seed']} | {r['proximal_wbc_rmse_rad']:.6f}/{r['wrist_wbc_rmse_rad']:.6f} | {fk['command_fk_error_m']['mean_m']*1000:.3f}/{fk['wbc_fk_error_m']['mean_m']*1000:.3f} | {steps} | {n}/{amount:.4f}")
    measurements.append(dict(seed=r['seed'],label=r['label'],fk=fk,adjustments=adj,dense_sampling=dense))
add('FK误差由相同机器人模型下关节参考/反馈计算，包含动态阶段；不是相机直接测得的末端误差。平均按各阶段样本数加权；逐阶段均值、P95、最大值和逐关节数据在JSON/CSV中。')
add('纠偏量是实际记录的release_prepare目标调整增量的L2长度累加，不是普通搬运动作量；准备步数包含纠偏与到位等待。seed4出现5个准备片段，其中1段未释放而进入支撑恢复，不能将该中断片段的最终误差当作释放误差。所有真正释放前的检查仍通过。')
add('')
add('保持偏差、延迟与振荡指标（各组最大关节值）：')
add('seed | 近端/腕部静态偏差绝对值 rad | 近端/腕部保持误差标准差 rad | 腕部动态相关延迟范围 sim s')
for r in selected:
    tr=r['reference']['tracking'];prox,wrist=tr['proximal'],tr['wrist']
    lag=[x['sim_lag_s'] for x in wrist['approximate_dynamic_lag_per_joint'] if x is not None]
    add(f"{r['seed']} | {max(map(abs,prox['static_bias_rad'])):.6f}/{max(map(abs,wrist['static_bias_rad'])):.6f} | {max(prox['static_position_std_rad']):.6f}/{max(wrist['static_position_std_rad']):.6f} | {min(lag):.2f}–{max(lag):.2f}")
add('静态样本要求参考在1秒仿真时间内变化<=0.003rad且实测速度<=0.03rad/s；标准差是保持误差变化，跨多个保持区间，不等于已证明无振荡。动态延迟来自速度相关峰值，是近似值；近端0秒命中搜索边界，不能解释为真实零延迟。腕部参考速度仍存在与位置变化不一致的记录，相关历史D速度候选未改善；未宣称此问题已全部消除。完整阶段未专门生成可辨识阶跃，因此不虚构整任务超调/阶跃到位时间，相关空手阶跃数据保留在原消融。')
add('')
add('限幅、等待与成功检查：')
total_samples=sum(r['pd_limits_50hz']['formal_policy']['samples'] for r in selected)
add(f'四轮正式阶段合计{total_samples}个约50Hz采样点，附加力矩/总力矩裁剪均为0。seed4初始化3个附加裁剪采样，最长约40.784ms，未裁剪原总力矩；其余三轮初始化无裁剪。不能把有限采样等同于全频段无瞬时裁剪。')
for r,m in zip(selected,measurements):
    released=[x for x in r['preparation'] if x['released']]
    assert len(released)==4
    tracking=max(x['final']['tracking_error_rad'] for x in released)
    movement=max(x['final']['settled_movement_rad'] for x in released)
    pad_wait=[x['steps'] for x in m['dense_sampling']['blocks']]
    add(f"seed{r['seed']}: target裁剪{r['target_clips']}，限速{r['velocity_limited_steps']}步；实际释放前最大跟踪{tracking:.6f}rad，最大运动{movement:.7f}rad；托盘观察步数{pad_wait}，每次均以3帧合格结束。")
add('各阶段耗时、逐包阶段耗时、保持/限速/纠偏、目标及WBC末端误差、关节边界与失败原因见final-phase-metrics-20261025.csv、final-parcel-phase-metrics-20261025.csv、MULTISEED-PAD-RESULTS-20261025.json及analysis-full/<label>。')
add('')
add('历史和消融不隐藏：')
add('旧bin-dense：seed0/1四包12679/12802步通过；seed2 3367步1包后桌面参考失败；seed3 4199步1包后托盘观察超时；原版seed0–3均释放准备失败。其配对结果在MULTISEED-RESULTS-20261025.json。2026-10-24 seed1 7630步2包因旧usage剩余20%规则中断，始终不算四包成功。2026-10-25用户明确恢复，本轮授权与旧停止回执分开留档。')
add('新seed0完整任务2072.117s，相比旧bin-dense seed0的1980.484s更慢；路径不同，不是严格控制消融，但不能据此宣称提速。新增视觉采样首先解决可靠性问题。旧简化URDF手指网格与实际四连杆不同，几何模型交集不冒充真实精确碰撞证明；没有启用掩码或缩小包裹包络。')
add('')
add('交付和恢复：')
add('冻结：path-control-freeze-pad-dense-20261025.json；policy manifest SHA256 '+d['freeze']['policy_manifest_sha256'])
add('路径：candidate-policy-executor-pad-dense、executor-table-dense.patch、executor-pad-dense.patch及verify_path_delivery.py中的完整43层链；所有manifest、RGB-D夹具和失败记录保留。')
add('控制：controller_variants.py及mpc-wbc-tracking-20261022/candidate-reference-origin、candidate-wrist-pd，参数与运行时身份分开交付。每轮runtime.json核对实际库/VR/task/QR。')
add('恢复：final-verification.json errors=[]；生产宿主/容器各124文件、v63备份115文件无缺失或差异，官方git工作区干净。原库 '+restored['restored_controller_library_sha256']+'；原VR '+restored['restored_executable_sha256']+'。')
delivery_path=P/'delivery-verification.json'
delivery=read(delivery_path)
add('全量交付校验：'+delivery.get('status','unknown')+'；errors='+json.dumps(delivery.get('errors'))+'；详见delivery-verification.json和delivery-verification-final-20261025.log。')
add('构建、独立同步、官方复位、日志、单次新标签运行与恢复命令见RUNBOOK.txt。原始证据在/media/krist/KINGSTON/scene1-path-tracking-continuation-20261023，工作目录保持可用符号链接。')
add('保留研究候选，生产未晋升。任务结束后无实验继续运行。')
(P/'FINAL_REPORT-20261025.txt').write_text('\n'.join(lines)+'\n')
(P/'FINAL-MEASUREMENTS-20261025.json').write_text(json.dumps(measurements,indent=2)+'\n')
fields=['label','seed','controller','parcel','phase','steps','elapsed_s','elapsed_sim_s','tracking_samples','proximal_wbc_rmse_rad','wrist_wbc_rmse_rad','command_fk_error_m','wbc_fk_error_m','plateau_bias_rad','explicit_hold_steps','bounds_clipped_steps','velocity_limited_steps','within_phase_command_variation_rad','within_phase_outer_residual_variation_rad']
for per_parcel,name in [(False,'final-phase-metrics-20261025.csv'),(True,'final-parcel-phase-metrics-20261025.csv')]:
    with (P/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in d['runs']:
            groups={k:v['phases'] for k,v in r['per_parcel'].items()} if per_parcel else {'all':r['phases']}
            for parcel,phases in groups.items():
                for phase,metrics in phases.items():
                    row=dict(label=r['label'],seed=r['seed'],controller=r['version'],parcel=parcel,phase=phase)
                    row.update({k:(json.dumps(metrics[k]) if isinstance(metrics[k],(list,dict)) else metrics[k]) for k in fields[5:] if k in metrics})
                    writer.writerow(row)
print('Report and phase/parcel tables written; delivery status:',delivery.get('status'))
