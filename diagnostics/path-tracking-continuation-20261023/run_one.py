"""Serial run, analysis and byte-verified archival; never retry a failed run."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('version', choices=['baseline','candidate','velocity','velocity_pd','mpc_q','proximal_kp','origin_pd','wrist_damping','arm_weight','pd_scale','pd_scale15','origin_mpc_q'])
    ap.add_argument('label')
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--parcels',type=int,choices=[1,2,4],required=True)
    ap.add_argument('--policy',choices=['axis_sweep','axis_feasible','executor','executor_geometry','executor_spatial','executor_release','executor_physical','executor_context','executor_hold','executor_open','executor_project','executor_obb','executor_fresh','executor_fast','executor_mpc_hold','executor_numerics','executor_prepare','executor_prepare_adapt','executor_grasp_frame','executor_grasp_gain','executor_grasp_settle','executor_support_heading','executor_support_gain','executor_pending_hold','executor_first_gain','executor_arm_prepare','executor_soft_open','executor_support_prepare','executor_insert_pitch','executor_support_recover','executor_placement_physical','executor_release_quiet','executor_first_transport','executor_first_pitch','executor_first_pickup','executor_first_select','executor_pickup_sweep','executor_first_clear','executor_first_reobserve','executor_pickup_hold_sweep','executor_first_bin','executor_bin_components','executor_bin_dense','executor_table_dense','executor_pad_dense'],default='axis_sweep')
    a = ap.parse_args()
    receipt = dict(label=a.label,version=a.version,seed=a.seed,parcels=a.parcels,policy=a.policy,stages={})
    destination = HERE/(a.label+'-pipeline.json')
    if destination.exists() or (HERE/a.label).exists():
        raise RuntimeError('Label already used; no implicit retries')

    def run(stage, script, arguments):
        receipt['current_stage']=stage
        destination.write_text(json.dumps(receipt,indent=2)+'\n')
        with (HERE/(a.label+'-'+stage+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,str(HERE/script),*arguments],stdout=log,stderr=subprocess.STDOUT)
        receipt['stages'][stage]=dict(returncode=result.returncode,
                                     ended_at=datetime.datetime.now().astimezone().isoformat())
        destination.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(label=a.label,stage=stage,returncode=result.returncode)),flush=True)
        return result.returncode

    if run('switch','switch_wrist_pd.py',[a.version]):
        raise SystemExit(1)
    trial_rc=run('driver','full_experiment.py',[a.version,a.label,'--policy',a.policy,
                    '--seed',str(a.seed),'--parcels',str(a.parcels),
                    '--max-steps',str({1:5000,2:9000,4:16800}[a.parcels])])
    d=HERE/a.label
    if (d/'output/steps.jsonl').is_file() and (d/'chain.jsonl').is_file():
        run('analysis','analyze_single.py',[a.label])
        run('audit','audit_run.py',[a.label])
    if (d/'experiment.json').is_file():
        run('archive','archive_completed.py',[a.label])
    run('summary','summarize_experiments.py',[])
    receipt['current_stage']='done'
    destination.write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(trial_rc or any(x['returncode'] for x in receipt['stages'].values()))


if __name__ == '__main__':
    main()
