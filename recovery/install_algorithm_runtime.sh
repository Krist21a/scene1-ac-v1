#!/usr/bin/env bash
# Run only inside a fresh compatible ROS Noetic container.
set -euo pipefail
cd "$(dirname "$0")/.."
test -f /opt/ros/noetic/setup.bash
test -x /root/miniforge3/bin/conda
for target in /root/miniforge3/envs/kls_dev /root/LeTools-Learning /root/scene1_act_optimization; do
  test ! -e "$target" || { echo "Already exists: $target; use a fresh container." >&2; exit 1; }
done
python3 recovery/verify.py --runtime
/root/miniforge3/bin/conda create -y -p /root/miniforge3/envs/kls_dev --file recovery/conda-production-explicit.txt
/root/miniforge3/envs/kls_dev/bin/python -m pip install --no-deps -r recovery/pip-production-nodeps.txt
cp -a runtime/LeTools-Learning /root/LeTools-Learning
/root/miniforge3/envs/kls_dev/bin/python -m pip install --no-deps -e /root/LeTools-Learning
/root/miniforge3/envs/kls_dev/bin/python - <<'PY'
import shutil,sysconfig
from pathlib import Path
site=Path(sysconfig.get_paths()['purelib'])
for name in ('kuavo_humanoid_sdk','kuavo_humanoid_sdk-1.3.3.dist-info'):
    target=site/name
    if target.exists():shutil.rmtree(target)
    shutil.copytree(Path('runtime')/name,target)
PY
cp -a runtime/scene1_act_optimization /root/scene1_act_optimization
mkdir -p /root/scene1_act_optimization/outputs
echo 'Python/SDK and legacy baseline restored. Configure simulator/GPU/X11 before running.'
