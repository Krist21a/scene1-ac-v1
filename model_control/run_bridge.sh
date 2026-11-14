#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
source /opt/ros/noetic/setup.bash
export PYTHONPATH="/root/LeTools-Learning:$PWD:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /root/miniforge3/envs/kls_dev/bin/python model_control/bridge.py "$@"
