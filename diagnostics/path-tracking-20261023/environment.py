"""Reuse previously verified process helpers without changing historical artifacts."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'diagnostics/mpc-wbc-tracking-20261022'
_spec = importlib.util.spec_from_file_location('tracking_previous_experiment', OLD / 'experiment.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ALGO, SIM, PROJECT, WS = _module.ALGO, _module.SIM, _module.PROJECT, _module.WS
command, wait_idle, stop = _module.command, _module.wait_idle, _module.stop
BASE_SHA = 'e46e9a800011f8c5bf43e269cb2ba0ab7d543c27e09ac10deaea6ae06e1179e4'
PD_SHA = 'ab20325136810cc785bfdb4abd80c3ecc2b2e73627ebebea42b904bd7bb3f0ba'

HOST_PROJECT = HERE / 'candidate-policy'
PROJECT = '/root/scene1_mpcwbc_candidates/reference_hold'
