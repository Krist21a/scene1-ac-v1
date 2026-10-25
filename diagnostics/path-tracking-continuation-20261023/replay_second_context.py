"""Persist the real second-parcel regression certificate; sends no commands."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
root=Path(__file__).resolve().parent
policy=root/'candidate-policy-executor-context'
sys.path[:0]=[str(policy),str(policy/'tests')]
from test_parcel_controller_context import ParcelControllerContextTests
from scene1.release_retreat import axis_retreat_goal
captured={}
def record(*args,**kwargs):
    goal,audit=axis_retreat_goal(*args,**kwargs)
    captured.update(audit)
    return goal,audit
test=ParcelControllerContextTests()
test.setUp()
with patch('scene1.residual.axis_retreat_goal',side_effect=record):
    test.test_retained_second_parcel_release_executes_certificate_after_fresh_construction()
(root/'second-context-replay.json').write_text(json.dumps(captured,indent=2)+'\n')
print(json.dumps({k:captured[k] for k in ['certified','nominal_steps','remaining_steps','position_residual_m']}))
