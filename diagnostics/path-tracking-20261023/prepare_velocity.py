"""Build two isolated single-factor source variants; never edit live sources."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

here = Path(__file__).resolve().parent
old = here.parent/'mpc-wbc-tracking-20261022'
block = '''
      // Research candidate: restore velocity consistency for wrist DOFs omitted
      // from the simplified MPC, in the NORMAL route only. Position, proximal
      // MPC outputs, filters, WBC gains/weights and torque limits stay unchanged.
      static Scene1WristVelocityReference scene1_wrist_reference;
      std::array<double, 14> scene1_filtered_positions{};
      const bool scene1_reference_enabled = !is_real_ && is_simplified_model_ &&
          armNumReal_ == 14 && armDofReal_ == 7 && armDofMPC_ == 4 &&
          !use_ros_arm_joint_trajectory_ && !use_mm_arm_joint_trajectory_ &&
          !only_half_up_body_ && !wbc_only_ && !isPullUp_ && enable_mpc &&
          resetting_mpc_state_ == ResettingMpcState::NOMAL &&
          mpcArmControlMode_ == ArmControlMode::EXTERN_CONTROL &&
          mpcArmControlMode_desired_ == ArmControlMode::EXTERN_CONTROL;
      if (armNumReal_ == 14) {
        for (int j = 0; j < 14; ++j)
          scene1_filtered_positions[j] = optimizedState2WBC_mrt_.tail(14)(j);
      }
      const auto scene1_velocity = scene1_wrist_reference.update(
          scene1_filtered_positions, currentObservation_.time,
          scene1_reference_enabled, arm_interpolation_max_velocity_);
      if (scene1_velocity.enabled) {
        for (int j : {4, 5, 6, 11, 12, 13})
          optimizedInput2WBC_mrt_.tail(14)(j) = scene1_velocity.bounded[j];
      }
      if (armNumReal_ == 14) {
        vector_t scene1_velocity_diagnostic = vector_t::Zero(47);
        for (int j = 0; j < 14; ++j) {
          scene1_velocity_diagnostic(j) = scene1_filtered_positions[j];
          scene1_velocity_diagnostic(14+j) = scene1_velocity.raw[j];
          scene1_velocity_diagnostic(28+j) = optimizedInput2WBC_mrt_.tail(14)(j);
        }
        scene1_velocity_diagnostic(42) = scene1_velocity.dt;
        scene1_velocity_diagnostic(43) = scene1_velocity.enabled;
        scene1_velocity_diagnostic(44) = scene1_velocity.clipped;
        scene1_velocity_diagnostic(45) = arm_interpolation_max_velocity_;
        scene1_velocity_diagnostic(46) = scene1_velocity.updated;
        ros_logger_->publishVector("/scene1_tracking/wrist_velocity_reference", scene1_velocity_diagnostic);
      }

'''
anchor = '      // optimizedInput2WBC_mrt_.segment(optimizedInput_mrt.size() - info.actuatedDofNum, jointNum_)'
for name, source in [('velocity',old/'baseline/src/humanoid-control/humanoid_controllers/src/humanoidController.cpp'),
                     ('velocity-pd',old/'candidate-wrist-pd/humanoidController.cpp')]:
    target = here/('candidate-'+name); target.mkdir(exist_ok=False)
    original = source.read_text()
    assert original.count(anchor) == 1
    text = original.replace('#include "humanoid_controllers/humanoidController.h"',
                            '#include "humanoid_controllers/humanoidController.h"\n#include "wrist_velocity_reference.h"')
    text = text.replace(anchor, block+anchor)
    (target/'humanoidController.cpp').write_text(text)
    shutil.copy2(here/'wrist_velocity_reference.h',target/'wrist_velocity_reference.h')
    shutil.copy2(old/'build_wrist_pd.py',target/'build.py')
    patch = ''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
                    fromfile='a/humanoidController.cpp',tofile='b/humanoidController.cpp'))
    patch += ''.join(difflib.unified_diff([], (here/'wrist_velocity_reference.h').read_text().splitlines(True),
                    fromfile='/dev/null',tofile='b/wrist_velocity_reference.h'))
    (target/'change.patch').write_text(patch)
    (target/'registration.json').write_text(json.dumps(dict(
        factor='normal MPC wrist velocity reference consistency', parent=str(source),
        parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(text.encode()).hexdigest(),
        helper_sha256=hashlib.sha256((target/'wrist_velocity_reference.h').read_bytes()).hexdigest(),
        evidence='path-reference-analysis.json: 100% of moving wrist samples had zero velocity reference',
        upper_policy='frozen-path.json', max_velocity='existing arm_interpolation_max_velocity_, no increase',
        next_gate='fixed empty motion vs original/C; reject sustained clipping or oscillation; then independent contact',
        unchanged=['position reference','normal MPC branch','proximal outputs','WBC gains and weights','torque limits','C parameters where present']),indent=2)+'\n')
    print(target)
