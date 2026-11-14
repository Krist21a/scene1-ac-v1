"""Optional learned short skill inside the existing visual pickup state machine.

The state machine owns contact, arrival, trial-lift evidence and its original
deadlines. A failed learned attempt returns to that same controller without
resetting any phase budget. Model outputs never establish success by themselves.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .command_sweep import sampled_sweep
from .motion import action_vector


class LocalACTGrasp:
    def __init__(self, model_path, manifest_path, device, config, low, high, output):
        import torch
        from kuavo_deploy.utils.policy_loader import load_native_policy_bundle
        model_path, manifest_path = Path(model_path), Path(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('stage') not in ('validation_candidate', 'frozen_final')
                or manifest.get('training_updates') != 20000
                or not manifest.get('training_receipt_sha256')
                or not manifest.get('files_sha256')):
            raise ValueError('Local ACT requires a frozen checkpoint from the completed training recipe')
        for name, expected in manifest['files_sha256'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe ACT checkpoint manifest path')
            if hashlib.sha256((model_path/relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Local ACT checkpoint changed: '+name)
        required = {'config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json'}
        if not required.issubset(manifest['files_sha256']):
            raise ValueError('Model manifest omits weights or preprocessing')
        if {str(p.relative_to(model_path)) for p in model_path.rglob('*') if p.is_file()} != set(manifest['files_sha256']):
            raise ValueError('Model manifest must cover every checkpoint and preprocessing file')
        self.policy, self.preprocess, self.postprocess, _ = load_native_policy_bundle(
            pretrained_path=str(model_path), device=torch.device(device), strict=True)
        if (self.policy.config.type != 'act' or self.policy.config.chunk_size != 30
                or self.policy.config.vision_backbone != 'resnet18'):
            raise ValueError('Local skill requires ACT / ResNet-18 / 30-step chunks')
        self.policy.eval()
        self.config = config
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.output = Path(output)
        self.manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.active = None
        self.attempts = []
        self.last_inference = None
        self.last_prediction = None

    def _fail(self, reason, step):
        if self.active is not None:
            self.active.update(success_before_recovery=False, failure_reason=reason,
                recovery_started_step=step, recovery_finished=False)
            self.active = None
        self.policy.reset()
        self.last_prediction = None

    def apply(self, observation, pickup, assist, measured, previous, traditional, info, step):
        import torch
        from kuavo_deploy.utils.policy_loader import inject_task_prompt
        parcel = pickup.completed_parcels
        record = next((row for row in self.attempts if row['parcel_index'] == parcel), None)
        if info.get('stage') == 'hover' and info.get('transition_to') == 'descend' and record is None:
            self.policy.reset()  # No action queue or temporal ensemble crosses skill boundaries.
            self.last_inference = self.last_prediction = None
            record = dict(parcel_index=parcel, start_step=step, start_wall_s=time.time(),
                success_before_recovery=None, recovery_finished=False, prediction_count=0,
                action_limits_applied=0, failure_reason=None)
            self.attempts.append(record)
            self.active = record
        if record is None:
            return traditional
        externally_complete = (info.get('stage') == 'lift'
            and info.get('transition_to') == 'observe_pad' and info.get('lift_evidence') is not None)
        if externally_complete:
            record.update(end_step=step, end_wall_s=time.time(), external_lift_evidence=info['lift_evidence'])
            if self.active is record:
                record['success_before_recovery'] = True
                self.active = None
            else:
                record.update(recovery_finished=True, success_after_recovery=True)
            self.policy.reset()
            self.last_prediction = None
        if self.active is not record:
            info['local_act'] = dict(record)
            return traditional
        stage = info.get('transition_to') if info.get('stage') == 'hover' else info.get('stage')
        if stage not in ('descend', 'local_grasp', 'lift'):
            self._fail('unexpected skill phase '+str(stage), step)
            info['local_act'] = dict(record)
            return traditional
        # Leave the remainder of each EXISTING phase budget to recovery. No
        # counters, contacts, references or retry allowances are reset here.
        pending = assist.pending or {}
        if stage == 'local_grasp':
            closing = pending.get('stage') == 'close_confirm'
            spent = pending.get('closing_steps', 0) if closing else pending.get('steps', 0)
            budget = assist.config['max_close_settle_steps'] if closing else assist.config['max_grasp_steps']
        else:
            spent, budget = pickup.motion.stage_steps, 500
        if spent >= budget//2:
            self._fail('learned attempt exhausted its share of the existing phase budget', step)
            info['local_act'] = dict(record)
            return traditional
        if info.get('table_reference_hold') or info.get('sensor_error'):
            info['local_act'] = dict(record, deferred_for_existing_sensor_hold=True)
            return traditional
        try:
            now = time.monotonic()
            # Replan at 10 Hz; any faster control cycles retain the latest
            # first action. Never execute the remainder of an old action chunk.
            if self.last_inference is None or now-self.last_inference >= .1:
                policy_observation = dict(observation)
                for key, value in policy_observation.items():
                    if key.startswith('observation.images.') and value.dtype == torch.uint8:
                        policy_observation[key] = value.to(torch.float32).div(255).contiguous()
                batch = self.preprocess(inject_task_prompt(policy_observation, 'scene1'))
                with torch.inference_mode():
                    chunk = self.policy.predict_action_chunk(batch)
                    decoded = self.postprocess(chunk[:, 0])
                self.last_prediction = action_vector(decoded.detach().cpu().numpy().reshape(-1))
                self.last_inference = now
                record['prediction_count'] += 1
            predicted = self.last_prediction.copy()
            proposed = predicted.copy()
            # The demonstrated skill is a right-hand pickup; retain the
            # prepared left hand while still consuming the canonical 16-vector.
            proposed[:8] = previous[:8]
            closing_hold = stage == 'local_grasp' and pending.get('stage') == 'close_confirm'
            if (not closing_hold and (np.any(proposed[8:15] < self.low[8:15])
                                      or np.any(proposed[8:15] > self.high[8:15]))):
                raise ValueError('learned target outside physical command bounds')
            limit = float(assist.config['max_joint_step_rad'])
            if stage in ('descend', 'lift'):
                profile = pickup.motion.config.get('servo_profiles', {}).get('pickup_'+stage, {})
                limit = float(profile.get('noncontact_step_rad',
                    pickup.motion.config.get('max_joint_step_rad', .012)))
                if pickup.motion.servo_precision[1]:
                    limit *= .5
            delta = proposed[8:15]-previous[8:15]
            scale = min(1., limit/max(np.max(abs(delta)), 1e-12))
            proposed[8:15] = previous[8:15]+scale*delta
            if scale < 1.:
                record['action_limits_applied'] += 1
            if stage == 'descend':
                proposed[15] = 0.
            elif stage == 'lift':
                proposed[15] = traditional[15]
            else:
                # Existing geometric/contact checks alone permit closure.
                proposed[15] = 0. if traditional[15] == 0. else min(proposed[15], traditional[15])
                if closing_hold:
                    proposed[8:15] = traditional[8:15]
                elif assist.first_grasp_complete:
                    proposed[15] = traditional[15]
            # Validate the components that can actually be sent. The existing
            # state machine owns open/lift grip targets and settled arm holds;
            # an unused prediction cannot override those contact decisions.
            if np.any(proposed < self.low) or np.any(proposed > self.high):
                raise ValueError('learned target outside physical command bounds')
            if stage == 'local_grasp' and 'base' in pending:
                if np.max(abs(proposed[8:15]-np.asarray(pending['base'])[8:15])) > assist.config['max_joint_offset_rad']:
                    raise ValueError('learned correction exceeds the existing grasp offset budget')
            if stage in ('descend', 'local_grasp') and not assist.closed[1]:
                audit = sampled_sweep(pickup.motion.k, measured[8:15], proposed[8:15],
                                      pickup.target, pickup.plane)
                if not audit['clear']:
                    raise ValueError('learned proposal rejected by the original open-finger sweep')
                if pending.get('stage') != 'close_confirm':
                    # Independently refresh feedback after inference in the
                    # runner's original final-send audit as well.
                    assist.command_sweep_context = dict(stage='visual_pickup_'+stage,
                        target=copy.deepcopy(pickup.target), plane=pickup.plane)
                    info['visual_pickup_sweep_required'] = True
            info.update(policy_controls_arms=True, strategy='local_act_with_external_skill_gates')
            info['local_act'] = dict(record, inference_age_s=time.monotonic()-self.last_inference,
                rolling_replan=True, predicted_action=predicted.tolist(), bounded_action=proposed.tolist())
            return proposed
        except Exception as exc:
            self._fail(str(exc), step)
            info['local_act'] = dict(record)
            return traditional

    def observe_sent(self, sent, assist):
        """Keep the existing local controller anchored to the actual command.

        During a learned approach its unused classical proposal is not the
        executed reference. Recovery must continue from the filtered, guarded
        target that was actually published, within the same pending budget.
        Settled-contact holds retain their original captured reference.
        """
        pending = assist.pending or {}
        if (self.active is not None and pending.get('kind') == 'grasp'
                and pending.get('stage') != 'close_confirm'):
            assist.offset[8:15] = np.asarray(sent)[8:15]-np.asarray(pending['base'])[8:15]

    def finish(self, reason, step):
        self._fail('episode ended: '+reason, step)
        for record in self.attempts:
            if record.get('success_before_recovery') is False and not record.get('recovery_finished'):
                record['success_after_recovery'] = False
        report = dict(model_manifest_sha256=self.manifest_sha256, attempts=self.attempts,
            episode_end_reason=reason, success_source='external contact and visual trial lift')
        (self.output/'local-act-grasps.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
