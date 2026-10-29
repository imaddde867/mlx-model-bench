"""Offline contracts for the separate paired LFM3 experiment."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experimental' / 'lfm3'))
spec = importlib.util.spec_from_file_location('lfm3_study', ROOT / 'experimental/lfm3/study.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class Lfm3StudyTests(unittest.TestCase):
    def test_schedule_is_seeded_and_first_four_blocks_balance_positions(self):
        orders = s.schedule(37029)
        self.assertEqual(orders, s.schedule(37029))
        self.assertEqual(len(orders), 5)
        for mid in s.MODELS:
            self.assertEqual([order.index(mid) for order in orders[:4]].count(0), 2)
        self.assertTrue(all(set(order) == set(s.MODELS) for order in orders))

    def test_commands_use_frozen_model_settings_and_isolated_output(self):
        plan = {'evidence': '/verified', 'models': [{'model_id': m, 'checkpoint': 'mlx-4bit'} for m in s.MODELS]}
        cmd = s.command('p2', plan, s.MODELS[0], Path('/new/raw'), 'clean', 'unique')
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[cmd.index('--max-tokens') + 1], '256')
        self.assertIn('--no-nonce', cmd)
        self.assertEqual(cmd[cmd.index('--trials') + 1], '5')
        self.assertEqual(cmd[cmd.index('--session-id') + 1], 'unique')
        for phase, script in [('p3', 'run_p3.py'), ('p4', 'run_p4.py')]:
            cmd = s.command(phase, plan, s.MODELS[0], Path('/new/raw'), 'clean', 'unique')
            self.assertEqual(Path(cmd[1]).name, script)
            self.assertEqual(cmd[cmd.index('--evidence') + 1], '/verified')
            self.assertEqual('--condition' in cmd, phase != 'p3')

    def test_quiet_gate_retries_cpu_and_fails_closed_on_power(self):
        quiet = {'power': 'ac', 'low_power_mode': 0, 'cpu_speed_limit_pct': None, 'top_cpu': []}
        busy = {**quiet, 'top_cpu': [{'proc': 'other', 'cpu_pct': 80}]}
        with mock.patch.object(s.sysinfo, 'snapshot', side_effect=[busy, quiet]), mock.patch.object(s.time, 'sleep') as sleep:
            self.assertEqual(s.wait_quiet(), quiet)
            sleep.assert_called_once_with(60)
        for bad in ({**quiet, 'power': 'battery'}, {**quiet, 'low_power_mode': None}, {**quiet, 'cpu_speed_limit_pct': 80}):
            with mock.patch.object(s.sysinfo, 'snapshot', return_value=bad), self.assertRaises(ValueError):
                s.wait_quiet()

    def test_failed_child_retains_exclusive_manifest_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'session'
            with mock.patch.object(s.sysinfo, 'snapshot', return_value={}), mock.patch.object(s.subprocess, 'run', return_value=mock.Mock(returncode=9)):
                with self.assertRaisesRegex(RuntimeError, '9'):
                    s.execute(folder, ['fixture'], {'model_id': 'model'})
            manifest = s.r.load_json(folder / 'session.json')
            self.assertEqual(manifest['return_code'], 9)
            self.assertEqual(manifest['artifacts_sha256'], s.r.evidence_hashes(folder))
            with self.assertRaises(FileExistsError):
                s.execute(folder, ['fixture'], {})

    def test_speed_rows_allow_cap_stops_but_reject_identity_and_summary_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mid = s.MODELS[0]
            raw = folder / 'raw' / mid / 'mlx-4bit-inproc'
            raw.mkdir(parents=True)
            model = {'model_id': mid, 'checkpoint': 'mlx-4bit', 'hf_repo': 'vendor/model',
                     'publisher': 'vendor', 'template_kwargs': {}}
            versions = {'mlx': 'pinned'}
            plan = {'models': [model], 'environment': {'machine': {'packages': versions}}}
            base = {**model, 'max_tokens': 256, 'trials': 5, 'condition': 'clean',
                    'session_id': 'unique', 'versions': versions, 'greedy': True}
            summary = {**base, 'outcome': 'completed', 'prompts': {}}
            for regime in s.PROMPTS:
                for trial in range(6):
                    row = {**base, 'prompt': regime, 'trial': trial, 'discarded': trial == 0,
                           'ok': True, 'nonce': False, 'system': {}, 'finish_reason': 'length',
                           'generation_tokens': 256, 'prompt_tokens': 100, 'text': 'output',
                           'wall_s': 1.2, 'request_ttft_s': 0.2, 'peak_gb': 1.1,
                           'prompt_file': str(s.PROMPT_DIR / f'{regime}.txt')}
                    name = f'{regime}-{"warmup" if trial == 0 else f"trial{trial}"}.json'
                    (raw / name).write_text(json.dumps(row))
                summary['prompts'][regime] = {'kept_ok': 5, 'warmup_ok': True,
                    **{k: {'median': row[k], 'min': row[k], 'max': row[k], 'n': 5}
                       for k in ('wall_s', 'request_ttft_s', 'peak_gb', 'generation_tokens')}}
            (raw / 'summary.json').write_text(json.dumps(summary))
            s.validate_speed(folder, plan, mid, 'clean', 'unique')
            path = raw / 'short-trial1.json'
            row = json.loads(path.read_text())
            row['session_id'] = 'other'
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'identity'):
                s.validate_speed(folder, plan, mid, 'clean', 'unique')
            row['session_id'] = 'unique'
            path.write_text(json.dumps(row))
            summary['prompts']['short']['wall_s']['median'] = 99
            (raw / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'median/range'):
                s.validate_speed(folder, plan, mid, 'clean', 'unique')

    def test_capability_rejects_runtime_cap_and_stop_drift_but_preserves_failed_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mid = s.MODELS[0]
            path = folder / 'raw' / mid / 'p4' / 'mlx-4bit-results.json'
            path.parent.mkdir(parents=True)
            cases = s.r.load_json(ROOT / 'prompts/agent/tasks.json')
            model = {'model_id': mid, 'checkpoint': 'mlx-4bit', 'hf_repo': 'vendor/model',
                     'publisher': 'vendor', 'template_kwargs': {'enable_thinking': False}}
            versions = {'mlx': 'pinned'}
            plan = {'models': [model], 'environment': {'machine': {'packages': versions}}}
            report = {**model, 'versions': versions, 'per_turn_max_tokens': 256,
                      'total': len(cases), 'passed': 0,
                      'tasks': [{'id': t['id'], 'passed': False, 'steps': [
                          {'response': '', 'raw_response': '<think>unfinished',
                           'finish_reason': 'length', 'system': {}}]} for t in cases]}
            path.write_text(json.dumps(report))
            s.validate_capability(folder, plan, mid, 'p4')
            for field, bad in [('versions', {}), ('per_turn_max_tokens', 128), ('template_kwargs', {})]:
                changed = {**report, field: bad}
                path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    s.validate_capability(folder, plan, mid, 'p4')
            report['tasks'][0]['steps'][0]['finish_reason'] = None
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'stop/system'):
                s.validate_capability(folder, plan, mid, 'p4')
            report['tasks'].pop()
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'completeness'):
                s.validate_capability(folder, plan, mid, 'p4')

    def test_capability_observations_use_nested_pre_generation_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            snap = {'top_cpu': [{'proc': 'other', 'cpu_pct': 90}]}
            (folder / 'report.json').write_text(json.dumps({
                'accuracy_runs': [{'system': snap}], 'quality_runs': [{'system': snap}]}))
            self.assertEqual(s.observations(folder), [{'system': snap}, {'system': snap}])
            (folder / 'report.json').write_text(json.dumps({
                'tasks': [{'steps': [{'system': snap}]}]}))
            self.assertEqual(s.observations(folder), [{'system': snap}])

    def test_plan_digest_rejects_tampering(self):
        plan = {'schema': 1, 'models': [], 'schedule': s.schedule(37029)}
        plan['plan_id'] = s.r.digest(plan)
        s.check_digest(plan)
        plan['schedule'][0].reverse()
        with self.assertRaisesRegex(ValueError, 'digest'):
            s.check_digest(plan)


if __name__ == '__main__':
    unittest.main()
