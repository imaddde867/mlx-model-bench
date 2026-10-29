#!/usr/bin/env python3
"""Separate five-block LFM3/MiniCPM experiment; inference stays in existing runners."""

import argparse
import importlib.metadata
import json
import random
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import registry
import replicated as r
import sysinfo
import lfm3_parity as lp

MODELS = ('lfm3-1.7b', 'minicpm5-2b')
PROMPTS = ('short', 'medium', 'long', 'decode-short', 'decode-long')
PROMPT_DIR = ROOT / 'prompts' / 'speed'
FILES = [*(PROMPT_DIR / f'{p}.txt' for p in PROMPTS),
         ROOT / 'p3_suite.py', ROOT / 'prompts/agent/tasks.json']


def schedule(seed):
    order = list(MODELS)
    random.Random(seed).shuffle(order)
    return [order[:] if block % 2 == 0 else order[::-1] for block in range(5)]


def environment():
    machine = r.machine()
    direct = json.loads(importlib.metadata.distribution('mlx-lm').read_text('direct_url.json') or '{}')
    commit = direct.get('vcs_info', {}).get('commit_id')
    if (machine['os'] != 'Darwin' or machine['chip'] != 'Apple M4'
            or machine['arch'] != 'arm64' or machine['cpu_count'] != 10
            or machine['ram_bytes'] != 16 * 2**30 or machine['python'] != '3.12.10'
            or machine['packages']['mlx'] != lp.PINS['mlx']
            or commit != lp.PINS['mlx_lm_commit']):
        raise ValueError('requires the pinned experimental runtime on the 16 GiB M4')
    return {'machine': machine, 'mlx_lm_commit': commit}


def make_plan(evidence, seed):
    plan = {'schema': 1, 'created_utc': r.now(), 'source_commit': r.git_head(),
            'environment': environment(), 'evidence': str(Path(evidence).resolve()),
            'models': [r.verified_identity(registry.get_model(mid), evidence) for mid in MODELS],
            'files_sha256': {str(p.relative_to(ROOT)): r.sha_file(p) for p in FILES},
            'seed': seed, 'schedule': schedule(seed),
            'settings': {'p2_trials': 5, 'p2_max_tokens': 256, 'prompts': list(PROMPTS),
                         'nonce': False, 'temperature': 0, 'cooldown_s': 120,
                         'p3_accuracy_max_tokens': 256, 'p3_quality_max_tokens': 384,
                         'p4_per_turn_max_tokens': 256, 'p4_max_steps': 7}}
    plan['plan_id'] = r.digest(plan)
    return plan


def check_digest(plan):
    if plan.get('schema') != 1 or plan.get('plan_id') != r.digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('plan digest mismatch')


def verify_host(plan):
    check_digest(plan)
    if plan['source_commit'] != r.git_head() or plan['environment'] != environment():
        raise ValueError('source/runtime drift')
    if plan['schedule'] != schedule(plan['seed']):
        raise ValueError('schedule drift')
    if plan['models'] != [r.verified_identity(registry.get_model(mid), plan['evidence']) for mid in MODELS]:
        raise ValueError('checkpoint/provenance drift')
    if any(r.sha_file(ROOT / name) != digest for name, digest in plan['files_sha256'].items()):
        raise ValueError('prompt/task drift')


def wait_quiet():
    for attempt in range(5):
        snap = sysinfo.snapshot()
        if snap.get('power') != 'ac' or snap.get('low_power_mode') != 0:
            raise ValueError('AC power and low-power mode off must be observed')
        if snap.get('cpu_speed_limit_pct') not in (None, 100):
            raise ValueError('reported thermal throttling')
        if snap.get('top_cpu') is not None and not any(p['cpu_pct'] >= 15 for p in snap['top_cpu']):
            return snap
        if attempt < 4:
            time.sleep(60)
    raise ValueError('desktop did not become quiet; no new session started')


def command(phase, plan, model_id, out, condition, session_id):
    script = {'p2': 'bench.py', 'p3': 'run_p3.py', 'p4': 'run_p4.py'}[phase]
    cmd = [sys.executable, str(ROOT / script), '--model', model_id,
           '--out', str(out), '--evidence', plan['evidence']]
    if phase != 'p3':
        cmd += ['--condition', condition]
    if phase == 'p2':
        cmd += ['--checkpoint', 'mlx-4bit', '--transport', 'inproc', '--trials', '5',
                '--max-tokens', '256', '--no-nonce', '--session-id', session_id,
                '--prompts', *PROMPTS, '--prompt-dir', str(PROMPT_DIR)]
    else:
        cmd += ['--checkpoints', 'mlx-4bit']
    return cmd


def observations(raw):
    rows = []
    for record in r.trial_rows(raw):
        if 'accuracy_runs' in record:
            rows.extend(record['accuracy_runs'])
            rows.extend(record['quality_runs'])
        elif 'tasks' in record:
            rows.extend(step for task in record['tasks'] for step in task['steps'])
        elif 'checkpoints' not in record:
            rows.append(record)
    return rows


def execute(folder, cmd, identity, ram_bytes=None):
    folder.mkdir(parents=True, exist_ok=False)
    before = sysinfo.snapshot()
    start = r.now()
    r.write_once(folder / 'started.json', {**identity, 'started_utc': start, 'before': before, 'command': cmd})
    sampler = r.MemorySampler(ram_bytes) if ram_bytes else None
    rc, error = None, None
    if sampler:
        sampler.start()
    try:
        with (folder / 'bench.log').open('x') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                timeout=3600, check=False).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        error = str(exc)
    finally:
        memory = sampler.stop() if sampler else None
        try:
            rows = observations(folder / 'raw')
        except (OSError, ValueError) as exc:
            rows = []
            error = error or str(exc)
        after = sysinfo.snapshot()
        payload = {**identity, 'started_utc': start, 'ended_utc': r.now(),
                   'return_code': rc, 'error': error,
                   'observed': {'before': before, 'after': after,
                                'caveats': r.observed_caveats(before, after, rows)},
                   'host_memory': memory}
        r.write_once(folder / 'session_payload.json', payload)
        r.write_once(folder / 'session.json', {**payload, 'artifacts_sha256': r.evidence_hashes(folder)})
    if rc != 0 or error:
        raise RuntimeError(f'child failed ({rc}): {folder}; {error}')


def validate_speed(folder, plan, model_id, condition, session_id):
    """Check all warmups/kept rows and recompute declared performance summaries."""
    model = next(m for m in plan['models'] if m['model_id'] == model_id)
    raw = folder / 'raw' / model_id / 'mlx-4bit-inproc'
    summary = r.load_json(raw / 'summary.json')
    expected = {'model_id': model_id, 'checkpoint': 'mlx-4bit', 'hf_repo': model['hf_repo'],
                'publisher': model['publisher'], 'template_kwargs': model['template_kwargs'],
                'max_tokens': 256, 'trials': 5, 'condition': condition, 'session_id': session_id,
                'versions': plan['environment']['machine']['packages'], 'greedy': True}
    if any(summary.get(k) != v for k, v in expected.items()) or summary.get('outcome') != 'completed':
        raise ValueError(f'summary identity/completion mismatch: {folder}')
    if set(summary['prompts']) != set(PROMPTS):
        raise ValueError('missing speed regimes')
    trial_files = set()
    for regime in PROMPTS:
        rows = []
        for trial in range(6):
            name = f'{regime}-{"warmup" if trial == 0 else f"trial{trial}"}.json'
            trial_files.add(name)
            row = r.load_json(raw / name)
            check = {k: v for k, v in expected.items() if k != 'trials'}
            check.update(prompt=regime, trial=trial, discarded=trial == 0, ok=True, nonce=False)
            if any(row.get(k) != v for k, v in check.items()) or not isinstance(row.get('system'), dict):
                raise ValueError(f'trial identity mismatch: {name}')
            if row.get('finish_reason') not in ('stop', 'length'):
                raise ValueError(f'unknown stop reason: {name}')
            if row.get('finish_reason') == 'length' and row.get('generation_tokens') != 256:
                raise ValueError(f'cap stop before token cap: {name}')
            if not 0 < row['generation_tokens'] <= 256:
                raise ValueError(f'invalid generation count: {name}')
            if Path(row.get('prompt_file', '')).resolve() != (PROMPT_DIR / f'{regime}.txt').resolve():
                raise ValueError(f'prompt identity mismatch: {name}')
            for metric in ('wall_s', 'request_ttft_s', 'peak_gb', 'prompt_tokens'):
                if type(row.get(metric)) not in (int, float) or not r.math.isfinite(row[metric]) or row[metric] <= 0:
                    raise ValueError(f'invalid metric: {name}/{metric}')
            rows.append(row)
        if len({row['prompt_tokens'] for row in rows}) != 1:
            raise ValueError('tokenizer prompt count drift')
        declared = summary['prompts'][regime]
        if declared.get('kept_ok') != 5 or declared.get('warmup_ok') is not True:
            raise ValueError('warmup/kept completeness mismatch')
        for metric in ('wall_s', 'request_ttft_s', 'peak_gb', 'generation_tokens'):
            values = [row[metric] for row in rows[1:]]
            stats = declared[metric]
            if not all(r.close(stats.get(k), v) for k, v in
                       [('median', statistics.median(values)), ('min', min(values)), ('max', max(values))]) or stats.get('n') != 5:
                raise ValueError(f'median/range mismatch: {regime}/{metric}')
    if {p.name for p in raw.glob('*.json')} != trial_files | {'summary.json'}:
        raise ValueError('unexpected/missing raw trial files')


def validate_capability(folder, plan, model_id, phase):
    report = r.load_json(folder / 'raw' / model_id / phase / 'mlx-4bit-results.json')
    model = next(m for m in plan['models'] if m['model_id'] == model_id)
    expected_identity = {k: model[k] for k in ('model_id', 'checkpoint', 'hf_repo', 'publisher', 'template_kwargs')}
    expected_identity['versions'] = plan['environment']['machine']['packages']
    if any(report.get(k) != v for k, v in expected_identity.items()):
        raise ValueError('capability identity/runtime mismatch')
    if phase == 'p3' and report.get('max_tokens') != {'accuracy': 256, 'quality': 384}:
        raise ValueError('P3 frozen cap mismatch')
    if phase == 'p4' and report.get('per_turn_max_tokens') != 256:
        raise ValueError('P4 frozen cap mismatch')
    if phase == 'p3':
        import p3_suite
        expected = {'accuracy_runs': p3_suite.MATH_PROMPTS + p3_suite.TOOL_PROMPTS + p3_suite.INSTR_PROMPTS,
                    'quality_runs': p3_suite.QUALITY_PROMPTS}
    else:
        expected = {'tasks': r.load_json(ROOT / 'prompts/agent/tasks.json')}
    for field, cases in expected.items():
        ids = [row['id'] for row in report[field]]
        if len(ids) != len(set(ids)) or set(ids) != {case['id'] for case in cases}:
            raise ValueError('capability task completeness mismatch')
    if phase == 'p3':
        generations = report['accuracy_runs'] + report['quality_runs']
    else:
        if any(not isinstance(t.get('steps'), list) or not 1 <= len(t['steps']) <= 7
               for t in report['tasks']):
            raise ValueError('missing/invalid P4 steps')
        generations = [step for task in report['tasks'] for step in task['steps']]
    for row in generations:
        if row.get('finish_reason') not in ('stop', 'length') or not isinstance(row.get('system'), dict):
            raise ValueError('capability stop/system provenance missing')
        if not isinstance(row.get('response'), str) or not isinstance(row.get('raw_response', row['response']), str):
            raise ValueError('capability response missing')
    if phase == 'p4' and (report.get('total') != len(report['tasks'])
                          or report.get('passed') != sum(t['passed'] for t in report['tasks'])):
        raise ValueError('P4 pass/count mismatch')


def validate(folder, capabilities=True):
    folder = Path(folder)
    plan = r.load_json(folder / 'plan.json')
    check_digest(plan)
    seen, previous_end = set(), None
    for block, order in enumerate(plan['schedule'], 1):
        for pos, mid in enumerate(order, 1):
            path = folder / 'sessions' / f'b{block:02d}-p{pos:02d}-{mid}'
            record = r.load_json(path / 'session.json')
            if record['return_code'] != 0 or record['error'] is not None:
                raise ValueError(f'failed session: {path}')
            if record['artifacts_sha256'] != r.evidence_hashes(path):
                raise ValueError(f'evidence hash mismatch: {path}')
            if {k: v for k, v in record.items() if k != 'artifacts_sha256'} != r.load_json(path / 'session_payload.json'):
                raise ValueError('manifest/payload mismatch')
            sid = record['session_id']
            if sid in seen or record['plan_id'] != plan['plan_id']:
                raise ValueError('duplicate UUID or plan identity mismatch')
            seen.add(sid)
            start, end = r.utc_time(record['started_utc']), r.utc_time(record['ended_utc'])
            if start >= end or (previous_end and start < previous_end):
                raise ValueError('session overlap or nonchronological schedule')
            previous_end = end
            validate_speed(path, plan, mid, record['condition'], sid)
    if capabilities:
        for mid in MODELS:
            for phase in ('p3', 'p4'):
                path = folder / 'capabilities' / f'{mid}-{phase}'
                record = r.load_json(path / 'session.json')
                if record['return_code'] != 0 or record['error'] or record['plan_id'] != plan['plan_id']:
                    raise ValueError('capability process failure')
                if record['artifacts_sha256'] != r.evidence_hashes(path):
                    raise ValueError('capability evidence hash mismatch')
                if {k: v for k, v in record.items() if k != 'artifacts_sha256'} != r.load_json(path / 'session_payload.json'):
                    raise ValueError('capability manifest/payload mismatch')
                validate_capability(path, plan, mid, phase)
    return plan


def run(folder, condition):
    folder = Path(folder)
    plan = r.load_json(folder / 'plan.json')
    with r.experiment_lock(folder):
        if (folder / 'sessions').exists() or (folder / 'capabilities').exists():
            raise FileExistsError('study already started; no automatic reruns or overwrites')
        for block, order in enumerate(plan['schedule'], 1):
            verify_host(plan)
            time.sleep(120)
            wait_quiet()
            for pos, mid in enumerate(order, 1):
                sid = uuid.uuid4().hex
                path = folder / 'sessions' / f'b{block:02d}-p{pos:02d}-{mid}'
                identity = {'plan_id': plan['plan_id'], 'model_id': mid, 'block': block,
                            'position': pos, 'session_id': sid, 'condition': condition, 'phase': 'p2'}
                execute(path, command('p2', plan, mid, path / 'raw', condition, sid), identity,
                        plan['environment']['machine']['ram_bytes'])
                validate_speed(path, plan, mid, condition, sid)
        validate(folder, capabilities=False)
        for mid in MODELS:
            for phase in ('p3', 'p4'):
                verify_host(plan)
                wait_quiet()
                path = folder / 'capabilities' / f'{mid}-{phase}'
                sid = uuid.uuid4().hex
                execute(path, command(phase, plan, mid, path / 'raw', condition, sid),
                        {'plan_id': plan['plan_id'], 'model_id': mid, 'phase': phase,
                         'session_id': sid, 'condition': condition})
                validate_capability(path, plan, mid, phase)
        validate(folder)
        r.write_once(folder / 'completed.json', {'plan_id': plan['plan_id'], 'ended_utc': r.now()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--out', type=Path, required=True)
    plan.add_argument('--evidence', type=Path, required=True)
    plan.add_argument('--seed', type=int, default=37029)
    run_parser = sub.add_parser('run')
    run_parser.add_argument('folder', type=Path)
    run_parser.add_argument('--condition', choices=('clean', 'loaded'), required=True)
    val = sub.add_parser('validate')
    val.add_argument('folder', type=Path)
    args = ap.parse_args()
    if args.action == 'plan':
        doc = make_plan(args.evidence, args.seed)
        args.out.mkdir(parents=True, exist_ok=False)
        r.write_once(args.out / 'plan.json', doc)
    elif args.action == 'run':
        run(args.folder, args.condition)
    else:
        validate(args.folder)
        print('10/10 paired performance sessions validated')


if __name__ == '__main__':
    main()
