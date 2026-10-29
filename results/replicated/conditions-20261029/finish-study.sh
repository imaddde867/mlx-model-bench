#!/bin/bash
set -euo pipefail
cd /Users/imadeddine/Documents/GitHub/mlx-model-bench
study=results/replicated/m4-4bit-20261029
conditions=results/replicated/conditions-20261029
trap 'result=$?; date -u +%FT%TZ > "$conditions/batch-ended.txt"; echo "$result" > "$conditions/batch-exit-code.txt"' EXIT
.venv-mlxlm/bin/python - <<'PY'
import replicated as r
p, sessions, errors = r.validate('results/replicated/m4-4bit-20261029')
assert len(sessions) == 4 and len(errors) == 16
assert all(e.endswith('missing independent session') for e in errors)
assert r.git_head() == p['source_commit']
PY
for block in 2 3 4 5; do
  sleep 120
  ready=false
  for attempt in 1 2 3 4 5; do
    if .venv-mlxlm/bin/python - <<'PY'
import json, sys
import sysinfo
s = sysinfo.snapshot()
print(json.dumps(s), flush=True)
assert s.get('power') == 'ac' and s.get('low_power_mode') == 0, 'AC/low-power gate failed'
limit = s.get('cpu_speed_limit_pct')
assert limit is None or limit == 100, 'reported thermal throttling'
sys.exit(2 if any(p['cpu_pct'] >= 15 for p in s.get('top_cpu', [])) else 0)
PY
    then
      ready=true
      break
    else
      result=$?
      if [ "$result" -ne 2 ]; then exit "$result"; fi
      sleep 60
    fi
  done
  if [ "$ready" != true ]; then echo "Desktop remained busy before block $block"; exit 1; fi
  .venv-mlxlm/bin/python sysinfo.py > "$conditions/before-b0$block.txt"
  pmset -g therm >> "$conditions/before-b0$block.txt"
  .venv-mlxlm/bin/python -u replicated.py run "$study" --block "$block" --condition clean
done
.venv-mlxlm/bin/python replicated.py validate "$study"
.venv-mlxlm/bin/python replicated.py report "$study" --out "$study/REPLICATED_STUDY.md"
echo 'ALL 20 SESSIONS VALIDATED; REPORT READY'
