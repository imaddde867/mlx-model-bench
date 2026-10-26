#!/usr/bin/env python3
"""What else is the machine doing right now?

Speed rows are only comparable if you know the conditions they ran under. This module
takes a cheap, read-only snapshot of the things that move MLX numbers on a 16 GB
Apple Silicon laptop: swap, memory pressure, power source, thermal throttling,
load average and the busiest processes.

Used two ways:

  python sysinfo.py            # preflight: print the snapshot and any warnings
  bench.snapshot per trial     # every result row records the snapshot taken just
                               # before its timed window opens (never inside it)

Every field degrades to None when its command is missing or its output does not
parse, so this is safe on Linux CI and never raises into a benchmark run.
The parsers take plain strings so the offline test suite can cover them.
"""

import json
import os
import re
import subprocess
import sys

TIMEOUT_S = 2
# Below these the snapshot is "quiet"; above, preflight() says why a row is suspect.
WARN_SWAP_GB = 0.5
WARN_FREE_PCT = 25
WARN_OTHER_CPU_PCT = 15.0


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def parse_swap(text):
    """`sysctl -n vm.swapusage` -> used GB. 'total = 2048.00M  used = 1024.50M  free = ...'"""
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", text or "")
    if not m:
        return None
    val = float(m.group(1))
    return round(val / 1024 if m.group(2) == "M" else val, 3)


def parse_free_pct(text):
    """`memory_pressure` -> system-wide free percentage (lower = more pressure)."""
    m = re.search(r"System-wide memory free percentage:\s*(\d+)%", text or "")
    return int(m.group(1)) if m else None


def parse_power(text):
    """`pmset -g batt` -> 'ac' | 'battery' | None."""
    m = re.search(r"Now drawing from '([^']+)'", text or "")
    if not m:
        return None
    return "ac" if "AC" in m.group(1) else "battery"


def parse_cpu_speed_limit(text):
    """`pmset -g therm` -> CPU_Speed_Limit percent (100 = not throttled), None if unreported."""
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def parse_top_cpu(text, own_pid=None, n=5, min_pct=1.0):
    """`ps -Ao pid=,pcpu=,comm=` lines -> the busiest other processes, by name.

    pcpu is a percent of ONE core, so 100 means one core saturated, not the machine.
    The benchmark's own process is excluded.
    """
    rows = []
    for line in (text or "").splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid, pcpu = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if own_pid is not None and pid == own_pid:
            continue
        if pcpu >= min_pct:
            rows.append({"proc": os.path.basename(parts[2]), "cpu_pct": pcpu})
    rows.sort(key=lambda r: r["cpu_pct"], reverse=True)
    return rows[:n]


def snapshot():
    """One read-only look at the machine. Costs a few short subprocess calls."""
    try:
        load1 = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):
        load1 = None
    return {
        "swap_used_gb": parse_swap(_run(["sysctl", "-n", "vm.swapusage"])),
        "mem_free_pct": parse_free_pct(_run(["memory_pressure"])),
        "power": parse_power(_run(["pmset", "-g", "batt"])),
        "cpu_speed_limit_pct": parse_cpu_speed_limit(_run(["pmset", "-g", "therm"])),
        "load_1m": load1,
        "top_cpu": parse_top_cpu(_run(["ps", "-Ao", "pid=,pcpu=,comm=", "-r"]), os.getpid()),
    }


def warnings(snap):
    """Human reasons a row taken under this snapshot deserves a caveat."""
    out = []
    if snap.get("power") == "battery":
        out.append("on battery: macOS may cap performance")
    if (snap.get("cpu_speed_limit_pct") or 100) < 100:
        out.append(f"thermal throttle: CPU speed limit {snap['cpu_speed_limit_pct']}%")
    if (snap.get("swap_used_gb") or 0) > WARN_SWAP_GB:
        out.append(f"swap in use: {snap['swap_used_gb']} GB (the model may be paging)")
    free = snap.get("mem_free_pct")
    if free is not None and free < WARN_FREE_PCT:
        out.append(f"memory free {free}%: under pressure")
    busy = [p for p in snap.get("top_cpu", []) if p["cpu_pct"] >= WARN_OTHER_CPU_PCT]
    if busy:
        out.append("busy: " + ", ".join(f"{p['proc']} {p['cpu_pct']:.0f}%" for p in busy))
    return out


def wired_limit_mb():
    """`sysctl iogpu.wired_limit_mb`; 0 means the OS default cap. None if the key is absent."""
    text = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    try:
        return int(text.strip())
    except (AttributeError, ValueError):
        return None


def main():
    snap = snapshot()
    snap["gpu_wired_limit_mb"] = wired_limit_mb()
    print(json.dumps(snap, indent=2))
    warn = warnings(snap)
    print("\nQUIET" if not warn else "\nCAVEATS:")
    for w in warn:
        print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
