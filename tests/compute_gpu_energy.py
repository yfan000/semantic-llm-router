"""
compute_gpu_energy.py — Compute total GPU energy from provisioner logs.

Captures idle energy (models loaded but not serving) in addition to serving
energy, giving the true system energy cost of static vs dynamic mode.

Static mode:  all models run entire time → idle GPUs still burn power.
Dynamic mode: only needed models run   → lower total GPU-hours.

Usage:
    python tests/compute_gpu_energy.py \\
        --log   ~/vllm_logs/prov_svd_static_node1.log \\
        --wall  459 \\
        --label "Static"

    python tests/compute_gpu_energy.py \\
        --log   ~/vllm_logs/prov_svd_dynamic_node1.log \\
        --wall  484 \\
        --label "Dynamic"
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime, date


# A100 SXM4 TDP per GPU (W). Used for energy estimation.
# Real power draw varies (250-400W), TDP is the spec maximum.
GPU_TDP_W: float = 400.0

# GPU count per model (must match dynamic_provisioner.py MODEL_CATALOG)
MODEL_GPUS: dict[str, int] = {
    "qwen-7b":          1,
    "deepseek-r1-7b":   1,
    "qwen3-coder-30b":  2,
    "gemma-3-27b":      2,
    "deepseek-r1-14b":  2,
    "coder-32b":        4,
    "llama4-scout":     8,
}


def parse_events(log_path: str) -> list[tuple[datetime, str, str]]:
    """Extract (timestamp, UP/DOWN, model_id) from provisioner log."""
    events: list[tuple[datetime, str, str]] = []
    today = date.today()
    for line in open(log_path, errors="replace"):
        # Match lines like: "14:23:07 [INFO] SPIN UP  qwen-7b  reason=initial..."
        m = re.search(r"(\d{2}:\d{2}:\d{2}).*SPIN (UP|DOWN)\s+(\S+)", line)
        if m:
            t = datetime.combine(today, datetime.strptime(m.group(1), "%H:%M:%S").time())
            action = m.group(2)
            model  = m.group(3).strip().rstrip(",")
            events.append((t, action, model))
    return events


def compute_gpu_energy(log_path: str, wall_time_s: float,
                       label: str = "") -> dict:
    """Parse provisioner log → total GPU energy including idle."""
    events = parse_events(log_path)

    if not events:
        print(f"  WARNING: no SPIN UP/DOWN events found in {log_path}")
        return {}

    running: dict[str, datetime] = {}   # model → spin-up time
    gpu_seconds: float = 0.0
    model_gpu_seconds: dict[str, float] = defaultdict(float)

    # Use first event time as t=0
    t0 = events[0][0]

    for t, action, model in events:
        n_gpus = MODEL_GPUS.get(model, 1)
        if action == "UP":
            running[model] = t
        elif action == "DOWN" and model in running:
            duration = (t - running.pop(model)).total_seconds()
            gpu_seconds += n_gpus * duration
            model_gpu_seconds[model] += n_gpus * duration

    # Models still running at experiment end
    t_end = t0 if not events else events[-1][0]
    elapsed_to_end = (t_end - t0).total_seconds()

    for model, start in running.items():
        n_gpus  = MODEL_GPUS.get(model, 1)
        # Model runs from start until experiment wall time
        duration = wall_time_s - (start - t0).total_seconds()
        duration = max(duration, 0)
        gpu_seconds += n_gpus * duration
        model_gpu_seconds[model] += n_gpus * duration

    total_wh  = (gpu_seconds / 3600) * GPU_TDP_W
    total_kwh = total_wh / 1000
    total_j   = total_wh * 3600

    result = {
        "label":          label or log_path,
        "gpu_seconds":    gpu_seconds,
        "gpu_hours":      gpu_seconds / 3600,
        "total_wh":       total_wh,
        "total_kwh":      total_kwh,
        "total_j":        total_j,
        "model_breakdown": dict(model_gpu_seconds),
        "wall_time_s":    wall_time_s,
        "avg_gpus":       gpu_seconds / max(wall_time_s, 1),
    }

    return result


def print_report(result: dict) -> None:
    if not result:
        return
    label = result["label"]
    print(f"\n  {'='*60}")
    print(f"  GPU Energy Report: {label}")
    print(f"  {'='*60}")
    print(f"  Wall time      : {result['wall_time_s']:.0f}s "
          f"({result['wall_time_s']/60:.1f} min)")
    print(f"  Avg GPUs active: {result['avg_gpus']:.1f}")
    print(f"  Total GPU-hours: {result['gpu_hours']:.3f} h")
    print(f"  Total energy   : {result['total_wh']:.1f} Wh  "
          f"= {result['total_kwh']:.4f} kWh  "
          f"= {result['total_j']:.0f} J")
    print(f"  (at {GPU_TDP_W:.0f}W TDP per GPU)")
    print(f"\n  Per-model GPU-seconds:")
    for model, gs in sorted(result["model_breakdown"].items(),
                            key=lambda x: -x[1]):
        wh = (gs / 3600) * GPU_TDP_W
        pct = gs / max(result["gpu_seconds"], 1) * 100
        print(f"    {model:<28} {gs:8.0f} GPU-s  ({wh:.1f} Wh, {pct:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute total GPU energy from provisioner log (including idle)")
    parser.add_argument("--log",   required=True,
                        help="Path to provisioner log (prov_svd_*.log)")
    parser.add_argument("--wall",  type=float, required=True,
                        help="Experiment wall time in seconds")
    parser.add_argument("--label", default="",
                        help="Label for the report (e.g. 'Static' or 'Dynamic')")
    args = parser.parse_args()

    result = compute_gpu_energy(args.log, args.wall, args.label)
    print_report(result)

    # Machine-readable summary line for scripting
    print(f"\n  SUMMARY: {result.get('label','')} "
          f"gpu_hours={result.get('gpu_hours',0):.3f} "
          f"total_wh={result.get('total_wh',0):.1f} "
          f"avg_gpus={result.get('avg_gpus',0):.1f}")


if __name__ == "__main__":
    main()
