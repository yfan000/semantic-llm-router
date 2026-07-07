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


def parse_events(log_path: str,
                 start_epoch: float = 0.0) -> list[tuple[datetime, str, str]]:
    """Extract (timestamp, UP/DOWN, model_id) from provisioner log.

    start_epoch: Unix timestamp of experiment start.  Events logged before
    this time are skipped — they belong to earlier runs appended to the
    same log file on the same day.
    """
    events: list[tuple[datetime, str, str]] = []
    today = date.today()
    # Earliest allowed wall-clock time (same-day, derived from epoch)
    start_dt = datetime.fromtimestamp(start_epoch) if start_epoch > 0 else None
    for line in open(log_path, errors="replace"):
        # Match lines like: "14:23:07 [INFO] SPIN UP  qwen-7b  reason=initial..."
        m = re.search(r"(\d{2}:\d{2}:\d{2}).*SPIN (UP|DOWN)\s+(\S+)", line)
        if m:
            t = datetime.combine(today, datetime.strptime(m.group(1), "%H:%M:%S").time())
            if start_dt and t < start_dt:
                continue  # stale event from a previous run logged earlier today
            action = m.group(2)
            model  = m.group(3).strip().rstrip(",")
            events.append((t, action, model))
    return events


def compute_gpu_energy(log_path: str, wall_time_s: float,
                       label: str = "",
                       start_epoch: float = 0.0) -> dict:
    """Parse provisioner log → total GPU energy including idle."""
    events = parse_events(log_path, start_epoch=start_epoch)

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


def print_comparison(static: dict, dynamic: dict) -> None:
    """Side-by-side GPU core-hours comparison: static vs dynamic."""
    if not static or not dynamic:
        return

    s_gh  = static["gpu_hours"]
    d_gh  = dynamic["gpu_hours"]
    saved = s_gh - d_gh
    pct   = saved / max(s_gh, 1e-9) * 100

    s_wh  = static["total_wh"]
    d_wh  = dynamic["total_wh"]

    # Node2: llama4-scout (8 GPUs) runs full wall time in both modes.
    # Energy is pure wall-time × TDP — no log parsing needed.
    NODE2_GPUS = 8
    s_node2_wh = (static["wall_time_s"]  * NODE2_GPUS * GPU_TDP_W) / 3600
    d_node2_wh = (dynamic["wall_time_s"] * NODE2_GPUS * GPU_TDP_W) / 3600
    s_total_wh = s_wh + s_node2_wh
    d_total_wh = d_wh + d_node2_wh

    print(f"\n  {'='*70}")
    print(f"  GPU CORE-HOURS COMPARISON: Static vs Dynamic")
    print(f"  {'='*70}")
    print(f"  {'Metric':<32} {'Static':>12}  {'Dynamic':>12}  {'Savings':>12}")
    print(f"  {'-'*68}")
    print(f"  {'Wall time (s)':<32} {static['wall_time_s']:>12.0f}  "
          f"{dynamic['wall_time_s']:>12.0f}  {'':>12}")
    print(f"  {'Avg GPUs active (node1)':<32} {static['avg_gpus']:>12.1f}  "
          f"{dynamic['avg_gpus']:>12.1f}  "
          f"{static['avg_gpus']-dynamic['avg_gpus']:>+12.1f}")
    print(f"  {'Total GPU-hours (node1)':<32} {s_gh:>12.3f}  "
          f"{d_gh:>12.3f}  "
          f"{saved:>+12.3f}  ({pct:.1f}% saved)")
    print(f"  {'Total energy (Wh, node1)':<32} {s_wh:>12.1f}  "
          f"{d_wh:>12.1f}  "
          f"{s_wh-d_wh:>+12.1f}  ({(s_wh-d_wh)/max(s_wh,1)*100:.1f}% saved)")
    print(f"  {'-'*68}")
    # Node2 breakdown
    s_node2_gh = (static["wall_time_s"]  * NODE2_GPUS) / 3600
    d_node2_gh = (dynamic["wall_time_s"] * NODE2_GPUS) / 3600
    print(f"  {'Node2: llama4-scout (8 GPUs)':<32}")
    print(f"  {'  GPU-hours (node2)':<32} {s_node2_gh:>12.3f}  "
          f"{d_node2_gh:>12.3f}  "
          f"{s_node2_gh-d_node2_gh:>+12.3f}  ({(s_node2_gh-d_node2_gh)/max(s_node2_gh,1)*100:.1f}% saved)")
    print(f"  {'  Energy (Wh, node2)':<32} {s_node2_wh:>12.1f}  "
          f"{d_node2_wh:>12.1f}  "
          f"{s_node2_wh-d_node2_wh:>+12.1f}")
    print(f"  {'-'*68}")
    # Combined totals
    total_saved_wh  = s_total_wh - d_total_wh
    total_saved_pct = total_saved_wh / max(s_total_wh, 1e-9) * 100
    s_total_gh = s_gh + s_node2_gh
    d_total_gh = d_gh + d_node2_gh
    print(f"  {'TOTAL (node1 + node2)':<32} {'':>12}  {'':>12}")
    print(f"  {'  Total GPU-hours (both)':<32} {s_total_gh:>12.3f}  "
          f"{d_total_gh:>12.3f}  "
          f"{s_total_gh-d_total_gh:>+12.3f}  ({(s_total_gh-d_total_gh)/max(s_total_gh,1)*100:.1f}% saved)")
    print(f"  {'  Total energy (Wh, both)':<32} {s_total_wh:>12.1f}  "
          f"{d_total_wh:>12.1f}  "
          f"{total_saved_wh:>+12.1f}  ({total_saved_pct:.1f}% saved)")
    print(f"  {'-'*68}")
    print(f"\n  Node1 savings: {pct:.1f}% (from dynamic provisioning decisions)")
    print(f"  Node2 savings: wall-time reduction only ({(s_node2_wh-d_node2_wh)/max(s_node2_wh,1)*100:.1f}%)")
    print(f"  Combined:      dynamic saves {total_saved_pct:.1f}% total GPU energy (both nodes)")
    print(f"  {'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute total GPU energy from provisioner log (including idle)")
    parser.add_argument("--log",   default=None,
                        help="Path to provisioner log (prov_svd_*.log)")
    parser.add_argument("--wall",  type=float, default=None,
                        help="Experiment wall time in seconds")
    parser.add_argument("--label", default="",
                        help="Label for the report (e.g. 'Static' or 'Dynamic')")
    parser.add_argument("--start-epoch", type=float, default=0.0,
                        help="[Single mode] Unix timestamp when this experiment started;"
                             " events before this time are ignored (fixes stale log entries)")
    # Comparison mode: pass both logs at once
    parser.add_argument("--static-log",  default=None,
                        help="[Compare mode] Static provisioner log")
    parser.add_argument("--static-wall", type=float, default=None,
                        help="[Compare mode] Static wall time (s)")
    parser.add_argument("--static-start-epoch", type=float, default=0.0,
                        help="[Compare mode] Unix timestamp when static experiment started")
    parser.add_argument("--dynamic-log",  default=None,
                        help="[Compare mode] Dynamic provisioner log")
    parser.add_argument("--dynamic-wall", type=float, default=None,
                        help="[Compare mode] Dynamic wall time (s)")
    parser.add_argument("--dynamic-start-epoch", type=float, default=0.0,
                        help="[Compare mode] Unix timestamp when dynamic experiment started")
    args = parser.parse_args()

    # Compare mode: both static and dynamic logs provided
    if args.static_log and args.dynamic_log:
        static  = compute_gpu_energy(args.static_log,  args.static_wall or 0,  "Static",
                                     start_epoch=args.static_start_epoch)
        dynamic = compute_gpu_energy(args.dynamic_log, args.dynamic_wall or 0, "Dynamic",
                                     start_epoch=args.dynamic_start_epoch)
        print_report(static)
        print_report(dynamic)
        print_comparison(static, dynamic)
        return

    # Single mode
    if not args.log:
        parser.error("Provide --log (single mode) or --static-log + --dynamic-log (compare mode)")
    result = compute_gpu_energy(args.log, args.wall or 0, args.label,
                                start_epoch=args.start_epoch)
    print_report(result)

    # Machine-readable summary line for scripting
    print(f"\n  SUMMARY: {result.get('label','')} "
          f"gpu_hours={result.get('gpu_hours',0):.3f} "
          f"total_wh={result.get('total_wh',0):.1f} "
          f"avg_gpus={result.get('avg_gpus',0):.1f}")


if __name__ == "__main__":
    main()
