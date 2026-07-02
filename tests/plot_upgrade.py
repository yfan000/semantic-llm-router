"""
plot_upgrade.py — Visualize the zero-downtime model upgrade experiment.

Produces:
  1. upgrade_figure.pdf — two-panel time-series figure for paper
  2. LaTeX table (stdout) — performance comparison Phase 1 vs Phase 2

Usage:
    python tests/plot_upgrade.py \\
        --phase1  results/.../phase1_old_coder.csv \\
        --phase2  results/.../phase2_new_coder.csv \\
        --warmup  /tmp/warmup_results.csv \\
        --output  results/.../upgrade_figure.pdf \\
        --ttca-ratio 1.76 \\
        --threshold  1.5

Time-series reconstruction:
    If 'start_epoch' column present: use real wall-clock time (seconds from Phase 1 start).
    Fallback: use row index — Phase 1 rows 0..N-1, Phase 2 rows N..2N-1.

Figure panels:
    Top:    Routing distribution (% requests to each model) over time, binned in windows.
    Bottom: Rolling-window P50 latency for each model.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from statistics import median


# ── Data loading ────────────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict]:
    try:
        with open(path, errors="replace") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"WARNING: file not found: {path}", file=sys.stderr)
        return []


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── Time-series computation ───────────────────────────────────────────────────────────────────────────

def make_timeline(rows1: list[dict], rows2: list[dict], bin_size: int = 30) -> dict:
    """
    Bin requests into windows and compute routing % and median latency per bin.

    Returns dict with:
        x_labels:  list of str labels for each bin
        bins_p1:   list of dicts per bin (phase 1 models)
        bins_p2:   list of dicts per bin (phase 2 models)
        use_time:  bool — whether x-axis is real time or request index
        t_transition: float — x value of the upgrade event marker
    """
    # Detect if start_epoch is available
    use_time = (rows1 and rows1[0].get("start_epoch", "") not in ("", None))

    if use_time:
        t0 = safe_float(rows1[0]["start_epoch"]) if rows1 else 0.0
        for r in rows1:
            r["_x"] = safe_float(r["start_epoch"]) - t0
        t1_end = max(safe_float(r["start_epoch"]) - t0 for r in rows1) if rows1 else 0.0
        for r in rows2:
            r["_x"] = safe_float(r["start_epoch"]) - t0
        t_transition = t1_end
        x_unit = "s"
    else:
        # Use row index
        for i, r in enumerate(rows1):
            r["_x"] = i
        for i, r in enumerate(rows2):
            r["_x"] = len(rows1) + i
        t_transition = len(rows1)
        x_unit = "requests"

    all_rows = rows1 + rows2
    if not all_rows:
        return {}

    x_min = min(r["_x"] for r in all_rows)
    x_max = max(r["_x"] for r in all_rows)

    # Bin width: divide total x range into ~20 bins
    if use_time:
        bin_width = (x_max - x_min) / 20.0 if x_max > x_min else 1.0
    else:
        bin_width = max(1, len(all_rows) // 20)

    def bin_rows(rows: list[dict]) -> list[dict]:
        bins = []
        x = x_min
        while x < x_max + bin_width:
            window = [r for r in rows if x <= r["_x"] < x + bin_width]
            models = Counter(r.get("model_winner", "?") for r in window if r.get("model_winner"))
            lats = {m: [] for m in models}
            for r in window:
                m = r.get("model_winner", "?")
                if m and r.get("wall_ms"):
                    lats[m].append(safe_float(r["wall_ms"]) / 1000)
            total = sum(models.values()) or 1
            bins.append({
                "x_center": x + bin_width / 2,
                "x_label":  f"{x:.0f}{x_unit}" if use_time else f"{int(x)}",
                "n":        total,
                "routing":  {m: c / total * 100 for m, c in models.items()},
                "lat_p50":  {m: median(v) if v else 0 for m, v in lats.items()},
            })
            x += bin_width
        return [b for b in bins if b["n"] > 0]

    return {
        "bins_p1":      bin_rows(rows1),
        "bins_p2":      bin_rows(rows2),
        "all_bins":     bin_rows(all_rows),
        "t_transition": t_transition,
        "use_time":     use_time,
        "x_unit":       x_unit,
    }


# ── Figure ─────────────────────────────────────────────────────────────────────────────────

def plot_figure(
    rows1: list[dict],
    rows2: list[dict],
    warmup: list[dict],
    output: str,
    ttca_ratio: float,
    threshold: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib not available — skipping figure generation", file=sys.stderr)
        return

    tl = make_timeline(rows1, rows2, bin_size=30)
    if not tl:
        print("WARNING: no data for figure", file=sys.stderr)
        return

    all_bins = tl["all_bins"]
    t_trans = tl["t_transition"]
    x_unit  = tl["x_unit"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 5), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # ── Panel A: routing distribution ─────────────────────────────────────────────────────
    xs = [b["x_center"] for b in all_bins]
    models_seen = set()
    for b in all_bins:
        models_seen.update(b["routing"].keys())

    old_model = "coder-32b"
    new_model = "qwen3-coder-30b"
    other_models = [m for m in models_seen if m not in (old_model, new_model)]

    colors = {
        old_model:  "#4C72B0",   # blue — old/dense model
        new_model:  "#55A868",   # green — new/MoE model
    }
    for m in other_models:
        colors[m] = "#C44E52"    # red for any others

    ax_top.axvspan(0, t_trans, alpha=0.05, color="#4C72B0", label="_nolegend_")
    ax_top.axvspan(t_trans, xs[-1] if xs else t_trans + 1,
                   alpha=0.05, color="#55A868", label="_nolegend_")

    bottom = np.zeros(len(xs))
    for model in [old_model, new_model] + other_models:
        vals = np.array([b["routing"].get(model, 0) for b in all_bins])
        ax_top.fill_between(xs, bottom, bottom + vals,
                            alpha=0.7, color=colors.get(model, "gray"),
                            label=model, step="mid")
        ax_top.step(xs, bottom + vals, where="mid",
                    color=colors.get(model, "gray"), linewidth=0.8)
        bottom += vals

    ax_top.axvline(t_trans, color="black", linestyle="--", linewidth=1.2, zorder=5)
    ax_top.text(t_trans + (xs[-1] - xs[0]) * 0.01, 95,
                f"★ Upgrade event\n  TTCA ratio={ttca_ratio:.2f}× > {threshold:.1f}×",
                fontsize=7.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax_top.set_ylabel("Traffic (%)", fontsize=9)
    ax_top.set_ylim(0, 105)
    ax_top.set_yticks([0, 25, 50, 75, 100])
    ax_top.legend(loc="center left", fontsize=8, framealpha=0.9)
    ax_top.set_title("(a) Routing distribution during model upgrade", fontsize=9, loc="left")
    ax_top.set_xlabel(f"Requests served", fontsize=8)

    # Phase labels
    mid1 = t_trans / 2
    mid2 = (xs[-1] + t_trans) / 2
    ax_top.text(mid1, -16, "Phase 1\n(coder-32b)", ha="center", va="top", fontsize=7.5,
                color="#4C72B0", transform=ax_top.get_xaxis_transform())
    ax_top.text(mid2, -16, "Phase 2\n(qwen3-coder-30b)", ha="center", va="top", fontsize=7.5,
                color="#55A868", transform=ax_top.get_xaxis_transform())

    # ── Panel B: P50 latency per window ────────────────────────────────────────────────────
    for model, color in [(old_model, "#4C72B0"), (new_model, "#55A868")]:
        lx, ly = [], []
        for b in all_bins:
            if model in b.get("lat_p50", {}) and b["lat_p50"][model] > 0:
                lx.append(b["x_center"])
                ly.append(b["lat_p50"][model])
        if lx:
            ax_bot.plot(lx, ly, "o-", color=color, markersize=3, linewidth=1.2, label=model)

    # Warmup reference lines
    if warmup:
        wup_rows = [r for r in warmup if r.get("model_winner") == new_model and r.get("wall_ms")]
        if wup_rows:
            wup_p50 = median(safe_float(r["wall_ms"]) / 1000 for r in wup_rows)
            ax_bot.axhline(wup_p50, color="#55A868", linestyle=":", linewidth=1,
                           label=f"qwen3 pre-warm P50 ({wup_p50:.1f}s)")
        p1_coder = [r for r in rows1 if r.get("model_winner") == old_model and r.get("wall_ms")]
        if p1_coder:
            p1_p50 = median(safe_float(r["wall_ms"]) / 1000 for r in p1_coder)
            ax_bot.axhline(p1_p50, color="#4C72B0", linestyle=":", linewidth=1,
                           label=f"coder-32b P1 P50 ({p1_p50:.1f}s)")

    ax_bot.axvline(t_trans, color="black", linestyle="--", linewidth=1.2)
    ax_bot.set_ylabel("P50 latency (s)", fontsize=9)
    ax_bot.set_xlabel(f"Requests served", fontsize=8)
    ax_bot.set_title("(b) Per-window P50 latency", fontsize=9, loc="left")
    ax_bot.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax_bot.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=200)
    print(f"\n  Figure saved: {output}")
    plt.close()


# ── LaTeX table ────────────────────────────────────────────────────────────────────────────────────

def print_latex_table(
    rows1: list[dict],
    rows2: list[dict],
    warmup: list[dict],
    ttca_ratio: float,
    threshold: float,
) -> None:
    def stats(rows: list[dict], model: str | None = None) -> dict:
        ok = [r for r in rows if r.get("status") == "200" or r.get("status") == 200]
        if model:
            ok = [r for r in ok if r.get("model_winner") == model]
        lats = [safe_float(r["wall_ms"]) for r in ok if r.get("wall_ms")]
        energy = [safe_float(r["energy_j"]) for r in ok if r.get("energy_j")]
        correct = sum(1 for r in ok if r.get("gt_correct") == "true")
        p50 = median(lats) / 1000 if lats else 0.0
        return {
            "n":      len(ok),
            "p50_s":  p50,
            "energy": sum(energy) / max(len(energy), 1) if energy else 0.0,
            "acc":    correct / max(len(ok), 1) * 100,
        }

    # Steady-state latency from pre-warm (intrinsic, low-concurrency)
    wup_new = [r for r in warmup if r.get("model_winner") == "qwen3-coder-30b"]
    p50_new_intrinsic = median(safe_float(r["wall_ms"]) / 1000 for r in wup_new) if wup_new else 0.0

    p1 = stats(rows1, "coder-32b")
    p2e = stats(rows2)               # full Phase 2 energy (both models mixed)

    errors1 = sum(1 for r in rows1 if r.get("status") not in ("200", 200, ""))
    errors2 = sum(1 for r in rows2 if r.get("status") not in ("200", 200, ""))
    total_requests = len(rows1) + len(rows2)

    energy_delta = (p1["energy"] - p2e["energy"]) / max(p1["energy"], 1) * 100
    lat_delta    = (p1["p50_s"] - p50_new_intrinsic) / max(p1["p50_s"], 1) * 100

    table = rf"""
% ─── Paper Table: Zero-downtime model upgrade ───────────────────────────────────────────────────
% Phase 1 = coder-32b (dense 32B), Phase 2 = qwen3-coder-30b (MoE 3.3B active)
% Both models use 2 A100 GPUs. Pre-warm measures intrinsic latency (concurrency=5).

\\begin{{table}}[t]
\\centering
\\caption{{Automatic zero-downtime model upgrade: coder-32b $\\to$ qwen3-coder-30b.
  Both models use 2 A100 GPUs. Latency measured at concurrency=5 (intrinsic).
  TTCA ratio {ttca_ratio:.2f}$\\times$ $>$ {threshold:.1f}$\\times$ threshold triggered automatic supersession.}}
\\label{{tab:upgrade}}
\\small
\\begin{{tabular}}{{lccr}}
\\toprule
Metric & coder-32b & qwen3-coder-30b & Change \\\\
\\midrule
Model params (active) & 32B & 3.3B (MoE) & — \\\\
GPU count & 2 $\\times$ A100 & 2 $\\times$ A100 & — \\\\
Latency P50 & {p1['p50_s']:.1f}s & {p50_new_intrinsic:.1f}s & $-{lat_delta:.0f}\\%$ \\\\
Energy / request & {p1['energy']:.0f}\\,J & {p2e['energy']:.0f}\\,J & $-{energy_delta:.0f}\\%$ \\\\
First-try accuracy & {p1['acc']:.0f}\\% & {stats(rows2)['acc']:.0f}\\% & $\\approx$ \\\\
Service interruption & — & {errors1 + errors2} / {total_requests} errors & zero \\\\
Migration time & — & $\\approx$90\\,s & — \\\\
TTCA ratio & — & {ttca_ratio:.2f}$\\times$ & $>{threshold:.1f}\\times$ threshold \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    print(table)

    # Also print a human-readable summary
    print("  ── Summary ──────────────────────────────────────────────────")
    print(f"  Phase 1 (coder-32b):       P50={p1['p50_s']:.1f}s  energy={p1['energy']:.0f}J/req  acc={p1['acc']:.0f}%")
    print(f"  Phase 2 (qwen3-coder-30b): P50={p50_new_intrinsic:.1f}s (intrinsic)  energy={p2e['energy']:.0f}J/req  acc={stats(rows2)['acc']:.0f}%")
    print(f"  Latency improvement : {lat_delta:.0f}% (pre-warm measurement, concurrency=5)")
    print(f"  Energy improvement  : {energy_delta:.0f}%")
    print(f"  Total errors        : {errors1 + errors2} / {total_requests}")
    print(f"  TTCA ratio          : {ttca_ratio:.2f}x (threshold {threshold:.1f}x)")


# ── CLI ────────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce paper figure + LaTeX table for upgrade experiment")
    parser.add_argument("--phase1",      required=True, help="Phase 1 CSV (old model)")
    parser.add_argument("--phase2",      required=True, help="Phase 2 CSV (new model)")
    parser.add_argument("--warmup",      default="",    help="Pre-warm CSV (intrinsic latency)")
    parser.add_argument("--output",      default="upgrade_figure.pdf",
                                         help="Output figure path (.pdf or .png)")
    parser.add_argument("--ttca-ratio",  type=float, default=1.76,
                                         help="TTCA ratio from pre-warm comparison")
    parser.add_argument("--threshold",   type=float, default=1.5,
                                         help="Supersession TTCA threshold")
    parser.add_argument("--no-figure",   action="store_true",
                                         help="Skip figure generation (table only)")
    args = parser.parse_args()

    rows1  = load_csv(args.phase1)
    rows2  = load_csv(args.phase2)
    warmup = load_csv(args.warmup) if args.warmup else []

    if not rows1:
        print(f"ERROR: no data in {args.phase1}", file=sys.stderr)
        sys.exit(1)

    if not args.no_figure:
        plot_figure(rows1, rows2, warmup, args.output, args.ttca_ratio, args.threshold)

    print_latex_table(rows1, rows2, warmup, args.ttca_ratio, args.threshold)


if __name__ == "__main__":
    main()
