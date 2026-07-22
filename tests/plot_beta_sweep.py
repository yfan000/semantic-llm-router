"""
plot_beta_sweep.py — Visualize the TTCA beta sweep: accuracy vs cost Pareto frontier.

Produces:
  1. beta_sweep_pareto.pdf — scatter plot for paper (accuracy vs cost/req)
  2. beta_sweep_bars.pdf   — bar chart comparing accuracy and cost side-by-side
  3. LaTeX table (stdout)  — ready to paste into paper/main.tex

Usage:
    python tests/plot_beta_sweep.py --results-dir results/beta_sweep_20260722_123456

    # With extra systems for comparison:
    python tests/plot_beta_sweep.py \\
        --results-dir results/beta_sweep_20260722_123456 \\
        --output results/beta_sweep_20260722_123456/beta_pareto.pdf

Options:
    --results-dir   Directory produced by submit_beta_sweep.sh
    --output        Output PDF path (default: <results-dir>/beta_sweep_pareto.pdf)
    --no-baselines  Skip baseline systems in the plot
    --latex         Print LaTeX table to stdout
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from statistics import mean


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv_stats(path: str) -> dict | None:
    """Load accuracy and cost stats from a results CSV."""
    if not path or not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path)) if r.get("status") == "200"]
    if not rows:
        return None
    scored  = [r for r in rows if r.get("gt_scored") == "true"]
    correct = [r for r in scored if r.get("gt_correct") == "true"]
    costs   = [float(r["charged_usd"]) for r in rows if r.get("charged_usd")]
    lats    = [float(r.get("actual_latency_ms") or r.get("wall_ms") or 0)
               for r in rows if r.get("actual_latency_ms") or r.get("wall_ms")]
    retries = [int(r.get("retries", 0)) for r in rows]
    return {
        "n":        len(rows),
        "accuracy": len(correct) / len(scored) * 100 if scored else None,
        "cost":     mean(costs) if costs else None,
        "lat_mean": mean(lats) / 1000 if lats else None,
        "avg_att":  1 + mean(retries),
    }


def load_sweep(results_dir: str) -> tuple[list, list]:
    """Return (beta_points, baseline_points) each as list of dicts."""
    beta_points = []
    baseline_points = []

    # Auto-detect beta runs from filenames
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("beta_") and fname.endswith("_results.csv"):
            label_raw = fname[len("beta_"):-len("_results.csv")]  # e.g. "0_5"
            beta_val = label_raw.replace("_", ".")
            stats = load_csv_stats(os.path.join(results_dir, fname))
            if stats:
                beta_points.append({"beta": float(beta_val), "label": f"β={beta_val}", **stats})

    beta_points.sort(key=lambda x: x["beta"])

    # Known baseline files
    for name, fname in [
        ("CARROT",         "baseline_carrot.csv"),
        ("Cascade",        "baseline_cascade.csv"),
        ("OmniRouter",     "baseline_omni_router.csv"),
        ("Round-Robin",    "rr_baseline.csv"),
        ("Tier-Opt-Acc",   "baseline_tier_acc_optimal.csv"),
        ("Tier-Opt-Cost",  "baseline_cost_optimal.csv"),
    ]:
        stats = load_csv_stats(os.path.join(results_dir, fname))
        if stats:
            baseline_points.append({"label": name, **stats})

    return beta_points, baseline_points


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Return non-dominated points (higher accuracy AND lower cost)."""
    dominated = set()
    pts = [(p["accuracy"], p["cost"], i) for i, p in enumerate(points)
           if p["accuracy"] is not None and p["cost"] is not None]
    for i, (ai, ci, _) in enumerate(pts):
        for j, (aj, cj, _) in enumerate(pts):
            if i != j and aj >= ai and cj <= ci and (aj > ai or cj < ci):
                dominated.add(i)
                break
    return [points[i] for i, _ in enumerate(points) if i not in dominated]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pareto(beta_points: list, baseline_points: list, output: str,
                show_baselines: bool = True) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Color palette
    BETA_COLOR     = "#1f77b4"   # blue
    BASELINE_COLOR = "#888888"   # gray
    PARETO_COLOR   = "#ff7f0e"   # orange

    # ── Plot beta curve ───────────────────────────────────────────────────────
    valid_betas = [p for p in beta_points
                   if p["accuracy"] is not None and p["cost"] is not None]

    if valid_betas:
        xs = [p["cost"] for p in valid_betas]
        ys = [p["accuracy"] for p in valid_betas]
        ax.plot(xs, ys, "-o", color=BETA_COLOR, zorder=3,
                linewidth=1.5, markersize=7, label="TTCA (β sweep)")

        for p in valid_betas:
            ax.annotate(
                p["label"],
                xy=(p["cost"], p["accuracy"]),
                xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=BETA_COLOR,
            )

    # ── Plot baselines ────────────────────────────────────────────────────────
    if show_baselines:
        MARKERS = {"CARROT": "^", "Cascade": "s", "OmniRouter": "D",
                   "Round-Robin": "x", "Tier-Opt-Acc": "*", "Tier-Opt-Cost": "P"}
        for p in baseline_points:
            if p["accuracy"] is None or p["cost"] is None:
                continue
            m = MARKERS.get(p["label"], "o")
            ax.scatter(p["cost"], p["accuracy"], marker=m, s=80, zorder=4,
                       color=BASELINE_COLOR, label=p["label"])
            ax.annotate(
                p["label"],
                xy=(p["cost"], p["accuracy"]),
                xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=BASELINE_COLOR,
            )

    # ── Mark Pareto-optimal points ────────────────────────────────────────────
    all_pts = valid_betas + (baseline_points if show_baselines else [])
    frontier = pareto_frontier(all_pts)
    frontier_sorted = sorted(frontier, key=lambda p: p["cost"])
    if len(frontier_sorted) >= 2:
        fx = [p["cost"] for p in frontier_sorted]
        fy = [p["accuracy"] for p in frontier_sorted]
        ax.plot(fx, fy, "--", color=PARETO_COLOR, linewidth=1.2, zorder=2,
                label="Pareto frontier", alpha=0.7)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Cost per request (USD)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("TTCA β sweep: Accuracy vs Cost tradeoff", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    # Format x-axis as scientific notation if values are small
    all_costs = [p["cost"] for p in all_pts if p.get("cost")]
    if all_costs and max(all_costs) < 0.01:
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.5f}")
        )
        plt.xticks(rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output}")
    plt.close()


def plot_bars(beta_points: list, baseline_points: list, output: str,
              show_baselines: bool = True) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed.")
        sys.exit(1)

    # Combine systems: baselines first, then betas in order
    systems = []
    if show_baselines:
        for p in baseline_points:
            if p["accuracy"] is not None:
                systems.append(p)
    for p in beta_points:
        if p["accuracy"] is not None:
            systems.append(p)

    if not systems:
        print("  No data to plot bars.")
        return

    labels   = [p["label"] for p in systems]
    accs     = [p["accuracy"] or 0 for p in systems]
    costs_raw = [p["cost"] or 0 for p in systems]

    x = range(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar([i - width/2 for i in x], accs,  width, label="Accuracy (%)",
                    color="#1f77b4", alpha=0.85)
    bars2 = ax2.bar([i + width/2 for i in x], costs_raw, width, label="Cost/req (USD)",
                    color="#ff7f0e", alpha=0.85)

    ax1.set_ylabel("Accuracy (%)", color="#1f77b4", fontsize=11)
    ax2.set_ylabel("Cost per request (USD)", color="#ff7f0e", fontsize=11)
    ax1.set_title("TTCA β sweep: Accuracy and Cost comparison", fontsize=12)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    # Value labels on bars
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, costs_raw):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"${val:.5f}", ha="center", va="bottom", fontsize=6, rotation=90)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output}")
    plt.close()


# ── LaTeX table ───────────────────────────────────────────────────────────────

def print_latex_table(beta_points: list, baseline_points: list) -> None:
    print("\n% ── LaTeX table (paste into paper/main.tex) ─────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{TTCA \(\beta\) sweep: accuracy–cost tradeoff. "
          r"Bold = Pareto-optimal.}")
    print(r"\label{tab:beta-sweep}")
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"System & Accuracy (\%) & Cost/req (USD) \\")
    print(r"\midrule")

    all_pts = baseline_points + beta_points
    frontier_labels = {p["label"] for p in pareto_frontier(all_pts)}

    def row(label, acc, cost):
        a_str = f"{acc:.1f}" if acc is not None else "--"
        c_str = f"\\${cost:.6f}" if cost is not None else "--"
        if label in frontier_labels:
            return f"\\textbf{{{label}}} & \\textbf{{{a_str}}} & \\textbf{{{c_str}}} \\\\"
        return f"{label} & {a_str} & {c_str} \\\\"

    print("% Baselines")
    for p in baseline_points:
        print(row(p["label"], p.get("accuracy"), p.get("cost")))
    print(r"\midrule")
    print("% TTCA beta sweep")
    for p in beta_points:
        print(row(p["label"], p.get("accuracy"), p.get("cost")))
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Directory produced by submit_beta_sweep.sh")
    parser.add_argument("--output", default="",
                        help="Base path for output PDFs (default: <results-dir>/beta_sweep)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Exclude baseline systems from plots")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table to stdout")
    args = parser.parse_args()

    rd = args.results_dir
    if not os.path.isdir(rd):
        print(f"ERROR: results dir not found: {rd}")
        sys.exit(1)

    base = args.output or os.path.join(rd, "beta_sweep")

    beta_points, baseline_points = load_sweep(rd)

    if not beta_points:
        print(f"ERROR: no beta_*_results.csv files found in {rd}")
        sys.exit(1)

    print(f"  Loaded {len(beta_points)} beta runs, {len(baseline_points)} baselines")
    print()

    # Print quick table to terminal
    show = (baseline_points if not args.no_baselines else []) + beta_points
    print(f"  {'System':<20} {'Accuracy':>10} {'Cost/req':>14} {'Lat Mean':>10}")
    print(f"  {'-'*58}")
    for p in show:
        acc_s  = f"{p['accuracy']:.1f}%"  if p.get('accuracy')  is not None else "    -"
        cost_s = f"${p['cost']:.6f}"      if p.get('cost')      is not None else "       -"
        lat_s  = f"{p['lat_mean']:.2f}s"  if p.get('lat_mean')  is not None else "    -"
        print(f"  {p['label']:<20} {acc_s:>10} {cost_s:>14} {lat_s:>10}")

    # Pareto-optimal set
    all_pts = (baseline_points if not args.no_baselines else []) + beta_points
    frontier = pareto_frontier(all_pts)
    print()
    print(f"  Pareto-optimal: {', '.join(p['label'] for p in frontier)}")

    # Generate plots
    print()
    show_baselines = not args.no_baselines
    plot_pareto(beta_points, baseline_points, base + "_pareto.pdf", show_baselines)
    plot_bars(beta_points, baseline_points, base + "_bars.pdf", show_baselines)

    if args.latex:
        print_latex_table(beta_points, baseline_points if show_baselines else [])


if __name__ == "__main__":
    main()
