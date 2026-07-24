"""
plot_sensitivity.py — Figure~\ref{fig:sensitivity}: sensitivity to TTCA α and β.

Produces a two-panel figure:
  Left panel:  accuracy and TTCA-mean vs α  (β=0 fixed)
  Right panel: accuracy and cost/req vs β   (α at paper defaults)

Usage:
    python tests/plot_sensitivity.py \
        --alpha-dir results/alpha_sweep_20260725_120000 \
        --beta-dir  results/beta_sweep_20260725_130000

    # Print LaTeX tables to stdout:
    python tests/plot_sensitivity.py \
        --alpha-dir results/alpha_sweep_... \
        --beta-dir  results/beta_sweep_... \
        --latex

Options:
    --alpha-dir   Directory produced by submit_alpha_sweep.sh
    --beta-dir    Directory produced by submit_beta_sweep.sh
    --output      Base path for output PDF (default: sensitivity.pdf in alpha-dir)
    --latex       Print LaTeX tables (tab:alpha_sweep, tab:beta_sweep) to stdout
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from statistics import mean, quantiles


# ── Data loading ──────────────────────────────────────────────────────────────

def _percentile(vals: list[float], p: int) -> float | None:
    if not vals:
        return None
    if len(vals) < 20:
        vals_s = sorted(vals)
        idx = max(0, int(len(vals_s) * p / 100) - 1)
        return vals_s[idx]
    return quantiles(vals, n=100)[p - 1]


def load_csv_stats(path: str) -> dict | None:
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
    # TTCA-mean: mean latency over correct-only responses
    correct_lats = [float(r.get("actual_latency_ms") or r.get("wall_ms") or 0)
                    for r in correct if r.get("actual_latency_ms") or r.get("wall_ms")]
    return {
        "n":         len(rows),
        "accuracy":  len(correct) / len(scored) * 100 if scored else None,
        "cost":      mean(costs) if costs else None,
        "lat_mean":  mean(lats) / 1000 if lats else None,
        "lat_p95":   (_percentile(lats, 95) or 0) / 1000 if lats else None,
        "ttca_mean": mean(correct_lats) / 1000 if correct_lats else None,
        "ttca_p95":  (_percentile(correct_lats, 95) or 0) / 1000 if correct_lats else None,
    }


def load_sweep_dir(results_dir: str, prefix: str) -> list[dict]:
    """Load alpha_*.csv or beta_*.csv files sorted by their numeric value."""
    points = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith(f"{prefix}_") and fname.endswith("_results.csv"):
            raw = fname[len(prefix) + 1 : -len("_results.csv")]  # e.g. "1_0"
            val = float(raw.replace("_", "."))
            stats = load_csv_stats(os.path.join(results_dir, fname))
            if stats:
                points.append({"val": val, "label": f"{val:.1f}", **stats})
    points.sort(key=lambda x: x["val"])
    return points


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sensitivity(alpha_pts: list, beta_pts: list, output: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: pip install matplotlib")
        sys.exit(1)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))

    ACC_COLOR  = "#1f77b4"
    LAT_COLOR  = "#ff7f0e"
    COST_COLOR = "#2ca02c"

    # ── Left panel: α sweep ───────────────────────────────────────────────────
    if alpha_pts:
        xs   = [p["val"]      for p in alpha_pts]
        accs = [p["accuracy"] for p in alpha_pts]
        ttca = [p["ttca_mean"] if p.get("ttca_mean") is not None else float("nan")
                for p in alpha_pts]
        ttca_p95 = [p["ttca_p95"] if p.get("ttca_p95") is not None else float("nan")
                    for p in alpha_pts]

        ax_a2 = ax_a.twinx()
        l1, = ax_a.plot(xs, accs,    "-o", color=ACC_COLOR,  linewidth=1.8, markersize=6,
                        label="Accuracy (%)")
        l2, = ax_a2.plot(xs, ttca,   "-s", color=LAT_COLOR,  linewidth=1.8, markersize=6,
                         label="TTCA mean (s)", linestyle="--")
        l3, = ax_a2.plot(xs, ttca_p95, "-^", color=LAT_COLOR, linewidth=1.2, markersize=5,
                         label="TTCA P95 (s)", linestyle=":", alpha=0.7)

        # Mark paper default α=1.0
        if any(p["val"] == 1.0 for p in alpha_pts):
            ax_a.axvline(x=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
            ax_a.text(1.0, ax_a.get_ylim()[0] + 1 if ax_a.get_ylim()[0] > 0 else 50,
                      "default\nα=1.0", ha="center", va="bottom", fontsize=7, color="gray")

        ax_a.set_xlabel("Latency exponent α  (β = 0, cost-blind)", fontsize=10)
        ax_a.set_ylabel("Accuracy (%)", color=ACC_COLOR, fontsize=10)
        ax_a2.set_ylabel("TTCA latency (s)",  color=LAT_COLOR, fontsize=10)
        ax_a.tick_params(axis="y", labelcolor=ACC_COLOR, labelsize=8)
        ax_a2.tick_params(axis="y", labelcolor=LAT_COLOR, labelsize=8)
        ax_a.tick_params(axis="x", labelsize=8)
        ax_a.set_title("(a) α sweep", fontsize=11)
        ax_a.grid(True, alpha=0.3)
        ax_a.legend(handles=[l1, l2, l3], fontsize=8, loc="lower left")
    else:
        ax_a.text(0.5, 0.5, "No alpha sweep data\n(run submit_alpha_sweep.sh first)",
                  ha="center", va="center", transform=ax_a.transAxes, fontsize=9, color="gray")
        ax_a.set_title("(a) α sweep", fontsize=11)

    # ── Right panel: β sweep ──────────────────────────────────────────────────
    if beta_pts:
        xs   = [p["val"]      for p in beta_pts]
        accs = [p["accuracy"] for p in beta_pts]
        costs = [p["cost"] if p.get("cost") is not None else float("nan")
                 for p in beta_pts]

        ax_b2 = ax_b.twinx()
        l1, = ax_b.plot(xs, accs,  "-o", color=ACC_COLOR,  linewidth=1.8, markersize=6,
                        label="Accuracy (%)")
        l2, = ax_b2.plot(xs, costs, "-s", color=COST_COLOR, linewidth=1.8, markersize=6,
                         label="Cost/req (USD)", linestyle="--")

        # Mark paper default β=0
        ax_b.axvline(x=0.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax_b.text(0.05, ax_b.get_ylim()[0] + 1 if ax_b.get_ylim()[0] > 0 else 50,
                  "default\nβ=0", ha="left", va="bottom", fontsize=7, color="gray")

        ax_b.set_xlabel("Cost exponent β  (α at paper defaults)", fontsize=10)
        ax_b.set_ylabel("Accuracy (%)", color=ACC_COLOR, fontsize=10)
        ax_b2.set_ylabel("Cost per request (USD)", color=COST_COLOR, fontsize=10)
        ax_b.tick_params(axis="y", labelcolor=ACC_COLOR, labelsize=8)
        ax_b2.tick_params(axis="y", labelcolor=COST_COLOR, labelsize=8)
        ax_b.tick_params(axis="x", labelsize=8)
        ax_b.set_title("(b) β sweep", fontsize=11)
        ax_b.grid(True, alpha=0.3)
        ax_b.legend(handles=[l1, l2], fontsize=8, loc="upper right")
    else:
        ax_b.text(0.5, 0.5, "No beta sweep data\n(run submit_beta_sweep.sh first)",
                  ha="center", va="center", transform=ax_b.transAxes, fontsize=9, color="gray")
        ax_b.set_title("(b) β sweep", fontsize=11)

    plt.suptitle("Sensitivity to TTCA parameters", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output}")
    plt.close()


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def print_latex_alpha(alpha_pts: list) -> None:
    print()
    print(r"% ── Table: alpha sweep (tab:alpha_sweep) ──────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Sensitivity to latency exponent $\alpha$ ($\beta=0$, "
          r"$N=300$). All per-domain $\alpha$ values are set uniformly. "
          r"\textbf{Bold} = paper default.}")
    print(r"\label{tab:alpha_sweep}")
    print(r"\begin{tabular}{crrrrr}")
    print(r"\toprule")
    print(r"$\alpha$ & $N$ & Accuracy (\%) & Lat.\ mean (s) & TTCA mean (s) & TTCA P95 (s) \\")
    print(r"\midrule")
    for p in alpha_pts:
        is_default = abs(p["val"] - 1.0) < 1e-9
        acc   = f"{p['accuracy']:.1f}"    if p.get("accuracy")  is not None else "--"
        lm    = f"{p['lat_mean']:.2f}"    if p.get("lat_mean")  is not None else "--"
        tm    = f"{p['ttca_mean']:.2f}"   if p.get("ttca_mean") is not None else "--"
        tp95  = f"{p['ttca_p95']:.2f}"    if p.get("ttca_p95")  is not None else "--"
        label = p["label"]
        if is_default:
            row = (f"\\textbf{{{label}}} & \\textbf{{{p['n']}}} & "
                   f"\\textbf{{{acc}}} & \\textbf{{{lm}}} & "
                   f"\\textbf{{{tm}}} & \\textbf{{{tp95}}} \\\\")
        else:
            row = f"{label} & {p['n']} & {acc} & {lm} & {tm} & {tp95} \\\\"
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_latex_beta(beta_pts: list) -> None:
    print()
    print(r"% ── Table: beta sweep (tab:beta_sweep) ───────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Sensitivity to cost exponent $\beta$ ($\alpha$ at paper "
          r"defaults, $N=300$). \textbf{Bold} = paper default.}")
    print(r"\label{tab:beta_sweep}")
    print(r"\begin{tabular}{crrrr}")
    print(r"\toprule")
    print(r"$\beta$ & $N$ & Accuracy (\%) & Cost/req (USD) & TTCA mean (s) \\")
    print(r"\midrule")
    for p in beta_pts:
        is_default = abs(p["val"] - 0.0) < 1e-9
        acc  = f"{p['accuracy']:.1f}"   if p.get("accuracy")  is not None else "--"
        cost = f"\\${p['cost']:.6f}"    if p.get("cost")      is not None else "--"
        tm   = f"{p['ttca_mean']:.2f}"  if p.get("ttca_mean") is not None else "--"
        label = p["label"]
        if is_default:
            row = (f"\\textbf{{{label}}} & \\textbf{{{p['n']}}} & "
                   f"\\textbf{{{acc}}} & \\textbf{{{cost}}} & \\textbf{{{tm}}} \\\\")
        else:
            row = f"{label} & {p['n']} & {acc} & {cost} & {tm} \\\\"
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-dir", default="",
                        help="Results directory from submit_alpha_sweep.sh")
    parser.add_argument("--beta-dir",  default="",
                        help="Results directory from submit_beta_sweep.sh")
    parser.add_argument("--output", default="",
                        help="Output PDF path (default: <alpha-dir>/sensitivity.pdf)")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX tables to stdout")
    args = parser.parse_args()

    if not args.alpha_dir and not args.beta_dir:
        parser.error("Provide at least one of --alpha-dir or --beta-dir")

    alpha_pts: list = []
    beta_pts:  list = []

    if args.alpha_dir:
        if not os.path.isdir(args.alpha_dir):
            print(f"ERROR: alpha-dir not found: {args.alpha_dir}")
            sys.exit(1)
        alpha_pts = load_sweep_dir(args.alpha_dir, "alpha")
        print(f"  Loaded {len(alpha_pts)} alpha runs from {args.alpha_dir}")

    if args.beta_dir:
        if not os.path.isdir(args.beta_dir):
            print(f"ERROR: beta-dir not found: {args.beta_dir}")
            sys.exit(1)
        beta_pts = load_sweep_dir(args.beta_dir, "beta")
        print(f"  Loaded {len(beta_pts)} beta runs from {args.beta_dir}")

    # Terminal summary
    if alpha_pts:
        print(f"\n  {'α':>6}  {'N':>5}  {'Accuracy':>9}  {'TTCA mean':>10}  {'TTCA P95':>9}  {'Cost/req':>12}")
        print(f"  {'-'*58}")
        for p in alpha_pts:
            acc  = f"{p['accuracy']:.1f}%"   if p.get("accuracy")  is not None else "     -"
            tm   = f"{p['ttca_mean']:.2f}s"  if p.get("ttca_mean") is not None else "       -"
            tp95 = f"{p['ttca_p95']:.2f}s"   if p.get("ttca_p95")  is not None else "      -"
            cost = f"${p['cost']:.6f}"        if p.get("cost")      is not None else "           -"
            mark = " *" if abs(p["val"] - 1.0) < 1e-9 else "  "
            print(f"  {p['label']:>6}{mark} {p['n']:>5}  {acc:>9}  {tm:>10}  {tp95:>9}  {cost:>12}")

    if beta_pts:
        print(f"\n  {'β':>6}  {'N':>5}  {'Accuracy':>9}  {'TTCA mean':>10}  {'Cost/req':>12}")
        print(f"  {'-'*52}")
        for p in beta_pts:
            acc  = f"{p['accuracy']:.1f}%"   if p.get("accuracy")  is not None else "     -"
            tm   = f"{p['ttca_mean']:.2f}s"  if p.get("ttca_mean") is not None else "       -"
            cost = f"${p['cost']:.6f}"        if p.get("cost")      is not None else "           -"
            mark = " *" if abs(p["val"] - 0.0) < 1e-9 else "  "
            print(f"  {p['label']:>6}{mark} {p['n']:>5}  {acc:>9}  {tm:>10}  {cost:>12}")

    # Plot
    base_dir = args.alpha_dir or args.beta_dir
    out = args.output or os.path.join(base_dir, "sensitivity.pdf")
    print()
    plot_sensitivity(alpha_pts, beta_pts, out)

    if args.latex:
        if alpha_pts:
            print_latex_alpha(alpha_pts)
        if beta_pts:
            print_latex_beta(beta_pts)


if __name__ == "__main__":
    main()
