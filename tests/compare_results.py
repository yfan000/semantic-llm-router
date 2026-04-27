"""
compare_results.py — Side-by-side comparison of semantic router vs round-robin.

Usage:
    python tests/compare_results.py \
        --router   results/router.csv \
        --baseline results/round_robin.csv \
        --electricity-price 0.08

When the router CSV contains multiple modes (cost/eco/accuracy/custom),
a per-mode breakdown is shown at the bottom comparing each mode group
against the round-robin baseline.
"""
from __future__ import annotations
import argparse
import csv
from statistics import mean


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sf(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def bar(fraction: float, width: int = 25) -> str:
    filled = int(min(fraction, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def pct_change(baseline: float, new: float) -> str:
    if baseline == 0:
        return "—"
    delta = (new - baseline) / baseline * 100
    sign  = "+" if delta > 0 else ""
    arrow = "▲" if delta > 0 else "▼"
    return f"{arrow} {sign}{delta:.1f}%"


def summarise(rows: list[dict], electricity_price: float) -> dict:
    ok = [r for r in rows if r.get("status") == "200"]

    wall_ms   = [sf(r["wall_ms"])           for r in ok if sf(r["wall_ms"])           is not None]
    actual_ms = [sf(r["actual_latency_ms"]) for r in ok if sf(r["actual_latency_ms"]) is not None]
    costs     = [sf(r["charged_usd"])       for r in ok if sf(r["charged_usd"])       is not None]
    energies  = [sf(r["energy_j"])          for r in ok if sf(r["energy_j"])          is not None]
    ec_per    = [(e / 3_600_000) * electricity_price for e in energies] if energies else []

    model_counts: dict[str, int] = {}
    for r in ok:
        m = r.get("model_winner") or "unknown"
        model_counts[m] = model_counts.get(m, 0) + 1

    domain_latency: dict[str, list[float]] = {}
    for r in ok:
        v = sf(r["actual_latency_ms"])
        if v:
            domain_latency.setdefault(r["domain"], []).append(v)

    complexity_counts: dict[str, dict] = {}
    for r in ok:
        cplx  = r["complexity"]
        model = r.get("model_winner", "")
        complexity_counts.setdefault(cplx, {})
        complexity_counts[cplx][model] = complexity_counts[cplx].get(model, 0) + 1

    return {
        "total":         len(rows),
        "ok":            len(ok),
        "errors":        len(rows) - len(ok),
        "model_counts":  model_counts,
        "wall_p50":      percentile(wall_ms, 50),
        "wall_p95":      percentile(wall_ms, 95),
        "wall_p99":      percentile(wall_ms, 99),
        "wall_mean":     mean(wall_ms) if wall_ms else 0,
        "actual_p50":    percentile(actual_ms, 50),
        "actual_p95":    percentile(actual_ms, 95),
        "actual_mean":   mean(actual_ms) if actual_ms else 0,
        "total_cost":    sum(costs) if costs else 0,
        "avg_cost":      mean(costs) if costs else 0,
        "total_energy":  sum(energies) if energies else 0,
        "avg_energy":    mean(energies) if energies else 0,
        "total_energy_cost": sum(ec_per) if ec_per else 0,
        "total_combined_cost": (sum(costs) if costs else 0) + (sum(ec_per) if ec_per else 0),
        "domain_latency":    domain_latency,
        "complexity_counts": complexity_counts,
    }


def compare(router_path: str, baseline_path: str, electricity_price: float = 0.10) -> None:
    router_rows   = read_csv(router_path)
    baseline_rows = read_csv(baseline_path)

    R = summarise(router_rows,   electricity_price)
    B = summarise(baseline_rows, electricity_price)

    W = 72
    C1, C2, C3 = 26, 22, 22

    def row(label, r_val, b_val, change_str="", highlight=False):
        mark = " ←" if highlight else ""
        print(f"  {label:<{C1}} {str(r_val):<{C2}} {str(b_val):<{C3}} {change_str}{mark}")

    print(f"\n{'='*W}")
    print(f"  SEMANTIC ROUTER  vs  ROUND-ROBIN — COMPARISON REPORT")
    print(f"{'='*W}")
    print(f"  Router:      {router_path}")
    print(f"  Baseline:    {baseline_path}")
    print(f"  Electricity: ${electricity_price}/kWh")

    # Overview
    print(f"\n{'='*W}")
    print(f"  {'OVERVIEW':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}}")
    print(f"  {'-'*68}")
    row("Total requests", R["total"], B["total"])
    row("Successful",
        f"{R['ok']} ({100*R['ok']//max(R['total'],1)}%)",
        f"{B['ok']} ({100*B['ok']//max(B['total'],1)}%)")
    row("Errors", R["errors"], B["errors"])

    # Routing distribution
    print(f"\n{'='*W}\n  ROUTING DISTRIBUTION\n{'='*W}")
    all_models = sorted(set(list(R["model_counts"]) + list(B["model_counts"])))
    for m in all_models:
        r_cnt = R["model_counts"].get(m, 0)
        b_cnt = B["model_counts"].get(m, 0)
        print(f"  {m:<{C1}}")
        print(f"    Router   {bar(r_cnt/max(R['ok'],1))} {100*r_cnt//max(R['ok'],1):3}%  ({r_cnt})")
        print(f"    Baseline {bar(b_cnt/max(B['ok'],1))} {100*b_cnt//max(B['ok'],1):3}%  ({b_cnt})")

    # Latency
    print(f"\n{'='*W}")
    print(f"  {'LATENCY (ms)':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
    print(f"  {'-'*68}")
    row("Wall P50",  f"{R['wall_p50']:.0f} ms",  f"{B['wall_p50']:.0f} ms",
        pct_change(B["wall_p50"],  R["wall_p50"]),  R["wall_p50"]  < B["wall_p50"])
    row("Wall P95",  f"{R['wall_p95']:.0f} ms",  f"{B['wall_p95']:.0f} ms",
        pct_change(B["wall_p95"],  R["wall_p95"]),  R["wall_p95"]  < B["wall_p95"])
    row("Wall P99",  f"{R['wall_p99']:.0f} ms",  f"{B['wall_p99']:.0f} ms",
        pct_change(B["wall_p99"],  R["wall_p99"]),  R["wall_p99"]  < B["wall_p99"])
    row("Wall mean", f"{R['wall_mean']:.0f} ms", f"{B['wall_mean']:.0f} ms",
        pct_change(B["wall_mean"], R["wall_mean"]), R["wall_mean"] < B["wall_mean"])

    # Latency by domain
    print(f"\n{'='*W}")
    print(f"  {'LATENCY BY DOMAIN (P50)':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
    print(f"  {'-'*68}")
    for domain in sorted(set(list(R["domain_latency"]) + list(B["domain_latency"]))):
        r_vals = R["domain_latency"].get(domain, [])
        b_vals = B["domain_latency"].get(domain, [])
        r_p50  = percentile(r_vals, 50)
        b_p50  = percentile(b_vals, 50)
        row(domain,
            f"{r_p50:.0f} ms" if r_vals else "—",
            f"{b_p50:.0f} ms" if b_vals else "—",
            pct_change(b_p50, r_p50) if r_vals and b_vals else "",
            r_p50 < b_p50 if r_vals and b_vals else False)

    # Routing by complexity
    print(f"\n{'='*W}\n  ROUTING BY COMPLEXITY\n{'='*W}")
    for cplx in ["easy", "medium", "hard"]:
        r_dist  = R["complexity_counts"].get(cplx, {})
        b_dist  = B["complexity_counts"].get(cplx, {})
        r_total = sum(r_dist.values())
        b_total = sum(b_dist.values())
        print(f"  {cplx.upper()} ({r_total} router / {b_total} baseline)")
        for m in all_models:
            rc = r_dist.get(m, 0)
            bc = b_dist.get(m, 0)
            print(f"    {m:<22} Router {100*rc//max(r_total,1):3}% ({rc:4})   "
                  f"Baseline {100*bc//max(b_total,1):3}% ({bc:4})")

    # Cost
    print(f"\n{'='*W}")
    print(f"  {'COST':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
    print(f"  {'-'*68}")
    row("Total inference cost",
        f"${R['total_cost']:.6f}", f"${B['total_cost']:.6f}",
        pct_change(B["total_cost"], R["total_cost"]), R["total_cost"] < B["total_cost"])
    row("Avg per request",
        f"${R['avg_cost']:.8f}", f"${B['avg_cost']:.8f}",
        pct_change(B["avg_cost"], R["avg_cost"]), R["avg_cost"] < B["avg_cost"])

    # Energy
    if R["total_energy"] > 0 or B["total_energy"] > 0:
        print(f"\n{'='*W}")
        print(f"  {'ENERGY':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
        print(f"  {'-'*68}")
        row("Total energy (J)",
            f"{R['total_energy']:.1f} J", f"{B['total_energy']:.1f} J",
            pct_change(B["total_energy"], R["total_energy"]), R["total_energy"] < B["total_energy"])
        row("Avg per request (J)",
            f"{R['avg_energy']:.3f} J", f"{B['avg_energy']:.3f} J",
            pct_change(B["avg_energy"], R["avg_energy"]), R["avg_energy"] < B["avg_energy"])
        row("Energy cost",
            f"${R['total_energy_cost']:.6f}", f"${B['total_energy_cost']:.6f}",
            pct_change(B["total_energy_cost"], R["total_energy_cost"]),
            R["total_energy_cost"] < B["total_energy_cost"])

    # Total combined cost
    print(f"\n{'='*W}")
    print(f"  {'TOTAL COST (inference + energy)':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}}")
    print(f"  {'-'*68}")
    row("Inference", f"${R['total_cost']:.6f}",       f"${B['total_cost']:.6f}")
    row("Energy",    f"${R['total_energy_cost']:.6f}", f"${B['total_energy_cost']:.6f}")
    total_r = R["total_combined_cost"]
    total_b = B["total_combined_cost"]
    print(f"  {'─'*68}")
    row("TOTAL", f"${total_r:.6f}", f"${total_b:.6f}",
        pct_change(total_b, total_r), total_r < total_b)

    if total_b > 0 and total_r < total_b:
        saved = total_b - total_r
        print(f"\n  ✓ Semantic router saved ${saved:.6f} ({(saved/total_b)*100:.1f}%) vs round-robin")
    elif total_r > total_b:
        extra = total_r - total_b
        print(f"\n  ✗ Semantic router cost ${extra:.6f} more ({(extra/total_b)*100:.1f}%) than round-robin")

    # ── Comparison grouped by mode ─────────────────────────────────────────────
    router_ok = [r for r in router_rows if r.get("status") == "200"]
    modes_present = sorted({
        r.get("mode", "") for r in router_ok
        if r.get("mode", "") not in ("", "round_robin")
    })

    if len(modes_present) > 1:
        print(f"\n{'='*W}")
        print(f"  COMPARISON BY MODE  (each router mode vs round-robin baseline)")
        print(f"{'='*W}")

        baseline_ok = [r for r in baseline_rows if r.get("status") == "200"]
        b_wall = [sf(r["wall_ms"])      for r in baseline_ok if sf(r["wall_ms"])      is not None]
        b_cost = [sf(r["charged_usd"])  for r in baseline_ok if sf(r["charged_usd"])  is not None]
        b_nrg  = [sf(r["energy_j"])     for r in baseline_ok if sf(r["energy_j"])     is not None]
        bw_p50 = percentile(b_wall, 50)
        bc_avg = mean(b_cost) if b_cost else 0
        be_avg = mean(b_nrg)  if b_nrg  else 0

        print(f"  Round-robin: wall_P50={bw_p50:.0f}ms  "
              f"avg_cost=${bc_avg:.8f}  avg_energy={be_avg:.3f}J\n")

        for mode in modes_present:
            mr = [r for r in router_ok if r.get("mode") == mode]
            if not mr:
                continue

            m_wall = [sf(r["wall_ms"])      for r in mr if sf(r["wall_ms"])      is not None]
            m_cost = [sf(r["charged_usd"])  for r in mr if sf(r["charged_usd"])  is not None]
            m_nrg  = [sf(r["energy_j"])     for r in mr if sf(r["energy_j"])     is not None]

            # Routing distribution for this mode
            mc2: dict[str, int] = {}
            for r in mr:
                mc2[r["model_winner"] or "?"] = mc2.get(r["model_winner"] or "?", 0) + 1

            print(f"  [{mode.upper()} mode]  ({len(mr)} requests)")
            for model, cnt in sorted(mc2.items(), key=lambda x: -x[1]):
                frac = cnt / len(mr)
                print(f"    {model:<22} {bar(frac, 20)}  {100*frac:5.1f}%  ({cnt})")

            w_p50  = percentile(m_wall, 50) if m_wall else 0
            w_mean = mean(m_wall)            if m_wall else 0
            c_avg  = mean(m_cost)            if m_cost else 0
            e_avg  = mean(m_nrg)             if m_nrg  else 0

            print(f"    Latency  P50={w_p50:.0f}ms  mean={w_mean:.0f}ms  "
                  f"  vs baseline {bw_p50:.0f}ms  {pct_change(bw_p50, w_p50)}")
            print(f"    Cost     avg=${c_avg:.8f}"
                  f"  vs baseline ${bc_avg:.8f}  {pct_change(bc_avg, c_avg)}")
            print(f"    Energy   avg={e_avg:.3f}J"
                  f"  vs baseline {be_avg:.3f}J  {pct_change(be_avg, e_avg)}")
            print()

    print(f"{'='*W}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router",            required=True, help="Semantic router CSV")
    parser.add_argument("--baseline",          required=True, help="Round-robin CSV")
    parser.add_argument("--electricity-price", type=float, default=0.10)
    args = parser.parse_args()
    compare(args.router, args.baseline, args.electricity_price)


if __name__ == "__main__":
    main()
