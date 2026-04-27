"""
analyze_results.py — Extract and report metrics from load_test CSV output.

Metrics reported:
  - Latency      : wall / actual / TTFT / ITL  (P50, P95, P99, mean) per model
  - Cost         : per-request and total monetary cost
  - Energy cost  : joules converted to kWh x electricity price (configurable)
  - Total cost   : inference cost + energy cost combined
  - Routing      : distribution by model, domain, complexity, mode

Usage:
    python tests/analyze_results.py results/load_test_20260427_143022.csv
    python tests/analyze_results.py results/run1.csv --electricity-price 0.12
    python tests/analyze_results.py results/run1.csv --model qwen2.5-1.5b
"""
from __future__ import annotations
import argparse
import csv
import sys
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


def bar(fraction: float, width: int = 28) -> str:
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


def fmt_ms(v) -> str:
    return f"{float(v):.1f}" if v is not None else "—"


def fmt_usd(v, digits: int = 8) -> str:
    return f"${float(v):.{digits}f}" if v is not None else "—"


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "n":    len(values),
        "mean": mean(values),
        "p50":  percentile(values, 50),
        "p95":  percentile(values, 95),
        "p99":  percentile(values, 99),
        "min":  min(values),
        "max":  max(values),
    }


def print_latency_row(label: str, stats: dict, unit: str = "ms") -> None:
    if not stats:
        print(f"  {label:<24} —")
        return
    print(
        f"  {label:<24} "
        f"P50={fmt_ms(stats['p50'])}{unit}  "
        f"P95={fmt_ms(stats['p95'])}{unit}  "
        f"P99={fmt_ms(stats['p99'])}{unit}  "
        f"mean={fmt_ms(stats['mean'])}{unit}  "
        f"(n={stats['n']})"
    )


def analyse(rows: list[dict], electricity_price: float = 0.10, filter_model: str | None = None) -> None:
    ok   = [r for r in rows if r.get("status") == "200"]
    errs = [r for r in rows if r.get("status") != "200"]

    if filter_model:
        ok = [r for r in ok if r.get("model_winner") == filter_model]
        print(f"\n  [Filtered to model: {filter_model}]")

    W = 70
    print(f"\n{'='*W}\n  LOAD TEST ANALYSIS\n{'='*W}")
    print(f"  Total requests  : {len(rows)}")
    print(f"  Successful      : {len(ok)}  ({100*len(ok)//max(len(rows),1)}%)")
    print(f"  Errors          : {len(errs)}")
    print(f"  Electricity rate: ${electricity_price}/kWh")

    if not ok:
        print("  No successful responses to analyse.")
        return

    models = sorted({r["model_winner"] for r in ok if r["model_winner"]})

    # Routing distribution
    print(f"\n{'='*W}\n  ROUTING DISTRIBUTION\n{'='*W}")
    mc: dict[str, int] = {}
    for r in ok:
        mc[r["model_winner"] or "unknown"] = mc.get(r["model_winner"] or "unknown", 0) + 1
    for model, cnt in sorted(mc.items(), key=lambda x: -x[1]):
        frac = cnt / len(ok)
        print(f"  {model:<22} {bar(frac)}  {100*frac:5.1f}%  ({cnt})")

    # Latency — all requests
    print(f"\n{'='*W}\n  LATENCY  (all successful requests)\n{'='*W}")
    print_latency_row("Wall latency",   latency_stats([sf(r["wall_ms"])           for r in ok if sf(r["wall_ms"])           is not None]))
    print_latency_row("Actual latency", latency_stats([sf(r["actual_latency_ms"]) for r in ok if sf(r["actual_latency_ms"]) is not None]))
    print_latency_row("TTFT",           latency_stats([sf(r.get("ttft_ms",""))    for r in ok if sf(r.get("ttft_ms",""))    is not None]))
    print_latency_row("ITL",            latency_stats([sf(r.get("itl_ms",""))     for r in ok if sf(r.get("itl_ms",""))     is not None and sf(r.get("itl_ms","")) > 0]))

    # Per-model latency
    print(f"\n{'='*W}\n  LATENCY PER MODEL\n{'='*W}")
    for model in models:
        mr = [r for r in ok if r["model_winner"] == model]
        print(f"\n  [{model}]  ({len(mr)} requests)")
        print_latency_row("  Wall",   latency_stats([sf(r["wall_ms"])        for r in mr if sf(r["wall_ms"])        is not None]))
        print_latency_row("  TTFT",   latency_stats([sf(r.get("ttft_ms","")) for r in mr if sf(r.get("ttft_ms","")) is not None]))
        print_latency_row("  ITL",    latency_stats([sf(r.get("itl_ms",""))  for r in mr if sf(r.get("itl_ms",""))  is not None and sf(r.get("itl_ms","")) > 0]))

    # Inference cost
    costs = [sf(r["charged_usd"]) for r in ok if sf(r["charged_usd"]) is not None]
    print(f"\n{'='*W}\n  INFERENCE COST\n{'='*W}")
    if costs:
        print(f"  Total cost      : {fmt_usd(sum(costs), 6)}")
        print(f"  Avg per request : {fmt_usd(mean(costs), 8)}")
        print(f"  P50 per request : {fmt_usd(percentile(costs, 50), 8)}")
        print(f"  P95 per request : {fmt_usd(percentile(costs, 95), 8)}")
        print(f"  Min / Max       : {fmt_usd(min(costs), 8)} / {fmt_usd(max(costs), 8)}")
        print(f"\n  Per-model:")
        for model in models:
            mc2 = [sf(r["charged_usd"]) for r in ok if r["model_winner"] == model and sf(r["charged_usd"]) is not None]
            if mc2:
                print(f"    {model:<22} avg={fmt_usd(mean(mc2),8)}  total={fmt_usd(sum(mc2),6)}")

    # Energy
    energies = [sf(r["energy_j"]) for r in ok if sf(r["energy_j"]) is not None]
    print(f"\n{'='*W}\n  ENERGY\n{'='*W}")
    if energies:
        total_j   = sum(energies)
        total_kwh = total_j / 3_600_000
        ec_per    = [(e / 3_600_000) * electricity_price for e in energies]
        print(f"  Total energy         : {total_j:.2f} J  =  {total_kwh:.8f} kWh")
        print(f"  Avg per request      : {mean(energies):.3f} J")
        print(f"  P95 per request      : {percentile(energies, 95):.3f} J")
        print(f"  Energy cost total    : {fmt_usd(sum(ec_per), 8)}  (@ ${electricity_price}/kWh)")
        print(f"  Energy cost avg/req  : {fmt_usd(mean(ec_per), 10)}")
        print(f"\n  Per-model:")
        for model in models:
            me = [sf(r["energy_j"]) for r in ok if r["model_winner"] == model and sf(r["energy_j"]) is not None]
            if me:
                e_cost = sum(e / 3_600_000 * electricity_price for e in me)
                print(f"    {model:<22} avg={mean(me):.3f} J  total={sum(me):.1f} J  cost={fmt_usd(e_cost,8)}")

    # Total cost
    print(f"\n{'='*W}\n  TOTAL COST  (inference + energy)\n{'='*W}")
    if costs and energies:
        n = min(len(costs), len(energies))
        combined = [costs[i] + (energies[i] / 3_600_000) * electricity_price for i in range(n)]
        inf_total    = sum(costs[:n])
        energy_total = sum(e / 3_600_000 * electricity_price for e in energies[:n])
        print(f"  Inference cost   : {fmt_usd(inf_total, 6)}")
        print(f"  Energy cost      : {fmt_usd(energy_total, 6)}")
        print(f"  {'─'*40}")
        print(f"  TOTAL            : {fmt_usd(inf_total + energy_total, 6)}")
        print(f"  Avg per request  : {fmt_usd(mean(combined), 8)}")

    # TTFT by domain
    print(f"\n{'='*W}\n  TTFT BY DOMAIN\n{'='*W}")
    for domain in sorted({r["domain"] for r in ok}):
        dr = [r for r in ok if r["domain"] == domain]
        tv = [sf(r.get("ttft_ms","")) for r in dr if sf(r.get("ttft_ms","")) is not None]
        if tv:
            print(f"  {domain:<14} P50={fmt_ms(percentile(tv,50))}ms  P95={fmt_ms(percentile(tv,95))}ms  mean={fmt_ms(mean(tv))}ms  (n={len(tv)})")

    # TTFT by complexity
    print(f"\n{'='*W}\n  TTFT BY COMPLEXITY\n{'='*W}")
    for cplx in ["easy", "medium", "hard"]:
        cr = [r for r in ok if r["complexity"] == cplx]
        tv = [sf(r.get("ttft_ms","")) for r in cr if sf(r.get("ttft_ms","")) is not None]
        if tv:
            print(f"  {cplx:<8}  P50={fmt_ms(percentile(tv,50))}ms  P95={fmt_ms(percentile(tv,95))}ms  mean={fmt_ms(mean(tv))}ms  (n={len(tv)})")

    # Errors
    if errs:
        print(f"\n{'='*W}\n  ERRORS\n{'='*W}")
        ec: dict[str, int] = {}
        for r in errs:
            e = r.get("error") or str(r.get("status", "?"))
            ec[e] = ec.get(e, 0) + 1
        for msg, cnt in sorted(ec.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:4}x  {msg[:70]}")

    print(f"\n{'='*W}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    parser.add_argument("--electricity-price", type=float, default=0.10,
                        help="USD per kWh (default: 0.10)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to a specific model_winner")
    args = parser.parse_args()
    rows = read_csv(args.csv_file)
    print(f"\n  Loaded {len(rows)} rows from {args.csv_file}")
    analyse(rows, args.electricity_price, args.model)


if __name__ == "__main__":
    main()
