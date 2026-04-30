"""
compare_results.py — Side-by-side comparison of semantic router vs round-robin.

Usage:
    python tests/compare_results.py \
        --router results/run_fix2_fix3.csv \
        --baseline results/round_robin.csv

    python tests/compare_results.py \
        --router results/run_fix2_fix3.csv \
        --baseline results/round_robin.csv \
        --electricity-price 0.08
"""
from __future__ import annotations
import argparse
import ast
import csv
import re
import subprocess
import sys
import tempfile
from statistics import mean


# ---------------------------------------------------------------------------
# Inline accuracy scoring (mirrors score_accuracy.py)
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]


def _score_math(response: str, ground_truth: str):
    pred = _extract_numbers(response)
    true = _extract_numbers(str(ground_truth))
    if not pred or not true:
        return None
    p, t = pred[-1], true[-1]
    if t == 0:
        return 1.0 if abs(p) < 0.01 else 0.0
    return 1.0 if (abs(p - t) / abs(t) < 0.01 or abs(p - t) < 0.01) else 0.0


def _score_code(response: str, ground_truth: str):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
    code = blocks[0].strip() if blocks else response.strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return 0.0
    gt = str(ground_truth)
    if "assert" in gt or "==" in gt:
        test_code = code + "\n" + gt
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(test_code)
            fname = f.name
        try:
            r = subprocess.run([sys.executable, fname], timeout=5, capture_output=True)
            return 1.0 if r.returncode == 0 else 0.0
        except Exception:
            return 0.0
    return 0.5  # syntax valid, no test cases


def _score_keyword(response: str, ground_truth: str):
    if not ground_truth or ground_truth.strip() in ("", "None"):
        return None
    gt_words = set(w.lower() for w in re.findall(r"\b\w{4,}\b", str(ground_truth)))
    if not gt_words:
        return None
    matches = sum(1 for w in gt_words if w in response.lower())
    overlap = matches / len(gt_words)
    return 1.0 if overlap >= 0.5 else overlap


_SCORERS = {
    "math":      _score_math,
    "code":      _score_code,
    "factual":   _score_keyword,
    "reasoning": _score_keyword,
    "creative":  lambda r, _: 1.0 if len(r.split()) >= 20 else 0.0,
}


def score_row(row: dict):
    """Return float accuracy score or None if unscorable."""
    if str(row.get("status", "")) != "200":
        return None
    response = row.get("response_text", "")
    if not response:
        return None
    scorer = _SCORERS.get(row.get("domain", "").lower())
    if scorer is None:
        return None
    return scorer(response, row.get("ground_truth", ""))


def accuracy_stats(rows: list[dict]) -> tuple[float | None, int, int]:
    """Return (mean_accuracy, n_scored, n_total)."""
    scores = [s for r in rows if (s := score_row(r)) is not None]
    return (mean(scores) if scores else None, len(scores), len(rows))


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

    complexity_counts: dict[tuple, dict] = {}
    for r in ok:
        key = (r["complexity"], r.get("model_winner", ""))
        complexity_counts.setdefault(r["complexity"], {})
        complexity_counts[r["complexity"]][r.get("model_winner", "")] = (
            complexity_counts[r["complexity"]].get(r.get("model_winner", ""), 0) + 1
        )

    # SLO pass rate — overall and per domain/complexity
    slo_rows = [r for r in ok if r.get("slo_ms") and r.get("slo_violated") in ("true", "false")]
    slo_pass  = sum(1 for r in slo_rows if r.get("slo_violated") == "false")
    slo_total = len(slo_rows)

    slo_by_domain: dict[str, tuple[int, int]] = {}   # domain -> (pass, total)
    slo_by_complexity: dict[str, tuple[int, int]] = {}
    for r in slo_rows:
        d = r.get("domain", "unknown")
        c = r.get("complexity", "unknown")
        p = 1 if r.get("slo_violated") == "false" else 0
        dp, dt = slo_by_domain.get(d, (0, 0))
        slo_by_domain[d] = (dp + p, dt + 1)
        cp, ct = slo_by_complexity.get(c, (0, 0))
        slo_by_complexity[c] = (cp + p, ct + 1)

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
        "domain_latency":     domain_latency,
        "complexity_counts":  complexity_counts,
        "slo_pass":           slo_pass,
        "slo_total":          slo_total,
        "slo_by_domain":      slo_by_domain,
        "slo_by_complexity":  slo_by_complexity,
    }


def compare(
    router_path: str,
    baseline_path: str,
    electricity_price: float = 0.10,
) -> None:
    router_rows   = read_csv(router_path)
    baseline_rows = read_csv(baseline_path)

    R = summarise(router_rows,   electricity_price)
    B = summarise(baseline_rows, electricity_price)

    W = 72
    C1, C2, C3 = 26, 22, 22   # column widths

    def row(label, r_val, b_val, change_str="", highlight=False):
        mark = " ←" if highlight else ""
        print(f"  {label:<{C1}} {r_val:<{C2}} {b_val:<{C3}} {change_str}{mark}")

    print(f"\n{'='*W}")
    print(f"  SEMANTIC ROUTER  vs  ROUND-ROBIN — COMPARISON REPORT")
    print(f"{'='*W}")
    print(f"  Router:   {router_path}")
    print(f"  Baseline: {baseline_path}")
    print(f"  Electricity: ${electricity_price}/kWh")

    # ── Overview ────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  {'OVERVIEW':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}}")
    print(f"  {'-'*68}")
    row("Total requests",   R["total"],       B["total"])
    row("Successful",       f"{R['ok']} ({100*R['ok']//max(R['total'],1)}%)",
                            f"{B['ok']} ({100*B['ok']//max(B['total'],1)}%)")
    row("Errors",           R["errors"],       B["errors"])

    # ── Routing distribution ─────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  ROUTING DISTRIBUTION")
    print(f"{'='*W}")
    all_models = sorted(set(list(R["model_counts"]) + list(B["model_counts"])))
    for m in all_models:
        r_cnt = R["model_counts"].get(m, 0)
        b_cnt = B["model_counts"].get(m, 0)
        r_pct = 100 * r_cnt // max(R["ok"], 1)
        b_pct = 100 * b_cnt // max(B["ok"], 1)
        print(f"  {m:<{C1}}")
        print(f"    Router   {bar(r_cnt/max(R['ok'],1))} {r_pct:3}%  ({r_cnt})")
        print(f"    Baseline {bar(b_cnt/max(B['ok'],1))} {b_pct:3}%  ({b_cnt})")

    # ── Latency ──────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  {'LATENCY (ms)':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
    print(f"  {'-'*68}")
    row("Wall P50",   f"{R['wall_p50']:.0f} ms",  f"{B['wall_p50']:.0f} ms",
        pct_change(B["wall_p50"],  R["wall_p50"]),  R["wall_p50"] < B["wall_p50"])
    row("Wall P95",   f"{R['wall_p95']:.0f} ms",  f"{B['wall_p95']:.0f} ms",
        pct_change(B["wall_p95"],  R["wall_p95"]),  R["wall_p95"] < B["wall_p95"])
    row("Wall P99",   f"{R['wall_p99']:.0f} ms",  f"{B['wall_p99']:.0f} ms",
        pct_change(B["wall_p99"],  R["wall_p99"]),  R["wall_p99"] < B["wall_p99"])
    row("Wall mean",  f"{R['wall_mean']:.0f} ms", f"{B['wall_mean']:.0f} ms",
        pct_change(B["wall_mean"], R["wall_mean"]),  R["wall_mean"] < B["wall_mean"])

    # ── Latency by domain ────────────────────────────────────────────────────
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

    # ── Routing by complexity ────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  ROUTING BY COMPLEXITY")
    print(f"{'='*W}")
    for cplx in ["easy", "medium", "hard"]:
        r_dist = R["complexity_counts"].get(cplx, {})
        b_dist = B["complexity_counts"].get(cplx, {})
        r_total = sum(r_dist.values())
        b_total = sum(b_dist.values())
        print(f"  {cplx.upper()} ({r_total} router / {b_total} baseline)")
        for m in all_models:
            rc = r_dist.get(m, 0)
            bc = b_dist.get(m, 0)
            rp = 100*rc//max(r_total,1)
            bp = 100*bc//max(b_total,1)
            print(f"    {m:<22} Router {rp:3}% ({rc:4})   Baseline {bp:3}% ({bc:4})")

    # ── Cost ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  {'COST':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
    print(f"  {'-'*68}")
    row("Total inference cost",
        f"${R['total_cost']:.6f}",  f"${B['total_cost']:.6f}",
        pct_change(B["total_cost"], R["total_cost"]),
        R["total_cost"] < B["total_cost"])
    row("Avg per request",
        f"${R['avg_cost']:.8f}",    f"${B['avg_cost']:.8f}",
        pct_change(B["avg_cost"],   R["avg_cost"]),
        R["avg_cost"] < B["avg_cost"])

    # ── Energy ───────────────────────────────────────────────────────────────
    if R["total_energy"] > 0 or B["total_energy"] > 0:
        print(f"\n{'='*W}")
        print(f"  {'ENERGY':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
        print(f"  {'-'*68}")
        row("Total energy (J)",
            f"{R['total_energy']:.1f} J",  f"{B['total_energy']:.1f} J",
            pct_change(B["total_energy"],  R["total_energy"]),
            R["total_energy"] < B["total_energy"])
        row("Avg per request (J)",
            f"{R['avg_energy']:.3f} J",    f"{B['avg_energy']:.3f} J",
            pct_change(B["avg_energy"],    R["avg_energy"]),
            R["avg_energy"] < B["avg_energy"])
        row("Energy cost",
            f"${R['total_energy_cost']:.6f}", f"${B['total_energy_cost']:.6f}",
            pct_change(B["total_energy_cost"], R["total_energy_cost"]),
            R["total_energy_cost"] < B["total_energy_cost"])

    # ── Total combined cost ───────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  {'TOTAL COST (inference + energy)':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}}")
    print(f"  {'-'*68}")
    row("Inference",  f"${R['total_cost']:.6f}",          f"${B['total_cost']:.6f}")
    row("Energy",     f"${R['total_energy_cost']:.6f}",    f"${B['total_energy_cost']:.6f}")
    total_r = R["total_combined_cost"]
    total_b = B["total_combined_cost"]
    print(f"  {'─'*68}")
    row("TOTAL",
        f"${total_r:.6f}", f"${total_b:.6f}",
        pct_change(total_b, total_r),
        total_r < total_b)

    if total_b > 0 and total_r < total_b:
        saved = (total_b - total_r)
        pct   = (total_b - total_r) / total_b * 100
        print(f"\n  Semantic router saved ${saved:.6f} ({pct:.1f}%) vs round-robin")
    elif total_r > total_b:
        extra = (total_r - total_b)
        pct   = (total_r - total_b) / total_b * 100
        print(f"\n  Semantic router cost ${extra:.6f} more ({pct:.1f}%) than round-robin")

    # -- SLO pass rate ---------------------------------------------------------
    if R["slo_total"] > 0 or B["slo_total"] > 0:
        print(f"\n{'='*W}")
        print(f"  {'SLO PASS RATE':<{C1}} {'Semantic Router':<{C2}} {'Round-Robin':<{C3}} {'Change'}")
        print(f"  {'-'*68}")

        r_rate = R["slo_pass"] / max(R["slo_total"], 1)
        b_rate = B["slo_pass"] / max(B["slo_total"], 1)
        row("Overall",
            f"{r_rate*100:.1f}%  ({R['slo_pass']}/{R['slo_total']})",
            f"{b_rate*100:.1f}%  ({B['slo_pass']}/{B['slo_total']})",
            pct_change(b_rate, r_rate),
            r_rate > b_rate)

        # Per domain
        all_domains = sorted(set(list(R["slo_by_domain"]) + list(B["slo_by_domain"])))
        for domain in all_domains:
            rp, rt = R["slo_by_domain"].get(domain, (0, 0))
            bp, bt = B["slo_by_domain"].get(domain, (0, 0))
            r_dr = rp / max(rt, 1)
            b_dr = bp / max(bt, 1)
            row(f"  {domain}",
                f"{r_dr*100:.1f}%  ({rp}/{rt})" if rt else "—",
                f"{b_dr*100:.1f}%  ({bp}/{bt})" if bt else "—",
                pct_change(b_dr, r_dr) if rt and bt else "",
                r_dr > b_dr if rt and bt else False)

        # Per complexity
        print(f"  {'-'*68}")
        for cplx in ["easy", "medium", "hard"]:
            rp, rt = R["slo_by_complexity"].get(cplx, (0, 0))
            bp, bt = B["slo_by_complexity"].get(cplx, (0, 0))
            r_cr = rp / max(rt, 1)
            b_cr = bp / max(bt, 1)
            row(f"  {cplx}",
                f"{r_cr*100:.1f}%  ({rp}/{rt})" if rt else "—",
                f"{b_cr*100:.1f}%  ({bp}/{bt})" if bt else "—",
                pct_change(b_cr, r_cr) if rt and bt else "",
                r_cr > b_cr if rt and bt else False)

    # ── Comparison grouped by mode ────────────────────────────────────────────
    router_ok = [r for r in router_rows if r.get("status") == "200"]
    modes_present = sorted({r.get("mode","") for r in router_ok if r.get("mode","") not in ("","round_robin")})

    if len(modes_present) > 1:   # only show if multiple modes exist
        print(f"\n{'='*W}")
        print(f"  COMPARISON BY MODE  (router mode vs round-robin baseline)")
        print(f"{'='*W}")

        baseline_ok = [r for r in baseline_rows if r.get("status") == "200"]
        b_wall = [sf(r["wall_ms"]) for r in baseline_ok if sf(r["wall_ms"]) is not None]
        b_cost = [sf(r["charged_usd"]) for r in baseline_ok if sf(r["charged_usd"]) is not None]
        b_nrg  = [sf(r["energy_j"])    for r in baseline_ok if sf(r["energy_j"])    is not None]
        bw_p50 = percentile(b_wall, 50) if b_wall else 0
        bc_avg = mean(b_cost) if b_cost else 0
        be_avg = mean(b_nrg)  if b_nrg  else 0

        # Compute baseline accuracy once (used as comparison for every mode)
        b_acc, b_acc_n, b_acc_total = accuracy_stats(baseline_ok)

        b_acc_str = f"{b_acc*100:.1f}%  ({b_acc_n}/{b_acc_total})" if b_acc is not None else "—"
        print(f"  Round-robin baseline: wall P50={bw_p50:.0f}ms  "
              f"avg_cost=${bc_avg:.8f}  avg_energy={be_avg:.3f}J  "
              f"accuracy={b_acc_str}")

        for mode in modes_present:
            mr = [r for r in router_ok if r.get("mode") == mode]
            if not mr:
                continue

            m_wall = [sf(r["wall_ms"])    for r in mr if sf(r["wall_ms"])    is not None]
            m_cost = [sf(r["charged_usd"]) for r in mr if sf(r["charged_usd"]) is not None]
            m_nrg  = [sf(r["energy_j"])    for r in mr if sf(r["energy_j"])    is not None]

            # Routing distribution for this mode
            mc2: dict[str, int] = {}
            for r in mr:
                mc2[r["model_winner"] or "?"] = mc2.get(r["model_winner"] or "?", 0) + 1

            print(f"\n  [{mode.upper()} mode]  ({len(mr)} requests)")
            for model, cnt in sorted(mc2.items(), key=lambda x: -x[1]):
                frac = cnt / len(mr)
                print(f"    {model:<22} {bar(frac, 20)}  {100*frac:5.1f}%  ({cnt})")

            w_p50  = percentile(m_wall, 50) if m_wall else 0
            w_mean = mean(m_wall)            if m_wall else 0
            c_avg  = mean(m_cost)            if m_cost else 0
            e_avg  = mean(m_nrg)             if m_nrg  else 0

            m_acc, m_acc_n, m_acc_total = accuracy_stats(mr)

            print(f"    Latency  P50={w_p50:.0f}ms  mean={w_mean:.0f}ms"
                  f"    vs baseline {bw_p50:.0f}ms  {pct_change(bw_p50, w_p50)}")
            print(f"    Cost     avg=${c_avg:.8f}"
                  f"  vs baseline ${bc_avg:.8f}  {pct_change(bc_avg, c_avg)}")
            print(f"    Energy   avg={e_avg:.3f}J"
                  f"  vs baseline {be_avg:.3f}J  {pct_change(be_avg, e_avg)}")
            if m_acc is not None:
                acc_str = f"{m_acc*100:.1f}%  (scored {m_acc_n}/{m_acc_total})"
                b_ref   = b_acc if b_acc is not None else 0.0
                print(f"    Accuracy avg={acc_str}"
                      f"  vs baseline {b_acc_str}  {pct_change(b_ref, m_acc)}")

    print(f"\n{'='*W}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router",            required=True, help="Semantic router CSV")
    parser.add_argument("--baseline",          required=True, help="Round-robin CSV")
    parser.add_argument("--electricity-price", type=float, default=0.10)
    args = parser.parse_args()
    compare(args.router, args.baseline, args.electricity_price)


if __name__ == "__main__":
    main()
