"""
extract_priors.py -- Extract per-domain x per-complexity accuracy priors
from eval_matrix.csv and output a JSON file ready for model registration.

Workflow:
    1. python tests/eval_all_models.py --dataset datasets/hf_1000.json \\
           --output results/eval_matrix.csv
    2. python tests/extract_priors.py --eval-matrix results/eval_matrix.csv \\
           --output results/priors.json
    3. python tests/register_with_priors.py --priors results/priors.json

Usage:
    python tests/extract_priors.py \\
        --eval-matrix results/eval_matrix.csv \\
        --output      results/priors.json \\
        --threshold   0.7
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DOMAINS      = ["math", "code", "factual", "reasoning", "creative"]
COMPLEXITIES = ["easy", "medium", "hard"]


def extract(eval_matrix: str, output: str, threshold: float) -> None:
    with open(eval_matrix, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Only keep scorable, successful rows
    scorable = [
        r for r in rows
        if r.get("is_correct") in ("true", "false") and str(r.get("status")) == "200"
    ]

    if not scorable:
        print("No scorable rows found. Check that eval_matrix.csv has is_correct column.")
        return

    # Aggregate: model -> domain -> complexity -> [is_correct booleans]
    agg: dict[str, dict[str, dict[str, list[bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in scorable:
        model_id   = r["model_id"]
        domain     = r.get("domain", "").lower()
        complexity = r.get("complexity", "").lower()
        correct    = r["is_correct"] == "true"
        if domain in DOMAINS and complexity in COMPLEXITIES:
            agg[model_id][domain][complexity].append(correct)

    priors: dict[str, dict[str, float]] = {}
    models = sorted(agg.keys())

    print(f"\n{'Model':<30} {'Key':<25} {'Accuracy':>10} {'N':>6}")
    print("-" * 75)

    for model_id in models:
        priors[model_id] = {}
        for domain in DOMAINS:
            for complexity in COMPLEXITIES:
                samples = agg[model_id][domain][complexity]
                if not samples:
                    continue
                acc = sum(samples) / len(samples)
                key = f"{domain}:{complexity}"
                priors[model_id][key] = round(acc, 4)
                print(f"  {model_id:<28} {key:<25} {acc:>10.1%} {len(samples):>6}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(priors, f, indent=2)

    print(f"\nSaved priors for {len(models)} models to: {output}")
    print(f"\nNext step:")
    print(f"  python tests/register_with_priors.py --priors {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-matrix", required=True,
                        help="Path to eval_matrix.csv from eval_all_models.py")
    parser.add_argument("--output",      default="results/priors.json",
                        help="Output path for priors JSON")
    parser.add_argument("--threshold",   type=float, default=0.7,
                        help="Correct answer threshold (default 0.7)")
    args = parser.parse_args()
    extract(args.eval_matrix, args.output, args.threshold)


if __name__ == "__main__":
    main()
