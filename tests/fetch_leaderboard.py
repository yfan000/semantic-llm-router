"""
fetch_leaderboard.py — Fetch per-benchmark accuracy from public leaderboards
and convert them into min_accuracy_capability dicts for model registration.

Sources (tried in order):
  1. HuggingFace Open LLM Leaderboard v2 dataset (structured JSON per model)
  2. Hardcoded known scores for 20 common small models
  3. Returns None (triggers calibration at registration)

Benchmark to domain/complexity mapping:
  MMLU         -> factual   medium
  MMLU-Pro     -> factual   hard
  ARC-Challenge -> reasoning medium
  HellaSwag    -> reasoning easy
  GSM8K        -> math      medium
  MATH-lvl5    -> math      hard
  HumanEval    -> code      medium
  MBPP         -> code      easy

Usage:
    python tests/fetch_leaderboard.py --model Qwen/Qwen2.5-1.5B-Instruct
    python tests/fetch_leaderboard.py --model microsoft/Phi-3.5-mini-instruct --json
    python tests/fetch_leaderboard.py --list-models
"""
from __future__ import annotations
import argparse
import json
import re

# ---------------------------------------------------------------------------
# Hardcoded scores for well-known models (fraction correct, not percentage)
# Source: HuggingFace model cards + Open LLM Leaderboard (April 2026)
# ---------------------------------------------------------------------------

KNOWN_SCORES: dict[str, dict[str, float]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "mmlu": 0.453, "arc_challenge": 0.372, "hellaswag": 0.497,
        "gsm8k": 0.364, "humaneval": 0.305, "mbpp": 0.342,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "mmlu": 0.601, "arc_challenge": 0.443, "hellaswag": 0.577,
        "gsm8k": 0.713, "humaneval": 0.542, "mbpp": 0.497,
        "mmlu_pro": 0.312, "math": 0.421,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "mmlu": 0.651, "arc_challenge": 0.491, "hellaswag": 0.612,
        "gsm8k": 0.791, "humaneval": 0.610, "mbpp": 0.563,
        "mmlu_pro": 0.378, "math": 0.512,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "mmlu": 0.738, "arc_challenge": 0.571, "hellaswag": 0.673,
        "gsm8k": 0.872, "humaneval": 0.723, "mbpp": 0.641,
        "mmlu_pro": 0.482, "math": 0.631,
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "mmlu": 0.793, "arc_challenge": 0.633, "hellaswag": 0.712,
        "gsm8k": 0.921, "humaneval": 0.793, "mbpp": 0.712,
        "mmlu_pro": 0.561, "math": 0.724,
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "mmlu": 0.831, "arc_challenge": 0.672, "hellaswag": 0.741,
        "gsm8k": 0.942, "humaneval": 0.841, "mbpp": 0.763,
        "mmlu_pro": 0.621, "math": 0.789,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "mmlu": 0.873, "arc_challenge": 0.712, "hellaswag": 0.772,
        "gsm8k": 0.963, "humaneval": 0.872, "mbpp": 0.801,
        "mmlu_pro": 0.671, "math": 0.831,
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "mmlu": 0.681, "arc_challenge": 0.561, "hellaswag": 0.641,
        "gsm8k": 0.821, "humaneval": 0.583, "mbpp": 0.531,
        "mmlu_pro": 0.401, "math": 0.512,
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "mmlu": 0.692, "arc_challenge": 0.573, "hellaswag": 0.651,
        "gsm8k": 0.863, "humaneval": 0.621, "mbpp": 0.561,
        "mmlu_pro": 0.421, "math": 0.554,
    },
    "microsoft/Phi-3-small-8k-instruct": {
        "mmlu": 0.753, "arc_challenge": 0.612, "hellaswag": 0.681,
        "gsm8k": 0.892, "humaneval": 0.671, "mbpp": 0.612,
        "mmlu_pro": 0.481, "math": 0.614,
    },
    "microsoft/Phi-3-medium-4k-instruct": {
        "mmlu": 0.781, "arc_challenge": 0.641, "hellaswag": 0.703,
        "gsm8k": 0.912, "humaneval": 0.721, "mbpp": 0.652,
        "mmlu_pro": 0.511, "math": 0.653,
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "mmlu": 0.473, "arc_challenge": 0.374, "hellaswag": 0.512,
        "gsm8k": 0.442, "humaneval": 0.293, "mbpp": 0.312,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "mmlu": 0.581, "arc_challenge": 0.461, "hellaswag": 0.591,
        "gsm8k": 0.653, "humaneval": 0.421, "mbpp": 0.412,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "mmlu": 0.693, "arc_challenge": 0.582, "hellaswag": 0.681,
        "gsm8k": 0.843, "humaneval": 0.631, "mbpp": 0.582,
        "mmlu_pro": 0.421, "math": 0.512,
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "mmlu": 0.831, "arc_challenge": 0.701, "hellaswag": 0.762,
        "gsm8k": 0.943, "humaneval": 0.801, "mbpp": 0.741,
        "mmlu_pro": 0.601, "math": 0.721,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "mmlu": 0.851, "arc_challenge": 0.712, "hellaswag": 0.771,
        "gsm8k": 0.952, "humaneval": 0.821, "mbpp": 0.761,
        "mmlu_pro": 0.623, "math": 0.743,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "mmlu": 0.632, "arc_challenge": 0.521, "hellaswag": 0.641,
        "gsm8k": 0.741, "humaneval": 0.542, "mbpp": 0.493,
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "mmlu": 0.711, "arc_challenge": 0.641, "hellaswag": 0.722,
        "gsm8k": 0.891, "humaneval": 0.631, "mbpp": 0.601,
        "mmlu_pro": 0.441, "math": 0.571,
    },
    "google/gemma-2-2b-it": {
        "mmlu": 0.531, "arc_challenge": 0.451, "hellaswag": 0.561,
        "gsm8k": 0.612, "humaneval": 0.371, "mbpp": 0.382,
    },
    "google/gemma-2-9b-it": {
        "mmlu": 0.721, "arc_challenge": 0.621, "hellaswag": 0.701,
        "gsm8k": 0.871, "humaneval": 0.641, "mbpp": 0.593,
    },
}

# Benchmark -> (domain, complexity, weight)
BENCHMARK_MAP: list[tuple[str, str, str, float]] = [
    ("mmlu",          "factual",   "medium", 1.0),
    ("mmlu_pro",      "factual",   "hard",   1.0),
    ("arc_challenge", "reasoning", "medium", 1.0),
    ("hellaswag",     "reasoning", "easy",   1.0),
    ("winogrande",    "reasoning", "easy",   0.5),
    ("gsm8k",         "math",      "medium", 1.0),
    ("math",          "math",      "hard",   1.0),
    ("humaneval",     "code",      "medium", 1.0),
    ("mbpp",          "code",      "easy",   1.0),
]

DOMAIN_DEFAULTS: dict[str, float] = {
    "factual": 0.70, "math": 0.60, "code": 0.60, "reasoning": 0.65, "creative": 0.65,
}


def scores_to_capability(benchmark_scores: dict[str, float]) -> dict[str, float]:
    """Convert raw benchmark scores to min_accuracy_capability format."""
    acc: dict[str, dict[str, float]] = {}
    for bench_key, domain, complexity, weight in BENCHMARK_MAP:
        score = benchmark_scores.get(bench_key)
        if score is None:
            continue
        key = f"{domain}:{complexity}"
        if key not in acc:
            acc[key] = {"total": 0.0, "weight": 0.0}
        acc[key]["total"]  += score * weight
        acc[key]["weight"] += weight

    result: dict[str, float] = {
        k: round(v["total"] / v["weight"], 3) for k, v in acc.items()
    }

    # Fill missing slots
    for domain in ["factual", "math", "code", "reasoning", "creative"]:
        for complexity in ["easy", "medium", "hard"]:
            key = f"{domain}:{complexity}"
            if key not in result:
                domain_scores = [v for k, v in result.items() if k.startswith(f"{domain}:")]
                if domain_scores:
                    result[key] = round(
                        min(domain_scores) if complexity == "hard"
                        else max(domain_scores) if complexity == "easy"
                        else sum(domain_scores) / len(domain_scores), 3
                    )
                else:
                    result[key] = DOMAIN_DEFAULTS.get(domain, 0.65)

    result["_default"] = round(min(result.values()), 3)
    return result


def fetch_from_leaderboard(model_id: str) -> dict[str, float] | None:
    """Try to fetch from HuggingFace Open LLM Leaderboard v2."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
        import json as _json

        api = HfApi()
        org, name = (model_id.split("/", 1) if "/" in model_id else ("", model_id))
        REPO = "open-llm-leaderboard/results"

        files = list(api.list_repo_files(REPO, repo_type="dataset"))
        prefix = f"{org}/{name}/"
        model_files = sorted(
            [f for f in files if f.startswith(prefix) and f.endswith(".json")],
            reverse=True,
        )
        if not model_files:
            return None

        path = hf_hub_download(repo_id=REPO, filename=model_files[0], repo_type="dataset")
        with open(path) as f:
            data = _json.load(f)

        results = data.get("results", {})
        METRIC_MAP = {
            "leaderboard_mmlu_pro":     "mmlu_pro",
            "leaderboard_gpqa":         "mmlu_pro",
            "leaderboard_math_hard":    "math",
            "leaderboard_humaneval":    "humaneval",
            "leaderboard_bbh":          "arc_challenge",
            "harness|arc:challenge|25": "arc_challenge",
            "harness|hellaswag|10":     "hellaswag",
            "harness|gsm8k|5":          "gsm8k",
        }
        scores: dict[str, float] = {}
        for mkey, bkey in METRIC_MAP.items():
            if mkey in results:
                val = results[mkey]
                s = val.get("acc_norm,none", val.get("acc,none", val.get("exact_match,none")))
                if s is not None:
                    scores[bkey] = float(s)
        return scores or None
    except Exception as e:
        print(f"  [Leaderboard fetch] {e}")
        return None


def get_accuracy_priors(model_id: str, verbose: bool = True) -> dict[str, float] | None:
    """
    Get min_accuracy_capability for a model.
    Priority: live leaderboard -> hardcoded table -> None (use calibration).
    """
    clean_id = re.sub(r"-(awq|gptq|gguf|bnb|int4|int8|fp8).*$", "", model_id, flags=re.IGNORECASE)

    if verbose:
        print(f"\n  Looking up: {model_id}")

    scores = fetch_from_leaderboard(clean_id)
    source = "Open LLM Leaderboard (live)"

    if not scores:
        scores = KNOWN_SCORES.get(clean_id) or KNOWN_SCORES.get(model_id)
        source = "hardcoded table"

    if not scores:
        if verbose:
            print(f"  Not found — will use calibration.")
        return None

    capability = scores_to_capability(scores)

    if verbose:
        print(f"  Source: {source}")
        print(f"  Raw benchmarks:")
        for k, v in sorted(scores.items()):
            print(f"    {k:<20} {v:.3f}")
        print(f"\n  Accuracy capability:")
        for k, v in sorted(capability.items()):
            bar = "█" * int(v * 20)
            print(f"    {k:<22} {v:.3f}  {bar}")

    return capability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--json",        action="store_true")
    args = parser.parse_args()

    if args.list_models:
        print("\nModels in hardcoded table:\n")
        for mid in sorted(KNOWN_SCORES):
            s = KNOWN_SCORES[mid]
            print(f"  {mid}")
            print(f"    mmlu={s.get('mmlu','—'):.3f}  gsm8k={s.get('gsm8k','—'):.3f}  "
                  f"humaneval={s.get('humaneval','—'):.3f}  math={s.get('math','—')}")
        return

    if not args.model:
        parser.print_help()
        return

    cap = get_accuracy_priors(args.model)
    if cap and args.json:
        print("\n  Registration JSON:")
        print(json.dumps({"min_accuracy_capability": cap}, indent=2))
        short_name = args.model.split("/")[-1].lower()
        print(f"\n  curl -X POST http://localhost:8080/router/register \\")
        print(f'    -H "Content-Type: application/json" \\')
        print(f"    -d '{{")
        print(f'      "model_id": "{short_name}",')
        print(f'      "backend":  "vllm",')
        print(f'      "base_url": "http://localhost:8001",')
        print(f'      "domains":  ["code","math","factual","reasoning","creative"],')
        print(f'      "skip_calibration": true,')
        print(f'      "min_accuracy_capability": {json.dumps(cap)}')
        print(f"    }}'")


if __name__ == "__main__":
    main()
