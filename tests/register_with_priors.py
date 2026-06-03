"""
register_with_priors.py — Re-register all models with accurate accuracy priors
extracted from eval_matrix.csv by extract_priors.py.

This replaces the initial static guesses with real observed accuracy,
eliminating the warm-up period where the router makes poor routing decisions.

Workflow:
    1. python tests/eval_all_models.py --dataset datasets/hf_1000.json \\
           --output results/eval_matrix.csv
    2. python tests/extract_priors.py  --eval-matrix results/eval_matrix.csv \\
           --output results/priors.json
    3. python tests/register_with_priors.py --priors results/priors.json

Usage:
    python tests/register_with_priors.py \\
        --priors      results/priors.json \\
        --router-url  http://localhost:8080
"""
from __future__ import annotations

import argparse
import json

import httpx


# ---------------------------------------------------------------------------
# Model definitions — edit to match your Sophia setup
# ---------------------------------------------------------------------------
MODELS = [
    {
        "model_id":   "qwen-7b",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "backend":    "vllm",
        "base_url":   "http://localhost:8000",
        "domains":    ["factual", "creative", "reasoning"],
        "min_accuracy_capability": {
            "factual":   0.72,
            "creative":  0.72,
            "reasoning": 0.70,
        },
    },
    {
        "model_id":   "qwen-14b",
        "model_name": "Qwen/Qwen2.5-14B-Instruct",
        "backend":    "vllm",
        "base_url":   "http://localhost:8001",
        "domains":    ["factual", "reasoning", "creative"],
        "min_accuracy_capability": {
            "factual":   0.80,
            "reasoning": 0.78,
            "creative":  0.78,
        },
    },
    {
        "model_id":   "deepseek-r1-7b",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "backend":    "vllm",
        "base_url":   "http://localhost:8002",
        "domains":    ["math", "reasoning"],
        "min_accuracy_capability": {
            "math":      0.82,
            "reasoning": 0.80,
        },
    },
    {
        "model_id":   "coder-32b",
        "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "backend":    "vllm",
        "base_url":   "http://localhost:8003",
        "domains":    ["code", "math", "reasoning"],
        "min_accuracy_capability": {
            "code":      0.90,
            "math":      0.88,
            "reasoning": 0.88,
        },
    },
    {
        "model_id":   "deepseek-v2-lite",
        "model_name": "deepseek-ai/DeepSeek-V2-Lite",
        "backend":    "vllm",
        "base_url":   "http://localhost:8005",
        "domains":    ["factual", "reasoning", "math"],
        "min_accuracy_capability": {
            "factual":   0.75,
            "reasoning": 0.78,
            "math":      0.80,
        },
    },
]


def register_all(priors_path: str, router_url: str) -> None:
    with open(priors_path) as f:
        all_priors: dict[str, dict[str, float]] = json.load(f)

    print(f"\nRegistering {len(MODELS)} models with accurate priors from {priors_path}\n")

    with httpx.Client(timeout=30.0) as client:
        for model in MODELS:
            mid = model["model_id"]
            try:
                r = client.delete(f"{router_url}/router/{mid}")
                if r.status_code in (200, 204, 404):
                    print(f"  Deregistered: {mid}")
            except Exception as e:
                print(f"  Warning: could not deregister {mid}: {e}")

        print()

        for model in MODELS:
            mid    = model["model_id"]
            priors = all_priors.get(mid, {})

            if not priors:
                print(f"  WARNING: No priors found for '{mid}' in {priors_path}")
                print(f"           Registering with static min_accuracy_capability only.")

            payload = {
                **model,
                "accuracy_priors":   priors,
                "skip_calibration":  True,
            }

            r = client.post(
                f"{router_url}/router/register",
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if r.status_code == 201:
                stored = r.json().get("accuracy_priors_stored", {})
                print(f"  ✓ Registered: {mid}")
                print(f"    Priors stored: {len(stored)} keys")
                for key, val in sorted(stored.items()):
                    print(f"      {key:<30} {val:.4f}")
            else:
                print(f"  ✗ Failed to register {mid}: {r.status_code} {r.text}")

            print()

    print("Done. Verify with:")
    print(f"  curl --noproxy '*' {router_url}/v1/models | python -m json.tool")
    print(f"  curl --noproxy '*' {router_url}/router/health")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors",     required=True)
    parser.add_argument("--router-url", default="http://localhost:8080")
    args = parser.parse_args()
    register_all(args.priors, args.router_url)


if __name__ == "__main__":
    main()
