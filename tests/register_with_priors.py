"""
register_with_priors.py — Re-register all models with accurate accuracy priors.

Workflow:
    1. python tests/eval_all_models.py --dataset datasets/hf_1000.json \\
           --output results/eval_matrix.csv --node2-host <NODE2>
    2. python tests/extract_priors.py  --eval-matrix results/eval_matrix.csv \\
           --output results/priors.json
    3. python tests/register_with_priors.py --priors results/priors.json \\
           --node2-host <NODE2>

Usage:
    python tests/register_with_priors.py \\
        --priors      results/priors.json \\
        --router-url  http://localhost:8080 \\
        --node2-host  sophia-gpu-09
"""
from __future__ import annotations

import argparse
import json

import httpx


# ---------------------------------------------------------------------------
# Node 1 models (always registered)
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
        "model_id":   "deepseek-r1-7b",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "backend":    "vllm",
        "base_url":   "http://localhost:8001",
        "domains":    ["math", "reasoning", "code"],
        "min_accuracy_capability": {
            "math":      0.82,
            "reasoning": 0.80,
            "code":      0.75,
        },
    },
    {
        "model_id":   "coder-32b",
        "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "backend":    "vllm",
        "base_url":   "http://localhost:8002",
        "domains":    ["code", "math", "reasoning"],
        "min_accuracy_capability": {
            "code":      0.90,
            "math":      0.88,
            "reasoning": 0.88,
        },
    },
    {
        "model_id":   "gemma-3-27b",
        "model_name": "google/gemma-3-27b-it",
        "backend":    "vllm",
        "base_url":   "http://localhost:8003",
        "domains":    ["factual", "reasoning", "creative", "math", "code"],
        "min_accuracy_capability": {"_default": 0.83},
    },
    {
        "model_id":   "deepseek-r1-14b",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "backend":    "vllm",
        "base_url":   "http://localhost:8004",
        "domains":    ["math", "reasoning", "code"],
        "min_accuracy_capability": {
            "math":      0.90,
            "reasoning": 0.88,
            "code":      0.82,
        },
    },
    # llama4-scout lives on node 2 -- base_url set dynamically via --node2-host
]

NODE2_MODELS = [
    {
        "model_id":   "llama4-scout",
        "model_name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "backend":    "vllm",
        "base_url":   "",   # filled at runtime from --node2-host
        "domains":    ["factual", "reasoning", "creative", "math", "code"],
        "min_accuracy_capability": {"_default": 0.88},
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
    parser.add_argument("--priors",     required=True,
                        help="Path to priors.json from extract_priors.py")
    parser.add_argument("--router-url", default="http://localhost:8080",
                        help="Router base URL (default: http://localhost:8080)")
    parser.add_argument("--node2-host", default=None,
                        help="Hostname of node 2 for llama4-scout registration (e.g. sophia-gpu-09)")
    args = parser.parse_args()

    if args.node2_host:
        NODE2_MODELS[0]["base_url"] = f"http://{args.node2_host}:8005"
        MODELS.extend(NODE2_MODELS)

    register_all(args.priors, args.router_url)


if __name__ == "__main__":
    main()
