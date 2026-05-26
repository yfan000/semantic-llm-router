"""
build_dataset.py — Download real benchmark queries from HuggingFace and
build a balanced 1000-request test set for the semantic LLM router.

Sources:
  Factual  : MMLU (57 subjects, graded by difficulty)
  Math     : GSM8K (grade-school, medium) + MATH benchmark (competition, hard)
  Code     : HumanEval (easy/medium split by prompt length) + APPS (hard)
  Reasoning: LogiQA (medium) + HellaSwag (easy) + ARC (easy/hard)

Install dependencies:
    pip install datasets tqdm

Usage:
    python tests/build_dataset.py                    # saves to datasets/hf_1000.json
    python tests/build_dataset.py --output my.json   # custom output path
    python tests/build_dataset.py --count 500        # build 500 instead of 1000

The output JSON can be used directly with load_test.py:
    python tests/load_test.py --dataset datasets/hf_1000.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_mmlu(n: int) -> list[dict]:
    """MMLU: 57 subjects. Map subjects to easy/medium/hard."""
    from datasets import load_dataset

    EASY_SUBJECTS = [
        "high_school_geography", "high_school_us_history", "high_school_world_history",
        "high_school_biology", "high_school_chemistry", "high_school_physics",
        "elementary_mathematics", "nutrition", "sociology",
    ]
    MEDIUM_SUBJECTS = [
        "college_biology", "college_chemistry", "college_computer_science",
        "college_mathematics", "college_physics", "college_medicine",
        "anatomy", "astronomy", "econometrics",
    ]
    HARD_SUBJECTS = [
        "professional_medicine", "professional_law", "professional_accounting",
        "professional_psychology", "medical_genetics", "clinical_knowledge",
        "abstract_algebra", "formal_logic", "logical_fallacies",
    ]

    results = []
    per_difficulty = n // 3

    for subjects, complexity, count in [
        (EASY_SUBJECTS,   "easy",   per_difficulty),
        (MEDIUM_SUBJECTS, "medium", per_difficulty),
        (HARD_SUBJECTS,   "hard",   n - 2 * per_difficulty),
    ]:
        per_subject = max(1, count // len(subjects))
        for subject in subjects:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=False)
                sample = ds.shuffle(seed=42).select(range(min(per_subject, len(ds))))
                for row in sample:
                    choices = row["choices"]
                    labels  = ["A", "B", "C", "D"]
                    opts    = "\n".join(f"{labels[i]}) {choices[i]}" for i in range(len(choices)))
                    query   = f"{row['question']}\n{opts}"
                    ans_idx = row["answer"]
                    ground_truth = choices[ans_idx] if ans_idx < len(choices) else labels[ans_idx]
                    results.append({
                        "domain": "factual", "complexity": complexity,
                        "query": query, "ground_truth": ground_truth,
                    })
            except Exception as e:
                print(f"  [MMLU] skipped {subject}: {e}")

    return random.sample(results, min(n, len(results)))


def load_gsm8k(n: int) -> list[dict]:
    """GSM8K: grade-school math word problems. Complexity = medium."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=False)
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        answer = row["answer"]
        parts = answer.split("####")
        ground_truth = parts[-1].strip().replace(",", "") if len(parts) > 1 else answer.strip()
        results.append({
            "domain": "math", "complexity": "medium",
            "query": row["question"], "ground_truth": ground_truth,
        })
    return results


def load_math_benchmark(n: int) -> list[dict]:
    """MATH benchmark: competition math. Level 1-2=easy, 3-4=medium, 5=hard."""
    from datasets import load_dataset
    import re as _re

    COMPLEXITY_MAP = {"1": "easy", "2": "easy", "3": "medium", "4": "medium", "5": "hard"}
    ds = None
    for name, cfg, split in [
        ("lighteval/MATH", "all", "test"),
        ("lighteval/MATH", None, "test"),
        ("EleutherAI/hendrycks_math", "algebra", "test"),
    ]:
        try:
            ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
            break
        except Exception as e:
            print(f"  [MATH/{name}] skipped: {e}")

    if ds is None:
        print("  [MATH] all sources failed, skipping competition math")
        return []

    ds = ds.shuffle(seed=42).select(range(min(n * 3, len(ds))))
    results = []
    for row in ds:
        level = str(row.get("level", "3")).replace("Level ", "")
        complexity = COMPLEXITY_MAP.get(level, "medium")
        problem = row.get("problem", row.get("question", ""))
        solution = row.get("solution", row.get("answer", ""))
        m = _re.search(r"\\boxed\{([^}]+)\}", solution)
        ground_truth = m.group(1) if m else solution.strip()[:50]
        if problem:
            results.append({
                "domain": "math", "complexity": complexity,
                "query": problem, "ground_truth": ground_truth,
            })

    return random.sample(results, min(n, len(results)))


def load_humaneval(n: int) -> list[dict]:
    """HumanEval: Python function synthesis.

    Splits by prompt length:
      short prompt (<=300 chars) -> easy
      longer prompt              -> medium

    HumanEval includes the function signature in the prompt so models
    know exactly what function name to implement, making scoring reliable.
    MBPP was previously used for code:easy but its tests call a hidden
    function name unknown to the model, yielding near-zero accuracy.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=False)
    except Exception as e:
        print(f"  [HumanEval] skipped: {e}")
        return []
    ds = ds.shuffle(seed=42)
    results = []
    for row in ds:
        prompt = row["prompt"].strip()
        query  = f"Complete the following Python function:\n\n{prompt}"
        ground_truth = row.get("test", "")
        # Short prompts with simple docstrings are easier problems
        complexity = "easy" if len(prompt) <= 300 else "medium"
        results.append({
            "domain": "code", "complexity": complexity,
            "query": query, "ground_truth": ground_truth,
        })
    random.shuffle(results)
    return results[:n]


def load_apps(n: int) -> list[dict]:
    """APPS: competitive programming. Complexity = hard."""
    from datasets import load_dataset
    try:
        ds = load_dataset("codeparrot/apps", "all", split="test", trust_remote_code=False)
        ds = ds.filter(lambda x: x.get("difficulty") == "interview")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        results = []
        for row in ds:
            q = row.get("question", "")[:500]
            results.append({"domain": "code", "complexity": "hard", "query": q})
        return results
    except Exception as e:
        print(f"  [APPS] skipped: {e}")
        return []


def load_logiqa(n: int) -> list[dict]:
    """LogiQA: logical reasoning MCQ. Complexity = medium."""
    from datasets import load_dataset
    try:
        ds = load_dataset("lucasmccabe/logiqa", split="test", trust_remote_code=False)
    except Exception:
        try:
            ds = load_dataset("logiqa", split="validation", trust_remote_code=False)
        except Exception as e:
            print(f"  [LogiQA] skipped: {e}")
            return []
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        context  = row.get("context", row.get("passage", ""))
        question = row.get("query",   row.get("question", ""))
        options  = row.get("options", row.get("answers",  []))
        labels   = ["A", "B", "C", "D"]
        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(min(len(options), 4)))
        query    = f"Context: {context}\n\nQuestion: {question}\n{opts_str}"
        ans_idx  = row.get("correct_option", row.get("answer", 0))
        ground_truth = options[ans_idx] if isinstance(ans_idx, int) and ans_idx < len(options) else str(ans_idx)
        results.append({"domain": "reasoning", "complexity": "medium", "query": query, "ground_truth": ground_truth})
    return results


def load_hellaswag(n: int) -> list[dict]:
    """HellaSwag: commonsense sentence completion. Complexity = easy."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
    except Exception as e:
        print(f"  [HellaSwag] skipped: {e}")
        return []
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        ctx    = row["ctx"].strip()
        labels = ["A", "B", "C", "D"]
        opts   = "\n".join(f"{labels[i]}) {row['endings'][i]}" for i in range(len(row["endings"])))
        query  = f"Choose the best continuation:\n\n{ctx}\n\n{opts}"
        label  = int(row.get("label", 0))
        ground_truth = row["endings"][label] if label < len(row["endings"]) else ""
        results.append({"domain": "reasoning", "complexity": "easy", "query": query, "ground_truth": ground_truth})
    return results


def load_arc(n: int) -> list[dict]:
    """ARC-Challenge: science MCQ. Easy->easy, Challenge->hard."""
    from datasets import load_dataset
    results = []
    for split_name, complexity, count in [("easy", "easy", n//2), ("challenge", "hard", n - n//2)]:
        try:
            ds = load_dataset("allenai/ai2_arc", f"ARC-{split_name.capitalize()}",
                              split="test", trust_remote_code=False)
            ds = ds.shuffle(seed=42).select(range(min(count, len(ds))))
            for row in ds:
                choices = row["choices"]
                opts    = "\n".join(f"{choices['label'][i]}) {choices['text'][i]}"
                          for i in range(len(choices["label"])))
                query   = f"{row['question']}\n{opts}"
                ans_key = row.get("answerKey", "A")
                ans_idx = choices["label"].index(ans_key) if ans_key in choices["label"] else 0
                ground_truth = choices["text"][ans_idx] if ans_idx < len(choices["text"]) else ans_key
                results.append({"domain": "reasoning", "complexity": complexity, "query": query, "ground_truth": ground_truth})
        except Exception as e:
            print(f"  [ARC-{split_name}] skipped: {e}")
    return results


# ---------------------------------------------------------------------------
# Balance and build final dataset
# ---------------------------------------------------------------------------

TARGET_DISTRIBUTION = {
    ("factual",   "easy"):   85,
    ("factual",   "medium"): 85,
    ("factual",   "hard"):   80,
    ("math",      "easy"):   85,
    ("math",      "medium"): 90,
    ("math",      "hard"):   75,
    ("code",      "easy"):   85,
    ("code",      "medium"): 90,
    ("code",      "hard"):   75,
    ("reasoning", "easy"):   85,
    ("reasoning", "medium"): 90,
    ("reasoning", "hard"):   75,
}


def build(total: int, output: str) -> None:
    print(f"\n  Building {total}-request dataset from HuggingFace benchmarks...")
    print(f"  Output: {output}\n")

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    print("  Loading datasets:")
    all_items: list[dict] = []

    def add(name: str, items: list[dict]) -> None:
        print(f"    {name:<20} {len(items):4} items loaded")
        all_items.extend(items)

    add("MMLU",           load_mmlu(260))
    add("GSM8K",          load_gsm8k(150))
    add("MATH benchmark", load_math_benchmark(100))
    add("HumanEval",      load_humaneval(164))  # all 164 items split into easy/medium by prompt length
    add("APPS",           load_apps(75))
    add("LogiQA",         load_logiqa(120))
    add("HellaSwag",      load_hellaswag(100))
    add("ARC",            load_arc(120))

    print(f"\n  Total raw items: {len(all_items)}")

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in all_items:
        key = (item["domain"], item["complexity"])
        buckets[key].append(item)

    print("\n  Available per bucket:")
    for key in sorted(TARGET_DISTRIBUTION):
        avail  = len(buckets[key])
        want   = TARGET_DISTRIBUTION[key]
        status = "OK" if avail >= want else f"only {avail}"
        print(f"    {key[0]:<12} {key[1]:<8} want={want:3}  {status}")

    dataset: list[dict] = []
    for key, want in TARGET_DISTRIBUTION.items():
        pool = buckets[key]
        if len(pool) >= want:
            dataset.extend(random.sample(pool, want))
        else:
            dataset.extend(pool)
            if pool:
                dataset.extend(random.choices(pool, k=want - len(pool)))

    if len(dataset) < total:
        dataset.extend(random.choices(all_items, k=total - len(dataset)))

    random.shuffle(dataset)
    dataset = dataset[:total]

    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n  Final dataset: {len(dataset)} requests")
    by_domain: dict[str, int] = defaultdict(int)
    by_cplx:   dict[str, int] = defaultdict(int)
    for item in dataset:
        by_domain[item["domain"]]   += 1
        by_cplx[item["complexity"]] += 1

    print("\n  By domain:")
    for d, c in sorted(by_domain.items()):
        print(f"    {d:<12} {c:4}")
    print("\n  By complexity:")
    for c, n2 in sorted(by_cplx.items()):
        print(f"    {c:<8} {n2:4}")

    print(f"\n  Saved to: {output}")
    print(f"  Run load test with:")
    print(f"    python tests/load_test.py --dataset {output}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/hf_1000.json")
    parser.add_argument("--count",  type=int, default=1000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    build(args.count, args.output)


if __name__ == "__main__":
    main()
