"""
build_dataset.py — Download real benchmark queries from HuggingFace and
build a balanced 1000-request test set for the semantic LLM router.

Sources:
  Factual  : MMLU (57 subjects, graded by difficulty)
  Math     : GSM8K (grade-school, medium) + MATH benchmark (competition, hard)
  Code     : HumanEval (medium) + MBPP (easy)
  Reasoning: LogiQA (medium) + HellaSwag (easy)
  Creative : WritingPrompts (medium/hard)

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
# Dataset loaders — each returns list of {domain, complexity, query, ground_truth}
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
        "anatomy", "astronomy", "economics",
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
                ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
                sample = ds.shuffle(seed=42).select(range(min(per_subject, len(ds))))
                for row in sample:
                    choices = row["choices"]
                    labels  = ["A", "B", "C", "D"]
                    opts    = "\n".join(f"{labels[i]}) {choices[i]}" for i in range(len(choices)))
                    query   = f"{row['question']}\n{opts}"
                    # ground_truth: the correct answer text (not just the letter)
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
    ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        # Answer format: "... #### 36" — extract the number after ####
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
    try:
        ds = load_dataset("lighteval/MATH", "all", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  [MATH] skipped: {e}")
            return []

    ds = ds.shuffle(seed=42).select(range(min(n * 3, len(ds))))
    results = []
    for row in ds:
        level = str(row.get("level", "3")).replace("Level ", "")
        complexity = COMPLEXITY_MAP.get(level, "medium")
        # Extract boxed answer from solution: \boxed{answer}
        solution = row.get("solution", "")
        m = _re.search(r"\\boxed\{([^}]+)\}", solution)
        ground_truth = m.group(1) if m else solution.strip()[:50]
        results.append({
            "domain": "math", "complexity": complexity,
            "query": row["problem"], "ground_truth": ground_truth,
        })

    return random.sample(results, min(n, len(results)))


def load_humaneval(n: int) -> list[dict]:
    """HumanEval: Python function synthesis. Complexity = medium."""
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  [HumanEval] skipped: {e}")
        return []
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        prompt = row["prompt"].strip()
        query  = f"Complete the following Python function:\n\n{prompt}"
        # ground_truth: the canonical test code (contains assert statements)
        ground_truth = row.get("test", "")
        results.append({
            "domain": "code", "complexity": "medium",
            "query": query, "ground_truth": ground_truth,
        })
    return results


def load_mbpp(n: int) -> list[dict]:
    """MBPP: basic Python programming. Complexity = easy."""
    from datasets import load_dataset
    try:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("mbpp", split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  [MBPP] skipped: {e}")
            return []
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        # ground_truth: join assert statements from test_list
        test_list = row.get("test_list", [])
        ground_truth = "\n".join(test_list) if test_list else ""
        results.append({
            "domain": "code", "complexity": "easy",
            "query": f"Write a Python function to: {row['text']}",
            "ground_truth": ground_truth,
        })
    return results


def load_apps(n: int) -> list[dict]:
    """APPS: competitive programming. Complexity = hard."""
    from datasets import load_dataset
    try:
        ds = load_dataset("codeparrot/apps", "all", split="test", trust_remote_code=True)
        ds = ds.filter(lambda x: x.get("difficulty") == "interview")
        ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
        results = []
        for row in ds:
            q = row.get("question", "")[:500]   # cap length
            results.append({"domain": "code", "complexity": "hard", "query": q, "ground_truth": ""})
        return results
    except Exception as e:
        print(f"  [APPS] skipped: {e}")
        return []


def load_logiqa(n: int) -> list[dict]:
    """LogiQA: logical reasoning MCQ. Complexity = medium."""
    from datasets import load_dataset
    try:
        ds = load_dataset("lucasmccabe/logiqa", split="test", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("logiqa", split="validation", trust_remote_code=True)
        except Exception as e:
            print(f"  [LogiQA] skipped: {e}")
            return []
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        context = row.get("context", row.get("passage", ""))
        question = row.get("query",   row.get("question", ""))
        options  = row.get("options", row.get("answers",  []))
        labels   = ["A", "B", "C", "D"]
        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(min(len(options), 4)))
        query    = f"Context: {context}\n\nQuestion: {question}\n{opts_str}"
        # ground_truth: the correct answer text
        ans_idx = row.get("correct_option", row.get("answer", 0))
        ground_truth = options[ans_idx] if isinstance(ans_idx, int) and ans_idx < len(options) else str(ans_idx)
        results.append({"domain": "reasoning", "complexity": "medium", "query": query, "ground_truth": ground_truth})
    return results


def load_hellaswag(n: int) -> list[dict]:
    """HellaSwag: commonsense sentence completion. Complexity = easy."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
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
        # ground_truth: the correct ending text
        label = int(row.get("label", 0))
        ground_truth = row["endings"][label] if label < len(row["endings"]) else ""
        results.append({"domain": "reasoning", "complexity": "easy", "query": query, "ground_truth": ground_truth})
    return results


def load_arc(n: int) -> list[dict]:
    """ARC-Challenge: science MCQ. Easy->medium, Challenge->hard."""
    from datasets import load_dataset
    results = []
    for split_name, complexity, count in [("easy", "easy", n//2), ("challenge", "hard", n - n//2)]:
        try:
            ds = load_dataset("allenai/ai2_arc", f"ARC-{split_name.capitalize()}",
                              split="test", trust_remote_code=True)
            ds = ds.shuffle(seed=42).select(range(min(count, len(ds))))
            for row in ds:
                choices = row["choices"]
                opts    = "\n".join(f"{choices['label'][i]}) {choices['text'][i]}"
                          for i in range(len(choices["label"])))
                query   = f"{row['question']}\n{opts}"
                # ground_truth: correct answer text
                ans_key = row.get("answerKey", "A")
                ans_idx = choices["label"].index(ans_key) if ans_key in choices["label"] else 0
                ground_truth = choices["text"][ans_idx] if ans_idx < len(choices["text"]) else ans_key
                results.append({"domain": "reasoning", "complexity": complexity, "query": query, "ground_truth": ground_truth})
        except Exception as e:
            print(f"  [ARC-{split_name}] skipped: {e}")
    return results


def load_writing_prompts(n: int) -> list[dict]:
    """WritingPrompts: creative writing. Short=medium, long=hard."""
    from datasets import load_dataset
    try:
        ds = load_dataset("euclaise/writingprompts", split="train", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("writing_prompts", split="train", trust_remote_code=True)
        except Exception as e:
            print(f"  [WritingPrompts] skipped: {e}")
            return []
    ds = ds.shuffle(seed=42).select(range(min(n * 2, len(ds))))
    results = []
    for row in ds:
        prompt = row.get("prompt", row.get("story", "")).strip()
        # Remove Reddit formatting markers
        prompt = prompt.replace("[WP]", "").replace("[SP]", "").replace("[EU]", "").strip()
        if len(prompt) < 20 or len(prompt) > 600:
            continue
        complexity = "hard" if len(prompt.split()) > 40 else "medium"
        results.append({"domain": "creative", "complexity": complexity,
                        "query": f"Write a short story or response to this prompt: {prompt}",
                        "ground_truth": ""})
    return random.sample(results, min(n, len(results)))


def load_creative_fallback(n: int) -> list[dict]:
    """Fallback creative prompts if WritingPrompts fails."""
    prompts = [
        ("Write a short story about a robot who discovers music.", "medium"),
        ("Write a poem about the passage of time using ocean imagery.", "medium"),
        ("Create a product description for an AI-powered dream recorder.", "medium"),
        ("Write a dialogue between the sun and the moon.", "easy"),
        ("Write a short story from the perspective of the last tree in a city.", "hard"),
        ("Create a myth explaining why humans dream.", "medium"),
        ("Write a letter from a future version of yourself to your present self.", "medium"),
        ("Write a story about a painter who can only paint tomorrow.", "hard"),
        ("Compose a haiku about artificial intelligence.", "easy"),
        ("Write a short story where the twist is revealed in the last word.", "hard"),
    ]
    results = []
    for prompt, complexity in prompts * (n // len(prompts) + 1):
        results.append({"domain": "creative", "complexity": complexity, "query": prompt, "ground_truth": ""})
    return results[:n]


# ---------------------------------------------------------------------------
# Balance and build final dataset
# ---------------------------------------------------------------------------

TARGET_DISTRIBUTION = {
    ("factual",   "easy"):   80,
    ("factual",   "medium"): 70,
    ("factual",   "hard"):   50,
    ("math",      "easy"):   50,
    ("math",      "medium"): 80,
    ("math",      "hard"):   70,
    ("code",      "easy"):   60,
    ("code",      "medium"): 80,
    ("code",      "hard"):   50,
    ("reasoning", "easy"):   70,
    ("reasoning", "medium"): 80,
    ("reasoning", "hard"):   50,
    ("creative",  "easy"):   40,
    ("creative",  "medium"): 80,
    ("creative",  "hard"):   40,
}
# Total = 950 -> pad to 1000 by sampling


def build(total: int, output: str) -> None:
    print(f"\n  Building {total}-request dataset from HuggingFace benchmarks...")
    print(f"  Output: {output}\n")

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    print("  Loading datasets:")

    all_items: list[dict] = []

    def add(name: str, items: list[dict]) -> None:
        print(f"    {name:<20} {len(items):4} items loaded")
        all_items.extend(items)

    add("MMLU",           load_mmlu(200))
    add("GSM8K",          load_gsm8k(100))
    add("MATH benchmark", load_math_benchmark(100))
    add("HumanEval",      load_humaneval(80))
    add("MBPP",           load_mbpp(80))
    add("APPS",           load_apps(50))
    add("LogiQA",         load_logiqa(100))
    add("HellaSwag",      load_hellaswag(80))
    add("ARC",            load_arc(100))

    wp = load_writing_prompts(150)
    if len(wp) < 30:
        wp = load_creative_fallback(150)
    add("WritingPrompts", wp)

    print(f"\n  Total raw items: {len(all_items)}")

    # Balance by domain/complexity
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in all_items:
        key = (item["domain"], item["complexity"])
        buckets[key].append(item)

    print("\n  Available per bucket:")
    for key in sorted(TARGET_DISTRIBUTION):
        avail = len(buckets[key])
        want  = TARGET_DISTRIBUTION[key]
        status = "ok" if avail >= want else f"only {avail}"
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

    # Pad to reach total if needed
    if len(dataset) < total:
        extra = random.choices(all_items, k=total - len(dataset))
        dataset.extend(extra)

    random.shuffle(dataset)
    dataset = dataset[:total]

    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n  Final dataset: {len(dataset)} requests")
    by_domain: dict[str, int] = defaultdict(int)
    by_cplx:   dict[str, int] = defaultdict(int)
    gt_count = 0
    for item in dataset:
        by_domain[item["domain"]]   += 1
        by_cplx[item["complexity"]] += 1
        if item.get("ground_truth"):
            gt_count += 1

    print("\n  By domain:")
    for d, c in sorted(by_domain.items()):
        print(f"    {d:<12} {c:4}")
    print("\n  By complexity:")
    for c, n2 in sorted(by_cplx.items()):
        print(f"    {c:<8} {n2:4}")
    print(f"\n  Items with ground_truth: {gt_count}/{total}")
    print(f"\n  Saved to: {output}")
    print(f"  Run load test with:")
    print(f"    python tests/load_test.py --dataset {output}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default="datasets/hf_1000.json")
    parser.add_argument("--count",   type=int, default=1000)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    build(args.count, args.output)


if __name__ == "__main__":
    main()
