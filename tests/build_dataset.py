"""build_dataset.py

Download real benchmark queries from HuggingFace and build a balanced
1500-request test set for the semantic LLM router.

Sources:
  Factual  : MMLU
  Math     : GSM8K + MATH benchmark
  Code     : HumanEval + BigCodeBench + APPS
  Reasoning: LogiQA + HellaSwag + ARC
  Code     : LiveCodeBench (easy/medium/hard) -- execution-scored, 300 samples

Usage:
    python tests/build_dataset.py
    python tests/build_dataset.py --output my.json
    python tests/build_dataset.py --count 1200
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict


def load_mmlu(n: int) -> list[dict]:
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
        complexity = "easy" if len(prompt) <= 300 else "medium"
        results.append({
            "domain": "code", "complexity": complexity,
            "query": query, "ground_truth": ground_truth,
        })
    random.shuffle(results)
    return results[:n]


def load_bigcodebench(n: int) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("bigcode/bigcodebench", trust_remote_code=True)
            ds = ds["test"] if "test" in ds else list(ds.values())[0]
        except Exception as e:
            print(f"  [BigCodeBench] skipped: {e}")
            return []
    ds = ds.shuffle(seed=42).select(range(min(n * 2, len(ds))))
    results = []
    for row in ds:
        prompt = row.get("complete_prompt", row.get("instruct_prompt", ""))
        if not prompt:
            continue
        difficulty = row.get("difficulty", 3)
        try:
            difficulty = int(difficulty)
        except Exception:
            difficulty = 3
        complexity = "medium" if difficulty <= 2 else "hard"
        test_code = row.get("test", "")
        results.append({
            "domain": "code",
            "complexity": complexity,
            "query": f"Complete the following Python task:\n\n{prompt.strip()}",
            "ground_truth": test_code,
        })
    return results[:n]


def load_apps(n: int) -> list[dict]:
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
        context = row.get("context", row.get("passage", ""))
        question = row.get("query",   row.get("question", ""))
        options  = row.get("options", row.get("answers",  []))
        labels   = ["A", "B", "C", "D"]
        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(min(len(options), 4)))
        query    = f"Context: {context}\n\nQuestion: {question}\n{opts_str}"
        ans_idx = row.get("correct_option", row.get("answer", 0))
        ground_truth = options[ans_idx] if isinstance(ans_idx, int) and ans_idx < len(options) else str(ans_idx)
        results.append({"domain": "reasoning", "complexity": "medium", "query": query, "ground_truth": ground_truth})
    return results


def load_hellaswag(n: int) -> list[dict]:
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
        label = int(row.get("label", 0))
        ground_truth = row["endings"][label] if label < len(row["endings"]) else ""
        results.append({"domain": "reasoning", "complexity": "easy", "query": query, "ground_truth": ground_truth})
    return results


def load_arc(n: int) -> list[dict]:
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


def load_writing_prompts(n: int) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("euclaise/writingprompts", split="train", trust_remote_code=False)
    except Exception:
        try:
            ds = load_dataset("writing_prompts", split="train", trust_remote_code=False)
        except Exception as e:
            print(f"  [WritingPrompts] skipped: {e}")
            return []
    ds = ds.shuffle(seed=42).select(range(min(n * 2, len(ds))))
    results = []
    for row in ds:
        prompt = row.get("prompt", row.get("story", "")).strip()
        prompt = prompt.replace("[WP]", "").replace("[SP]", "").replace("[EU]", "").strip()
        if len(prompt) < 20 or len(prompt) > 600:
            continue
        complexity = "hard" if len(prompt.split()) > 40 else "medium"
        results.append({"domain": "creative", "complexity": complexity,
                        "query": f"Write a short story or response to this prompt: {prompt}"})
    return random.sample(results, min(n, len(results)))


def _strip_html(text: str) -> str:
    import re as _re
    text = _re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                         ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
        text = text.replace(entity, char)
    return _re.sub(r"\s{2,}", " ", text).strip()


def load_livecodebench(n: int) -> list[dict]:
    """LiveCodeBench: competitive programming problems.

    Uses livecodebench/code_generation_lite. Each item stores test cases as
    a JSON string in ground_truth, which triggers the execution scorer.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("livecodebench/code_generation_lite",
                          split="test", trust_remote_code=False)
    except Exception:
        try:
            ds = load_dataset("livecodebench/code_generation_lite",
                              trust_remote_code=False)
            ds = ds["test"] if "test" in ds else list(ds.values())[0]
        except Exception as e:
            print(f"  [LiveCodeBench] skipped: {e}")
            return []

    COMPLEXITY_MAP = {"easy": "easy", "medium": "medium", "hard": "hard"}
    ds = ds.shuffle(seed=42)
    results = []
    for row in ds:
        difficulty = str(row.get("difficulty", "medium")).lower().strip()
        complexity = COMPLEXITY_MAP.get(difficulty, "medium")
        content = _strip_html(row.get("question_content", "")).strip()
        starter = (row.get("starter_code") or "").strip()
        if not content:
            continue
        prompt = f"Solve the following programming problem and provide a complete Python solution:\n\n{content}"
        if starter:
            prompt += f"\n\nUse this function signature:\n```python\n{starter}\n```"
        try:
            raw = row.get("public_test_cases", "[]")
            test_cases = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            test_cases = []
        if not test_cases:
            continue
        results.append({
            "domain":       "code",
            "complexity":   complexity,
            "query":        prompt,
            "ground_truth": json.dumps(test_cases),
            "source":       "livecodebench",
        })
        if len(results) >= n * 3:
            break

    per_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for item in results:
        per_diff[item["complexity"]].append(item)
    per_bucket = n // 3
    balanced: list[dict] = []
    for diff, pool in per_diff.items():
        want = per_bucket if diff != "hard" else n - 2 * per_bucket
        balanced.extend(random.sample(pool, min(want, len(pool))))
    return balanced


TARGET_DISTRIBUTION = {
    ("factual",   "easy"):   100,
    ("factual",   "medium"): 100,
    ("factual",   "hard"):   100,
    ("math",      "easy"):   100,
    ("math",      "medium"): 110,
    ("math",      "hard"):    90,
    ("code",      "easy"):   100,
    ("code",      "medium"): 110,
    ("code",      "hard"):    90,
    ("reasoning", "easy"):   100,
    ("reasoning", "medium"): 110,
    ("reasoning", "hard"):    90,
}


def build(total: int, output: str, lcb_count: int = 300) -> None:
    print(f"\n  Building {total}-request dataset from HuggingFace benchmarks...")
    print(f"  Output: {output}  (base={total - lcb_count} + LiveCodeBench={lcb_count})\n")
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    all_items: list[dict] = []

    def add(name: str, items: list[dict]) -> None:
        print(f"    {name:<20} {len(items):4} items loaded")
        all_items.extend(items)

    add("MMLU",           load_mmlu(320))
    add("GSM8K",          load_gsm8k(180))
    add("MATH benchmark", load_math_benchmark(130))
    add("HumanEval",      load_humaneval(200))
    add("BigCodeBench",   load_bigcodebench(130))
    add("APPS",           load_apps(90))
    add("LogiQA",         load_logiqa(150))
    add("HellaSwag",      load_hellaswag(130))
    add("ARC",            load_arc(150))

    print(f"\n  Total raw base items: {len(all_items)}")

    base_target = total - lcb_count
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in all_items:
        key = (item["domain"], item["complexity"])
        buckets[key].append(item)

    dataset: list[dict] = []
    for key, want in TARGET_DISTRIBUTION.items():
        pool = buckets[key]
        if len(pool) >= want:
            dataset.extend(random.sample(pool, want))
        else:
            dataset.extend(pool)
            if pool:
                dataset.extend(random.choices(pool, k=want - len(pool)))

    if len(dataset) < base_target:
        extra = random.choices(all_items, k=base_target - len(dataset))
        dataset.extend(extra)
    dataset = dataset[:base_target]

    print(f"\n  Loading LiveCodeBench ({lcb_count} samples)...")
    lcb_items = load_livecodebench(lcb_count)
    print(f"    LiveCodeBench      {len(lcb_items):4} items loaded")

    if lcb_items:
        dataset.extend(lcb_items)
    else:
        print("  LiveCodeBench unavailable -- padding with existing code items")
        code_pool = [x for x in all_items if x["domain"] == "code"]
        dataset.extend(random.choices(code_pool, k=lcb_count))

    random.shuffle(dataset)

    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n  Final dataset: {len(dataset)} requests")
    by_domain: dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    for item in dataset:
        by_domain[item["domain"]] += 1
        by_source[item.get("source", "hf")] += 1
    print("\n  By domain:")
    for d, c in sorted(by_domain.items()):
        print(f"    {d:<12} {c:4}")
    print("\n  By source:")
    for s, n2 in sorted(by_source.items()):
        print(f"    {s:<20} {n2:4}")
    print(f"\n  Saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",    default="datasets/hf_1500.json")
    parser.add_argument("--count",     type=int, default=1500)
    parser.add_argument("--lcb-count", type=int, default=300)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    build(args.count, args.output, lcb_count=args.lcb_count)


if __name__ == "__main__":
    main()
