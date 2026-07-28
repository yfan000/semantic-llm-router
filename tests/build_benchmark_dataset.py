"""
build_benchmark_dataset.py — Download five benchmarks and build a test dataset.

Benchmarks:
  GPQA Diamond   : Graduate-level science MCQ (A-D) → reasoning:hard
  MMLU-Pro       : Expert-level factual MCQ (A-J)   → factual:easy/medium/hard
  GSM1K          : Contamination-resistant grade-school math → math:easy/medium
  OlympiadBench  : Olympiad math/physics problems   → math:hard
  LiveCodeBench  : Post-cutoff competitive programming → code:easy/medium/hard

Each item carries an `answer_type` field consumed by run_benchmark_eval.py:
  "mcq"        — letter choice (A/B/C/D/…), scored by letter extraction
  "numeric"    — final number, scored by numeric comparison ±1%
  "expression" — LaTeX/symbolic answer, scored by expression normalization
  "code"       — code execution against stdin/stdout test cases

Install:
    pip install datasets tqdm

Usage:
    python tests/build_benchmark_dataset.py
    python tests/build_benchmark_dataset.py --output datasets/benchmark_1000.json --total 1000
    python tests/build_benchmark_dataset.py --total 500 --cutoff 2024-09-01
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

CONTAMINATION_CUTOFF = "2024-06-01"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gpqa_diamond(n: int) -> list[dict]:
    """GPQA Diamond: 198 graduate-level science MCQ. reasoning:hard.

    Correct answer is shuffled into a random position (A/B/C/D) with a
    per-question seed so the assignment is deterministic across runs.
    Ground truth is the letter (e.g. "B"), not the answer text.
    """
    from datasets import load_dataset

    ds = None
    for ds_id, cfg in [
        ("Idavidrein/gpqa", "gpqa_diamond"),
        ("Idavidrein/gpqa", "gpqa_main"),
    ]:
        try:
            ds = load_dataset(ds_id, cfg, split="train", trust_remote_code=False)
            print(f"  [GPQA] loaded from {ds_id}/{cfg} ({len(ds)} items)")
            break
        except Exception as e:
            print(f"  [GPQA/{ds_id}] skipped: {e}")

    if ds is None:
        return []

    ds = ds.shuffle(seed=42)
    results = []
    for row in ds:
        question = str(row.get("Question", row.get("question", ""))).strip()
        correct  = str(row.get("Correct Answer",    row.get("correct_answer",  ""))).strip()
        wrong1   = str(row.get("Incorrect Answer 1", row.get("distractor1", ""))).strip()
        wrong2   = str(row.get("Incorrect Answer 2", row.get("distractor2", ""))).strip()
        wrong3   = str(row.get("Incorrect Answer 3", row.get("distractor3", ""))).strip()

        if not question or not correct:
            continue

        # Shuffle options with per-question seed so results are deterministic
        rng = random.Random(hash(question) & 0xFFFFFFFF)
        options = [correct, wrong1, wrong2, wrong3]
        rng.shuffle(options)
        labels = ["A", "B", "C", "D"]
        gt_letter = labels[options.index(correct)]

        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(4) if options[i])
        query = (
            f"{question}\n\n{opts_str}\n\n"
            f"Reply with only the letter of the correct answer (A, B, C, or D)."
        )

        results.append({
            "domain":       "reasoning",
            "complexity":   "hard",
            "query":        query,
            "ground_truth": gt_letter,
            "source":       "gpqa_diamond",
            "answer_type":  "mcq",
        })
        if len(results) >= n:
            break

    return results


def load_mmlu_pro(n: int) -> list[dict]:
    """MMLU-Pro: expert-level factual MCQ with 10 choices. factual:easy/medium/hard.

    Complexity by category:
      STEM (math, physics, chemistry, biology, engineering, CS, medical) → hard
      Social science (law, econ, psychology, philosophy, business, history) → medium
      Other → easy
    Ground truth is the correct letter (A-J).
    """
    from datasets import load_dataset

    HARD_CATS   = {"math", "physics", "chemistry", "biology", "engineering",
                   "computer science", "medical"}
    MEDIUM_CATS = {"law", "economics", "psychology", "philosophy",
                   "business", "history"}

    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=False)
    except Exception as e:
        print(f"  [MMLU-Pro] skipped: {e}")
        return []

    ds = ds.shuffle(seed=42)
    results = []
    for row in ds:
        category = str(row.get("category", "")).lower()
        if any(h in category for h in HARD_CATS):
            complexity = "hard"
        elif any(m in category for m in MEDIUM_CATS):
            complexity = "medium"
        else:
            complexity = "easy"

        question = row.get("question", "")
        options  = row.get("options", [])
        if not question or not options:
            continue

        labels   = [chr(ord("A") + i) for i in range(len(options))]
        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(len(options)))
        query    = (
            f"{question}\n\n{opts_str}\n\n"
            f"Reply with only the letter of the correct answer."
        )

        ans_idx = row.get("answer_index", None)
        if ans_idx is not None and 0 <= ans_idx < len(labels):
            gt_letter = labels[ans_idx]
        else:
            raw = str(row.get("answer", "A")).strip().upper()
            gt_letter = raw[0] if raw and raw[0].isalpha() else "A"

        results.append({
            "domain":       "factual",
            "complexity":   complexity,
            "query":        query,
            "ground_truth": gt_letter,
            "source":       "mmlu_pro",
            "answer_type":  "mcq",
        })
        if len(results) >= n * 2:
            break

    return random.sample(results, min(n, len(results)))


def load_gsm1k(n: int) -> list[dict]:
    """GSM1K: contamination-resistant grade-school math. math:easy/medium.

    GSM1K is a 1250-problem benchmark designed to avoid memorization artifacts
    in GSM8K. Falls back to GSM8K if the HuggingFace dataset is unavailable.
    Complexity: short problems (≤60 words) → easy, longer → medium.
    """
    from datasets import load_dataset

    ds = None
    for ds_id, cfg, split in [
        ("gsm1k/gsm1k",              None,   "test"),
        ("math-ai/gsm1k",            None,   "test"),
        ("openai/gsm8k",             "main", "test"),   # fallback
    ]:
        try:
            if cfg:
                ds = load_dataset(ds_id, cfg, split=split, trust_remote_code=False)
            else:
                ds = load_dataset(ds_id, split=split, trust_remote_code=False)
            print(f"  [GSM1K] loaded from {ds_id} ({len(ds)} items)")
            break
        except Exception as e:
            print(f"  [GSM1K/{ds_id}] skipped: {e}")

    if ds is None:
        return []

    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        question = str(row.get("question", row.get("problem", ""))).strip()
        answer   = str(row.get("answer",   row.get("solution",  ""))).strip()

        if not question:
            continue

        # Extract numeric answer after "####" separator (GSM8K-style)
        parts = answer.split("####")
        if len(parts) > 1:
            gt = parts[-1].strip().replace(",", "")
        else:
            nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
            gt = nums[-1] if nums else answer.strip()[:50]

        complexity = "easy" if len(question.split()) <= 60 else "medium"

        results.append({
            "domain":       "math",
            "complexity":   complexity,
            "query":        question,
            "ground_truth": gt,
            "source":       "gsm1k",
            "answer_type":  "numeric",
        })

    return results


def load_olympiadbench(n: int) -> list[dict]:
    """OlympiadBench: math and physics olympiad problems. math:hard.

    Covers AMC, AIME, IMO, USAMO, and national olympiad problems.
    Ground truth may be numeric (e.g. "12") or symbolic (e.g. "\\frac{3}{2}").
    The query instructs the model to put its answer in \\boxed{}.
    """
    from datasets import load_dataset

    ds = None
    for ds_id in [
        "OpenBMB/OlympiadBench",
        "olympiadbench/OlympiadBench",
    ]:
        try:
            raw = load_dataset(ds_id, trust_remote_code=True)
            split_name = "train" if "train" in raw else list(raw.keys())[0]
            ds = raw[split_name]
            print(f"  [OlympiadBench] loaded from {ds_id} split={split_name} ({len(ds)} items)")
            break
        except Exception as e:
            print(f"  [OlympiadBench/{ds_id}] skipped: {e}")

    if ds is None:
        return []

    ds = ds.shuffle(seed=42).select(range(min(n * 4, len(ds))))
    results = []
    for row in ds:
        problem  = str(row.get("problem",  row.get("question", ""))).strip()
        answer   = str(row.get("answer",   row.get("solution", ""))).strip()
        subject  = str(row.get("subject",  row.get("category", "math"))).lower()

        if not problem or not answer:
            continue
        # Only math and physics
        if "math" not in subject and "physics" not in subject:
            continue

        # Unwrap \boxed{} from stored answer if present
        m = re.search(r"\\boxed\{([^}]+)\}", answer)
        gt = m.group(1).strip() if m else answer.strip()[:120]

        query = (
            "Solve the following olympiad problem. Show your reasoning step by step, "
            "then put your final answer in \\boxed{}:\n\n"
            f"{problem}"
        )

        results.append({
            "domain":       "math",
            "complexity":   "hard",
            "query":        query,
            "ground_truth": gt,
            "source":       "olympiadbench",
            "answer_type":  "expression",
        })
        if len(results) >= n:
            break

    return results[:n]


def load_livecodebench(n: int, cutoff: str = CONTAMINATION_CUTOFF) -> list[dict]:
    """LiveCodeBench: post-cutoff competitive programming. code:easy/medium/hard.

    Filters to problems released after `cutoff` so no model training set
    includes them. Ground truth is a JSON array of {input, output} test cases.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("livecodebench/code_generation_lite",
                          split="test", trust_remote_code=False)
    except Exception:
        try:
            raw = load_dataset("livecodebench/code_generation_lite",
                               trust_remote_code=False)
            ds = raw["test"] if "test" in raw else list(raw.values())[0]
        except Exception as e:
            print(f"  [LiveCodeBench] skipped: {e}")
            return []

    COMPLEXITY_MAP = {"easy": "easy", "medium": "medium", "hard": "hard"}

    def _strip_html(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        for entity, char in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                              ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
            text = text.replace(entity, char)
        return re.sub(r"\s{2,}", " ", text).strip()

    ds = ds.shuffle(seed=42)
    results = []
    skipped = 0
    for row in ds:
        contest_date = str(row.get("contest_date", "") or row.get("start_date", ""))
        if contest_date and contest_date < cutoff:
            skipped += 1
            continue

        difficulty = str(row.get("difficulty", "medium")).lower()
        complexity = COMPLEXITY_MAP.get(difficulty, "medium")
        content    = _strip_html(row.get("question_content", "")).strip()
        starter    = (row.get("starter_code") or "").strip()
        if not content:
            continue

        prompt = (
            "Solve the following competitive programming problem. "
            "Provide a complete, runnable Python solution:\n\n"
            f"{content}"
        )
        if starter:
            prompt += f"\n\nUse this starter code:\n```python\n{starter}\n```"

        try:
            raw_tc = row.get("public_test_cases", "[]")
            test_cases = json.loads(raw_tc) if isinstance(raw_tc, str) else raw_tc
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
            "answer_type":  "code",
        })
        if len(results) >= n * 3:
            break

    if skipped:
        print(f"  [LiveCodeBench] skipped {skipped} items before cutoff {cutoff}")

    # Balance easy/medium/hard
    per_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for item in results:
        per_diff[item["complexity"]].append(item)
    per_bucket = n // 3
    balanced: list[dict] = []
    for diff, pool in per_diff.items():
        want = per_bucket if diff != "hard" else n - 2 * per_bucket
        balanced.extend(random.sample(pool, min(want, len(pool))))

    print(f"  [LiveCodeBench] {len(balanced)} items selected (cutoff={cutoff})")
    return balanced


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(total: int, output: str, cutoff: str = CONTAMINATION_CUTOFF) -> None:
    print(f"\nBuilding benchmark dataset ({total} samples total)...")
    print(f"  Cutoff : {cutoff}")
    print(f"  Output : {output}\n")

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    per_benchmark = total // 5

    all_items: list[dict] = []

    def add(name: str, items: list[dict]) -> None:
        print(f"  {name:<22} {len(items):4} items loaded")
        all_items.extend(items)

    print("Loading benchmarks:")
    add("GPQA Diamond",  load_gpqa_diamond(per_benchmark))
    add("MMLU-Pro",      load_mmlu_pro(per_benchmark))
    add("GSM1K",         load_gsm1k(per_benchmark))
    add("OlympiadBench", load_olympiadbench(per_benchmark))
    add("LiveCodeBench", load_livecodebench(per_benchmark, cutoff=cutoff))

    random.shuffle(all_items)
    for i, item in enumerate(all_items):
        item["req_id"] = i

    with open(output, "w") as f:
        json.dump(all_items, f, indent=2)

    # Summary
    by_src   = defaultdict(int)
    by_dom   = defaultdict(int)
    by_cpx   = defaultdict(int)
    by_atype = defaultdict(int)
    for item in all_items:
        by_src[item.get("source", "?")]        += 1
        by_dom[item["domain"]]                 += 1
        by_cpx[item["complexity"]]             += 1
        by_atype[item.get("answer_type", "?")] += 1

    print(f"\nFinal dataset: {len(all_items)} items\n")
    print(f"  {'Source':<22} {'N':>5}")
    for k, v in sorted(by_src.items()):
        print(f"  {k:<22} {v:>5}")
    print(f"\n  {'Domain':<14} {'N':>5}")
    for k, v in sorted(by_dom.items()):
        print(f"  {k:<14} {v:>5}")
    print(f"\n  {'Complexity':<12} {'N':>5}")
    for k, v in sorted(by_cpx.items()):
        print(f"  {k:<12} {v:>5}")
    print(f"\n  {'answer_type':<14} {'N':>5}")
    for k, v in sorted(by_atype.items()):
        print(f"  {k:<14} {v:>5}")
    print(f"\nSaved to: {output}")
    print(f"Run eval with:\n  python tests/run_benchmark_eval.py --dataset {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GPQA/MMLU-Pro/GSM1K/OlympiadBench/LiveCodeBench and build a test dataset."
    )
    parser.add_argument("--output",  default="datasets/benchmark_1000.json")
    parser.add_argument("--total",   type=int, default=1000,
                        help="Total items (~total/5 per benchmark)")
    parser.add_argument("--cutoff",  default=CONTAMINATION_CUTOFF,
                        help="Contamination cutoff date YYYY-MM-DD (LiveCodeBench filter)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    build(args.total, args.output, cutoff=args.cutoff)


if __name__ == "__main__":
    main()
