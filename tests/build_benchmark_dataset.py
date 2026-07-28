"""
build_benchmark_dataset.py — Download five benchmarks and build a test dataset.

Benchmarks (primary → fallback):
  GPQA Diamond   → ARC-Challenge  : Hard reasoning MCQ       → reasoning:hard
  MMLU-Pro                        : Expert factual MCQ        → factual:easy/medium/hard
  GSM1K          → GSM8K          : Grade-school math         → math:easy/medium
  OlympiadBench  → MATH           : Olympiad/competition math → math:hard
  LiveCodeBench  → HumanEval+MBPP : Competitive coding        → code:easy/medium/hard

GPQA requires HuggingFace access:
  1. Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa
  2. Run: huggingface-cli login   (or set HF_TOKEN env var)
  Without access, falls back to ARC-Challenge.

Each item carries an `answer_type` field:
  "mcq"        — letter choice (A/B/C/D/…)
  "numeric"    — final number, ±1% tolerance
  "expression" — LaTeX/symbolic, \boxed{} extraction
  "code"       — stdin/stdout or assert execution

Usage:
    python tests/build_benchmark_dataset.py
    python tests/build_benchmark_dataset.py --output datasets/benchmark_1000.json --total 1000
    HF_TOKEN=hf_xxx python tests/build_benchmark_dataset.py  # enables GPQA
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict

CONTAMINATION_CUTOFF = "2024-06-01"

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gpqa_diamond(n: int) -> list[dict]:
    """GPQA Diamond (gated) → falls back to ARC-Challenge.

    GPQA: 198 graduate-level science MCQ, reasoning:hard.
    ARC-Challenge: high school science MCQ, same format, freely available.
    Set HF_TOKEN env var and accept GPQA terms to get the real dataset.
    """
    from datasets import load_dataset

    ds = None
    for ds_id, cfg in [
        ("Idavidrein/gpqa", "gpqa_diamond"),
        ("Idavidrein/gpqa", "gpqa_main"),
    ]:
        try:
            ds = load_dataset(ds_id, cfg, split="train", token=_HF_TOKEN)
            print(f"  [GPQA] loaded from {ds_id}/{cfg} ({len(ds)} items)")
            break
        except Exception as e:
            print(f"  [GPQA/{ds_id}] skipped: {e}")

    if ds is not None:
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
            rng = random.Random(hash(question) & 0xFFFFFFFF)
            options = [correct, wrong1, wrong2, wrong3]
            rng.shuffle(options)
            labels = ["A", "B", "C", "D"]
            gt_letter = labels[options.index(correct)]
            opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(4) if options[i])
            query = (
                f"{question}\n\n{opts_str}\n\n"
                "Reply with only the letter of the correct answer (A, B, C, or D)."
            )
            results.append({
                "domain":      "reasoning",
                "complexity":  "hard",
                "query":       query,
                "ground_truth": gt_letter,
                "source":      "gpqa_diamond",
                "answer_type": "mcq",
            })
            if len(results) >= n:
                break
        return results

    # Fallback: ARC-Challenge
    print("  [GPQA] Falling back to ARC-Challenge"
          " (set HF_TOKEN + accept GPQA terms for real GPQA data)")
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test+validation")
        print(f"  [ARC-Challenge] loaded {len(arc)} items")
    except Exception as e:
        print(f"  [ARC-Challenge] skipped: {e}")
        return []

    arc = arc.shuffle(seed=42)
    results = []
    for row in arc:
        question = str(row.get("question", "")).strip()
        choices  = row.get("choices", {})
        labels   = choices.get("label", [])
        texts    = choices.get("text", [])
        ans_key  = str(row.get("answerKey", "A")).strip().upper()
        if not question or not labels:
            continue
        # Normalize numeric answer keys (1→A, 2→B, …)
        if ans_key.isdigit():
            ans_key = chr(ord("A") + int(ans_key) - 1)
        opts_str = "\n".join(
            f"{labels[i]}) {texts[i]}" for i in range(min(len(labels), len(texts)))
        )
        query = (
            f"{question}\n\n{opts_str}\n\n"
            "Reply with only the letter of the correct answer."
        )
        results.append({
            "domain":      "reasoning",
            "complexity":  "hard",
            "query":       query,
            "ground_truth": ans_key,
            "source":      "arc_challenge",
            "answer_type": "mcq",
        })
        if len(results) >= n:
            break
    return results


def load_mmlu_pro(n: int) -> list[dict]:
    """MMLU-Pro: expert-level factual MCQ with 10 choices. factual:easy/medium/hard."""
    from datasets import load_dataset

    HARD_CATS   = {"math", "physics", "chemistry", "biology", "engineering",
                   "computer science", "medical"}
    MEDIUM_CATS = {"law", "economics", "psychology", "philosophy",
                   "business", "history"}

    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
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
            "Reply with only the letter of the correct answer."
        )

        ans_idx = row.get("answer_index", None)
        if ans_idx is not None and 0 <= ans_idx < len(labels):
            gt_letter = labels[ans_idx]
        else:
            raw = str(row.get("answer", "A")).strip().upper()
            gt_letter = raw[0] if raw and raw[0].isalpha() else "A"

        results.append({
            "domain":      "factual",
            "complexity":  complexity,
            "query":       query,
            "ground_truth": gt_letter,
            "source":      "mmlu_pro",
            "answer_type": "mcq",
        })
        if len(results) >= n * 2:
            break

    return random.sample(results, min(n, len(results)))


def load_gsm1k(n: int) -> list[dict]:
    """GSM1K → GSM8K fallback: grade-school math. math:easy/medium."""
    from datasets import load_dataset

    ds = None
    for ds_id, cfg, split in [
        ("gsm1k/gsm1k",   None,   "test"),
        ("math-ai/gsm1k", None,   "test"),
        ("openai/gsm8k",  "main", "test"),
    ]:
        try:
            if cfg:
                ds = load_dataset(ds_id, cfg, split=split)
            else:
                ds = load_dataset(ds_id, split=split)
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
        parts = answer.split("####")
        if len(parts) > 1:
            gt = parts[-1].strip().replace(",", "")
        else:
            nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
            gt = nums[-1] if nums else answer.strip()[:50]
        complexity = "easy" if len(question.split()) <= 60 else "medium"
        results.append({
            "domain":      "math",
            "complexity":  complexity,
            "query":       question,
            "ground_truth": gt,
            "source":      "gsm1k",
            "answer_type": "numeric",
        })
    return results


def load_olympiadbench(n: int) -> list[dict]:
    """OlympiadBench → MATH competition fallback. math:hard, answer_type=expression.

    OlympiadBench uses a loading script (trust_remote_code) which is no longer
    supported in newer datasets versions. Falls back to the MATH competition dataset
    (AMC/AIME/competition_math problems, level 4-5 difficulty).
    """
    from datasets import load_dataset

    ds = None

    # Try OlympiadBench without trust_remote_code (works if dataset was converted to Parquet)
    for ds_id in ["OpenBMB/OlympiadBench", "olympiadbench/OlympiadBench"]:
        try:
            raw = load_dataset(ds_id)
            split_name = "train" if "train" in raw else list(raw.keys())[0]
            ds_raw = raw[split_name]
            print(f"  [OlympiadBench] loaded from {ds_id} split={split_name} ({len(ds_raw)} items)")
            ds_raw = ds_raw.shuffle(seed=42).select(range(min(n * 4, len(ds_raw))))
            results = []
            for row in ds_raw:
                problem  = str(row.get("problem",  row.get("question", ""))).strip()
                answer   = str(row.get("answer",   row.get("solution", ""))).strip()
                subject  = str(row.get("subject",  row.get("category", "math"))).lower()
                if not problem or not answer:
                    continue
                if "math" not in subject and "physics" not in subject:
                    continue
                m = re.search(r"\\boxed\{([^}]+)\}", answer)
                gt = m.group(1).strip() if m else answer.strip()[:120]
                query = (
                    "Solve the following olympiad problem. Show your reasoning step by step, "
                    "then put your final answer in \\boxed{}:\n\n"
                    f"{problem}"
                )
                results.append({
                    "domain":      "math",
                    "complexity":  "hard",
                    "query":       query,
                    "ground_truth": gt,
                    "source":      "olympiadbench",
                    "answer_type": "expression",
                })
                if len(results) >= n:
                    break
            return results[:n]
        except Exception as e:
            print(f"  [OlympiadBench/{ds_id}] skipped: {e}")

    # Fallback: MATH competition dataset (level 4-5 = hard competition problems)
    print("  [OlympiadBench] Falling back to MATH competition dataset (level 4-5)")
    for ds_id, cfg in [
        ("lighteval/MATH", "all"),
        ("EleutherAI/hendrycks_math", "all"),
        ("hendrycks/competition_mathematics", None),
    ]:
        try:
            if cfg:
                raw = load_dataset(ds_id, cfg, split="test")
            else:
                raw = load_dataset(ds_id, split="test")
            ds = raw
            print(f"  [MATH] loaded from {ds_id} ({len(ds)} items)")
            break
        except Exception as e:
            print(f"  [MATH/{ds_id}] skipped: {e}")

    if ds is None:
        return []

    ds = ds.shuffle(seed=42)
    results = []
    for row in ds:
        level   = str(row.get("level", "")).strip()
        problem = str(row.get("problem", row.get("question", ""))).strip()
        solution = str(row.get("solution", row.get("answer", ""))).strip()
        if not problem:
            continue
        # Keep only hard problems (level 4 or 5)
        if level and not any(x in level for x in ("4", "5")):
            continue
        m = re.search(r"\\boxed\{([^}]+)\}", solution)
        gt = m.group(1).strip() if m else re.findall(r"-?\d+(?:\.\d+)?", solution)[-1] if re.findall(r"-?\d+(?:\.\d+)?", solution) else solution[:80]
        query = (
            "Solve the following math problem. Show your reasoning step by step, "
            "then put your final answer in \\boxed{}:\n\n"
            f"{problem}"
        )
        results.append({
            "domain":      "math",
            "complexity":  "hard",
            "query":       query,
            "ground_truth": gt,
            "source":      "competition_math",
            "answer_type": "expression",
        })
        if len(results) >= n:
            break
    return results[:n]


def load_livecodebench(n: int, cutoff: str = CONTAMINATION_CUTOFF) -> list[dict]:
    """LiveCodeBench → HumanEval + MBPP fallback. code:easy/medium/hard.

    LiveCodeBench uses a loading script which is blocked in datasets >= 3.0.
    Falls back to HumanEval (164 problems, medium) + MBPP (374 problems, easy).
    """
    from datasets import load_dataset

    COMPLEXITY_MAP = {"easy": "easy", "medium": "medium", "hard": "hard"}

    def _strip_html(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        for entity, char in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                              ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
            text = text.replace(entity, char)
        return re.sub(r"\s{2,}", " ", text).strip()

    # Try LiveCodeBench
    lcb_ds = None
    for ds_id in ["livecodebench/code_generation_lite"]:
        try:
            raw = load_dataset(ds_id)
            lcb_ds = raw["test"] if "test" in raw else list(raw.values())[0]
            print(f"  [LiveCodeBench] loaded from {ds_id} ({len(lcb_ds)} items)")
            break
        except Exception as e:
            print(f"  [LiveCodeBench/{ds_id}] skipped: {e}")

    if lcb_ds is not None:
        lcb_ds = lcb_ds.shuffle(seed=42)
        results = []
        skipped = 0
        for row in lcb_ds:
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
                "domain":      "code",
                "complexity":  complexity,
                "query":       prompt,
                "ground_truth": json.dumps(test_cases),
                "source":      "livecodebench",
                "answer_type": "code",
            })
            if len(results) >= n * 3:
                break
        if skipped:
            print(f"  [LiveCodeBench] skipped {skipped} items before cutoff {cutoff}")
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

    # Fallback: HumanEval + MBPP
    print("  [LiveCodeBench] Falling back to HumanEval + MBPP (loading scripts not supported)")
    results = []

    for he_id in ["openai_humaneval", "openai/openai_humaneval"]:
        try:
            he = load_dataset(he_id, split="test")
            print(f"  [HumanEval] loaded {len(he)} items from {he_id}")
            for row in he:
                prompt      = str(row.get("prompt", "")).strip()
                test_code   = str(row.get("test", "")).strip()
                entry_point = str(row.get("entry_point", "solve")).strip()
                if not prompt:
                    continue
                query = (
                    "Solve the following Python coding problem. "
                    "Implement the function exactly as specified:\n\n"
                    f"{prompt}"
                )
                # Score using assert-style tests
                gt = f"{test_code}\ncheck({entry_point})" if "check" in test_code else test_code
                results.append({
                    "domain":      "code",
                    "complexity":  "medium",
                    "query":       query,
                    "ground_truth": gt,
                    "source":      "humaneval",
                    "answer_type": "code",
                })
            break
        except Exception as e:
            print(f"  [HumanEval/{he_id}] skipped: {e}")

    for mbpp_id in ["google-research-datasets/mbpp", "mbpp"]:
        try:
            mbpp = load_dataset(mbpp_id, "full", split="test")
            print(f"  [MBPP] loaded {len(mbpp)} items from {mbpp_id}")
            mbpp = mbpp.shuffle(seed=42)
            for row in mbpp:
                text      = str(row.get("text", "")).strip()
                test_list = row.get("test_list", [])
                code      = str(row.get("code", "")).strip()
                if not text or not test_list:
                    continue
                query = (
                    "Write a Python function to solve the following problem:\n\n"
                    f"{text}"
                )
                gt = "\n".join(test_list)
                results.append({
                    "domain":      "code",
                    "complexity":  "easy",
                    "query":       query,
                    "ground_truth": gt,
                    "source":      "mbpp",
                    "answer_type": "code",
                })
            break
        except Exception as e:
            print(f"  [MBPP/{mbpp_id}] skipped: {e}")

    random.shuffle(results)
    selected = results[:n]
    print(f"  [HumanEval+MBPP] {len(selected)} items selected")
    return selected


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(total: int, output: str, cutoff: str = CONTAMINATION_CUTOFF) -> None:
    print(f"\nBuilding benchmark dataset ({total} samples total)...")
    print(f"  Cutoff : {cutoff}")
    print(f"  Output : {output}\n")
    if not _HF_TOKEN:
        print("  Tip: set HF_TOKEN env var to enable gated datasets (GPQA Diamond).\n")

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    per_benchmark = total // 5
    all_items: list[dict] = []

    def add(name: str, items: list[dict]) -> None:
        print(f"  {name:<22} {len(items):4} items loaded")
        all_items.extend(items)

    print("Loading benchmarks:")
    add("GPQA / ARC-Challenge", load_gpqa_diamond(per_benchmark))
    add("MMLU-Pro",             load_mmlu_pro(per_benchmark))
    add("GSM1K / GSM8K",        load_gsm1k(per_benchmark))
    add("OlympiadBench / MATH", load_olympiadbench(per_benchmark))
    add("LiveCodeBench / HE+MBPP", load_livecodebench(per_benchmark, cutoff=cutoff))

    random.shuffle(all_items)
    for i, item in enumerate(all_items):
        item["req_id"] = i

    with open(output, "w") as f:
        json.dump(all_items, f, indent=2)

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
    print(f"  {'Source':<26} {'N':>5}")
    for k, v in sorted(by_src.items()):
        print(f"  {k:<26} {v:>5}")
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
        description="Download benchmarks and build a test dataset."
    )
    parser.add_argument("--output",  default="datasets/benchmark_1000.json")
    parser.add_argument("--total",   type=int, default=1000,
                        help="Total items (~total/5 per benchmark)")
    parser.add_argument("--cutoff",  default=CONTAMINATION_CUTOFF,
                        help="Contamination cutoff YYYY-MM-DD (LiveCodeBench filter)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    build(args.total, args.output, cutoff=args.cutoff)


if __name__ == "__main__":
    main()
