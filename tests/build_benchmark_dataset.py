"""
build_benchmark_dataset.py — Download five benchmarks and build a test dataset.

Benchmarks:
  GPQA Diamond   → ARC-Challenge  : Hard reasoning MCQ         reasoning:hard
  MMLU-Pro                        : Expert factual MCQ          factual:easy/medium/hard
  GSM1K          → GSM8K          : Grade-school math           math:easy/medium
  OlympiadBench                   : Olympiad math/physics       math:hard
  LiveCodeBench                   : Competitive programming     code:easy/medium/hard

OlympiadBench and LiveCodeBench are loaded directly from their raw data files on
HuggingFace Hub (bypassing the deprecated loading scripts).

GPQA requires HuggingFace account access:
  1. Accept terms at https://huggingface.co/datasets/Idavidrein/gpqa
  2. huggingface-cli login  (or set HF_TOKEN env var)
  Without access, falls back to ARC-Challenge.

Usage:
    python tests/build_benchmark_dataset.py
    python tests/build_benchmark_dataset.py --total 1000 --output datasets/benchmark_1000.json
    HF_TOKEN=hf_xxx python tests/build_benchmark_dataset.py
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
# Helpers
# ---------------------------------------------------------------------------

def _list_repo_data_files(repo_id: str) -> list[str]:
    """Return all data file paths in a HuggingFace dataset repo."""
    from huggingface_hub import list_repo_files
    try:
        files = list(list_repo_files(repo_id, repo_type="dataset", token=_HF_TOKEN))
        exts = (".parquet", ".json", ".jsonl", ".arrow", ".csv")
        skip = ("readme", ".gitattributes", ".py", ".md", "license")
        return [
            f for f in files
            if any(f.lower().endswith(e) for e in exts)
            and not any(s in f.lower() for s in skip)
        ]
    except Exception as e:
        print(f"    [list_files/{repo_id}] {e}")
        return []


def _load_raw(repo_id: str, files: list[str]) -> object | None:
    """Load a dataset directly from a list of repo-relative file paths."""
    from datasets import load_dataset
    if not files:
        return None
    # Prefer parquet; fall back to json/jsonl
    pq   = [f for f in files if f.endswith(".parquet")]
    js   = [f for f in files if f.endswith((".json", ".jsonl"))]
    use  = pq if pq else js
    if not use:
        return None
    fmt  = "parquet" if pq else "json"
    urls = [f"hf://datasets/{repo_id}/{f}" for f in use]
    raw  = load_dataset(fmt, data_files=urls)
    return raw["train"] if "train" in raw else list(raw.values())[0]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gpqa_diamond(n: int) -> list[dict]:
    """GPQA Diamond (gated) → ARC-Challenge fallback.

    Set HF_TOKEN and accept terms at huggingface.co/datasets/Idavidrein/gpqa
    to get the real GPQA data. Otherwise uses ARC-Challenge (free, same format).
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
            labels    = ["A", "B", "C", "D"]
            gt_letter = labels[options.index(correct)]
            opts_str  = "\n".join(f"{labels[i]}) {options[i]}" for i in range(4) if options[i])
            query = (
                f"{question}\n\n{opts_str}\n\n"
                "Reply with only the letter of the correct answer (A, B, C, or D)."
            )
            results.append({
                "domain": "reasoning", "complexity": "hard",
                "query": query, "ground_truth": gt_letter,
                "source": "gpqa_diamond", "answer_type": "mcq",
            })
            if len(results) >= n:
                break
        return results

    # Fallback: ARC-Challenge
    print("  [GPQA] Falling back to ARC-Challenge"
          " (accept GPQA terms + set HF_TOKEN for real data)")
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
        if ans_key.isdigit():
            ans_key = chr(ord("A") + int(ans_key) - 1)
        opts_str = "\n".join(
            f"{labels[i]}) {texts[i]}" for i in range(min(len(labels), len(texts)))
        )
        query = f"{question}\n\n{opts_str}\n\nReply with only the letter of the correct answer."
        results.append({
            "domain": "reasoning", "complexity": "hard",
            "query": query, "ground_truth": ans_key,
            "source": "arc_challenge", "answer_type": "mcq",
        })
        if len(results) >= n:
            break
    return results


def load_mmlu_pro(n: int) -> list[dict]:
    """MMLU-Pro: expert-level factual MCQ with 10 choices. factual:easy/medium/hard."""
    from datasets import load_dataset

    HARD_CATS   = {"math", "physics", "chemistry", "biology", "engineering",
                   "computer science", "medical"}
    MEDIUM_CATS = {"law", "economics", "psychology", "philosophy", "business", "history"}

    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception as e:
        print(f"  [MMLU-Pro] skipped: {e}")
        return []

    ds = ds.shuffle(seed=42)
    results = []
    for row in ds:
        category = str(row.get("category", "")).lower()
        complexity = (
            "hard"   if any(h in category for h in HARD_CATS)   else
            "medium" if any(m in category for m in MEDIUM_CATS) else
            "easy"
        )
        question = row.get("question", "")
        options  = row.get("options", [])
        if not question or not options:
            continue
        labels   = [chr(ord("A") + i) for i in range(len(options))]
        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(len(options)))
        query    = f"{question}\n\n{opts_str}\n\nReply with only the letter of the correct answer."
        ans_idx  = row.get("answer_index", None)
        if ans_idx is not None and 0 <= ans_idx < len(labels):
            gt_letter = labels[ans_idx]
        else:
            raw = str(row.get("answer", "A")).strip().upper()
            gt_letter = raw[0] if raw and raw[0].isalpha() else "A"
        results.append({
            "domain": "factual", "complexity": complexity,
            "query": query, "ground_truth": gt_letter,
            "source": "mmlu_pro", "answer_type": "mcq",
        })
        if len(results) >= n * 2:
            break
    return random.sample(results, min(n, len(results)))


def load_gsm1k(n: int) -> list[dict]:
    """GSM1K: contamination-resistant grade-school math. Falls back to GSM8K.

    Tries known HuggingFace dataset IDs for GSM1K. If none exist, searches
    via HuggingFace Hub API for any public dataset named gsm1k, then falls
    back to GSM8K.
    """
    from datasets import load_dataset

    ds = None

    # Known possible IDs for GSM1K
    candidates = [
        ("gsm1k/gsm1k",          None,   "test"),
        ("math-ai/gsm1k",        None,   "test"),
        ("gsm1k/gsm1k",          None,   "train"),
        ("gsm1k",                None,   "test"),
        ("mlfoundations/gsm1k",  None,   "test"),
    ]

    for ds_id, cfg, split in candidates:
        try:
            if cfg:
                ds = load_dataset(ds_id, cfg, split=split, token=_HF_TOKEN)
            else:
                ds = load_dataset(ds_id, split=split, token=_HF_TOKEN)
            print(f"  [GSM1K] loaded from {ds_id} ({len(ds)} items)")
            break
        except Exception:
            pass

    # Try searching HuggingFace Hub for any gsm1k dataset
    if ds is None:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            results_search = list(api.list_datasets(search="gsm1k", limit=10))
            for info in results_search:
                repo_id = info.id
                if "gsm1k" not in repo_id.lower():
                    continue
                try:
                    ds = load_dataset(repo_id, split="test", token=_HF_TOKEN)
                    print(f"  [GSM1K] found and loaded from {repo_id} ({len(ds)} items)")
                    break
                except Exception:
                    try:
                        ds = load_dataset(repo_id, split="train", token=_HF_TOKEN)
                        print(f"  [GSM1K] found and loaded from {repo_id} ({len(ds)} items)")
                        break
                    except Exception:
                        pass
        except Exception as e:
            print(f"  [GSM1K/search] {e}")

    # Fallback: GSM8K
    if ds is None:
        print("  [GSM1K] not found — falling back to GSM8K")
        try:
            ds = load_dataset("openai/gsm8k", "main", split="test")
            print(f"  [GSM8K] loaded {len(ds)} items")
        except Exception as e:
            print(f"  [GSM8K] skipped: {e}")
            return []

    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    results = []
    for row in ds:
        question = str(row.get("question", row.get("problem", ""))).strip()
        answer   = str(row.get("answer",   row.get("solution", ""))).strip()
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
            "domain": "math", "complexity": complexity,
            "query": question, "ground_truth": gt,
            "source": "gsm1k", "answer_type": "numeric",
        })
    return results


def load_olympiadbench(n: int) -> list[dict]:
    """OlympiadBench (or NuminaMath fallback): hard math/physics competition problems.

    Tries OlympiadBench first via streaming Parquet (English-only files, capped).
    Falls back to AI-MO/NuminaMath-CoT which is openly accessible and has AMC/AIME
    competition problems of comparable difficulty.
    """
    import itertools
    from datasets import load_dataset

    def _parse_olympiad_row(row: dict) -> dict | None:
        problem = str(
            row.get("problem", row.get("Problem", row.get("question", "")))
        ).strip()
        answer = str(
            row.get("answer", row.get("Answer", row.get("solution", "")))
        ).strip()
        subject = str(
            row.get("subject", row.get("Subject", row.get("category", "math")))
        ).lower()
        if not problem or not answer:
            return None
        if "math" not in subject and "phys" not in subject and subject not in ("", "unknown"):
            return None
        m  = re.search(r"\\boxed\{([^}]+)\}", answer)
        gt = m.group(1).strip() if m else answer.strip()[:120]
        return {
            "domain": "math", "complexity": "hard",
            "query": (
                "Solve the following olympiad problem. Show your reasoning step by step, "
                f"then put your final answer in \\boxed{{}}:\n\n{problem}"
            ),
            "ground_truth": gt,
            "source": "olympiadbench", "answer_type": "expression",
        }

    # ── Try OlympiadBench directly via streaming ────────────────────────────
    for repo_id in ["OpenBMB/OlympiadBench", "olympiadbench/OlympiadBench"]:
        try:
            all_files = _list_repo_data_files(repo_id)
            if not all_files:
                print(f"  [OlympiadBench/{repo_id}] no data files found")
                continue

            # Prefer English math/physics files; cap at 4 files to avoid huge downloads
            en_files = [
                f for f in all_files
                if any(x in f.lower() for x in ["_en", "english", "maths_en", "phys_en"])
            ]
            use_files = (en_files if en_files else all_files)[:4]
            fmt  = "parquet" if any(f.endswith(".parquet") for f in use_files) else "json"
            urls = [f"hf://datasets/{repo_id}/{f}" for f in use_files]
            print(f"  [OlympiadBench/{repo_id}] streaming {len(urls)} files ({fmt})")

            raw    = load_dataset(fmt, data_files=urls, streaming=True)
            split  = raw["train"] if "train" in raw else list(raw.values())[0]
            sample = list(itertools.islice(split, n * 6))  # stream only what's needed
            print(f"  [OlympiadBench/{repo_id}] streamed {len(sample)} candidate rows")

            results = []
            for row in sample:
                parsed = _parse_olympiad_row(row)
                if parsed:
                    results.append(parsed)
                if len(results) >= n:
                    break

            if results:
                print(f"  [OlympiadBench] {len(results)} items selected")
                return results[:n]

        except Exception as e:
            print(f"  [OlympiadBench/{repo_id}] skipped: {e}")

    # ── Fallback: NuminaMath-CoT (AMC / AIME / competition math, open access) ──
    print("  [OlympiadBench] falling back to AI-MO/NuminaMath-CoT")
    try:
        raw   = load_dataset("AI-MO/NuminaMath-CoT", streaming=True)
        split = raw["train"] if "train" in raw else list(raw.values())[0]

        # Only keep competition-level sources (AIME, AMC, Olympiad)
        hard_sources = {"aime", "amc", "olympiad", "imo", "usamo", "putnam", "hmmt",
                        "arml", "mathcounts"}
        results = []
        for row in split:
            src = str(row.get("source", "")).lower()
            if not any(h in src for h in hard_sources):
                continue
            problem  = str(row.get("problem", "")).strip()
            solution = str(row.get("solution", "")).strip()
            if not problem or not solution:
                continue
            m  = re.search(r"\\boxed\{([^}]+)\}", solution)
            gt = m.group(1).strip() if m else solution.strip()[:120]
            results.append({
                "domain": "math", "complexity": "hard",
                "query": (
                    "Solve the following competition math problem. Show your reasoning "
                    f"step by step, then put your final answer in \\boxed{{}}:\n\n{problem}"
                ),
                "ground_truth": gt,
                "source": "numina_math", "answer_type": "expression",
            })
            if len(results) >= n:
                break

        if results:
            print(f"  [NuminaMath-CoT] {len(results)} items selected")
            return results[:n]

    except Exception as e:
        print(f"  [NuminaMath-CoT] skipped: {e}")

    print("  [OlympiadBench] could not load from any source")
    return []


def load_livecodebench(n: int, cutoff: str = CONTAMINATION_CUTOFF) -> list[dict]:
    """LiveCodeBench: post-cutoff competitive programming. code:easy/medium/hard.

    Loads raw data files directly from HuggingFace Hub, bypassing the
    deprecated loading script.
    """

    COMPLEXITY_MAP = {"easy": "easy", "medium": "medium", "hard": "hard"}

    def _strip_html(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        for entity, char in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                              ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
            text = text.replace(entity, char)
        return re.sub(r"\s{2,}", " ", text).strip()

    ds = None
    for repo_id in [
        "livecodebench/code_generation_lite",
        "livecodebench/code_generation",
    ]:
        try:
            all_files = _list_repo_data_files(repo_id)
            if not all_files:
                print(f"  [LiveCodeBench/{repo_id}] no data files found")
                continue
            print(f"  [LiveCodeBench/{repo_id}] found {len(all_files)} files")

            ds = _load_raw(repo_id, all_files)
            if ds is None:
                continue
            print(f"  [LiveCodeBench] loaded {len(ds)} rows via direct file access")
            break
        except Exception as e:
            print(f"  [LiveCodeBench/{repo_id}] skipped: {e}")

    if ds is None:
        print("  [LiveCodeBench] could not load from any source")
        return []

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
        content    = _strip_html(str(row.get("question_content", ""))).strip()
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
            "domain": "code", "complexity": complexity,
            "query": prompt, "ground_truth": json.dumps(test_cases),
            "source": "livecodebench", "answer_type": "code",
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


def load_livebench_reasoning(n: int) -> list[dict]:
    """LiveBench reasoning tasks → reasoning:easy and reasoning:medium.

    Three tasks from livebench/reasoning:
      - web_of_lies_v2  : truth/lie chain logic         → easy
      - zebra_puzzle    : logic grid (level-based)      → easy (≤12) / medium (13-16)
      - spatial         : geometric piece-counting       → medium

    Answer types:
      - spatial         : "numeric"    (bold integer answer already in query)
      - web_of_lies_v2  : "structured" (comma-separated yes/no)
      - zebra_puzzle    : "structured" (comma-separated attribute values)
    """
    import json as _json
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "parquet",
            data_files="hf://datasets/livebench/reasoning/data/test-00000-of-00001.parquet",
            split="train",
        )
        print(f"  [LiveBench/reasoning] loaded {len(ds)} items")
    except Exception as e:
        print(f"  [LiveBench/reasoning] skipped: {e}")
        return []

    def _query(row: dict) -> str:
        turns = row.get("turns", "")
        try:
            parsed = _json.loads(turns)
            return parsed[0] if parsed else str(turns)
        except Exception:
            return str(turns)

    def _complexity(row: dict) -> str:
        task = row.get("task", "")
        if task == "web_of_lies_v2":
            return "easy"
        if task == "spatial":
            return "medium"
        # zebra_puzzle: use level field
        try:
            lvl = int(row.get("level") or 0)
        except (TypeError, ValueError):
            lvl = 0
        return "easy" if lvl <= 12 else "medium"

    def _answer_type(row: dict) -> str:
        return "numeric" if row.get("task") == "spatial" else "structured"

    results: list[dict] = []
    ds = ds.shuffle(seed=42)
    for row in ds:
        task = row.get("task", "")
        if task not in ("zebra_puzzle", "web_of_lies_v2", "spatial"):
            continue
        query = _query(row)
        gt    = str(row.get("ground_truth", "")).strip()
        if not query or not gt:
            continue
        complexity   = _complexity(row)
        answer_type  = _answer_type(row)
        # Append output-format hint for structured tasks
        if answer_type == "structured":
            n_fields = len([f for f in gt.split(",") if f.strip()])
            query = (
                query.rstrip()
                + f"\n\nProvide your final answer as a comma-separated list of "
                  f"{n_fields} value(s) matching the order asked."
            )
        results.append({
            "domain": "reasoning",
            "complexity": complexity,
            "query": query,
            "ground_truth": gt,
            "source": f"livebench_{task}",
            "answer_type": answer_type,
        })
        if len(results) >= n:
            break

    easy   = [r for r in results if r["complexity"] == "easy"]
    medium = [r for r in results if r["complexity"] == "medium"]
    per_c  = n // 2
    balanced = (
        random.sample(easy,   min(per_c, len(easy)))
        + random.sample(medium, min(per_c, len(medium)))
    )
    print(f"  [LiveBench/reasoning] {len(balanced)} items selected "
          f"(easy={sum(1 for r in balanced if r['complexity']=='easy')}, "
          f"medium={sum(1 for r in balanced if r['complexity']=='medium')})")
    return balanced


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(total: int, output: str, cutoff: str = CONTAMINATION_CUTOFF) -> None:
    print(f"\nBuilding benchmark dataset ({total} samples total)...")
    print(f"  Cutoff : {cutoff}")
    print(f"  Output : {output}")
    if not _HF_TOKEN:
        print("  Tip   : set HF_TOKEN to enable gated datasets (GPQA Diamond)\n")
    else:
        print()

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    per_benchmark = total // 5
    all_items: list[dict] = []

    def add(name: str, items: list[dict]) -> None:
        print(f"  {name:<34} {len(items):4} items loaded")
        all_items.extend(items)

    print("Loading benchmarks:")
    # Reasoning slot: GPQA Diamond (hard) + LiveBench (easy/medium), split evenly
    gpqa_n      = per_benchmark // 2
    livebench_n = per_benchmark - gpqa_n
    add("GPQA / ARC-Challenge (hard)",    load_gpqa_diamond(gpqa_n))
    add("LiveBench Reasoning (easy/med)", load_livebench_reasoning(livebench_n))
    add("MMLU-Pro",                       load_mmlu_pro(per_benchmark))
    add("GSM1K / GSM8K",                  load_gsm1k(per_benchmark))
    add("OlympiadBench",                  load_olympiadbench(per_benchmark))
    add("LiveCodeBench",                  load_livecodebench(per_benchmark, cutoff=cutoff))

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
                        help="Contamination cutoff YYYY-MM-DD (LiveCodeBench)")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    build(args.total, args.output, cutoff=args.cutoff)


if __name__ == "__main__":
    main()
