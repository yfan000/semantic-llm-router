"""
build_dataset.py — Download real benchmark queries from HuggingFace and
build a balanced 3000-request test set for the semantic LLM router.

Sources:
  Factual  : MMLU-Pro (post-cutoff, harder than MMLU, less contamination)
  Math     : GSM8K (grade-school, medium) + MATH benchmark (competition, hard)
  Code     : HumanEval (easy/medium) + MBPP (easy) + BigCodeBench (medium/hard) + APPS (hard)
  Reasoning: LogiQA (medium) + HellaSwag (easy) + ARC (easy/hard)
  Code     : LiveCodeBench (post-cutoff, execution-scored, 400 samples)
  Code     : SWE-bench Verified (post-cutoff GitHub issues, 200 samples)

Dataset limits — why MBPP is added:
  HumanEval has only 164 problems; not enough for code:easy at 3000-sample scale.
  MBPP (374 sanitized problems) fills the gap; combined 538 unique easy code samples.

Contamination strategy:
  All models have training cutoff <= March 2025 (latest: Qwen3-Coder-30B, May 2025).
  We filter time-sensitive benchmarks to problems created after CONTAMINATION_CUTOFF
  so no model could have seen them in training data.

  MMLU-Pro:      released June 2024, human-curated harder questions not in MMLU;
                 no per-question date -- treated as lower-contamination-risk as-is.
  LiveCodeBench: filtered by contest_date >= CONTAMINATION_CUTOFF.
  SWE-bench:     filtered by created_at  >= CONTAMINATION_CUTOFF.
                 Source files fetched from GitHub and stored in repo_files for
                 real execution scoring in eval_all_models.py.

Install dependencies:
    pip install datasets tqdm

Usage:
    python tests/build_dataset.py                       # saves to datasets/hf_3000.json
    python tests/build_dataset.py --output my.json      # custom output path
    python tests/build_dataset.py --cutoff 2025-05-01   # custom contamination cutoff

The output JSON can be used directly with load_test.py:
    python tests/load_test.py --dataset datasets/hf_3000.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict

# Problems created AFTER this date are safe from training-data contamination
# for all 6 models (latest: Qwen3-Coder-30B, released May 2025, cutoff ~Mar 2025).
CONTAMINATION_CUTOFF = "2025-05-01"

# ---------------------------------------------------------------------------
# Dataset loaders — each returns list of {domain, complexity, query}
# ---------------------------------------------------------------------------

def load_mmlu_pro(n: int) -> list[dict]:
    """MMLU-Pro: harder, reasoning-focused version of MMLU (TIGER-Lab, Jun 2024).

    10 answer choices instead of 4, requires multi-step reasoning.
    Lower contamination risk than MMLU because these questions were not in
    the original MMLU and were released after most models' training cutoffs.

    Complexity mapping based on category:
      STEM categories        -> hard
      Social sciences/law    -> medium
      Other                  -> easy
    """
    from datasets import load_dataset

    HARD_CATEGORIES   = {"math", "physics", "chemistry", "biology", "engineering",
                         "computer science", "medical"}
    MEDIUM_CATEGORIES = {"law", "economics", "psychology", "philosophy",
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
        if any(h in category for h in HARD_CATEGORIES):
            complexity = "hard"
        elif any(m in category for m in MEDIUM_CATEGORIES):
            complexity = "medium"
        else:
            complexity = "easy"

        question = row.get("question", "")
        options  = row.get("options", [])
        labels   = [chr(ord("A") + i) for i in range(len(options))]
        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(len(options)))
        query    = f"{question}\n{opts_str}"

        ans_idx = row.get("answer_index", None)
        if ans_idx is not None and ans_idx < len(options):
            ground_truth = options[ans_idx]
        else:
            ground_truth = str(row.get("answer", ""))

        if not question:
            continue
        results.append({
            "domain": "factual", "complexity": complexity,
            "query": query, "ground_truth": ground_truth,
            "source": "mmlu_pro",
        })
        if len(results) >= n * 2:
            break

    return random.sample(results, min(n, len(results)))


# ---------------------------------------------------------------------------
# SWE-bench helpers for fetching source files from GitHub
# ---------------------------------------------------------------------------

def _swe_fetch_file(repo: str, commit: str, filepath: str) -> str | None:
    """Fetch one source file from GitHub raw content at a specific commit.

    No auth needed for public repositories (all SWE-bench repos are public).
    Rate limit: ~4 requests/s (we sleep 0.25s between calls in load_swe_bench).
    """
    import urllib.request as _ur
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{filepath}"
    try:
        req = _ur.Request(url, headers={"User-Agent": "swe-eval-dataset/1.0"})
        with _ur.urlopen(req, timeout=15) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _swe_affected_files(patch: str) -> list[str]:
    """Parse a unified diff to extract the list of modified file paths."""
    import re as _re
    files = []
    for line in patch.splitlines():
        m = _re.match(r"^--- a/(.+)$", line)
        if m and m.group(1) != "/dev/null":
            files.append(m.group(1))
    return files


def load_swe_bench(n: int, cutoff: str = CONTAMINATION_CUTOFF) -> list[dict]:
    """SWE-bench Verified: real GitHub issues with known fixes (princeton-nlp).

    Filters to issues created AFTER cutoff (no model has seen them in training).

    For each item, the source files affected by the gold patch are fetched from
    GitHub and stored as repo_files so eval_all_models.py can:
      1. Apply the model's patch with patch(1) -p1
      2. Run the FAIL_TO_PASS tests with pytest
      3. Return the real pass/fail score

    The prompt asks the model to output a unified diff patch + explanation.

    Complexity mapping:
      patch changed lines <= 20  -> easy
      patch changed lines <= 80  -> medium
      patch changed lines > 80   -> hard
    """
    import time as _time
    import re as _re
    from datasets import load_dataset

    ds = None
    for ds_name in ["princeton-nlp/SWE-bench_Verified", "princeton-nlp/SWE-bench"]:
        try:
            ds = load_dataset(ds_name, split="test", trust_remote_code=False)
            print(f"  [SWE-bench] loaded from {ds_name}")
            break
        except Exception as e:
            print(f"  [SWE-bench/{ds_name}] skipped: {e}")

    if ds is None:
        return []

    # First pass: collect all candidates that pass the date filter
    candidates = []
    for row in ds:
        created_at = str(row.get("created_at", "") or row.get("pull_created_at", ""))
        if created_at and created_at < cutoff:
            continue

        problem     = str(row.get("problem_statement", "")).strip()
        repo        = str(row.get("repo", "")).strip()
        patch       = str(row.get("patch", "")).strip()
        base_commit = str(row.get("base_commit", "")).strip()
        instance_id = str(row.get("instance_id", "")).strip()
        test_patch  = str(row.get("test_patch", "")).strip()

        fail_to_pass = row.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except Exception:
                fail_to_pass = []

        if not problem or not patch or not base_commit:
            continue

        patch_lines = len([l for l in patch.splitlines()
                           if l.startswith("+") or l.startswith("-")])
        if patch_lines <= 20:
            complexity = "easy"
        elif patch_lines <= 80:
            complexity = "medium"
        else:
            complexity = "hard"

        patch_symbols = " ".join(_re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", patch))
        ground_truth  = patch_symbols[:500] if patch_symbols else patch[:200]

        candidates.append({
            "domain":       "code",
            "complexity":   complexity,
            "ground_truth": ground_truth,
            "source":       "swe_bench",
            "created_at":   created_at,
            "instance_id":  instance_id,
            "repo":         repo,
            "base_commit":  base_commit,
            "gold_patch":   patch,
            "test_patch":   test_patch,
            "fail_to_pass": fail_to_pass,
            "_problem":     problem,
        })
        if len(candidates) >= n * 2:
            break

    if not candidates:
        print(f"  [SWE-bench] no items found after cutoff {cutoff}")
        return []

    # Balance per complexity and select final items
    per_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for item in candidates:
        per_diff[item["complexity"]].append(item)
    per_bucket = n // 3
    balanced: list[dict] = []
    for diff, pool in per_diff.items():
        want = per_bucket if diff != "hard" else n - 2 * per_bucket
        balanced.extend(random.sample(pool, min(want, len(pool))))

    # Fetch original source files for real execution scoring
    print(f"  [SWE-bench] fetching source files for {len(balanced)} items "
          f"(cutoff={cutoff}, ~{len(balanced) * 3 * 0.25:.0f}s)...")
    fetched = 0
    for item in balanced:
        affected  = _swe_affected_files(item["gold_patch"])[:5]  # max 5 files
        repo_files: dict[str, str] = {}
        for fp in affected:
            content = _swe_fetch_file(item["repo"], item["base_commit"], fp)
            if content:
                repo_files[fp] = content
            _time.sleep(0.25)  # stay under GitHub rate limit (~4 req/s)
        item["repo_files"] = repo_files
        if repo_files:
            fetched += 1

        # Build the prompt — ask for a diff patch + explanation
        problem_text = item.pop("_problem", "")
        item["query"] = (
            f"Repository: {item['repo']}\n\n"
            f"Bug report:\n{problem_text[:600]}\n\n"
            f"Generate a unified diff patch (git format) to fix this bug. "
            f"Wrap the patch in a ```diff code block. "
            f"Also briefly explain the root cause."
        )

    print(f"  [SWE-bench] {len(balanced)} items selected, "
          f"{fetched}/{len(balanced)} have source files for execution scoring")
    return balanced


def load_mmlu(n: int) -> list[dict]:
    """MMLU: 57 subjects. Kept for backward compatibility — load_mmlu_pro is preferred."""
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
    """HumanEval: 164 Python function synthesis problems.

    Hard limit of 164 problems — supplement with load_mbpp() for larger datasets.
    Short prompts (<=300 chars) -> easy, longer -> medium.
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
        complexity = "easy" if len(prompt) <= 300 else "medium"
        results.append({
            "domain": "code", "complexity": complexity,
            "query": query, "ground_truth": ground_truth,
        })
    random.shuffle(results)
    return results[:n]


def load_mbpp(n: int) -> list[dict]:
    """MBPP (Mostly Basic Python Problems): 374 sanitized function-synthesis problems.

    Used to supplement HumanEval for code:easy since HumanEval only has 164 samples.
    The sanitized version uses consistent function names so tests can be run directly.
    Ground truth = assert statements -> execution-scored via _score_code().
    Complexity = easy (simple one-function problems).
    """
    from datasets import load_dataset
    for name, cfg, split in [
        ("google-research-datasets/mbpp", "sanitized", "test"),
        ("google-research-datasets/mbpp", "sanitized", "validation"),
        ("google-research-datasets/mbpp", None, "test"),
    ]:
        try:
            ds = load_dataset(name, cfg, split=split) if cfg else load_dataset(name, split=split)
            break
        except Exception as e:
            print(f"  [MBPP/{cfg}] skipped: {e}")
            ds = None

    if ds is None:
        print("  [MBPP] all sources failed, skipping")
        return []

    ds = ds.shuffle(seed=42)
    results = []
    for row in ds:
        text  = str(row.get("prompt", row.get("text", ""))).strip()
        tests = row.get("test_list", row.get("tests", []))
        code  = str(row.get("code", ""))
        if not text:
            continue
        if isinstance(tests, list):
            ground_truth = "\n".join(tests)
        else:
            ground_truth = str(tests)
        if not ground_truth and "assert" in code:
            ground_truth = "\n".join(l.strip() for l in code.splitlines() if "assert" in l)
        query = f"Write a Python function to solve the following problem:\n\n{text}"
        results.append({
            "domain": "code", "complexity": "easy",
            "query": query, "ground_truth": ground_truth,
        })
        if len(results) >= n:
            break

    return results[:n]


def load_bigcodebench(n: int) -> list[dict]:
    """BigCodeBench: real-world library usage (numpy, pandas, requests, etc.).

    Complexity mapping:
      difficulty <= 2 -> medium
      difficulty >= 3 -> hard
    """
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
        context  = row.get("context",  row.get("passage",  ""))
        question = row.get("query",    row.get("question", ""))
        options  = row.get("options",  row.get("answers",  []))
        labels   = ["A", "B", "C", "D"]
        opts_str = "\n".join(f"{labels[i]}) {options[i]}" for i in range(min(len(options), 4)))
        query    = f"Context: {context}\n\nQuestion: {question}\n{opts_str}"
        ans_idx  = row.get("correct_option", row.get("answer", 0))
        ground_truth = options[ans_idx] if isinstance(ans_idx, int) and ans_idx < len(options) else str(ans_idx)
        results.append({"domain": "reasoning", "complexity": "medium",
                        "query": query, "ground_truth": ground_truth})
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
        results.append({"domain": "reasoning", "complexity": "easy",
                        "query": query, "ground_truth": ground_truth})
    return results


def load_arc(n: int) -> list[dict]:
    """ARC: science MCQ. Easy->easy, Challenge->hard."""
    from datasets import load_dataset
    results = []
    for split_name, complexity, count in [("easy", "easy", n // 2), ("challenge", "hard", n - n // 2)]:
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
                results.append({"domain": "reasoning", "complexity": complexity,
                                "query": query, "ground_truth": ground_truth})
        except Exception as e:
            print(f"  [ARC-{split_name}] skipped: {e}")
    return results


def _strip_html(text: str) -> str:
    import re as _re
    text = _re.sub(r"<[^>]+>", " ", text)
    for entity, char in [("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                         ("&nbsp;", " "), ("&#39;", "'"), ("&quot;", '"')]:
        text = text.replace(entity, char)
    return _re.sub(r"\s{2,}", " ", text).strip()


def load_livecodebench(n: int, cutoff: str = CONTAMINATION_CUTOFF) -> list[dict]:
    """LiveCodeBench: competitive programming (LeetCode, Codeforces, AtCoder).

    Filters by contest_date >= cutoff — problems are outside all model training windows.
    Scored by subprocess execution with stdin/stdout comparison (pass@1).
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
    skipped_date = 0
    for row in ds:
        contest_date = str(row.get("contest_date", "") or row.get("start_date", ""))
        if contest_date and contest_date < cutoff:
            skipped_date += 1
            continue

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

    if skipped_date > 0:
        print(f"  [LiveCodeBench] skipped {skipped_date} items before cutoff {cutoff}")

    per_diff: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
    for item in results:
        per_diff[item["complexity"]].append(item)

    per_bucket = n // 3
    balanced: list[dict] = []
    for diff, pool in per_diff.items():
        want = per_bucket if diff != "hard" else n - 2 * per_bucket
        balanced.extend(random.sample(pool, min(want, len(pool))))

    return balanced


# ---------------------------------------------------------------------------
# Balance and build final dataset
# ---------------------------------------------------------------------------

# 2400 base samples across 4 domains (factual, math, code, reasoning).
# + 400 LiveCodeBench (post-cutoff, execution-scored)
# + 200 SWE-bench Verified (post-cutoff GitHub issues, with source files)
# = 3000 total
TARGET_DISTRIBUTION = {
    ("factual",   "easy"):   200,
    ("factual",   "medium"): 200,
    ("factual",   "hard"):   200,
    ("math",      "easy"):   200,
    ("math",      "medium"): 220,
    ("math",      "hard"):   180,
    ("code",      "easy"):   200,
    ("code",      "medium"): 220,
    ("code",      "hard"):   180,
    ("reasoning", "easy"):   200,
    ("reasoning", "medium"): 220,
    ("reasoning", "hard"):   180,
}


def build(total: int, output: str, lcb_count: int = 400,
          swe_count: int = 200, cutoff: str = CONTAMINATION_CUTOFF) -> None:
    print(f"\n  Building {total}-request dataset from HuggingFace benchmarks...")
    print(f"  Contamination cutoff : {cutoff}")
    print(f"  Output : {output}")
    print(f"  Breakdown: base={total - lcb_count - swe_count}"
          f" + LiveCodeBench={lcb_count} + SWE-bench={swe_count}\n")

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    print("  Loading base datasets:")
    all_items: list[dict] = []

    def add(name: str, items: list[dict]) -> None:
        print(f"    {name:<20} {len(items):4} items loaded")
        all_items.extend(items)

    add("MMLU-Pro",       load_mmlu_pro(650))
    add("GSM8K",          load_gsm8k(360))
    add("MATH benchmark", load_math_benchmark(260))
    add("HumanEval",      load_humaneval(164))   # hard limit: only 164 exist
    add("MBPP",           load_mbpp(374))         # supplements HumanEval for code:easy
    add("BigCodeBench",   load_bigcodebench(260))
    add("APPS",           load_apps(180))
    add("LogiQA",         load_logiqa(300))
    add("HellaSwag",      load_hellaswag(260))
    add("ARC",            load_arc(300))

    print(f"\n  Total raw base items: {len(all_items)}")

    base_target = total - lcb_count - swe_count
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for item in all_items:
        key = (item["domain"], item["complexity"])
        buckets[key].append(item)

    print("\n  Available per bucket (base):")
    for key in sorted(TARGET_DISTRIBUTION):
        avail  = len(buckets[key])
        want   = TARGET_DISTRIBUTION[key]
        status = "OK" if avail >= want else f"WARN only {avail}"
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

    if len(dataset) < base_target:
        extra = random.choices(all_items, k=base_target - len(dataset))
        dataset.extend(extra)
    dataset = dataset[:base_target]

    print(f"\n  Loading LiveCodeBench ({lcb_count} samples, cutoff={cutoff})...")
    lcb_items = load_livecodebench(lcb_count, cutoff=cutoff)
    print(f"    LiveCodeBench      {len(lcb_items):4} items loaded")

    if lcb_items:
        dataset.extend(lcb_items)
    else:
        print("  WARNING: LiveCodeBench unavailable or no items after cutoff -- padding")
        code_pool = [x for x in all_items if x["domain"] == "code"]
        dataset.extend(random.choices(code_pool, k=lcb_count))

    print(f"\n  Loading SWE-bench Verified ({swe_count} samples, cutoff={cutoff})...")
    print(f"  NOTE: fetching source files from GitHub takes ~{swe_count*3*0.25:.0f}s")
    swe_items = load_swe_bench(swe_count, cutoff=cutoff)
    print(f"    SWE-bench          {len(swe_items):4} items loaded")

    if swe_items:
        dataset.extend(swe_items)
    else:
        print("  WARNING: SWE-bench unavailable or no items after cutoff -- padding")
        code_pool = [x for x in all_items if x["domain"] == "code"]
        dataset.extend(random.choices(code_pool, k=swe_count))

    random.shuffle(dataset)

    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n  Final dataset: {len(dataset)} requests")

    by_domain: dict[str, int] = defaultdict(int)
    by_cplx:   dict[str, int] = defaultdict(int)
    by_source: dict[str, int] = defaultdict(int)
    swe_with_files = 0
    for item in dataset:
        by_domain[item["domain"]]           += 1
        by_cplx[item["complexity"]]         += 1
        by_source[item.get("source", "hf")] += 1
        if item.get("source") == "swe_bench" and item.get("repo_files"):
            swe_with_files += 1

    print("\n  By domain:")
    for d, c in sorted(by_domain.items()):
        print(f"    {d:<12} {c:4}")
    print("\n  By complexity:")
    for c, n2 in sorted(by_cplx.items()):
        print(f"    {c:<8} {n2:4}")
    print("\n  By source:")
    for s, n2 in sorted(by_source.items()):
        print(f"    {s:<20} {n2:4}")
    print(f"\n  Contamination cutoff : {cutoff}")
    print(f"  Post-cutoff items    : LiveCodeBench={len(lcb_items)}, SWE-bench={len(swe_items)}")
    print(f"  SWE-bench with files : {swe_with_files}/{len(swe_items)} "
          f"(items with source files for execution scoring)")
    print(f"\n  Saved to: {output}")
    print(f"  Run load test with:")
    print(f"    python tests/load_test.py --dataset {output}\n")
    print(f"  Evaluate with real SWE-bench execution:")
    print(f"    python tests/eval_all_models.py --dataset {output}           # subprocess scorer")
    print(f"    python tests/eval_all_models.py --dataset {output} --use-docker  # Docker scorer\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",    default="datasets/hf_3000.json")
    parser.add_argument("--count",     type=int, default=3000,
                        help="Total samples (base + LiveCodeBench + SWE-bench)")
    parser.add_argument("--lcb-count", type=int, default=400,
                        help="LiveCodeBench samples (post-cutoff, execution-scored)")
    parser.add_argument("--swe-count", type=int, default=200,
                        help="SWE-bench Verified samples (post-cutoff, with source files)")
    parser.add_argument("--cutoff",    default=CONTAMINATION_CUTOFF,
                        help=f"Contamination cutoff YYYY-MM-DD (default: {CONTAMINATION_CUTOFF}). "
                             "Excludes LiveCodeBench/SWE-bench problems before this date.")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    build(args.count, args.output,
          lcb_count=args.lcb_count,
          swe_count=args.swe_count,
          cutoff=args.cutoff)


if __name__ == "__main__":
    main()
