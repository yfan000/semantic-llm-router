"""
build_dataset.py — Download real benchmark queries from HuggingFace and
build a balanced 1000-request test set for the semantic LLM router.

Sources:
  Factual  : MMLU (57 subjects, graded by difficulty)
  Math     : GSM8K (grade-school, medium) + MATH benchmark (competition, hard)
  Code     : HumanEval (medium) + MBPP (easy)
  Reasoning: LogiQA (medium) + HellaSwag (easy) + ARC (easy/hard)
  Creative : WritingPrompts (medium/hard)

Install:
    pip install datasets

Usage:
    python tests/build_dataset.py                    # saves to datasets/hf_1000.json
    python tests/build_dataset.py --output my.json
    python tests/build_dataset.py --count 500

Use with load_test.py:
    python tests/load_test.py --dataset datasets/hf_1000.json
"""
from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict


def load_mmlu(n: int) -> list[dict]:
    from datasets import load_dataset
    EASY   = ["high_school_geography","high_school_us_history","high_school_biology",
               "high_school_chemistry","high_school_physics","elementary_mathematics","nutrition"]
    MEDIUM = ["college_biology","college_chemistry","college_computer_science",
               "college_mathematics","college_physics","anatomy","astronomy"]
    HARD   = ["professional_medicine","professional_law","professional_accounting",
               "medical_genetics","abstract_algebra","formal_logic","logical_fallacies"]
    results = []
    per = n // 3
    for subjects, complexity, count in [(EASY,"easy",per),(MEDIUM,"medium",per),(HARD,"hard",n-2*per)]:
        ps = max(1, count // len(subjects))
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                ds = ds.shuffle(seed=42).select(range(min(ps, len(ds))))
                for row in ds:
                    choices = row["choices"]
                    opts = "\n".join(f"{['A','B','C','D'][i]}) {choices[i]}" for i in range(len(choices)))
                    results.append({"domain":"factual","complexity":complexity,
                                    "query": f"{row['question']}\n{opts}"})
            except Exception as e:
                print(f"  [MMLU/{subj}] skipped: {e}")
    return random.sample(results, min(n, len(results)))


def load_gsm8k(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k","main",split="test",trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(n,len(ds))))
    return [{"domain":"math","complexity":"medium","query":row["question"]} for row in ds]


def load_math_benchmark(n: int) -> list[dict]:
    from datasets import load_dataset
    CMAP = {"1":"easy","2":"easy","3":"medium","4":"medium","5":"hard"}
    for name in ["lighteval/MATH","hendrycks/competition_math"]:
        try:
            ds = load_dataset(name,"all",split="test",trust_remote_code=True)
            ds = ds.shuffle(seed=42).select(range(min(n*3,len(ds))))
            results = []
            for row in ds:
                lv = str(row.get("level","3")).replace("Level ","")
                results.append({"domain":"math","complexity":CMAP.get(lv,"medium"),"query":row["problem"]})
            return random.sample(results, min(n, len(results)))
        except Exception as e:
            print(f"  [MATH/{name}] skipped: {e}")
    return []


def load_humaneval(n: int) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/openai_humaneval",split="test",trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(n,len(ds))))
        return [{"domain":"code","complexity":"medium",
                 "query":f"Complete the following Python function:\n\n{row['prompt'].strip()}"} for row in ds]
    except Exception as e:
        print(f"  [HumanEval] skipped: {e}"); return []


def load_mbpp(n: int) -> list[dict]:
    from datasets import load_dataset
    for name,cfg in [("google-research-datasets/mbpp","sanitized"),("mbpp","full")]:
        try:
            ds = load_dataset(name,cfg,split="test",trust_remote_code=True)
            ds = ds.shuffle(seed=42).select(range(min(n,len(ds))))
            return [{"domain":"code","complexity":"easy",
                     "query":f"Write a Python function to: {row['text']}"} for row in ds]
        except Exception as e:
            print(f"  [MBPP/{name}] skipped: {e}")
    return []


def load_apps(n: int) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("codeparrot/apps","all",split="test",trust_remote_code=True)
        ds = ds.filter(lambda x: x.get("difficulty")=="interview")
        ds = ds.shuffle(seed=42).select(range(min(n,len(ds))))
        return [{"domain":"code","complexity":"hard","query":row.get("question","")[:500]} for row in ds]
    except Exception as e:
        print(f"  [APPS] skipped: {e}"); return []


def load_logiqa(n: int) -> list[dict]:
    from datasets import load_dataset
    for name,split in [("lucasmccabe/logiqa","test"),("logiqa","validation")]:
        try:
            ds = load_dataset(name,split=split,trust_remote_code=True)
            ds = ds.shuffle(seed=42).select(range(min(n,len(ds))))
            results = []
            for row in ds:
                ctx  = row.get("context", row.get("passage",""))
                q    = row.get("query",   row.get("question",""))
                opts = row.get("options", row.get("answers",[]))
                L    = ["A","B","C","D"]
                o    = "\n".join(f"{L[i]}) {opts[i]}" for i in range(min(len(opts),4)))
                results.append({"domain":"reasoning","complexity":"medium",
                                 "query":f"Context: {ctx}\n\nQuestion: {q}\n{o}"})
            return results
        except Exception as e:
            print(f"  [LogiQA/{name}] skipped: {e}")
    return []


def load_hellaswag(n: int) -> list[dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag",split="validation",trust_remote_code=True)
        ds = ds.shuffle(seed=42).select(range(min(n,len(ds))))
        results = []
        for row in ds:
            L    = ["A","B","C","D"]
            opts = "\n".join(f"{L[i]}) {row['endings'][i]}" for i in range(len(row["endings"])))
            results.append({"domain":"reasoning","complexity":"easy",
                             "query":f"Choose the best continuation:\n\n{row['ctx'].strip()}\n\n{opts}"})
        return results
    except Exception as e:
        print(f"  [HellaSwag] skipped: {e}"); return []


def load_arc(n: int) -> list[dict]:
    from datasets import load_dataset
    results = []
    for sname, complexity, count in [("Easy","easy",n//2),("Challenge","hard",n-n//2)]:
        try:
            ds = load_dataset("allenai/ai2_arc",f"ARC-{sname}",split="test",trust_remote_code=True)
            ds = ds.shuffle(seed=42).select(range(min(count,len(ds))))
            for row in ds:
                c   = row["choices"]
                opts = "\n".join(f"{c['label'][i]}) {c['text'][i]}" for i in range(len(c["label"])))
                results.append({"domain":"reasoning","complexity":complexity,
                                 "query":f"{row['question']}\n{opts}"})
        except Exception as e:
            print(f"  [ARC/{sname}] skipped: {e}")
    return results


def load_writing_prompts(n: int) -> list[dict]:
    from datasets import load_dataset
    for name in ["euclaise/writingprompts","writing_prompts"]:
        try:
            ds = load_dataset(name,split="train",trust_remote_code=True)
            ds = ds.shuffle(seed=42).select(range(min(n*3,len(ds))))
            results = []
            for row in ds:
                p = row.get("prompt",row.get("story","")).strip()
                p = p.replace("[WP]","").replace("[SP]","").replace("[EU]","").strip()
                if 20 <= len(p) <= 600:
                    cplx = "hard" if len(p.split()) > 40 else "medium"
                    results.append({"domain":"creative","complexity":cplx,
                                    "query":f"Write a short story or poem for this prompt: {p}"})
            return random.sample(results, min(n, len(results)))
        except Exception as e:
            print(f"  [WritingPrompts/{name}] skipped: {e}")
    # Fallback
    fallback = [
        ("Write a short story about a robot who discovers music.", "medium"),
        ("Write a poem about the passage of time using ocean imagery.", "medium"),
        ("Create a product description for an AI-powered dream recorder.", "medium"),
        ("Write a dialogue between the sun and the moon.", "easy"),
        ("Write a story from the perspective of the last tree in a city.", "hard"),
        ("Create a myth explaining why humans dream.", "medium"),
        ("Write a short story where the twist is revealed in the last word.", "hard"),
        ("Compose a haiku about artificial intelligence.", "easy"),
        ("Write a letter from a future version of yourself.", "medium"),
        ("Write a story about a painter who can only paint tomorrow.", "hard"),
    ]
    results = []
    for p, c in fallback * (n // len(fallback) + 1):
        results.append({"domain":"creative","complexity":c,"query":p})
    return results[:n]


TARGET = {
    ("factual","easy"):80, ("factual","medium"):70, ("factual","hard"):50,
    ("math","easy"):50,    ("math","medium"):80,    ("math","hard"):70,
    ("code","easy"):60,    ("code","medium"):80,    ("code","hard"):50,
    ("reasoning","easy"):70, ("reasoning","medium"):80, ("reasoning","hard"):50,
    ("creative","easy"):40,  ("creative","medium"):80,  ("creative","hard"):40,
}


def build(total: int, output: str) -> None:
    print(f"\n  Building {total}-request dataset from HuggingFace benchmarks...")
    all_items: list[dict] = []

    def add(name, items):
        print(f"  {name:<22} {len(items):4} items")
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
    add("WritingPrompts", load_writing_prompts(150))

    print(f"\n  Total raw items: {len(all_items)}")

    buckets: dict = defaultdict(list)
    for item in all_items:
        buckets[(item["domain"], item["complexity"])].append(item)

    dataset: list[dict] = []
    for key, want in TARGET.items():
        pool = buckets[key]
        if len(pool) >= want:
            dataset.extend(random.sample(pool, want))
        else:
            dataset.extend(pool)
            if pool:
                dataset.extend(random.choices(pool, k=want - len(pool)))

    if len(dataset) < total and all_items:
        dataset.extend(random.choices(all_items, k=total - len(dataset)))

    random.shuffle(dataset)
    dataset = dataset[:total]

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(dataset, f, indent=2)

    by_d: dict[str, int] = defaultdict(int)
    by_c: dict[str, int] = defaultdict(int)
    for item in dataset:
        by_d[item["domain"]] += 1
        by_c[item["complexity"]] += 1

    print(f"\n  Final: {len(dataset)} requests")
    print("  By domain:    " + "  ".join(f"{d}:{n}" for d,n in sorted(by_d.items())))
    print("  By complexity:" + "  ".join(f"{c}:{n}" for c,n in sorted(by_c.items())))
    print(f"\n  Saved: {output}")
    print(f"  Usage: python tests/load_test.py --dataset {output}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="datasets/hf_1000.json")
    p.add_argument("--count",  type=int, default=1000)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)
    build(args.count, args.output)


if __name__ == "__main__":
    main()
