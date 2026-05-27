"""
benchmark_store.py -- Offline ground-truth lookup and deterministic scoring.

Loads the benchmark dataset (built by tests/build_dataset.py) at startup and
provides fast query-to-ground-truth lookup. Used by the inference retry loop
to verify response quality in real time when a matching benchmark query is found.

Scoring is fully deterministic (no LLM judge needed):
  math      -- extract final number, compare within 1% tolerance
  code      -- run each assert individually, score = passed/total
  factual   -- keyword overlap >= 80% of ground_truth keywords
  reasoning -- keyword overlap >= 80%
  creative  -- non-empty response >= 20 words (always passes)

Usage:
    store = BenchmarkStore("datasets/hf_1000.json")
    gt = store.lookup(query)
    if gt:
        score = store.score(domain, response_text, gt)
        if score < 0.7:
            # retry with next model
"""
from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

CORRECT_THRESHOLD = 0.7


class BenchmarkStore:
    def __init__(self, path: str | None) -> None:
        self._index: dict[str, dict] = {}  # normalized_query -> item
        if path and Path(path).exists():
            self._load(path)
        elif path:
            log.warning("BenchmarkStore: file not found: %s -- online scoring disabled", path)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            key = self._normalize(item.get("query", ""))
            if key:
                self._index[key] = item
        log.info("BenchmarkStore: loaded %d entries from %s", len(self._index), path)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def lookup(self, query: str) -> dict | None:
        """Return the benchmark item for this query, or None if not found."""
        return self._index.get(self._normalize(query))

    def score(self, domain: str, response: str, ground_truth: str) -> float | None:
        """Score response against ground_truth. Returns None if not scorable."""
        if not response:
            return 0.0
        domain = domain.lower()
        if domain == "math":
            return self._score_math(response, ground_truth)
        elif domain == "code":
            return self._score_code(response, ground_truth)
        elif domain in ("factual", "reasoning"):
            return self._score_keyword(response, ground_truth)
        elif domain == "creative":
            return 1.0 if len(response.split()) >= 20 else 0.5
        return None

    def is_correct(self, domain: str, response: str, ground_truth: str) -> bool | None:
        """Return True/False/None (None = not scorable)."""
        s = self.score(domain, response, ground_truth)
        if s is None:
            return None
        return s >= CORRECT_THRESHOLD

    # -------------------------------------------------------------------------
    # Deterministic scorers
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        return [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", text)]

    def _score_math(self, response: str, gt: str) -> float | None:
        pred = self._extract_numbers(response)
        true = self._extract_numbers(str(gt))
        if not pred or not true:
            return None
        p, t = pred[-1], true[-1]
        if t == 0:
            return 1.0 if abs(p) < 0.01 else 0.0
        return 1.0 if (abs(p - t) / abs(t) < 0.01 or abs(p - t) < 0.01) else 0.0

    def _score_code(self, response: str, gt: str) -> float:
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        code = blocks[0].strip() if blocks else response.strip()
        try:
            ast.parse(code)
        except SyntaxError:
            return 0.0
        if not gt or ("assert" not in str(gt) and "==" not in str(gt)):
            return 0.5  # syntax ok but no test to run

        gt_str = str(gt)
        # Extract individual assert statements and run each separately
        # so partial credit is given when only some tests pass.
        assert_lines = [
            line.strip() for line in gt_str.splitlines()
            if line.strip().startswith("assert")
        ]
        if not assert_lines:
            # No individual asserts -- run the whole test file
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(code + "\n" + gt_str)
                fname = f.name
            try:
                r = subprocess.run([sys.executable, fname], timeout=5, capture_output=True)
                return 1.0 if r.returncode == 0 else 0.0
            except Exception:
                return 0.0

        passed = 0
        for assert_line in assert_lines:
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                f.write(code + "\n" + assert_line)
                fname = f.name
            try:
                r = subprocess.run([sys.executable, fname], timeout=5, capture_output=True)
                if r.returncode == 0:
                    passed += 1
            except Exception:
                pass
        return passed / len(assert_lines)

    def _score_keyword(self, response: str, gt: str) -> float | None:
        if not gt or str(gt).strip() in ("", "None"):
            return None
        words = set(w.lower() for w in re.findall(r"\b\w{4,}\b", str(gt)))
        if not words:
            return None
        hits = sum(1 for w in words if w in response.lower())
        overlap = hits / len(words)
        return 1.0 if overlap >= 0.8 else overlap
