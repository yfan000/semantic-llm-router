"""
accuracy_sampler.py — Async accuracy judge pipeline.

Uses already-running vLLM endpoints as judges via HTTP (no extra model downloads).
Cross-family pairing avoids self-enhancement bias:
  e.g. phi-3.5-mini judges qwen2.5-1.5b responses, and vice versa.

Configure judge endpoints in config.py:
    JUDGE_ENDPOINTS = {
        "qwen2.5-1.5b": ("http://localhost:8002", "phi-3.5-mini"),
        "phi-3.5-mini":  ("http://localhost:8001", "qwen2.5-1.5b"),
    }

Math and code are scored deterministically — no LLM judge needed.
"""
from __future__ import annotations
import asyncio
import logging
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from semantic_router.config import (
    SAMPLE_RATE, JUDGE_BATCH_SIZE, JUDGE_BATCH_WINDOW_S, JUDGE_QUEUE_MAX,
    JUDGE_ENDPOINTS,
)

if TYPE_CHECKING:
    from semantic_router.reputation_tracker import ReputationTracker

log = logging.getLogger(__name__)

RUBRICS = {
    "factual": (
        "Score 1.0 if all stated facts are accurate and complete. "
        "Score 0.5 if partially correct or incomplete. Score 0.0 if factually wrong."
    ),
    "reasoning": (
        "Score 1.0 if the reasoning chain is valid and the conclusion is correct. "
        "Deduct 0.3 per logical error. Score 0.0 if the conclusion is wrong."
    ),
    "creative": (
        "Score 1.0 if the response is coherent, creative, and relevant to the prompt. "
        "Score 0.5 if partially relevant. Score 0.0 if off-topic or incoherent."
    ),
}


@dataclass
class SampleItem:
    model_id: str
    domain: str
    complexity: str
    query: str
    response: str


# ---------------------------------------------------------------------------
# Deterministic scorers
# ---------------------------------------------------------------------------

def _score_math(response: str, query: str) -> float:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    return 1.0 if numbers else 0.0


def _score_code(response: str) -> float:
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    code = m.group(1) if m else response
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(
            ["python3", "-c", f"import ast; ast.parse(open('{path}').read())"],
            timeout=2, capture_output=True,
        )
        if r.returncode != 0:
            return 0.0
        r = subprocess.run(["python3", path], timeout=5, capture_output=True)
        return 1.0 if r.returncode == 0 else 0.3
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# LLM judge via HTTP
# ---------------------------------------------------------------------------

async def _llm_judge(
    model_id: str, domain: str, query: str, response: str
) -> float | None:
    if model_id not in JUDGE_ENDPOINTS:
        return None

    judge_url, judge_model = JUDGE_ENDPOINTS[model_id]
    rubric = RUBRICS.get(domain, RUBRICS["factual"])

    prompt = (
        f"You are a fair evaluator. Rubric: {rubric}\n"
        f"Query: {query}\n"
        f"Response: {response}\n"
        f'Give a score from 0.0 to 1.0. Return ONLY: {{"score": <float>}}'
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.0,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{judge_url}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
        if m:
            return min(max(float(m.group(1)), 0.0), 1.0)
        m = re.search(r"\b([01](?:\.\d+)?)\b", text)
        return float(m.group(1)) if m else None
    except Exception as e:
        log.debug("LLM judge call failed for %s: %s", model_id, e)
        return None


# ---------------------------------------------------------------------------
# AccuracySampler
# ---------------------------------------------------------------------------

class AccuracySampler:
    def __init__(self, reputation_tracker: "ReputationTracker") -> None:
        self._tracker = reputation_tracker
        self._queue: asyncio.Queue[SampleItem] = asyncio.Queue(maxsize=JUDGE_QUEUE_MAX)

    def enqueue(self, item: SampleItem) -> None:
        if random.random() >= SAMPLE_RATE:
            return
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def configure_judges(self, endpoints: dict[str, tuple[str, str]]) -> None:
        """Set cross-family judge endpoints dynamically after model registration."""
        JUDGE_ENDPOINTS.update(endpoints)
        log.info("Judge endpoints updated: %s", list(JUDGE_ENDPOINTS.keys()))

    async def _score_item(self, item: SampleItem) -> float:
        if item.domain == "math":
            return await asyncio.get_event_loop().run_in_executor(
                None, _score_math, item.response, item.query
            )
        if item.domain == "code":
            return await asyncio.get_event_loop().run_in_executor(
                None, _score_code, item.response
            )
        score = await _llm_judge(item.model_id, item.domain, item.query, item.response)
        if score is not None:
            return score
        log.debug("No judge configured for %s — set JUDGE_ENDPOINTS in config.py.", item.model_id)
        return -1.0  # sentinel: skip EMA update

    async def _process_batch(self, batch: list[SampleItem]) -> None:
        for item in batch:
            try:
                score = await self._score_item(item)
                if score >= 0.0:
                    self._tracker.update_accuracy_prior(
                        item.model_id, item.domain, item.complexity, score
                    )
            except Exception as e:
                log.warning("Judge failed for %s: %s", item.model_id, e)

    async def run(self) -> None:
        while True:
            batch: list[SampleItem] = []
            deadline = asyncio.get_event_loop().time() + JUDGE_BATCH_WINDOW_S
            while len(batch) < JUDGE_BATCH_SIZE:
                timeout = deadline - asyncio.get_event_loop().time()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            if batch:
                await self._process_batch(batch)
