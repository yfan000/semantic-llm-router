from __future__ import annotations
import asyncio, logging, random, re, subprocess, tempfile, textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from semantic_router.config import (
    SAMPLE_RATE, JUDGE_BATCH_SIZE, JUDGE_BATCH_WINDOW_S, JUDGE_QUEUE_MAX,
    CALIBRATION_RATE, CALIBRATION_DRIFT_THRESHOLD,
)

if TYPE_CHECKING:
    from semantic_router.reputation_tracker import ReputationTracker

log = logging.getLogger(__name__)

RUBRICS = {
    "factual":   "Score 1.0 if all stated facts are accurate and complete. Score 0.5 if partially correct. Score 0.0 if factually wrong.",
    "reasoning": "Score 1.0 if the reasoning chain is valid and the conclusion correct. Deduct 0.3 per logical error. Score 0.0 if conclusion is wrong.",
    "creative":  "Score 1.0 if coherent, creative, and relevant. Score 0.5 if partially relevant. Score 0.0 if off-topic.",
}


@dataclass
class SampleItem:
    model_id: str
    domain: str
    complexity: str
    query: str
    response: str
    bid_accuracy: float = 0.70  # model's self-reported estimated_accuracy at bid time


def _score_code(response: str) -> float:
    code = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    code = code.group(1) if code else response
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    r = subprocess.run(["python", "-c", f"import ast; ast.parse(open('{path}').read())"], timeout=2, capture_output=True)
    if r.returncode != 0: return 0.0
    return 1.0 if subprocess.run(["python", path], timeout=5, capture_output=True).returncode == 0 else 0.3


def _score_math(response: str, query: str) -> float:
    return 1.0 if re.findall(r"-?\d+(?:\.\d+)?", response) else 0.0


class AccuracySampler:
    def __init__(self, reputation_tracker: "ReputationTracker") -> None:
        self._tracker = reputation_tracker
        self._queue: asyncio.Queue[SampleItem] = asyncio.Queue(maxsize=JUDGE_QUEUE_MAX)
        self._judge_model = self._creative_model = None
        self._calib_pairs: list[tuple[float, float]] = []

    def enqueue(self, item: SampleItem) -> None:
        if random.random() >= SAMPLE_RATE: return
        try: self._queue.put_nowait(item)
        except asyncio.QueueFull: pass

    async def _load_judge(self) -> None:
        if self._judge_model is not None: return
        from transformers import pipeline
        self._judge_model    = pipeline("text-generation", model="kaist-ai/prometheus-2-7b",   device_map="auto", max_new_tokens=64)
        self._creative_model = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct", device_map="auto", max_new_tokens=64)

    def _prometheus_score(self, domain: str, query: str, response: str) -> float:
        prompt = textwrap.dedent(f"""
            [INST] Rubric: {RUBRICS.get(domain, RUBRICS['factual'])}
            Query: {query}\nResponse: {response}
            Score 0.0-1.0. Output: {{"score": <float>, "reason": "<str>"}} [/INST]""").strip()
        out = self._judge_model(prompt)[0]["generated_text"]
        m = re.search(r'"score"\s*:\s*([0-9.]+)', out)
        return float(m.group(1)) if m else 0.5

    def _qwen_score(self, query: str, response: str) -> float:
        out = self._creative_model(
            f"Rubric: {RUBRICS['creative']}\nQuery: {query}\nResponse: {response}\n"
            f'Rate 0.0-1.0. Return only {{"score": <float>}}.'
        )[0]["generated_text"]
        m = re.search(r'"score"\s*:\s*([0-9.]+)', out)
        return float(m.group(1)) if m else 0.5

    async def _score_item(self, item: SampleItem) -> float:
        loop = asyncio.get_event_loop()
        if item.domain == "code":     return await loop.run_in_executor(None, _score_code, item.response)
        if item.domain == "math":     return await loop.run_in_executor(None, _score_math, item.response, item.query)
        await self._load_judge()
        if item.domain == "creative": return await loop.run_in_executor(None, self._qwen_score, item.query, item.response)
        return await loop.run_in_executor(None, self._prometheus_score, item.domain, item.query, item.response)

    async def _process_batch(self, batch: list[SampleItem]) -> None:
        for item in batch:
            try:
                score = await self._score_item(item)
                # Update eligibility floor (accuracy_priors EMA)
                self._tracker.update_accuracy_prior(
                    item.model_id, item.domain, item.complexity, score
                )
                # Update bid reliability penalty (accuracy_bid_reliability EMA)
                self._tracker.record_accuracy_bid(
                    item.model_id, item.domain, item.complexity,
                    item.bid_accuracy, score,
                )
                if random.random() < CALIBRATION_RATE:
                    asyncio.create_task(self._calibrate(item, score))
            except Exception as e:
                log.warning("Judge failed for %s: %s", item.model_id, e)

    async def _calibrate(self, item: SampleItem, p_score: float) -> None:
        try:
            import httpx, os
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.post("https://api.openai.com/v1/chat/completions",
                    json={"model": "gpt-4o-mini", "messages": [
                        {"role": "system", "content": "Rate 0.0-1.0. Return only a float."},
                        {"role": "user", "content": f"Query: {item.query}\nResponse: {item.response}"}],
                    "max_tokens": 10},
                    headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"})
            gpt_score = float(re.search(r"[0-9.]+", r.json()["choices"][0]["message"]["content"]).group())
            self._calib_pairs.append((p_score, gpt_score))
            if len(self._calib_pairs) >= 1000:
                mae = sum(abs(a-b) for a,b in self._calib_pairs) / len(self._calib_pairs)
                if mae > CALIBRATION_DRIFT_THRESHOLD:
                    log.warning("Prometheus-2 calibration drift: MAE=%.3f", mae)
                self._calib_pairs.clear()
        except Exception as e:
            log.debug("Calibration failed: %s", e)

    async def run(self) -> None:
        while True:
            batch: list[SampleItem] = []
            deadline = asyncio.get_event_loop().time() + JUDGE_BATCH_WINDOW_S
            while len(batch) < JUDGE_BATCH_SIZE:
                t = deadline - asyncio.get_event_loop().time()
                if t <= 0: break
                try: batch.append(await asyncio.wait_for(self._queue.get(), timeout=t))
                except asyncio.TimeoutError: break
            if batch: await self._process_batch(batch)
