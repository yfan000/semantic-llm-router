"""
calibration.py — Measures per-domain accuracy for a model at registration time.

Runs a small curated benchmark (~10 prompts per domain) against the model's
vLLM endpoint, scores each response, and returns:
    {"factual": 0.82, "math": 0.45, "code": 0.75, "reasoning": 0.60, "creative": 0.65}

These become the model's min_accuracy_capability per domain.
Creative domain uses a lightweight rubric check since there is no ground truth.
"""
from __future__ import annotations
import asyncio
import logging
import re
import subprocess
import tempfile
import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark prompts with ground-truth answers
# ---------------------------------------------------------------------------

_FACTUAL: list[dict] = [
    {"q": "What is the chemical symbol for gold?\nA) Ag  B) Au  C) Fe  D) Cu", "a": "B"},
    {"q": "Which planet is closest to the Sun?\nA) Venus  B) Earth  C) Mercury  D) Mars", "a": "C"},
    {"q": "Who wrote 'Romeo and Juliet'?\nA) Dickens  B) Austen  C) Shakespeare  D) Hemingway", "a": "C"},
    {"q": "What is the speed of light in km/s (approx)?\nA) 100,000  B) 300,000  C) 500,000  D) 1,000,000", "a": "B"},
    {"q": "Which organ produces insulin?\nA) Liver  B) Kidney  C) Pancreas  D) Spleen", "a": "C"},
    {"q": "What is the capital of Australia?\nA) Sydney  B) Melbourne  C) Brisbane  D) Canberra", "a": "D"},
    {"q": "How many bones are in the adult human body?\nA) 106  B) 206  C) 306  D) 406", "a": "B"},
    {"q": "What gas do plants absorb during photosynthesis?\nA) Oxygen  B) Nitrogen  C) Carbon dioxide  D) Hydrogen", "a": "C"},
    {"q": "Which element has atomic number 1?\nA) Helium  B) Hydrogen  C) Carbon  D) Lithium", "a": "B"},
    {"q": "What is the largest ocean on Earth?\nA) Atlantic  B) Indian  C) Arctic  D) Pacific", "a": "D"},
]

_MATH: list[dict] = [
    {"q": "A store sells apples for $0.50 each. If you buy 12 apples, how much do you pay in dollars?", "a": "6"},
    {"q": "A train travels 60 km/h for 2.5 hours. How many km does it travel?", "a": "150"},
    {"q": "If 3x + 7 = 22, what is x?", "a": "5"},
    {"q": "A rectangle is 8 m wide and 5 m tall. What is its area in square metres?", "a": "40"},
    {"q": "There are 24 students in a class. 1/3 are absent today. How many are present?", "a": "16"},
    {"q": "What is 15% of 200?", "a": "30"},
    {"q": "A car uses 8 litres per 100 km. How many litres for a 350 km trip?", "a": "28"},
    {"q": "If you invest $1000 at 5% simple interest for 3 years, how much interest do you earn?", "a": "150"},
    {"q": "The sum of three consecutive integers is 81. What is the smallest?", "a": "26"},
    {"q": "A circle has radius 7 cm. What is its area? (Use pi=3.14, round to nearest integer.)", "a": "154"},
]

_CODE: list[dict] = [
    {"q": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
     "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n"},
    {"q": "Write a Python function `is_even(n)` that returns True if n is even, False otherwise.",
     "test": "assert is_even(4) == True\nassert is_even(7) == False\nassert is_even(0) == True\n"},
    {"q": "Write a Python function `reverse_string(s)` that returns the string reversed.",
     "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('') == ''\n"},
    {"q": "Write a Python function `factorial(n)` that returns n! for non-negative integers.",
     "test": "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(3) == 6\n"},
    {"q": "Write a Python function `max_of_list(lst)` that returns the maximum value in a list.",
     "test": "assert max_of_list([3, 1, 4, 1, 5]) == 5\nassert max_of_list([-1, -5, -2]) == -1\n"},
    {"q": "Write a Python function `count_vowels(s)` that counts vowels (a,e,i,o,u) in a string (case-insensitive).",
     "test": "assert count_vowels('hello') == 2\nassert count_vowels('AEIOU') == 5\nassert count_vowels('xyz') == 0\n"},
    {"q": "Write a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
     "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\n"},
    {"q": "Write a Python function `flatten(lst)` that flattens one level of a nested list.",
     "test": "assert flatten([[1,2],[3,4]]) == [1,2,3,4]\nassert flatten([[1],[2,3],[4]]) == [1,2,3,4]\n"},
    {"q": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed, fib(0)=0).",
     "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(6) == 8\n"},
    {"q": "Write a Python function `second_largest(lst)` that returns the second largest unique value.",
     "test": "assert second_largest([1,2,3,4]) == 3\nassert second_largest([5,5,4]) == 4\n"},
]

_REASONING: list[dict] = [
    {"q": "All mammals are warm-blooded. Whales are mammals. Which conclusion must be true?\nA) Whales are cold-blooded  B) Whales are warm-blooded  C) Whales are fish  D) Mammals are whales", "a": "B"},
    {"q": "If it rains, the ground is wet. The ground is not wet. What can we conclude?\nA) It is raining  B) It did not rain  C) The ground is dry  D) B and C", "a": "D"},
    {"q": "Every dog in the shelter has been vaccinated. Max is a dog in the shelter. Is Max vaccinated?\nA) Yes  B) No  C) Cannot determine  D) Only if Max is old", "a": "A"},
    {"q": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?\nA) Alice  B) Bob  C) Carol  D) Cannot determine", "a": "C"},
    {"q": "All prime numbers greater than 2 are odd. 7 is a prime number greater than 2. Therefore:\nA) 7 is even  B) 7 is odd  C) 7 is not prime  D) 2 is odd", "a": "B"},
    {"q": "Some cats are black. All black animals are nocturnal. Which must be true?\nA) All cats are nocturnal  B) Some cats are nocturnal  C) No cats are nocturnal  D) All nocturnal animals are cats", "a": "B"},
    {"q": "If a shape is a square, it is a rectangle. Shape X is not a rectangle. What can we say?\nA) X is a square  B) X is not a square  C) X is a circle  D) Cannot determine", "a": "B"},
    {"q": "Either the meeting is on Monday or Tuesday. The meeting is not on Monday. When is it?\nA) Monday  B) Tuesday  C) Wednesday  D) Cannot determine", "a": "B"},
    {"q": "All students who passed studied hard. Jane did not study hard. Did Jane pass?\nA) Yes  B) No  C) Maybe  D) Only if she is smart", "a": "B"},
    {"q": "A is twice as old as B. B is 10 years old. How old is A?\nA) 5  B) 10  C) 15  D) 20", "a": "D"},
]

_CREATIVE: list[dict] = [
    {"q": "Write a two-sentence story about a robot learning to cook.", "keywords": ["robot", "cook", "food", "kitchen", "recipe", "meal", "learned"]},
    {"q": "Write a haiku about the ocean.", "keywords": ["ocean", "sea", "wave", "water", "shore", "deep", "tide", "blue", "salt"]},
    {"q": "Write one sentence describing the feeling of rain.", "keywords": ["rain", "drops", "wet", "sky", "fall", "pour", "gentle", "cold", "fresh"]},
    {"q": "In one sentence, create a metaphor for time.", "keywords": ["time", "river", "arrow", "thief", "sand", "clock", "flows", "passes", "runs"]},
    {"q": "Write a two-sentence product description for a magic pen.", "keywords": ["pen", "write", "ink", "magic", "words", "create", "draw", "spell"]},
]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_mcq(response: str, answer: str) -> float:
    text = response.strip()[:200].upper()
    if re.search(rf'\b{answer}\b[).:\s]|^{answer}$|ANSWER.*\b{answer}\b|\b{answer}\b.*CORRECT', text):
        return 1.0
    m = re.search(r'\b([A-D])\b', text)
    return 1.0 if m and m.group(1) == answer else 0.0


def _score_math(response: str, answer: str) -> float:
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
    if not numbers:
        return 0.0
    try:
        return 1.0 if abs(float(numbers[-1]) - float(answer)) < 0.01 else 0.0
    except ValueError:
        return 0.0


def _score_code(response: str, test_code: str) -> float:
    m = re.search(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    code = m.group(1) if m else response
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + test_code)
        path = f.name
    try:
        r = subprocess.run(["python", path], timeout=5, capture_output=True)
        return 1.0 if r.returncode == 0 else 0.0
    except Exception:
        return 0.0


def _score_creative(response: str, keywords: list[str]) -> float:
    text = response.lower()
    hits = sum(1 for kw in keywords if kw in text)
    return 1.0 if hits >= 2 else (0.5 if hits == 1 else 0.0)


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

async def _call(base_url: str, model_id: str, prompt: str, timeout: float = 30.0) -> str:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Main calibration entry point
# ---------------------------------------------------------------------------

async def calibrate(base_url: str, model_id: str) -> dict[str, float]:
    """
    Run benchmark prompts against the model and return per-domain accuracy scores.
    Takes ~1-3 minutes depending on model speed.
    """
    log.info("Starting calibration for %s at %s", model_id, base_url)
    results: dict[str, float] = {}

    async def run_domain(name: str, items: list[dict], score_fn) -> float:
        scores = []
        for item in items:
            try:
                resp = await _call(base_url, model_id, item["q"])
                scores.append(score_fn(resp, item))
            except Exception as e:
                log.warning("%s prompt failed: %s", name, e)
                scores.append(0.0)
        acc = round(sum(scores) / len(scores), 3)
        log.info("  %-10s %.2f  (%d/%d)", name + ":", acc, int(sum(scores)), len(scores))
        return acc

    results["factual"]   = await run_domain("factual",   _FACTUAL,   lambda r, i: _score_mcq(r, i["a"]))
    results["math"]      = await run_domain("math",      _MATH,      lambda r, i: _score_math(r, i["a"]))
    results["code"]      = await run_domain("code",      _CODE,      lambda r, i: _score_code(r, i["test"]))
    results["reasoning"] = await run_domain("reasoning", _REASONING, lambda r, i: _score_mcq(r, i["a"]))
    results["creative"]  = await run_domain("creative",  _CREATIVE,  lambda r, i: _score_creative(r, i["keywords"]))
    results["_default"]  = round(min(results.values()), 3)

    log.info("Calibration complete for %s: %s", model_id, results)
    return results
