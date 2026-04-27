"""
load_test.py — 1000-request load test against the real running router.

Sends diverse requests across all domains and complexities, logs every
result to a CSV, and prints a summary analysis report.

Usage:
    cd ~/semantic-llm-router
    python tests/load_test.py                          # 1000 requests, concurrency 10
    python tests/load_test.py --requests 500 --concurrency 20
    python tests/load_test.py --router http://my-server:8080
    python tests/load_test.py --output results/run1.csv

Output:
    results/load_test_YYYYMMDD_HHMMSS.csv   (every request logged)
    Printed analysis report at the end
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import os
import random
import time
from datetime import datetime
from statistics import mean

import httpx

# ---------------------------------------------------------------------------
# Diverse queries: 5 domains x 3 complexities
# ---------------------------------------------------------------------------

QUERIES: dict[tuple[str, str], list[str]] = {
    ("factual", "easy"): [
        "What is the capital of France?",
        "What is the chemical symbol for gold?",
        "How many days are in a leap year?",
        "Who invented the telephone?",
        "What is the speed of light in km/s?",
        "What is the largest planet in our solar system?",
        "How many continents are there on Earth?",
        "What is the boiling point of water in Celsius?",
        "Who painted the Mona Lisa?",
        "What is the capital of Japan?",
        "How many bones are in the adult human body?",
        "What language is spoken in Brazil?",
        "What is the currency of the United Kingdom?",
        "How many sides does a hexagon have?",
        "What is the smallest country in the world?",
        "Who wrote Romeo and Juliet?",
        "What is the tallest mountain on Earth?",
        "What gas do plants use during photosynthesis?",
        "What is the atomic number of carbon?",
        "How many planets are in our solar system?",
    ],
    ("factual", "medium"): [
        "Explain the difference between DNA and RNA.",
        "What caused the fall of the Roman Empire?",
        "How does the immune system fight viruses?",
        "What is the significance of the Magna Carta?",
        "Explain how vaccines work.",
        "What are the main causes of climate change?",
        "How does the stock market work?",
        "What is the difference between a virus and a bacterium?",
        "Explain the greenhouse effect.",
        "What were the main causes of World War I?",
        "How does the human digestive system work?",
        "What is the significance of the Industrial Revolution?",
        "Explain the concept of supply and demand.",
        "How does GPS navigation work?",
        "What is the role of the United Nations?",
        "Explain how antibiotics work.",
        "What is the difference between renewable and non-renewable energy?",
        "How does the brain process memories?",
        "What is the significance of the Silk Road?",
        "Explain what causes earthquakes.",
    ],
    ("factual", "hard"): [
        "Explain the geopolitical implications of the Bretton Woods collapse.",
        "Analyse the long-term economic consequences of hyperinflation in Weimar Germany.",
        "What are the epistemological foundations of the scientific method?",
        "Explain the relationship between quantum mechanics and general relativity.",
        "What are the sociological implications of mass surveillance on democracy?",
        "Analyse the role of cognitive biases in financial decision-making.",
        "Explain the thermodynamic principles behind entropy and the arrow of time.",
        "What are the geopolitical consequences of rare earth mineral scarcity?",
        "Analyse the philosophical implications of the Turing Test for consciousness.",
        "Explain the game-theoretic foundations of nuclear deterrence.",
    ],
    ("math", "easy"): [
        "What is 15% of 240?",
        "What is the square root of 144?",
        "If a rectangle is 8m wide and 5m tall, what is its area?",
        "What is 3/4 expressed as a decimal?",
        "What is 2 to the power of 8?",
        "If you buy 6 items at $3.50 each, what is the total?",
        "What is the perimeter of a square with side 7cm?",
        "Convert 100 kilometres to miles (1km = 0.621 miles).",
        "What is 20% of 85?",
        "If a train travels at 60 km/h for 2.5 hours, how far does it go?",
        "What is the sum of angles in a triangle?",
        "What is 1/3 + 1/4?",
        "How many seconds are in 2.5 hours?",
        "If x + 7 = 15, what is x?",
        "What is the area of a circle with radius 5? (use pi=3.14)",
        "What is 25% of 200?",
        "If a dozen eggs cost $4.80, how much does each egg cost?",
        "What is 7 squared minus 5 squared?",
        "Convert 37 degrees Celsius to Fahrenheit.",
        "If 3x = 21, what is x?",
    ],
    ("math", "medium"): [
        "Solve the quadratic equation x squared minus 5x plus 6 equals 0.",
        "Find the derivative of f(x) = 3x cubed minus 2x squared plus x minus 5.",
        "What is the probability of rolling two sixes in a row with a fair die?",
        "Calculate the compound interest on $1000 at 5% annually for 3 years.",
        "Find the sum of the arithmetic series 3 + 7 + 11 + ... + 99.",
        "What is the determinant of the matrix [[2,3],[1,4]]?",
        "Solve the system: 2x + y = 7 and x - y = 2.",
        "What is the integral of x squared from 0 to 3?",
        "Find the nth term formula for the sequence 2, 5, 10, 17, 26.",
        "What is the standard deviation of {2, 4, 4, 4, 5, 5, 7, 9}?",
    ],
    ("math", "hard"): [
        "Prove by induction that the sum of the first n natural numbers is n(n+1)/2.",
        "Find the eigenvalues and eigenvectors of the matrix [[3,1],[1,3]].",
        "Prove that the square root of 2 is irrational.",
        "Solve the differential equation dy/dx = 2xy with y(0) = 1.",
        "What is the Fourier transform of a Gaussian function?",
        "Prove the Cauchy-Schwarz inequality for inner product spaces.",
        "Find the general solution to y'' minus 3y' plus 2y = e to the x.",
        "Prove that there are infinitely many prime numbers.",
        "What is the gradient and Hessian of f(x,y) = x squared y + xy squared?",
        "Derive the formula for the volume of a sphere using integration.",
    ],
    ("code", "easy"): [
        "Write a Python function to add two numbers.",
        "Write a function that returns the length of a string.",
        "Write a Python function to check if a number is even.",
        "Write a function to find the maximum of two numbers.",
        "Write a Python function to reverse a string.",
        "Write a function that returns True if a list is empty.",
        "Write a Python function to convert Celsius to Fahrenheit.",
        "Write a function to count the vowels in a string.",
        "Write a Python function to check if a string is a palindrome.",
        "Write a function that returns the first element of a list.",
        "Write a Python function to calculate the factorial of n.",
        "Write a function that squares all numbers in a list.",
        "Write a Python function to find the minimum value in a list.",
        "Write a function that checks if a number is positive.",
        "Write a Python function to concatenate two lists.",
        "Write a Python function to check if a year is a leap year.",
        "Write a function that removes duplicates from a list.",
        "Write a Python function to return the nth Fibonacci number.",
        "Write a function that checks if a string contains only digits.",
        "Write a Python function to flatten a list one level deep.",
    ],
    ("code", "medium"): [
        "Implement a binary search algorithm in Python.",
        "Write a Python class for a stack data structure.",
        "Implement merge sort in Python.",
        "Write a function to find all permutations of a string.",
        "Implement a simple LRU cache in Python.",
        "Write a Python decorator that measures execution time.",
        "Implement a binary tree with insert and search methods.",
        "Write a Python generator that yields prime numbers.",
        "Implement quicksort in Python.",
        "Write a function that validates an email address using regex.",
        "Implement a basic graph with BFS and DFS.",
        "Write a Python function to memoize any function.",
        "Write a function to find the longest common subsequence of two strings.",
        "Implement a trie data structure in Python.",
        "Write a Python context manager for file operations.",
        "Implement a rate limiter using a token bucket algorithm.",
        "Write a function that flattens a deeply nested list.",
        "Implement the producer-consumer pattern using threading.",
        "Write a Python class for a queue using two stacks.",
        "Implement a simple event emitter in Python.",
    ],
    ("code", "hard"): [
        "Implement a thread-safe singleton pattern in Python.",
        "Write a Python implementation of a consistent hash ring.",
        "Implement an async connection pool in Python using asyncio.",
        "Write a Python implementation of the Raft consensus algorithm skeleton.",
        "Implement a lock-free queue using compare-and-swap semantics in Python.",
        "Write a Python implementation of a B-tree.",
        "Implement a distributed key-value store client with automatic failover.",
        "Write a Python metaclass that enforces interface contracts.",
        "Implement a reactive stream pipeline with backpressure in Python.",
        "Write a Python implementation of a persistent red-black tree.",
    ],
    ("reasoning", "easy"): [
        "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
        "If it rains, the ground gets wet. The ground is not wet. Did it rain?",
        "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?",
        "All squares are rectangles. Shape X is a square. Is shape X a rectangle?",
        "John has more apples than Mary. Mary has more than Tom. Who has fewest?",
        "Every dog in the shelter is vaccinated. Max is a shelter dog. Is Max vaccinated?",
        "All prime numbers greater than 2 are odd. Is 7 prime and odd?",
        "Either the meeting is Monday or Tuesday. It is not Monday. When is it?",
        "Jane is older than Tom. Tom is older than Sue. Is Jane older than Sue?",
        "All students who passed studied hard. Kim did not study. Did Kim pass?",
        "If P implies Q, and Q is false, what can we say about P?",
        "A is north of B. B is north of C. Is A north of C?",
        "Every even number is divisible by 2. 14 is even. Is 14 divisible by 2?",
        "If you are taller than 180cm you can ride. Sam is 175cm. Can Sam ride?",
        "A is twice B. B is 5. What is A?",
        "All cats have tails. Whiskers is a cat. Does Whiskers have a tail?",
        "Five birds were on a branch, three flew away. How many remain?",
        "If today is Monday, what day is it in 3 days?",
        "If A then B. A is true. Is B true?",
        "A is twice as old as B. B is 10. How old is A?",
    ],
    ("reasoning", "medium"): [
        "Compare and contrast SQL and NoSQL databases for a high-traffic web app.",
        "What are the trade-offs between microservices and monolithic architecture?",
        "Should a startup choose Python or Go for their backend? Justify your answer.",
        "Compare gradient descent with Adam optimiser for neural network training.",
        "What are the trade-offs between consistency and availability in distributed systems?",
        "Compare TCP and UDP protocols for video streaming applications.",
        "Should companies store data in the cloud or on-premises? Discuss trade-offs.",
        "Compare supervised and reinforcement learning for robotics applications.",
        "Compare containerisation with virtualisation for deployment.",
        "Analyse the trade-offs between code readability and performance optimisation.",
        "Compare horizontal and vertical scaling strategies for databases.",
        "What are the ethical implications of facial recognition technology?",
        "Analyse the pros and cons of remote work for productivity.",
        "Compare agile and waterfall development methodologies.",
        "What are the implications of strong AI on employment markets?",
        "Evaluate the pros and cons of electric vehicles versus combustion engines.",
        "Analyse whether open-source software is more secure than proprietary.",
        "Compare early and late binding in programming languages.",
        "What are the implications of quantum computing for current encryption?",
        "Analyse the arguments for and against a universal basic income.",
    ],
    ("reasoning", "hard"): [
        "Analyse the game-theoretic implications of the prisoner's dilemma for climate agreements.",
        "Evaluate the philosophical tension between determinism and moral responsibility.",
        "What are the second-order economic effects of automating white-collar jobs?",
        "Analyse the logical consistency of Rawls' veil of ignorance argument.",
        "How does Gödel's incompleteness theorem limit the foundations of mathematics?",
        "Evaluate the strategic implications of first-mover advantage in network-effects markets.",
        "What are the epistemological challenges of training AI on internet-scale data?",
        "Analyse the causal mechanisms linking financial deregulation to systemic risk.",
        "Evaluate the philosophical implications of the hard problem of consciousness.",
        "How do information asymmetries affect market efficiency in healthcare?",
    ],
    ("creative", "easy"): [
        "Write a two-sentence story about a friendly dragon.",
        "Write a haiku about the ocean.",
        "Create a tagline for a coffee shop.",
        "Write one sentence describing the feeling of rain.",
        "Write a two-line rhyme about Monday mornings.",
        "Describe a sunset in one sentence.",
        "Write a short fortune cookie message.",
        "Create a slogan for a bookstore.",
        "Write a two-sentence story about a lost key.",
        "Describe the smell of fresh bread in one sentence.",
        "Write a one-line joke about programmers.",
        "Create a name for a new ice cream flavour.",
        "Write a two-sentence bedtime story.",
        "Describe the sound of a thunderstorm in one sentence.",
        "Create a tagline for a gym.",
        "Write a haiku about autumn leaves.",
        "Describe the feeling of finishing a good book.",
        "Write a two-sentence story about a talking cat.",
        "Create a product name for a smart umbrella.",
        "Write a one-sentence description of the colour blue.",
    ],
    ("creative", "medium"): [
        "Write a short story about a robot learning to paint.",
        "Write a poem about the loneliness of space exploration.",
        "Create a product description for a time machine.",
        "Write a dialogue between two stars discussing humanity.",
        "Write a short story about a chef who can taste emotions.",
        "Write a letter from the ocean to the moon.",
        "Create a myth explaining why the sky is blue.",
        "Write a short story about the last library on Earth.",
        "Write a poem about the relationship between humans and technology.",
        "Create a news headline from 100 years in the future.",
        "Write a short story about a city that runs on music.",
        "Write a poem about impermanence using imagery of seasons.",
        "Create a product pitch for an app that translates animal sounds.",
        "Write a dialogue between a tree and the wind.",
        "Write a short story about a painter who only paints in darkness.",
        "Write a poem about the first person to see fire.",
        "Create a travel brochure for a planet made entirely of water.",
        "Write a short story about a map that shows the future.",
        "Write a poem about the moment between waking and sleeping.",
        "Create an advertisement for selling memories.",
    ],
    ("creative", "hard"): [
        "Write a short story told from the perspective of a dying star.",
        "Write a poem in the style of Emily Dickinson about artificial intelligence.",
        "Create a myth about the origin of mathematics using characters from multiple cultures.",
        "Write a philosophical dialogue between Socrates and a modern AI.",
        "Write a short story that uses the same sentence at the beginning and end with opposite meaning.",
        "Create a world-building document for a society where music determines social hierarchy.",
        "Write a poem simultaneously about love and quantum entanglement.",
        "Write a short story that explores identity through the metaphor of translation.",
        "Create a fable about the dangers of optimisation without purpose.",
        "Write flash fiction where the narrator is unreliable in a way revealed only in the last line.",
    ],
}

MODES = ["cost", "eco", "accuracy", "custom"]
MODE_WEIGHTS = [0.35, 0.25, 0.25, 0.15]


def build_request_list(n: int) -> list[dict]:
    all_queries = []
    for (domain, complexity), queries in QUERIES.items():
        for q in queries:
            all_queries.append({"domain": domain, "complexity": complexity, "query": q})
    random.shuffle(all_queries)
    if n > len(all_queries):
        all_queries += random.choices(all_queries, k=n - len(all_queries))
    return all_queries[:n]


def random_router_params() -> dict:
    mode = random.choices(MODES, weights=MODE_WEIGHTS)[0]
    if mode == "custom":
        w = [random.random() for _ in range(4)]
        s = sum(w)
        return {
            "mode": "custom",
            "cost_weight":     round(w[0]/s, 3),
            "latency_weight":  round(w[1]/s, 3),
            "accuracy_weight": round(w[2]/s, 3),
            "energy_weight":   round(w[3]/s, 3),
        }
    return {"mode": mode}


async def send_request(client: httpx.AsyncClient, router_url: str, req_id: int, item: dict) -> dict:
    params = random_router_params()
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": item["query"]}],
        "extra_body": {"router": params},
    }
    t0 = time.monotonic()
    result = {
        "req_id": req_id, "domain": item["domain"], "complexity": item["complexity"],
        "query": item["query"][:100], "mode": params.get("mode", "custom"),
        "status": "", "model_winner": "", "bid_latency_ms": "", "actual_latency_ms": "",
        "charged_usd": "", "energy_j": "", "load": "", "wall_ms": "", "error": "",
    }
    try:
        resp = await client.post(f"{router_url}/v1/chat/completions", json=payload)
        result["wall_ms"] = int((time.monotonic() - t0) * 1000)
        result["status"]  = resp.status_code
        h = resp.headers
        result["model_winner"]      = h.get("x-router-model", "")
        result["bid_latency_ms"]    = h.get("x-router-bid-latency-ms", "")
        result["actual_latency_ms"] = h.get("x-router-actual-latency-ms", "")
        result["charged_usd"]       = h.get("x-router-charged-usd", "")
        result["energy_j"]          = h.get("x-router-energy-j", "")
        result["load"]              = h.get("x-router-load", "")
        if resp.status_code != 200:
            result["error"] = resp.json().get("detail", str(resp.status_code))
    except Exception as e:
        result["wall_ms"] = int((time.monotonic() - t0) * 1000)
        result["status"]  = "error"
        result["error"]   = str(e)
    return result


def analyse(rows: list[dict]) -> None:
    W = 62
    ok   = [r for r in rows if r["status"] == 200]
    errs = [r for r in rows if r["status"] != 200]

    def sf(v): 
        try: return float(v)
        except: return None

    print(f"\n{'='*W}\n  LOAD TEST RESULTS\n{'='*W}")
    print(f"  Total requests : {len(rows)}")
    print(f"  Successful     : {len(ok)}  ({100*len(ok)//max(len(rows),1)}%)")
    print(f"  Errors/fallback: {len(errs)}")

    if not ok:
        return

    # Routing distribution
    print(f"\n{'='*W}\n  ROUTING DISTRIBUTION\n{'='*W}")
    mc: dict[str, int] = {}
    for r in ok:
        m = r["model_winner"] or "unknown"
        mc[m] = mc.get(m, 0) + 1
    for model, cnt in sorted(mc.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / len(ok) * 30)
        print(f"  {model:<22} {bar:<30} {100*cnt//len(ok):3}%  ({cnt})")

    # Wall latency
    wv = sorted([sf(r["wall_ms"]) for r in ok if sf(r["wall_ms"])])
    if wv:
        print(f"\n{'='*W}\n  WALL LATENCY (end-to-end ms)\n{'='*W}")
        print(f"  P50={wv[len(wv)//2]:.0f}  P95={wv[int(len(wv)*.95)]:.0f}  P99={wv[int(len(wv)*.99)]:.0f}  mean={mean(wv):.0f}")

    # Per-model latency
    print(f"\n{'='*W}\n  ACTUAL LATENCY PER MODEL\n{'='*W}")
    bm: dict[str, list] = {}
    for r in ok:
        m = r["model_winner"] or "unknown"
        v = sf(r["actual_latency_ms"])
        if v: bm.setdefault(m, []).append(v)
    for model, vals in sorted(bm.items()):
        vals.sort()
        print(f"  {model:<22} P50={vals[len(vals)//2]:.0f}ms  P95={vals[int(len(vals)*.95)]:.0f}ms  mean={mean(vals):.0f}ms  (n={len(vals)})")

    # Cost & energy
    costs    = [sf(r["charged_usd"]) for r in ok if sf(r["charged_usd"]) is not None]
    energies = [sf(r["energy_j"])    for r in ok if sf(r["energy_j"])    is not None]
    if costs:
        print(f"\n{'='*W}\n  COST & ENERGY SUMMARY\n{'='*W}")
        print(f"  Total cost     : ${sum(costs):.6f}")
        print(f"  Avg per request: ${mean(costs):.8f}")
        if energies:
            print(f"  Total energy   : {sum(energies):.2f} J")
            print(f"  Avg per request: {mean(energies):.3f} J")

    # Routing by domain
    print(f"\n{'='*W}\n  ROUTING BY DOMAIN\n{'='*W}")
    for domain in sorted({r["domain"] for r in ok}):
        dr = [r for r in ok if r["domain"] == domain]
        dc: dict[str, int] = {}
        for r in dr: dc[r["model_winner"] or "?"] = dc.get(r["model_winner"] or "?", 0) + 1
        parts = "  ".join(f"{m}:{c}" for m, c in sorted(dc.items(), key=lambda x: -x[1]))
        print(f"  {domain:<12} ({len(dr):4} reqs)  {parts}")

    # Routing by complexity
    print(f"\n{'='*W}\n  ROUTING BY COMPLEXITY\n{'='*W}")
    for cplx in ["easy", "medium", "hard"]:
        cr = [r for r in ok if r["complexity"] == cplx]
        cc: dict[str, int] = {}
        for r in cr: cc[r["model_winner"] or "?"] = cc.get(r["model_winner"] or "?", 0) + 1
        parts = "  ".join(f"{m}:{c}" for m, c in sorted(cc.items(), key=lambda x: -x[1]))
        print(f"  {cplx:<8} ({len(cr):4} reqs)  {parts}")

    # Routing by mode
    print(f"\n{'='*W}\n  ROUTING BY MODE\n{'='*W}")
    for mode in ["cost", "eco", "accuracy", "custom"]:
        mr = [r for r in ok if r["mode"] == mode]
        if not mr: continue
        mmc: dict[str, int] = {}
        for r in mr: mmc[r["model_winner"] or "?"] = mmc.get(r["model_winner"] or "?", 0) + 1
        parts = "  ".join(f"{m}:{c}" for m, c in sorted(mmc.items(), key=lambda x: -x[1]))
        print(f"  {mode:<10} ({len(mr):4} reqs)  {parts}")

    if errs:
        print(f"\n{'='*W}\n  ERRORS\n{'='*W}")
        ec: dict[str, int] = {}
        for r in errs:
            e = r["error"] or str(r["status"])
            ec[e] = ec.get(e, 0) + 1
        for msg, cnt in sorted(ec.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cnt:4}x  {msg[:70]}")

    print(f"\n{'='*W}\n")


async def run(router_url: str, n_requests: int, concurrency: int, output: str) -> None:
    requests_list = build_request_list(n_requests)
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    fieldnames = [
        "req_id", "domain", "complexity", "query", "mode", "status",
        "model_winner", "bid_latency_ms", "actual_latency_ms",
        "charged_usd", "energy_j", "load", "wall_ms", "error",
    ]

    results: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    done = 0
    t0 = time.monotonic()

    print(f"\n  Sending {n_requests} requests to {router_url} (concurrency={concurrency})")
    print(f"  Output: {output}\n")

    async def bounded(req_id: int, item: dict, writer, f) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await send_request(client, router_url, req_id, item)
            results.append(r)
            writer.writerow(r)
            f.flush()
            done += 1
            if done % max(n_requests // 20, 1) == 0:
                elapsed = time.monotonic() - t0
                bar = "=" * int(done / n_requests * 40)
                print(f"\r  [{bar:<40}] {done}/{n_requests}  {done/elapsed:.1f} req/s", end="", flush=True)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        await asyncio.gather(*[bounded(i, requests_list[i], writer, f) for i in range(n_requests)])

    elapsed = time.monotonic() - t0
    print(f"\r  [{'='*40}] {n_requests}/{n_requests}  ({elapsed:.1f}s, {n_requests/elapsed:.1f} req/s)\n")

    analyse(results)
    print(f"  Full results saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router",      default="http://localhost:8080")
    parser.add_argument("--requests",    type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--output",      default="")
    args = parser.parse_args()
    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/load_test_{ts}.csv"
    asyncio.run(run(args.router, args.requests, args.concurrency, args.output))


if __name__ == "__main__":
    main()
