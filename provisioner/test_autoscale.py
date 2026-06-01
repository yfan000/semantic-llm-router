"""
test_autoscale.py -- Test auto-scaling of the dynamic provisioner.

Phase 1 (flood): Send concurrent requests to trigger queue overload -> spin up
Phase 2 (idle):  Stop requests, wait for idle timeout -> spin down

Usage:
    python provisioner/test_autoscale.py                  # both phases
    python provisioner/test_autoscale.py --phase flood    # only flood
    python provisioner/test_autoscale.py --phase idle     # only wait for spin-down
    python provisioner/test_autoscale.py --phase monitor  # watch models live
"""
from __future__ import annotations

import argparse
import asyncio
import time
from statistics import mean

import httpx

ROUTER_URL  = "http://localhost:8080"
CONCURRENCY = 40    # concurrent requests -- enough to build a queue
TOTAL       = 200   # total requests in flood phase
MAX_TOKENS  = 300   # longer responses = more GPU time = longer queue buildup
IDLE_WAIT_S = 400   # seconds to wait after flood (> IDLE_WINDOW_S=300 in provisioner)


async def get_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{ROUTER_URL}/v1/models")
            return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []


async def send_request(client: httpx.AsyncClient, req_id: int) -> dict:
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"{ROUTER_URL}/v1/chat/completions",
            json={
                "model": "router",
                "messages": [{"role": "user",
                               "content": "Explain machine learning in detail with examples."}],
                "max_tokens": MAX_TOKENS,
                "extra_body": {"router": {"user_id": f"tester-{req_id % 5}",
                                          "mode": "accuracy"}},
            },
        )
        wall_ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            h = r.headers
            return {
                "ok": True, "req_id": req_id, "wall_ms": wall_ms,
                "model":      h.get("x-router-model", "?"),
                "latency_ms": h.get("x-router-actual-latency-ms", "?"),
                "attempt":    h.get("x-router-attempt", "1"),
            }
        return {"ok": False, "req_id": req_id, "wall_ms": wall_ms,
                "error": str(r.status_code)}
    except Exception as e:
        return {"ok": False, "req_id": req_id,
                "wall_ms": int((time.monotonic() - t0) * 1000),
                "error": str(e)[:80]}


async def flood_phase() -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 1: FLOOD  ({TOTAL} requests, concurrency={CONCURRENCY})")
    print(f"  Goal: queue > 10 -> provisioner spins up a new model")
    print(f"{'='*60}")

    models_before = await get_models()
    print(f"  Models before: {models_before}")
    print(f"  Sending {TOTAL} requests...\n")

    results = []
    sem  = asyncio.Semaphore(CONCURRENCY)
    done = 0
    t0   = time.monotonic()

    async def bounded(req_id: int) -> None:
        nonlocal done
        async with sem:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await send_request(client, req_id)
            results.append(r)
            done += 1
            if done % 10 == 0:
                elapsed = time.monotonic() - t0
                models  = await get_models()
                ok      = sum(1 for x in results if x["ok"])
                print(f"  [{elapsed:5.1f}s] {done:3d}/{TOTAL}  "
                      f"({ok} ok)  models={models}")

    await asyncio.gather(*[bounded(i) for i in range(TOTAL)])

    elapsed     = time.monotonic() - t0
    models_after = await get_models()
    ok           = [r for r in results if r["ok"]]
    failed       = [r for r in results if not r["ok"]]
    wall_ms      = sorted(r["wall_ms"] for r in ok)
    by_model: dict[str, int] = {}
    for r in ok:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1

    print(f"\n{'='*60}")
    print(f"  FLOOD RESULTS")
    print(f"{'='*60}")
    print(f"  Total: {TOTAL}  OK: {len(ok)}  Failed: {len(failed)}")
    print(f"  Time:  {elapsed:.1f}s  ({TOTAL/elapsed:.1f} req/s)")
    if wall_ms:
        print(f"  P50={wall_ms[len(wall_ms)//2]}ms  "
              f"P95={wall_ms[int(len(wall_ms)*.95)]}ms  "
              f"mean={int(mean(wall_ms))}ms")
    print(f"  Routing: {by_model}")
    print(f"  Models before: {models_before}")
    print(f"  Models after:  {models_after}")

    new  = set(models_after) - set(models_before)
    gone = set(models_before) - set(models_after)
    if new:
        print(f"\n  SCALE-UP: {new} was spun up!")
    if gone:
        print(f"\n  SCALE-DOWN: {gone} was spun down!")
    if not new and not gone:
        print(f"\n  No model changes yet.")
        print(f"  Check provisioner log: tail -f ~/vllm_logs/provisioner.log")
        print(f"  The new model may still be starting up (takes 2-5 min).")

    if failed:
        print(f"\n  Errors ({len(failed)}):")
        for r in failed[:3]:
            print(f"    req {r['req_id']}: {r.get('error')}")


async def idle_phase() -> None:
    print(f"\n{'='*60}")
    print(f"  PHASE 2: IDLE WAIT ({IDLE_WAIT_S}s)")
    print(f"  Goal: no requests -> provisioner spins down extra models")
    print(f"{'='*60}")

    models_before = await get_models()
    print(f"  Models now: {models_before}")
    print(f"  Waiting {IDLE_WAIT_S}s with no traffic...\n")

    t0           = time.monotonic()
    prev_models  = models_before

    while time.monotonic() - t0 < IDLE_WAIT_S:
        await asyncio.sleep(10)
        models  = await get_models()
        elapsed = int(time.monotonic() - t0)
        if models != prev_models:
            print(f"\n  [{elapsed:4d}s] CHANGE: {prev_models} -> {models}")
            prev_models = models
        else:
            remaining = int(IDLE_WAIT_S - (time.monotonic() - t0))
            print(f"  [{elapsed:4d}s] Models: {models}  ({remaining}s left)", end="\r")

    models_after = await get_models()
    print(f"\n\n  Models after idle: {models_after}")
    if len(models_after) < len(models_before):
        print(f"  SCALE-DOWN: {set(models_before) - set(models_after)} spun down!")
    else:
        print(f"  No change — idle window may not have elapsed yet in provisioner.")


async def main(phase: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{ROUTER_URL}/router/health")
        print(f"Router: {r.json()}")
    except Exception as e:
        print(f"Cannot reach router: {e}")
        return

    if phase in ("flood", "both"):
        await flood_phase()
    if phase in ("idle", "both"):
        await idle_phase()
    if phase == "monitor":
        print("Watching models (Ctrl+C to stop)...")
        prev = []
        while True:
            models = await get_models()
            if models != prev:
                print(f"  {time.strftime('%H:%M:%S')} Models: {models}")
                prev = models
            else:
                print(f"  {time.strftime('%H:%M:%S')} Models: {models}", end="\r")
            await asyncio.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url",  default=ROUTER_URL)
    parser.add_argument("--phase",       default="both",
                        choices=["both", "flood", "idle", "monitor"])
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--total",       type=int, default=TOTAL)
    args = parser.parse_args()

    ROUTER_URL  = args.router_url
    CONCURRENCY = args.concurrency
    TOTAL       = args.total

    asyncio.run(main(args.phase))
