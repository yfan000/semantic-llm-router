"""
mock_backend.py — Fake vLLM server for manual integration testing.

Run two instances to simulate a 2-GPU setup:

  # Terminal 1 — fast cheap model (low load)
  python tests/mock_backend.py --port 8001 --model-id llama-3.1-8b \\
    --latency-ms 200 --accuracy 0.72 --load 0.2

  # Terminal 2 — slow accurate model (high load)
  python tests/mock_backend.py --port 8002 --model-id llama-3.1-70b \\
    --latency-ms 800 --accuracy 0.91 --load 0.6

  # Terminal 3 — start the router
  uvicorn semantic_router.main:app --port 8080

Then register both models and send requests normally.
"""
import argparse
import asyncio
import time
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()
CFG: dict = {}


@app.get("/health")
def health():
    return {"status": "ok", "model": CFG["model_id"]}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    load    = CFG["load"]
    waiting = int(load * CFG["max_seqs"] * 0.5)
    return (
        f"# HELP vllm:gpu_cache_usage_perc KV cache usage\n"
        f"vllm:gpu_cache_usage_perc {load}\n"
        f"# HELP vllm:num_requests_waiting Waiting requests\n"
        f"vllm:num_requests_waiting {waiting}\n"
        f"# HELP vllm:num_requests_running Running requests\n"
        f"vllm:num_requests_running {int(load * CFG['max_seqs'])}\n"
    )


@app.post("/v1/chat/completions")
async def completions(request: dict):
    await asyncio.sleep(CFG["latency_ms"] / 1000)
    prompt = "".join(m.get("content", "") for m in request.get("messages", []) if m.get("role") == "user")
    input_tokens  = max(len(prompt.split()) * 2, 10)
    output_tokens = 50
    return {
        "id": f"mock-{int(time.time())}",
        "object": "chat.completion",
        "model": CFG["model_id"],
        "choices": [{"index": 0,
                     "message": {"role": "assistant",
                                 "content": f"[{CFG['model_id']}] Mock response to: {prompt[:80]}"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens},
    }


def main():
    parser = argparse.ArgumentParser(description="Mock vLLM backend")
    parser.add_argument("--port",       type=int,   default=8001)
    parser.add_argument("--model-id",   type=str,   default="mock-model")
    parser.add_argument("--latency-ms", type=float, default=300.0)
    parser.add_argument("--accuracy",   type=float, default=0.75)
    parser.add_argument("--load",       type=float, default=0.3)
    parser.add_argument("--max-seqs",   type=int,   default=256)
    args = parser.parse_args()
    CFG.update({"model_id": args.model_id, "latency_ms": args.latency_ms,
                "accuracy": args.accuracy, "load": args.load, "max_seqs": args.max_seqs})
    print(f"Starting mock backend '{args.model_id}' on port {args.port}")
    print(f"  latency={args.latency_ms}ms  accuracy={args.accuracy}  load={args.load}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
