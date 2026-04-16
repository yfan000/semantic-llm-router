# Semantic LLM Router

An auction-based semantic router for self-hosted LLM inference fleets (vLLM, NVIDIA Dynamo, Ray Serve).

## Quick start

```bash
pip install -r requirements.txt
uvicorn semantic_router.main:app --host 0.0.0.0 --port 8080 --workers 1
```

## Features
- OpenAI-compatible `/v1/chat/completions` endpoint
- Auction-based bidding: models self-quote cost / latency / accuracy / energy
- KV cache + request queue as load signal for dynamic pricing
- 4D preference-weighted selection (cost, latency, accuracy, energy)
- Per-user mode presets: `accuracy` / `eco` / `cost`
- Per-user token and energy (tokens/J) budgets
- Latency reputation EMA with penalty multiplier
- Async accuracy sampler with Prometheus-2 and Qwen2.5 judges
- Adapters for vLLM, NVIDIA Dynamo, and Ray Serve
