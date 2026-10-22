# llama4-remastered

An independent, from-scratch reimplementation of the **Llama 4** architecture, trained on public data, plus a "remastered" variant that swaps in newer techniques, each measured against the faithful baseline.
There is no autograd: every layer has a hand-written `forward` and `backward`, verified by a gradient check. No Meta code or weights are used.

The layers come from [nnkit](https://github.com/Gevornairi/nnkit); this repo holds the model, data pipeline, training and generation.
All math goes through one array module, so the same code runs on the Apple GPU (MLX) or the CPU (NumPy).

## Plan

Every remastered change is a flag; the faithful Llama 4 configuration is the default baseline.

| Llama 4 | Remastered | Source |
| --- | --- | --- |
| MoE: top-1 routing, router score on the expert input, shared expert | Many small experts, top-k, normalised gates on the expert output | DeepSeek-V3 |
| Load balancing (not published) | Bias-based balancing without an auxiliary loss | DeepSeek-V3 |
| QK-norm without a learned scale, after RoPE | RMSNorm with a learned scale, before RoPE | Qwen3, OLMo 2, Gemma 3 |
| Chunked local attention on RoPE layers | Sliding-window attention | Gemma 3, gpt-oss |
| Untied embeddings | Tied embeddings (better at small scale) | Gemma |
| AdamW + cosine schedule | Muon + warmup-stable-decay | Kimi K2, MiniCPM |

Results will replace this plan as each change is trained and compared.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). MLX needs Apple silicon; elsewhere, use the NumPy backend.

```sh
uv sync
```

## References

- Meta, *The Llama 4 herd* (April 2025), and the reference implementation in Hugging Face `transformers` (`models/llama4/modeling_llama4.py`).
