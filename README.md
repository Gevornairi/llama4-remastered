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

## Results

Preliminary: 2,000 steps (16.4M tokens, 3.5% of TinyStories), one seed, 200 warmup steps, decay to a tenth of the peak learning rate.
Validation loss is measured on the same 410k held-out tokens for every run.
A second seed moved the baseline by 0.004 (one pair of seeds, so a rough noise level only).

### Learning rate

| Peak LR | Llama 4 (AdamW + cosine, 21.25M) | Remastered (Muon + WSD, 17.58M) |
| --- | --- | --- |
| 3e-4 | 2.796 | 2.633 |
| 1e-3 | 2.289 | 2.130 |
| 3e-3 | **2.100** | 1.939 |
| 6e-3 | 2.144 | **1.909** |
| 1e-2 | 2.238 | 1.917 |

Remastered is 0.19 lower at each preset's best learning rate, with 17% fewer parameters and the same active FFN width per token.

### One change at a time

Each row is the Llama 4 preset with one remastered change, at lr 3e-3 (baseline 2.100).
Expert imbalance is the busiest expert's token share over the mean share, at the last step (1 = even).

| Change | Valid loss | Δ | Params | Expert imbalance |
| --- | --- | --- | --- | --- |
| Muon + WSD + bias-based balancing (lr 6e-3) | 1.990 | −0.110 | 21.25M | 1.10x |
| Muon + WSD (lr 6e-3) | 1.995 | −0.105 | 21.25M | 3.40x |
| Muon + WSD | 2.005 | −0.095 | 21.25M | 3.41x |
| Tied embeddings | 2.064 | −0.036 | 19.15M | 2.07x |
| QK-norm with a learned scale, before RoPE | 2.082 | −0.018 | 21.25M | 2.36x |
| Many small experts, top-3, normalised gates on the output | 2.091 | −0.009 | 19.68M | 1.84x |
| Bias-based balancing | 2.096 | −0.004 | 21.25M | 1.09x |
| Sliding-window attention | 2.096 | −0.004 | 21.25M | 2.31x |
| Many small experts, top-3 | 2.100 | 0.000 | 19.68M | 1.79x |
| Gates on the expert output (top-1) | 2.102 | +0.002 | 21.25M | 2.24x |
| Baseline | 2.100 | — | 21.25M | 2.22x |

Repeated with a second seed:

| | Seed 0 | Seed 1 |
| --- | --- | --- |
| Baseline | 2.100 | 2.096 |
| QK-norm with a learned scale, before RoPE | 2.082 (−0.018) | 2.071 (−0.025) |

Three changes clearly help: Muon + WSD, tied embeddings and QK-norm (on both seeds).
The rest are neutral on loss at this scale; the smaller experts get there with 1.6M fewer parameters, and balancing evens out the routing.
The single changes add up to about −0.18, close to the full preset's −0.19.
Muon + WSD is the largest gain, even though without balancing one expert takes 94–100% of the tokens in half of the layers; adding balancing evens out the routing but lowers the loss by only 0.005.
Where the gate is applied makes no measurable difference. Normalised gates need top-k > 1: with one chosen expert every gate is 1 and the router never trains, so the configuration rejects that combination.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). MLX needs Apple silicon; elsewhere, use the NumPy backend.

```sh
uv sync
```

## References

- Meta, *The Llama 4 herd* (April 2025), and the reference implementation in Hugging Face `transformers` (`models/llama4/modeling_llama4.py`).
