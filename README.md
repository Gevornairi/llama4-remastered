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
Over a full epoch the gap closes (see [Full epoch](#full-epoch)).

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

### Full epoch

Both presets trained for 56,000 steps (459M tokens, 98% of TinyStories) at their best learning rate from the sweep, with 500 warmup steps and decay to a tenth of the peak.
Both saw the same batches in the same order.

| | Llama 4 | Remastered |
| --- | --- | --- |
| Valid loss | 1.591 | 1.584 |
| Params | 21.25M | 17.58M |
| Throughput | 26.5k tokens/s | 23.8k tokens/s |
| Busiest expert's token share, per layer | 0.34–1.00 (even: 0.25) | 0.13 (even: 0.125) |

Remastered minus Llama 4 validation loss during training:

| Step | 1k | 2k | 5k | 10k | 20k | 30k | 44k | 50k | 54k | 56k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Δ | −0.19 | −0.13 | −0.05 | 0.00 | +0.06 | +0.13 | +0.27 | +0.14 | +0.05 | −0.007 |

At a full epoch the presets end level: the 0.007 gap is about the size of seed noise, so the 0.19 lead at 2,000 steps reflected faster early progress, not a lower final loss.
At a constant 6e-3, remastered's validation loss stays around 1.93–1.94 from step 15,000 until the decay starts at step 44,800, and all of the remaining gain comes during the decay.
It was still falling by 0.03 per 1,000 steps at the end, while Llama 4 had levelled off.
Both learning rates were tuned at 2,000 steps, so neither is necessarily the best choice for a full epoch.
Remastered matches Llama 4 with 17% fewer parameters and evenly loaded experts (in one Llama 4 layer, one expert takes 99.8% of the tokens), at 10% lower throughput.

A second remastered run decays over the last half of training (step 28,000 on) instead of the last fifth, with everything else unchanged:

| Step | 28k | 35k | 44k | 50k | 54k | 56k |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 4 | 1.815 | 1.753 | 1.663 | 1.614 | 1.594 | 1.591 |
| Remastered, decay over the last 20% | 1.931 | 1.925 | 1.928 | 1.749 | 1.645 | 1.584 |
| Remastered, decay over the last 50% | 1.932 | 1.817 | 1.734 | 1.668 | 1.613 | 1.582 |

The longer decay leaves the plateau earlier but ends 0.002 lower, so the final loss depends on reaching the minimum learning rate, not on how long the decay takes.
Both remastered runs were still falling at the end (by 0.018 and 0.031 over the last 1,000 steps), possibly because they end at 6e-4, twice Llama 4's final learning rate; a lower minimum is untested.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). MLX needs Apple silicon; elsewhere, use the NumPy backend.

```sh
uv sync
```

## References

- Meta, *The Llama 4 herd* (April 2025), and the reference implementation in Hugging Face `transformers` (`models/llama4/modeling_llama4.py`).
