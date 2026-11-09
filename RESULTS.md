# Results

CPU (2 cores), WikiText-2 test, 32 windows of 1024 tokens, one run per setting. Final package defaults:
shaped coding with λ = 1 and κ = 1 (nothing tuned) and dynamic group grids. RTN uses the same GPTQ grid.

| model | group | dense | RTN INT4 | shaped INT4 | RTN mixed | shaped mixed | mixed bits/w |
|---|---|---|---|---|---|---|---|
| GPT-2 124M | 128 | 26.84 | 29.91 | **29.14** | 29.62 | **28.86** | 4.75 |
| OPT-125m | 128 | 29.27 | 31.98 | **31.39** | 30.71 | **30.44** | 4.75 |
| SmolLM2-135M (Llama) | 64 | 15.46 | **19.28** | 19.29 | **17.82** | 17.83 | 5.0 |

Uniform INT4 is 4 + 32/group bits per quantized weight: 4.25 for groups of 128, 4.5 for groups of 64.
"Mixed" is INT4/INT8/FP16 per module chosen by the type rule at the stated budget.

Change in perplexity from shaped coding vs RTN:
* GPT-2 124M: INT4 -2.6%, mixed -2.6%
* OPT-125m: INT4 -1.9%, mixed -0.9%
* SmolLM2-135M (Llama): INT4 +0.1%, mixed +0.1%

Static group grids (grids fixed from the original weights), for comparison:
* GPT-2 124M: shaped INT4 29.09, shaped mixed 28.82
* OPT-125m: shaped INT4 32.03, shaped mixed 31.02
* SmolLM2-135M (Llama): shaped INT4 19.19, shaped mixed 17.74

Exact per-layer output error vs RTN under the true input covariance (OPT-125m, layers 0/4/8, evaluation only):
geometric-mean ratio 0.868 (data-free H), 0.282 (true covariance). The layer-level gain only partly reaches perplexity.

## 70B scaling

Estimates from the real configs (`shapequant memory`):

| model | params | device peak | host buffer | INT4 checkpoint | largest H |
|---|---|---|---|---|---|
| Llama-3.1-70B | 70.6B | 21.2 GB | 5.6 GB | 37.8 GB | 28672 |
| Qwen2.5-72B | 72.7B | 22.1 GB | 5.6 GB | 39.4 GB | 29568 |
| Llama-2-70B | 69.0B | 21.2 GB | 5.6 GB | 34.8 GB | 28672 |

Per-layer core at Llama-3.1-70B shapes, executed on cpu (2 cores): stacked q/k/v (10240×8192) 26 s, o_proj (8192×8192) 19 s, attention-output covariance 1.7 s.
The MLP-sized steps (28672-wide covariance, Monte Carlo) need ~7 GB for two 28672² fp32 matrices and were not run
on this 7.6 GB machine. 70B needs a GPU (≥ 24 GB, 40–80 GB comfortable). Work per layer is the same order as GPTQ's.
