# shapequant

Data-free, training-free quantization of Hugging Face causal language models.

shapequant uses no calibration text and no fine-tuning. Each layer's input second moment is derived from the weights and
the architecture. The weights are then rounded with error feedback under the design matrix $H = I + \lambda S$.
The output is a standard GPTQ v1 checkpoint (INT4/INT8, asymmetric, group-wise, optional per-module mixed precision).

Documentation with figures: `docs/index.html`, served by GitHub Pages from the `docs/` folder.

---

## Install

```bash
pip install -e .                 # torch>=2.1, transformers>=4.40, accelerate, safetensors, huggingface_hub
pip install -e ".[eval,test]"    # + datasets (WikiText-2 download) and pytest
```

## Quick start

```python
from shapequant import quantize_model, load_quantized, perplexity, wikitext2_windows, estimate_memory

estimate_memory("path/or/config.json")                         # peak-memory estimate from a config only
quantize_model("meta-llama/Llama-3.3-70B-Instruct", "out70b", bits=4, group_size=128, device="cuda")
quantize_model("HuggingFaceTB/SmolLM2-135M", "smol", budget=5.0, group_size=64)      # mixed INT4/INT8/FP16
quantize_model("facebook/opt-125m", "opt-rtn", bits=4, method="rtn")                  # baseline, same grid
model = load_quantized("smol", device="cpu")                                          # or device_map="auto"
```

```bash
shapequant memory  meta-llama/Llama-3.1-70B
shapequant quantize meta-llama/Llama-3.1-70B out/ --bits 4 --group-size 128 --device cuda
shapequant quantize HuggingFaceTB/SmolLM2-135M out/ --budget 5.0 --group-size 64
shapequant eval out/ --windows 32 [--text wiki.test.raw] [--bos]
```

### `quantize_model(model, out_dir, ...)`

| argument | default | meaning |
|---|---|---|
| `bits` | 4 | uniform bit width, 4 or 8 |
| `group_size` | 128 | quantization group along the input dimension; must divide every input width |
| `method` | `"shaped"` | `"shaped"` (data-free design $H=I+\lambda S$) or `"rtn"` (round to nearest, same grid) |
| `budget` | `None` | average bits/weight for mixed INT4/INT8/FP16 per module; must be ≥ $4 + 32/g$ |
| `lam`, `kappa` | 1.0, 1.0 | $H = I + \lambda S$; attention concentration $\kappa$ (1 = theory value) |
| `n_samples` | 16384 | Monte Carlo samples for the MLP-input moment |
| `dynamic_groups` | `True` | fit each group's grid to the error-fed weights at the start of the group (GPTQ behaviour) |
| `device` | cuda if available | device for the per-layer work |
| `max_shard_size_gb` | 4.0 | output shard size |

Supported `model_type`:
* `llama`, `mistral`, `qwen2` (SiLU-gated MLP, RMSNorm, GQA, optional q/k/v biases); covers Llama-2/3/3.1/3.3 and Qwen2.5 up to 70–72B;
* `opt` (pre-LN only);
* `gpt2`.

---

## Method

### Objective

For a linear block $y = xW$ with quantization error $R = W - \hat W$ and input second moment $\Sigma = \mathbb{E}[x x^\top]$,

$$\mathbb{E}\,\lVert xR \rVert^2 = \operatorname{tr}\left(R^\top \Sigma R\right).$$

Round-to-nearest is optimal only for $\Sigma \propto I$. shapequant replaces the unknown $\Sigma$ with a model-implied
surrogate $S$ (trace-normalised to $n$) and minimises $\operatorname{tr}(R H R^\top)$ with

$$H = I + \lambda S,\qquad \lambda = 1 .$$

If $\Sigma = S + E$ with $0 \preceq E \preceq \rho I$, the objective increases with $\Sigma$ in the PSD order. The worst
case is therefore $S + \rho I$, and the minimax design is that matrix. The identity term is the uncertainty radius, not a tuned damping.

### Model-implied second moments (no data)

**Residual stream.**
* Byte-level BPE ids follow merge order, which tracks frequency, so a Zipf prior on the id gives the layer-0 moment.
  $e_t$ are the token embeddings and $p_s$ the learned positions (OPT, GPT-2):

$$\Sigma^{(0)} = \sum_t \pi_t\, e_t e_t^\top + \frac1T \sum_{s} p_s p_s^\top + (\bar e\,\bar p^\top + \bar p\,\bar e^\top),\qquad \pi_t \propto \frac{1}{t + 10}.$$

* Each writer then adds its contribution, assuming isotropic inputs to the writer:

$$\Sigma \leftarrow \Sigma + W_o^\top W_o \quad\text{(after attention)},\qquad \Sigma \leftarrow \Sigma + W_{\mathrm{down}}^\top W_{\mathrm{down}} \quad\text{(after the MLP)}.$$

**Normalisation outputs** (inputs of q/k/v and of gate/up or fc1). Gain $g$, bias $\beta$, $P = I - \tfrac1n\mathbf 1\mathbf 1^\top$, residual mean $m$:

$$\text{LayerNorm: } C = \frac{P\Sigma P}{\operatorname{tr}(P\Sigma P)/n},\quad \operatorname{cov} = g\circ C\circ g,\quad \mu = \beta,\quad S = \operatorname{cov} + \beta\beta^\top$$

$$\text{RMSNorm: } C = \frac{\Sigma}{\operatorname{tr}(\Sigma)/n},\quad S = g\circ C\circ g,\quad \mu = \frac{g\circ m}{\sqrt{\operatorname{tr}(\Sigma)/n}},\quad \operatorname{cov} = S - \mu\mu^\top$$

**MLP second input** (input of down_proj / fc2 / c_proj). Monte Carlo over the model-implied Gaussian:

$$S_{\mathrm{act}} = \mathbb{E}\left[\varphi(x)\varphi(x)^\top\right],\qquad x \sim \mathcal N(\mu, \operatorname{cov}),$$

$$\varphi(x) = \operatorname{silu}(xW_g)\odot(xW_u)\ \text{(Llama family)},\qquad \operatorname{relu}(xW_1 + b_1)\ \text{(OPT)},\qquad \operatorname{gelu}(xW_1 + b_1)\ \text{(GPT-2)}.$$

**Attention output** (input of o_proj). Each head mixes value vectors; $g(h)$ is the KV group of query head $h$:

$$S_{\mathrm{attn}} = \mu_o\mu_o^\top + \kappa\,\operatorname{blockdiag}_h\left(W_v^{(g(h))}\operatorname{cov}\,W_v^{(g(h))\top}\right),\qquad \kappa = \mathbb{E}\sum_t a_t^2 = 1 .$$

### Rounding

Fix the asymmetric GPTQ grid per (output row, input group):
* the range includes 0;
* $s = (\max - \min)/(2^b - 1)$ and $z = \operatorname{round}(-\min/s)$, kept $\ge 1$.

Factor $H^{-1} = U^\top U$ with $U$ upper-triangular. Then for every input column $i$:

$$q_i = \operatorname{clamp}\left(\operatorname{round}(w_i/s) + z,\ 0,\ 2^b - 1\right),\qquad e_i = \frac{w_i - (q_i - z)s}{U_{ii}},\qquad w_j \leftarrow w_j - e_i U_{ij}\ \ (j > i).$$

* $H = I$ gives exactly round-to-nearest.
* Updates are applied lazily in blocks of 128 columns.
* With `dynamic_groups=True` each group's $(s, z)$ is fitted to the error-fed weights when the group starts.
* Modules that read the same input and have the same bit width (q/k/v, gate/up) are stacked and rounded in one pass.

**Gain bound.** In the high-resolution model with design $H = LDL^\top$ the output MSE is
$\tfrac{\Delta^2}{12}\, m \operatorname{tr}(L^{-1}\Sigma L^{-\top})$. This is minimised at $H = \Sigma$, so for any design

$$\frac{\text{MSE}_{\mathrm{RTN}}}{\text{MSE}_{H}} \le \frac{\operatorname{tr}\Sigma}{\operatorname{tr}D_\Sigma} \le \frac{\operatorname{AM}(\lambda(\Sigma))}{\operatorname{GM}(\lambda(\Sigma))}.$$

### Mixed precision

Per module $\ell$ with $n_\ell$ weights, the bit width $b_\ell \in \{4, 8, 16\}$ (cost $4 + 32/g$, $8 + 32/g$, $16$ bits/weight) solves

$$\min_{b}\ \sum_\ell s_\ell\, D_\ell(b_\ell)\quad \text{s.t.}\quad \sum_\ell n_\ell\, c(b_\ell) \le B \sum_\ell n_\ell,\qquad D_\ell(b) = \lVert W_\ell - Q_b(W_\ell)\rVert_F^2,\qquad s_\ell = \frac{w_{\mathrm{type}(\ell)}}{\lVert W_\ell\rVert_F^2}.$$

* The type weights are 4 for modules that write linearly into the residual stream (v, o, down) and 1 for the rest.
* The problem is solved by Lagrangian bisection; leftover budget is then spent greedily on the best marginal upgrades.

---

## Checkpoint format

GPTQ v1 (AutoGPTQ / GPTQModel layout), `quantization_config.quant_method = "gptq"`, `checkpoint_format = "gptq"`:

| tensor | dtype | shape | content |
|---|---|---|---|
| `qweight` | int32 | (in·b/32, out) | codes packed along the input dimension |
| `qzeros` | int32 | (in/g, out·b/32) | zero-points minus 1, packed along the output dimension |
| `scales` | fp16 | (in/g, out) | group scales |
| `g_idx` | int32 | (in,) | `arange(in) // g` |
| `bias` | fp16 | (out,) | if present |

* `sym=False`, `desc_act=False`.
* Mixed-precision overrides use the GPTQModel `dynamic` convention: `"+:<regex>": {"bits": 8}` and `"-:<regex>": {}` (FP16, not quantized).
* Embeddings, norms and an untied `lm_head` are stored in fp16.
* `shapequant_summary.json` records the method, the parameters and the per-module bit widths.

## Scaling

* **Streaming.** Tensors are read one at a time from the safetensors shards. Only one decoder layer is resident, and it is written before the next is read.
* **Per-layer state:**
  * the residual moment $\Sigma$ ($d\times d$, fp64);
  * the four $S$ matrices;
  * the stacked weights of one input group;
  * $H$ and its Cholesky chain.
* **Peak device memory** is about one layer in fp32 plus two $n\times n$ fp32 matrices for the widest input
  ($n = 28672$ for Llama-3.1-70B).

| model | params | est. device peak | INT4 g128 checkpoint |
|---|---|---|---|
| Llama-2-7B | 6.7B | 4.0 GB | 3.7 GB |
| Llama-3.1-8B | 8.0B | 5.4 GB | 5.4 GB |
| Llama-2-13B | 13.0B | 6.2 GB | 6.9 GB |
| Qwen2.5-32B | 32.8B | 16.1 GB | 18.3 GB |
| Llama-3.1-70B | 70.6B | 21.2 GB | 37.8 GB |
| Qwen2.5-72B | 72.7B | 22.1 GB | 39.4 GB |

These are estimates (`shapequant memory`). The attention-sized steps were executed at full Llama-3.1-70B shapes on a
2-core CPU: stacked q/k/v (10240×8192) in 26 s, o_proj (8192×8192) in 19 s.

## Results

WikiText-2 test, 32 windows × 1024 tokens, CPU, one run per setting. Defaults: $\lambda = 1$, $\kappa = 1$, dynamic group grids.
"Mixed" budget: 4.75 bits/weight (GPT-2, OPT) or 5.0 (SmolLM2).

| model | group | fp16 | RTN INT4 | shaped INT4 | RTN mixed | shaped mixed |
|---|---|---|---|---|---|---|
| GPT-2 124M | 128 | 26.84 | 29.91 | **29.14** | 29.62 | **28.86** |
| OPT-125m | 128 | 29.27 | 31.98 | **31.39** | 30.71 | **30.44** |
| SmolLM2-135M (Llama) | 64 | 15.46 | **19.28** | 19.29 | **17.82** | 17.83 |

Exact per-layer output error relative to RTN under the true input covariance (OPT-125m, layers 0/4/8):
* geometric mean 0.868 with the data-free $H$;
* 0.282 with the true covariance, which uses text and is shown for reference only.

`RESULTS.md` has the full numbers, including static-grid variants.

## Verification

```bash
pytest -q tests/test_core.py tests/test_end_to_end.py
python tests/real_models.py <hf_model> <out_dir> --group 128 --budget 4.75 --windows 32
python tests/scale_70b.py scale_70b.json
```

* `test_core.py` checks that:
  * $H = I$ reproduces RTN in both grid modes;
  * the blocked recursion equals the unblocked one for every block size and grid mode;
  * GPTQ pack/unpack is exact for 4 and 8 bits;
  * zero-points are always ≥ 1;
  * shaping lowers the $H$-weighted error.
* `test_end_to_end.py` uses tiny random Llama, Qwen2, OPT and GPT-2 models, uniform and mixed. In each case
  quantize → save → load gives logits equal to a reference model holding the dequantized weights.

## Limits

* **Gains:** at standard INT4 the perplexity gains are small (−2.6%, −1.9%, 0.0% on the three models tested). The research experiments
  showed much larger gains at 3 bits, but 3-bit is not exported.
* **70B:** no end-to-end run yet. It needs a GPU (≥ 24 GB).
* **Serving:** loading with GPTQModel or vLLM kernels has not been tested. The format follows the GPTQ v1 layout.
* **Architectures:** not supported are Gemma-style norms, fused-QKV Llama variants (Phi-3), MoE and post-LN OPT.

## Layout

```
shapequant/        package: arch, io, sigma, core, quantize, qlinear, evaluate, memory, cli
tests/             unit, end-to-end, real-model and 70B-scaling scripts
docs/              GitHub Pages site (index.html, images/)
tools/             figure data (figure_data.json) and scripts that rebuild docs/images
RESULTS.md         full result tables
```

## Citation

```bibtex
@software{majumdar_shapequant_2026,
  author = {Majumdar, Angshul},
  title  = {shapequant: data-free noise-shaped quantization of large language models},
  year   = {2026},
  url    = {https://github.com/AngshulMajumdar/shapequant}
}
```

License: MIT (see `LICENSE`).
