"""Collects the small derived data behind the documentation figures into tools/figure_data.json.
Needs the research artifacts (true input covariances from held-out text, local model folders); the figures
themselves are rebuilt from the JSON by tools/make_figures.py without any of that."""
import json, sys, numpy as np, torch
from shapequant import sigma, estimate_memory
from shapequant.arch import from_config
from shapequant.io import Reader
from shapequant.quantize import _get_linear

OPT_DIR, ST_OPT = sys.argv[1], sys.argv[2]           # HF folder of facebook/opt-125m, true-Σ npz (evaluation only)
RES = {k: json.load(open(p)) for k, p in zip(("gpt2", "opt125m", "smollm2"), sys.argv[3:6])}
out = {}
a = from_config(json.load(open(OPT_DIR + "/config.json"))); r = Reader(OPT_DIR); res = sigma.ResidualModel(a, r, "cpu")
ST = np.load(ST_OPT)
spec = {}
for L in range(5):
    lins = {l.name: l for l in a.layers[L]}; W = {n: _get_linear(r, l, "cpu") for n, l in lins.items()}
    m1 = res.norm_moments(L, "ln1")
    if L == 4:
        S_hat = sigma.S_norm(m1).numpy(); T = ST[f"{L}_ln1"]; T = T * len(T) / np.trace(T)
        e_hat, e_true = np.sort(np.linalg.eigvalsh(S_hat))[::-1], np.sort(np.linalg.eigvalsh(T))[::-1]
        H_hat = np.eye(len(T)) + S_hat
        spec = {"layer": 4, "input": "attention input (after LayerNorm)", "true": e_true.clip(1e-6).tolist(),
                "data_free": e_hat.clip(1e-6).tolist(),
                "flatness_bound_true": float(np.mean(e_true.clip(1e-9)) / np.exp(np.mean(np.log(e_true.clip(1e-9))))),
                "cos_sim": float(np.sum(S_hat * T) / np.linalg.norm(S_hat) / np.linalg.norm(T))}
    res.add_writer(W[[l for l in lins.values() if l.role == "o"][0].name][0])
    res.add_writer(W[[l for l in lins.values() if l.role == "down"][0].name][0])
out["spectrum"] = spec
out["layer_ratios"] = {  # exact output-error ratio vs RTN under the true covariance, OPT-125m (tests: e2e/diag.py)
    "modules": ["q", "k", "v", "o", "up", "down"], "layers": [0, 4, 8],
    "data_free": [[0.771, 0.767, 0.770, 0.895, 0.903, 0.515], [0.956, 0.968, 0.988, 0.936, 0.918, 0.830], [0.974, 0.972, 0.988, 0.974, 0.937, 0.738]],
    "true_cov":  [[0.331, 0.337, 0.330, 0.191, 0.263, 0.146], [0.194, 0.194, 0.264, 0.318, 0.265, 0.355], [0.360, 0.354, 0.387, 0.299, 0.355, 0.308]]}
out["results"] = {k: {kk: (v if isinstance(v, float) else v["ppl"]) for kk, v in R.items() if kk in ("dense", "rtn_int4", "shaped_int4", "rtn_mixed", "shaped_mixed")} for k, R in RES.items()}
C = {"Llama-2-7B": (4096, 11008, 32, 32, 32, 32000), "Llama-3.1-8B": (4096, 14336, 32, 32, 8, 128256),
     "Llama-2-13B": (5120, 13824, 40, 40, 40, 32000), "Qwen2.5-32B": (5120, 27648, 64, 40, 8, 152064),
     "Llama-3.1-70B": (8192, 28672, 80, 64, 8, 128256), "Qwen2.5-72B": (8192, 29568, 80, 64, 8, 152064)}
out["memory"] = {}
for k, (d, i, L, H, kv, V) in C.items():
    e = estimate_memory({"model_type": "qwen2" if "Qwen" in k else "llama", "hidden_size": d, "intermediate_size": i, "num_hidden_layers": L,
                         "num_attention_heads": H, "num_key_value_heads": kv, "vocab_size": V, "hidden_act": "silu", "tie_word_embeddings": False})
    out["memory"][k] = e
json.dump(out, open("tools/figure_data.json", "w"), indent=1)
print({k: (v if k != "spectrum" else {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}) for k, v in out.items() if k != "layer_ratios"})
