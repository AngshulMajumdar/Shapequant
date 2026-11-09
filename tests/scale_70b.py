"""70B scaling check without the weights: memory estimates from real configs, and timing of the per-layer core at
Llama-3.1-70B shapes on whatever device is available (random weights; the arithmetic is shape-dependent only)."""
import json, sys, time, torch
from shapequant import estimate_memory, core, sigma
from shapequant.arch import from_config

CFGS = {
 "Llama-3.1-70B": {"model_type": "llama", "hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
                   "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 128256, "hidden_act": "silu",
                   "rms_norm_eps": 1e-5, "max_position_embeddings": 131072, "tie_word_embeddings": False},
 "Qwen2.5-72B":   {"model_type": "qwen2", "hidden_size": 8192, "intermediate_size": 29568, "num_hidden_layers": 80,
                   "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 152064, "hidden_act": "silu",
                   "rms_norm_eps": 1e-6, "max_position_embeddings": 32768, "tie_word_embeddings": False},
 "Llama-2-70B":   {"model_type": "llama", "hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
                   "num_attention_heads": 64, "num_key_value_heads": 8, "vocab_size": 32000, "hidden_act": "silu",
                   "rms_norm_eps": 1e-5, "max_position_embeddings": 4096, "tie_word_embeddings": False},
}
out = {k: estimate_memory(v) for k, v in CFGS.items()}
print(json.dumps(out, indent=1))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = from_config(CFGS["Llama-3.1-70B"]); d = a.d; t = {}
torch.manual_seed(0)
# attention-input group (q,k,v stacked: 10240 x 8192) and attention-output (8192 x 8192) at full 70B size
for name, rows in (("qkv_stacked", 64 * 128 + 2 * 8 * 128), ("o_proj", d)):
    W = torch.randn(rows, d, device=dev) * 0.02
    A = torch.randn(d, 256, device=dev) / 16; H = A @ A.T; H.diagonal().add_(1.0)
    t0 = time.time(); q, s, z = core.quantize(W, 4, 128, H=H); t[name] = time.time() - t0
    pk = core.pack(q, s, z, 4, 128); assert pk["qweight"].shape == (d // 8, rows)
    del W, H, A, q, s, z, pk
# attention-output covariance construction at 70B size (GQA 64 query / 8 KV heads)
mu = torch.zeros(d, dtype=torch.float64, device=dev); cov = torch.eye(d, dtype=torch.float64, device=dev)
t0 = time.time(); S = sigma.S_attn(a, (mu, cov, cov), torch.randn(8 * 128, d, device=dev) * 0.02, None); t["S_attn"] = time.time() - t0
assert S.shape == (d, d); del S, cov
print(json.dumps({"device": str(dev), "seconds": t}, indent=1))
json.dump({"estimates": out, "timing": {"device": str(dev), "seconds": t}}, open(sys.argv[1] if len(sys.argv) > 1 else "scale_70b.json", "w"), indent=1)
