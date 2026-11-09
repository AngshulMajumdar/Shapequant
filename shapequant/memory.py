"""Peak-memory estimate for quantize_model from a config (no weights needed)."""
import json, os
from .arch import from_config


def estimate(config, n_samples=16384, batch=2048, method="shaped"):
    """config: dict, path to config.json, or a directory. Returns bytes for the accelerator and for host RAM."""
    if isinstance(config, str):
        p = os.path.join(config, "config.json") if os.path.isdir(config) else config
        config = json.load(open(p))
    a = from_config(config)
    f32, f64 = 4, 8
    lin = sum(_numel(a, l) for l in a.layers[0])                       # one decoder layer, fp32 on device
    ins = {"ln1": a.d, "attn_out": a.n_head * a.head_dim, "ln2": a.d, "act": a.inter}
    # quantizing the largest input group: stacked W (fp32) + H + Cholesky chain (2 alive) + error buffer
    worst = 0
    for key, n in ins.items():
        rows = sum(_out(a, l) for l in a.layers[0] if l.input_key == key)
        worst = max(worst, rows * n * f32 * 2 + 2 * n * n * f32 + rows * 128 * f32 * 2)
    shaped = 0
    if method == "shaped":
        shaped = (3 * a.d * a.d * f64                                   # residual Sig, moments
                  + 2 * a.inter * a.inter * f32                         # S_act accumulator + H
                  + batch * (a.d + 2 * a.inter) * f32                   # Monte Carlo batch
                  + 4 * (a.n_head * a.head_dim) ** 2 * f64)             # attention-output S
    embed = a.vocab * a.d * f32
    device_peak = lin * f32 + max(worst + shaped, embed + 2 * a.d * a.d * f64)
    n_params = a.n_layer * lin + a.vocab * a.d * (1 if a.lm_head is None else 2)
    out_bytes = a.n_layer * lin * (4.25 / 8) + a.vocab * a.d * 2 * (1 if a.lm_head is None else 2)
    return {"model_type": a.model_type, "params_B": n_params / 1e9, "device_peak_GB": device_peak / 2 ** 30,
            "host_buffer_GB": 4 + lin * 2 / 2 ** 30, "int4_checkpoint_GB": out_bytes / 2 ** 30,
            "largest_H_dim": max(ins.values())}


def _out(a, l):
    return {"q": a.n_head * a.head_dim, "k": a.n_kv * a.head_dim, "v": a.n_kv * a.head_dim, "qkv": 3 * a.d,
            "o": a.d, "gate": a.inter, "up": a.inter, "down": a.d}[l.role]


def _in(a, l):
    return {"ln1": a.d, "attn_out": a.n_head * a.head_dim, "ln2": a.d, "act": a.inter}[l.input_key]


def _numel(a, l):
    return _out(a, l) * _in(a, l)
