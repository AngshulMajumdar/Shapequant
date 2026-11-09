"""Data-free, training-free quantization of Hugging Face causal LMs, streamed one decoder layer at a time.

    quantize_model("meta-llama/Llama-3.3-70B-Instruct", "out/llama70b-sq-int4", bits=4, group_size=128)

method="shaped": noise-shaped rounding with the model-implied input covariance H = I + lam*S (default, lam = 1).
method="rtn":    plain round-to-nearest on the same grid (baseline).
budget=<bits/weight>: mixed INT4 / INT8 / FP16 per module by the type rule (writers into the residual stream,
                  i.e. V, attention-output and MLP-output projections, count 4x). None = uniform `bits`.
Output: GPTQ-format checkpoint (quant_method "gptq", checkpoint_format "gptq", sym=False, desc_act=False),
per-module bit overrides in quantization_config["dynamic"], plus everything else unquantized in fp16.
"""
import os, json, time, math
import torch

from . import core, sigma
from .arch import from_config
from .io import resolve, Reader, ShardWriter, copy_side_files

__all__ = ["quantize_model", "allocate"]

TYPE_WEIGHT = {"q": 1.0, "k": 1.0, "qkv": 1.0, "gate": 1.0, "up": 1.0, "v": 4.0, "o": 4.0, "down": 4.0}


def _get_linear(reader, lin, device):
    W = reader.get(lin.name + ".weight", device=device)
    if lin.conv1d:
        W = W.T.contiguous()
    b = reader.get(lin.name + ".bias", device=device) if reader.has(lin.name + ".bias") else None
    return W, b


def _default_device(device):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def allocate(arch, reader, budget, group_size, device, log=print):
    """Per-module bits in {4, 8, 16} minimising sum_l s_l D_l(b_l) under an average bits/weight budget.
    s_l = type weight / ||W_l||_F^2, D_l(b) = ||W_l - RTN_b(W_l)||_F^2 (exact). One streaming pass over the weights."""
    over = 32.0 / group_size
    cost = torch.tensor([4 + over, 8 + over, 16.0], dtype=torch.float64)
    names, D, n, s = [], [], [], []
    for L in range(arch.n_layer):
        for lin in arch.layers[L]:
            W, _ = _get_linear(reader, lin, device)
            row = []
            for b in (4, 8):
                q, sc, z = core.quantize(W, b, group_size)
                row.append(float(((W - core.dequantize(q, sc, z, group_size)) ** 2).sum()))
            row.append(float(((W - W.half().float()) ** 2).sum()))
            names.append(lin.name); D.append(row); n.append(W.numel()); s.append(TYPE_WEIGHT[lin.role] / float((W ** 2).sum()))
        log(f"[allocate] scanned layer {L + 1}/{arch.n_layer}")
    D, n, s = torch.tensor(D, dtype=torch.float64), torch.tensor(n, dtype=torch.float64), torch.tensor(s, dtype=torch.float64)
    N = n.sum()
    used = lambda k: float((n * cost[k]).sum() / N)
    pick = lambda mu: (s[:, None] * D + mu * n[:, None] * cost[None, :]).argmin(1)
    if used(torch.zeros(len(n), dtype=torch.long)) > budget + 1e-9:
        raise ValueError(f"budget {budget} is below the INT4 floor {cost[0]:.3f} bits/weight")
    lo, hi = 0.0, 1.0
    while used(pick(hi)) > budget:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if used(pick(mid)) > budget else (lo, mid)
    k = pick(hi)
    while True:                                        # spend leftover budget on the best marginal upgrades
        best, gain = None, 0.0
        for l in range(len(k)):
            if k[l] < 2:
                extra = float(n[l] * (cost[k[l] + 1] - cost[k[l]]) / N)
                if used(k) + extra <= budget + 1e-12:
                    r = float(s[l] * (D[l, k[l]] - D[l, k[l] + 1]) / (n[l] * (cost[k[l] + 1] - cost[k[l]])))
                    if r > gain:
                        best, gain = l, r
        if best is None:
            break
        k[best] += 1
    bits = {nm: (4, 8, 16)[int(x)] for nm, x in zip(names, k)}
    return bits, used(k)


def quantize_model(model, out_dir, bits=4, group_size=128, method="shaped", budget=None, lam=1.0, kappa=1.0,
                   n_samples=16384, device=None, max_shard_size_gb=4.0, revision=None, seed=0, dynamic_groups=True, log=print):
    """Quantize a Hugging Face causal LM (local dir or hub id) and write a GPTQ-format checkpoint to out_dir.
    Peak memory: one decoder layer of weights plus a few (in x in) matrices; see shapequant.memory.estimate."""
    t0 = time.time()
    assert method in ("shaped", "rtn")
    assert bits in (4, 8) or budget is not None
    dev = _default_device(device)
    src = resolve(model, revision)
    cfg = json.load(open(os.path.join(src, "config.json")))
    arch = from_config(cfg)
    reader = Reader(src)
    if budget is not None:
        bits_of, bpw = allocate(arch, reader, budget, group_size, dev, log)
        log(f"[allocate] {sum(v == 4 for v in bits_of.values())} INT4, {sum(v == 8 for v in bits_of.values())} INT8, "
            f"{sum(v == 16 for v in bits_of.values())} FP16 modules; {bpw:.3f} bits/weight")
    else:
        bits_of = None

    writer = ShardWriter(out_dir, int(max_shard_size_gb * 2 ** 30))
    quantized = set()
    prefixes = tuple(_layer_prefixes(arch))
    for key in reader.all_keys():                          # everything outside the decoder layers: fp16 copy
        ck = _canon(key, arch)
        if ck.startswith("lm_head") and arch.lm_head is None:     # tied to the input embedding
            continue
        if not ck.startswith(prefixes):
            writer.add(ck, reader.get(key, dtype=torch.float16))

    res = sigma.ResidualModel(arch, reader, dev) if method == "shaped" else None
    for L in range(arch.n_layer):
        tl = time.time()
        lins = {lin.name: lin for lin in arch.layers[L]}
        W = {nm: _get_linear(reader, lin, dev) for nm, lin in lins.items()}
        S = {}
        if method == "shaped":
            m1 = res.norm_moments(L, "ln1")
            S["ln1"] = sigma.S_norm(m1)
            v = [lin for lin in lins.values() if lin.role in ("v", "qkv")][0]
            Wv, bv = W[v.name]
            if v.role == "qkv":
                Wv, bv = Wv[2 * arch.d:3 * arch.d], (None if bv is None else bv[2 * arch.d:3 * arch.d])
            S["attn_out"] = sigma.S_attn(arch, m1, Wv, bv, kappa)
            o = [lin for lin in lins.values() if lin.role == "o"][0]
            res.add_writer(W[o.name][0])
            m2 = res.norm_moments(L, "ln2")
            S["ln2"] = sigma.S_norm(m2)
            up = [lin for lin in lins.values() if lin.role == "up"][0]
            gate = [lin for lin in lins.values() if lin.role == "gate"]
            S["act"] = sigma.S_act(arch, m2, W[up.name][0], W[up.name][1], W[gate[0].name][0] if gate else None,
                                   n_samples=n_samples, seed=seed + L, device=dev)
            dn = [lin for lin in lins.values() if lin.role == "down"][0]
            res.add_writer(W[dn.name][0])
        # group modules that read the same input and use the same bit width: one shaped pass on the stacked rows
        groups = {}
        for nm, lin in lins.items():
            b = bits if bits_of is None else bits_of[nm]
            groups.setdefault((lin.input_key, b), []).append(nm)
        uses = {}
        for (ik, b) in groups:
            if b != 16:
                uses[ik] = uses.get(ik, 0) + 1
        for (ik, b), nms in groups.items():
            if b == 16:
                for nm in nms:
                    writer.add(nm + ".weight", (W[nm][0].T if lins[nm].conv1d else W[nm][0]).half())
                    if W[nm][1] is not None:
                        writer.add(nm + ".bias", W[nm][1].half())
                continue
            Wcat = torch.cat([W[nm][0] for nm in nms], 0)
            H = None
            if method == "shaped":
                uses[ik] -= 1                                     # last use: reuse S's memory for H
                H = S.pop(ik) if uses[ik] == 0 else S[ik].clone()
                H = H.to(device=dev, dtype=torch.float32); H.mul_(lam); H.diagonal().add_(1.0)
            q, sc, z = core.quantize(Wcat, b, group_size, H=H, dynamic_groups=dynamic_groups)
            del H
            r0 = 0
            for nm in nms:
                r1 = r0 + W[nm][0].shape[0]
                pk = core.pack(q[r0:r1], sc[r0:r1], z[r0:r1], b, group_size)
                for k_, t in pk.items():
                    writer.add(nm + "." + k_, t)
                if W[nm][1] is not None:
                    writer.add(nm + ".bias", W[nm][1].half())
                quantized.add(nm); r0 = r1
        for which in ("ln1", "ln2"):
            gname, bname = arch.norm_names(L, which)
            writer.add(gname, reader.get(gname, dtype=torch.float16))
            if bname and reader.has(bname):
                writer.add(bname, reader.get(bname, dtype=torch.float16))
        del W, S
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log(f"[quantize] layer {L + 1}/{arch.n_layer} done in {time.time() - tl:.1f}s")
    total = writer.close()

    qc = {"quant_method": "gptq", "bits": bits if bits_of is None else 4, "group_size": group_size, "desc_act": False,
          "sym": False, "static_groups": not dynamic_groups, "true_sequential": False, "damp_percent": 0.0, "checkpoint_format": "gptq",
          "pack_dtype": "int32", "lm_head": False,
          "quantizer": ["shapequant:0.1.0"],
          "shapequant": {"method": method, "lam": lam, "kappa": kappa, "dynamic_groups": dynamic_groups, "n_samples": n_samples, "budget": budget,
                         "data_free": True, "training_free": True}}
    if bits_of is not None:
        qc["dynamic"] = {}
        for nm, b in bits_of.items():
            if b == 8:
                qc["dynamic"]["+:^" + nm.replace(".", r"\.") + "$"] = {"bits": 8}
            elif b == 16:
                qc["dynamic"]["-:^" + nm.replace(".", r"\.") + "$"] = {}
    cfg_out = dict(cfg); cfg_out["quantization_config"] = qc; cfg_out["torch_dtype"] = "float16"
    json.dump(cfg_out, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    copy_side_files(src, out_dir)
    summary = {"model": model, "out_dir": out_dir, "method": method, "group_size": group_size,
               "bits": bits if bits_of is None else "mixed", "checkpoint_bytes": total, "minutes": (time.time() - t0) / 60}
    if bits_of is not None:
        summary["bits_per_module"] = bits_of
    json.dump(summary, open(os.path.join(out_dir, "shapequant_summary.json"), "w"), indent=2)
    log(f"[done] {out_dir}: {total / 2 ** 30:.2f} GiB in {summary['minutes']:.1f} min")
    return summary


def _layer_prefixes(arch):
    return [arch.norms[L][0].rsplit(".", 1)[0] + "." for L in range(arch.n_layer)]


def _canon(key, arch):
    """Key with the canonical root used by the architecture ('model.' for llama/opt, 'transformer.' for gpt2)."""
    root = "transformer." if arch.model_type == "gpt2" else "model."
    if key.startswith((root, "lm_head")):
        return key
    return root + key
