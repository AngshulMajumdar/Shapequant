"""Loading shapequant / GPTQ-v1 checkpoints with plain transformers + torch (no CUDA kernels needed).

The checkpoint is standard GPTQ format, so GPTQ-capable stacks (GPTQModel, vLLM, TGI) can load it with their
fast kernels. This loader is the dependency-light path: each quantized layer keeps packed int32 weights and
dequantizes on the fly (optionally caching the fp16 weight).
"""
import os, re, json, glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import core


class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bits, group_size, bias, dtype=torch.float16, cache=False):
        super().__init__()
        self.in_features, self.out_features, self.bits, self.group_size, self.cache = in_features, out_features, bits, group_size, cache
        per = 32 // bits; G = in_features // group_size
        self.register_buffer("qweight", torch.zeros((in_features // per, out_features), dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros((G, out_features // per), dtype=torch.int32))
        self.register_buffer("scales", torch.zeros((G, out_features), dtype=torch.float16))
        self.register_buffer("g_idx", torch.zeros(in_features, dtype=torch.int32))
        self.register_buffer("bias", torch.zeros(out_features, dtype=dtype) if bias else None)
        self._w = None

    def weight_fp(self, dtype):
        if self._w is not None and self._w.dtype == dtype:
            return self._w
        w = core.unpack(self.qweight, self.qzeros, self.scales, self.g_idx, self.bits).to(dtype)
        if self.cache:
            self._w = w
        return w

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight_fp(x.dtype), b)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, bits={self.bits}, group={self.group_size}"


def _bits_for(name, qc):
    bits = qc.get("bits", 4)
    for pat, ov in (qc.get("dynamic") or {}).items():
        sign, rx = (pat[0], pat[2:]) if pat[:2] in ("+:", "-:") else ("+", pat)
        if re.match(rx, name):
            if sign == "-":
                return None
            bits = ov.get("bits", bits)
    return bits


def _set_module(model, name, mod):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, mod)


def load_quantized(path, device="cpu", dtype=torch.float16, cache_weights=False, device_map=None):
    """Load a checkpoint written by quantize_model. device_map="auto" dispatches layers over GPUs/CPU (accelerate)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    cfg = AutoConfig.from_pretrained(path)
    qc = cfg.quantization_config if isinstance(cfg.quantization_config, dict) else cfg.quantization_config.to_dict()
    del cfg.quantization_config
    with init_empty_weights(include_buffers=False):
        try:
            model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)          # transformers >= 4.56
        except TypeError:
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    sd = {}
    for f in files:
        sd.update(load_file(f))
    qnames = sorted({k[: -len(".qweight")] for k in sd if k.endswith(".qweight")})
    msd = model.state_dict()
    def find(name):                      # map checkpoint module name to model module name (prefix differences)
        for cand in (name, name.split(".", 1)[1], "model." + name):
            try:
                model.get_submodule(cand); return cand
            except AttributeError:
                continue
        raise KeyError(name)
    ren = {}
    for qn in qnames:
        mn = find(qn); old = model.get_submodule(mn)
        if hasattr(old, "in_features"):
            inn, out = old.in_features, old.out_features
        else:                            # GPT-2 Conv1D: weight (in, out)
            inn, out = old.weight.shape
        b = _bits_for(qn, qc)
        _set_module(model, mn, QuantLinear(inn, out, b, qc["group_size"], (qn + ".bias") in sd, dtype, cache_weights))
        ren[qn] = mn
    fixed = {}
    for k, v in sd.items():
        mod = k.rsplit(".", 1)[0]
        if mod in ren:
            k = ren[mod] + "." + k.rsplit(".", 1)[1]
        elif k not in msd:
            for cand in (k.split(".", 1)[1], "model." + k):
                if cand in msd:
                    k = cand; break
        fixed[k] = v.to(dtype) if v.is_floating_point() and not k.endswith(".scales") else v
    missing, unexpected = model.load_state_dict(fixed, strict=False, assign=True)
    model.tie_weights()
    still = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if still:
        raise RuntimeError(f"weights missing from checkpoint: {still[:5]}")
    model.eval()
    if device_map is not None:
        from accelerate import dispatch_model, infer_auto_device_map
        return dispatch_model(model, infer_auto_device_map(model, no_split_module_classes=model._no_split_modules))
    return model.to(device)
