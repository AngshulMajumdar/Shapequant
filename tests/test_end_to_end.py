"""Tiny random models of every supported architecture: quantize -> save -> load -> logits must equal a model whose
weights are the dequantized values (exactness of format, loader and module mapping)."""
import json, os, tempfile
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, OPTConfig, GPT2Config, Qwen2Config

from shapequant import quantize_model, load_quantized, core

CFGS = {
    "llama": LlamaConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, max_position_embeddings=256, tie_word_embeddings=False),
    "qwen2": Qwen2Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=512, max_position_embeddings=256, tie_word_embeddings=True),
    "opt": OPTConfig(hidden_size=128, ffn_dim=256, num_hidden_layers=2, num_attention_heads=4, vocab_size=512,
                     max_position_embeddings=256, word_embed_proj_dim=128),
    "gpt2": GPT2Config(n_embd=128, n_layer=2, n_head=4, vocab_size=512, n_positions=256),
}


@pytest.mark.parametrize("kind", list(CFGS))
@pytest.mark.parametrize("budget", [None, 6.5])
def test_roundtrip(kind, budget):
    torch.manual_seed(0)
    src = AutoModelForCausalLM.from_config(CFGS[kind]).eval()
    with tempfile.TemporaryDirectory() as d:
        src.save_pretrained(os.path.join(d, "src"), safe_serialization=True)
        s = quantize_model(os.path.join(d, "src"), os.path.join(d, "q"), bits=4, group_size=64, budget=budget,
                           n_samples=2048, device="cpu", log=lambda *a: None)
        qm = load_quantized(os.path.join(d, "q"), dtype=torch.float32)
        # reference: same architecture, weights replaced by the dequantized values
        ref = AutoModelForCausalLM.from_config(CFGS[kind]).eval()
        ref.load_state_dict(src.state_dict())
        sd = ref.state_dict()
        from safetensors.torch import load_file
        ck = load_file(os.path.join(d, "q", "model.safetensors"))
        qc = json.load(open(os.path.join(d, "q", "config.json")))["quantization_config"]
        from shapequant.qlinear import _bits_for
        n_q = 0
        for k in ck:
            if k.endswith(".qweight"):
                mod = k[:-8]; bits = _bits_for(mod, qc)
                Wd = core.unpack(ck[mod + ".qweight"], ck[mod + ".qzeros"], ck[mod + ".scales"], ck[mod + ".g_idx"], bits)
                key = next(c for c in (mod + ".weight", mod.split(".", 1)[1] + ".weight") if c in sd)
                sd[key] = Wd.T if kind == "gpt2" else Wd
                n_q += 1
        ref.load_state_dict(sd)
        ids = torch.randint(0, 512, (1, 40))
        with torch.no_grad():
            a, b, c = qm(ids).logits, ref(ids).logits, src(ids).logits
        assert n_q > 0
        assert torch.allclose(a, b, atol=2e-3), (a - b).abs().max()      # loader == dequantized reference
        assert (a - c).abs().max() < 5 * (c.std() + 1e-3)                  # close to the original model
        if budget is not None:
            assert "dynamic" in qc and s["bits"] == "mixed"
