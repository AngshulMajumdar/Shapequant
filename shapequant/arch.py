"""Architecture adapters: tensor names and structure for the supported Hugging Face causal LMs.

Supported model_type: llama, mistral, qwen2 (incl. Llama-2/3/3.1/3.3-70B, Qwen2.5-72B), opt (pre-LN), gpt2.
A linear module is described by (module_name, input_key, role):
  input_key in {"ln1", "attn_out", "ln2", "act"}  - which activation it reads
  role      in {"q","k","v","qkv","o","gate","up","down"}
Weights are returned in torch Linear layout (out, in); GPT-2 Conv1D weights are transposed on read.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Linear:
    name: str              # module path, e.g. "model.layers.3.self_attn.q_proj"
    input_key: str
    role: str
    conv1d: bool = False   # stored (in, out)


@dataclass
class Arch:
    model_type: str
    n_layer: int
    d: int
    n_head: int
    n_kv: int
    head_dim: int
    inter: int
    norm: str              # "rms" | "ln"
    norm_eps: float
    act: str               # "swiglu" | "relu" | "gelu_new" | "gelu"
    embed: str
    pos_embed: Optional[str]
    pos_offset: int
    final_norm: str
    lm_head: Optional[str]
    vocab: int
    max_pos: int
    layers: List[List[Linear]] = field(default_factory=list)
    norms: List[Tuple[str, str]] = field(default_factory=list)       # per layer: (ln1 prefix, ln2 prefix)

    def norm_names(self, L, which):
        p = self.norms[L][0 if which == "ln1" else 1]
        return p + ".weight", (p + ".bias" if self.norm == "ln" else None)


def from_config(cfg: dict) -> Arch:
    mt = cfg.get("model_type")
    if mt in ("llama", "mistral", "qwen2"):
        d, H = cfg["hidden_size"], cfg["num_attention_heads"]
        act = cfg.get("hidden_act", "silu")
        if act != "silu":
            raise NotImplementedError(f"{mt} with hidden_act={act}")
        a = Arch(mt, cfg["num_hidden_layers"], d, H, cfg.get("num_key_value_heads", H), cfg.get("head_dim") or d // H,
                 cfg["intermediate_size"], "rms", cfg.get("rms_norm_eps", 1e-6), "swiglu",
                 "model.embed_tokens.weight", None, 0, "model.norm.weight",
                 None if cfg.get("tie_word_embeddings", False) else "lm_head.weight",
                 cfg["vocab_size"], cfg.get("max_position_embeddings", 4096))
        for L in range(a.n_layer):
            p = f"model.layers.{L}."
            a.layers.append([Linear(p + "self_attn.q_proj", "ln1", "q"), Linear(p + "self_attn.k_proj", "ln1", "k"),
                             Linear(p + "self_attn.v_proj", "ln1", "v"), Linear(p + "self_attn.o_proj", "attn_out", "o"),
                             Linear(p + "mlp.gate_proj", "ln2", "gate"), Linear(p + "mlp.up_proj", "ln2", "up"),
                             Linear(p + "mlp.down_proj", "act", "down")])
            a.norms.append((p + "input_layernorm", p + "post_attention_layernorm"))
        return a
    if mt == "opt":
        if not cfg.get("do_layer_norm_before", True):
            raise NotImplementedError("post-LN OPT (opt-350m) is not supported")
        if cfg.get("word_embed_proj_dim", cfg["hidden_size"]) != cfg["hidden_size"]:
            raise NotImplementedError("OPT with project_in/out is not supported")
        d, H = cfg["hidden_size"], cfg["num_attention_heads"]
        a = Arch(mt, cfg["num_hidden_layers"], d, H, H, d // H, cfg["ffn_dim"], "ln", 1e-5,
                 {"relu": "relu"}.get(cfg.get("activation_function", "relu"), cfg.get("activation_function")),
                 "model.decoder.embed_tokens.weight", "model.decoder.embed_positions.weight", 2,
                 "model.decoder.final_layer_norm.weight", None if cfg.get("tie_word_embeddings", True) else "lm_head.weight",
                 cfg["vocab_size"], cfg.get("max_position_embeddings", 2048))
        for L in range(a.n_layer):
            p = f"model.decoder.layers.{L}."
            a.layers.append([Linear(p + "self_attn.q_proj", "ln1", "q"), Linear(p + "self_attn.k_proj", "ln1", "k"),
                             Linear(p + "self_attn.v_proj", "ln1", "v"), Linear(p + "self_attn.out_proj", "attn_out", "o"),
                             Linear(p + "fc1", "ln2", "up"), Linear(p + "fc2", "act", "down")])
            a.norms.append((p + "self_attn_layer_norm", p + "final_layer_norm"))
        return a
    if mt == "gpt2":
        d, H = cfg["n_embd"], cfg["n_head"]
        a = Arch(mt, cfg["n_layer"], d, H, H, d // H, cfg.get("n_inner") or 4 * d, "ln", cfg.get("layer_norm_epsilon", 1e-5),
                 {"gelu_new": "gelu_new"}.get(cfg.get("activation_function", "gelu_new"), cfg.get("activation_function")),
                 "transformer.wte.weight", "transformer.wpe.weight", 0, "transformer.ln_f.weight", None,
                 cfg["vocab_size"], cfg.get("n_positions", 1024))
        for L in range(a.n_layer):
            p = f"transformer.h.{L}."
            a.layers.append([Linear(p + "attn.c_attn", "ln1", "qkv", True), Linear(p + "attn.c_proj", "attn_out", "o", True),
                             Linear(p + "mlp.c_fc", "ln2", "up", True), Linear(p + "mlp.c_proj", "act", "down", True)])
            a.norms.append((p + "ln_1", p + "ln_2"))
        return a
    raise NotImplementedError(f"model_type={mt!r} is not supported (supported: llama, mistral, qwen2, opt, gpt2)")
