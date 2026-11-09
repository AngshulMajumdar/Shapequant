"""Real-model check: quantize -> load -> WikiText-2 perplexity, for RTN and shaped, uniform INT4 and mixed.
usage: python tests/real_models.py <hf_model_dir_or_id> <out_dir> [--group 128] [--budget 4.75] [--windows 32] [--bos ID] [--text wt2_test.txt]"""
import argparse, json, os, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from shapequant import quantize_model, load_quantized, perplexity, wikitext2_windows

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("out"); p.add_argument("--group", type=int, default=128)
p.add_argument("--budget", type=float, default=4.75); p.add_argument("--windows", type=int, default=32)
p.add_argument("--bos", type=int, default=None); p.add_argument("--text", default=None); p.add_argument("--device", default="cpu")
a = p.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
W = wikitext2_windows(tok, 1024, a.windows, a.text)
R = {"dense": perplexity(AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval().to(a.device), W, a.bos)}
for tag, kw in (("rtn_int4", dict(bits=4, method="rtn")), ("shaped_int4", dict(bits=4)),
                ("rtn_mixed", dict(budget=a.budget, method="rtn")), ("shaped_mixed", dict(budget=a.budget))):
    t = time.time(); d = os.path.join(a.out, tag)
    s = quantize_model(a.model, d, group_size=a.group, device=a.device, log=lambda *x: None, **kw)
    R[tag] = {"ppl": perplexity(load_quantized(d, dtype=torch.float32, device=a.device), W, a.bos),
              "minutes": (time.time() - t) / 60, "checkpoint_MB": s["checkpoint_bytes"] / 2 ** 20}
    print(tag, R[tag], flush=True)
json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1)
