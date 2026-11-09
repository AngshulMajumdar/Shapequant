import argparse, json, sys


def main(argv=None):
    p = argparse.ArgumentParser("shapequant", description="Data-free, training-free quantization of Hugging Face causal LMs")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize", help="quantize a model (hub id or local dir) to a GPTQ-format checkpoint")
    q.add_argument("model"); q.add_argument("out")
    q.add_argument("--bits", type=int, default=4, choices=[4, 8])
    q.add_argument("--group-size", type=int, default=128)
    q.add_argument("--method", default="shaped", choices=["shaped", "rtn"])
    q.add_argument("--budget", type=float, default=None, help="average bits/weight for mixed INT4/INT8/FP16 (e.g. 4.75)")
    q.add_argument("--lam", type=float, default=1.0); q.add_argument("--kappa", type=float, default=1.0)
    q.add_argument("--samples", type=int, default=16384); q.add_argument("--device", default=None)
    q.add_argument("--revision", default=None); q.add_argument("--shard-gb", type=float, default=4.0)
    q.add_argument("--static-groups", action="store_true", help="fit group grids to the original weights instead of the error-fed ones")
    e = sub.add_parser("eval", help="perplexity of a quantized checkpoint on WikiText-2 (or a text file)")
    e.add_argument("ckpt"); e.add_argument("--text", default=None); e.add_argument("--seq-len", type=int, default=1024)
    e.add_argument("--windows", type=int, default=32); e.add_argument("--device", default="cpu"); e.add_argument("--bos", action="store_true")
    m = sub.add_parser("memory", help="peak-memory estimate from a config (hub id, dir or config.json)")
    m.add_argument("model"); m.add_argument("--method", default="shaped", choices=["shaped", "rtn"])
    a = p.parse_args(argv)
    if a.cmd == "quantize":
        from .quantize import quantize_model
        s = quantize_model(a.model, a.out, bits=a.bits, group_size=a.group_size, method=a.method, budget=a.budget, lam=a.lam,
                           kappa=a.kappa, n_samples=a.samples, device=a.device, revision=a.revision, max_shard_size_gb=a.shard_gb,
                           dynamic_groups=not a.static_groups)
        print(json.dumps({k: v for k, v in s.items() if k != "bits_per_module"}, indent=2))
    elif a.cmd == "eval":
        from transformers import AutoTokenizer
        from .qlinear import load_quantized
        from .evaluate import wikitext2_windows, perplexity
        tok = AutoTokenizer.from_pretrained(a.ckpt)
        model = load_quantized(a.ckpt, device=a.device)
        W = wikitext2_windows(tok, a.seq_len, a.windows, a.text)
        print(f"perplexity {perplexity(model, W, tok.bos_token_id if a.bos else None):.3f} on {W.numel()} tokens")
    elif a.cmd == "memory":
        import os
        from .memory import estimate
        src = a.model
        if not os.path.exists(src):
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(a.model, "config.json")
        print(json.dumps(estimate(src, method=a.method), indent=2))


if __name__ == "__main__":
    sys.exit(main())
