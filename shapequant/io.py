"""Lazy weight reading (one tensor at a time) and sharded safetensors writing. Nothing loads the whole model."""
import os, json, glob, shutil
import torch


def resolve(path_or_repo, revision=None):
    """Local directory, or a Hugging Face repo id (downloads config, tokenizer and weight files)."""
    if os.path.isdir(path_or_repo):
        return path_or_repo
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.bin"],
                             ignore_patterns=["*onnx*", "*.msgpack", "*.h5", "*consolidated*", "original/*"])


class Reader:
    """Reads individual tensors from safetensors shards (or a small pytorch_model.bin) by name.
    Accepts names with or without a leading 'model.' / 'transformer.' prefix."""

    def __init__(self, path):
        self.path = path
        st = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        self.where, self._bin = {}, None
        if st:
            from safetensors import safe_open
            self._open = {}
            for f in st:
                with safe_open(f, framework="pt") as h:
                    for k in h.keys():
                        self.where[k] = f
        else:
            bins = sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))
            if not bins:
                raise FileNotFoundError(f"no *.safetensors or pytorch_model*.bin in {path}")
            self._bin = {}
            for f in bins:
                sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
                for k, v in sd.items():
                    self._bin[k] = v; self.where[k] = f

    def _key(self, name):
        if name in self.where:
            return name
        for cand in (name.split(".", 1)[1] if "." in name else name, "model." + name, "transformer." + name):
            if cand in self.where:
                return cand
        tail = name.split(".", 1)[1] if name.startswith(("model.", "transformer.")) else name
        for k in self.where:
            if k.endswith(tail) and (len(k) == len(tail) or k[-len(tail) - 1] == "."):
                return k
        return None

    def has(self, name):
        return self._key(name) is not None

    def get(self, name, device="cpu", dtype=torch.float32):
        k = self._key(name)
        if k is None:
            raise KeyError(name)
        if self._bin is not None:
            t = self._bin[k]
        else:
            from safetensors import safe_open
            with safe_open(self.where[k], framework="pt", device="cpu") as h:
                t = h.get_tensor(k)
        return t.to(device=device, dtype=dtype)

    def all_keys(self):
        return list(self.where)


class ShardWriter:
    """Accumulates tensors and writes model-0000x-of-0000y.safetensors shards plus the index json."""

    def __init__(self, out_dir, max_shard_bytes=4 * 2 ** 30):
        os.makedirs(out_dir, exist_ok=True)
        self.out, self.max, self.buf, self.size, self.shards, self.index = out_dir, max_shard_bytes, {}, 0, [], {}

    def add(self, name, t):
        t = t.detach().to("cpu").contiguous().clone()
        nb = t.numel() * t.element_size()
        if self.buf and self.size + nb > self.max:
            self.flush()
        self.buf[name] = t; self.size += nb

    def flush(self):
        if not self.buf:
            return
        from safetensors.torch import save_file
        fn = f"tmp-{len(self.shards):05d}.safetensors"
        save_file(self.buf, os.path.join(self.out, fn), metadata={"format": "pt"})
        self.shards.append((fn, list(self.buf)))
        self.buf, self.size = {}, 0

    def close(self):
        self.flush()
        n = len(self.shards); total = 0; wm = {}
        for i, (fn, names) in enumerate(self.shards):
            final = "model.safetensors" if n == 1 else f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            os.replace(os.path.join(self.out, fn), os.path.join(self.out, final))
            total += os.path.getsize(os.path.join(self.out, final))
            for k in names:
                wm[k] = final
        if n > 1:
            json.dump({"metadata": {"total_size": total}, "weight_map": wm},
                      open(os.path.join(self.out, "model.safetensors.index.json"), "w"), indent=2)
        return total


def copy_side_files(src, dst):
    for f in os.listdir(src):
        if f.startswith(("tokenizer", "special_tokens_map", "generation_config", "vocab", "merges")) or f.endswith((".model", ".tiktoken")):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
