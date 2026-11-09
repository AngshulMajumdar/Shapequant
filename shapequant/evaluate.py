"""Perplexity on fixed windows (same protocol as the paper experiments)."""
import math
import torch


def windows_from_text(tokenizer, text, seq_len=1024, n=None, offset=0):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    k = len(ids) // seq_len
    W = ids[: k * seq_len].view(k, seq_len)[offset:]
    return W if n is None else W[:n]


def wikitext2_windows(tokenizer, seq_len=1024, n=None, path=None):
    if path is not None:
        text = open(path).read()
    else:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    return windows_from_text(tokenizer, text, seq_len, n)


@torch.no_grad()
def perplexity(model, windows, bos_token_id=None, device=None):
    """windows: LongTensor (n, L). If bos_token_id is set it is prepended to every window and all L tokens are scored."""
    device = device or next(model.parameters()).device
    nll, cnt = 0.0, 0
    for w in windows:
        ids = w if bos_token_id is None else torch.cat([torch.tensor([bos_token_id]), w])
        ids = ids.to(device)[None]
        logits = model(ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        nll += float(torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")); cnt += len(tgt)
    return math.exp(nll / cnt)
