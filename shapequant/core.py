"""Quantizer core (torch, CPU or GPU).

Grid: standard asymmetric GPTQ grid per (output row, input group): range includes 0,
scale = (max - min) / (2^b - 1), integer zero-point z = round(-min / scale), q = clamp(round(w/scale) + z, 0, 2^b - 1),
w_hat = (q - z) * scale. Grids are fixed from the original weights ("static groups").
Zero-points are kept >= 1 so the checkpoint can use the GPTQ v1 storage convention (stores z - 1).

Rounding: sequential error feedback that minimises tr(R H R^T) for a design matrix H (in x in), i.e. the
OBQ/GPTQ recursion with lazy block updates. H = None gives round-to-nearest. With H = I it is identical to RTN.
"""
import torch


def group_params(W, bits, group):
    """W: (out, in). Returns scale, zero of shape (out, in // group)."""
    out, inn = W.shape
    assert inn % group == 0, f"input dim {inn} not divisible by group size {group}"
    maxq = 2 ** bits - 1
    Wg = W.reshape(out, inn // group, group)
    xmin = torch.clamp(Wg.amin(-1), max=0.0)
    xmax = torch.clamp(Wg.amax(-1), min=0.0)
    both0 = (xmin == 0) & (xmax == 0)
    xmin, xmax = torch.where(both0, -torch.ones_like(xmin), xmin), torch.where(both0, torch.ones_like(xmax), xmax)
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    # keep z >= 1 (GPTQ v1 storage): if the group has no negative part, use z = 1 and rescale so max still fits
    low = zero < 1
    scale = torch.where(low, xmax / (maxq - 1), scale)
    zero = torch.where(low, torch.ones_like(zero), zero)
    zero = torch.clamp(zero, 1, maxq)
    return scale, zero


def _q(w, scale, zero, maxq):
    return torch.clamp(torch.round(w / scale) + zero, 0, maxq)


def quantize(W, bits, group, H=None, block=128, dynamic_groups=True):
    """W: (out, in) float tensor (torch Linear layout). Returns (q uint8 (out, in), scale (out, G), zero (out, G)).
    H: (in, in) positive-definite design matrix, or None for round-to-nearest.
    dynamic_groups: each group's grid is fitted to the error-fed (updated) weights when the group starts,
    as in standard GPTQ, so feedback never pushes weights outside their grid. False = grids from the original W."""
    W = W.to(torch.float32)
    out, inn = W.shape
    maxq = 2 ** bits - 1
    scale, zero = group_params(W, bits, group)
    gid = torch.arange(inn, device=W.device) // group
    if H is None:
        Q = _q(W, scale[:, gid], zero[:, gid], maxq)
        return Q.to(torch.uint8), scale, zero
    W = W.clone()
    H = H.to(device=W.device, dtype=torch.float32)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L); del L
    U = torch.linalg.cholesky(Hinv, upper=True); del Hinv
    Q = torch.zeros((out, inn), dtype=torch.uint8, device=W.device)
    for i1 in range(0, inn, block):
        i2 = min(i1 + block, inn); n = i2 - i1
        W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); U1 = U[i1:i2, i1:i2]
        for i in range(n):
            c = i1 + i; g = c // group
            if dynamic_groups and c % group == 0:
                seg = W1[:, i:i + group] if i + group <= n else torch.cat([W1[:, i:], W[:, i2:c + group]], 1)
                sg, zg = group_params(seg, bits, group)
                scale[:, g], zero[:, g] = sg[:, 0], zg[:, 0]
            w = W1[:, i]
            q = _q(w, scale[:, g], zero[:, g], maxq)
            Q[:, c] = q.to(torch.uint8)
            err = (w - (q - zero[:, g]) * scale[:, g]) / U1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * U1[i, i:].unsqueeze(0)
            E1[:, i] = err
        if i2 < inn:
            W[:, i2:] -= E1 @ U[i1:i2, i2:]
    return Q, scale, zero


def dequantize(Q, scale, zero, group):
    gid = torch.arange(Q.shape[1], device=Q.device) // group
    return (Q.to(torch.float32) - zero[:, gid]) * scale[:, gid]


# --------------------------------------------------------------- GPTQ v1 packing (AutoGPTQ / GPTQModel "gptq" format)
def pack(Q, scale, zero, bits, group):
    """Returns dict(qweight int32 (in*bits/32, out), qzeros int32 (G, out*bits/32), scales fp16 (G, out), g_idx int32 (in,))."""
    assert bits in (2, 4, 8)
    per = 32 // bits
    out, inn = Q.shape
    q = Q.t().contiguous().to(torch.int64)                      # (in, out)
    assert inn % per == 0 and out % per == 0
    qw = torch.zeros((inn // per, out), dtype=torch.int64, device=Q.device)
    for k in range(per):
        qw |= q[k::per, :] << (bits * k)
    z = (zero.t().contiguous().to(torch.int64) - 1)             # v1 convention: store z - 1
    qz = torch.zeros((z.shape[0], out // per), dtype=torch.int64, device=Q.device)
    for k in range(per):
        qz |= z[:, k::per] << (bits * k)
    to32 = lambda t: (t & 0xFFFFFFFF).to(torch.int64).where(t < 2 ** 31, t - 2 ** 32).to(torch.int32)
    return {"qweight": to32(qw), "qzeros": to32(qz), "scales": scale.t().contiguous().to(torch.float16),
            "g_idx": (torch.arange(inn, device=Q.device) // group).to(torch.int32)}


def unpack(qweight, qzeros, scales, g_idx, bits):
    """Inverse of pack: dequantized weight in torch Linear layout (out, in), float32."""
    per = 32 // bits; mask = 2 ** bits - 1
    qw = qweight.to(torch.int64) & 0xFFFFFFFF
    inn = qw.shape[0] * per
    q = torch.empty((inn, qw.shape[1]), dtype=torch.int64, device=qw.device)
    for k in range(per):
        q[k::per] = (qw >> (bits * k)) & mask
    qz = qzeros.to(torch.int64) & 0xFFFFFFFF
    z = torch.empty((qz.shape[0], qz.shape[1] * per), dtype=torch.int64, device=qz.device)
    for k in range(per):
        z[:, k::per] = ((qz >> (bits * k)) & mask) + 1
    g = g_idx.to(torch.int64)
    W = (q.to(torch.float32) - z[g].to(torch.float32)) * scales[g].to(torch.float32)   # (in, out)
    return W.t().contiguous()
