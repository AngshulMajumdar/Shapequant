"""Model-implied input second moments, built from the weights only and streamed layer by layer.

Residual stream:  Sig_0 = E_{t~pi}[e_t e_t^T] (+ positional embeddings), pi = Zipf prior on token id
                  (BPE ids are assigned in merge order, i.e. by frequency); then Sig += W^T W for every writer
                  (attention output and MLP output) in order.
Norm outputs:     LayerNorm: C = P Sig P / (tr/n), cov = g C g, mean = beta, M2 = cov + beta beta^T.
                  RMSNorm:   C = Sig / (tr/n),     M2 = g C g, mean = g m / s, cov = M2 - mean mean^T.
MLP 2nd input:    Monte Carlo of the block's own nonlinearity on x ~ N(mean, cov).
Attention output: E[o o^T] = mu_o mu_o^T + kappa * blockdiag_heads(C_v of the head's KV group).
Every S is normalised to trace n. Design H = I + lam * S.
"""
import math
import torch

F64 = torch.float64


class ResidualModel:
    def __init__(self, arch, reader, device, ctx=1024, zipf_shift=10.0, chunk=8192):
        self.a, self.r, self.dev = arch, reader, device
        E = reader.get(arch.embed, device=device, dtype=torch.float32)
        V = E.shape[0]
        pi = 1.0 / (torch.arange(V, device=device, dtype=F64) + zipf_shift); pi /= pi.sum()
        Sig = torch.zeros((arch.d, arch.d), dtype=F64, device=device); m = torch.zeros(arch.d, dtype=F64, device=device)
        for s in range(0, V, chunk):
            e = E[s:s + chunk].to(F64); w = pi[s:s + chunk]
            Sig += (e * w[:, None]).T @ e; m += w @ e
        del E
        if arch.pos_embed is not None and reader.has(arch.pos_embed):
            P = reader.get(arch.pos_embed, device=device, dtype=F64)
            T = min(ctx, P.shape[0] - arch.pos_offset)
            P = P[arch.pos_offset: arch.pos_offset + T]
            pm = P.mean(0)
            Sig += P.T @ P / T + torch.outer(m, pm) + torch.outer(pm, m); m = m + pm
        self.Sig, self.m = Sig, m

    def add_writer(self, W_out_in):
        """W in torch layout (out=d, in); the writer adds W h with isotropic h."""
        W = W_out_in.to(device=self.dev, dtype=F64)
        self.Sig += W @ W.T

    def norm_moments(self, L, which):
        a = self.a; n = a.d
        gname, bname = a.norm_names(L, which)
        g = self.r.get(gname, device=self.dev, dtype=F64)
        if a.norm == "ln":
            Sg = self.Sig - self.Sig.mean(0, keepdim=True)
            Sg = Sg - Sg.mean(1, keepdim=True)                    # P Sig P
            C = Sg / (torch.trace(Sg) / n)
            cov = g[:, None] * C * g[None, :]
            b = self.r.get(bname, device=self.dev, dtype=F64) if bname and self.r.has(bname) else torch.zeros(n, dtype=F64, device=self.dev)
            return b, cov, cov + torch.outer(b, b)
        s2 = torch.trace(self.Sig) / n
        M2 = g[:, None] * (self.Sig / s2) * g[None, :]
        mu = g * self.m / torch.sqrt(s2)
        w, Q = torch.linalg.eigh(M2 - torch.outer(mu, mu))
        cov = (Q * w.clamp(min=0)) @ Q.T
        return mu, cov, M2


def normalise(M):
    return M.mul_(M.shape[0] / torch.trace(M))


def S_norm(moments):
    return normalise(moments[2].clone())


def S_act(arch, moments, W_up, b_up, W_gate, n_samples=16384, batch=2048, seed=0, device="cpu"):
    """W_up/W_gate: torch layout (inter, d). Returns (inter, inter) float32."""
    mu, cov, _ = moments
    w, Q = torch.linalg.eigh(cov)
    R = (Q * w.clamp(min=0).sqrt()).to(torch.float32)
    mu = mu.to(torch.float32)
    Wu = W_up.to(device=device, dtype=torch.float32).T
    Wg = None if W_gate is None else W_gate.to(device=device, dtype=torch.float32).T
    bu = None if b_up is None else b_up.to(device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(seed)
    M = torch.zeros((Wu.shape[1], Wu.shape[1]), dtype=torch.float32, device=device); done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        x = mu + torch.randn((b, R.shape[0]), generator=gen, device=device) @ R.T
        u = x @ Wu
        if bu is not None:
            u = u + bu
        if arch.act == "swiglu":
            z = x @ Wg; h = z * torch.sigmoid(z) * u
        elif arch.act == "relu":
            h = torch.relu(u)
        elif arch.act == "gelu_new":
            h = 0.5 * u * (1 + torch.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u ** 3)))
        elif arch.act == "gelu":
            h = torch.nn.functional.gelu(u)
        else:
            raise NotImplementedError(arch.act)
        M += h.T @ h; done += b
    return normalise(M.div_(done))


def S_attn(arch, moments, W_v, b_v, kappa=1.0):
    """W_v: torch layout (n_kv*head_dim, d). Returns (n_head*head_dim)^2 float64."""
    mu, cov, _ = moments
    Wv = W_v.to(device=cov.device, dtype=F64)
    mv = Wv @ mu + (0 if b_v is None else b_v.to(device=cov.device, dtype=F64))
    Cv = Wv @ cov @ Wv.T
    H, dh = arch.n_head, arch.head_dim; per = H // arch.n_kv
    n = H * dh
    mo = torch.zeros(n, dtype=F64, device=cov.device); M = torch.zeros((n, n), dtype=F64, device=cov.device)
    for h in range(H):
        gk = h // per; s, t = slice(h * dh, (h + 1) * dh), slice(gk * dh, (gk + 1) * dh)
        mo[s] = mv[t]; M[s, s] = kappa * Cv[t, t]
    return normalise(M + torch.outer(mo, mo))
