import torch
from shapequant import core


def _ref_sequential(W, bits, group, H, dynamic=False):
    """Unblocked reference of the error-feedback recursion."""
    W = W.clone().float(); out, inn = W.shape; maxq = 2 ** bits - 1
    sc, z = core.group_params(W, bits, group)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for c in range(inn):
        g = c // group
        if dynamic and c % group == 0:
            s_, z_ = core.group_params(W[:, c:c + group], bits, group); sc[:, g], z[:, g] = s_[:, 0], z_[:, 0]
        q = torch.clamp(torch.round(W[:, c] / sc[:, g]) + z[:, g], 0, maxq); Q[:, c] = q
        err = (W[:, c] - (q - z[:, g]) * sc[:, g]) / U[c, c]
        W[:, c:] -= err[:, None] * U[c, c:][None, :]
    return Q


def test_identity_design_is_rtn():
    torch.manual_seed(0)
    W = torch.randn(64, 256)
    q0, s0, z0 = core.quantize(W, 4, 64)
    for dyn in (False, True):
        q1, s1, z1 = core.quantize(W, 4, 64, H=torch.eye(256), dynamic_groups=dyn)
        assert torch.equal(q0, q1) and torch.equal(s0, s1)


def test_blocked_equals_sequential():
    torch.manual_seed(1)
    W = torch.randn(48, 384); A = torch.randn(384, 384); H = A @ A.T / 384 + torch.eye(384)
    for dyn in (False, True):
        for blk in (32, 64, 128, 256):
            q, _, _ = core.quantize(W, 4, 128, H=H, block=blk, dynamic_groups=dyn)
            assert (q.float() - _ref_sequential(W, 4, 128, H, dyn)).abs().max() == 0, (dyn, blk)


def test_pack_roundtrip():
    torch.manual_seed(2)
    for bits in (4, 8):
        W = torch.randn(96, 256) * 0.05
        q, s, z = core.quantize(W, bits, 128)
        pk = core.pack(q, s, z, bits, 128)
        Wd = core.unpack(pk["qweight"], pk["qzeros"], pk["scales"], pk["g_idx"], bits)
        ref = core.dequantize(q, s.half().float(), z, 128)
        assert torch.allclose(Wd, ref, atol=0, rtol=0)
        assert pk["qweight"].dtype == torch.int32 and pk["qweight"].shape == (256 * bits // 32, 96)


def test_zero_points_v1_safe():
    W = torch.rand(32, 128)                       # all positive groups: zero-point must still be >= 1
    q, s, z = core.quantize(W, 4, 64)
    assert z.min() >= 1
    err = (W - core.dequantize(q, s, z, 64)).abs().max() / W.abs().max()
    assert err < 0.05


def test_shaping_reduces_weighted_error():
    torch.manual_seed(3)
    W = torch.randn(128, 256); A = torch.randn(256, 32); H = torch.eye(256) + A @ A.T
    def cost(q, s, z):
        R = W - core.dequantize(q, s, z, 64); return float(torch.trace(R @ H @ R.T))
    assert cost(*core.quantize(W, 4, 64, H=H)) < cost(*core.quantize(W, 4, 64))
