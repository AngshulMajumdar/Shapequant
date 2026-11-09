"""Rebuilds the eight documentation figures (docs/images/*.png) from tools/figure_data.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "images")
D = json.load(open(os.path.join(HERE, "figure_data.json")))
os.makedirs(OUT, exist_ok=True)

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SHAPED, RTN, TRUE = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
SOFT = {"blue": "#cde2fb", "orange": "#fbe0d4", "aqua": "#cdeee1", "gray": "#efeeea"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11.5, "axes.titlelocation": "left", "legend.frameon": False})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc, ec=GRID, fs=10, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight="bold" if bold else None, wrap=True)


def arrow(ax, p, q, color=INK2, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw, linestyle=ls, shrinkA=2, shrinkB=2))


# 1 ---------------------------------------------------------------- pipeline
fig, ax = plt.subplots(figsize=(12, 4.4)); ax.set_axis_off(); ax.set_xlim(-0.01, 1.02); ax.set_ylim(0, 1)
ax.set_title("Figure 1. One decoder layer at a time: read, model the inputs, round, pack, write", pad=10)
box(ax, 0.01, 0.62, 0.15, 0.22, "safetensors shards\n(original model)", SOFT["gray"])
box(ax, 0.22, 0.62, 0.17, 0.22, "layer ℓ weights\nq k v o · gate up down", SOFT["gray"])
box(ax, 0.45, 0.62, 0.22, 0.22, "input model for layer ℓ\nS_ln1  S_attn  S_ln2  S_act", SOFT["blue"], ec=SHAPED)
box(ax, 0.73, 0.62, 0.12, 0.22, "shaped\nrounding\nH = I + λS", SOFT["blue"], ec=SHAPED)
box(ax, 0.88, 0.62, 0.12, 0.22, "GPTQ pack\nINT4 / INT8", SOFT["gray"])
box(ax, 0.45, 0.16, 0.22, 0.2, "residual-stream state\nΣ_res  (d × d), carried to ℓ+1", SOFT["blue"], ec=SHAPED)
box(ax, 0.80, 0.16, 0.2, 0.2, "output shards\n+ config.json", SOFT["gray"])
box(ax, 0.01, 0.16, 0.33, 0.2, "embeddings, Zipf prior on token id\n(+ positions)  →  Σ_res at layer 0", SOFT["gray"], fs=9.5)
for p, q in (((0.16, 0.73), (0.22, 0.73)), ((0.39, 0.73), (0.45, 0.73)), ((0.67, 0.73), (0.73, 0.73)), ((0.85, 0.73), (0.88, 0.73)),
             ((0.935, 0.62), (0.90, 0.36)), ((0.34, 0.26), (0.45, 0.26))):
    arrow(ax, p, q)
arrow(ax, (0.53, 0.36), (0.53, 0.62), color=SHAPED); arrow(ax, (0.60, 0.62), (0.60, 0.36), color=SHAPED)
ax.text(0.505, 0.49, "read Σ_res", fontsize=9, color=INK2, ha="right"); ax.text(0.615, 0.49, "+ WᵀW of\nwriters o, down", fontsize=9, color=INK2)
ax.text(0.01, 0.02, "Peak memory: one layer + a few (in × in) matrices. No calibration text, no gradients.", fontsize=9.5, color=INK2)
save(fig, "fig1_pipeline.png")

# 2 ---------------------------------------------------------------- where the error goes (2-D)
th = np.deg2rad(32); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Sig = Rm @ np.diag([9.0, 0.35]) @ Rm.T
rng = np.random.default_rng(0); X = rng.multivariate_normal([0, 0], Sig, 900)
C_rtn = np.eye(2) * 0.55
lam, V = np.linalg.eigh(Sig); C_sh = V @ np.diag(0.55 * 2 * (1 / np.sqrt(lam)) / (1 / np.sqrt(lam)).sum()) @ V.T   # same total power
C_sh = C_sh * (np.trace(C_rtn) / np.trace(C_sh))
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True, sharey=True)
fig.suptitle("Figure 2. Same rounding-error power, different direction: output error is tr(Rᵀ Σ R)", x=0.01, ha="left", weight="bold", fontsize=11.5)
for ax, C, nm, col in ((axs[0], C_rtn, "round to nearest: error is white", RTN), (axs[1], C_sh, "shaped: error moved where inputs are small", SHAPED)):
    ax.scatter(X[:, 0], X[:, 1], s=6, color=MUTED, alpha=0.35, lw=0, label="inputs x ~ N(0, Σ)")
    w, U = np.linalg.eigh(C)
    for k in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=np.degrees(np.arctan2(U[1, 1], U[0, 1])),
                             fc="none", ec=col, lw=2))
    ax.set_title(nm, fontsize=10.5); ax.set_aspect("equal"); ax.set_xlim(-8, 8); ax.set_ylim(-5.5, 5.5)
    ax.text(-7.6, -5.0, f"output MSE ∝ tr(ΣC) = {np.trace(Sig @ C):.2f}", fontsize=10, color=INK)
    ax.text(-7.6, 4.5, "grey: inputs   ring: error covariance C", fontsize=9, color=INK2)
save(fig, "fig2_where_error_goes.png")

# 3 ---------------------------------------------------------------- residual-stream model
fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_axis_off(); ax.set_xlim(-0.01, 1.0); ax.set_ylim(0, 1)
ax.set_title("Figure 3. The model supplies its own input statistics", pad=10)
box(ax, 0.0, 0.40, 0.17, 0.22, "Σ₀ = Σ_t π_t e_t e_tᵀ\nπ_t ∝ 1/(t + 10)", SOFT["gray"], fs=10)
xs = [0.215, 0.48, 0.745]
labs = [("LayerNorm / RMSNorm", "S_ln1 = normalised\nsecond moment", "reads: q, k, v"),
        ("attention output", "S_attn = μμᵀ + κ·blockdiag(C_v)", "reads: o_proj"),
        ("MLP input → nonlinearity", "S_ln2;  S_act = E[φ(x)φ(x)ᵀ]\nby Monte Carlo, x ~ N(μ, C)", "reads: gate, up · down")]
for x, (t1, t2, t3) in zip(xs, labs):
    box(ax, x, 0.58, 0.225, 0.16, t1, SOFT["blue"], ec=SHAPED, bold=True, fs=10)
    box(ax, x, 0.30, 0.225, 0.2, t2, SURF, fs=9)
    ax.text(x + 0.1125, 0.22, t3, ha="center", fontsize=9, color=INK2)
    arrow(ax, (x + 0.1125, 0.58), (x + 0.1125, 0.50), color=SHAPED)
ax.add_patch(FancyArrowPatch((0.16, 0.86), (0.97, 0.86), arrowstyle="-|>", mutation_scale=16, color=INK, lw=2))
ax.text(0.17, 0.90, "residual stream:  Σ_res ← Σ_res + W_oᵀW_o  (after attention),  Σ_res ← Σ_res + W_downᵀW_down  (after MLP)", fontsize=9.5)
arrow(ax, (0.17, 0.51), (0.22, 0.66))
ax.text(0.01, 0.05, "LayerNorm: C = PΣP / (tr/n), cov = g∘C∘g, mean = β     RMSNorm: C = Σ/(tr/n), mean = g∘m/s     Every S is scaled to trace n.",
        fontsize=9.5, color=INK2)
save(fig, "fig3_input_model.png")

# 4 ---------------------------------------------------------------- error-feedback recursion
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
fig.suptitle("Figure 4. Sequential rounding with error feedback minimises tr(R H Rᵀ)", x=0.01, ha="left", weight="bold", fontsize=11.5)
ax = axs[0]; ax.set_axis_off(); n = 12; i = 4
A = np.zeros((6, n)); rng = np.random.default_rng(3)
for c in range(n):
    for r in range(6):
        col = SOFT["gray"] if c > i else ("#9ec5f4" if c < i else SHAPED)
        ax.add_patch(plt.Rectangle((c, 5 - r), 0.92, 0.92, fc=col, ec=SURF, lw=2))
for c in range(i + 1, n):
    w = np.exp(-(c - i) / 3)
    ax.add_patch(FancyArrowPatch((i + 0.5, 6.25), (c + 0.5, 6.25), connectionstyle=f"arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=9, color=RTN, lw=0.6 + 2 * w, alpha=0.4 + 0.6 * w))
ax.text(i + 0.46, -0.55, "column i", ha="center", fontsize=9.5, color=SHAPED, weight="bold")
ax.text(1.8, -0.55, "rounded", ha="center", fontsize=9.5, color=INK2); ax.text(9, -0.55, "still real-valued", ha="center", fontsize=9.5, color=INK2)
ax.text(8.0, 8.3, "error eᵢ pushed to later columns", color=RTN, fontsize=9.5, ha="center")
ax.set_xlim(-0.3, n + 0.3); ax.set_ylim(-1.0, 8.9)
ax = axs[1]; ax.set_axis_off()
txt = ("H = I + λS,   H⁻¹ = UᵀU   (U upper-triangular)\n\n"
       "for each input column i:\n"
       "    qᵢ = round_grid(wᵢ)\n"
       "    eᵢ = (wᵢ − qᵢ) / U_ii\n"
       "    w_j ← w_j − eᵢ U_ij      for all j > i\n\n"
       "H = I  ⇒  U = I  ⇒  exactly round-to-nearest\n"
       "lazy updates in blocks of 128 columns;\n"
       "grid of each group fitted when the group starts")
ax.text(0.0, 0.5, txt, family="DejaVu Sans Mono", fontsize=10, va="center", color=INK)
save(fig, "fig4_error_feedback.png")

# 5 ---------------------------------------------------------------- spectra (real)
sp = D["spectrum"]; et, eh = np.array(sp["true"]), np.array(sp["data_free"])
fig, ax = plt.subplots(figsize=(9.5, 4.4))
k = np.arange(1, len(et) + 1)
ax.plot(k, et, color=TRUE, lw=2, label="true Σ (held-out text, evaluation only)")
ax.plot(k, eh, color=SHAPED, lw=2, label="data-free S (weights only)")
ax.axhline(1.0, color=RTN, lw=2, ls="--", label="round-to-nearest assumes Σ ∝ I")
ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("eigenvalue (trace-normalised to n)")
ax.set_title(f"Figure 5. Input spectrum, OPT-125m layer {sp['layer']}, {sp['input']}")
ax.text(k[2], et[2] * 1.6, "true Σ", color=INK, fontsize=9.5); ax.text(k[2], eh[2] * 0.45, "data-free S", color=INK, fontsize=9.5)
ax.text(200, 1.25, "identity", color=INK, fontsize=9.5)
ax.text(1.05, 2e-4, f"AM/GM of true spectrum = {sp['flatness_bound_true']:.1f}  (upper bound on any error-feedback gain)\n"
        f"cosine similarity ⟨S, Σ⟩ = {sp['cos_sim']:.2f}", fontsize=9.5, color=INK2)
ax.legend(loc="upper right", fontsize=9.5)
save(fig, "fig5_spectrum.png")

# 6 ---------------------------------------------------------------- per-layer error ratio (real)
lr = D["layer_ratios"]; mods = lr["modules"]
fig, axs = plt.subplots(1, 3, figsize=(11, 4.3), sharey=True)
fig.suptitle("Figure 6. Exact output error relative to round-to-nearest, OPT-125m (INT4, group 128); lower is better",
             x=0.01, ha="left", weight="bold", fontsize=11.5, y=1.02)
x = np.arange(len(mods)); wdt = 0.38
for j, (ax, L) in enumerate(zip(axs, lr["layers"])):
    a, b = np.array(lr["data_free"][j]), np.array(lr["true_cov"][j])
    ax.bar(x - wdt / 2, a, wdt, color=SHAPED, ec=SURF, lw=2)
    ax.bar(x + wdt / 2, b, wdt, color=TRUE, ec=SURF, lw=2)
    ax.axhline(1.0, color=RTN, lw=2, ls="--")
    for xi, v in zip(x, a):
        ax.text(xi - wdt / 2, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=7, color="white", rotation=90)
    for xi, v in zip(x, b):
        ax.text(xi + wdt / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2, rotation=90)
    ax.set_xticks(x, mods); ax.set_title(f"layer {L}", fontsize=10.5); ax.set_ylim(0, 1.08); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("tr(RᵀΣR) / tr(R_rtnᵀΣR_rtn)")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Patch(color=SHAPED, label="data-free H (shapequant)"), Patch(color=TRUE, label="true covariance (reference, uses text)"),
                    Line2D([], [], color=RTN, lw=2, ls="--", label="round to nearest = 1")], loc="upper left", ncol=3,
           bbox_to_anchor=(0.01, 0.98), fontsize=9.5)
fig.subplots_adjust(top=0.80)
save(fig, "fig6_layer_error.png")

# 7 ---------------------------------------------------------------- perplexity (real)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
R = D["results"]; names = {"gpt2": "GPT-2 124M", "opt125m": "OPT-125m", "smollm2": "SmolLM2-135M (Llama)"}
fig, axs = plt.subplots(1, 3, figsize=(11, 4.4))
fig.suptitle("Figure 7. WikiText-2 perplexity (32 × 1024 tokens); lower is better. Each panel has its own scale.",
             x=0.01, ha="left", weight="bold", fontsize=11.5, y=1.02)
for ax, key in zip(axs, ("gpt2", "opt125m", "smollm2")):
    r = R[key]; xs = np.array([0, 1, 2.6, 3.6])
    vals = [r["rtn_int4"], r["shaped_int4"], r["rtn_mixed"], r["shaped_mixed"]]
    ax.bar(xs, vals, 0.85, color=[RTN, SHAPED, RTN, SHAPED], ec=SURF, lw=2)
    lo = min(vals + [r["dense"]]) * 0.97; hi = max(vals) * 1.02
    ax.set_ylim(lo, hi); ax.set_xlim(-0.6, 4.2); ax.axhline(r["dense"], color=INK2, lw=1.2, ls=":")
    for xi, v in zip(xs, vals):
        ax.text(xi, v + (hi - lo) * 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks([0.5, 3.1], ["INT4", "mixed\nINT4/INT8"]); ax.set_title(names[key], fontsize=10.5); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("perplexity")
fig.legend(handles=[Patch(color=RTN, label="round to nearest"), Patch(color=SHAPED, label="shapequant (data-free)"),
                    Line2D([], [], color=INK2, lw=1.2, ls=":", label="unquantized model")], loc="upper left", ncol=3,
           bbox_to_anchor=(0.01, 0.98), fontsize=9.5)
fig.subplots_adjust(top=0.80, wspace=0.28)
save(fig, "fig7_perplexity.png")

# 8 ---------------------------------------------------------------- scaling (estimates)
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
M = D["memory"]; ks = list(M); P = np.array([M[k]["params_B"] for k in ks])
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4))
fig.suptitle("Figure 8. Scaling to 70B: quantization memory stays per-layer; the INT4 checkpoint is about 3.6× smaller than fp16",
             x=0.01, ha="left", weight="bold", fontsize=11.5, y=1.02)
ticks = [7, 8, 13, 32, 72]
def logx(ax):
    ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator()); ax.set_xlim(5.8, 95); ax.set_xlabel("parameters (billions)")
ax = axs[0]; dp = np.array([M[k]["device_peak_GB"] for k in ks])
ax.plot(P, dp, color=SHAPED, lw=2, marker="o", ms=8, mec=SURF, mew=2)
for g in (24, 48, 80):
    ax.axhline(g, color=MUTED, lw=1, ls="--"); ax.text(6.0, g + 1.2, f"{g} GB GPU", fontsize=8.5, color=INK2)
off = {"Llama-2-7B": (1.0, -6.0, "center"), "Llama-3.1-8B": (1.0, 3.5, "left"), "Llama-2-13B": (1.0, 3.5, "center"),
       "Qwen2.5-32B": (1.0, 3.5, "center"), "Llama-3.1-70B": (1.0, -7, "center"), "Qwen2.5-72B": (1.0, 4, "center")}
for k, p_, v in zip(ks, P, dp):
    if p_ < 20:
        continue
    fx, dy, ha = off[k]; ax.text(p_ * fx, v + dy, k, fontsize=8, color=INK2, ha=ha)
ax.text(6.0, 34, "7 = Llama-2-7B   8 = Llama-3.1-8B\n13 = Llama-2-13B", fontsize=8, color=INK2)
logx(ax); ax.set_ylabel("estimated device peak (GB)"); ax.set_ylim(0, 88); ax.set_title("device memory during quantization", fontsize=10.5)
ax = axs[1]; ck = np.array([M[k]["int4_checkpoint_GB"] for k in ks]); fp = P * 2 * 1e9 / 2 ** 30
ax.plot(P, fp, color=MUTED, lw=2, marker="o", ms=7, mec=SURF, mew=2, label="fp16 checkpoint")
ax.plot(P, ck, color=SHAPED, lw=2, marker="o", ms=8, mec=SURF, mew=2, label="INT4, group 128")
ax.text(P[-1] * 1.06, fp[-1], f"{fp[-1]:.0f} GB", fontsize=8.5, color=INK2, va="center")
ax.text(P[-1] * 1.06, ck[-1], f"{ck[-1]:.0f} GB", fontsize=8.5, color=INK, va="center")
logx(ax); ax.set_yscale("log"); ax.set_ylabel("checkpoint size (GB)"); ax.set_title("checkpoint size", fontsize=10.5)
ax.legend(loc="upper left", fontsize=9)
fig.subplots_adjust(wspace=0.28)
save(fig, "fig8_scaling.png")
print(sorted(os.listdir(OUT)))
