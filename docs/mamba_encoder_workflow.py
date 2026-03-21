"""Workflow diagram for the 4-layer Bidirectional Mamba encoder used in ATLAS Muon hit filtering.

Run this script to regenerate ``mamba_encoder_workflow.png`` in the same directory::

    python docs/mamba_encoder_workflow.py

The diagram shows every computational step from raw detector-hit features all the way through
to the final binary keep/discard prediction, with special attention to the *sequence-packing*
batching technique that avoids compute on padded positions.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C = {
    "input":    "#4A90D9",   # blue  – raw data / I/O
    "embed":    "#7B68EE",   # purple – embedding / positional encoding
    "pack":     "#F5A623",   # orange – packing / batching
    "mamba":    "#2ECC71",   # green  – Mamba SSM blocks
    "norm":     "#95A5A6",   # grey   – normalisation
    "gate":     "#E74C3C",   # red    – gating / sigmoid
    "residual": "#1ABC9C",   # teal   – residual / skip
    "unpack":   "#F5A623",   # orange – unpack (same as pack)
    "head":     "#9B59B6",   # violet – classification head
    "pred":     "#27AE60",   # dark green – prediction output
    "layer_bg": "#FAFAFA",   # very light – layer repeat background
    "arrow":    "#2C3E50",   # dark – arrows
    "text":     "#2C3E50",
}

# ---------------------------------------------------------------------------
# Helper drawing utilities
# ---------------------------------------------------------------------------

def rounded_box(ax, xy, width, height, label, sublabel="", color="#4A90D9",
                fontsize=9, text_color="white", alpha=0.92, bold=False):
    """Draw a rounded rectangle with centred label (and optional sublabel)."""
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width, height,
        boxstyle="round,pad=0.03",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(x, y + height * 0.12, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=4)
        ax.text(x, y - height * 0.2, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color, alpha=0.85, zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight, zorder=4)
    return box


def arrow(ax, x0, y0, x1, y1, color="#2C3E50", lw=1.4, style="->"):
    """Draw a straight arrow between two points."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=5)


def bracket_arrow(ax, x0, y0, x1, y1, color="#2C3E50", lw=1.2, rad=0.3):
    """Draw a curved arrow (for skip/residual connections)."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=5)


# ---------------------------------------------------------------------------
# Main diagram
# ---------------------------------------------------------------------------

def draw_workflow():
    fig_w, fig_h = 14, 24
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")

    cx = fig_w / 2          # centre x
    bw = 7.0                # default box width
    bh = 0.75               # default box height

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(cx, fig_h - 0.5, "4-Layer Bidirectional Mamba Encoder – Hit-Filtering Workflow",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C["text"])
    ax.text(cx, fig_h - 1.05, "Batching technique: Sequence Packing  |  Task: ATLAS Muon hit keep/discard",
            ha="center", va="center", fontsize=9, color=C["text"], alpha=0.7)

    # ── Step positions (top → bottom) ────────────────────────────────────────
    y_input      = fig_h - 2.0
    y_norm_input = fig_h - 3.1
    y_dense      = fig_h - 4.2
    y_posenc     = fig_h - 5.3
    y_add        = fig_h - 6.1
    y_sort       = fig_h - 7.1
    y_pack       = fig_h - 8.2

    # Mamba layer block (4 layers)
    layer_top    = fig_h - 9.2
    layer_h      = 5.4       # total height of the "repeat × 4" bracket
    layer_bottom = layer_top - layer_h

    # Inside the layer block
    y_ln_sa   = layer_top - 0.55
    y_fwd     = layer_top - 1.55
    y_bwd_box = layer_top - 2.65
    y_gate    = layer_top - 3.75
    y_res     = layer_top - 4.65

    y_final_norm = layer_bottom - 0.9
    y_unpack     = layer_bottom - 1.9
    y_dense_head = layer_bottom - 3.0
    y_sigmoid    = layer_bottom - 4.0
    y_pred       = layer_bottom - 4.9

    # ── 1. Input ─────────────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_input), bw, bh,
                "Input: Detector Hits",
                "Batch of B events · N_max hits per event · 18 features per hit\n"
                "(coordinates, detector info, derived: r, s, θ, φ, η)",
                color=C["input"], bold=True)

    arrow(ax, cx, y_input - bh/2, cx, y_norm_input + bh/2)

    # ── 2. InputNet: LayerNorm ────────────────────────────────────────────────
    rounded_box(ax, (cx, y_norm_input), bw, bh,
                "InputNet – LayerNorm",
                "Normalise the 18 raw features (norm_input=True)",
                color=C["norm"])

    arrow(ax, cx, y_norm_input - bh/2, cx, y_dense + bh/2)

    # ── 3. InputNet: Dense MLP ────────────────────────────────────────────────
    rounded_box(ax, (cx, y_dense), bw, bh,
                "InputNet – Dense MLP  (Linear → SwiGLU → Linear)",
                "18 → 256 (gated) → 128   |   output: (B, N, D=128)",
                color=C["embed"])

    arrow(ax, cx, y_dense - bh/2, cx, y_posenc + bh/2)

    # ── 4. PositionEncoder ────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_posenc), bw, bh,
                "PositionEncoder  (r, η, φ → sinusoidal)",
                "per_input_dim = D // 3 = 42 per field · concat → D=128",
                color=C["embed"])

    arrow(ax, cx, y_posenc - bh/2, cx, y_add + bh/2)

    # ── 5. Add embeddings ─────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_add), bw * 0.7, bh * 0.8,
                "⊕  Add position encoding to MLP embedding",
                color=C["embed"], fontsize=9)

    arrow(ax, cx, y_add - bh*0.8/2, cx, y_sort + bh/2)

    # ── 6. Sort by φ ─────────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_sort), bw, bh,
                "Sort hits by φ (azimuthal angle)",
                "argsort(φ) → gather x by sort_idx   |   (B, N, D) → (B, N, D) phi-ordered",
                color=C["pack"])

    arrow(ax, cx, y_sort - bh/2, cx, y_pack + bh/2)

    # ── 7. Packing ────────────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_pack), bw, bh * 1.6,
                "Sequence Packing  (batching technique)",
                "boolean-mask valid tokens: (B, N_max, D) → (1, ΣL, D)\n"
                "seq_idx (1, ΣL) – event labels; Mamba2 resets SSM at boundaries\n"
                "flip_idx (ΣL,) – precomputed reverse-gather index for backward pass",
                color=C["pack"], bold=True, fontsize=8.5)

    arrow(ax, cx, y_pack - bh*1.6/2, cx, layer_top + 0.05)

    # ── Layer repeat background ───────────────────────────────────────────────
    layer_bg = FancyBboxPatch(
        (cx - bw/2 - 0.35, layer_bottom - 0.15),
        bw + 0.7, layer_h + 0.35,
        boxstyle="round,pad=0.05",
        facecolor="#E8F8F5",
        edgecolor="#2ECC71",
        linewidth=2.0,
        linestyle="--",
        alpha=0.4,
        zorder=1,
    )
    ax.add_patch(layer_bg)
    ax.text(cx + bw/2 + 0.45, (layer_top + layer_bottom) / 2,
            "× 4\nlayers", ha="center", va="center", fontsize=10,
            fontweight="bold", color="#2ECC71", zorder=4)

    # ── 8a. Pre-LayerNorm ─────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_ln_sa), bw * 0.82, bh,
                "Pre-LayerNorm  (norm each packed token)",
                "x_norm = LayerNorm(x).contiguous()   |   (1, ΣL, D)",
                color=C["norm"])

    # skip connection start
    skip_x = cx - bw/2 - 0.1
    skip_y_start = y_ln_sa + bh / 2
    skip_y_end   = y_res   - bh / 2 - 0.05

    arrow(ax, cx, y_ln_sa - bh/2, cx, y_fwd + bh/2)

    # ── 8b. Forward Mamba2 ────────────────────────────────────────────────────
    fw = 2.8
    x_fwd = cx - 1.8
    rounded_box(ax, (x_fwd, y_fwd), fw, bh,
                "Forward Mamba2",
                "Conv1D (d_conv=4)\n→ SSM selective scan\n(seq_idx resets state)",
                color=C["mamba"], fontsize=8)

    # ── 8c. Backward Mamba2 ───────────────────────────────────────────────────
    x_bwd = cx + 1.8
    rounded_box(ax, (x_bwd, y_bwd_box), fw, bh * 1.35,
                "Backward Mamba2",
                "x_rev = x_norm[:, flip_idx]  (gather)\n"
                "Conv1D → SSM selective scan\n"
                "x_bwd = x_bwd[:, flip_idx]  (un-flip)",
                color=C["mamba"], fontsize=8)

    # arrows from norm into fwd and bwd
    arrow(ax, x_fwd, y_ln_sa - bh/2, x_fwd, y_fwd + bh/2, color=C["arrow"], lw=1.2)
    arrow(ax, x_bwd, y_ln_sa - bh/2, x_bwd, y_bwd_box + bh*1.35/2, color=C["arrow"], lw=1.2)

    # arrows from fwd/bwd into gate
    arrow(ax, x_fwd, y_fwd - bh/2,            cx, y_gate + bh/2, color=C["arrow"], lw=1.2)
    arrow(ax, x_bwd, y_bwd_box - bh*1.35/2,   cx, y_gate + bh/2, color=C["arrow"], lw=1.2)

    # ── 8d. Gating ────────────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_gate), bw * 0.82, bh * 1.05,
                "Gating  (Sigmoid)",
                "gate = σ( Linear(x_norm) )\n"
                "x_comb = gate ⊗ x_fwd  +  (1 − gate) ⊗ x_bwd",
                color=C["gate"])

    arrow(ax, cx, y_gate - bh*1.05/2, cx, y_res + bh/2)

    # ── 8e. Residual ──────────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_res), bw * 0.82, bh,
                "Residual connection  (⊕ skip)",
                "x = skip + x_comb   |   (1, ΣL, D)",
                color=C["residual"])

    # draw skip arrow on the left side
    ax.annotate("",
                xy=(skip_x + 0.05, skip_y_end),
                xytext=(skip_x, skip_y_start),
                arrowprops=dict(arrowstyle="->", color=C["residual"],
                                lw=1.5, connectionstyle="arc3,rad=0.0"),
                zorder=6)
    ax.plot([skip_x, skip_x], [skip_y_start, skip_y_end],
            color=C["residual"], lw=1.5, zorder=6)
    ax.plot([skip_x, cx - bw*0.82/2], [skip_y_end, y_res],
            color=C["residual"], lw=1.5, zorder=6)

    ax.text(skip_x - 0.25, (skip_y_start + skip_y_end) / 2,
            "skip", fontsize=7.5, color=C["residual"],
            ha="center", va="center", rotation=90, zorder=7)

    # arrow out of the layer block
    arrow(ax, cx, y_res - bh/2, cx, y_final_norm + bh/2)

    # ── 9. Final LayerNorm ────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_final_norm), bw, bh,
                "Final LayerNorm  (on packed sequence)",
                "x_packed = LayerNorm(x_packed)   |   (1, ΣL, D)",
                color=C["norm"])

    arrow(ax, cx, y_final_norm - bh/2, cx, y_unpack + bh/2)

    # ── 10. Unpack + unsort ───────────────────────────────────────────────────
    rounded_box(ax, (cx, y_unpack), bw, bh * 1.35,
                "Unpack  +  Unsort",
                "x_out[pad_mask] = x_packed.squeeze(0)  → (B, N_max, D)\n"
                "gather with unsort_idx → restore original hit order",
                color=C["unpack"], bold=True, fontsize=8.5)

    arrow(ax, cx, y_unpack - bh*1.35/2, cx, y_dense_head + bh/2)

    # ── 11. Classification head (Dense 128→1) ─────────────────────────────────
    rounded_box(ax, (cx, y_dense_head), bw, bh,
                "HitFilterTask – Dense  (Linear 128 → 1)",
                "per-hit logit   |   (B, N_max, 1) → squeeze → (B, N_max)",
                color=C["head"])

    arrow(ax, cx, y_dense_head - bh/2, cx, y_sigmoid + bh/2)

    # ── 12. Sigmoid ───────────────────────────────────────────────────────────
    rounded_box(ax, (cx, y_sigmoid), bw * 0.7, bh * 0.85,
                "σ  Sigmoid  →  probability ∈ (0, 1)",
                color=C["gate"], fontsize=9)

    arrow(ax, cx, y_sigmoid - bh*0.85/2, cx, y_pred + bh/2)

    # ── 13. Binary prediction ─────────────────────────────────────────────────
    rounded_box(ax, (cx, y_pred), bw, bh,
                "Prediction: keep hit if  σ(logit) ≥ threshold (0.01)",
                "True → hit belongs to a reconstructable muon track",
                color=C["pred"], bold=True)

    # ── Step number labels ────────────────────────────────────────────────────
    steps = [
        (cx - bw/2 - 0.6, y_input,      " 1 "),
        (cx - bw/2 - 0.6, y_norm_input, " 2 "),
        (cx - bw/2 - 0.6, y_dense,      " 3 "),
        (cx - bw/2 - 0.6, y_posenc,     " 4 "),
        (cx - bw/2 - 0.6, y_add,        " 5 "),
        (cx - bw/2 - 0.6, y_sort,       " 6 "),
        (cx - bw/2 - 0.6, y_pack,       " 7 "),
        (cx - bw/2 - 0.6, y_ln_sa,      "8a"),
        (cx - bw/2 - 0.6, y_fwd,        "8b"),
        (cx - bw/2 - 0.6, y_bwd_box,    "8c"),
        (cx - bw/2 - 0.6, y_gate,       "8d"),
        (cx - bw/2 - 0.6, y_res,        "8e"),
        (cx - bw/2 - 0.6, y_final_norm, " 9 "),
        (cx - bw/2 - 0.6, y_unpack,     "10"),
        (cx - bw/2 - 0.6, y_dense_head, "11"),
        (cx - bw/2 - 0.6, y_sigmoid,    "12"),
        (cx - bw/2 - 0.6, y_pred,       "13"),
    ]
    for sx, sy, label in steps:
        ax.text(sx, sy, label, ha="center", va="center",
                fontsize=8.5, color=C["text"], alpha=0.6, zorder=7)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (C["input"],    "Input / Output"),
        (C["embed"],    "Embedding / Positional Encoding"),
        (C["pack"],     "Packing / Batching / Sort"),
        (C["mamba"],    "Mamba2 SSM Block"),
        (C["norm"],     "Normalisation"),
        (C["gate"],     "Gating / Sigmoid"),
        (C["residual"], "Residual Connection"),
        (C["head"],     "Classification Head"),
    ]
    lx0, ly0 = 0.35, 1.45
    for i, (c, label) in enumerate(legend_items):
        patch = mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.8, label=label)
        ax.add_patch(FancyBboxPatch((lx0 + i * 1.7, ly0 - 0.18), 0.28, 0.28,
                                    boxstyle="round,pad=0.02",
                                    facecolor=c, edgecolor="white", linewidth=0.6,
                                    zorder=8))
        ax.text(lx0 + i * 1.7 + 0.35, ly0 - 0.04, label,
                ha="left", va="center", fontsize=7, color=C["text"], zorder=8)

    ax.text(cx, 0.65,
            "Tensor shapes: (B, N_max, D) padded   ↔   (1, ΣL, D) packed\n"
            "D = 128 · d_state = 64 · d_conv = 4 · expand = 2 · headdim = 32/64",
            ha="center", va="center", fontsize=8, color=C["text"],
            style="italic", alpha=0.7)

    plt.tight_layout(pad=0.2)
    out_path = Path(__file__).parent / "mamba_encoder_workflow.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    draw_workflow()
