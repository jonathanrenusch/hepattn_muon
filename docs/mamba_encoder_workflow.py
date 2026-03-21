"""Workflow diagram for the 4-layer Bidirectional Mamba encoder used in ATLAS Muon hit filtering.

Config: atlas_muon_filtering_mamba_bidirectional_2.yaml
        (ATLAS-Muon-VisionMamba-Bidirectional_layers4)

The diagram has two sections:
  TOP  – full HitFilter pipeline (input → pack → 4x BiMamba layers → unpack → prediction)
  BOTTOM – zoomed-in detail of ONE BidirectionalMambaEncoderLayer showing exactly what
           happens to the packed sequence inside each of the 4 repeated layers.

Run this script to regenerate ``mamba_encoder_workflow.png`` in the same directory::

    python docs/mamba_encoder_workflow.py
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Colour palette  (consistent throughout both sections)
# ---------------------------------------------------------------------------
C = {
    "input":    "#3A86FF",   # blue   – raw data / I/O
    "embed":    "#8338EC",   # purple – embedding / pos-enc
    "pack":     "#FB8500",   # orange – packing / sort
    "mamba_f":  "#06A77D",   # teal   – forward Mamba2
    "mamba_b":  "#D62246",   # rose   – backward Mamba2
    "norm":     "#6C757D",   # grey   – normalisation (RMSNorm)
    "gate":     "#E76F51",   # coral  – sigmoid gate
    "residual": "#2DC2BD",   # cyan   – residual / skip
    "head":     "#9B59B6",   # violet – classification head
    "pred":     "#27AE60",   # green  – final prediction
    "arrow":    "#343A40",
    "text":     "#212529",
    "bg":       "#F8F9FA",
    "layer_bg": "#E8F5E9",
    "zoom_bg":  "#EEF2FF",
    "mamba_int":"#AEDFF7",   # light blue – Mamba2 internals
}

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def box(ax, cx, cy, w, h, label, sub="", color="#3A86FF",
        fs=9, sub_fs=7.5, tc="white", bold=False, alpha=0.93, zorder=3):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="white",
        linewidth=1.3, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy + h * 0.15, label, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight, zorder=zorder + 1)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=sub_fs, color=tc, alpha=0.88, style="italic",
                zorder=zorder + 1)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, color=tc, fontweight=weight, zorder=zorder + 1)


def arr(ax, x0, y0, x1, y1, color="#343A40", lw=1.5, rad=0.0, zorder=5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=lw,
                    connectionstyle=f"arc3,rad={rad}",
                    mutation_scale=10,
                ),
                zorder=zorder)


def hline(ax, x0, x1, y, color, lw=1.5, zorder=5):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=zorder)


def vline(ax, x, y0, y1, color, lw=1.5, zorder=5):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=zorder)


def bracket_bg(ax, x0, y0, w, h, color, lw=2.0, ls="--", alpha=0.35, zorder=1):
    patch = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor=color.replace("E8", "A8").replace("EE", "AA"),
        linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)


# ---------------------------------------------------------------------------
# TOP SECTION  –  full HitFilter pipeline (compact, left column)
# ---------------------------------------------------------------------------

def draw_pipeline(ax, x0=0.5, x1=7.8):
    """Draw the full pipeline in a tall column between x0 and x1."""
    cx = (x0 + x1) / 2
    bw = x1 - x0 - 0.2
    bh = 0.72
    bh_lg = 1.05  # large box

    # y positions from top to bottom
    top = 27.8
    ys = {}
    ys["title"]  = top
    ys["input"]  = top - 1.1
    ys["ln"]     = top - 2.15
    ys["dense"]  = top - 3.20
    ys["posenc"] = top - 4.25
    ys["add"]    = top - 5.10
    ys["sort"]   = top - 6.05
    ys["pack"]   = top - 7.25
    layer_top    = top - 8.45
    ys["layer_top"] = layer_top
    layer_h      = 4.20
    ys["layer_bot"] = layer_top - layer_h
    ys["fn"]     = layer_top - layer_h - 1.00
    ys["unpack"] = layer_top - layer_h - 2.10
    ys["head"]   = layer_top - layer_h - 3.20
    ys["sigmoid"]= layer_top - layer_h - 4.10
    ys["pred"]   = layer_top - layer_h - 5.00

    # title of section
    ax.text(cx, ys["title"], "FULL PIPELINE",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C["text"])

    # boxes
    box(ax, cx, ys["input"], bw, bh,
        "Input: Detector Hits",
        "B events x N_max hits x 18 features",
        color=C["input"], bold=True)

    box(ax, cx, ys["ln"], bw, bh * 0.8,
        "InputNet: LayerNorm",
        "normalise 18 raw features",
        color=C["norm"])

    box(ax, cx, ys["dense"], bw, bh,
        "InputNet: Dense MLP",
        "Linear(18,72) -> SwiGLU -> Linear(36,128)",
        color=C["embed"])

    box(ax, cx, ys["posenc"], bw, bh,
        "PositionEncoder (r, eta, phi)",
        "sinusoidal enc, concat -> D=128",
        color=C["embed"])

    box(ax, cx, ys["add"], bw * 0.65, bh * 0.7,
        "add pos encoding to embedding",
        color=C["embed"], fs=8.5)

    box(ax, cx, ys["sort"], bw, bh,
        "Sort hits by phi",
        "argsort(phi) -> gather  (B, N_max, D)",
        color=C["pack"])

    box(ax, cx, ys["pack"], bw, bh_lg,
        "PACK  (batching technique)",
        "pad_mask -> (1, SL, D)  no padding\n"
        "seq_idx: SSM resets at event boundaries\n"
        "flip_idx: precomputed reverse gather",
        color=C["pack"], bold=True, fs=8.5, sub_fs=7)

    # layer repeat bracket
    bracket_bg(ax,
               cx - bw / 2 - 0.25, ys["layer_bot"] - 0.15,
               bw + 0.5, layer_h + 0.35,
               C["layer_bg"], lw=2.0, ls="--")
    ax.text(cx + bw / 2 + 0.45,
            (layer_top + ys["layer_bot"]) / 2,
            "x 4\nlayers",
            ha="center", va="center", fontsize=9,
            fontweight="bold", color="#388E3C", zorder=4)

    # inside the layer block – condensed single summary box
    ly_mid = (layer_top + ys["layer_bot"]) / 2
    box(ax, cx, ly_mid + 0.55, bw * 0.88, bh * 0.75,
        "RMSNorm  (pre-norm)", color=C["norm"], fs=8.5)
    box(ax, cx - bw * 0.24, ly_mid - 0.30, bw * 0.38, bh * 0.8,
        "Forward\nMamba2",
        color=C["mamba_f"], fs=8, tc="white")
    box(ax, cx + bw * 0.24, ly_mid - 0.30, bw * 0.38, bh * 0.8,
        "Backward\nMamba2",
        color=C["mamba_b"], fs=8, tc="white")
    box(ax, cx, ly_mid - 1.35, bw * 0.88, bh * 0.75,
        "Sigmoid Gate  +  Residual (+skip)",
        color=C["gate"], fs=8.5)

    # small arrows inside layer
    arr(ax, cx, ly_mid + 0.55 - bh*0.75/2,
           cx - bw*0.24, ly_mid - 0.30 + bh*0.8/2,
           color=C["mamba_f"], lw=1.1)
    arr(ax, cx, ly_mid + 0.55 - bh*0.75/2,
           cx + bw*0.24, ly_mid - 0.30 + bh*0.8/2,
           color=C["mamba_b"], lw=1.1)
    arr(ax, cx - bw*0.24, ly_mid - 0.30 - bh*0.8/2,
           cx, ly_mid - 1.35 + bh*0.75/2,
           color=C["mamba_f"], lw=1.1)
    arr(ax, cx + bw*0.24, ly_mid - 0.30 - bh*0.8/2,
           cx, ly_mid - 1.35 + bh*0.75/2,
           color=C["mamba_b"], lw=1.1)

    box(ax, cx, ys["fn"], bw, bh * 0.8,
        "Final RMSNorm  (after all 4 layers)",
        color=C["norm"], fs=8.5)

    box(ax, cx, ys["unpack"], bw, bh,
        "UNPACK  +  Unsort",
        "(1,SL,D) -> (B,N_max,D), undo phi sort",
        color=C["pack"], bold=True)

    box(ax, cx, ys["head"], bw, bh,
        "HitFilterTask: Dense  Linear(128->1)",
        "per-hit logit  (B, N_max)",
        color=C["head"])

    box(ax, cx, ys["sigmoid"], bw * 0.65, bh * 0.75,
        "Sigmoid  ->  probability in (0,1)",
        color=C["gate"], fs=8.5)

    box(ax, cx, ys["pred"], bw, bh,
        "keep if  sigmoid(logit) >= 0.01",
        "True = hit on reconstructable muon track",
        color=C["pred"], bold=True)

    # inter-box arrows (main flow)
    flow = [
        (ys["input"],  bh,      ys["ln"],     bh * 0.8),
        (ys["ln"],     bh*0.8,  ys["dense"],  bh),
        (ys["dense"],  bh,      ys["posenc"], bh),
        (ys["posenc"], bh,      ys["add"],    bh*0.7),
        (ys["add"],    bh*0.7,  ys["sort"],   bh),
        (ys["sort"],   bh,      ys["pack"],   bh_lg),
        (ys["pack"],   bh_lg,   layer_top + 0.15, 0),
    ]
    for (ya, ha_, yb, hb) in flow:
        arr(ax, cx, ya - ha_/2, cx, yb + hb/2)

    # layer block -> final norm
    arr(ax, cx, ys["layer_bot"] - 0.15, cx, ys["fn"] + bh*0.8/2)
    arr(ax, cx, ys["fn"] - bh*0.8/2, cx, ys["unpack"] + bh/2)
    arr(ax, cx, ys["unpack"] - bh/2, cx, ys["head"] + bh/2)
    arr(ax, cx, ys["head"] - bh/2, cx, ys["sigmoid"] + bh*0.75/2)
    arr(ax, cx, ys["sigmoid"] - bh*0.75/2, cx, ys["pred"] + bh/2)

    return ys


# ---------------------------------------------------------------------------
# BOTTOM / RIGHT SECTION  –  zoomed detail of ONE BiMamba layer
# ---------------------------------------------------------------------------

def draw_layer_detail(ax, x0=8.5, x1=19.5, y_top=27.8, y_bot=10.5):
    """
    Draw a detailed diagram of ONE BidirectionalMambaEncoderLayer.
    The packed sequence (1, SL, D) flows top-to-bottom through the layer.
    Forward and backward paths are shown side-by-side.
    """
    W = x1 - x0
    cx = (x0 + x1) / 2

    # vertical rhythm
    bh = 0.80
    gap = 0.28
    y = y_top - 0.55        # current y cursor (top of each box)

    def cur_y():
        return y

    # title
    ax.text(cx, y_top,
            "ZOOM-IN: One BidirectionalMambaEncoderLayer  (repeated x4)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C["text"])
    ax.text(cx, y_top - 0.50,
            "Config: d_state=32  d_conv=4  expand=2  headdim=32  norm=RMSNorm  D=128",
            ha="center", va="center", fontsize=8.5, color=C["text"],
            style="italic", alpha=0.75)

    y -= 0.55  # now at first box centre

    # ------ helper to advance y ------
    def step(h):
        nonlocal y
        y -= h / 2 + gap

    # ── (A) Incoming packed sequence ─────────────────────────────────────────
    bw_full = W - 0.6
    y_in = y - bh / 2 - 0.3
    box(ax, cx, y_in, bw_full, bh,
        "Input:  x  shape (1, SL, D=128)  — packed sequence",
        "SL = total valid hits across all B events; no padding",
        color=C["input"], bold=True, fs=9.5)
    y = y_in
    step(bh)

    # ── (B) Save skip ─────────────────────────────────────────────────────────
    y_skip_save = y
    box(ax, cx, y_skip_save, bw_full * 0.55, bh * 0.65,
        "skip = x                  (save for residual)",
        color=C["residual"], fs=8.5, tc="white")
    step(bh * 0.65)

    # skip rail on the left
    skip_rail_x = x0 + 0.18
    skip_rail_y_start = y_skip_save - bh * 0.65 / 2
    # we'll set skip_rail_y_end later once we know where residual box lands

    # ── (C) RMSNorm ──────────────────────────────────────────────────────────
    y_norm = y
    box(ax, cx, y_norm, bw_full * 0.72, bh * 0.75,
        "RMSNorm  (pre-norm)      x_norm = RMSNorm(x).contiguous()",
        color=C["norm"], fs=8.5)
    step(bh * 0.75)

    arr(ax, cx, y_skip_save - bh*0.65/2, cx, y_norm + bh*0.75/2, lw=1.4)

    # ── split arrow: two branches ─────────────────────────────────────────────
    split_y = y - 0.05
    arr(ax, cx, y_norm - bh*0.75/2, cx, split_y + 0.05, lw=1.4)

    # left branch (forward) and right branch (backward)
    fwd_cx = x0 + W * 0.27
    bwd_cx = x0 + W * 0.73
    branch_w = W * 0.40

    # horizontal split line
    arr(ax, cx, split_y, fwd_cx, split_y, color=C["mamba_f"], lw=1.5, rad=0.0)
    arr(ax, cx, split_y, bwd_cx, split_y, color=C["mamba_b"], lw=1.5, rad=0.0)

    # ── FORWARD branch labels ─────────────────────────────────────────────────
    y_branch_label = split_y - 0.30
    ax.text(fwd_cx, y_branch_label, "FORWARD PATH  (left -> right)",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C["mamba_f"])
    ax.text(bwd_cx, y_branch_label, "BACKWARD PATH  (right -> left)",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C["mamba_b"])

    y = y_branch_label - 0.35

    # ── FORWARD (D) ──────────────────────────────────────────────────────────
    int_bh = 0.68
    int_bw = branch_w - 0.3

    # ── BACKWARD flip box ────────────────────────────────────────────────────
    y_fwd_pass = y
    y_bwd_flip = y

    box(ax, fwd_cx, y_fwd_pass, int_bw, int_bh * 0.7,
        "pass x_norm unchanged",
        color=C["mamba_f"], fs=8, tc="white", alpha=0.5)

    box(ax, bwd_cx, y_bwd_flip, int_bw, int_bh,
        "FLIP via flip_idx  (vectorised)",
        "x_rev = x_norm[:, flip_idx]  — one GPU gather\n"
        "reverses each event's sub-seq in the packed tensor",
        color=C["mamba_b"], fs=8, sub_fs=7, tc="white")

    y -= max(int_bh, int_bh * 0.7) + gap

    arr(ax, fwd_cx, split_y, fwd_cx, y_fwd_pass + int_bh*0.7/2,
        color=C["mamba_f"], lw=1.3)
    arr(ax, bwd_cx, split_y, bwd_cx, y_bwd_flip + int_bh/2,
        color=C["mamba_b"], lw=1.3)

    # ── Mamba2 internals box (same for both paths) ────────────────────────────
    def draw_mamba2_internal(ax, mcx, top_y, color, label_color):
        """Draw the three sub-steps inside one Mamba2 call."""
        iw = int_bw - 0.1
        ih = 0.58
        iy = top_y
        steps_int = [
            ("1. Linear projections\n   (x, B, C, dt from x_norm)", "#C8E6C9" if label_color == C["mamba_f"] else "#FFCDD2"),
            ("2. 1-D Causal Conv  (d_conv=4)\n   short-range context mixing", "#A5D6A7" if label_color == C["mamba_f"] else "#EF9A9A"),
            ("3. SSM Selective Scan\n   d_state=32 recurrent state\n   seq_idx resets at event\n   boundaries", "#81C784" if label_color == C["mamba_f"] else "#E57373"),
            ("4. Output projection\n   -> (1, SL, D=128)", "#4CAF50" if label_color == C["mamba_f"] else "#F44336"),
        ]
        sub_ih = [0.58, 0.58, 0.90, 0.58]
        y_cur = iy
        for (lbl, fc), sh in zip(steps_int, sub_ih):
            tc = "black" if fc in ("#C8E6C9", "#FFCDD2", "#A5D6A7", "#EF9A9A") else "white"
            box(ax, mcx, y_cur, iw * 0.92, sh, lbl,
                color=fc, fs=7, tc=tc, alpha=0.95, zorder=5)
            y_cur -= sh + 0.12
            if y_cur > iy - (sum(sub_ih) + 4*0.12):
                arr(ax, mcx, y_cur + 0.06, mcx, y_cur - 0.02, color=color, lw=1.0)
        return y_cur  # bottom of last sub-box

    # forward path Mamba2 internals
    y_mamba2_top_fwd = y - 0.05
    y_mamba2_bot_fwd = draw_mamba2_internal(
        ax, fwd_cx, y_mamba2_top_fwd, C["mamba_f"], C["mamba_f"])

    # backward path Mamba2 internals
    y_mamba2_top_bwd = y - 0.05
    y_mamba2_bot_bwd = draw_mamba2_internal(
        ax, bwd_cx, y_mamba2_top_bwd, C["mamba_b"], C["mamba_b"])

    arr(ax, fwd_cx, y_fwd_pass - int_bh*0.7/2, fwd_cx, y_mamba2_top_fwd,
        color=C["mamba_f"], lw=1.3)
    arr(ax, bwd_cx, y_bwd_flip - int_bh/2, bwd_cx, y_mamba2_top_bwd,
        color=C["mamba_b"], lw=1.3)

    y = min(y_mamba2_bot_fwd, y_mamba2_bot_bwd) - gap

    # ── UN-FLIP backward ──────────────────────────────────────────────────────
    y_bwd_unflip = y
    box(ax, bwd_cx, y_bwd_unflip, int_bw, int_bh,
        "UN-FLIP via flip_idx",
        "x_bwd = x_bwd[:, flip_idx]\nrestores original token order",
        color=C["mamba_b"], fs=8, sub_fs=7, tc="white")

    box(ax, fwd_cx, y_bwd_unflip, int_bw, int_bh * 0.7,
        "x_fwd ready",
        color=C["mamba_f"], fs=8, tc="white", alpha=0.5)

    arr(ax, fwd_cx, y_mamba2_bot_fwd,
        fwd_cx, y_bwd_unflip + int_bh*0.7/2, color=C["mamba_f"], lw=1.3)
    arr(ax, bwd_cx, y_mamba2_bot_bwd,
        bwd_cx, y_bwd_unflip + int_bh/2, color=C["mamba_b"], lw=1.3)

    y -= max(int_bh, int_bh * 0.7) + gap + 0.1

    # horizontal merge arrows
    y_merge = y
    arr(ax, fwd_cx, y_bwd_unflip - int_bh*0.7/2,
        cx, y_merge + 0.02, color=C["mamba_f"], lw=1.3)
    arr(ax, bwd_cx, y_bwd_unflip - int_bh/2,
        cx, y_merge + 0.02, color=C["mamba_b"], lw=1.3)

    # ── Gate ──────────────────────────────────────────────────────────────────
    y_gate = y_merge - 0.2
    box(ax, cx, y_gate, bw_full * 0.78, int_bh * 1.25,
        "Sigmoid GATE  (learned combination)",
        "gate = sigmoid( Linear(x_norm) )  shape (1, SL, D)\n"
        "x_combined = gate * x_fwd  +  (1 - gate) * x_bwd\n"
        "gate close to 1 -> trust forward; close to 0 -> trust backward",
        color=C["gate"], fs=9, sub_fs=7.5, bold=False)
    step(int_bh * 1.25)

    arr(ax, cx, y_merge, cx, y_gate + int_bh*1.25/2, lw=1.5)
    step(0)

    # ── Residual ──────────────────────────────────────────────────────────────
    y_res = y
    box(ax, cx, y_res, bw_full * 0.72, bh * 0.8,
        "Residual:  x_out  =  skip  +  x_combined",
        "shape (1, SL, D=128)  — ready for next layer",
        color=C["residual"], fs=9, sub_fs=8)
    step(bh * 0.8)

    arr(ax, cx, y_gate - int_bh*1.25/2, cx, y_res + bh*0.8/2, lw=1.5)

    # skip residual rail (left side)
    skip_rail_y_end = y_res - bh * 0.8 / 2 - 0.05
    vline(ax, skip_rail_x, skip_rail_y_start, skip_rail_y_end,
          color=C["residual"], lw=2.0)
    hline(ax, skip_rail_x,
          cx - bw_full * 0.72 / 2, skip_rail_y_end,
          color=C["residual"], lw=2.0)
    ax.annotate("",
                xy=(cx - bw_full*0.72/2 + 0.02, skip_rail_y_end),
                xytext=(skip_rail_x, skip_rail_y_end),
                arrowprops=dict(arrowstyle="-|>", color=C["residual"],
                                lw=2.0, mutation_scale=10),
                zorder=6)
    ax.text(skip_rail_x - 0.22,
            (skip_rail_y_start + skip_rail_y_end) / 2,
            "skip", fontsize=8, color=C["residual"],
            ha="center", va="center", rotation=90, zorder=7)

    # ── Output ────────────────────────────────────────────────────────────────
    y_out = y
    box(ax, cx, y_out, bw_full, bh * 0.7,
        "Output:  x  shape (1, SL, D=128)  — passed to next layer (or final RMSNorm)",
        color=C["input"], bold=True, fs=9)
    arr(ax, cx, y_res - bh*0.8/2, cx, y_out + bh*0.7/2, lw=1.5)

    return y_out - bh * 0.7 / 2  # bottom of last box


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def draw_legend(ax, y_bottom, fig_w):
    items = [
        (C["input"],    "Input / Output tensor"),
        (C["embed"],    "Embedding / Pos-Enc"),
        (C["pack"],     "Pack / Sort / Unpack"),
        (C["mamba_f"],  "Forward Mamba2"),
        (C["mamba_b"],  "Backward Mamba2"),
        (C["norm"],     "RMSNorm"),
        (C["gate"],     "Sigmoid Gate"),
        (C["residual"], "Residual / Skip"),
        (C["head"],     "Classification Head"),
    ]
    n = len(items)
    total_w = fig_w - 1.0
    col_w = total_w / n
    lx0 = 0.5
    ly = y_bottom - 0.55
    ax.text(fig_w / 2, ly + 0.35, "Legend",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=C["text"])
    for i, (c, label) in enumerate(items):
        cx_ = lx0 + i * col_w + col_w / 2
        ax.add_patch(FancyBboxPatch(
            (cx_ - col_w*0.38, ly - 0.18), col_w*0.76, 0.34,
            boxstyle="round,pad=0.03",
            facecolor=c, edgecolor="white", linewidth=0.8, alpha=0.9, zorder=8))
        ax.text(cx_, ly - 0.01, label,
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", zorder=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def draw_workflow():
    fig_w = 20.0
    fig_h = 30.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    # ── Main title ────────────────────────────────────────────────────────────
    ax.text(fig_w / 2, fig_h - 0.45,
            "ATLAS-Muon-VisionMamba-Bidirectional_layers4  —  Hit Filtering Workflow",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["text"])
    ax.text(fig_w / 2, fig_h - 0.95,
            "Config: atlas_muon_filtering_mamba_bidirectional_2.yaml  |  "
            "num_layers=4  dim=128  d_state=32  d_conv=4  expand=2  headdim=32  norm=RMSNorm",
            ha="center", va="center", fontsize=8.5, color=C["text"], alpha=0.7,
            style="italic")

    # vertical divider between left pipeline and right detail
    divider_x = 8.1
    ax.plot([divider_x, divider_x], [1.2, fig_h - 1.3],
            color="#DEE2E6", lw=1.2, zorder=1)

    # ── LEFT: pipeline ────────────────────────────────────────────────────────
    draw_pipeline(ax, x0=0.3, x1=divider_x - 0.3)

    # ── RIGHT: layer detail ───────────────────────────────────────────────────
    draw_layer_detail(ax, x0=divider_x + 0.15, x1=fig_w - 0.3,
                      y_top=fig_h - 1.55, y_bot=2.0)

    # ── Legend at bottom ─────────────────────────────────────────────────────
    draw_legend(ax, y_bottom=1.8, fig_w=fig_w)

    plt.tight_layout(pad=0.1)
    out_path = Path(__file__).parent / "mamba_encoder_workflow.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    draw_workflow()
