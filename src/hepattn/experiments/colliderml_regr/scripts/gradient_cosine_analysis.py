"""Per-head gradient cosine-similarity analysis.

Motivation: d0 uses a binned-DFL *classification* head (420 bins + pinball
coupling), while z0/phi/theta/qop use continuous quantile/circular
regression heads. All heads share the same trunk (encoder + pool_head).
If the d0-classification gradient direction fights the regression-head
gradient directions on the shared trunk, multi-task training will converge
sub-optimally.

For N minibatches we compute, on the *shared trunk parameters* (encoder +
pool_head), the per-parameter loss-gradient vectors g_d0, g_z0, g_phi,
g_theta, g_qop, and report the pairwise cosine-similarity distribution.

Usage:
    pixi run python -m hepattn.experiments.colliderml_regr.scripts.gradient_cosine_analysis \
        --config logs/comet_offline/<run>/config.yaml \
        --ckpt logs/comet_offline/<run>/ckpts/last.ckpt \
        --data-dir /scratch/colliderml/p200_core_finetune \
        --output-dir /shared/tracking/logs/<contextual-name>/grad_cos \
        --n-batches 20 --batch-size 2048
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import tempfile
import torch
import yaml
from lightning.pytorch.cli import LightningCLI

from hepattn.experiments.colliderml_regr.data import ColliderMLRegrDataModule
from hepattn.experiments.colliderml_regr.model import TrackRegressionWrapper


PARAMS = ["d0", "z0", "phi", "theta", "qop"]


def _build_model_and_datamodule(config_path: Path, data_dir: str, batch_size: int):
    """Build the Lightning module + datamodule via LightningCLI(run=False).

    The saved LightningCLI config has a number of top-level keys that the
    ``run=False`` parser does not accept (``ckpt_path``, a populated
    ``trainer.callbacks`` list that wants to resolve relative paths, etc).
    We drop/neutralise them before handing off.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw.pop("ckpt_path", None)
    # Keep only model + data; neutralise trainer so no callbacks/loggers load.
    raw["trainer"] = {"devices": 1, "logger": False, "callbacks": None,
                      "accelerator": "auto"}
    raw.get("data", {}).update({
        "preprocessed_dir": str(data_dir),
        "batch_size": int(batch_size),
        "num_workers": 0,
    })
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f)
        tmp_cfg = f.name

    old_argv = sys.argv
    sys.argv = ["gradient_cosine_analysis", "--config", tmp_cfg]
    try:
        cli = LightningCLI(
            model_class=TrackRegressionWrapper,
            datamodule_class=ColliderMLRegrDataModule,
            seed_everything_default=42,
            run=False,
            save_config_callback=None,
        )
    finally:
        sys.argv = old_argv
    return cli.model, cli.datamodule


def _flatten(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.detach().reshape(-1).float() for g in grads.values()])


def _compute_param_grads(
    wrapper: TrackRegressionWrapper,
    trunk_params: dict[str, torch.nn.Parameter],
    inputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return flattened trunk-gradients for each per-param loss in PARAMS."""
    # Single forward; reuse the graph for 5 targeted backward passes.
    wrapper.zero_grad(set_to_none=True)
    outputs = wrapper.model(inputs)
    valid_mask = targets.get("track_valid")
    losses = wrapper.model.compute_loss(outputs, targets, valid_mask=valid_mask)

    grads_out: dict[str, torch.Tensor] = {}
    param_list = list(trunk_params.values())
    for i, name in enumerate(PARAMS):
        retain = i < len(PARAMS) - 1
        g = torch.autograd.grad(
            losses[name],
            param_list,
            retain_graph=retain,
            allow_unused=True,
        )
        # allow_unused=True can return None for params that didn't touch the loss;
        # replace with zeros of the correct shape so cosine is well-defined.
        flat = torch.cat([
            (gi if gi is not None else torch.zeros_like(p))
                .detach().reshape(-1).float()
            for gi, p in zip(g, param_list, strict=True)
        ])
        grads_out[name] = flat.cpu()
    return grads_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="val",
                    help="Which dataloader to pull batches from")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] config = {args.config}")
    print(f"[setup] ckpt   = {args.ckpt}")
    print(f"[setup] data   = {args.data_dir}")

    wrapper, dm = _build_model_and_datamodule(args.config, args.data_dir, args.batch_size)

    # Load weights (strict=False to tolerate mismatched keys across optimizer changes).
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    print(f"[ckpt] missing={len(missing)}, unexpected={len(unexpected)}")

    wrapper.to(args.device).eval()
    # We need grads even in eval; no dropout either way but keep it explicit.
    for p in wrapper.parameters():
        p.requires_grad_(False)

    # Trunk parameters: encoder + pool/projection heads, but NOT the final
    # output_head (it's the merge point, conflict there is irrelevant because
    # each param lives in its own output slice).
    trunk: dict[str, torch.nn.Parameter] = {}
    for pname, p in wrapper.model.named_parameters():
        if pname.startswith("output_head"):
            continue
        trunk[pname] = p
        p.requires_grad_(True)
    print(f"[trunk] n_params={sum(p.numel() for p in trunk.values()):,}  "
          f"n_tensors={len(trunk)}")

    # Datamodule setup
    dm.setup(stage="fit")
    if args.split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.train_dataloader()

    # Collect per-batch grads
    all_grads: list[dict[str, torch.Tensor]] = []
    batch_iter = iter(loader)
    for i in range(args.n_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        inputs, targets = batch
        inputs = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        targets = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in targets.items()}
        grads = _compute_param_grads(wrapper, trunk, inputs, targets)
        all_grads.append(grads)
        print(f"[batch {i+1}/{args.n_batches}] grad norms: " +
              "  ".join(f"{n}={grads[n].norm().item():.3e}" for n in PARAMS))

    # Stack to (N_batches, 5, D)
    G = torch.stack([torch.stack([b[n] for n in PARAMS]) for b in all_grads])
    print(f"[stack] G shape = {tuple(G.shape)}")

    # Within-batch pairwise cosine similarity: (N_batches, 5, 5)
    Gn = G / (G.norm(dim=-1, keepdim=True) + 1e-30)
    cos_per_batch = torch.einsum("bpd,bqd->bpq", Gn, Gn).numpy()  # (N, 5, 5)
    cos_mean = cos_per_batch.mean(axis=0)
    cos_std = cos_per_batch.std(axis=0)

    # Save raw values
    np.savez(
        args.output_dir / "grad_cosines.npz",
        cos_per_batch=cos_per_batch,
        grad_norms=G.norm(dim=-1).numpy(),
        params=np.array(PARAMS),
    )

    # ---------- Plot 1: mean cosine heatmap ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, mat, title in ((axes[0], cos_mean, "Mean cosine similarity"),
                           (axes[1], cos_std, "Std across batches")):
        im = ax.imshow(mat, vmin=-1 if title.startswith("Mean") else 0, vmax=1,
                       cmap="RdBu_r" if title.startswith("Mean") else "viridis")
        ax.set_xticks(range(len(PARAMS)), PARAMS)
        ax.set_yticks(range(len(PARAMS)), PARAMS)
        ax.set_title(title)
        for i in range(len(PARAMS)):
            for j in range(len(PARAMS)):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        color="black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Per-parameter loss-gradient cosine on shared trunk\n"
        f"({G.shape[0]} minibatches × BS={args.batch_size}, "
        f"trunk={sum(p.numel() for p in trunk.values()):,} params)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "cos_heatmap.png", dpi=150)
    plt.close(fig)

    # ---------- Plot 2: histogram of d0-vs-others cosine across batches ----------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, name in enumerate(PARAMS):
        if name == "d0":
            continue
        vals = cos_per_batch[:, 0, j]  # d0 is index 0
        ax.hist(vals, bins=max(5, args.n_batches // 2), alpha=0.55,
                label=f"d0 ↔ {name}  (μ={vals.mean():+.2f}, σ={vals.std():.2f})")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("cosine similarity (d0-DFL loss grad vs regression loss grad)")
    ax.set_ylabel("#batches")
    ax.set_title("Does the binned-DFL d0 head fight the regression heads?")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(args.output_dir / "cos_d0_vs_others_hist.png", dpi=150)
    plt.close(fig)

    # ---------- Plot 3: all off-diagonal pairs, boxplot ----------
    pairs = []
    labels = []
    for i in range(len(PARAMS)):
        for j in range(i + 1, len(PARAMS)):
            pairs.append(cos_per_batch[:, i, j])
            labels.append(f"{PARAMS[i]}↔{PARAMS[j]}")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot(pairs, labels=labels, showmeans=True)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(-1, 1)
    ax.set_title(
        "Pairwise trunk-gradient cosine across minibatches "
        "(>0: constructive, <0: conflicting)"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "cos_all_pairs_boxplot.png", dpi=150)
    plt.close(fig)

    # ---------- Text summary ----------
    with open(args.output_dir / "summary.txt", "w") as f:
        f.write(f"Run config : {args.config}\n")
        f.write(f"Checkpoint : {args.ckpt}\n")
        f.write(f"Data       : {args.data_dir}\n")
        f.write(f"N batches  : {G.shape[0]}  BS={args.batch_size}\n")
        f.write(f"Trunk params included: {sum(p.numel() for p in trunk.values()):,}\n\n")
        f.write("Mean cosine matrix:\n")
        f.write("        " + "  ".join(f"{p:>7s}" for p in PARAMS) + "\n")
        for i, p in enumerate(PARAMS):
            f.write(f"{p:>7s} " +
                    "  ".join(f"{cos_mean[i, j]:+7.3f}" for j in range(len(PARAMS))) +
                    "\n")
        f.write("\nStd across batches:\n")
        for i, p in enumerate(PARAMS):
            f.write(f"{p:>7s} " +
                    "  ".join(f"{cos_std[i, j]:7.3f}" for j in range(len(PARAMS))) +
                    "\n")
        # Rule-of-thumb interpretation
        d0_vs_reg = cos_mean[0, 1:]  # d0 vs z0, phi, theta, qop
        f.write(f"\nd0 vs regression heads (mean cos): "
                f"{dict(zip(PARAMS[1:], d0_vs_reg.round(3)))}\n")
        if (d0_vs_reg < 0).any():
            f.write("=> at least one head is ANTI-ALIGNED with d0 -> gradient conflict present.\n")
        elif (d0_vs_reg < 0.1).all():
            f.write("=> d0 is near-orthogonal to regression heads (no strong conflict nor synergy).\n")
        else:
            f.write("=> d0 broadly aligned with regression heads; no classification-vs-regression conflict evident.\n")

    print(f"[done] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
