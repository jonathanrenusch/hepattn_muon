"""Per-head trunk-gradient norm logger.

Motivates: the standard :class:`GradientLoggerCallback` logs per-module
gradient norms *after* the combined loss has backward'd, which cannot
separate the contribution of each per-parameter loss on the **shared
trunk** (encoder + shared pool_head).  When d0 uses a DFL classification
loss and the others use continuous quantile/circular losses, the
classification gradient can dominate the trunk by 10–150× while module-
level norms only reveal the branch-specific contributions.

This callback runs every ``probe_every_n_steps`` training steps.  It
fetches one batch, does a single forward pass, and calls
``torch.autograd.grad`` once per per-parameter loss to obtain each head's
contribution to the trunk L2 gradient norm — then logs:

- ``grad_probe/per_head/<param>/trunk_norm`` for each param in
  ``parameter_order``.
- ``grad_probe/per_head/ratio_max`` — the highest head's norm divided
  by the geometric mean of the others.  A clean "is d0 still dominating"
  number.

Cost: one extra forward + k backward-on-scalar-loss calls per invocation.
With the default ``probe_every_n_steps=500`` and five heads on a BS=2048
batch, measured overhead is ~0.8 s per probe — well below 1 % of step
time.

The callback intentionally uses the *current* mini-batch draining the
train dataloader, so probes reflect the live training distribution at
that point of the schedule.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from lightning import Callback, LightningModule, Trainer


class PerHeadTrunkGradLogger(Callback):
    """Periodically log per-head L2 gradient norms on the shared trunk.

    Parameters
    ----------
    probe_every_n_steps
        Run a probe this often.  0 disables.
    include_pool_head
        If True, include the shared ``pool_head`` parameters in the
        "trunk" definition.  Default True — the pool_head is shared by
        the regression heads and is exactly the projection the separate-
        branch architecture isolates d0 from.
    include_output_head
        If True, include the main ``output_head`` parameters too.  Default
        False — output_head slices are per-parameter by construction, so
        conflict there is uninformative.
    """

    def __init__(
        self,
        probe_every_n_steps: int = 500,
        include_pool_head: bool = True,
        include_output_head: bool = False,
    ):
        self.probe_every_n_steps = int(probe_every_n_steps)
        self.include_pool_head = bool(include_pool_head)
        self.include_output_head = bool(include_output_head)
        self._sync_dist = False

    # ------------------------------------------------------------------

    def setup(self, trainer: Trainer, module: LightningModule, stage: str) -> None:
        if trainer.fast_dev_run or stage != "fit":
            return
        self._sync_dist = len(trainer.device_ids) > 1

    # ------------------------------------------------------------------

    def _trunk_params(self, model: torch.nn.Module) -> list[torch.nn.Parameter]:
        """Collect shared-trunk parameters from the inner regressor."""
        out: list[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("encoder."):
                out.append(p); continue
            if name.startswith("input_net."):
                out.append(p); continue
            if self.include_pool_head and name.startswith("pool_head."):
                out.append(p); continue
            if self.include_output_head and name.startswith("output_head."):
                out.append(p); continue
            if name.startswith(("fwd_head.", "bwd_head.")):
                out.append(p); continue
            # d0_pool_head / d0_output_head are branch-specific and
            # excluded from "trunk" by definition.
        return out

    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self.probe_every_n_steps <= 0:
            return
        if trainer.global_step == 0:
            return
        if trainer.global_step % self.probe_every_n_steps != 0:
            return

        # Re-use the last training batch from args; Lightning passes it in.
        inputs, targets = batch
        inner = pl_module.model  # TrackParameterRegressor
        trunk_params = self._trunk_params(inner)
        if not trunk_params:
            return

        loss_module = inner.loss_module
        param_order: Iterable[str] = loss_module.parameter_order

        # Snapshot training state; restore at the end so we do not
        # interfere with Lightning's step bookkeeping.
        was_training = pl_module.training
        grads_that_need_none = [p for p in pl_module.parameters()]
        saved_grads = {id(p): p.grad for p in grads_that_need_none}

        try:
            pl_module.eval()  # disable dropout during the probe
            with torch.enable_grad():
                outputs_fwd = inner(inputs)
                losses = inner.compute_loss(
                    outputs_fwd, targets,
                    valid_mask=targets.get("track_valid"),
                )

            norms: dict[str, float] = {}
            param_list = list(trunk_params)
            n = len(param_order)
            for i, name in enumerate(param_order):
                retain = i < n - 1
                g = torch.autograd.grad(
                    losses[name], param_list,
                    retain_graph=retain, allow_unused=True,
                )
                sq = 0.0
                for gi, p in zip(g, param_list, strict=True):
                    if gi is None:
                        continue
                    sq += float(gi.detach().norm(2).item() ** 2)
                norms[name] = math.sqrt(sq)
        except Exception as e:
            # Never kill training because of the probe.  Log and move on.
            pl_module.print(f"[per-head probe] skipped: {e!r}")
            return
        finally:
            for p in grads_that_need_none:
                p.grad = saved_grads.get(id(p))
            if was_training:
                pl_module.train()

        # Log per-head trunk norms
        for name, v in norms.items():
            pl_module.log(
                f"grad_probe/{name}/trunk_norm", v,
                on_step=True, on_epoch=False, logger=True,
                sync_dist=self._sync_dist,
            )

        # Dominance ratio: max-head / geometric-mean-of-the-rest.
        #   >1 means one head dominates the trunk.  =1 is parity.
        sorted_items = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)
        if len(sorted_items) >= 2:
            top_name, top_v = sorted_items[0]
            rest = [v for _, v in sorted_items[1:] if v > 0]
            if rest:
                gm = math.exp(sum(math.log(v) for v in rest) / len(rest))
                pl_module.log(
                    "grad_probe/ratio_max_over_geomean_rest", top_v / gm,
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )
                pl_module.log(
                    f"grad_probe/dominant_head/{top_name}", 1.0,
                    on_step=True, on_epoch=False, logger=True,
                    sync_dist=self._sync_dist,
                )
