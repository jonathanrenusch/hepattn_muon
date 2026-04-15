"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

Minimal implementation following Foret et al., "Sharpness-Aware Minimization
for Efficiently Improving Generalization" (ICLR 2021), and the reference
code at https://github.com/davda54/sam .

Usage (Lightning manual optimization, two forward/backward passes per step):

    opt = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-4)

    # forward 1 — ascent direction from full loss
    loss = criterion(model(x), y)
    loss.backward()
    opt.first_step(zero_grad=True)   # theta -> theta + rho * g / ||g||

    # forward 2 — descent gradient at perturbed weights
    loss2 = criterion(model(x), y)
    loss2.backward()
    opt.second_step(zero_grad=True)  # restore theta, base_optimizer.step()

Incompatibilities to be aware of
--------------------------------
* ``precision: 16-mixed`` (fp16 + GradScaler) is not supported by this class.
  fp32 and bf16-mixed are fine.
* Gradient accumulation ``> 1`` is not supported: each SAM step expects
  freshly-computed grads, not accumulated ones.  The LightningModule using
  this optimizer should assert ``accumulate_grad_batches == 1``.
* Under DDP, the grad norm in ``_grad_norm`` is computed locally per rank.
  This matches the reference SAM implementation and the literature; no
  cross-rank all-reduce of the norm is needed.
"""

from __future__ import annotations

from typing import Any, Type

import torch
from torch.optim.optimizer import Optimizer


class SAM(Optimizer):
    """SAM wrapper around a base optimizer.

    Parameters
    ----------
    params : iterable
        Parameters to optimize (same as any ``torch.optim.Optimizer``).
    base_optimizer : type[Optimizer]
        The inner optimizer class (e.g. ``torch.optim.AdamW``).
        NOT an instance — SAM constructs it internally from the same
        parameter groups so that LR schedulers and other code that touch
        ``opt.param_groups`` transparently drive the underlying optimizer.
    rho : float
        Neighborhood radius.  Reference default is ``0.05``.
    adaptive : bool
        If True, use ASAM (per-parameter adaptive ε scaling).  Default False.
    **base_kwargs
        Forwarded to the base optimizer's ``__init__`` (lr, weight_decay,
        betas, ...).
    """

    def __init__(
        self,
        params,
        base_optimizer: Type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **base_kwargs: Any,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho (must be non-negative): {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **base_kwargs)
        super().__init__(params, defaults)

        # Build the base optimizer on the SAME param_groups so LR schedulers
        # modifying ``self.param_groups`` transparently flow through.
        self.base_optimizer = base_optimizer(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        # Mirror the base defaults so external code sees a consistent view.
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb weights in the ascent direction of the current gradients.

        Saves the per-parameter perturbation ``e_w`` into ``self.state[p]``
        so that :meth:`second_step` can restore the original weights.  Any
        pre-existing ``e_w`` entries are cleared first to guarantee no
        drift across steps if some parameter's grad transiently becomes
        ``None``.
        """
        # Clear any stale e_w from a previous step.
        for state in self.state.values():
            state.pop("e_w", None)

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Adaptive SAM multiplies by |θ| so the perturbation is
                # element-wise scale-invariant (Kwon et al. 2021).
                if group["adaptive"]:
                    e_w = (torch.abs(p) * p.grad) * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore the original weights and take a base-optimizer step.

        Iterates over params that **have** a stored ``e_w`` (not over params
        whose ``.grad`` is currently non-None) — this is deliberate: a
        parameter may have received a grad in the first pass but not the
        second (e.g. dropout path differences), and we still need to roll
        back its ascent perturbation.
        """
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, None)
                if state is None or "e_w" not in state:
                    continue
                p.sub_(state["e_w"])
                # Drop the e_w reference so memory is released promptly.
                del state["e_w"]

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _grad_norm(self) -> torch.Tensor:
        """L2 norm of the current gradients across all parameter groups.

        Computed locally per-rank under DDP — this matches the reference
        SAM implementation.  ``shared_device`` guards against parameter
        groups living on different devices (rare, but safe).
        """
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["adaptive"]:
                    term = (torch.abs(p) * p.grad).norm(p=2)
                else:
                    term = p.grad.norm(p=2)
                norms.append(term.to(shared_device))
        if not norms:
            return torch.zeros((), device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """Fall through to the base optimizer's step.

        During SAM-active epochs the caller uses ``first_step`` /
        ``second_step`` explicitly and never reaches here.  During warmup
        epochs (before ``sam_start_epoch``) the manual training loop calls
        ``opt.step()`` for a normal base-optimizer update — Lightning routes
        that through this method, so we forward to the base optimizer.
        """
        self.base_optimizer.step(closure=closure)

    def load_state_dict(self, state_dict):  # type: ignore[override]
        super().load_state_dict(state_dict)
        # Keep the base optimizer's param_groups alias in sync with ours
        # after a state dict reload.
        self.base_optimizer.param_groups = self.param_groups
