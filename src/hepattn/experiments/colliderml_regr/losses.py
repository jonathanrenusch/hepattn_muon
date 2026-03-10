"""Config-driven loss functions for track parameter regression.

Every parameter's loss behaviour is determined *entirely* by the YAML config.
The :class:`TrackParameterLoss` module reads the ``losses`` section of the
config and instantiates the appropriate per-parameter loss object.

Supported loss types
--------------------
``smooth_l1``
    Direct Smooth-L1 on (optionally pre-normalised) targets.
``spline_l1``
    Smooth-L1 in quantile-spline-normalised ``[0, 1]`` space.
``quantile``
    Pinball (quantile) loss directly on physical / normalised targets.
``spline_quantile``
    Pinball loss in spline-normalised space.
``circular``
    ``SmoothL1(sin) + SmoothL1(cos)`` for angular parameters (phi).

All individual components expose ``.forward(pred, target) → loss``
and ``.predict(raw_output) → physical_value``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from hepattn.experiments.colliderml_regr.spline import MonotonicSplineTransform


# ============================================================================
# Per-parameter loss components
# ============================================================================


class SmoothL1Loss(nn.Module):
    """Direct Smooth-L1 on normalised targets.

    The model predicts a single scalar per parameter.  Targets are linearly
    normalised to ``[-1, 1]`` using the configured ``norm_min`` / ``norm_max``
    range before computing the loss.

    Parameters
    ----------
    norm_min : float
        Lower bound of the physical range for linear normalisation.
    norm_max : float
        Upper bound of the physical range for linear normalisation.
    weight : float
        Multiplicative weight applied to this parameter's loss.
    beta : float
        Smooth-L1 transition point.
    """

    num_outputs: int = 1

    def __init__(
        self,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    def _normalise(self, x: Tensor) -> Tensor:
        """Map physical value to [-1, 1]."""
        return 2.0 * (x - self.norm_min) / (self.norm_max - self.norm_min) - 1.0

    def _denormalise(self, u: Tensor) -> Tensor:
        """Map [-1, 1] back to physical value."""
        return (u + 1.0) / 2.0 * (self.norm_max - self.norm_min) + self.norm_min

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute loss.  Both tensors have shape ``(N,)`` or ``(N, 1)``."""
        t_norm = self._normalise(target)
        p = pred.squeeze(-1) if pred.dim() > target.dim() else pred
        return self.weight * F.smooth_l1_loss(p, t_norm, beta=self.beta)

    def predict(self, raw: Tensor) -> Tensor:
        """Convert raw model output to physical value."""
        return self._denormalise(raw.squeeze(-1))


class SplineL1Loss(nn.Module):
    """Smooth-L1 in spline-normalised ``[0, 1]`` space.

    Parameters
    ----------
    spline_config : str | dict
        Path to a YAML spline config or an inline dict.
    weight : float
        Loss weight.
    beta : float
        Smooth-L1 transition point.
    """

    num_outputs: int = 1

    def __init__(
        self,
        spline_config: str | dict[str, Any],
        weight: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.weight = weight
        self.beta = beta

        if isinstance(spline_config, (str, Path)):
            self.spline = MonotonicSplineTransform.from_config(spline_config)
        else:
            self.spline = MonotonicSplineTransform.from_dict(spline_config)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        u_target = self.spline.forward(target)
        p = pred.squeeze(-1) if pred.dim() > target.dim() else pred
        # Model predicts in [0, 1] spline space; apply sigmoid to ensure range
        p_clamped = torch.sigmoid(p)
        return self.weight * F.smooth_l1_loss(p_clamped, u_target, beta=self.beta)

    def predict(self, raw: Tensor) -> Tensor:
        return self.spline.inverse(torch.sigmoid(raw.squeeze(-1)))


class QuantileLoss(nn.Module):
    """Pinball (quantile / check) loss on normalised physical values.

    The model outputs one value per quantile.  The median (τ=0.5) serves
    as the point prediction.

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels, e.g. ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    norm_min : float
        Lower bound for linear normalisation.
    norm_max : float
        Upper bound for linear normalisation.
    weight : float
        Loss weight.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.weight = weight
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def _normalise(self, x: Tensor) -> Tensor:
        return 2.0 * (x - self.norm_min) / (self.norm_max - self.norm_min) - 1.0

    def _denormalise(self, u: Tensor) -> Tensor:
        return (u + 1.0) / 2.0 * (self.norm_max - self.norm_min) + self.norm_min

    @staticmethod
    def _pinball(pred: Tensor, target: Tensor, tau: Tensor) -> Tensor:
        """Pinball loss: τ * max(0, target-pred) + (1-τ) * max(0, pred-target)."""
        diff = target - pred
        return torch.mean(torch.max(tau * diff, (tau - 1) * diff))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred: (N, num_quantiles),  target: (N,)."""
        t_norm = self._normalise(target).unsqueeze(-1)  # (N, 1)
        tau = self.quantiles.unsqueeze(0)  # (1, Q)
        diff = t_norm - pred  # (N, Q)
        loss = torch.mean(torch.max(tau * diff, (tau - 1) * diff))
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Return median quantile mapped back to physical space."""
        median_idx = (self.quantiles - 0.5).abs().argmin()
        return self._denormalise(raw[..., median_idx])


class SplineQuantileLoss(nn.Module):
    """Pinball (quantile) loss in spline-normalised ``[0, 1]`` space.

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels.
    spline_config : str | dict
        Spline configuration.
    weight : float
        Loss weight.
    """

    def __init__(
        self,
        spline_config: str | dict[str, Any],
        quantiles: list[float] | None = None,
        weight: float = 1.0,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.weight = weight
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

        if isinstance(spline_config, (str, Path)):
            self.spline = MonotonicSplineTransform.from_config(spline_config)
        else:
            self.spline = MonotonicSplineTransform.from_dict(spline_config)

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        u_target = self.spline.forward(target).unsqueeze(-1)  # (N, 1)
        # Model predicts in logit space; sigmoid maps to [0, 1] spline space
        p = torch.sigmoid(pred)  # (N, Q)
        tau = self.quantiles.unsqueeze(0)  # (1, Q)
        diff = u_target - p  # (N, Q)
        loss = torch.mean(torch.max(tau * diff, (tau - 1) * diff))
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        median_idx = (self.quantiles - 0.5).abs().argmin()
        return self.spline.inverse(torch.sigmoid(raw[..., median_idx]))


class CircularPhiLoss(nn.Module):
    """Smooth-L1 on sin/cos components for circular angular regression.

    The model outputs two values ``(sin_pred, cos_pred)``.  Loss is::

        SmoothL1(sin_pred - sin(phi_true)) + SmoothL1(cos_pred - cos(phi_true))

    Recovery: ``phi = atan2(sin_pred, cos_pred)``.

    Parameters
    ----------
    weight : float
        Loss weight.
    beta : float
        Smooth-L1 transition point.
    """

    num_outputs: int = 2

    def __init__(self, weight: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.weight = weight
        self.beta = beta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """pred: (N, 2) with [sin, cos];  target: (N,) with phi in radians."""
        sin_true = torch.sin(target)
        cos_true = torch.cos(target)
        sin_pred = pred[..., 0]
        cos_pred = pred[..., 1]
        loss_sin = F.smooth_l1_loss(sin_pred, sin_true, beta=self.beta)
        loss_cos = F.smooth_l1_loss(cos_pred, cos_true, beta=self.beta)
        return self.weight * (loss_sin + loss_cos)

    def predict(self, raw: Tensor) -> Tensor:
        """Recover phi from (sin, cos) outputs."""
        return torch.atan2(raw[..., 0], raw[..., 1])


# ============================================================================
# Registry
# ============================================================================

LOSS_REGISTRY: dict[str, type] = {
    "smooth_l1": SmoothL1Loss,
    "spline_l1": SplineL1Loss,
    "quantile": QuantileLoss,
    "spline_quantile": SplineQuantileLoss,
    "circular": CircularPhiLoss,
}


# ============================================================================
# Composite loss over all five track parameters
# ============================================================================


class TrackParameterLoss(nn.Module):
    """Config-driven composite loss for the five perigee track parameters.

    Instantiated entirely from a ``losses`` dict in the YAML config::

        losses:
          d0:
            type: smooth_l1
            weight: 1.0
            norm_min: -2.0
            norm_max: 2.0
          phi:
            type: circular
            weight: 1.0
          ...

    The ``parameters`` ordering determines the slice of the model's output
    that each sub-loss reads.

    Parameters
    ----------
    config : dict[str, dict]
        Per-parameter loss config keyed by parameter name.
        Each value is a dict with at least ``type`` plus any kwargs
        for that loss class.
    parameter_order : list[str]
        Canonical ordering of the five parameters.
    """

    def __init__(
        self,
        config: dict[str, dict[str, Any]],
        parameter_order: list[str] | None = None,
    ):
        super().__init__()

        if parameter_order is None:
            parameter_order = ["d0", "z0", "phi", "theta", "qop"]

        self.parameter_order = parameter_order
        self.losses = nn.ModuleDict()

        for name in parameter_order:
            assert name in config, f"Missing loss config for parameter '{name}'"
            cfg = dict(config[name])  # shallow copy to pop from
            loss_type = cfg.pop("type")
            assert loss_type in LOSS_REGISTRY, (
                f"Unknown loss type '{loss_type}' for parameter '{name}'. "
                f"Available: {list(LOSS_REGISTRY.keys())}"
            )
            self.losses[name] = LOSS_REGISTRY[loss_type](**cfg)

        # Compute output slice boundaries
        self._output_slices: dict[str, tuple[int, int]] = {}
        offset = 0
        for name in parameter_order:
            n = self.losses[name].num_outputs
            self._output_slices[name] = (offset, offset + n)
            offset += n

        self._total_outputs = offset

    @property
    def total_outputs(self) -> int:
        """Total number of raw regression outputs the model must produce."""
        return self._total_outputs

    def get_output_slice(self, name: str) -> tuple[int, int]:
        """Return (start, end) indices into the model's output vector."""
        return self._output_slices[name]

    def forward(
        self,
        pred: Tensor,
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute per-parameter losses.

        Parameters
        ----------
        pred : Tensor
            Model output of shape ``(B, total_outputs)`` or
            ``(N, total_outputs)`` after masking.
        targets : dict[str, Tensor]
            Per-parameter target tensors, each ``(B,)`` or ``(N,)``.
        valid_mask : Tensor | None
            Boolean mask ``(B,)`` selecting valid tracks.  If given,
            both ``pred`` and ``targets`` are masked before computing losses.

        Returns
        -------
        dict[str, Tensor]
            Per-parameter scalar loss values, plus a ``"total"`` entry.
        """
        losses: dict[str, Tensor] = {}
        device = pred.device
        total = torch.tensor(0.0, device=device)

        for name in self.parameter_order:
            start, end = self._output_slices[name]
            p = pred[..., start:end] if end - start > 1 else pred[..., start]
            t = targets[name]

            if valid_mask is not None:
                p = p[valid_mask]
                t = t[valid_mask]

            if t.numel() == 0:
                losses[name] = torch.tensor(0.0, device=device)
                continue

            loss = self.losses[name](p, t)
            losses[name] = loss
            total = total + loss

        losses["total"] = total
        return losses

    def predict(
        self,
        pred: Tensor,
    ) -> dict[str, Tensor]:
        """Convert raw model outputs to physical predictions.

        Parameters
        ----------
        pred : Tensor
            Raw model output ``(B, total_outputs)`` or ``(N, total_outputs)``.

        Returns
        -------
        dict[str, Tensor]
            Per-parameter physical predictions, each ``(B,)`` or ``(N,)``.
        """
        preds: dict[str, Tensor] = {}
        for name in self.parameter_order:
            start, end = self._output_slices[name]
            raw = pred[..., start:end] if end - start > 1 else pred[..., start:start + 1]
            preds[name] = self.losses[name].predict(raw)
        return preds

    def predict_quantiles(
        self,
        pred: Tensor,
    ) -> dict[str, Tensor]:
        """Return full quantile predictions (for quantile-based losses only).

        For non-quantile parameters the single point prediction is returned.
        """
        preds: dict[str, Tensor] = {}
        for name in self.parameter_order:
            start, end = self._output_slices[name]
            raw = pred[..., start:end]
            loss_fn = self.losses[name]
            if isinstance(loss_fn, (QuantileLoss, SplineQuantileLoss)):
                # Return all quantile predictions in physical space
                if isinstance(loss_fn, SplineQuantileLoss):
                    preds[name] = loss_fn.spline.inverse(torch.sigmoid(raw))
                else:
                    preds[name] = loss_fn._denormalise(raw)
            else:
                preds[name] = loss_fn.predict(raw)
        return preds
