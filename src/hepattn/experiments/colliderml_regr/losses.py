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

# Directory containing this file — used to resolve relative spline config paths
_EXPERIMENT_DIR = Path(__file__).resolve().parent


def _resolve_spline_path(spline_config: str | Path) -> Path:
    """Resolve a spline config path relative to the experiment directory."""
    p = Path(spline_config)
    if p.is_absolute():
        return p
    # Try relative to experiment dir first (handles 'config/splines/...')
    resolved = _EXPERIMENT_DIR / p
    if resolved.exists():
        return resolved
    # Fall back to CWD-relative (original behaviour)
    return p


# ============================================================================
# Shared normalisation helpers
# ============================================================================


def _linear_normalise(x: Tensor, norm_min: Tensor, norm_max: Tensor) -> Tensor:
    """Map physical value to [-1, 1]."""
    return 2.0 * (x - norm_min) / (norm_max - norm_min) - 1.0


def _linear_denormalise(u: Tensor, norm_min: Tensor, norm_max: Tensor) -> Tensor:
    """Map [-1, 1] back to physical value."""
    return (u + 1.0) / 2.0 * (norm_max - norm_min) + norm_min


def _validate_quantiles(quantiles: list[float]) -> None:
    """Validate quantile levels are inside (0, 1) and strictly increasing."""
    if len(quantiles) == 0:
        raise ValueError("quantiles must contain at least one value")
    if any((q <= 0.0 or q >= 1.0) for q in quantiles):
        raise ValueError(f"quantiles must be in (0, 1), got: {quantiles}")
    if any(q2 <= q1 for q1, q2 in zip(quantiles[:-1], quantiles[1:], strict=False)):
        raise ValueError(f"quantiles must be strictly increasing, got: {quantiles}")


def _raw_crossing_stats(raw: Tensor) -> dict[str, Tensor]:
    """Compute crossing stats from raw quantile channels.

    NOTE: These statistics are computed on the **raw unconstrained** model
    outputs (base + delta channels), NOT on the ordered quantile predictions.
    The actual quantile predictions never cross thanks to the softplus + cumsum
    construction.  A high crossing rate here simply means the raw network
    outputs do not naturally maintain ordering — which is expected and harmless.

    Crossing is measured on adjacent raw channels as
    ``max(raw_i - raw_{i+1}, 0)``.
    """
    if raw.numel() == 0 or raw.shape[-1] < 2:
        z = raw.new_tensor(0.0)
        return {"rate": z, "mean_gap": z, "max_gap": z}

    gaps = (raw[..., :-1] - raw[..., 1:]).clamp_min(0.0)
    # Fraction of individual adjacent pairs that violate ordering
    rate = (gaps > 0).to(dtype=raw.dtype).mean()
    return {
        "rate": rate,
        "mean_gap": gaps.mean(),
        "max_gap": gaps.max(),
    }


def _quantile_calibration(
    ordered_quantiles: Tensor,
    target: Tensor,
    quantile_levels: Tensor,
) -> dict[str, Tensor]:
    """Compute quantile calibration: empirical coverage vs nominal levels.

    For a well-calibrated model, the fraction of targets below the τ-th
    predicted quantile should equal τ.

    Parameters
    ----------
    ordered_quantiles : Tensor
        Ordered quantile predictions ``(N, Q)`` in physical space.
    target : Tensor
        Ground truth values ``(N,)``.
    quantile_levels : Tensor
        Nominal quantile levels ``(Q,)``, e.g. [0.05, 0.1, 0.25, 0.5, ...].

    Returns
    -------
    dict[str, Tensor]
        ``calibration_error``: mean |empirical_coverage - nominal| across quantiles.
    """
    if target.numel() == 0 or ordered_quantiles.numel() == 0:
        z = target.new_tensor(0.0)
        return {"calibration_error": z}

    # (N, 1) < (N, Q) → (N, Q)
    below = (target.unsqueeze(-1) < ordered_quantiles).float()
    empirical_coverage = below.mean(dim=0)  # (Q,)
    calibration_error = (empirical_coverage - quantile_levels).abs().mean()
    return {"calibration_error": calibration_error}


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

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """Compute loss.  Both tensors have shape ``(N,)`` or ``(N, 1)``."""
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max)
        p = pred.squeeze(-1) if pred.dim() > target.dim() else pred
        per_sample = F.smooth_l1_loss(p, t_norm, beta=self.beta, reduction="none")
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Convert raw model output to physical value."""
        return _linear_denormalise(raw.squeeze(-1), self.norm_min, self.norm_max)


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
            self.spline = MonotonicSplineTransform.from_config(_resolve_spline_path(spline_config))
        else:
            self.spline = MonotonicSplineTransform.from_dict(spline_config)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        u_target = self.spline.forward(target)
        p = pred.squeeze(-1) if pred.dim() > target.dim() else pred
        # Model predicts in [0, 1] spline space; apply sigmoid to ensure range
        p_clamped = torch.sigmoid(p)
        per_sample = F.smooth_l1_loss(p_clamped, u_target, beta=self.beta, reduction="none")
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

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
        monotone_eps: float = 1.0e-6,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        _validate_quantiles(quantiles)
        self.weight = weight
        self.monotone_eps = monotone_eps
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def _ordered_from_raw(self, raw: Tensor) -> Tensor:
        """Map raw channels to strictly ordered quantile values."""
        if raw.shape[-1] < 2:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + self.monotone_eps
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, num_quantiles),  target: (N,)."""
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max).unsqueeze(-1)  # (N, 1)
        p = self._ordered_from_raw(pred)
        tau = self.quantiles.unsqueeze(0)  # (1, Q)
        diff = t_norm - p  # (N, Q)
        per_sample = torch.max(tau * diff, (tau - 1) * diff).mean(dim=-1)  # (N,)
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Return median quantile mapped back to physical space."""
        ordered = self._ordered_from_raw(raw)
        median_idx = (self.quantiles - 0.5).abs().argmin()
        return _linear_denormalise(ordered[..., median_idx], self.norm_min, self.norm_max)

    def predict_quantiles(self, raw: Tensor) -> Tensor:
        """Return all ordered quantile predictions in physical space."""
        ordered = self._ordered_from_raw(raw)
        return _linear_denormalise(ordered, self.norm_min, self.norm_max)

    def raw_crossing_metrics(self, raw: Tensor) -> dict[str, Tensor]:
        """Return crossing metrics computed on the raw (unconstrained) channels."""
        return _raw_crossing_stats(raw)

    def calibration_metrics(self, raw: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Return quantile calibration metrics on physical-space predictions."""
        ordered = self._ordered_from_raw(raw)
        ordered_phys = _linear_denormalise(ordered, self.norm_min, self.norm_max)
        return _quantile_calibration(ordered_phys, target, self.quantiles)


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
        monotone_eps: float = 1.0e-6,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        _validate_quantiles(quantiles)
        self.weight = weight
        self.monotone_eps = monotone_eps
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))

        if isinstance(spline_config, (str, Path)):
            self.spline = MonotonicSplineTransform.from_config(_resolve_spline_path(spline_config))
        else:
            self.spline = MonotonicSplineTransform.from_dict(spline_config)

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def _ordered_logits_from_raw(self, raw: Tensor) -> Tensor:
        """Map raw channels to strictly ordered quantile logits."""
        if raw.shape[-1] < 2:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + self.monotone_eps
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        u_target = self.spline.forward(target).unsqueeze(-1)  # (N, 1)
        # Model predicts in logit space; sigmoid maps to [0, 1] spline space
        ordered_logits = self._ordered_logits_from_raw(pred)
        p = torch.sigmoid(ordered_logits)  # (N, Q)
        tau = self.quantiles.unsqueeze(0)  # (1, Q)
        diff = u_target - p  # (N, Q)
        per_sample = torch.max(tau * diff, (tau - 1) * diff).mean(dim=-1)  # (N,)
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        ordered_logits = self._ordered_logits_from_raw(raw)
        median_idx = (self.quantiles - 0.5).abs().argmin()
        return self.spline.inverse(torch.sigmoid(ordered_logits[..., median_idx]))

    def predict_quantiles(self, raw: Tensor) -> Tensor:
        """Return all ordered quantile predictions in physical space."""
        ordered_logits = self._ordered_logits_from_raw(raw)
        return self.spline.inverse(torch.sigmoid(ordered_logits))

    def raw_crossing_metrics(self, raw: Tensor) -> dict[str, Tensor]:
        """Return crossing metrics computed on the raw (unconstrained) channels."""
        return _raw_crossing_stats(raw)

    def calibration_metrics(self, raw: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Return quantile calibration metrics on physical-space predictions."""
        ordered_logits = self._ordered_logits_from_raw(raw)
        ordered_phys = self.spline.inverse(torch.sigmoid(ordered_logits))
        return _quantile_calibration(ordered_phys, target, self.quantiles)


def _theta_to_eta(theta: Tensor) -> Tensor:
    """Convert polar angle θ ∈ (0, π) to pseudorapidity η = -ln(tan(θ/2))."""
    half = theta.clamp(1e-7, torch.pi - 1e-7) * 0.5
    return -torch.log(torch.tan(half))


def _eta_to_theta(eta: Tensor) -> Tensor:
    """Convert pseudorapidity η back to polar angle θ = 2·arctan(exp(-η))."""
    return 2.0 * torch.atan(torch.exp(-eta))


class EtaQuantileLoss(nn.Module):
    """Quantile loss that operates in pseudorapidity (η) space.

    Targets arrive as θ ∈ (0, π), are converted to η = -ln(tan(θ/2)),
    normalised to [-1, 1] using ``norm_min``/``norm_max`` (in η units),
    and the pinball loss is computed in that space.

    **Predictions are always returned in θ space** so that downstream
    metrics (MAE, precision) remain comparable to the θ-native loss.

    Parameters
    ----------
    quantiles : list[float]
        Quantile levels, e.g. ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    norm_min : float
        Lower bound of η range for linear normalisation (e.g. -5.0).
    norm_max : float
        Upper bound of η range for linear normalisation (e.g. 5.0).
    weight : float
        Loss weight.
    monotone_eps : float
        Minimum gap between adjacent quantile predictions.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        norm_min: float = -5.0,
        norm_max: float = 5.0,
        weight: float = 1.0,
        monotone_eps: float = 1.0e-6,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        _validate_quantiles(quantiles)
        self.weight = weight
        self.monotone_eps = monotone_eps
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    @property
    def num_outputs(self) -> int:
        return len(self.quantiles)

    def _ordered_from_raw(self, raw: Tensor) -> Tensor:
        """Map raw channels to strictly ordered quantile values."""
        if raw.shape[-1] < 2:
            return raw
        base = raw[..., :1]
        deltas = F.softplus(raw[..., 1:]) + self.monotone_eps
        return torch.cat([base, base + torch.cumsum(deltas, dim=-1)], dim=-1)

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, num_quantiles),  target: (N,) in θ space."""
        eta = _theta_to_eta(target)
        t_norm = _linear_normalise(eta, self.norm_min, self.norm_max).unsqueeze(-1)  # (N, 1)
        p = self._ordered_from_raw(pred)
        tau = self.quantiles.unsqueeze(0)  # (1, Q)
        diff = t_norm - p  # (N, Q)
        per_sample = torch.max(tau * diff, (tau - 1) * diff).mean(dim=-1)  # (N,)
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Return median quantile mapped back to θ space."""
        ordered = self._ordered_from_raw(raw)
        median_idx = (self.quantiles - 0.5).abs().argmin()
        eta_pred = _linear_denormalise(ordered[..., median_idx], self.norm_min, self.norm_max)
        return _eta_to_theta(eta_pred)

    def predict_quantiles(self, raw: Tensor) -> Tensor:
        """Return all ordered quantile predictions in θ space."""
        ordered = self._ordered_from_raw(raw)
        eta_preds = _linear_denormalise(ordered, self.norm_min, self.norm_max)
        return _eta_to_theta(eta_preds)

    def raw_crossing_metrics(self, raw: Tensor) -> dict[str, Tensor]:
        """Return crossing metrics computed on the raw (unconstrained) channels."""
        return _raw_crossing_stats(raw)

    def calibration_metrics(self, raw: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Return quantile calibration metrics in θ space."""
        ordered = self._ordered_from_raw(raw)
        eta_preds = _linear_denormalise(ordered, self.norm_min, self.norm_max)
        theta_preds = _eta_to_theta(eta_preds)
        return _quantile_calibration(theta_preds, target, self.quantiles)


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

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, 2) with [sin, cos];  target: (N,) with phi in radians."""
        sin_true = torch.sin(target)
        cos_true = torch.cos(target)
        sin_pred = pred[..., 0]
        cos_pred = pred[..., 1]
        per_sample = (
            F.smooth_l1_loss(sin_pred, sin_true, beta=self.beta, reduction="none")
            + F.smooth_l1_loss(cos_pred, cos_true, beta=self.beta, reduction="none")
        )
        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

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
    "quantile_eta": EtaQuantileLoss,
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
        qop_tail_weight: float = 0.0,
        qop_scale: float = 2.0,
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
        self.qop_tail_weight = qop_tail_weight
        self.qop_scale = qop_scale

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

        # Per-track |qop| weighting — opt-in, off by default (qop_tail_weight=0)
        # Tracks with high |qop| (large curvature, significant scattering) get
        # proportionally more gradient signal.  Weights are normalised so the
        # mean weight is 1, preserving the overall loss scale.
        sample_weights: Tensor | None = None
        if self.qop_tail_weight > 0.0 and "qop" in targets:
            qop_vals = targets["qop"]
            if valid_mask is not None:
                qop_vals = qop_vals[valid_mask]
            w = 1.0 + self.qop_tail_weight * (qop_vals.abs() / self.qop_scale).clamp(max=1.0)
            sample_weights = w / w.mean()

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

            loss = self.losses[name](p, t, sample_weights=sample_weights)
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
            if isinstance(loss_fn, (QuantileLoss, EtaQuantileLoss, SplineQuantileLoss)):
                preds[name] = loss_fn.predict_quantiles(raw)
            else:
                preds[name] = loss_fn.predict(raw)
        return preds

    def quantile_crossing_metrics(
        self,
        pred: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return crossing diagnostics from raw quantile channels.

        Metrics are computed before monotone reconstruction to monitor how often
        the unconstrained model outputs violate quantile ordering.
        """
        metrics: dict[str, Tensor] = {}
        rates: list[Tensor] = []
        mean_gaps: list[Tensor] = []
        max_gaps: list[Tensor] = []

        for name in self.parameter_order:
            loss_fn = self.losses[name]
            if not isinstance(loss_fn, (QuantileLoss, EtaQuantileLoss, SplineQuantileLoss)):
                continue

            start, end = self._output_slices[name]
            raw = pred[..., start:end]
            if valid_mask is not None:
                raw = raw[valid_mask]

            stats = loss_fn.raw_crossing_metrics(raw)
            metrics[f"{name}/raw_crossing_rate"] = stats["rate"]
            metrics[f"{name}/raw_crossing_mean_gap"] = stats["mean_gap"]
            metrics[f"{name}/raw_crossing_max_gap"] = stats["max_gap"]

            rates.append(stats["rate"])
            mean_gaps.append(stats["mean_gap"])
            max_gaps.append(stats["max_gap"])

        if rates:
            metrics["quantiles/raw_crossing_rate_mean"] = torch.stack(rates).mean()
            metrics["quantiles/raw_crossing_mean_gap_mean"] = torch.stack(mean_gaps).mean()
            metrics["quantiles/raw_crossing_max_gap_max"] = torch.stack(max_gaps).max()

        return metrics

    def quantile_calibration_metrics(
        self,
        pred: Tensor,
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return quantile calibration diagnostics.

        For each quantile-based parameter, computes how well the empirical
        coverage matches the nominal quantile levels.
        """
        metrics: dict[str, Tensor] = {}
        cal_errors: list[Tensor] = []

        for name in self.parameter_order:
            loss_fn = self.losses[name]
            if not isinstance(loss_fn, (QuantileLoss, EtaQuantileLoss, SplineQuantileLoss)):
                continue

            start, end = self._output_slices[name]
            raw = pred[..., start:end]
            t = targets[name]
            if valid_mask is not None:
                raw = raw[valid_mask]
                t = t[valid_mask]

            if t.numel() == 0:
                continue

            stats = loss_fn.calibration_metrics(raw, t)
            metrics[f"{name}/quantile_calibration_error"] = stats["calibration_error"]
            cal_errors.append(stats["calibration_error"])

        if cal_errors:
            metrics["quantiles/calibration_error_mean"] = torch.stack(cal_errors).mean()

        return metrics
