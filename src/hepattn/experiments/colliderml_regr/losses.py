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
``quantile_eta``
    Pinball loss in pseudorapidity space (for θ).
``spline_quantile``
    Pinball loss in spline-normalised space.
``circular``
    ``SmoothL1(sin) + SmoothL1(cos)`` for angular parameters (phi).
``gaussian``
    Gaussian NLL on normalised targets, outputs ``(mu, log_var)``.
``gaussian_eta``
    Gaussian NLL in η-space (for θ), outputs ``(mu, log_var)``.
``cosine_phi``
    ``1 - cos(phi_pred - phi_true)`` via (sin, cos) outputs for stability.

All individual components expose ``.forward(pred, target) → loss``
and ``.predict(raw_output) → physical_value``.
"""

from __future__ import annotations

import math
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
    reduction : str
        ``"mean"`` (default) returns scalar; ``"none"`` returns the
        per-sample ``weight * loss_i`` tensor with ``sample_weights``
        multiplied in but not normalised.  Used by batch-trimming.
    """

    num_outputs: int = 2

    def __init__(self, weight: float = 1.0, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.beta = beta
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self.reduction = reduction

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

        if self.reduction == "none":
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Recover phi from (sin, cos) outputs."""
        return torch.atan2(raw[..., 0], raw[..., 1])


# ============================================================================
# Gaussian NLL losses (for the "Gaussian" loss family — sharpens core
# resolution at the cost of the quantile loss's tail focus)
# ============================================================================


class GaussianParameterLoss(nn.Module):
    """Gaussian negative log-likelihood on linearly normalised targets.

    The model outputs ``(mu, log_var)`` per parameter.  Loss is the standard
    Gaussian NLL with the constant term dropped (irrelevant for gradients)::

        per_sample = 0.5 * (exp(-log_var) * (target_norm - mu)**2 + log_var)

    Using ``log_var`` instead of ``var`` guarantees positive variance without
    clamping and prevents divide-by-zero / NaN gradient explosions.

    Targets are linearly normalised to ``[-1, 1]`` using ``norm_min`` /
    ``norm_max`` before the NLL is computed; predictions are denormalised
    back to physical space.  Analytic quantile predictions are available via
    ``predict_quantiles(raw, quantiles)``.

    Parameters
    ----------
    norm_min : float
        Lower bound of the physical range for linear normalisation.
    norm_max : float
        Upper bound of the physical range for linear normalisation.
    weight : float
        Multiplicative weight applied to this parameter's loss.
    log_var_clamp : float
        Symmetric clamp applied to ``log_var`` inside the NLL.  Protects
        against ``exp(-log_var)`` overflow under sharp loss-landscape
        perturbations (e.g. SAM ascent step) which can otherwise push
        ``log_var`` arbitrarily negative → NaN.  Default ``5.0`` limits
        σ_min to exp(-2.5) ≈ 0.08 in normalised space, preventing the
        catastrophic exp(8) ≈ 3000 multiplier that caused gradient spikes
        with the previous default of 8.0.
    reduction : str
        ``"mean"`` (default): return scalar, current behaviour.
        ``"none"``: return per-sample ``weight * per_sample`` with shape
        ``(N,)``, with ``sample_weights`` multiplied in but **not
        normalised**.  Used by batch-trimming to rank samples.
    """

    num_outputs: int = 2

    def __init__(
        self,
        norm_min: float = -1.0,
        norm_max: float = 1.0,
        weight: float = 1.0,
        log_var_clamp: float = 5.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.log_var_clamp = float(log_var_clamp)
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self.reduction = reduction
        self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
        self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))

    def _nll(self, mu: Tensor, log_var: Tensor, target_norm: Tensor) -> Tensor:
        """Full Gaussian NLL including the 0.5·log(2π) constant.

        Including the constant keeps the loss non-negative when the model
        is well-calibrated (σ ≈ residual std), which prevents confusing
        negative loss values on the dashboard and makes the loss landscape
        smoother near the optimum.
        """
        # Clamp log_var to prevent exp(-log_var) overflow.  The clamp is
        # placed inside the same autograd-tracked expression so that the
        # gradient is zero outside the clamp region (saturated), which
        # is the desired behaviour — we do NOT want the optimizer to
        # keep driving log_var past the clamp.
        log_var = log_var.clamp(-self.log_var_clamp, self.log_var_clamp)
        return 0.5 * (torch.exp(-log_var) * (target_norm - mu) ** 2 + log_var + math.log(2.0 * math.pi))

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, 2) with [mu, log_var];  target: (N,) in physical units."""
        mu = pred[..., 0]
        log_var = pred[..., 1]
        t_norm = _linear_normalise(target, self.norm_min, self.norm_max)
        per_sample = self._nll(mu, log_var, t_norm)

        if self.reduction == "none":
            # Per-sample form: apply sample_weights as a multiplier (not a
            # normaliser) so downstream code can sum across parameters.
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

        if sample_weights is not None:
            loss = (sample_weights * per_sample).sum() / sample_weights.sum()
        else:
            loss = per_sample.mean()
        return self.weight * loss

    def predict(self, raw: Tensor) -> Tensor:
        """Return the mean (``mu``) denormalised to physical units."""
        return _linear_denormalise(raw[..., 0], self.norm_min, self.norm_max)

    def predict_quantiles(self, raw: Tensor, quantiles: list[float] | Tensor | None = None) -> Tensor:
        """Analytic quantiles from (mu, log_var).

        For a Gaussian, ``Q(τ) = μ + σ · Φ⁻¹(τ)``.  The inverse CDF is
        computed via ``Φ⁻¹(τ) = √2 · erfinv(2τ - 1)``.  Both ``mu`` and
        ``sigma`` are converted to physical units before the quantile is
        formed, so the return value is in physical space.

        Parameters
        ----------
        raw : Tensor
            Model output of shape ``(N, 2)`` with ``[mu, log_var]``.
        quantiles : list[float] | Tensor | None
            Quantile levels to sample.  Defaults to the standard
            ``[0.05, 0.25, 0.5, 0.75, 0.95]`` five-point summary.
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        if not isinstance(quantiles, Tensor):
            quantiles = torch.tensor(list(quantiles), dtype=raw.dtype, device=raw.device)

        mu_phys = _linear_denormalise(raw[..., 0], self.norm_min, self.norm_max)
        # σ is in normalised space; the linear de-normalisation scales it by
        # (norm_max - norm_min) / 2 (same scale factor applied to every coord)
        half_range = 0.5 * (self.norm_max - self.norm_min)
        sigma_phys = torch.exp(0.5 * raw[..., 1]) * half_range
        z = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)  # Φ⁻¹(τ)
        return mu_phys.unsqueeze(-1) + sigma_phys.unsqueeze(-1) * z  # (N, Q)


class GaussianEtaLoss(GaussianParameterLoss):
    """Gaussian NLL in pseudorapidity (η) space for the θ parameter.

    Mirrors :class:`EtaQuantileLoss`: targets arrive as θ ∈ (0, π), are
    converted to η = -ln(tan(θ/2)), normalised to [-1, 1] in η units, and
    the Gaussian NLL is computed in that space.  **Predictions are returned
    in θ space** (via the inverse η→θ map) so downstream metrics remain
    comparable to the θ-native quantile loss.
    """

    num_outputs: int = 2

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        # Convert θ → η once and delegate to the parent's NLL on the η-space target.
        return super().forward(pred, _theta_to_eta(target), sample_weights)

    def predict(self, raw: Tensor) -> Tensor:
        """Return the mean η denormalised and converted back to θ."""
        eta_pred = _linear_denormalise(raw[..., 0], self.norm_min, self.norm_max)
        return _eta_to_theta(eta_pred)

    def predict_quantiles(self, raw: Tensor, quantiles: list[float] | Tensor | None = None) -> Tensor:
        """Analytic quantiles in θ space.

        Compute quantiles in η space (where the Gaussian lives), then apply
        the monotonic η→θ map.  Note: η→θ is monotonically decreasing, so the
        resulting θ quantiles are in *reverse* order.  We sort descending to
        keep the returned tensor monotonically increasing in θ.
        """
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        if not isinstance(quantiles, Tensor):
            quantiles = torch.tensor(list(quantiles), dtype=raw.dtype, device=raw.device)

        mu_eta = raw[..., 0]  # in normalised η-space
        log_var = raw[..., 1]
        sigma_norm = torch.exp(0.5 * log_var)
        z = math.sqrt(2.0) * torch.erfinv(2.0 * quantiles - 1.0)
        # quantiles in normalised η-space: μ + σ · z
        eta_norm_q = mu_eta.unsqueeze(-1) + sigma_norm.unsqueeze(-1) * z  # (N, Q)
        eta_phys_q = _linear_denormalise(eta_norm_q, self.norm_min, self.norm_max)
        theta_q = _eta_to_theta(eta_phys_q)  # (N, Q), decreasing in q
        # Sort to ascending θ order so downstream code sees monotonic quantiles.
        return torch.sort(theta_q, dim=-1).values


class CosinePhiLoss(nn.Module):
    """Angular loss ``1 - cos(phi_pred - phi_true)`` via (sin, cos) outputs.

    The model outputs ``(sin_raw, cos_raw)``.  These are unit-normalised
    inside the forward (divided by ``sqrt(sin² + cos²)``) to prevent the
    unbounded-drift failure mode of direct-φ prediction.  The loss is::

        1 - cos(phi_pred - phi_true)
            = 1 - (sin_p · sin(phi_true) + cos_p · cos(phi_true))

    which is naturally wraparound-safe.  Recovery: ``φ = atan2(sin, cos)``.
    Uses ``num_outputs=2`` to match :class:`CircularPhiLoss` so the rest of
    the metric / slice machinery does not need to branch on loss type.

    Parameters
    ----------
    weight : float
        Loss weight.
    reduction : str
        ``"mean"`` (default) returns scalar; ``"none"`` returns the
        per-sample ``weight * (1 - cos(Δφ))`` tensor with ``sample_weights``
        multiplied in but not normalised.  Used by batch-trimming.
    """

    num_outputs: int = 2

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        if reduction not in ("mean", "none"):
            raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor, sample_weights: Tensor | None = None) -> Tensor:
        """pred: (N, 2) with [sin, cos];  target: (N,) with phi in radians."""
        sin_p = pred[..., 0]
        cos_p = pred[..., 1]
        # Unit-normalise to the circle; small epsilon guards against zero norm.
        norm = torch.sqrt(sin_p * sin_p + cos_p * cos_p + 1e-8)
        sin_p = sin_p / norm
        cos_p = cos_p / norm
        # 1 - cos(pred - target) = 1 - (sin_p sin_t + cos_p cos_t)
        per_sample = 1.0 - (sin_p * torch.sin(target) + cos_p * torch.cos(target))

        if self.reduction == "none":
            if sample_weights is not None:
                per_sample = sample_weights * per_sample
            return self.weight * per_sample

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
    "gaussian": GaussianParameterLoss,
    "gaussian_eta": GaussianEtaLoss,
    "cosine_phi": CosinePhiLoss,
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
        trim_mask: Tensor | None = None,
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
        trim_mask : Tensor | None
            Optional per-sample weight mask of shape ``(N_valid,)``
            (i.e. same size as ``valid_mask.sum()`` if a valid_mask is
            given, else same size as ``pred.shape[0]``).  Used by
            batch-trimming: a float tensor of 0/1 values where 0 drops
            the sample from the loss and 1 keeps it with its full
            weight.  The trim mask is multiplied into ``sample_weights``
            so that the weighted-mean normalisation denominator is
            ``(sample_weights * trim_mask).sum()`` — i.e. a proper mean
            over retained samples, not a scaled mean over all samples.

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

        # Fold the trim_mask into sample_weights (creating one if needed).
        # The sub-losses use ``sum(sw * per) / sum(sw)`` so a hard 0/1 mask
        # acts as a dropout filter on the batch.
        if trim_mask is not None:
            if sample_weights is None:
                sample_weights = trim_mask
            else:
                sample_weights = sample_weights * trim_mask

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

    def per_sample_total(
        self,
        pred: Tensor,
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Return per-sample weighted sum of the five parameter losses.

        Shape: ``(N_valid,)`` — one scalar per *kept* track, which is the
        quantity a batch-trimming rule should rank on.  The per-parameter
        loss weights from the YAML config are applied (so this matches the
        objective the optimizer sees), but the ``qop_tail_weight`` meta
        sample weights are **deliberately NOT** applied — those are an
        orthogonal curvature re-weighting and would bias the trimmer
        toward keeping high-|qop| tracks, which is the opposite of what
        we want.

        Each sub-loss is invoked in its ``reduction="none"`` path, which
        requires that every loss class used in the config supports it.
        The currently supported ones are ``gaussian``, ``gaussian_eta``,
        and ``cosine_phi``.  If a quantile loss is plugged in later a
        ``reduction`` argument will have to be added there too — we fail
        loudly with a clear error rather than silently returning a wrong
        aggregate.

        Parameters
        ----------
        pred : Tensor
            Raw model output ``(B, total_outputs)``.
        targets : dict[str, Tensor]
            Per-parameter targets, each ``(B,)``.
        valid_mask : Tensor | None
            Boolean mask selecting valid tracks.

        Returns
        -------
        Tensor
            Shape ``(N_valid,)`` — per-sample aggregate loss, detached
            from the autograd graph (safe to feed into ``kthvalue``).
        """
        device = pred.device

        # Determine N_valid for the zero-init accumulator.
        n_valid = (
            int(valid_mask.sum().item())
            if valid_mask is not None
            else int(pred.shape[0])
        )
        if n_valid == 0:
            return torch.zeros(0, device=device)

        aggregate = torch.zeros(n_valid, device=device)

        for name in self.parameter_order:
            sub_loss = self.losses[name]
            if not hasattr(sub_loss, "reduction"):
                raise RuntimeError(
                    f"per_sample_total: loss class {type(sub_loss).__name__} "
                    f"for parameter {name!r} does not support reduction='none'. "
                    "Batch trimming requires per-sample losses on every parameter."
                )

            start, end = self._output_slices[name]
            p = pred[..., start:end] if end - start > 1 else pred[..., start]
            t = targets[name]
            if valid_mask is not None:
                p = p[valid_mask]
                t = t[valid_mask]
            if t.numel() == 0:
                continue

            prev = sub_loss.reduction
            sub_loss.reduction = "none"
            try:
                per_sample = sub_loss(p, t, sample_weights=None)
            finally:
                sub_loss.reduction = prev

            aggregate = aggregate + per_sample

        return aggregate.detach()

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
        Gaussian losses (``GaussianParameterLoss`` / ``GaussianEtaLoss``)
        expose an analytic ``predict_quantiles`` from ``(mu, log_var)`` and are
        therefore included in the dispatch.
        """
        preds: dict[str, Tensor] = {}
        for name in self.parameter_order:
            start, end = self._output_slices[name]
            raw = pred[..., start:end]
            loss_fn = self.losses[name]
            if isinstance(
                loss_fn,
                (QuantileLoss, EtaQuantileLoss, SplineQuantileLoss, GaussianParameterLoss),
            ):
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
