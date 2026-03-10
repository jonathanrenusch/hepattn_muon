"""Monotonic spline transform for track parameter normalisation.

Provides a :class:`MonotonicSplineTransform` that maps physical track
parameter values to a normalised ``[0, 1]`` space (and back) using a
monotone piecewise-cubic Hermite interpolant (PCHIP).

Knot positions are determined from the **quantiles** of the training data
distribution (after track selection) so that each inter-knot interval
contains roughly the same number of tracks.  This yields an approximately
uniform distribution in the transformed space, giving the regression loss
equal sensitivity across the full parameter range.

The knot tables are stored in a small YAML config produced by the companion
script ``scripts/fit_splines.py`` and loaded at model-construction time.

Usage (inside a loss module)::

    transform = MonotonicSplineTransform.from_config("spline_d0.yaml")
    u_pred  = transform.forward(y_pred)   # physical → normalised
    u_truth = transform.forward(y_truth)
    loss    = F.smooth_l1_loss(u_pred, u_truth)
    y_hat   = transform.inverse(u_pred)   # normalised → physical
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn


class MonotonicSplineTransform(nn.Module):
    """Piecewise-cubic Hermite monotone transform backed by fixed knot tables.

    The transform is **not learnable** — the knots are registered as buffers
    so that they move to the correct device/dtype automatically but do not
    receive gradients.

    The forward mapping ``physical → [0, 1]`` and its inverse are both
    evaluated with ``torch`` ops only and are therefore fully
    differentiable (gradients flow through them during training).

    Parameters
    ----------
    knot_x : list[float]
        Knot positions in physical space (strictly increasing).
    knot_y : list[float]
        Corresponding positions in normalised ``[0, 1]`` space
        (strictly increasing, first=0, last=1).
    name : str
        Human-readable name of the parameter (used in error messages).
    """

    def __init__(self, knot_x: list[float], knot_y: list[float], name: str = ""):
        super().__init__()
        assert len(knot_x) == len(knot_y) >= 2, "Need at least 2 knots"
        self.name = name

        kx = torch.tensor(knot_x, dtype=torch.float64)
        ky = torch.tensor(knot_y, dtype=torch.float64)

        # Precompute monotone Hermite slopes at each knot (Fritsch–Carlson)
        slopes = self._fritsch_carlson_slopes(kx, ky)

        # Register as non-trainable buffers
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("slopes", slopes)

        # Also pre-compute inverse knots (swap roles)
        inv_slopes = self._fritsch_carlson_slopes(ky, kx)
        self.register_buffer("inv_kx", ky)
        self.register_buffer("inv_ky", kx)
        self.register_buffer("inv_slopes", inv_slopes)

    # ----- construction helpers -------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "MonotonicSplineTransform":
        """Load from a YAML file produced by ``fit_splines.py``.

        Expected YAML format::

            name: d0
            knot_x: [...]
            knot_y: [...]
        """
        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        return cls(
            knot_x=cfg["knot_x"],
            knot_y=cfg["knot_y"],
            name=cfg.get("name", ""),
        )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "MonotonicSplineTransform":
        """Construct from an in-memory config dict."""
        return cls(
            knot_x=cfg["knot_x"],
            knot_y=cfg["knot_y"],
            name=cfg.get("name", ""),
        )

    # ----- Fritsch–Carlson monotone slopes --------------------------------

    @staticmethod
    def _fritsch_carlson_slopes(x: Tensor, y: Tensor) -> Tensor:
        """Compute monotone cubic Hermite slopes (Fritsch–Carlson method).

        Guarantees the interpolant is monotone between every pair of knots.
        """
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        delta = dy / dx  # secant slopes

        n = len(x)
        m = torch.zeros_like(x)

        # Interior points: harmonic mean of adjacent secants (only when same sign)
        for k in range(1, n - 1):
            if delta[k - 1].sign() != delta[k].sign() or delta[k - 1] == 0 or delta[k] == 0:
                m[k] = 0.0
            else:
                m[k] = 2.0 * delta[k - 1] * delta[k] / (delta[k - 1] + delta[k])

        # End-point slopes: one-sided
        m[0] = delta[0]
        m[-1] = delta[-1]

        # Fritsch–Carlson adjustment for monotonicity
        for k in range(n - 1):
            if delta[k] == 0:
                m[k] = 0.0
                m[k + 1] = 0.0
            else:
                alpha = m[k] / delta[k]
                beta = m[k + 1] / delta[k]
                # Ensure we stay within the monotonicity region
                mag = alpha**2 + beta**2
                if mag > 9.0:
                    tau = 3.0 / mag.sqrt()
                    m[k] = tau * alpha * delta[k]
                    m[k + 1] = tau * beta * delta[k]

        return m

    # ----- core Hermite evaluation ----------------------------------------

    @staticmethod
    def _hermite_eval(x: Tensor, kx: Tensor, ky: Tensor, slopes: Tensor) -> Tensor:
        """Evaluate the piecewise cubic Hermite interpolant.

        Parameters
        ----------
        x : Tensor
            Query points (any shape).
        kx, ky, slopes : Tensor
            Knot tables (1-D, on the same device as *x*).

        Returns
        -------
        Tensor
            Interpolated values, same shape as *x*.
        """
        # Cast to float64 for precision, will cast back at end
        orig_dtype = x.dtype
        x = x.to(torch.float64)

        # Clamp to knot range
        x_clamped = x.clamp(kx[0], kx[-1])

        # Find interval index for each query point via searchsorted
        # searchsorted returns the index where x would be inserted to keep kx sorted
        idx = torch.searchsorted(kx, x_clamped, right=True) - 1
        idx = idx.clamp(0, len(kx) - 2)

        # Local coordinates
        x0 = kx[idx]
        x1 = kx[idx + 1]
        y0 = ky[idx]
        y1 = ky[idx + 1]
        m0 = slopes[idx]
        m1 = slopes[idx + 1]
        h = x1 - x0
        t = (x_clamped - x0) / h

        # Hermite basis functions
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        result = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        return result.to(orig_dtype)

    # ----- public API -----------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Map from physical space to normalised ``[0, 1]`` space."""
        return self._hermite_eval(x, self.kx, self.ky, self.slopes)

    def inverse(self, u: Tensor) -> Tensor:
        """Map from normalised ``[0, 1]`` space back to physical space."""
        return self._hermite_eval(u, self.inv_kx, self.inv_ky, self.inv_slopes)

    def extra_repr(self) -> str:
        return f"name={self.name!r}, n_knots={len(self.kx)}"
