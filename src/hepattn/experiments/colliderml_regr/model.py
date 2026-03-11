"""Track parameter regression model using Bidirectional Mamba-2.

This module provides:

- :class:`TrackParameterRegressor` — the core ``nn.Module`` that embeds
  per-hit features (with Fourier encoding and min-max normalisation),
  encodes them with a bidirectional Mamba-2 encoder,
  extracts the concatenated final SSM hidden states of the last layer,
  and regresses the five perigee track parameters through a Dense head.

- :class:`TrackRegressionWrapper` — a ``LightningModule`` that wraps the
  regressor, configures the optimiser / scheduler, and handles the
  train / val / test loops with MAE and precision metrics.

Data flow
---------
1. Hits arrive as ``(B, N, D_in)`` with 12 features per hit:
   [x, y, z, r, phi_hit, theta_hit, s, volume_id, layer_id, surface_id, detector, eta_hit]
2. Min-max normalisation scales each feature to [0, 1].
3. Fourier encoding expands each feature into multi-scale sin/cos components.
4. :class:`hepattn.models.Dense` projects Fourier features to dimension ``dim``.
5. :class:`BidirectionalMambaEncoder` produces sequence output and a global
   hidden state from the last SSM layer.
6. A regression head (Dense) maps the hidden state to the target vector.
7. :class:`TrackParameterLoss` computes the config-driven composite loss.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
from lion_pytorch import Lion
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW

from hepattn.models.dense import Dense
from hepattn.experiments.colliderml_regr.losses import TrackParameterLoss
from hepattn.experiments.colliderml_regr.mamba_state import BidirectionalMambaEncoder


# ============================================================================
# Fourier encoding
# ============================================================================


def fourier_encode(
    x: Tensor,
    fourier_scales: list[int] | None = None,
    fourier_base: int = 3,
) -> Tensor:
    """Encode input tensor with multi-scale Fourier features.

    For input of shape ``(*, D)``, produces ``(*, 2 * len(fourier_scales) * D)``
    by concatenating ``sin(x / base^n)`` and ``cos(x / base^n)`` for each scale.

    Parameters
    ----------
    x : Tensor
        Input features ``(*, D)``.
    fourier_scales : list[int]
        Exponent scales for the Fourier basis.
    fourier_base : int
        Base for the exponential frequency scaling.
    """
    if fourier_scales is None:
        fourier_scales = [-3, -2, -1, 0, 1, 2, 3]
    sin = [torch.sin(x / (fourier_base**n)) for n in fourier_scales]
    cos = [torch.cos(x / (fourier_base**n)) for n in fourier_scales]
    return torch.cat(sin + cos, dim=-1)



class TrackParameterRegressor(nn.Module):
    """Bidirectional Mamba-2 regressor for perigee track parameters.

    The encoder produces two SSM hidden states — one from the forward
    scan and one from the backward scan.  Instead of concatenating the
    two 16k-dim vectors into a single 32k-dim input for a single head,
    each direction is first projected through its own configurable
    ``Dense`` network (``fwd_head`` / ``bwd_head``).  The two
    lower-dimensional embeddings are then concatenated and fed through
    a final ``output_head`` that produces the regression targets.

    This factored design dramatically reduces parameters (32k × 256
    dense → two 16k × 128 dense) while giving independent capacity to
    each scan direction.

    Parameters
    ----------
    input_dim : int
        Number of raw per-hit features (before Fourier encoding).
    dim : int
        Internal model dimension (embedding size).
    encoder : BidirectionalMambaEncoder
        Pre-constructed encoder module.
    loss_module : TrackParameterLoss
        Pre-constructed composite loss module.

    state_head_output_dim : int
        Output dimensionality of each per-direction projection head.
    state_head_hidden_layers : int | list[int] | None
        Hidden layers inside each per-direction head (``None`` = linear).
    state_head_dropout : float
        Dropout in each per-direction head.
    state_head_activation : str
        Activation in each per-direction head.

    output_head_hidden_layers : int | list[int] | None
        Hidden layers in the final output head (``None`` = linear).
    output_head_dropout : float
        Dropout in the final output head.
    output_head_activation : str
        Activation in the final output head.

    input_net_hidden_layers : int | list[int] | None
        Hidden layer config for the input embedding network.
    input_net_dropout : float
        Dropout in the input embedding network.
    input_net_activation : str
        Activation function for the input embedding network.

    input_fields : list[str]
        Names of hit-level input features. Used for validation against
        ``input_dim``.
    fourier_scales : list[int] | None
        Fourier encoding scales.
    fourier_base : int
        Base for Fourier frequency scaling.
    norm_min / norm_max : list[float] | None
        Per-feature bounds for min-max normalisation.
    """

    @staticmethod
    def _resolve_activation(name: str) -> nn.Module | str:
        """Map an activation name string to a Module instance.

        ``'SwiGLU'`` is returned as a string so that :class:`Dense` can
        handle the gated linear unit bookkeeping internally.
        """
        _map: dict[str, nn.Module | str] = {
            "silu": nn.SiLU(),
            "gelu": nn.GELU(),
            "swiglu": "SwiGLU",
            "relu": nn.ReLU(),
            "mish": nn.Mish(),
        }
        key = name.lower()
        if key not in _map:
            raise ValueError(f"Unknown activation '{name}'. Choose from: {list(_map.keys())}")
        return _map[key]

    def __init__(
        self,
        input_dim: int,
        dim: int,
        encoder: BidirectionalMambaEncoder,
        loss_module: TrackParameterLoss,
        # Per-direction state projection heads
        state_head_output_dim: int = 256,
        state_head_hidden_layers: int | list[int] | None = 0,
        state_head_dropout: float = 0.1,
        state_head_activation: str = "SiLU",
        # Final output head (after combining fwd + bwd projections)
        output_head_hidden_layers: int | list[int] | None = 0,
        output_head_dropout: float = 0.0,
        output_head_activation: str = "SiLU",
        # Input embedding network
        input_net_hidden_layers: int | list[int] | None = 0,
        input_net_dropout: float = 0.0,
        input_net_activation: str = "SiLU",
        # Data config
        input_fields: list[str] | None = None,
        sort_field: str = "s",
        fourier_scales: list[int] | None = None,
        fourier_base: int = 3,
        norm_min: list[float] | None = None,
        norm_max: list[float] | None = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.dim = dim
        self.input_fields = input_fields or []

        # Fourier encoding config
        self.fourier_scales = fourier_scales if fourier_scales is not None else [-3, -2, -1, 0, 1, 2, 3]
        self.fourier_base = fourier_base
        fourier_dim = input_dim * 2 * len(self.fourier_scales)

        # Min-max normalisation buffers
        if norm_min is not None and norm_max is not None:
            self.register_buffer("norm_min", torch.tensor(norm_min, dtype=torch.float32))
            self.register_buffer("norm_max", torch.tensor(norm_max, dtype=torch.float32))
            self.use_norm = True
        else:
            self.use_norm = False

        # Validate input_fields matches input_dim if both are provided
        if self.input_fields and len(self.input_fields) != input_dim:
            raise ValueError(
                f"len(input_fields)={len(self.input_fields)} != input_dim={input_dim}"
            )

        # Hit embedding: fourier_dim → dim
        self.input_net = Dense(
            input_size=fourier_dim,
            output_size=dim,
            hidden_layers=input_net_hidden_layers,
            dropout=input_net_dropout,
            activation=self._resolve_activation(input_net_activation),
        )

        # Bidirectional Mamba-2 encoder with state extraction
        self.encoder = encoder

        # Loss module (holds sub-losses and knows output dimensionality)
        self.loss_module = loss_module

        # ---- Factored regression heads ----
        # Each direction's SSM state is projected independently before
        # being combined for the final regression output.
        self.per_dir_dim = encoder.state_dim // 2

        self.fwd_head = Dense(
            input_size=self.per_dir_dim,
            output_size=state_head_output_dim,
            hidden_layers=state_head_hidden_layers,
            dropout=state_head_dropout,
            activation=self._resolve_activation(state_head_activation),
        )
        self.bwd_head = Dense(
            input_size=self.per_dir_dim,
            output_size=state_head_output_dim,
            hidden_layers=state_head_hidden_layers,
            dropout=state_head_dropout,
            activation=self._resolve_activation(state_head_activation),
        )
        # Final output head — no final_activation (linear output for regression)
        self.output_head = Dense(
            input_size=2 * state_head_output_dim,
            output_size=loss_module.total_outputs,
            hidden_layers=output_head_hidden_layers,
            dropout=output_head_dropout,
            activation=self._resolve_activation(output_head_activation),
        )

    def _normalise(self, x: Tensor) -> Tensor:
        """Min-max normalise features to [0, 1]."""
        if not self.use_norm:
            return x
        span = (self.norm_max - self.norm_min).clamp(min=1e-8)
        return (x - self.norm_min) / span

    def forward(
        self,
        inputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        inputs : dict[str, Tensor]
            Must contain:
            - ``"hit_features"`` : ``(B, N, input_dim)``
            - ``"hit_s"`` : ``(B, N)`` distance-from-IP for ordering
            - ``"hit_valid"`` : ``(B, N)`` bool mask (for padded batches)
            Optionally:
            - ``"seq_idx"`` : ``(B, N)`` sequence indices for packed batches

        Returns
        -------
        dict[str, Tensor]
            ``"pred"`` — raw regression output ``(B, total_outputs)``
            ``"hidden_state"`` — encoder final state ``(B, state_dim)``
        """
        x = inputs["hit_features"]
        s = inputs["hit_s"]
        seq_idx = inputs.get("seq_idx")

        # Min-max normalise
        x = self._normalise(x)

        # Fourier encode: (B, N, D) → (B, N, 2*n_scales*D)
        x = fourier_encode(x, self.fourier_scales, self.fourier_base)

        # Embed hits: (B, N, fourier_dim) → (B, N, dim)
        x = self.input_net(x)

        # Encode with bidirectional Mamba-2 (sort by s, extract hidden state)
        _seq_out, hidden_state = self.encoder(x, x_sort_value=s, seq_idx=seq_idx)

        # Split into forward / backward SSM states and project independently
        h_fwd = hidden_state[:, :self.per_dir_dim]
        h_bwd = hidden_state[:, self.per_dir_dim:]

        z_fwd = self.fwd_head(h_fwd)
        z_bwd = self.bwd_head(h_bwd)
        z = torch.cat([z_fwd, z_bwd], dim=-1)

        # Final regression output (linear final activation)
        pred = self.output_head(z)

        return {"pred": pred, "hidden_state": hidden_state}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert raw outputs to physical predictions."""
        return self.loss_module.predict(outputs["pred"])

    def compute_loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all parameter losses.

        Parameters
        ----------
        outputs : dict
            From ``forward()``.
        targets : dict[str, Tensor]
            Must contain ``d0``, ``z0``, ``phi``, ``theta``, ``qop`` tensors.
        valid_mask : Tensor | None
            ``(B,)`` bool mask for valid tracks.
        """
        return self.loss_module(outputs["pred"], targets, valid_mask=valid_mask)


# ============================================================================
# LightningModule wrapper
# ============================================================================


class TrackRegressionWrapper(LightningModule):
    """Lightning wrapper for track parameter regression training.

    Parameters
    ----------
    model : TrackParameterRegressor
        The regression model.
    lrs_config : dict
        Learning-rate scheduler configuration with keys:
        ``initial``, ``max``, ``end``, ``pct_start``, ``weight_decay``,
        ``skip_scheduler``.
    optimizer : str
        ``"AdamW"`` or ``"Lion"``.
    """

    def __init__(
        self,
        model: TrackParameterRegressor,
        lrs_config: dict[str, Any],
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        name: str = "TrackRegression",
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.lrs_config = lrs_config
        self.opt_name = optimizer

    # -- forward / predict --------------------------------------------------

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        preds = self.model.predict(outputs)
        return preds, targets

    # -- step helpers -------------------------------------------------------

    def _shared_step(
        self,
        batch: tuple[dict[str, Tensor], dict[str, Tensor]],
        stage: str,
    ) -> Tensor:
        inputs, targets = batch
        outputs = self.model(inputs)

        valid_mask = targets.get("track_valid")
        losses = self.model.compute_loss(outputs, targets, valid_mask=valid_mask)

        # Log every component
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, sync_dist=True, prog_bar=(name == "total"))

        # Compute and log per-parameter metrics for all stages
        preds = self.model.predict(outputs)
        self._log_metrics(preds, targets, valid_mask, stage)

        return losses["total"]

    # Units and scale factors for precision logging
    _PRECISION_UNITS: dict[str, tuple[str, float]] = {
        "d0": ("[mm]", 1.0),
        "z0": ("[mm]", 1.0),
        "phi": ("[mrad]", 1000.0),
        "theta": ("[mrad]", 1000.0),
        "qop": ("[1/GeV]", 1.0),
    }

    def _log_metrics(
        self,
        preds: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor | None,
        stage: str,
    ) -> None:
        """Log per-parameter metrics: MAE and precision.

        Metrics
        -------
        - ``{stage}/{name}/mae``: mean absolute error
        - ``{stage}/{name}/precision {unit}``: std of (pred - truth) residuals,
          scaled to physical units (mrad for angles, mm for d0/z0, 1/GeV for qop)
        """
        for name in self.model.loss_module.parameter_order:
            if name not in preds or name not in targets:
                continue
            p = preds[name]
            t = targets[name]
            if valid_mask is not None:
                p = p[valid_mask]
                t = t[valid_mask]
            if t.numel() == 0:
                continue

            residual = p - t

            # MAE
            self.log(f"{stage}/{name}/mae", residual.abs().mean(), sync_dist=True)

            # Precision: std of residuals in physical units
            if residual.numel() > 1:
                unit, scale = self._PRECISION_UNITS.get(name, ("", 1.0))
                self.log(
                    f"{stage}/{name}/precision {unit}",
                    residual.std() * scale,
                    sync_dist=True,
                )

    # -- train / val / test ------------------------------------------------

    def training_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "train")}

    def validation_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "val")}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        valid_mask = targets.get("track_valid")
        losses = self.model.compute_loss(outputs, targets, valid_mask=valid_mask)

        # Log every component
        for name, value in losses.items():
            self.log(f"test/{name}", value, sync_dist=True, prog_bar=(name == "total"))

        # Compute predictions and metrics
        preds = self.model.predict(outputs)
        self._log_metrics(preds, targets, valid_mask, "test")

        return {"loss": losses["total"], "preds": preds, "targets": targets}

    # -- optimiser / scheduler ---------------------------------------------

    def configure_optimizers(self):
        if self.opt_name.lower() == "adamw":
            opt_cls = AdamW
        elif self.opt_name.lower() == "lion":
            opt_cls = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.opt_name}")

        opt = opt_cls(
            self.model.parameters(),
            lr=self.lrs_config["initial"],
            weight_decay=self.lrs_config["weight_decay"],
        )

        if not self.lrs_config.get("skip_scheduler"):
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            return [opt], [{"scheduler": sch, "interval": "step"}]

        return opt
