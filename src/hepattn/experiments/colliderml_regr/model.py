"""Track parameter regression model using Bidirectional Mamba-2 or Transformer.

This module provides:

- :class:`TrackParameterRegressor` — the core ``nn.Module`` that embeds
  per-hit features (with Fourier encoding and min-max normalisation),
  encodes them with one of three backends (selected via ``pool``), and
  regresses the five perigee track parameters through a Dense head.

- :class:`TrackRegressionWrapper` — a ``LightningModule`` that wraps the
  regressor, configures the optimiser / scheduler, and handles the
  train / val / test loops with MAE, precision, IQR/σ and RMS metrics.

Pool / backbone selection
-------------------------
The ``pool`` argument determines how a per-track summary is obtained from
the encoder and which encoder API is used:

``ssm_state``
    :class:`BidirectionalMambaEncoder` — extracts the concatenated
    forward + backward final SSM hidden states (``2 * nheads * headdim *
    d_state``) from the last layer, projects each direction through its
    own ``state_head``, concatenates, and feeds ``output_head``.
``ssm_cls``
    :class:`BidirectionalMambaCLSEncoder` — appends learned CLS tokens to
    each scan direction, reads the final token's sequence output per
    direction, and concatenates to shape ``(B, 2 * dim)``.  Feeds
    ``output_head`` directly.
``register_token``
    Transformer :class:`EncoderWithCLS` (flash-attn2) with a single
    learned register token.  The register's final representation is
    read out at shape ``(B, dim)`` and feeds ``output_head`` directly.

Hybrid FP32/bf16 precision
--------------------------
The Lightning trainer runs in strict FP32 (``trainer.precision: 32-true``).
Mamba-2 CUDA kernels and flash-attn2 both require bf16/fp16, so the
``self.encoder(...)`` call is wrapped in a ``torch.amp.autocast`` block and
the encoder outputs are explicitly cast back to FP32.  Heads, loss, and
metrics therefore remain in full precision — the FP32 regime the NeurIPS
precision-focused study targets.

Data flow
---------
1. Hits arrive as ``(B, N, D_in)`` with 12 features per hit:
   [x, y, z, r, phi_hit, theta_hit, s, volume_id, layer_id, surface_id, detector, eta_hit]
2. Min-max normalisation scales each feature to [0, 1].
3. Fourier encoding expands each feature into multi-scale sin/cos components.
4. :class:`hepattn.models.Dense` projects Fourier features to dimension ``dim``.
5. Encoder (one of the three above) produces sequence output and a pooled
   per-track summary tensor.
6. A regression head (Dense) maps the pooled summary to the target vector.
7. :class:`TrackParameterLoss` computes the config-driven composite loss.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Literal

import torch
import torch.distributed as dist
from lion_pytorch import Lion
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW

from hepattn.models.dense import Dense
from hepattn.experiments.colliderml_regr.losses import MixtureDensityLoss, TrackParameterLoss
from hepattn.experiments.colliderml_regr.mamba_state import BidirectionalMambaEncoder
from hepattn.experiments.colliderml_regr.muon import MuonHybrid, split_params_for_muon
from hepattn.experiments.colliderml_regr.sam import SAM


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
    """Track parameter regressor supporting SSM, SSM-CLS and Transformer backbones.

    The ``pool`` argument selects which pooling strategy is used:

    - ``ssm_state`` (default): :class:`BidirectionalMambaEncoder` exposes
      two per-direction SSM hidden states.  Each is projected through its
      own ``state_head`` (``fwd_head`` / ``bwd_head``), concatenated, and
      fed through ``output_head``.
    - ``ssm_cls``: encoder appends learned CLS tokens to each scan
      direction and returns the concatenated pair of ``(cls_fwd, cls_bwd)``
      shape ``(B, 2 * dim)``.  Goes straight into ``output_head`` (no
      state_head).
    - ``register_token``: Transformer encoder with a learned register
      token; the register's final representation shape ``(B, dim)`` is
      fed straight into ``output_head`` (no state_head).

    Parameters
    ----------
    input_dim : int
        Number of raw per-hit features (before Fourier encoding).
    dim : int
        Internal model dimension (embedding size).
    encoder : nn.Module
        Pre-constructed encoder module.  Its ``forward`` is expected to
        return ``(sequence_output, pooled_summary)``.
    loss_module : TrackParameterLoss
        Pre-constructed composite loss module.
    pool : str
        One of ``"ssm_state"``, ``"ssm_cls"``, ``"register_token"``.

    state_head_output_dim : int
        Output dimensionality of each per-direction projection head.
        Only used when ``pool == "ssm_state"``.
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
    encoder_autocast_dtype : str
        Autocast dtype used for the encoder forward pass.  Defaults to
        ``"bfloat16"``, which is required by Mamba-2 Triton kernels and
        flash-attn2.  Set to ``"float32"`` to disable the autocast (only
        meaningful for the Transformer baseline on CPU).
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
        encoder: nn.Module,
        loss_module: TrackParameterLoss,
        pool: Literal["ssm_state", "ssm_cls", "register_token"] = "ssm_state",
        # Per-direction state projection heads (only used when pool == "ssm_state")
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
        # Precision
        encoder_autocast_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        # Output head initialisation scale — multiply the last Linear layer's
        # weights by this factor at init.  Keeps initial predictions near zero
        # regardless of hidden-state magnitude, which is critical for Gaussian
        # NLL (exp(-log_var) amplifies large initial outputs).  Default 1.0
        # is a no-op.
        output_head_init_scale: float = 1.0,
    ):
        super().__init__()

        if pool not in ("ssm_state", "ssm_cls", "register_token"):
            raise ValueError(
                f"Unknown pool='{pool}'. Must be one of "
                "('ssm_state', 'ssm_cls', 'register_token')"
            )
        self.pool = pool

        self.input_dim = input_dim
        self.dim = dim
        self.input_fields = input_fields or []
        self.encoder_autocast_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[encoder_autocast_dtype]

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

        # Backbone encoder (SSM-state, SSM-CLS, or Transformer with register token)
        self.encoder = encoder

        # Loss module (holds sub-losses and knows output dimensionality)
        self.loss_module = loss_module

        # ---- Pool-dependent regression heads ----
        if pool == "ssm_state":
            # Two per-direction SSM states of dim (state_dim / 2), each
            # projected independently before being combined.
            if not hasattr(encoder, "state_dim"):
                raise ValueError(
                    "pool='ssm_state' requires the encoder to expose a "
                    "`state_dim` attribute (use BidirectionalMambaEncoder)."
                )
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
            output_head_input_dim = 2 * state_head_output_dim
        elif pool == "ssm_cls":
            # Two learned CLS tokens concatenated → (B, 2 * dim).
            # Projection head provides a gradient bottleneck matching the
            # ssm_state path — without it, raw encoder-output gradients
            # enter the bf16 backward pass at ~7x higher magnitude.
            self.per_dir_dim = None
            self.fwd_head = None
            self.bwd_head = None
            self.pool_head = Dense(
                input_size=2 * dim,
                output_size=state_head_output_dim,
                hidden_layers=state_head_hidden_layers,
                dropout=state_head_dropout,
                activation=self._resolve_activation(state_head_activation),
            )
            output_head_input_dim = state_head_output_dim
        elif pool == "register_token":
            # Single learned register token → (B, dim).
            self.per_dir_dim = None
            self.fwd_head = None
            self.bwd_head = None
            self.pool_head = Dense(
                input_size=dim,
                output_size=state_head_output_dim,
                hidden_layers=state_head_hidden_layers,
                dropout=state_head_dropout,
                activation=self._resolve_activation(state_head_activation),
            )
            output_head_input_dim = state_head_output_dim
        else:  # unreachable — guarded above
            raise ValueError(f"Unknown pool '{pool}'")

        # Final output head — no final_activation (linear output for regression)
        self.output_head = Dense(
            input_size=output_head_input_dim,
            output_size=loss_module.total_outputs,
            hidden_layers=output_head_hidden_layers,
            dropout=output_head_dropout,
            activation=self._resolve_activation(output_head_activation),
        )

        # Scale down the output head's last Linear layer so that initial
        # predictions are near zero regardless of hidden-state magnitude.
        # Critical for Gaussian NLL where large initial outputs push log_var
        # into saturated clamp regions with zero gradient.
        if output_head_init_scale != 1.0:
            last_layer = self.output_head.net[-1]
            assert isinstance(last_layer, nn.Linear), (
                f"output_head_init_scale requires the last layer of output_head "
                f"to be nn.Linear, got {type(last_layer).__name__}"
            )
            with torch.no_grad():
                last_layer.weight.mul_(output_head_init_scale)
                if last_layer.bias is not None:
                    last_layer.bias.zero_()

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
            ``"hidden_state"`` — pooled per-track summary ``(B, pool_dim)``
        """
        x = inputs["hit_features"]
        s = inputs["hit_s"]
        seq_idx = inputs.get("seq_idx")
        hit_valid = inputs.get("hit_valid")

        # Min-max normalise
        x = self._normalise(x)

        # Fourier encode: (B, N, D) → (B, N, 2*n_scales*D)
        x = fourier_encode(x, self.fourier_scales, self.fourier_base)

        # Embed hits: (B, N, fourier_dim) → (B, N, dim)
        x = self.input_net(x)

        # Encoder forward pass in reduced precision.  Mamba-2 Triton kernels
        # and flash-attn2 both require bf16/fp16; everything outside this
        # context (heads, loss, metrics) stays in FP32 when the trainer is
        # configured with precision=32-true.
        use_autocast = (
            x.is_cuda and self.encoder_autocast_dtype != torch.float32
        )
        with torch.amp.autocast(
            device_type="cuda",
            dtype=self.encoder_autocast_dtype,
            enabled=use_autocast,
        ):
            if self.pool == "register_token":
                # Transformer encoder path — uses `kv_mask` for padding.
                _, pooled = self.encoder(x, x_sort_value=s, kv_mask=hit_valid)
            else:
                # Mamba-2 encoder path — uses `seq_idx` (None for padded batches).
                _, pooled = self.encoder(x, x_sort_value=s, seq_idx=seq_idx)

        # Cast pooled summary back to FP32 for the heads / loss / metrics.
        # (The per-hit sequence output is discarded — only used by the
        # encoder-internal DDP tie that keeps its `gate` / norm params in the
        # autograd graph.)
        pooled = pooled.to(torch.float32)

        if self.pool == "ssm_state":
            # Split into forward / backward SSM states, project independently.
            h_fwd = pooled[:, :self.per_dir_dim]
            h_bwd = pooled[:, self.per_dir_dim:]
            z_fwd = self.fwd_head(h_fwd)
            z_bwd = self.bwd_head(h_bwd)
            z = torch.cat([z_fwd, z_bwd], dim=-1)
        else:
            # ssm_cls / register_token — project through pool_head bottleneck.
            z = self.pool_head(pooled)

        # Final regression output (linear final activation)
        pred = self.output_head(z)

        return {"pred": pred, "hidden_state": pooled}

    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert raw outputs to physical predictions."""
        return self.loss_module.predict(outputs["pred"])

    def compute_loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        valid_mask: Tensor | None = None,
        trim_mask: Tensor | None = None,
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
        trim_mask : Tensor | None
            Optional per-sample 0/1 weight mask passed through to
            :meth:`TrackParameterLoss.forward` — used by the batch-trimming
            path in the LightningModule to drop the worst-residual samples
            from the backward pass.  ``None`` (the default) means no
            trimming.
        """
        return self.loss_module(
            outputs["pred"], targets, valid_mask=valid_mask, trim_mask=trim_mask
        )


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
        optimizer: Literal["AdamW", "Lion", "MuonHybrid"] = "AdamW",
        name: str = "TrackRegression",
        pretrained_ckpt_path: str | None = None,
        use_sam: bool = False,
        sam_rho: float = 0.05,
        sam_adaptive: bool = False,
        sam_start_epoch: int = 0,
        trim_frac: float = 0.0,
        trim_warmup_frac: float = 0.1,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.lrs_config = lrs_config
        self.opt_name = optimizer
        if "betas" in lrs_config:
            self.opt_betas = tuple(lrs_config["betas"])
        else:
            # Sensible default per-optimizer: AdamW uses (0.9, 0.999), Lion
            # uses (0.9, 0.99).  configure_optimizers() only forwards this to
            # the optimizer if the user actually set ``betas``.
            self.opt_betas = (0.9, 0.999)

        # SAM + batch-trimming hyperparameters.  When ``use_sam`` is True the
        # wrapper switches to manual optimization so we can run two
        # forward/backward passes per step (ascent + descent).  Trimming can
        # operate independently of SAM: if ``use_sam=False`` but
        # ``trim_frac>0``, we still need manual optimization because
        # automatic mode would require re-computing the trim mask from the
        # same loss tensor that's being backprop'd, which is awkward.
        self.use_sam = bool(use_sam)
        self.sam_rho = float(sam_rho)
        self.sam_adaptive = bool(sam_adaptive)
        self.sam_start_epoch = int(sam_start_epoch)
        self.trim_frac = float(trim_frac)
        self.trim_warmup_frac = float(trim_warmup_frac)
        self.gradient_clip_val = float(gradient_clip_val)

        # Lightning: switch to manual optimization for SAM / trim path.
        # All other runs retain the default automatic optimization.
        self._manual_opt_needed = self.use_sam or self.trim_frac > 0.0
        if self._manual_opt_needed:
            self.automatic_optimization = False

        # Load model weights only (no optimizer/scheduler state) for fine-tuning.
        # Use this instead of --ckpt_path when you want a fresh optimizer and LR schedule.
        if pretrained_ckpt_path is not None:
            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
            # Lightning checkpoints prefix all keys with "model." (from self.model = model)
            state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            missing, unexpected = self.model.load_state_dict(state, strict=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint weight mismatch.\nMissing: {missing}\nUnexpected: {unexpected}"
                )
            print(f"[fine-tune] Loaded model weights from {pretrained_ckpt_path}")

    def setup(self, stage: str) -> None:
        """Log a one-shot summary of architecture/parameter/precision state.

        Prints are gated on ``trainer.is_global_zero`` to avoid duplication
        under DDP.  Only fires for ``fit`` stage (not sanity-check/validate).
        """
        if stage != "fit":
            return

        # SAM + gradient accumulation is non-trivial (needs boundary-gating
        # the two-pass ascent/descent).  Reject the combination loudly.
        if self._manual_opt_needed:
            accum = int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1)
            if accum != 1:
                raise RuntimeError(
                    "TrackRegressionWrapper manual-optimization path "
                    "(SAM / batch trimming) requires "
                    "trainer.accumulate_grad_batches == 1, got "
                    f"{accum}. Reduce the batch size instead."
                )

        if not getattr(self.trainer, "is_global_zero", True):
            return

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pool = getattr(self.model, "pool", "unknown")
        autocast_dtype = getattr(self.model, "encoder_autocast_dtype", None)
        try:
            precision = self.trainer.precision
        except RuntimeError:
            precision = "<not attached>"

        print("-" * 80)
        print(f"[{self.name}] Pool type:             {pool}")
        print(f"[{self.name}] Trainer precision:     {precision}")
        print(f"[{self.name}] Encoder autocast:      {autocast_dtype}")
        print(f"[{self.name}] Total parameters:      {total / 1e6:.3f} M")
        print(f"[{self.name}] Trainable parameters:  {trainable / 1e6:.3f} M")
        print(f"[{self.name}] Manual optimization:   {self._manual_opt_needed}")
        if self.use_sam:
            print(
                f"[{self.name}] SAM:                   rho={self.sam_rho} "
                f"adaptive={self.sam_adaptive} start_epoch={self.sam_start_epoch}"
            )
        if self.trim_frac > 0.0:
            warmup_epochs = int(self.trim_warmup_frac * max(1, int(self.trainer.max_epochs)))
            print(
                f"[{self.name}] Batch trimming:        drop {self.trim_frac:.1%} "
                f"after epoch {warmup_epochs} ({self.trim_warmup_frac:.0%} warmup)"
            )
        print("-" * 80)

    # -- forward / predict --------------------------------------------------

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        preds = self.model.loss_module.predict_physical(outputs["pred"], targets)
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
        crossing_metrics = self.model.loss_module.quantile_crossing_metrics(outputs["pred"], valid_mask=valid_mask)
        calibration_metrics = self.model.loss_module.quantile_calibration_metrics(outputs["pred"], targets, valid_mask=valid_mask)

        # Only use sync_dist for val/test — training metrics are local per-rank
        # averages.  Using sync_dist=True during training adds many NCCL
        # allreduces per step which can cause DDP synchronisation issues.
        do_sync = stage != "train"

        # Log every component
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, sync_dist=do_sync, prog_bar=(name == "total"))

        # Monitor quantile crossings on raw (unconstrained) quantile channels
        for name, value in crossing_metrics.items():
            self.log(f"{stage}/{name}", value, sync_dist=do_sync)

        # Monitor quantile calibration (empirical coverage vs nominal levels)
        for name, value in calibration_metrics.items():
            self.log(f"{stage}/{name}", value, sync_dist=do_sync)

        # MDN component diagnostics (π_k, μ_k, σ_k per mixture component)
        self._log_mdn_components(outputs["pred"], valid_mask, stage)

        # Compute and log per-parameter metrics for all stages.
        # Use predict_physical to add back delta anchors for correct residuals.
        preds = self.model.loss_module.predict_physical(outputs["pred"], targets)
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
        """Log per-parameter metrics: MAE, precision, and SSM precision on tight DM subset.

        Metrics
        -------
        - ``{stage}/{name}/mae``: mean absolute error (all stages)
        - ``{stage}/{name}/precision {unit}``: std of (pred - truth) residuals,
          scaled to physical units (all stages)
        - ``{stage}/{name}/ssm_precision_dm {unit}``: SSM precision on
          the double-matched subset — val/test only (needs ACTS DM data).
        - ``{stage}/{name}/ssm_iqr_dm {unit}``: robust σ estimator
          ``(Q75 - Q25) / 1.349`` on the double-matched subset — val/test.
        - ``{stage}/{name}/ssm_rms_dm {unit}``: RMS of the residuals on the
          double-matched subset — val/test.

        ``_dm`` metrics filter to tracks that are **both** ``valid`` AND
        ``acts_dm_mask`` (ACTS Kalman Filter purity > 0.75 AND efficiency
        > 0.75).  The residual is always ``p - t`` against the truth
        target ``t`` — ACTS reco values are only used as the presence
        guard.  Any change here must preserve this double-matched filter
        semantics so that the three DM metrics compare apples to apples.

        ACTS precision is never logged (precomputed / static).
        """
        do_sync = stage != "train"

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

            # Match evaluation script behavior: wrap phi residuals to [-pi, pi].
            if name == "phi":
                residual = torch.where(residual > math.pi, residual - 2.0 * math.pi, residual)
                residual = torch.where(residual < -math.pi, residual + 2.0 * math.pi, residual)

            # MAE
            self.log(f"{stage}/{name}/mae", residual.abs().mean(), sync_dist=do_sync)

            # Precision: std of residuals in physical units
            if residual.numel() > 1:
                unit, scale = self._PRECISION_UNITS.get(name, ("", 1.0))
                self.log(
                    f"{stage}/{name}/precision {unit}",
                    residual.std() * scale,
                    sync_dist=do_sync,
                )

            # SSM precision on DM subset — val/test only (train has no ACTS data).
            # See the _log_metrics docstring for the exact DM filter semantics.
            if stage in ("val", "test"):
                acts_key = f"acts_reco_{name}"
                if acts_key in targets and "acts_dm_mask" in targets:
                    dm_mask = targets["acts_dm_mask"]
                    if valid_mask is not None:
                        dm_mask = dm_mask[valid_mask]

                    if dm_mask.any():
                        p_dm = p[dm_mask]
                        t_dm = t[dm_mask]
                        ssm_residual_dm = p_dm - t_dm
                        if name == "phi":
                            ssm_residual_dm = torch.where(ssm_residual_dm > math.pi, ssm_residual_dm - 2.0 * math.pi, ssm_residual_dm)
                            ssm_residual_dm = torch.where(ssm_residual_dm < -math.pi, ssm_residual_dm + 2.0 * math.pi, ssm_residual_dm)
                        if ssm_residual_dm.numel() > 1:
                            unit, scale = self._PRECISION_UNITS.get(name, ("", 1.0))
                            # σ — standard deviation (shared DM mask construction)
                            self.log(
                                f"{stage}/{name}/ssm_precision_dm {unit}",
                                ssm_residual_dm.std() * scale,
                                sync_dist=True,
                            )
                            # IQR/1.349 — robust σ estimator (tail-robust)
                            quant_levels = torch.tensor(
                                [0.25, 0.75],
                                device=ssm_residual_dm.device,
                                dtype=ssm_residual_dm.dtype,
                            )
                            q25, q75 = torch.quantile(ssm_residual_dm, quant_levels)
                            self.log(
                                f"{stage}/{name}/ssm_iqr_dm {unit}",
                                (q75 - q25) / 1.349 * scale,
                                sync_dist=True,
                            )
                            # RMS — captures outlier contribution (tail-sensitive)
                            rms = torch.sqrt((ssm_residual_dm ** 2).mean())
                            self.log(
                                f"{stage}/{name}/ssm_rms_dm {unit}",
                                rms * scale,
                                sync_dist=True,
                            )

    def _log_mdn_components(
        self,
        raw: Tensor,
        valid_mask: Tensor | None,
        stage: str,
    ) -> None:
        """Log per-component MDN diagnostics (mean π, μ, σ, effective K, MAP fraction).

        For every parameter whose loss is a ``MixtureDensityLoss``, record
        one scalar per component plus two aggregate diagnostics:
          - ``{stage}/{name}/mdn/pi_{k}``       — batch-mean mixture weight
          - ``{stage}/{name}/mdn/mu_{k}``       — batch-mean predicted location [physical units]
          - ``{stage}/{name}/mdn/sigma_{k}``    — batch-mean predicted spread  [physical units]
          - ``{stage}/{name}/mdn/effective_k``  — ⟨1/Σₖπₖ²⟩, measures how balanced π is
          - ``{stage}/{name}/mdn/map_frac_{k}`` — fraction of tracks whose MAP component is k

        Watching these during training diagnoses two failure modes:
          * One component capturing all tracks (`map_frac_{k} → 1`,
            `effective_k → 1`) → collapse back to single-Gaussian behaviour.
          * A component going dead (`pi_{k} → 0`, unchanging μ/σ) → K is
            too large; reduce in the next run.
        """
        loss_module = self.model.loss_module
        if not any(isinstance(sub, MixtureDensityLoss) for sub in loss_module.losses.values()):
            return

        do_sync = stage != "train"
        for name, sub_loss in loss_module.losses.items():
            if not isinstance(sub_loss, MixtureDensityLoss):
                continue
            start, end = loss_module.get_output_slice(name)
            r = raw[..., start:end]
            if valid_mask is not None:
                r = r[valid_mask]
            if r.numel() == 0:
                continue

            comps = sub_loss.predict_components(r)
            pi = comps["pi"]        # (N, K)
            mu = comps["mu"]        # (N, K) physical
            sigma = comps["sigma"]  # (N, K) physical
            K = sub_loss.n_components

            for k in range(K):
                self.log(f"{stage}/{name}/mdn/pi_{k}", pi[..., k].mean(), sync_dist=do_sync)
                self.log(f"{stage}/{name}/mdn/mu_{k}", mu[..., k].mean(), sync_dist=do_sync)
                self.log(f"{stage}/{name}/mdn/sigma_{k}", sigma[..., k].mean(), sync_dist=do_sync)

            pi_sq_sum = (pi ** 2).sum(dim=-1).clamp_min(1e-12)
            self.log(
                f"{stage}/{name}/mdn/effective_k",
                (1.0 / pi_sq_sum).mean(),
                sync_dist=do_sync,
            )

            k_map = pi.argmax(dim=-1)
            for k in range(K):
                self.log(
                    f"{stage}/{name}/mdn/map_frac_{k}",
                    (k_map == k).float().mean(),
                    sync_dist=do_sync,
                )

    # -- train / val / test ------------------------------------------------

    def training_step(self, batch, batch_idx):
        # Fast path — plain Lightning automatic optimization.
        if self.automatic_optimization:
            return {"loss": self._shared_step(batch, "train")}

        # Manual path — SAM and/or batch trimming.
        return self._manual_training_step(batch, batch_idx)

    def _manual_training_step(self, batch, batch_idx):
        """Manual-optimization training step for SAM + batch trimming.

        Flow:

        1. **Forward 1** on the full batch.  Build the trim mask (if active)
           from a detached per-sample loss tensor, *then* compute the loss
           that will actually be backpropped.  This avoids a double
           ``compute_loss`` on the same forward in the trim-only path.
        2. If SAM is active: the ascent loss is deliberately the **untrimmed**
           full-batch loss (plan §3, Forward 1 is "no trim") — we want SAM
           to find a flat minimum over the full batch, then apply the trim
           only at the descent step.  ``manual_backward(loss)`` (wrapped in
           ``no_sync()`` under DDP to save an unnecessary all-reduce),
           clip, ``opt.first_step``.  Forward 2 at the perturbed weights
           runs with ``trim_mask`` applied and drives ``opt.second_step``.
        3. If SAM is NOT active (warmup phase or purely trim-only run):
           we do a single forward whose loss is already the trimmed loss,
           one backward, one ``opt.step``.
        4. Advance the LR scheduler manually (Lightning does not auto-step
           schedulers under manual optimization).
        5. Mirror the metric / loss logging that ``_shared_step`` emits so
           the manual path lights up the same CometML panels.
        """
        opt = self.optimizers()
        sch = self.lr_schedulers()

        inputs, targets = batch
        valid_mask = targets.get("track_valid")

        sam_active = self.use_sam and self.current_epoch >= self.sam_start_epoch
        trim_active = (
            self.trim_frac > 0.0
            and self.current_epoch
            >= int(self.trim_warmup_frac * max(1, int(self.trainer.max_epochs)))
        )

        # ---- Forward 1 -----------------------------------------------------
        outputs = self.model(inputs)

        # Build the trim mask (no_grad — just a ranking).
        trim_mask: Tensor | None = None
        if trim_active:
            with torch.no_grad():
                per_sample = self.model.loss_module.per_sample_total(
                    outputs["pred"], targets, valid_mask=valid_mask
                )
                if per_sample.numel() > 0:
                    # keep_k = number of samples to retain (the smallest
                    # (1 - trim_frac) fraction).  ``kthvalue`` is O(n)
                    # average — cheaper than a full sort.
                    keep_k = max(1, int((1.0 - self.trim_frac) * per_sample.numel()))
                    thresh = per_sample.kthvalue(keep_k).values
                    trim_mask = (per_sample <= thresh).to(per_sample.dtype)

        if sam_active:
            # Ascent direction — full-batch (untrimmed) loss.
            losses = self.model.compute_loss(outputs, targets, valid_mask=valid_mask)

            # DDP bandwidth optimisation: the ascent grads are discarded
            # after ``first_step``, so there is no need to all-reduce them.
            # ``no_sync`` is only valid under a DDP strategy.
            if self.trainer.world_size > 1 and hasattr(
                getattr(self.trainer.strategy, "model", None), "no_sync"
            ):
                no_sync_cm = self.trainer.strategy.model.no_sync()
            else:
                no_sync_cm = nullcontext()
            with no_sync_cm:
                self.manual_backward(losses["total"])
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            opt.first_step(zero_grad=True)

            # ---- Forward 2 (at perturbed weights, trim applied) ------------
            outputs = self.model(inputs)
            losses = self.model.compute_loss(
                outputs, targets, valid_mask=valid_mask, trim_mask=trim_mask
            )
            self.manual_backward(losses["total"])
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            opt.second_step(zero_grad=True)
        else:
            # Non-SAM path: compute the (optionally trimmed) loss once and
            # take a single base-optimizer step.  This reuses Forward 1's
            # autograd graph via ``outputs``, so there is no wasted forward
            # even when a trim mask is in play.
            losses = self.model.compute_loss(
                outputs, targets, valid_mask=valid_mask, trim_mask=trim_mask
            )
            self.manual_backward(losses["total"])
            self.clip_gradients(
                opt,
                gradient_clip_val=self.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Advance the LR scheduler.  Lightning does NOT auto-step schedulers
        # in manual mode, even when they're returned from configure_optimizers
        # with ``interval: step``, so we call step() ourselves.  Our
        # configure_optimizers always returns a single SequentialLR, so this
        # is never a list under manual optimization.
        if sch is not None:
            sch.step()

        # ---- Logging (mirror _shared_step) --------------------------------
        # ``outputs`` / ``losses`` above now refer to the final (post-step)
        # forward: in SAM mode these are Forward 2 at perturbed weights; in
        # the non-SAM path they are the single trimmed forward.
        for name, value in losses.items():
            self.log(
                f"train/{name}",
                value,
                sync_dist=False,
                prog_bar=(name == "total"),
            )

        # quantile_crossing_metrics / quantile_calibration_metrics return
        # empty dicts when the loss is non-quantile (gaussian / cosine_phi).
        crossing_metrics = self.model.loss_module.quantile_crossing_metrics(
            outputs["pred"], valid_mask=valid_mask
        )
        for name, value in crossing_metrics.items():
            self.log(f"train/{name}", value, sync_dist=False)
        calibration_metrics = self.model.loss_module.quantile_calibration_metrics(
            outputs["pred"], targets, valid_mask=valid_mask
        )
        for name, value in calibration_metrics.items():
            self.log(f"train/{name}", value, sync_dist=False)

        # MDN component diagnostics (mirrors _shared_step) — no-op when no MDN in play.
        self._log_mdn_components(outputs["pred"], valid_mask, "train")

        # Per-parameter MAE / precision / IQR / RMS logging.
        preds = self.model.loss_module.predict_physical(outputs["pred"], targets)
        self._log_metrics(preds, targets, valid_mask, "train")

        return {"loss": losses["total"]}

    def validation_step(self, batch, batch_idx):
        return {"loss": self._shared_step(batch, "val")}

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        valid_mask = targets.get("track_valid")
        losses = self.model.compute_loss(outputs, targets, valid_mask=valid_mask)
        crossing_metrics = self.model.loss_module.quantile_crossing_metrics(outputs["pred"], valid_mask=valid_mask)
        calibration_metrics = self.model.loss_module.quantile_calibration_metrics(outputs["pred"], targets, valid_mask=valid_mask)

        # Log every component
        for name, value in losses.items():
            self.log(f"test/{name}", value, sync_dist=True, prog_bar=(name == "total"))

        # Monitor quantile crossings on raw (unconstrained) quantile channels
        for name, value in crossing_metrics.items():
            self.log(f"test/{name}", value, sync_dist=True)

        # Monitor quantile calibration
        for name, value in calibration_metrics.items():
            self.log(f"test/{name}", value, sync_dist=True)

        # Compute predictions and metrics (predict_physical adds back delta anchors)
        preds = self.model.loss_module.predict_physical(outputs["pred"], targets)
        self._log_metrics(preds, targets, valid_mask, "test")

        # Full quantile predictions (for quantile-based losses)
        quantile_preds = self.model.loss_module.predict_quantiles(outputs["pred"])

        return {
            "loss": losses["total"],
            "preds": preds,
            "targets": targets,
            "quantile_preds": quantile_preds,
        }

    # -- optimiser / scheduler ---------------------------------------------

    def configure_optimizers(self):
        if self.opt_name.lower() == "muonhybrid":
            # Muon for 2-D interior matrix weights, AdamW for 1-D params
            # (norms, biases, SSM A_log/D, etc.) plus the input_net,
            # pool_head, and output_head Dense stacks.  The split is handled
            # by :func:`split_params_for_muon`; each group's initial LR is
            # set to its *peak* (``muon_max`` and ``max``) so the WSD / cosine
            # schedulers scale both groups proportionally from their own peak.
            # The LR ratio (``initial`` / ``max``) is shared across groups,
            # so the user must pick ``muon_max`` and ``max`` such that their
            # warmup-start and final LRs line up with the single scheduler
            # factor — typically ``muon_max ≈ 3–10× max`` per the Muon blog.
            if self.use_sam or self._manual_opt_needed:
                raise ValueError("MuonHybrid is not compatible with SAM/trim manual-opt paths.")
            muon_peak = float(self.lrs_config["muon_max"])
            adamw_peak = float(self.lrs_config["max"])
            param_groups = split_params_for_muon(
                self.model,
                muon_lr=muon_peak,
                muon_weight_decay=float(self.lrs_config.get("muon_weight_decay", self.lrs_config["weight_decay"])),
                adamw_lr=adamw_peak,
                adamw_weight_decay=float(self.lrs_config["weight_decay"]),
                adamw_betas=self.opt_betas if "betas" in self.lrs_config else (0.9, 0.95),
            )
            opt = MuonHybrid(
                param_groups,
                lr=adamw_peak,  # group-level lrs override this default
                momentum=float(self.lrs_config.get("muon_momentum", 0.95)),
                ns_steps=int(self.lrs_config.get("muon_ns_steps", 5)),
            )
        else:
            if self.opt_name.lower() == "adamw":
                opt_cls = AdamW
            elif self.opt_name.lower() == "lion":
                opt_cls = Lion
            else:
                raise ValueError(f"Unknown optimizer: {self.opt_name}")

            opt_kwargs: dict[str, Any] = dict(
                lr=self.lrs_config["initial"],
                weight_decay=self.lrs_config["weight_decay"],
            )
            # Both AdamW and lion_pytorch.Lion accept a ``betas`` kwarg; Lion
            # defaults to (0.9, 0.99) vs AdamW's (0.9, 0.999).  If the user did not
            # set betas in lrs_config we skip passing them so each optimizer keeps
            # its own default.
            if "betas" in self.lrs_config:
                opt_kwargs["betas"] = self.opt_betas

            if self.use_sam:
                # SAM wraps the base optimizer and shares its param_groups, so
                # any LR scheduler written against ``opt.param_groups`` (cosine,
                # onecycle, cosine_freeze below) will transparently drive the
                # underlying AdamW/Lion without any scheduler-side changes.
                opt = SAM(
                    self.model.parameters(),
                    base_optimizer=opt_cls,
                    rho=self.sam_rho,
                    adaptive=self.sam_adaptive,
                    **opt_kwargs,
                )
            else:
                opt = opt_cls(self.model.parameters(), **opt_kwargs)

        if not self.lrs_config.get("skip_scheduler"):
            schedule = self.lrs_config.get("schedule", "onecycle")
            total_steps = self.trainer.estimated_stepping_batches

            if schedule == "cosine":
                # Linear warmup + cosine annealing
                # Set optimizer LR to max BEFORE creating schedulers so that
                # base_lrs is captured correctly.  LinearLR's start_factor then
                # scales it down to `initial` at step 0, ramping up to `max`.
                for pg in opt.param_groups:
                    pg["lr"] = self.lrs_config["max"]
                warmup_steps = int(float(self.lrs_config["pct_start"]) * total_steps)
                warmup_sch = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=self.lrs_config["initial"] / self.lrs_config["max"],
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=total_steps - warmup_steps,
                    eta_min=self.lrs_config["end"],
                )
                sch = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sch, cosine_sch],
                    milestones=[warmup_steps],
                )
            elif schedule == "cosine_freeze":
                # Three-phase schedule for frozen-backbone fine-tuning:
                #   Phase 1: [0, warmup_steps)        LinearLR  initial → max    (frozen)
                #   Phase 2: [warmup_steps, unfreeze) ConstantLR at max          (frozen)
                #   Phase 3: [unfreeze, total_steps)  CosineAnnealingLR
                #                                     unfreeze_lr → end         (unfrozen)
                # The unfreeze happens via :class:`MambaBackboneFreeze`
                # (sets requires_grad).  This schedule drops the LR at the
                # unfreeze milestone so that the now-trainable backbone
                # doesn't receive the large (max) LR.
                warmup_steps = int(float(self.lrs_config["pct_start"]) * total_steps)
                unfreeze_epoch = int(self.lrs_config["unfreeze_epoch"])
                max_epochs = max(1, int(self.trainer.max_epochs))
                steps_per_epoch = max(1, total_steps // max_epochs)
                unfreeze_step = unfreeze_epoch * steps_per_epoch
                unfreeze_lr = float(self.lrs_config["unfreeze_lr"])

                if not (0 < warmup_steps <= unfreeze_step < total_steps):
                    raise ValueError(
                        "Invalid cosine_freeze schedule: "
                        f"warmup_steps={warmup_steps}, unfreeze_step={unfreeze_step}, "
                        f"total_steps={total_steps} — require "
                        "0 < warmup_steps <= unfreeze_step < total_steps"
                    )

                # Capture base_lr = max at scheduler construction time so that
                # LinearLR can scale it down to `initial` via start_factor.
                for pg in opt.param_groups:
                    pg["lr"] = self.lrs_config["max"]

                warmup_sch = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=self.lrs_config["initial"] / self.lrs_config["max"],
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                hold_sch = torch.optim.lr_scheduler.ConstantLR(
                    opt,
                    factor=1.0,
                    total_iters=unfreeze_step - warmup_steps,
                )
                cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=total_steps - unfreeze_step,
                    eta_min=self.lrs_config["end"],
                )
                # Override the cosine scheduler's base_lrs to start decay from
                # unfreeze_lr (typically ~10x lower than max).  CosineAnnealingLR
                # reads from `self.base_lrs` at every step, so this reliably
                # produces `unfreeze_lr → end` over the remaining steps
                # regardless of what the optimizer's param-group LR currently is.
                cosine_sch.base_lrs = [unfreeze_lr for _ in cosine_sch.base_lrs]

                sch = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sch, hold_sch, cosine_sch],
                    milestones=[warmup_steps, unfreeze_step],
                )
            elif schedule == "wsd":
                # Warmup-Stable-Decay (Hägele et al. 2024, arXiv:2405.18392).
                # Three phases, all driven by a single SequentialLR:
                #   [0, warmup)             LinearLR   initial → max
                #   [warmup, decay_start)   ConstantLR at max
                #   [decay_start, total)    CosineAnnealingLR  max → end
                # Recommended default for continuing from a converged checkpoint
                # (domain-shift fine-tuning): short warmup (pct_start ≈ 0.02),
                # long stable plateau, short cosine cooldown (decay_pct ≈ 0.15).
                # For multi-group optimizers (MuonHybrid) each group keeps its
                # own peak LR — LinearLR / CosineAnnealingLR apply the same
                # relative factor to every group's base_lr.
                warmup_pct = float(self.lrs_config.get("pct_start", 0.02))
                decay_pct = float(self.lrs_config.get("decay_pct", 0.15))
                warmup_steps = int(warmup_pct * total_steps)
                decay_start = int((1.0 - decay_pct) * total_steps)
                if not (0 < warmup_steps < decay_start < total_steps):
                    raise ValueError(
                        "Invalid WSD schedule: "
                        f"warmup_steps={warmup_steps}, decay_start={decay_start}, "
                        f"total_steps={total_steps} — require "
                        "0 < warmup_steps < decay_start < total_steps"
                    )

                if len(opt.param_groups) == 1:
                    # Single-group path: mirror the cosine branch and set the
                    # base LR to `max` so LinearLR's start_factor scales it down
                    # to `initial` at step 0.
                    for pg in opt.param_groups:
                        pg["lr"] = self.lrs_config["max"]
                # else: MuonHybrid — each group's 'lr' was set to its own peak
                # at construction time via split_params_for_muon, so we leave
                # them alone and LinearLR applies the same factor per group.

                start_factor = float(self.lrs_config["initial"]) / float(self.lrs_config["max"])
                warmup_sch = torch.optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
                hold_sch = torch.optim.lr_scheduler.ConstantLR(
                    opt,
                    factor=1.0,
                    total_iters=decay_start - warmup_steps,
                )
                cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=total_steps - decay_start,
                    eta_min=float(self.lrs_config["end"]),
                )
                sch = torch.optim.lr_scheduler.SequentialLR(
                    opt,
                    schedulers=[warmup_sch, hold_sch, cosine_sch],
                    milestones=[warmup_steps, decay_start],
                )
            else:
                sch = torch.optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=self.lrs_config["max"],
                    total_steps=total_steps,
                    div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                    final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                    pct_start=float(self.lrs_config["pct_start"]),
                )

            return [opt], [{"scheduler": sch, "interval": "step"}]

        return opt
