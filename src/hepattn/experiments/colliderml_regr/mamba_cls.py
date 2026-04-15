"""Bidirectional Mamba-2 encoder with learned CLS tokens.

This is an alternative pooling strategy to :mod:`mamba_state`.  Instead of
extracting the final SSM recurrent hidden state (``nheads * headdim *
d_state`` per direction), we learn two CLS tokens that are inserted at the
terminal positions of each scan direction:

- ``cls_fwd`` is **appended** to the sorted sequence so the forward SSM
  sees it *last* — after accumulating state across every hit.
- ``cls_bwd`` is **prepended** to the sorted sequence so that, after the
  ``torch.flip`` inside :class:`BidirectionalMambaLayer`, it becomes the
  last token the backward SSM sees — again, after accumulating state
  across every hit.

Both CLS tokens flow through all ``num_layers`` encoder layers so they
accumulate representation across depth, ViT/Vision-Mamba style.  The
final layer is a custom :class:`BidirectionalMambaCLSFinalLayer` that
exposes the per-direction Mamba-2 outputs at the CLS positions *before*
the gated bidirectional merge — which would otherwise contaminate the
two readouts with their opposite-direction partner's irrelevant
"only-seen-one-token" output.

The encoder returns a pooled tensor of shape ``(B, 2 * dim)`` formed by
concatenating ``(cls_fwd_out, cls_bwd_out)``, alongside the (sorted, then
un-sorted) per-hit sequence output.  Paired with the
``pool='ssm_cls'`` branch of :class:`TrackParameterRegressor`, this
bypasses the ``state_head`` projection entirely (the output_head takes
``2 * dim``-dim input directly).

DDP unused-parameter tie
------------------------
If the downstream model consumes only the CLS output and discards the
per-hit sequence output, the ``gate`` parameters inside each layer would
have no gradient — causing DDP to raise "unused parameter" errors.  To
avoid this, the forward adds ``0.0 * sequence_output.sum()`` to the CLS
output, which forces the sequence-output path (and therefore all
internal layer parameters) into the autograd graph without changing the
numerical value.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from mamba_ssm.modules.mamba2 import Mamba2

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba2 = nn.Module  # type: ignore[assignment, misc]

from hepattn.experiments.colliderml_regr.mamba_state import BidirectionalMambaLayer


class BidirectionalMambaCLSFinalLayer(nn.Module):
    """Final bidirectional Mamba layer that exposes per-direction CLS readouts.

    Structurally identical to :class:`BidirectionalMambaLayer` (norm →
    forward Mamba-2 + backward Mamba-2 → gated merge → residual) but in
    addition to the gated sequence output it returns:

    - ``cls_fwd_out = x_fwd[:, -1, :]`` — the forward Mamba-2's output at
      the terminal position of the sequence (where ``cls_fwd`` lives).
      This is ungated (and therefore uncontaminated by the backward
      output at the same position, which has only seen one token).
    - ``cls_bwd_out = x_bwd[:, 0, :]`` — the backward Mamba-2's output at
      position 0 after the post-scan flip, which is the flipped-scan's
      terminal position (where ``cls_bwd`` lives after flipping).

    Layout convention expected on input: ``(cls_bwd, h_0, …, h_{L-1},
    cls_fwd)`` of shape ``(B, L+2, D)``.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.norm = nn.RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm: {norm}")

        mamba_kwargs = {
            "d_model": dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
            "ngroups": ngroups,
            "chunk_size": chunk_size,
        }
        self.forward_mamba = Mamba2(**mamba_kwargs)
        self.backward_mamba = Mamba2(**mamba_kwargs)

        self.gate = nn.Linear(dim, dim)
        self.gate_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: Tensor,
        seq_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        skip = x
        x_norm = self.norm(x).contiguous()

        x_fwd = self.forward_mamba(x_norm, seq_idx=seq_idx)  # (B, L+2, D)

        x_bwd_in = torch.flip(x_norm, dims=[1]).contiguous()
        x_bwd_flipped = self.backward_mamba(x_bwd_in, seq_idx=seq_idx)  # (B, L+2, D)
        x_bwd = torch.flip(x_bwd_flipped, dims=[1])  # (B, L+2, D), original order

        gate = self.gate_activation(self.gate(x_norm))
        x_combined = gate * x_fwd + (1 - gate) * x_bwd
        output = skip + self.dropout(x_combined)

        # Per-direction CLS readouts — ungated.
        cls_fwd_out = x_fwd[:, -1, :]   # forward scan terminal → cls_fwd
        cls_bwd_out = x_bwd[:, 0, :]    # backward scan terminal (post-flip) → cls_bwd

        return output, cls_fwd_out, cls_bwd_out


class BidirectionalMambaCLSEncoder(nn.Module):
    """Bidirectional Mamba-2 encoder with learned CLS tokens.

    Stacks ``num_layers - 1`` plain :class:`BidirectionalMambaLayer`
    layers followed by one :class:`BidirectionalMambaCLSFinalLayer`.  All
    layers see the CLS-augmented sequence ``(cls_bwd, hits, cls_fwd)``;
    the CLS tokens therefore accumulate representation across depth.

    Parameters
    ----------
    num_layers, dim, d_state, d_conv, expand, headdim, ngroups, chunk_size, norm, dropout
        See :class:`BidirectionalMambaLayer`.
    cls_init_scale : float
        Standard deviation of the CLS-token initialisation.
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
        cls_init_scale: float = 0.02,
    ):
        super().__init__()
        assert num_layers >= 1, "Need at least one layer"

        self.num_layers = num_layers
        self.dim = dim

        # Learned CLS tokens.  Initialised with a small Gaussian.
        self.cls_fwd = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)
        self.cls_bwd = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)

        common = dict(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            norm=norm,
            dropout=dropout,
        )

        # Intermediate layers (plain bidirectional, CLS passes through as just
        # two more tokens — no special handling needed at intermediate depth).
        self.layers = nn.ModuleList(
            [BidirectionalMambaLayer(**common) for _ in range(num_layers - 1)]
        )

        # Final layer returns per-direction CLS outputs without gating.
        self.final_layer = BidirectionalMambaCLSFinalLayer(**common)

        # Post-encoder normalisation on the full augmented sequence.
        if norm == "LayerNorm":
            self.final_norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.final_norm = nn.RMSNorm(dim)
        else:
            self.final_norm = nn.Identity()

    @property
    def pool_dim(self) -> int:
        """Dimension of the concatenated ``(cls_fwd, cls_bwd)`` pooled output."""
        return 2 * self.dim

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        seq_idx: Tensor | None = None,
        **kwargs,  # noqa: ARG002 — API compatibility with other encoders
    ) -> tuple[Tensor, Tensor]:
        """Encode a sequence and return ``(seq_output, cls_concat)``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, N, D)``.
        x_sort_value : Tensor | None
            Values to sort tokens by (e.g. ``s`` — distance from IP).
            Tokens are sorted before processing and un-sorted after; CLS
            tokens are inserted *after* sorting and stripped *before*
            un-sorting so they never participate in the permutation.
        seq_idx : Tensor | None
            Optional per-token sequence index (unused for padded batches;
            defaults to ``None``).

        Returns
        -------
        tuple[Tensor, Tensor]
            - Sequence output ``(B, N, D)`` (in original token order; CLS
              tokens stripped).
            - Pooled CLS summary ``(B, 2 * D)``, formed by concatenating
              ``(cls_fwd_out, cls_bwd_out)``.
        """
        B = x.shape[0]

        # Optional sort (e.g. by distance from IP).  CLS tokens are inserted
        # AFTER this step — the plan calls this out explicitly because
        # appending/prepending before sort would scramble the CLS positions.
        x_sort_idx = None
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x)).contiguous()

        # Insert CLS tokens at the terminal positions of each scan direction.
        # Layout: (cls_bwd, h_0, ..., h_{L-1}, cls_fwd).
        cls_bwd_tok = self.cls_bwd.expand(B, -1, -1).to(dtype=x.dtype)
        cls_fwd_tok = self.cls_fwd.expand(B, -1, -1).to(dtype=x.dtype)
        x_aug = torch.cat([cls_bwd_tok, x, cls_fwd_tok], dim=1).contiguous()

        # Intermediate layers — CLS tokens flow through unchanged.
        for layer in self.layers:
            x_aug = layer(x_aug, seq_idx=seq_idx)

        # Final layer returns per-direction CLS readouts ungated.
        x_aug, cls_fwd_out, cls_bwd_out = self.final_layer(x_aug, seq_idx=seq_idx)

        x_aug = self.final_norm(x_aug)

        # Strip CLS tokens from the sequence output — keep only the hit positions.
        x_hits = x_aug[:, 1:-1, :]

        # Un-sort hits back to original order.
        if x_sort_idx is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            x_hits = torch.gather(x_hits, -2, x_unsort_idx.unsqueeze(-1).expand_as(x_hits))

        # Concatenate the per-direction CLS readouts → (B, 2*dim).
        cls_concat = torch.cat([cls_fwd_out, cls_bwd_out], dim=-1)

        # DDP unused-parameter tie: pull the per-hit sequence-output path
        # (and therefore the `gate` weights inside each layer) into the
        # autograd graph even when the downstream model discards x_hits.
        # Numerically a no-op — just a 0.0-scaled sum.
        cls_concat = cls_concat + 0.0 * x_hits.sum()

        return x_hits, cls_concat
