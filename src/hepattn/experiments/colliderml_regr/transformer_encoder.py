"""Transformer encoder baseline with a learned CLS token.

This is a thin wrapper around :class:`hepattn.models.Encoder` that:

- Adds a single learned ``cls_token`` at the front of the sequence.
- Returns ``(sequence_output, cls_tensor)`` instead of just the sequence
  output — matching the API of the SSM encoders so that
  :class:`TrackParameterRegressor` can use the same ``pool='register_token'``
  pathway.

Design: composition, not inheritance.
--------------------------------------
Rather than subclassing :class:`Encoder` and duplicating its sophisticated
forward logic (flash-varlen unpadding, sliding window masks, sorting,
etc.), we *contain* an ``Encoder`` that is constructed without its own
register-token support (``num_register_tokens=None``).  ``EncoderWithCLS``
then performs:

1. Optional sort-by-``x_sort_value`` (same convention as the SSM encoder).
2. Prepend the learned CLS token to the sorted sequence (and the
   corresponding mask if present).
3. Call the inner :class:`Encoder`'s ``forward`` with the CLS-augmented
   sequence — but with ``x_sort_value=None`` so the inner encoder does
   not re-sort.
4. Split the output into ``(cls_tensor, hit_output)`` and un-sort the hit
   output back to the original input order.
5. Return ``(hit_output, cls_tensor)``.

This keeps the shared :class:`Encoder` untouched — the trackml experiment
is unaffected.

Also includes a DDP unused-parameter tie (``cls_out + 0.0 *
hit_output.sum()``) so that every encoder parameter contributes to the
autograd graph even when the downstream model only consumes the CLS
output.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from hepattn.models.encoder import Encoder


class EncoderWithCLS(nn.Module):
    """Transformer encoder that returns both the sequence output and a CLS token.

    Parameters
    ----------
    dim : int
        Model dimension (embedding size).
    cls_init_scale : float
        Standard deviation of the learned CLS token initialisation.
    **encoder_kwargs
        All other kwargs are forwarded to :class:`hepattn.models.Encoder`.
        ``num_register_tokens`` is silently dropped if present — the CLS
        token is managed at this layer, not by the inner Encoder.
    """

    def __init__(
        self,
        dim: int,
        cls_init_scale: float = 0.02,
        **encoder_kwargs: Any,
    ) -> None:
        super().__init__()
        self.dim = dim

        # The inner Encoder must not own a register token — we manage the
        # CLS ourselves so we can read it out cleanly.  Silently drop the
        # key to keep YAML configs flexible.
        encoder_kwargs.pop("num_register_tokens", None)

        self.encoder = Encoder(dim=dim, **encoder_kwargs)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * cls_init_scale)

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        kv_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """Encode a sequence and return ``(hit_output, cls_tensor)``.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, N, D)``.
        x_sort_value : Tensor | None
            Values to sort tokens by before encoding (optional).  The CLS
            token is inserted *after* sorting.
        kv_mask : Tensor | None
            Boolean mask of shape ``(B, N)``.  ``True`` = valid token,
            ``False`` = padding.  A column of ``True`` is prepended for
            the CLS token.
        **kwargs
            Forwarded to the inner :class:`Encoder`.

        Returns
        -------
        tuple[Tensor, Tensor]
            - ``hit_output`` of shape ``(B, N, D)`` in the original input
              order (CLS token stripped, un-sorted).
            - ``cls_tensor`` of shape ``(B, D)``.
        """
        B = x.shape[0]

        # Sort first (if requested) — keep track of the sort index so we
        # can un-sort the hit output later.  The inner Encoder is called
        # with ``x_sort_value=None`` below so it does not re-sort.
        x_sort_idx: Tensor | None = None
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x))
            if kv_mask is not None:
                kv_mask = torch.gather(kv_mask, -1, x_sort_idx)

        # Prepend the learned CLS token.
        cls_tok = self.cls_token.expand(B, -1, -1).to(dtype=x.dtype)
        x_aug = torch.cat([cls_tok, x], dim=1)

        # Extend the mask with a ``True`` for the CLS position so the CLS
        # participates in attention.
        kv_mask_aug: Tensor | None = None
        if kv_mask is not None:
            cls_mask = torch.ones(
                (B, 1), dtype=kv_mask.dtype, device=kv_mask.device
            )
            kv_mask_aug = torch.cat([cls_mask, kv_mask], dim=1)

        # Run the inner Encoder without its own sorting.
        seq_out = self.encoder(
            x_aug,
            x_sort_value=None,
            kv_mask=kv_mask_aug,
            **kwargs,
        )

        # Split CLS and hit outputs.
        cls_out = seq_out[:, 0, :]         # (B, D)
        hit_out = seq_out[:, 1:, :]        # (B, N, D)

        # Un-sort the hit output back to the original input order.
        if x_sort_idx is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            hit_out = torch.gather(
                hit_out, -2, x_unsort_idx.unsqueeze(-1).expand_as(hit_out)
            )

        # DDP unused-parameter tie: pull the hit-output path (and therefore
        # every internal encoder parameter) into the autograd graph even
        # when the downstream model consumes only the CLS output.
        # Numerically a no-op.
        cls_out = cls_out + 0.0 * hit_out.sum()

        return hit_out, cls_out
