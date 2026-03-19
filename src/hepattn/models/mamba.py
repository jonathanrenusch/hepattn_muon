"""Mamba State Space Model encoder implementations for hit filtering.

This module provides Mamba-based encoders as alternatives to transformer encoders:
- MambaEncoder: Unidirectional Mamba encoder (simple baseline)
- BidirectionalMambaEncoder: Vision Mamba-style bidirectional encoder

Both encoders follow the same interface as the Transformer Encoder, accepting
(B, N, D) shaped tensors and optionally sorting by phi for proper sequence ordering.

References:
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)
- Vision Mamba: Efficient Visual Representation Learning with Bidirectional SSM (arXiv:2401.09417)
"""

import torch
from torch import Tensor, nn

try:
    from mamba_ssm import Mamba, Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    Mamba = None
    Mamba2 = None



class MambaEncoderLayer(nn.Module):
    """Single Mamba encoder layer with pre-normalization and residual connection.
    
    This follows a similar pattern to transformer encoder layers:
    x = x + Mamba(norm(x))
    x = x + FFN(norm(x))
    
    Parameters
    ----------
    dim : int
        Model dimension.
    d_state : int
        SSM state expansion factor.
    d_conv : int
        Local convolution width.
    expand : int
        Block expansion factor.
    use_mamba2 : bool
        Whether to use Mamba-2 architecture (more efficient for longer sequences).
    headdim : int
        Head dimension for Mamba-2 (default: 64). Must divide dim * expand evenly.
        Use smaller values (e.g., 32) for smaller model dimensions.
    norm : str
        Normalization layer type ('LayerNorm' or 'RMSNorm').
    dropout : float
        Dropout rate.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba2: bool = True,
        headdim: int = 64,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required for Mamba models. "
                "Install with: pip install mamba-ssm[causal-conv1d]"
            )
        
        self.dim = dim
        
        # Pre-normalization
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.norm = nn.RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm type: {norm}")
        
        # Mamba block
        MambaClass = Mamba2 if use_mamba2 else Mamba
        mamba_kwargs = {
            "d_model": dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }
        if use_mamba2:
            mamba_kwargs["headdim"] = headdim
        self.mamba = MambaClass(**mamba_kwargs)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D).
            
        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D).
        """
        # Pre-norm residual: x = x + dropout(mamba(norm(x)))
        # Note: .contiguous() is required for Mamba2 causal_conv1d kernel
        x = x + self.dropout(self.mamba(self.norm(x).contiguous()))
        return x


class BidirectionalMambaEncoderLayer(nn.Module):
    """Bidirectional Mamba encoder layer (Vision Mamba style).
    
    This implements the bidirectional processing from Vision Mamba paper:
    - Forward Mamba processes sequence left-to-right
    - Backward Mamba processes sequence right-to-left (flipped)
    - Outputs are combined with gating mechanism
    
    Parameters
    ----------
    dim : int
        Model dimension.
    d_state : int
        SSM state expansion factor.
    d_conv : int
        Local convolution width.
    expand : int
        Block expansion factor.
    use_mamba2 : bool
        Whether to use Mamba-2 architecture.
    headdim : int
        Head dimension for Mamba-2 (default: 64). Must divide dim * expand evenly.
        Use smaller values (e.g., 32) for smaller model dimensions.
    norm : str
        Normalization layer type.
    dropout : float
        Dropout rate.
    """
    
    def __init__(
        self,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba2: bool = True,
        headdim: int = 64,
        ngroups: int = 1,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package is required for Mamba models. "
                "Install with: pip install mamba-ssm[causal-conv1d]"
            )

        self.dim = dim

        # Pre-normalization
        if norm == "LayerNorm":
            self.norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.norm = nn.RMSNorm(dim)
        else:
            raise ValueError(f"Unknown norm type: {norm}")

        # Forward and backward Mamba blocks
        MambaClass = Mamba2 if use_mamba2 else Mamba
        mamba_kwargs = {
            "d_model": dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
        }
        if use_mamba2:
            mamba_kwargs["headdim"] = headdim
            mamba_kwargs["ngroups"] = ngroups
        self.forward_mamba = MambaClass(**mamba_kwargs)
        self.backward_mamba = MambaClass(**mamba_kwargs)
        
        # Gating mechanism for combining forward and backward
        self.gate = nn.Linear(dim, dim)
        self.gate_activation = nn.Sigmoid()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: Tensor,
        seq_idx: Tensor | None = None,
        flip_idx: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with bidirectional processing.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D). When packing is used this is
            (1, sum(lengths), D) with events concatenated along the sequence dim.
        seq_idx : Tensor, optional
            Integer tensor of shape (1, sum(lengths)) mapping each token to its
            event index. Passed to Mamba2 to reset SSM state at event boundaries.
        flip_idx : Tensor, optional
            Long tensor of shape (sum(lengths),) precomputed by the encoder.
            When provided, the backward flip is done via a single gather instead
            of torch.flip over the full sequence.

        Returns
        -------
        Tensor
            Output tensor of same shape as ``x``.
        """
        # Skip connection
        skip = x

        # Normalize and make contiguous for Mamba2 causal_conv1d kernel
        x_norm = self.norm(x).contiguous()

        # Forward pass (left-to-right)
        x_forward = self.forward_mamba(x_norm, seq_idx=seq_idx)

        # Backward pass (right-to-left)
        # Packed: single gather via precomputed flip_idx (one GPU dispatch).
        # Padded: torch.flip over the full sequence dimension.
        if flip_idx is not None:
            x_rev = x_norm[:, flip_idx].contiguous()
        else:
            x_rev = torch.flip(x_norm, dims=[1]).contiguous()
        x_backward = self.backward_mamba(x_rev, seq_idx=seq_idx)
        if flip_idx is not None:
            x_backward = x_backward[:, flip_idx].contiguous()
        else:
            x_backward = torch.flip(x_backward, dims=[1])

        # Gating mechanism to combine forward and backward
        gate = self.gate_activation(self.gate(x_norm))
        x_combined = gate * x_forward + (1 - gate) * x_backward

        # Residual connection with dropout
        return skip + self.dropout(x_combined)


class MambaEncoder(nn.Module):
    """Unidirectional Mamba encoder (simple baseline).
    
    Stacks multiple MambaEncoderLayers for sequence processing.
    Supports optional phi-sorting for proper sequence ordering of detector hits.
    
    Parameters
    ----------
    num_layers : int
        Number of Mamba layers.
    dim : int
        Model dimension.
    d_state : int
        SSM state expansion factor (default: 64 for Mamba-2).
    d_conv : int
        Local convolution width (default: 4).
    expand : int
        Block expansion factor (default: 2).
    use_mamba2 : bool
        Whether to use Mamba-2 architecture (default: True).
    headdim : int
        Head dimension for Mamba-2 (default: 64). Must divide dim * expand evenly.
        Use smaller values (e.g., 32) for smaller model dimensions like 128.
    norm : str
        Normalization layer type (default: 'LayerNorm').
    dropout : float
        Dropout rate (default: 0.0).
    """
    
    def __init__(
        self,
        num_layers: int,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba2: bool = True,
        headdim: int = 64,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        
        self.layers = nn.ModuleList([
            MambaEncoderLayer(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_mamba2=use_mamba2,
                headdim=headdim,
                norm=norm,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        if norm == "LayerNorm":
            self.final_norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.final_norm = nn.RMSNorm(dim)
        else:
            self.final_norm = nn.Identity()
    
    def forward(self, x: Tensor, x_sort_value: Tensor | None = None, **kwargs) -> Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D).
        x_sort_value : Tensor, optional
            Values to sort tokens by (e.g., phi angle). If provided, tokens are
            sorted before processing and unsorted after.
        **kwargs
            Additional arguments (ignored, for compatibility with Transformer encoder).
            
        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D).
        """
        # Sort tokens by provided value (e.g., phi angle) for proper sequence ordering
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x)).contiguous()
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Unsort tokens back to original order
        if x_sort_value is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            x = torch.gather(x, -2, x_unsort_idx.unsqueeze(-1).expand_as(x))
        
        return x


class BidirectionalMambaEncoder(nn.Module):
    """Bidirectional Mamba encoder (Vision Mamba style).
    
    Stacks multiple BidirectionalMambaEncoderLayers for sequence processing.
    Each layer processes the sequence in both forward and backward directions,
    combining outputs via a learned gating mechanism.
    
    This is useful for particle tracking where information can propagate in
    both directions along the detector (e.g., along phi angle).
    
    Parameters
    ----------
    num_layers : int
        Number of bidirectional Mamba layers.
    dim : int
        Model dimension.
    d_state : int
        SSM state expansion factor (default: 64 for Mamba-2).
    d_conv : int
        Local convolution width (default: 4).
    expand : int
        Block expansion factor (default: 2).
    use_mamba2 : bool
        Whether to use Mamba-2 architecture (default: True).
    headdim : int
        Head dimension for Mamba-2 (default: 64). Must divide dim * expand evenly.
        Use smaller values (e.g., 32) for smaller model dimensions like 128.
    norm : str
        Normalization layer type (default: 'LayerNorm').
    dropout : float
        Dropout rate (default: 0.0).
    """
    
    def __init__(
        self,
        num_layers: int,
        dim: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba2: bool = True,
        headdim: int = 64,
        ngroups: int = 1,
        norm: str = "LayerNorm",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dim = dim

        self.layers = nn.ModuleList([
            BidirectionalMambaEncoderLayer(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_mamba2=use_mamba2,
                headdim=headdim,
                ngroups=ngroups,
                norm=norm,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        if norm == "LayerNorm":
            self.final_norm = nn.LayerNorm(dim)
        elif norm == "RMSNorm":
            self.final_norm = nn.RMSNorm(dim)
        else:
            self.final_norm = nn.Identity()

    def forward(
        self,
        x: Tensor,
        x_sort_value: Tensor | None = None,
        pad_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass with optional sequence packing.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N_max, D). May contain padding.
        x_sort_value : Tensor, optional
            Per-token sort key (e.g. phi angle) of shape (B, N_max). Tokens
            are sorted within each event before processing and unsorting after.
        pad_mask : Tensor, optional
            Boolean validity mask of shape (B, N_max). True marks valid tokens.
            When provided, valid tokens are packed into a single contiguous
            sequence so no compute is wasted on padding positions.
        **kwargs
            Additional arguments (ignored, for compatibility with Transformer encoder).

        Returns
        -------
        Tensor
            Output tensor of shape (B, N_max, D).
        """
        if pad_mask is not None:
            return self._forward_packed(x, x_sort_value, pad_mask)

        # ── Padded path (original behaviour) ────────────────────────────────
        if x_sort_value is not None:
            x_sort_idx = torch.argsort(x_sort_value, dim=-1)
            x = torch.gather(x, -2, x_sort_idx.unsqueeze(-1).expand_as(x)).contiguous()

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)

        if x_sort_value is not None:
            x_unsort_idx = torch.argsort(x_sort_idx, dim=-1)
            x = torch.gather(x, -2, x_unsort_idx.unsqueeze(-1).expand_as(x))

        return x

    def _forward_packed(
        self,
        x: Tensor,
        x_sort_value: Tensor | None,
        pad_mask: Tensor,
    ) -> Tensor:
        """Packing path: concatenate valid tokens across the batch, process once,
        then unpack back to the original padded layout.

        Relies on the collator placing valid tokens at the *start* of each row
        (i.e. x[b, :l] are valid, x[b, l:] is padding), which is guaranteed by
        AtlasMuonCollator's pad_and_concat.

        All pack / sort / unpack steps are fully vectorised (no Python loops over
        the batch dimension in the hot path).
        """
        B, N_max, D = x.shape
        # Derive lengths directly on GPU — avoids a CPU→GPU round-trip.
        lengths_t = pad_mask.sum(-1).to(torch.int32)   # (B,)

        # ── 1. Sort valid tokens (optional) + pack ──────────────────────────
        # Set phi = +inf at padding positions so argsort places all valid tokens
        # in positions 0..l-1 of each row, sorted by phi.  Padding floats to
        # the end and is then dropped when we boolean-index with pad_mask.
        if x_sort_value is not None:
            phi = x_sort_value.clone()
            phi[~pad_mask] = float("inf")
            sort_idx = torch.argsort(phi, dim=-1)          # (B, N_max)
            x = torch.gather(x, -2, sort_idx.unsqueeze(-1).expand_as(x)).contiguous()

        # Boolean-mask indexing gathers all valid tokens in (batch, seq) order.
        # Result shape: (ΣL, D) → unsqueeze to (1, ΣL, D) for Mamba2.
        x_packed = x[pad_mask].unsqueeze(0)

        # seq_idx: (1, ΣL) — integer label per token; Mamba2 resets SSM/conv
        # states at each label change, i.e. at every event boundary.
        seq_idx = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32, device=x.device), lengths_t
        ).unsqueeze(0)

        # flip_idx: precomputed once for all layers, fully vectorised.
        # For token at flat position i belonging to event b (offset o, length l):
        #   reversed position = o + (l - 1) - (i - o) = 2o + l - 1 - i
        offsets_t = torch.cat([lengths_t.new_zeros(1), lengths_t[:-1].cumsum(0)])
        flat_i = torch.arange(int(lengths_t.sum()), dtype=torch.long, device=x.device)
        event_idx = seq_idx.squeeze(0).long()
        flip_idx = (
            2 * offsets_t[event_idx].long() + lengths_t[event_idx].long() - 1 - flat_i
        )

        # ── 2. Encoder layers on packed sequence ────────────────────────────
        for layer in self.layers:
            x_packed = layer(x_packed, seq_idx=seq_idx, flip_idx=flip_idx)

        x_packed = self.final_norm(x_packed)

        # ── 3. Unpack + unsort back to (B, N_max, D) ────────────────────────
        # Scatter processed tokens back into the (phi-)sorted padded layout …
        x_out = x.new_zeros(B, N_max, D)
        x_out[pad_mask] = x_packed.squeeze(0)

        # … then undo the phi sort so tokens are in their original positions.
        if x_sort_value is not None:
            unsort_idx = torch.argsort(sort_idx, dim=-1)  # (B, N_max)
            x_out = torch.gather(x_out, -2, unsort_idx.unsqueeze(-1).expand_as(x_out))

        return x_out

