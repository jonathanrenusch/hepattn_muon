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


# TODO: Understand how closely this code aligns with the original Mamba and Vision Mamba that you found online


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
        self.forward_mamba = MambaClass(**mamba_kwargs)
        self.backward_mamba = MambaClass(**mamba_kwargs)
        
        # Gating mechanism for combining forward and backward
        self.gate = nn.Linear(dim, dim)
        self.gate_activation = nn.Sigmoid()
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with bidirectional processing.
        
        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D).
            
        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D).
        """
        # Skip connection
        skip = x
        
        # Normalize and make contiguous for Mamba2 causal_conv1d kernel
        x_norm = self.norm(x).contiguous()
        
        # Forward pass (left-to-right)
        x_forward = self.forward_mamba(x_norm)
        
        # Backward pass (right-to-left) - flip, process, flip back
        # Note: .contiguous() is required for Mamba2 causal_conv1d kernel
        x_backward = torch.flip(x_norm, dims=[1]).contiguous()
        x_backward = self.backward_mamba(x_backward)
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

