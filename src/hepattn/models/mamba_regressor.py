"""Mamba-based track parameter regression model.

This module provides a bidirectional Mamba model for track parameter regression
using ground truth hit-to-track assignments. It reuses the existing
BidirectionalMambaEncoder from hepattn.models.mamba.

The model:
1. Projects hit features to model dimension
2. Adds a learnable CLS token at the start of each sequence
3. Processes through bidirectional Mamba encoder
4. Uses CLS token output for regression (eta, phi, pt) and classification (charge)
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hepattn.models.mamba import BidirectionalMambaEncoder
from hepattn.models.activation import SwiGLU


class MambaTrackRegressor(nn.Module):
    """Bidirectional Mamba model for track parameter regression.
    
    Uses the existing BidirectionalMambaEncoder for sequence processing,
    with a CLS token for aggregating track information.
    
    The model outputs physics-informed representations:
    - eta: direct regression (already centered around 0, range ~[-3, 3])
    - sin_phi, cos_phi: unit circle representation for periodic phi
    - log_pt: log-transformed pt for better handling of long-tail distribution
    
    Parameters
    ----------
    input_dim : int
        Number of input features per hit.
    dim : int
        Model embedding dimension.
    num_layers : int
        Number of bidirectional Mamba layers.
    d_state : int
        SSM state expansion factor.
    d_conv : int
        Local convolution width.
    expand : int
        Block expansion factor.
    use_mamba2 : bool
        Whether to use Mamba-2 architecture.
    headdim : int
        Head dimension for Mamba-2.
    norm : str
        Normalization type ('LayerNorm' or 'RMSNorm').
    dropout : float
        Dropout rate for Mamba encoder (typically 0.0 for SSMs).
    head_dropout : float
        Dropout rate for MLP regression/classification heads.
    head_hidden_dim : int
        Hidden dimension for regression/classification heads.
    regression_fields : list[str]
        Names of regression targets (user-facing: eta, phi, pt).
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        dim: int = 64,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba2: bool = True,
        headdim: int = 16,
        norm: str = "RMSNorm",
        dropout: float = 0.0,
        head_dropout: float = 0.2,
        head_hidden_dim: int = 128,
        regression_fields: list[str] | None = None,
    ):
        super().__init__()
        
        self.dim = dim
        self.regression_fields = regression_fields or ['eta', 'phi', 'pt']
        # Internal output: eta, sin_phi, cos_phi, log_pt (4 outputs)
        self.num_regression_outputs = 4
        
        # Input projection: hit features -> model dimension
        # Simpler projection without extra dropout
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
        )
        
        # Learnable CLS token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional embedding for sequence position
        self.max_seq_len = 256  # Max hits + CLS
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Bidirectional Mamba encoder (reuses existing implementation)
        self.encoder = BidirectionalMambaEncoder(
            num_layers=num_layers,
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mamba2=use_mamba2,
            headdim=headdim,
            norm=norm,
            dropout=dropout,
        )
        
        # Regression head: single hidden layer with GELU
        self.regression_head = nn.Sequential(
            nn.Linear(dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, self.num_regression_outputs),
        )
        
        # Classification head for charge (binary)
        self.classification_head = nn.Sequential(
            nn.Linear(dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in [self.regression_head, self.classification_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        hit_features: Tensor,
        hit_mask: Tensor,
        sequence_lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        hit_features : Tensor
            Hit features of shape (B, seq_len, input_dim).
            Position 0 is placeholder for CLS token.
        hit_mask : Tensor
            Boolean mask of shape (B, seq_len), True for valid positions.
        sequence_lengths : Tensor, optional
            Actual sequence lengths of shape (B,).
            
        Returns
        -------
        dict with keys:
            - regression: (B, num_regression_outputs) - eta, phi, pt predictions
            - charge_logit: (B, 1) - charge classification logit
            - cls_embedding: (B, dim) - CLS token embedding (for analysis)
        """
        B, seq_len, _ = hit_features.shape
        
        # Project hit features to model dimension
        x = self.input_projection(hit_features)  # (B, seq_len, dim)
        
        # Replace position 0 with learnable CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat([cls_tokens, x[:, 1:]], dim=1)  # (B, seq_len, dim)
        
        # Add positional embeddings
        if seq_len <= self.max_seq_len:
            x = x + self.pos_embedding[:, :seq_len]
        else:
            # Interpolate positional embeddings for longer sequences
            pos_embed = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=seq_len,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1)
            x = x + pos_embed
        
        # Apply mask by zeroing out invalid positions
        # (Mamba doesn't use attention masks, but we zero out padded positions)
        x = x * hit_mask.unsqueeze(-1).float()
        
        # Process through bidirectional Mamba encoder
        # Note: We don't use x_sort_value since hits are already sorted by r
        x = self.encoder(x)  # (B, seq_len, dim)
        
        # Extract CLS token representation (position 0)
        cls_output = x[:, 0]  # (B, dim)
        
        # Regression predictions
        regression_output = self.regression_head(cls_output)  # (B, num_regression_outputs)
        
        # Classification prediction (charge)
        charge_logit = self.classification_head(cls_output)  # (B, 1)
        
        return {
            'regression': regression_output,
            'charge_logit': charge_logit.squeeze(-1),
            'cls_embedding': cls_output,
        }
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert model outputs to physics predictions.
        
        The model outputs [eta, sin_phi, cos_phi, log_pt]. This method:
        - Recovers phi from sin/cos using atan2 (handles periodicity)
        - Recovers pt from log_pt using exp
        - Converts charge probability to -1/+1
        
        Parameters
        ----------
        outputs : dict
            Model outputs from forward pass.
            
        Returns
        -------
        dict with keys:
            - eta: (B,) eta predictions (direct)
            - phi: (B,) phi predictions recovered from sin/cos
            - pt: (B,) pt predictions recovered from log
            - sin_phi, cos_phi, log_pt: (B,) raw network outputs
            - charge_prob: (B,) probability of positive charge
            - charge: (B,) predicted charge (-1 or 1)
        """
        regression = outputs['regression']
        charge_logit = outputs['charge_logit']
        
        # Extract raw outputs: [eta, sin_phi, cos_phi, log_pt]
        eta_pred = regression[:, 0]
        sin_phi_pred = regression[:, 1]
        cos_phi_pred = regression[:, 2]
        log_pt_pred = regression[:, 3]
        
        # Recover phi from sin/cos (NOT normalized - use raw predictions)
        phi_pred = torch.atan2(sin_phi_pred, cos_phi_pred)
        
        # Recover pt from log
        pt_pred = torch.exp(log_pt_pred)
        
        # Charge predictions
        charge_prob = torch.sigmoid(charge_logit)
        charge_pred = torch.where(charge_prob > 0.5, 
                                   torch.ones_like(charge_prob),
                                   -torch.ones_like(charge_prob))
        
        return {
            # Recovered physics values
            'eta': eta_pred,
            'phi': phi_pred,
            'pt': pt_pred,
            # Raw network outputs (for debugging/analysis)
            'sin_phi': sin_phi_pred,
            'cos_phi': cos_phi_pred,
            'log_pt': log_pt_pred,
            # Charge predictions
            'charge_prob': charge_prob,
            'charge': charge_pred,
        }
