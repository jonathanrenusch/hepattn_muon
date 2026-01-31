"""Mamba-based track parameter regression model.

This module provides a bidirectional Mamba model for track parameter regression
using ground truth hit-to-track assignments. It reuses the existing
BidirectionalMambaEncoder from hepattn.models.mamba.

The model:
1. Projects hit features to model dimension
2. Adds a learnable CLS token at the start of each sequence
3. Processes through bidirectional Mamba encoder
4. Uses CLS token output for regression (eta, phi, pt) and classification (charge)

Delta prediction mode (optional):
- Instead of predicting absolute eta/phi, predict deltas from innermost hit
- This provides easier training targets, especially for phi (avoids periodicity issues)
- Final predictions are: ref_value + delta
"""

import math
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
    
    Delta prediction mode (use_delta_prediction=True):
    - Predicts delta_eta, delta_phi from innermost hit reference
    - Much simpler training target for phi (no periodicity handling needed)
    - Final output: ref_value + delta
    
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
    use_delta_prediction : bool
        If True, predict deltas from reference point instead of absolute values.
    delta_reference : str
        Reference point for delta prediction: 'innermost' (default) or 'average'.
    eta_feature_idx : int
        Index of eta in hit_fields (default: 17 based on standard config).
    phi_feature_idx : int
        Index of phi in hit_fields (default: 16 based on standard config).
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
        use_delta_prediction: bool = False,
        delta_reference: str = "innermost",
        eta_feature_idx: int = 17,
        phi_feature_idx: int = 16,
    ):
        super().__init__()
        
        self.dim = dim
        self.regression_fields = regression_fields or ['eta', 'phi', 'pt']
        
        # Delta prediction configuration
        self.use_delta_prediction = use_delta_prediction
        self.delta_reference = delta_reference
        self.eta_feature_idx = eta_feature_idx
        self.phi_feature_idx = phi_feature_idx
        
        # Internal output dimensions depend on mode
        if use_delta_prediction:
            # Delta mode: delta_eta, delta_phi, log_pt (3 outputs)
            self.num_regression_outputs = 3
        else:
            # Standard mode: eta, sin_phi, cos_phi, log_pt (4 outputs)
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
    
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        inputs : dict
            Dictionary containing:
            - hit_features: Tensor of shape (B, seq_len, input_dim).
              Position 0 is placeholder for CLS token.
            - hit_mask: Boolean mask of shape (B, seq_len), True for valid positions.
            - sequence_lengths: Optional tensor of actual sequence lengths (B,).
            
        Returns
        -------
        dict with keys:
            - regression: (B, num_regression_outputs) - raw network outputs
            - charge_logit: (B, 1) - charge classification logit
            - cls_embedding: (B, dim) - CLS token embedding (for analysis)
            
        In delta prediction mode, also includes:
            - ref_eta: (B,) - reference eta from innermost hit
            - ref_phi: (B,) - reference phi from innermost hit
        """
        hit_features = inputs['hit_features']
        hit_mask = inputs['hit_mask']
        # sequence_lengths = inputs.get('sequence_lengths')  # Currently unused
        
        B, seq_len, _ = hit_features.shape
        
        # Extract reference values for delta prediction BEFORE modifying hit_features
        ref_eta = None
        ref_phi = None
        if self.use_delta_prediction:
            if self.delta_reference == "innermost":
                # Hits are sorted by r ascending with CLS at position 0
                # Position 1 is the innermost hit
                ref_eta = hit_features[:, 1, self.eta_feature_idx]  # (B,)
                ref_phi = hit_features[:, 1, self.phi_feature_idx]  # (B,)
            elif self.delta_reference == "average":
                # Compute masked average over all valid hits (excluding CLS at position 0)
                hit_eta = hit_features[:, 1:, self.eta_feature_idx]  # (B, seq_len-1)
                hit_phi = hit_features[:, 1:, self.phi_feature_idx]  # (B, seq_len-1)
                valid_mask = hit_mask[:, 1:].float()  # (B, seq_len-1)
                
                # Masked mean
                ref_eta = (hit_eta * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
                ref_phi = (hit_phi * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
            else:
                raise ValueError(f"Unknown delta_reference: {self.delta_reference}")
        
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
        
        result = {
            'regression': regression_output,
            'charge_logit': charge_logit.squeeze(-1),
            'cls_embedding': cls_output,
        }
        
        # Add reference values for delta prediction mode
        if self.use_delta_prediction:
            result['ref_eta'] = ref_eta
            result['ref_phi'] = ref_phi
        
        return result
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert model outputs to physics predictions.
        
        Standard mode:
        - Model outputs [eta, sin_phi, cos_phi, log_pt]
        - Recovers phi from sin/cos using atan2 (handles periodicity)
        - Recovers pt from log_pt using exp
        
        Delta prediction mode:
        - Model outputs [delta_eta, delta_phi, log_pt]
        - Final eta = ref_eta + delta_eta
        - Final phi = ref_phi + delta_phi (wrapped to [-π, π])
        - Recovers pt from log_pt using exp
        
        Parameters
        ----------
        outputs : dict
            Model outputs from forward pass.
            
        Returns
        -------
        dict with keys:
            - eta: (B,) eta predictions
            - phi: (B,) phi predictions
            - pt: (B,) pt predictions
            - sin_phi, cos_phi, log_pt: (B,) raw network outputs (standard mode only)
            - delta_eta, delta_phi: (B,) deltas (delta mode only)
            - ref_eta, ref_phi: (B,) reference values (delta mode only)
            - charge_prob: (B,) probability of positive charge
            - charge: (B,) predicted charge (-1 or 1)
        """
        regression = outputs['regression']
        charge_logit = outputs['charge_logit']
        
        if self.use_delta_prediction:
            # Delta mode: [delta_eta, delta_phi, log_pt]
            delta_eta = regression[:, 0]
            delta_phi = regression[:, 1]
            log_pt_pred = regression[:, 2]
            
            ref_eta = outputs['ref_eta']
            ref_phi = outputs['ref_phi']
            
            # Recover final values: reference + delta
            eta_pred = ref_eta + delta_eta
            phi_pred_raw = ref_phi + delta_phi
            
            # Wrap phi to [-π, π]
            phi_pred = torch.atan2(torch.sin(phi_pred_raw), torch.cos(phi_pred_raw))
            
            # Recover pt from log
            pt_pred = torch.exp(log_pt_pred)
            
            # Charge predictions
            charge_prob = torch.sigmoid(charge_logit)
            charge_pred = torch.where(charge_prob > 0.5, 
                                       torch.ones_like(charge_prob),
                                       -torch.ones_like(charge_prob))
            
            return {
                # Recovered physics values (for evaluation)
                'eta': eta_pred,
                'phi': phi_pred,
                'pt': pt_pred,
                # Raw deltas (for debugging/analysis)
                'delta_eta': delta_eta,
                'delta_phi': delta_phi,
                'log_pt': log_pt_pred,
                # Reference values
                'ref_eta': ref_eta,
                'ref_phi': ref_phi,
                # Charge predictions
                'charge_prob': charge_prob,
                'charge': charge_pred,
            }
        else:
            # Standard mode: [eta, sin_phi, cos_phi, log_pt]
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
