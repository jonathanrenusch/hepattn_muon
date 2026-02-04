"""Single-task Mamba models for isolated track parameter regression/classification.

This module provides separate Mamba models for each track parameter:
- MambaEtaRegressor: Predicts eta (pseudorapidity)
- MambaPhiRegressor: Predicts phi (azimuthal angle via sin/cos)
- MambaPtRegressor: Predicts pT (transverse momentum in log scale)
- MambaChargeClassifier: Predicts charge sign (+1/-1)

Each model is completely independent with its own encoder and head,
allowing isolated training without gradient interference from other tasks.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hepattn.models.mamba import BidirectionalMambaEncoder


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class MambaSingleTaskBase(nn.Module):
    """Base class for single-task Mamba models.
    
    Provides shared encoder architecture, subclasses define task-specific heads.
    
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
        Dropout rate for Mamba encoder.
    head_dropout : float
        Dropout rate for task head.
    head_hidden_dim : int
        Hidden dimension for task head.
    eta_feature_idx : int
        Index of eta in hit_fields (for reference extraction).
    phi_feature_idx : int
        Index of phi in hit_fields (for reference extraction).
    """
    
    def __init__(
        self,
        input_dim: int = 27,
        dim: int = 256,
        num_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba2: bool = True,
        headdim: int = 64,
        norm: str = "RMSNorm",
        dropout: float = 0.0,
        head_dropout: float = 0.15,
        head_hidden_dim: int = 512,
        eta_feature_idx: int = 26,
        phi_feature_idx: int = 25,
    ):
        super().__init__()
        
        self.dim = dim
        self.eta_feature_idx = eta_feature_idx
        self.phi_feature_idx = phi_feature_idx
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
        )
        
        # Single CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional embedding
        self.max_seq_len = 256
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Bidirectional Mamba encoder
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
        
        # Subclasses define self.head
        self.head_dropout = head_dropout
        self.head_hidden_dim = head_hidden_dim
    
    def _create_head(self, output_dim: int) -> nn.Module:
        """Create MLP head for the task."""
        return nn.Sequential(
            nn.Linear(self.dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.head_dropout),
            nn.Linear(self.head_hidden_dim, output_dim),
        )
    
    def _init_head_weights(self, head: nn.Module):
        """Initialize head weights."""
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, inputs: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Encode input sequence and return CLS embedding plus references.
        
        Returns
        -------
        cls_output : Tensor (B, dim)
            CLS token representation.
        ref_eta : Tensor (B,)
            Reference eta from innermost hit.
        ref_phi : Tensor (B,)
            Reference phi from innermost hit.
        """
        hit_features = inputs['hit_features']
        hit_mask = inputs['hit_mask']
        
        B, seq_len, _ = hit_features.shape
        
        # Extract reference values from innermost hit (position 1, after CLS placeholder)
        ref_eta = hit_features[:, 1, self.eta_feature_idx]
        ref_phi = hit_features[:, 1, self.phi_feature_idx]
        
        # Project to model dimension
        x = self.input_projection(hit_features)
        
        # Replace position 0 with CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x[:, 1:]], dim=1)
        
        # Add positional embeddings
        if seq_len <= self.max_seq_len:
            x = x + self.pos_embedding[:, :seq_len]
        else:
            pos_embed = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=seq_len,
                mode='linear',
                align_corners=False,
            ).permute(0, 2, 1)
            x = x + pos_embed
        
        # Apply mask
        x = x * hit_mask.unsqueeze(-1).float()
        
        # Encode
        x = self.encoder(x)
        
        # Extract CLS token
        cls_output = x[:, 0]
        
        return cls_output, ref_eta, ref_phi


class MambaEtaRegressor(MambaSingleTaskBase):
    """Single-task model for eta (pseudorapidity) regression.
    
    Predicts delta_eta from innermost hit reference.
    Final eta = ref_eta + delta_eta
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = self._create_head(output_dim=1)
        self._init_head_weights(self.head)
    
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        cls_output, ref_eta, ref_phi = self.encode(inputs)
        
        # Predict delta_eta
        delta_eta = self.head(cls_output).squeeze(-1)  # (B,)
        
        return {
            'delta_eta': delta_eta,
            'ref_eta': ref_eta,
            'cls_embedding': cls_output,
        }
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert to final eta prediction."""
        eta = outputs['ref_eta'] + outputs['delta_eta']
        return {'eta': eta}


class MambaPhiRegressor(MambaSingleTaskBase):
    """Single-task model for phi (azimuthal angle) regression.
    
    Predicts sin(phi) and cos(phi) directly (handles periodicity).
    Final phi = atan2(sin_phi, cos_phi)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = self._create_head(output_dim=2)  # sin_phi, cos_phi
        self._init_head_weights(self.head)
    
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        cls_output, ref_eta, ref_phi = self.encode(inputs)
        
        # Predict sin_phi, cos_phi
        phi_output = self.head(cls_output)  # (B, 2)
        sin_phi = phi_output[:, 0]
        cos_phi = phi_output[:, 1]
        
        return {
            'sin_phi': sin_phi,
            'cos_phi': cos_phi,
            'ref_phi': ref_phi,  # For reference/logging
            'cls_embedding': cls_output,
        }
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert to final phi prediction."""
        phi = torch.atan2(outputs['sin_phi'], outputs['cos_phi'])
        return {'phi': phi}


class MambaPtRegressor(MambaSingleTaskBase):
    """Single-task model for pT (transverse momentum) regression.
    
    Predicts log(pT) to handle the long-tail distribution.
    Final pT = exp(log_pt)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = self._create_head(output_dim=1)
        self._init_head_weights(self.head)
    
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        cls_output, ref_eta, ref_phi = self.encode(inputs)
        
        # Predict log_pt
        log_pt = self.head(cls_output).squeeze(-1)  # (B,)
        
        return {
            'log_pt': log_pt,
            'cls_embedding': cls_output,
        }
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert to final pT prediction."""
        pt = torch.exp(outputs['log_pt'])
        return {'pt': pt}


class MambaChargeClassifier(MambaSingleTaskBase):
    """Single-task model for charge classification.
    
    Predicts charge sign as binary classification (+1 vs -1).
    Output is logit; apply sigmoid for probability.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = self._create_head(output_dim=1)
        self._init_head_weights(self.head)
    
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        cls_output, ref_eta, ref_phi = self.encode(inputs)
        
        # Predict charge logit
        charge_logit = self.head(cls_output).squeeze(-1)  # (B,)
        
        return {
            'charge_logit': charge_logit,
            'cls_embedding': cls_output,
        }
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert to final charge prediction."""
        charge_prob = torch.sigmoid(outputs['charge_logit'])
        charge = torch.where(charge_prob > 0.5, 
                            torch.ones_like(charge_prob), 
                            -torch.ones_like(charge_prob))
        return {'charge': charge, 'charge_prob': charge_prob}
