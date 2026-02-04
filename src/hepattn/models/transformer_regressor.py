"""Transformer-based track parameter regression model.

This module provides a Transformer encoder model for track parameter regression
using ground truth hit-to-track assignments. It serves as a baseline to compare
against the bidirectional Mamba model.

The model:
1. Projects hit features to model dimension
2. Adds a learnable CLS token at the start of each sequence
3. Processes through Transformer encoder (self-attention + FFN layers)
4. Uses CLS token output for regression (eta, phi, pt) and classification (charge)

Delta prediction mode (optional):
- Instead of predicting absolute eta/phi, predict deltas from innermost hit
- This provides easier training targets, especially for phi (avoids periodicity issues)
- Final predictions are: ref_value + delta

This is designed to be a parameter-count-matched baseline for MambaTrackRegressor.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hepattn.models.transformer import Encoder


class TransformerTrackRegressor(nn.Module):
    """Transformer encoder model for track parameter regression.
    
    Uses the existing Transformer Encoder for sequence processing,
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
        Number of Transformer encoder layers.
    num_heads : int
        Number of attention heads.
    attn_type : str
        Attention type: "torch", "flash", or "flex".
    dense_hidden_scale : int
        Scale factor for FFN hidden dimension (hidden_dim = dim * scale).
    norm : str
        Normalization type ('LayerNorm' or 'RMSNorm').
    dropout : float
        Dropout rate for Transformer encoder.
    head_dropout : float
        Dropout rate for MLP regression/classification heads.
    head_hidden_dim : int
        Hidden dimension for regression/classification heads.
    regression_fields : list[str]
        Names of regression targets (user-facing: eta, phi, pt).
    use_delta_prediction : bool
        If True, predict deltas from reference point instead of absolute values.
    use_delta_phi : bool
        If True (and use_delta_prediction=True), predict delta_phi.
        If False (and use_delta_prediction=True), use sin/cos for phi (hybrid mode).
        Default: False (hybrid mode - delta_eta + sin/cos phi).
    delta_reference : str
        Reference point for delta prediction: 'innermost' (default) or 'average'.
    eta_feature_idx : int
        Index of eta in hit_fields (default: 17 based on standard config).
    phi_feature_idx : int
        Index of phi in hit_fields (default: 16 based on standard config).
    use_multi_cls : bool
        If True, use 4 separate CLS tokens (one per task: eta, phi, pt, charge).
        Each task gets its own dedicated CLS token and MLP head.
        Default: False (single shared CLS token).
    """
    
    def __init__(
        self,
        input_dim: int = 18,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        attn_type: str = "torch",
        dense_hidden_scale: int = 2,
        norm: str = "RMSNorm",
        dropout: float = 0.0,
        head_dropout: float = 0.2,
        head_hidden_dim: int = 128,
        regression_fields: list[str] | None = None,
        use_delta_prediction: bool = False,
        use_delta_phi: bool = False,  # If False + delta_prediction=True: hybrid mode
        delta_reference: str = "innermost",
        eta_feature_idx: int = 17,
        phi_feature_idx: int = 16,
        use_multi_cls: bool = False,  # If True, use 4 separate CLS tokens
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.regression_fields = regression_fields or ['eta', 'phi', 'pt']
        self.use_multi_cls = use_multi_cls
        
        # Delta prediction configuration
        self.use_delta_prediction = use_delta_prediction
        self.use_delta_phi = use_delta_phi
        self.delta_reference = delta_reference
        self.eta_feature_idx = eta_feature_idx
        self.phi_feature_idx = phi_feature_idx
        
        # Internal output dimensions depend on mode
        # Modes:
        # 1. Standard (use_delta_prediction=False): [eta, sin_phi, cos_phi, log_pt] = 4 outputs
        # 2. Full delta (use_delta_prediction=True, use_delta_phi=True): [delta_eta, delta_phi, log_pt] = 3 outputs
        # 3. Hybrid (use_delta_prediction=True, use_delta_phi=False): [delta_eta, sin_phi, cos_phi, log_pt] = 4 outputs
        if use_delta_prediction:
            if use_delta_phi:
                # Full delta mode: delta_eta, delta_phi, log_pt (3 outputs)
                self.num_regression_outputs = 3
                self.phi_output_dim = 1  # delta_phi
            else:
                # Hybrid mode: delta_eta, sin_phi, cos_phi, log_pt (4 outputs)
                self.num_regression_outputs = 4
                self.phi_output_dim = 2  # sin_phi, cos_phi
        else:
            # Standard mode: eta, sin_phi, cos_phi, log_pt (4 outputs)
            self.num_regression_outputs = 4
            self.phi_output_dim = 2  # sin_phi, cos_phi
        
        # Input projection: hit features -> model dimension
        # Simpler projection without extra dropout
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
        )
        
        # CLS token(s) - either single shared or 4 separate tokens
        if use_multi_cls:
            # 4 separate CLS tokens: [CLS_eta, CLS_phi, CLS_pt, CLS_charge]
            # Shape: (1, 4, dim) - will be placed at positions 0-3
            self.num_cls_tokens = 4
            self.cls_tokens = nn.Parameter(torch.zeros(1, 4, dim))
            nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        else:
            # Single shared CLS token
            self.num_cls_tokens = 1
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Positional embedding for sequence position
        self.max_seq_len = 256  # Max hits + CLS token(s)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer encoder (reuses existing implementation)
        # Note: Dense layer uses SwiGLU by default with hidden_dim = dim * hidden_scale
        dense_kwargs = {
            "hidden_dim_scale": dense_hidden_scale,
            "dropout": dropout,
        }
        attn_kwargs = {
            "num_heads": num_heads,
            "attn_type": attn_type,
        }
        
        self.encoder = Encoder(
            num_layers=num_layers,
            dim=dim,
            attn_type=attn_type,
            norm=norm,
            dense_kwargs=dense_kwargs,
            attn_kwargs=attn_kwargs,
        )
        
        # Task heads - either shared or separate per task
        if use_multi_cls:
            # Separate heads for each task
            # Each head: dim -> head_hidden_dim -> output_dim
            
            # Eta head: outputs 1 value (eta or delta_eta)
            self.eta_head = nn.Sequential(
                nn.Linear(dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
            
            # Phi head: outputs 1 (delta_phi) or 2 (sin_phi, cos_phi) values
            self.phi_head = nn.Sequential(
                nn.Linear(dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, self.phi_output_dim),
            )
            
            # Pt head: outputs 1 value (log_pt)
            self.pt_head = nn.Sequential(
                nn.Linear(dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
            
            # Charge head: outputs 1 value (charge logit)
            self.charge_head = nn.Sequential(
                nn.Linear(dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            # Shared regression head: single hidden layer with GELU
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
        if self.use_multi_cls:
            heads = [self.eta_head, self.phi_head, self.pt_head, self.charge_head]
        else:
            heads = [self.regression_head, self.classification_head]
        
        for module in heads:
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
              Position 0 (or 0-3 for multi-CLS) is placeholder for CLS token(s).
            - hit_mask: Boolean mask of shape (B, seq_len), True for valid positions.
            - sequence_lengths: Optional tensor of actual sequence lengths (B,).
            
        Returns
        -------
        dict with keys:
            - regression: (B, num_regression_outputs) - raw network outputs
            - charge_logit: (B, 1) - charge classification logit
            - cls_embedding: (B, dim) or (B, 4, dim) - CLS token embedding(s) (for analysis)
            
        In delta prediction mode, also includes:
            - ref_eta: (B,) - reference eta from innermost hit
            - ref_phi: (B,) - reference phi from innermost hit
        """
        hit_features = inputs['hit_features']
        hit_mask = inputs['hit_mask']
        
        B, seq_len, _ = hit_features.shape
        
        # Extract reference values for delta prediction BEFORE modifying hit_features
        hit_start_idx = self.num_cls_tokens
        ref_eta = None
        ref_phi = None
        if self.use_delta_prediction:
            if self.delta_reference == "innermost":
                # Hits are sorted by r ascending with CLS at position(s) 0(-3)
                ref_eta = hit_features[:, hit_start_idx, self.eta_feature_idx]  # (B,)
                ref_phi = hit_features[:, hit_start_idx, self.phi_feature_idx]  # (B,)
            elif self.delta_reference == "average":
                # Compute masked average over all valid hits (excluding CLS tokens)
                hit_eta = hit_features[:, hit_start_idx:, self.eta_feature_idx]  # (B, seq_len-num_cls)
                hit_phi = hit_features[:, hit_start_idx:, self.phi_feature_idx]  # (B, seq_len-num_cls)
                valid_mask = hit_mask[:, hit_start_idx:].float()  # (B, seq_len-num_cls)
                
                # Masked mean
                ref_eta = (hit_eta * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
                ref_phi = (hit_phi * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
            else:
                raise ValueError(f"Unknown delta_reference: {self.delta_reference}")
        
        # Project hit features to model dimension
        x = self.input_projection(hit_features)  # (B, seq_len, dim)
        
        if self.use_multi_cls:
            # Multi-CLS mode: Replace positions 0-3 with 4 separate CLS tokens
            cls_tokens = self.cls_tokens.expand(B, -1, -1)  # (B, 4, dim)
            x = torch.cat([cls_tokens, x[:, self.num_cls_tokens:]], dim=1)  # (B, seq_len, dim)
        else:
            # Single CLS mode: Replace position 0 with learnable CLS token
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
        
        # Pass padding mask via q_mask/kv_mask parameters
        # The Attention layer merges these with any explicit attn_mask
        # True values indicate valid positions (not masked out)
        
        # Process through Transformer encoder with padding masks
        x = self.encoder(x, q_mask=hit_mask, kv_mask=hit_mask)  # (B, seq_len, dim)
        
        if self.use_multi_cls:
            # Extract 4 separate CLS token representations
            cls_eta = x[:, 0]     # (B, dim) - for eta prediction
            cls_phi = x[:, 1]     # (B, dim) - for phi prediction
            cls_pt = x[:, 2]      # (B, dim) - for pt prediction
            cls_charge = x[:, 3]  # (B, dim) - for charge prediction
            
            # Apply separate heads to each CLS token
            eta_output = self.eta_head(cls_eta)        # (B, 1)
            phi_output = self.phi_head(cls_phi)        # (B, 1) or (B, 2)
            pt_output = self.pt_head(cls_pt)           # (B, 1)
            charge_logit = self.charge_head(cls_charge)  # (B, 1)
            
            # Concatenate regression outputs: [eta, phi..., pt]
            regression = torch.cat([eta_output, phi_output, pt_output], dim=-1)  # (B, 3) or (B, 4)
            
            # Stack CLS embeddings for analysis
            cls_embedding = torch.stack([cls_eta, cls_phi, cls_pt, cls_charge], dim=1)  # (B, 4, dim)
        else:
            # Extract CLS token output (position 0)
            cls_embedding = x[:, 0]  # (B, dim)
            
            # Apply heads
            regression = self.regression_head(cls_embedding)  # (B, num_regression_outputs)
            charge_logit = self.classification_head(cls_embedding)  # (B, 1)
        
        outputs = {
            'regression': regression,
            'charge_logit': charge_logit,
            'cls_embedding': cls_embedding,
        }
        
        # Include reference values for delta computation during loss
        if self.use_delta_prediction:
            outputs['ref_eta'] = ref_eta
            outputs['ref_phi'] = ref_phi
        
        return outputs
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert model outputs to physics predictions.
        
        Handles the conversion from internal representation to physical quantities:
        - Standard mode: eta, phi (from sin/cos), pt (from log)
        - Delta mode: eta = ref + delta, phi = ref + delta, pt (from log)
        - Hybrid mode: eta = ref + delta, phi (from sin/cos), pt (from log)
        
        Parameters
        ----------
        outputs : dict
            Model forward outputs.
            
        Returns
        -------
        dict with keys:
            - eta: (B,) predicted eta values
            - phi: (B,) predicted phi values in [-pi, pi]
            - pt: (B,) predicted transverse momentum
            - charge: (B,) predicted charge (-1 or +1)
        """
        regression = outputs['regression']
        charge_logit = outputs['charge_logit']
        
        # Extract components based on mode
        if self.use_delta_prediction:
            if self.use_delta_phi:
                # Full delta mode: [delta_eta, delta_phi, log_pt]
                delta_eta = regression[:, 0]
                delta_phi = regression[:, 1]
                log_pt = regression[:, 2]
                
                # Reconstruct absolute values
                ref_eta = outputs['ref_eta']
                ref_phi = outputs['ref_phi']
                eta = ref_eta + delta_eta
                phi = ref_phi + delta_phi
                # Normalize phi to [-pi, pi]
                phi = torch.atan2(torch.sin(phi), torch.cos(phi))
            else:
                # Hybrid mode: [delta_eta, sin_phi, cos_phi, log_pt]
                delta_eta = regression[:, 0]
                sin_phi = regression[:, 1]
                cos_phi = regression[:, 2]
                log_pt = regression[:, 3]
                
                # Reconstruct eta
                ref_eta = outputs['ref_eta']
                eta = ref_eta + delta_eta
                
                # phi from sin/cos (ignores ref_phi)
                phi = torch.atan2(sin_phi, cos_phi)
        else:
            # Standard mode: [eta, sin_phi, cos_phi, log_pt]
            eta = regression[:, 0]
            sin_phi = regression[:, 1]
            cos_phi = regression[:, 2]
            log_pt = regression[:, 3]
            
            # Reconstruct phi
            phi = torch.atan2(sin_phi, cos_phi)
        
        # Reconstruct pt from log_pt
        pt = torch.exp(log_pt)
        
        # Charge prediction from logit
        charge = torch.where(charge_logit[:, 0] > 0, 1.0, -1.0)
        
        return {
            'eta': eta,
            'phi': phi,
            'pt': pt,
            'charge': charge,
        }
