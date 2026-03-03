"""YOLO-style track parameter regressor using Mamba encoder.

3-stage YOLO-inspired training pipeline:
1. Classification pretraining: Classify track parameters into discrete bins
2. Regression warmup: Train regression heads with frozen encoder  
3. Joint finetuning: End-to-end refinement of all heads

Architecture:
1. Input projection: hit features -> model dimension
2. CLS token (or attention pooling) for sequence aggregation
3. Bidirectional Mamba encoder
4. Classification heads: eta, phi, pt, charge (4 separate heads)
   - Supports hierarchical multi-resolution heads (up to 3 levels per variable)
5. Regression heads: eta, phi, pt (for stages 2 & 3, output per-bin offsets)

Features:
- Focal loss for pT (handles class imbalance in quantile bins)
- Variable bin counts for different resolution experiments
- Hierarchical classification: multiple resolution levels per variable
- 1/pt binning support (detector measures curvature directly)
- Freezing mechanisms for multi-stage training
- Regression heads output per-bin offset predictions (same size as classification)
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np

from hepattn.models.mamba import BidirectionalMambaEncoder


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values focus more on hard examples.
    alpha : float or None
        Class weight. If None, no weighting.
    reduction : str
        'mean', 'sum', or 'none'.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.
        
        Parameters
        ----------
        logits : Tensor
            Raw logits, shape (B, num_classes).
        targets : Tensor
            Class indices, shape (B,).
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE) when CE = -log(p_t)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            focal_weight = self.alpha * focal_weight
            
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence.
    
    Uses a learnable query to compute attention weights over the sequence,
    then returns a weighted sum as the pooled representation.
    """
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        
        # Key and value projections
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Pool sequence into single vector.
        
        Parameters
        ----------
        x : Tensor
            Sequence features, shape (B, seq_len, dim).
        mask : Tensor
            Boolean mask, True for valid positions, shape (B, seq_len).
            
        Returns
        -------
        Tensor
            Pooled representation, shape (B, dim).
        """
        B, N, C = x.shape
        
        # Query: (1, 1, dim) -> (B, 1, num_heads, head_dim) -> (B, num_heads, 1, head_dim)
        q = self.query.expand(B, -1, -1).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Key, Value: (B, N, dim) -> (B, N, 2, num_heads, head_dim) -> 2 x (B, num_heads, N, head_dim)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention: (B, num_heads, 1, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, 1, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum: (B, num_heads, 1, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, 1, head_dim)
        out = attn @ v
        
        # Reshape: (B, num_heads, 1, head_dim) -> (B, 1, dim) -> (B, dim)
        out = out.transpose(1, 2).reshape(B, 1, C).squeeze(1)
        out = self.proj(out)
        
        return out


class YOLORegressor(nn.Module):
    """Bidirectional Mamba model for YOLO-style track parameter regression.
    
    Designed for YOLO-inspired 3-stage training:
    - Stage 1: Classification pretraining (classification heads only)
    - Stage 2: Regression warmup (regression heads with frozen encoder)
    - Stage 3: Joint finetuning (all heads, unfrozen encoder)
    
    Architecture:
    - Input projection: hit features -> model dimension
    - CLS token (or attention pooling) for sequence aggregation
    - Bidirectional Mamba encoder
    - Classification heads: eta, phi, pt/inv_pt, charge
      - Supports hierarchical multi-resolution: pass list of bin counts per level
    - Regression heads: eta, phi, pt (output per-bin offsets, same size as finest cls)
    
    Hierarchical classification:
        Pass eta_bins=[50, 200, 501] for 3 levels of refinement.
        Level 0 (coarsest) classifies into 50 bins, level 1 into 200, level 2 into 501.
        All levels share the same encoder embedding.
        Pass a scalar (e.g., eta_bins=501) for single-level (backward compatible).
    
    1/pt binning:
        Set use_inv_pt=True to classify in 1/pt space instead of pt.
        The model output key is 'inv_pt_logits' (or 'inv_pt_logits_L{i}' for hierarchical).
        The training module handles converting predictions back to pt for metrics.
    
    Parameters
    ----------
    input_dim : int
        Number of input features per hit.
    dim : int
        Model embedding dimension.
    num_layers : int
        Number of bidirectional Mamba layers.
    eta_bins : int or list[int]
        Number of eta classification bins per level. List for hierarchical.
    phi_bins : int or list[int]
        Number of phi classification bins per level. List for hierarchical.
    pt_bins : int or list[int]
        Number of pT (or 1/pT) classification bins per level. List for hierarchical.
    use_inv_pt : bool
        If True, pt heads classify 1/pt instead of pt. Keys use 'inv_pt' prefix.
    use_attention_pooling : bool
        If True, use attention pooling instead of CLS token.
    enable_regression_heads : bool
        If True, create regression heads (for stages 2 & 3).
        Regression uses finest level bins only.
    regression_hidden_dim : int
        Hidden dimension for regression heads.
    regression_num_layers : int
        Number of layers in regression heads (1, 2, or 3).
    """
    
    def __init__(
        self,
        input_dim: int = 27,
        dim: int = 128,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba2: bool = True,
        headdim: int = 32,
        norm: str = "RMSNorm",
        dropout: float = 0.0,
        head_dropout: float = 0.15,
        head_hidden_dim: int = 256,
        # Classification/regression bins — scalar or list for hierarchical
        eta_bins: int | list[int] = 100,
        phi_bins: int | list[int] = 100,
        pt_bins: int | list[int] = 50,
        use_inv_pt: bool = False,
        use_attention_pooling: bool = False,
        attention_heads: int = 4,
        # Regression heads (disabled by default for Stage 1)
        enable_regression_heads: bool = False,
        regression_hidden_dim: int = 256,
        regression_num_layers: int = 2,
        regression_dropout: float = 0.15,
    ):
        super().__init__()
        
        self.dim = dim
        self.use_inv_pt = use_inv_pt
        self.use_attention_pooling = use_attention_pooling
        self.enable_regression_heads = enable_regression_heads
        
        # Normalize bins to lists for hierarchical support
        self.eta_bins_list = [eta_bins] if isinstance(eta_bins, int) else list(eta_bins)
        self.phi_bins_list = [phi_bins] if isinstance(phi_bins, int) else list(phi_bins)
        self.pt_bins_list = [pt_bins] if isinstance(pt_bins, int) else list(pt_bins)
        
        self.num_eta_levels = len(self.eta_bins_list)
        self.num_phi_levels = len(self.phi_bins_list)
        self.num_pt_levels = len(self.pt_bins_list)
        
        # Backward-compatible properties: finest level bin count
        self.eta_bins = self.eta_bins_list[-1]
        self.phi_bins = self.phi_bins_list[-1]
        self.pt_bins = self.pt_bins_list[-1]
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
        )
        
        # Pooling mechanism
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(dim, num_heads=attention_heads)
            self.num_cls_tokens = 0
        else:
            # CLS token
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.num_cls_tokens = 1
        
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
        
        # ===== Classification heads (hierarchical) =====
        # Each variable has one head per resolution level
        self.eta_cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, n_bins),
            )
            for n_bins in self.eta_bins_list
        ])
        
        self.phi_cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, n_bins),
            )
            for n_bins in self.phi_bins_list
        ])
        
        # pt (or inv_pt) heads
        self.pt_cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, n_bins),
            )
            for n_bins in self.pt_bins_list
        ])
        
        # Charge head (always single, binary)
        self.charge_head = nn.Sequential(
            nn.Linear(dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        
        # Backward-compatible aliases: point to finest-level head
        self.eta_cls_head = self.eta_cls_heads[-1]
        self.phi_cls_head = self.phi_cls_heads[-1]
        self.pt_cls_head = self.pt_cls_heads[-1]
        
        # ===== Regression heads (optional, finest level only) =====
        if enable_regression_heads:
            self.eta_reg_head = self._make_regression_head(
                dim, regression_hidden_dim, regression_num_layers, regression_dropout,
                output_dim=self.eta_bins
            )
            self.phi_reg_head = self._make_regression_head(
                dim, regression_hidden_dim, regression_num_layers, regression_dropout,
                output_dim=self.phi_bins
            )
            self.pt_reg_head = self._make_regression_head(
                dim, regression_hidden_dim, regression_num_layers, regression_dropout,
                output_dim=self.pt_bins
            )
        
        # Initialize weights
        self._init_weights()
        
    def _make_regression_head(
        self, 
        dim: int, 
        hidden_dim: int, 
        num_layers: int,
        dropout: float,
        output_dim: int,
    ) -> nn.Module:
        """Create a regression head with variable depth.
        
        Parameters
        ----------
        dim : int
            Input dimension (model embedding dim).
        hidden_dim : int
            Hidden layer dimension.
        num_layers : int
            Number of layers (1, 2, or 3).
        dropout : float
            Dropout rate.
        output_dim : int
            Number of output values (same as number of bins).
            
        Returns
        -------
        nn.Module
            Regression head that outputs per-bin offset predictions.
        """
        if num_layers == 1:
            return nn.Linear(dim, output_dim)
        elif num_layers == 2:
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:  # num_layers >= 3
            return nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
    
    def _init_weights(self):
        """Initialize head weights."""
        heads = list(self.eta_cls_heads) + list(self.phi_cls_heads) + list(self.pt_cls_heads) + [self.charge_head]
        if self.enable_regression_heads:
            heads.extend([self.eta_reg_head, self.phi_reg_head, self.pt_reg_head])
            
        for head in heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def freeze_encoder(self):
        """Freeze encoder parameters for regression warmup (Stage 2)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.input_projection.parameters():
            param.requires_grad = False
        self.pos_embedding.requires_grad = False
        if not self.use_attention_pooling:
            self.cls_token.requires_grad = False
        else:
            for param in self.attention_pool.parameters():
                param.requires_grad = False
                
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for joint finetuning (Stage 3)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.input_projection.parameters():
            param.requires_grad = True
        self.pos_embedding.requires_grad = True
        if not self.use_attention_pooling:
            self.cls_token.requires_grad = True
        else:
            for param in self.attention_pool.parameters():
                param.requires_grad = True
                
    def freeze_classification_heads(self):
        """Freeze classification heads."""
        for head_list in [self.eta_cls_heads, self.phi_cls_heads, self.pt_cls_heads]:
            for head in head_list:
                for param in head.parameters():
                    param.requires_grad = False
        for param in self.charge_head.parameters():
            param.requires_grad = False
                
    def unfreeze_classification_heads(self):
        """Unfreeze classification heads."""
        for head_list in [self.eta_cls_heads, self.phi_cls_heads, self.pt_cls_heads]:
            for head in head_list:
                for param in head.parameters():
                    param.requires_grad = True
        for param in self.charge_head.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        inputs: dict[str, Tensor],
        run_classification: bool = True,
        run_regression: bool = False,
    ) -> dict[str, Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        inputs : dict
            - hit_features: (B, seq_len, input_dim)
            - hit_mask: (B, seq_len), True for valid positions
        run_classification : bool
            If True, run classification heads.
        run_regression : bool
            If True, run regression heads (requires enable_regression_heads=True).
            
        Returns
        -------
        dict with:
            Classification outputs (if run_classification=True):
            For single-level (backward compatible):
                - eta_logits: (B, eta_bins)
                - phi_logits: (B, phi_bins)
                - pt_logits or inv_pt_logits: (B, pt_bins)
                - charge_logit: (B, 1)
            For multi-level hierarchical:
                - eta_logits_L0, eta_logits_L1, ...: coarse to fine
                - phi_logits_L0, phi_logits_L1, ...
                - pt_logits_L0, pt_logits_L1, ... (or inv_pt_logits_L0, ...)
                - eta_logits: alias for finest level
                - phi_logits: alias for finest level
                - pt_logits/inv_pt_logits: alias for finest level
                - charge_logit: (B, 1)
            
            Regression outputs (if run_regression=True):
                - eta_offsets: (B, eta_bins) finest level
                - phi_offsets: (B, phi_bins) finest level
                - pt_offsets/inv_pt_offsets: (B, pt_bins) finest level
            
            Always returned:
                - cls_embedding: (B, dim)
        """
        hit_features = inputs['hit_features']
        hit_mask = inputs['hit_mask']
        B, seq_len, _ = hit_features.shape
        
        # Project hit features
        x = self.input_projection(hit_features)
        
        if self.use_attention_pooling:
            x = x + self.pos_embedding[:, :seq_len, :]
            x = self.encoder(x)
            cls_embedding = self.attention_pool(x, hit_mask)
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x[:, 1:, :]], dim=1)
            x = x + self.pos_embedding[:, :seq_len, :]
            x = self.encoder(x)
            cls_embedding = x[:, 0, :]
        
        outputs = {'cls_embedding': cls_embedding}
        
        # Determine pt key prefix
        pt_prefix = 'inv_pt' if self.use_inv_pt else 'pt'
        
        # Classification heads
        if run_classification:
            # Eta heads (all levels)
            for i, head in enumerate(self.eta_cls_heads):
                key = f'eta_logits_L{i}' if self.num_eta_levels > 1 else 'eta_logits'
                outputs[key] = head(cls_embedding)
            
            # Phi heads (all levels)
            for i, head in enumerate(self.phi_cls_heads):
                key = f'phi_logits_L{i}' if self.num_phi_levels > 1 else 'phi_logits'
                outputs[key] = head(cls_embedding)
            
            # PT/inv_pt heads (all levels)
            for i, head in enumerate(self.pt_cls_heads):
                key = f'{pt_prefix}_logits_L{i}' if self.num_pt_levels > 1 else f'{pt_prefix}_logits'
                outputs[key] = head(cls_embedding)
            
            # Always provide finest-level aliases for backward compatibility
            if self.num_eta_levels > 1:
                outputs['eta_logits'] = outputs[f'eta_logits_L{self.num_eta_levels - 1}']
            if self.num_phi_levels > 1:
                outputs['phi_logits'] = outputs[f'phi_logits_L{self.num_phi_levels - 1}']
            if self.num_pt_levels > 1:
                outputs[f'{pt_prefix}_logits'] = outputs[f'{pt_prefix}_logits_L{self.num_pt_levels - 1}']
            
            # Charge head (always single level)
            outputs['charge_logit'] = self.charge_head(cls_embedding)
            
            # For backward compatibility: if using inv_pt, also provide pt_logits alias
            if self.use_inv_pt:
                outputs['pt_logits'] = outputs['inv_pt_logits']
        
        # Regression heads (finest level only)
        if run_regression:
            if not self.enable_regression_heads:
                raise ValueError(
                    "Regression heads not enabled. Set enable_regression_heads=True "
                    "in model config to use regression."
                )
            outputs['eta_offsets'] = self.eta_reg_head(cls_embedding)
            outputs['phi_offsets'] = self.phi_reg_head(cls_embedding)
            outputs[f'{pt_prefix}_offsets'] = self.pt_reg_head(cls_embedding)
            # Backward compatibility alias
            if self.use_inv_pt:
                outputs['pt_offsets'] = outputs['inv_pt_offsets']
        
        return outputs


# Backward compatibility alias
MambaClassifier = YOLORegressor
