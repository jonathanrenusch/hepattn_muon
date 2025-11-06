from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn

from hepattn.models.dense import Dense
from hepattn.models.loss import cost_fns, focal_loss, loss_fns


class Task(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.has_intermediate_loss = False

    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the forward pass of the task."""

    @abstractmethod
    def predict(self, outputs: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """Return predictions from model outputs."""

    @abstractmethod
    def loss(self, outputs: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute loss between outputs and targets."""

    def cost(self, outputs: dict[str, Tensor], targets: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        return {}

    def attn_mask(self, outputs, **kwargs):
        return {}

    def key_mask(self, outputs, **kwargs):
        return {}

    def query_mask(self, outputs, **kwargs):
        return None


class ObjectValidTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_queries: bool = False,
    ):
        """Task used for classifying whether object candidates / seeds should be
        taken as reconstructed / pred objects or not.

        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_object : str
            Name of the input object feature
        output_object : str
            Name of the output object feature which will denote if the predicted object slot is used or not.
        target_object: str
            Name of the target object feature that we want to predict is valid or not.
        losses : dict[str, float]
            Dict specifying which losses to use. Keys denote the loss function name,
            whiel value denotes loss weight.
        costs : dict[str, float]
            Dict specifying which costs to use. Keys denote the cost function name,
            whiel value denotes cost weight.
        dim : int
            Embedding dimension of the input features.
        null_weight : float
            Weight applied to the null class in the loss. Useful if many instances of
            the target class are null, and we need to reweight to overcome class imbalance.
        """
        super().__init__()

        self.name = name
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_queries = mask_queries

        # Internal
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_logit"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Network projects the embedding down into a scalar
        x_logit = self.net(x[self.input_object + "_embed"])
        return {self.output_object + "_logit": x_logit.squeeze(-1)}

    def predict(self, outputs, threshold=0.5):
        # Objects that have a predicted probability aove the threshold are marked as predicted to exist
        return {self.output_object + "_valid": outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold}

    def cost(self, outputs, targets):
        output = outputs[self.output_object + "_logit"].detach().to(torch.float32)
        target = targets[self.target_object + "_valid"].to(torch.float32)
        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
        return costs

    # def cost(self, outputs, targets):
    #     output = outputs[self.output_object + "_logit"].detach().to(torch.float32)
    #     target = targets[self.target_object + "_valid"].to(torch.float32)
    #     costs = {}
    #     for cost_fn, cost_weight in self.costs.items():
    #         cost_value = cost_weight * cost_fns[cost_fn](output, target)
            
    #         # Add debugging here
    #         # if torch.isnan(cost_value).any() or torch.isinf(cost_value).any():
    #         #     print(f"Invalid cost in {self.name}.{cost_fn}:")
    #         #     print(f"  NaN count: {torch.isnan(cost_value).sum()}")
    #         #     print(f"  Inf count: {torch.isinf(cost_value).sum()}")
    #         #     print(f"  Cost shape: {cost_value.shape}")
    #         #     print(f"  Output shape: {output.shape}, min: {output.min()}, max: {output.max()}")
    #         #     print(f"  Target shape: {target.shape}, min: {target.min()}, max: {target.max()}")
    #         #     print(f"  Cost weight: {cost_weight}")
            
    #         costs[cost_fn] = cost_value
    #     return costs


    def loss(self, outputs, targets):
        losses = {}
        output = outputs[self.output_object + "_logit"]
        target = targets[self.target_object + "_valid"].type_as(output)
        weight = target + self.null_weight * (1 - target)
        # Calculate the loss from each specified loss function.
        for loss_fn, loss_weight in self.losses.items():
            losses[loss_fn] = loss_weight * loss_fns[loss_fn](output, target, mask=None, weight=weight)
        return losses

    def query_mask(self, outputs, threshold=0.1):
        if not self.mask_queries:
            return None

        return outputs[self.output_object + "_logit"].detach().sigmoid() >= threshold


class HitFilterTask(Task):
    def __init__(
        self,
        name: str,
        input_object: str,
        target_field: str,
        dim: int,
        threshold: float = 0.1,
        mask_keys: bool = False,
        loss_fn: Literal["bce", "focal", "both"] = "bce",
    ):
        """Task used for classifying whether hits belong to reconstructable objects or not."""
        super().__init__()

        self.name = name
        self.input_object = input_object
        self.target_field = target_field
        self.dim = dim
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.mask_keys = mask_keys

        # Internal
        self.input_objects = [f"{input_object}_embed"]
        self.net = Dense(dim, 1)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x_logit = self.net(x[f"{self.input_object}_embed"])
        return {f"{self.input_object}_logit": x_logit.squeeze(-1)}

    def predict(self, outputs: dict) -> dict:
        return {f"{self.input_object}_{self.target_field}": outputs[f"{self.input_object}_logit"].sigmoid() >= self.threshold}

    def loss(self, outputs: dict, targets: dict) -> dict:
        # Pick out the field that denotes whether a hit is on a reconstructable object or not
        output = outputs[f"{self.input_object}_logit"]
        target = targets[f"{self.input_object}_{self.target_field}"].type_as(output)

        # Calculate the BCE loss with class weighting
        if self.loss_fn == "bce":
            weight = 1 / target.float().mean()
            loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "focal":
            loss = focal_loss(output, target)
            return {f"{self.input_object}_{self.loss_fn}": loss}
        if self.loss_fn == "both":
            weight = 1 / target.float().mean()
            bce_loss = nn.functional.binary_cross_entropy_with_logits(output, target, pos_weight=weight)
            focal_loss_value = focal_loss(output, target)
            return {
                f"{self.input_object}_bce": bce_loss,
                f"{self.input_object}_focal": focal_loss_value,
            }
        raise ValueError(f"Unknown loss function: {self.loss_fn}")

    def key_mask(self, outputs, threshold=0.1):
        if not self.mask_keys:
            return {}

        return {self.input_object: outputs[f"{self.input_object}_logit"].detach().sigmoid() >= threshold}


class ObjectHitMaskTask(Task):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        losses: dict[str, float],
        costs: dict[str, float],
        dim: int,
        null_weight: float = 1.0,
        mask_attn: bool = True,
    ):
        super().__init__()



        self.name = name
        self.input_hit = input_hit
        self.input_object = input_object
        self.output_object = output_object
        self.target_object = target_object
        self.losses = losses
        self.costs = costs
        self.dim = dim
        self.null_weight = null_weight
        self.mask_attn = mask_attn
        self.has_intermediate_loss = mask_attn

        self.output_object_hit = output_object + "_" + input_hit
        self.target_object_hit = target_object + "_" + input_hit
        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object_hit + "_logit"]
        self.hit_net = Dense(dim, dim)
        self.object_net = Dense(dim, dim)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Produce new task-specific embeddings for the hits and objects
        x_object = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_hit + "_embed"])

        # Object-hit probability is the dot product between the hit and object embedding
        object_hit_logit = torch.einsum("bnc,bmc->bnm", x_object, x_hit)

        # Zero out entries for any hit slots that are not valid
        object_hit_logit[~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(object_hit_logit)] = torch.finfo(object_hit_logit.dtype).min

        return {self.output_object_hit + "_logit": object_hit_logit}

    # def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
    #     # Produce new task-specific embeddings for the hits and objects
    #     x_object = self.object_net(x[self.input_object + "_embed"])
    #     x_hit = self.hit_net(x[self.input_hit + "_embed"])

    #     # print(f"x_object stats: shape={x_object.shape}, min={x_object.min()}, max={x_object.max()}, has_nan={torch.isnan(x_object).any()}")
    #     # print(f"x_hit stats: shape={x_hit.shape}, min={x_hit.min()}, max={x_hit.max()}, has_nan={torch.isnan(x_hit).any()}")

    #     # Object-hit probability is the dot product between the hit and object embedding
    #     object_hit_logit = torch.einsum("bnc,bmc->bnm", x_object, x_hit)
        
    #     # print(f"After einsum: shape={object_hit_logit.shape}, min={object_hit_logit.min()}, max={object_hit_logit.max()}, has_nan={torch.isnan(object_hit_logit).any()}")

    #     # Zero out entries for any hit slots that are not valid
    #     mask_value = torch.finfo(object_hit_logit.dtype).min
    #     # print(f"Mask value: {mask_value}")
        
    #     mask = ~x[self.input_hit + "_valid"].unsqueeze(-2).expand_as(object_hit_logit)
    #     # print(f"Mask stats: shape={mask.shape}, true_count={mask.sum()}")
        
    #     object_hit_logit[mask] = mask_value
        
    #     # print(f"After masking: shape={object_hit_logit.shape}, min={object_hit_logit.min()}, max={object_hit_logit.max()}, has_nan={torch.isnan(object_hit_logit).any()}")

    #     return {self.output_object_hit + "_logit": object_hit_logit}

    def attn_mask(self, outputs, threshold=0.1):
        if not self.mask_attn:
            return {}

        attn_mask = outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold

        # If the attn mask is completely padded for a given entry, unpad it - tested and is required (?)
        # TODO: See if the query masking stops this from being necessary
        attn_mask[torch.where(torch.all(attn_mask, dim=-1))] = False

        return {self.input_hit: attn_mask}

    def predict(self, outputs, threshold=0.5):
        # Object-hit pairs that have a predicted probability above the threshold are predicted as being associated to one-another
        return {self.output_object_hit + "_valid": outputs[self.output_object_hit + "_logit"].detach().sigmoid() >= threshold}

    # def cost(self, outputs, targets):
    #     output = outputs[self.output_object_hit + "_logit"].detach().to(torch.float32)
    #     target = targets[self.target_object_hit + "_valid"].to(torch.float32)

    #     costs = {}
    #     for cost_fn, cost_weight in self.costs.items():
    #         costs[cost_fn] = cost_weight * cost_fns[cost_fn](output, target)
    #     return costs

    def cost(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"].detach().to(torch.float32)
        
        target = targets[self.target_object_hit + "_valid"].to(torch.float32)
        # print("This is output from cost function:", output)
        # print("This is target from cost function!", target)
        # print("This is output from cost function:", output.shape)
        # print("This is target from cost function!", target.shape)
        # print("Quick gut check:", torch.sum(target, dim=-1))

        costs = {}
        for cost_fn, cost_weight in self.costs.items():
            cost_value = cost_weight * cost_fns[cost_fn](output, target)
            
            # Add debugging here
            # if torch.isnan(cost_value).any() or torch.isinf(cost_value).any():
            #     print(f"Invalid cost in {self.name}.{cost_fn}:")
            #     print(f"  NaN count: {torch.isnan(cost_value).sum()}")
            #     print(f"  Inf count: {torch.isinf(cost_value).sum()}")
            #     print(f"  Cost shape: {cost_value.shape}")
            #     print(f"  Output shape: {output.shape}, min: {output.min()}, max: {output.max()}")
            #     print(f"  Target shape: {target.shape}, min: {target.min()}, max: {target.max()}")
            #     print(f"  Cost weight: {cost_weight}")
            #     print(f"  Raw cost (before weight): min: {(cost_fns[cost_fn](output, target)).min()}, max: {(cost_fns[cost_fn](output, target)).max()}")
            
            costs[cost_fn] = cost_value
        
        return costs

    def loss(self, outputs, targets):
        output = outputs[self.output_object_hit + "_logit"]
        target = targets[self.target_object_hit + "_valid"].type_as(output)

        # Build a padding mask for object-hit pairs
        hit_pad = targets[self.input_hit + "_valid"].unsqueeze(-2).expand_as(target)
        object_pad = targets[self.target_object + "_valid"].unsqueeze(-1).expand_as(target)
        # An object-hit is valid slot if both its object and hit are valid slots
        # TODO: Maybe calling this a mask is confusing since true entries are
        object_hit_mask = object_pad & hit_pad

        weight = target + self.null_weight * (1 - target)

        losses = {}
        for loss_fn, loss_weight in self.losses.items():
            loss = loss_fns[loss_fn](output, target, mask=object_hit_mask, weight=weight)
            losses[loss_fn] = loss_weight * loss
        return losses


class RegressionTask(Task):
    def __init__(
        self,
        name: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
    ):
        super().__init__()

        self.name = name
        self.output_object = output_object
        self.target_object = target_object
        self.fields = fields
        self.loss_weight = loss_weight
        self.k = len(fields)
        # For standard regression number of DoFs is just the number of targets
        self.ndofs = self.k

    def forward(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        # For a standard regression task, the raw network output is the final prediction
        latent = self.latent(x, pads=pads)
        return {self.output_object + "_regr": latent}

    def predict(self, outputs):
        # Split the regression vectior into the separate fields
        latent = outputs[self.output_object + "_regr"]
        return {self.output_object + "_" + field: latent[..., i] for i, field in enumerate(self.fields)}

    def loss(self, outputs, targets):
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        output = outputs[self.output_object + "_regr"]

        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]

        # Compute the loss
        loss = torch.nn.functional.smooth_l1_loss(output, target, reduction="none")

        # Average over all the features
        loss = torch.mean(loss, dim=-1)

        # Compute the regression loss only for valid objects
        return {"smooth_l1": self.loss_weight * loss.mean()}

    def metrics(self, preds, targets):
        metrics = {}
        for field in self.fields:
            # Get the target and prediction only for valid targets
            pred = preds[self.output_object + "_" + field][targets[self.target_object + "_valid"]]
            target = targets[self.target_object + "_" + field][targets[self.target_object + "_valid"]]
            # Get the error between the prediction and target for this field
            err = pred - target
            # Compute the RMSE and log it
            metrics[field + "_rmse"] = torch.sqrt(torch.mean(torch.square(err)))
            # Compute the relative error / resolution and log it
            metrics[field + "_mean_res"] = torch.mean(err / target)
            metrics[field + "_std_res"] = torch.std(err / target)

        return metrics


class ObjectRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight)

        self.input_object = input_object
        self.inputs = [input_object + "_embed"]
        self.outputs = [output_object + "_regr"]

        self.dim = dim
        self.net = Dense(self.dim, self.ndofs)

    def latent(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> Tensor:
        return self.net(x[self.input_object + "_embed"])


class ObjectHitRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        fields: list[str],
        loss_weight: float,
        dim: int,
    ):
        super().__init__(name, output_object, target_object, fields, loss_weight)

        self.input_hit = input_hit
        self.input_object = input_object

        self.inputs = [input_object + "_embed", input_hit + "_embed"]
        self.outputs = [self.output_object + "_regr"]

        self.dim = dim
        self.dim_per_dof = self.dim // self.ndofs

        self.hit_net = Dense(dim, self.ndofs * self.dim_per_dof)
        self.object_net = Dense(dim, self.ndofs * self.dim_per_dof)

    def latent(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> Tensor:
        # Embed the hits and tracks and reshape so we have a separate embedding for each DoF
        x_obj = self.object_net(x[self.input_object + "_embed"])
        x_hit = self.hit_net(x[self.input_hit + "_embed"])

        x_obj = x_obj.reshape(x_obj.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BNDE
        x_hit = x_hit.reshape(x_hit.size()[:-1] + torch.Size((self.ndofs, self.dim_per_dof)))  # Shape BMDE

        # Take the dot product between the hits and tracks over the last embedding dimension so we are left
        # with just a scalar for each degree of freedom
        x_obj_hit = torch.einsum("...nie,...mie->...nmi", x_obj, x_hit)  # Shape BNMD

        # If padding data is provided, use it to zero out predictions for any hit slots that are not valid
        if pads is not None:
            # Shape of padding goes BM -> B1M -> B1M1 -> BNMD
            x_obj_hit = x_obj_hit * pads[self.input_hit + "_valid"].unsqueeze(-2).unsqueeze(-1).expand_as(x_obj_hit).float()
        return x_obj_hit


class WeightedPoolingObjectHitRegressionTask(RegressionTask):
    def __init__(
        self,
        name: str,
        input_hit: str,
        input_object: str,
        output_object: str,
        target_object: str,
        hit_fields: list[str],
        fields: list[str],
        loss_weight: float,
        dim: int,
        assignment_threshold: float = 0.1,
        regression_hidden_dim: int | None = None,
        regression_num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Regression task that aggregates hit information using weighted pooling.
        
        This task uses the hit-track assignment logits from ObjectHitMaskTask to create
        a weighted aggregation of both raw and encoded hit features for each track.
        
        Architecture:
        1. Get assignment probabilities from ObjectHitMaskTask logits
        2. Apply hard threshold (default 0.1) to filter hits
        3. Process raw hits through task-specific network (full dim)
        4. Process encoded hits through task-specific network (full dim)
        5. Apply multiple pooling operations:
           - Weighted average (uses assignment weights)
           - Weighted sum (uses assignment weights)
           - Max pooling (unweighted, over thresholded hits)
           - Min pooling (unweighted, over thresholded hits)
        6. Concatenate [query_embed, raw_pooled, encoded_pooled]
        7. Multi-layer regression head with dropout and layer norm
        
        Parameters
        ----------
        name : str
            Name of the task - will be used as the key to separate task outputs.
        input_hit : str
            Name of the input hit feature (e.g., "hit")
        input_object : str
            Name of the input object feature (e.g., "query")
        output_object : str
            Name of the output object feature (e.g., "track")
        target_object : str
            Name of the target object feature (e.g., "particle")
        hit_fields : list[str]
            List of raw hit field names to use as input features
        fields : list[str]
            List of regression target field names (e.g., ["truthMuon_eta", "truthMuon_phi", ...])
        loss_weight : float
            Weight applied to the regression loss
        dim : int
            Embedding dimension of the encoder output
        assignment_threshold : float
            Threshold for hard filtering of hit-track assignments (default: 0.1)
        regression_hidden_dim : int | None
            Hidden dimension for regression head. If None, defaults to aggregated_dim // 2.
        regression_num_layers : int
            Number of layers in regression head (default: 3). 
            1 = direct projection, 2 = one hidden layer, 3 = two hidden layers, etc.
        dropout : float
            Dropout probability for regularization in regression head (default: 0.1)
        """
        super().__init__(name, output_object, target_object, fields, loss_weight)

        self.input_hit = input_hit
        self.input_object = input_object
        self.hit_fields = hit_fields
        self.assignment_threshold = assignment_threshold

        self.inputs = [input_object + "_embed", input_hit + "_embed", input_hit]
        self.outputs = [self.output_object + "_regr"]

        self.dim = dim
        self.num_raw_features = len(hit_fields)
        
        # Task-specific networks for processing hits - USE FULL DIM (not dim//2)
        # This preserves more information from raw detector measurements and encoded features
        self.raw_hit_net = Dense(self.num_raw_features, dim)
        self.encoded_hit_net = Dense(dim, dim)
        
        # Build multi-layer regression head
        # Input: query_embed (dim) + raw_pooled (4*dim) + encoded_pooled (4*dim) = 9*dim
        aggregated_dim = 9 * dim
        
        if regression_hidden_dim is None:
            regression_hidden_dim = aggregated_dim // 2
        
        self.regression_head = self._build_regression_head(
            aggregated_dim,
            self.ndofs,
            regression_hidden_dim,
            regression_num_layers,
            dropout
        )
    
    def _build_regression_head(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build a multi-layer MLP for regression.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Number of regression targets
            hidden_dim: Hidden layer dimension
            num_layers: Total number of layers (including input and output)
            dropout: Dropout probability
            
        Returns:
            Sequential module implementing the regression head
        """
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        
        layers = []
        
        if num_layers == 1:
            # Direct projection (no hidden layers)
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # First layer: input → hidden
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            
            # Middle layers: hidden → hidden
            for _ in range(num_layers - 2):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout)
                ])
            
            # Output layer: hidden → output
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)

    def latent(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> Tensor:
        """Compute aggregated hit features for regression.
        
        Args:
            x: Dictionary containing:
                - input_object + "_embed": Query embeddings [B, N, D]
                - input_hit + "_embed": Encoded hit embeddings [B, M, D]
                - input_hit: Raw hit features [B, M, num_raw_features]
                - output_object + "_" + input_hit + "_logit": Assignment logits [B, N, M]
            pads: Optional padding information
            
        Returns:
            Aggregated feature tensor [B, N, 9*D] ready for regression head
        """
        batch_size = x[self.input_object + "_embed"].size(0)
        num_tracks = x[self.input_object + "_embed"].size(1)
        
        # Get query embeddings [B, N, D]
        query_embed = x[self.input_object + "_embed"]
        
        # Get assignment logits from ObjectHitMaskTask [B, N, M]
        assignment_logits = x[self.output_object + "_" + self.input_hit + "_logit"]
        num_hits = assignment_logits.size(2)  # M from [B, N, M]
        
        # Compute assignment probabilities
        assignment_probs = assignment_logits.sigmoid()  # [B, N, M]
        
        # Apply hard threshold to create mask
        assignment_mask = assignment_probs >= self.assignment_threshold  # [B, N, M]
        
        # Account for hit validity if available
        if self.input_hit + "_valid" in x:
            hit_valid = x[self.input_hit + "_valid"].unsqueeze(1).expand_as(assignment_mask)  # [B, N, M]
            assignment_mask = assignment_mask & hit_valid
        
        # Compute normalized weights for weighted pooling (only over thresholded hits)
        masked_probs = assignment_probs * assignment_mask.float()  # [B, N, M]
        weight_sum = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]
        weights = masked_probs / weight_sum  # [B, N, M], normalized
        
        # === Process Raw Hits ===
        # Stack raw hit features [B, M, num_raw_features]
        raw_hits = torch.stack([x[self.input_hit + "_" + field] for field in self.hit_fields], dim=-1)
        
        # Project raw hits through task-specific network [B, M, D]
        raw_hit_features = self.raw_hit_net(raw_hits)
        
        # Expand for broadcasting [B, 1, M, D] -> for track dimension
        raw_hit_features_expanded = raw_hit_features.unsqueeze(1).expand(batch_size, num_tracks, num_hits, self.dim)
        
        # Weighted average pooling [B, N, D]
        raw_weighted_avg = (weights.unsqueeze(-1) * raw_hit_features_expanded).sum(dim=2)
        
        # Weighted sum pooling [B, N, D]
        raw_weighted_sum = (masked_probs.unsqueeze(-1) * raw_hit_features_expanded).sum(dim=2)
        
        # Max pooling (unweighted, over thresholded hits) [B, N, D]
        raw_for_max = torch.where(
            assignment_mask.unsqueeze(-1).expand_as(raw_hit_features_expanded),
            raw_hit_features_expanded,
            torch.tensor(float('-inf'), device=raw_hit_features.device, dtype=raw_hit_features.dtype)
        )
        raw_max = raw_for_max.max(dim=2)[0]  # [B, N, D]
        # Replace -inf with 0 for tracks with no assigned hits
        raw_max = torch.where(torch.isinf(raw_max), torch.zeros_like(raw_max), raw_max)
        
        # Min pooling (unweighted, over thresholded hits) [B, N, D]
        raw_for_min = torch.where(
            assignment_mask.unsqueeze(-1).expand_as(raw_hit_features_expanded),
            raw_hit_features_expanded,
            torch.tensor(float('inf'), device=raw_hit_features.device, dtype=raw_hit_features.dtype)
        )
        raw_min = raw_for_min.min(dim=2)[0]  # [B, N, D]
        # Replace +inf with 0 for tracks with no assigned hits
        raw_min = torch.where(torch.isinf(raw_min), torch.zeros_like(raw_min), raw_min)
        
        # === Process Encoded Hits ===
        # Project encoded hits through task-specific network [B, M, D]
        encoded_hit_features = self.encoded_hit_net(x[self.input_hit + "_embed"])
        
        # Expand for broadcasting [B, 1, M, D] -> for track dimension
        encoded_hit_features_expanded = encoded_hit_features.unsqueeze(1).expand(batch_size, num_tracks, num_hits, self.dim)
        
        # Weighted average pooling [B, N, D]
        encoded_weighted_avg = (weights.unsqueeze(-1) * encoded_hit_features_expanded).sum(dim=2)
        
        # Weighted sum pooling [B, N, D]
        encoded_weighted_sum = (masked_probs.unsqueeze(-1) * encoded_hit_features_expanded).sum(dim=2)
        
        # Max pooling (unweighted, over thresholded hits) [B, N, D]
        encoded_for_max = torch.where(
            assignment_mask.unsqueeze(-1).expand_as(encoded_hit_features_expanded),
            encoded_hit_features_expanded,
            torch.tensor(float('-inf'), device=encoded_hit_features.device, dtype=encoded_hit_features.dtype)
        )
        encoded_max = encoded_for_max.max(dim=2)[0]  # [B, N, D]
        # Replace -inf with 0 for tracks with no assigned hits
        encoded_max = torch.where(torch.isinf(encoded_max), torch.zeros_like(encoded_max), encoded_max)
        
        # Min pooling (unweighted, over thresholded hits) [B, N, D]
        encoded_for_min = torch.where(
            assignment_mask.unsqueeze(-1).expand_as(encoded_hit_features_expanded),
            encoded_hit_features_expanded,
            torch.tensor(float('inf'), device=encoded_hit_features.device, dtype=encoded_hit_features.dtype)
        )
        encoded_min = encoded_for_min.min(dim=2)[0]  # [B, N, D]
        # Replace +inf with 0 for tracks with no assigned hits
        encoded_min = torch.where(torch.isinf(encoded_min), torch.zeros_like(encoded_min), encoded_min)
        
        # === Concatenate all features ===
        # query_embed: [B, N, D]
        # raw poolings: 4 x [B, N, D] = [B, N, 4*D]
        # encoded poolings: 4 x [B, N, D] = [B, N, 4*D]
        # Total: [B, N, 9*D]
        aggregated = torch.cat([
            query_embed,                # [B, N, D]
            raw_weighted_avg,          # [B, N, D]
            raw_weighted_sum,          # [B, N, D]
            raw_max,                   # [B, N, D]
            raw_min,                   # [B, N, D]
            encoded_weighted_avg,      # [B, N, D]
            encoded_weighted_sum,      # [B, N, D]
            encoded_max,               # [B, N, D]
            encoded_min,               # [B, N, D]
        ], dim=-1)
        
        return aggregated
    
    def forward(self, x: dict[str, Tensor], pads: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """Forward pass that aggregates hits and produces regression outputs.
        
        Args:
            x: Dictionary of input tensors
            pads: Optional padding information
            
        Returns:
            Dictionary with regression outputs
        """
        # Get aggregated features [B, N, 9*D]
        aggregated = self.latent(x, pads=pads)
        
        # Pass through multi-layer regression head [B, N, ndofs]
        regression_output = self.regression_head(aggregated)
        
        return {self.output_object + "_regr": regression_output}
    
    def predict(self, outputs):
        """Return predictions from model outputs.
        
        Args:
            outputs: Dictionary containing model outputs
            
        Returns:
            Dictionary with predictions
        """
        raw_output = outputs[self.output_object + "_regr"]  # [B, N, num_fields]
        
        predictions = {}
        for i, field in enumerate(self.fields):
            pred_value = raw_output[..., i]
            predictions[self.output_object + "_" + field] = pred_value
        
        return predictions
    
    def loss(self, outputs, targets):
        """Compute loss between outputs and targets.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Dictionary containing target values
            
        Returns:
            Dictionary with loss values
        """
        output = outputs[self.output_object + "_regr"]  # [B, N, num_fields]
        
        # Stack target fields
        target = torch.stack([targets[self.target_object + "_" + field] for field in self.fields], dim=-1)
        
        # Only compute loss for valid targets
        mask = targets[self.target_object + "_valid"].clone()
        target = target[mask]
        output = output[mask]
        
        # Compute the loss
        loss = torch.nn.functional.smooth_l1_loss(output, target, reduction="none")
        
        # Average over all the features
        loss = torch.mean(loss, dim=-1)
        
        # Compute the regression loss only for valid objects
        # Handle edge case where no valid objects exist (avoid NaN)
        num_valid = mask.sum()
        if num_valid > 0:
            return {"smooth_l1": self.loss_weight * loss.sum() / num_valid}
        else:
            # Return zero loss when no valid targets (gradient will be zero, no update)
            return {"smooth_l1": torch.tensor(0.0, device=output.device, dtype=output.dtype, requires_grad=True)}


class FrozenEncoderRegressionTask(WeightedPoolingObjectHitRegressionTask):
    """Regression task with frozen encoder loaded from checkpoint.
    
    This task loads a pre-trained model checkpoint and freezes all encoder/decoder
    weights, only training the regression head. This enables:
    - Fast iteration on regression architecture
    - Decoupled training from hit-to-track assignment
    - Lower memory usage (no gradients through transformer)
    - Easier hyperparameter tuning for regression
    
    The frozen model is used to generate hit-track assignments and encoded features,
    which are then used for regression exactly as in WeightedPoolingObjectHitRegressionTask.
    
    Usage:
        This task should be used as the ONLY task in a training run, with a checkpoint
        path pointing to a model trained with ObjectValidTask and ObjectHitMaskTask.
        
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file (.ckpt) containing the pre-trained model.
        The checkpoint should contain a model with encoder, decoder, and at minimum
        the ObjectHitMaskTask to generate hit-track assignment logits.
    freeze_all : bool
        If True (default), freezes all parameters from the checkpoint including
        input nets, encoder, decoder, and all tasks. If False, only freezes the
        encoder and decoder (allowing input nets to adapt).
    **kwargs : dict
        All other arguments are passed to WeightedPoolingObjectHitRegressionTask.
        
    Example Config:
        ```yaml
        tasks:
          - class_path: hepattn.models.task.FrozenEncoderRegressionTask
            init_args:
              name: regr
              checkpoint_path: logs/ckpts/epoch=004-val_loss=9.00107.ckpt
              freeze_all: true
              input_hit: hit
              input_object: query
              output_object: track
              target_object: truthMuon
              hit_fields: [...all 23 fields including eta...]
              fields:
                - truthMuon_eta
                - truthMuon_phi
                - truthMuon_pt_norm
                - truthMuon_charge
              loss_weight: 1.0
              dim: 32
              regression_hidden_dim: 144
              regression_num_layers: 3
              dropout: 0.1
        ```
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        freeze_all: bool = True,
        **kwargs
    ):
        """Initialize frozen encoder regression task.
        
        Args:
            checkpoint_path: Path to pre-trained model checkpoint
            freeze_all: Whether to freeze all model parameters (vs just encoder/decoder)
            **kwargs: Arguments passed to WeightedPoolingObjectHitRegressionTask
        """
        super().__init__(**kwargs)
        
        self.checkpoint_path = checkpoint_path
        self.freeze_all = freeze_all
        self.frozen_model = None  # Will be set during model initialization
        
    def setup_frozen_model(self, model):
        """Load and freeze the pre-trained model.
        
        This is called by the parent MaskFormer after all modules are initialized.
        We can't load the checkpoint in __init__ because we need access to the
        full model structure.
        
        Args:
            model: The parent MaskFormer model instance
        """
        import torch
        
        # Load the checkpoint
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Extract the state dict (handle both wrapped and unwrapped checkpoints)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (from LightningModule wrapper)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        # Load weights into the model
        # We only load the encoder, decoder, input_nets, and ObjectHitMaskTask
        # The regression head stays randomly initialized for training
        
        # Filter out the regression task weights (this task's weights)
        filtered_state_dict = {}
        regression_task_prefix = f"tasks.{self.name}."
        
        for key, value in state_dict.items():
            # Skip this task's parameters (we want to train these)
            if key.startswith(f"tasks."):
                # Check if this is our regression task
                task_key = key.split('.')[1] if len(key.split('.')) > 1 else None
                if task_key == self.name:
                    continue  # Skip regression task weights
            filtered_state_dict[key] = value
        
        # Load filtered weights
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"Loaded checkpoint with {len(filtered_state_dict)} parameters")
        print(f"Missing keys (expected for regression task): {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        
        # Freeze parameters
        if self.freeze_all:
            print("Freezing all parameters except regression head")
            for name, param in model.named_parameters():
                # Only train this task's regression head
                if not name.startswith(f"tasks.{self.name}."):
                    param.requires_grad = False
        else:
            print("Freezing only encoder and decoder")
            # Freeze encoder and decoder only
            if model.encoder is not None:
                for param in model.encoder.parameters():
                    param.requires_grad = False
            for decoder_layer in model.decoder_layers:
                for param in decoder_layer.parameters():
                    param.requires_grad = False
            
            # Also freeze other tasks (not this one)
            for task in model.tasks:
                if task.name != self.name:
                    for param in task.parameters():
                        param.requires_grad = False
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        self.frozen_model = model
