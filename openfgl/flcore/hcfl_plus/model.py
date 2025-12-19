# openfgl/flcore/hcfl_plus/model.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class HCFLPlusSplitModel(nn.Module):
    """
    ϕ / θ split for OpenFGL models:
      - encoder ϕ: all layers except last (self.layers[:-1])
      - heads θₖ: K copies of the last layer (self.layers[-1])
    
    Works with GCN, GAT, GraphSAGE, MLP, SGC, etc.
    """

    def __init__(self, base_model: nn.Module, num_heads: int):
        super().__init__()
        self.base_model = base_model
        self.num_heads = int(num_heads)

        # OpenFGL models use self.layers (ModuleList)
        # Last layer is the classifier: self.layers[-1]
        if not hasattr(base_model, "layers"):
            raise AttributeError(
                f"Base model has no 'layers' attribute. "
                f"OpenFGL models (GCN, GAT, GraphSAGE, etc.) should have self.layers."
            )
        
        if len(base_model.layers) == 0:
            raise ValueError("Base model has no layers.")

        # Extract classifier (last layer) as template for heads
        self.head_template = copy.deepcopy(base_model.layers[-1])
        
        # Don't modify base_model - we'll just use layers[:-1] in forward_embed
        # Create K cluster heads
        self.heads = nn.ModuleList([copy.deepcopy(self.head_template) for _ in range(self.num_heads)])

        # --- for soft clustering at inference ---
        self.default_head_idx = 0
        self.default_head_weights = None   # torch.Tensor shape [K] or None


    def resize_heads(self, new_k: int):
        """Resize heads to new_k, keeping existing heads when possible."""
        new_k = int(new_k)
        if new_k <= 0:
            new_k = 1
        if new_k == len(self.heads):
            return

        old_heads = list(self.heads)
        self.heads = nn.ModuleList()

        # keep as many as possible
        keep = min(len(old_heads), new_k)
        for i in range(keep):
            self.heads.append(old_heads[i])

        # add new ones if expanding
        for _ in range(new_k - keep):
            self.heads.append(copy.deepcopy(self.head_template))

        self.num_heads = new_k
        
    def forward_embed(self, data):
        """
        Returns embedding z from encoder ϕ (all layers except last).
        
        Args:
            data: torch_geometric.data.Data object
        
        Returns:
            z: embedding tensor (output of encoder)
        """
        x = data.x
        edge_index = data.edge_index if hasattr(data, 'edge_index') else None
        
        # Forward through encoder (all layers except last)
        for i, layer in enumerate(self.base_model.layers[:-1]):
            if edge_index is not None:
                # Graph layer (GCN, GAT, GraphSAGE, SGC)
                x = layer(x, edge_index)
            else:
                # Linear layer (MLP)
                x = layer(x)
            
            x = F.relu(x)
            #x = F.dropout(x, p=self.base_model.dropout, training=self.training)
            x = F.dropout(x, p=getattr(self.base_model, "dropout", 0.0), training=self.training)

        
        return x

    def forward_heads(self, z: torch.Tensor, data):
        """
        Return list of logits for each cluster head.
        
        Args:
            z: embedding tensor from encoder
            data: torch_geometric.data.Data (needed for edge_index if graph layer)
        
        Returns:
            list of logits tensors, one per cluster head
        """
        edge_index = data.edge_index if hasattr(data, 'edge_index') else None
        
        logits_list = []
        for head in self.heads:
            if edge_index is not None:
                # Graph layer
                logits = head(z, edge_index)
            else:
                # Linear layer
                logits = head(z)
            logits_list.append(logits)
        
        return logits_list
    
    def forward(self, data, head_idx: int = 0, return_all_heads: bool = False):
        """
        Default behavior (for OpenFGL tasks):
        - If return_all_heads=True: returns (z, [logits_k ...])
        - Else:
            * If self.default_head_weights is set -> returns (z, mixed_logits)
            * Else -> returns (z, logits_of_one_head) using:
                  provided head_idx OR self.default_head_idx
        """
        z = self.forward_embed(data)
        print("HCFLPlusSplitModel.forward called | soft=", self.default_head_weights is not None)


        # Return all heads if requested (training utilities etc.)
        if return_all_heads:
            logits_list = self.forward_heads(z, data)
            return z, logits_list

        logits_list = self.forward_heads(z, data)   # list of K tensors [N, C]
        K = len(logits_list)

        # -------------------------
        # ✅ SOFT mixture inference
        # -------------------------
        w = getattr(self, "default_head_weights", None)
        if w is not None:
            w = w.to(logits_list[0].device).float().view(-1)

            # pad/truncate safely
            if w.numel() != K:
                if w.numel() < K:
                    pad = torch.full((K - w.numel(),), 1e-6, device=w.device)
                    w = torch.cat([w, pad], dim=0)
                else:
                    w = w[:K]

            w = w / w.sum().clamp_min(1e-12)

            mixed = 0.0
            for k in range(K):
                mixed = mixed + w[k] * logits_list[k]
            return z, mixed

        # -------------------------
        # hard fallback (single head)
        # -------------------------
        idx = int(getattr(self, "default_head_idx", head_idx))
        idx = max(0, min(idx, K - 1))
        return z, logits_list[idx]

