# OpenFGL Model Structure Guide

## How OpenFGL Models Are Structured

### Standard Pattern (GCN, GAT, GraphSAGE, MLP, SGC)

All OpenFGL models follow this structure:

```python
class ModelName(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()  # All layers here
        
        # Encoder layers (all except last)
        if num_layers > 1:
            self.layers.append(LayerType(input_dim, hid_dim))
            for _ in range(num_layers - 2):
                self.layers.append(LayerType(hid_dim, hid_dim))
        
        # Classifier (last layer)
        self.layers.append(LayerType(hid_dim, output_dim))
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index  # for graph models
        
        # Encoder: all layers except last
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)  # or layer(x) for MLP
            x = F.relu(x)
            x = F.dropout(x, ...)
        
        # Classifier: last layer
        logits = self.layers[-1](x, edge_index)  # or self.layers[-1](x)
        
        return x, logits  # (embedding, logits)
```

### Key Points:

1. **No `classifier` attribute**: Models don't have `model.classifier`
2. **`self.layers` is a ModuleList**: Contains all layers
3. **Last layer is classifier**: `model.layers[-1]` is the classifier
4. **Encoder is all but last**: `model.layers[:-1]` is the encoder
5. **Forward returns `(embedding, logits)`**: Embedding from encoder, logits from classifier

---

## Examples

### GCN Model
```python
from openfgl.model.gcn import GCN

model = GCN(input_dim=1433, hid_dim=64, output_dim=7, num_layers=2)

# Structure:
# model.layers[0] = GCNConv(1433, 64)  # encoder layer
# model.layers[1] = GCNConv(64, 7)      # classifier

# Forward:
embedding, logits = model(data)
# embedding: output of layers[0] (64-dim)
# logits: output of layers[1] (7-dim)
```

### GAT Model
```python
from openfgl.model.gat import GAT

model = GAT(input_dim=1433, hid_dim=64, output_dim=7, num_layers=2)

# Structure:
# model.layers[0] = GATConv(1433, 64)  # encoder
# model.layers[1] = GATConv(64, 7)      # classifier
```

### MLP Model
```python
from openfgl.model.mlp import MLP

model = MLP(input_dim=1433, hid_dim=64, output_dim=7, num_layers=2)

# Structure:
# model.layers[0] = Linear(1433, 64)  # encoder
# model.layers[1] = Linear(64, 7)      # classifier
```

---

## How to Extract Classifier for HCFL+

### Method 1: Direct Access (Current Implementation)
```python
# Classifier is always the last layer
classifier = model.layers[-1]

# Encoder is all layers except last
encoder_layers = model.layers[:-1]
```

### Method 2: Using HCFLPlusSplitModel (Updated)
The updated `model.py` now correctly handles OpenFGL models:

```python
from openfgl.flcore.hcfl_plus.model import HCFLPlusSplitModel

# Wrap your OpenFGL model
base_model = GCN(input_dim=1433, hid_dim=64, output_dim=7)
split_model = HCFLPlusSplitModel(base_model, num_heads=3)

# Now you have:
# - split_model.base_model.layers[:-1] = encoder ϕ
# - split_model.heads = [θ₁, θ₂, θ₃] (3 cluster heads)

# Forward:
z, logits_list = split_model.forward(data, head_idx=None)  # all heads
z, logits = split_model.forward(data, head_idx=0)         # head 0 only
```

---

## For Your HCFL+ Implementation

### If you're doing encoder/head split:

**Option A: Use the wrapper** (recommended)
```python
# In client or server initialization
from openfgl.flcore.hcfl_plus.model import HCFLPlusSplitModel

# Wrap the task's model
split_model = HCFLPlusSplitModel(self.task.model, num_heads=args.num_clusters)
self.task.model = split_model
```

**Option B: Manual split** (if you need more control)
```python
# Encoder: all layers except last
encoder_layers = model.layers[:-1]

# Classifier: last layer
classifier = model.layers[-1]

# Create K heads
heads = [copy.deepcopy(classifier) for _ in range(K)]
```

---

## Summary

| Model Component | OpenFGL Structure | Access |
|----------------|-------------------|--------|
| **All layers** | `model.layers` | `model.layers` |
| **Encoder** | `model.layers[:-1]` | `model.layers[:-1]` |
| **Classifier** | `model.layers[-1]` | `model.layers[-1]` |
| **Forward output** | `(embedding, logits)` | `embedding, logits = model(data)` |

**Important**: OpenFGL models do NOT have a `classifier` attribute. Always use `model.layers[-1]` for the classifier.




