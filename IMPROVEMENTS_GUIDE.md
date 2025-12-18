# Where to Add HCFL+ Improvements

## 1. µ̃ (mu_tilde) Regularization Loss

### File: `openfgl/flcore/hcfl_plus/clientv2.py`

### Where to add: In the `execute()` method, after line 38 (after loading cluster models)

**Current code (lines 19-40)**:
```python
def execute(self):
    server_msg = self.message_pool.get("server", None)

    if server_msg is not None:
        key = f"client_{self.client_id}"
        cluster_models = server_msg.get("cluster_models", None)
        alpha_map = server_msg.get("client_cluster_weights", {}).get(key, None)

        if cluster_models is not None and alpha_map:
            with torch.no_grad():
                for p in self.task.model.parameters():
                    p.zero_()

                for cid, alpha in alpha_map.items():
                    cid = int(cid)
                    if cid not in cluster_models:
                        continue
                    params_k = cluster_models[cid]
                    for p, pk in zip(self.task.model.parameters(), params_k):
                        p.add_(pk.to(self.device), alpha=float(alpha))

    self.task.train()  # <-- This is where we need to inject µ̃ loss
```

### What to add:

**Step 1**: Store target model (weighted combination) before training
```python
# After line 38, before self.task.train():
# Store target model for µ̃ penalty
self.target_model_params = None
if cluster_models is not None and alpha_map:
    # Compute weighted combination: θ_target = Σₖ ω̃ᵢₖ θₖ
    target_params = []
    for p in self.task.model.parameters():
        target_params.append(p.detach().clone())
    self.target_model_params = target_params
```

**Step 2**: Override loss function to include µ̃ penalty
```python
# Before self.task.train(), add:
mu_tilde = float(getattr(self.args, "hcfl_mu_tilde", 0.4))
if self.target_model_params is not None and mu_tilde > 0:
    original_loss_fn = self.task.loss_fn
    
    def hcfl_loss_fn(embedding, logits, labels, mask):
        # Base loss
        base_loss = original_loss_fn(embedding, logits, labels, mask)
        
        # µ̃ penalty: ||θ_client - θ_target||²
        penalty = 0.0
        for p_client, p_target in zip(self.task.model.parameters(), self.target_model_params):
            penalty += torch.nn.functional.mse_loss(p_client, p_target.to(self.device))
        
        return base_loss + mu_tilde * penalty
    
    self.task.loss_fn = hcfl_loss_fn
```

**Complete modified `execute()` method**:
```python
def execute(self):
    server_msg = self.message_pool.get("server", None)
    mu_tilde = float(getattr(self.args, "hcfl_mu_tilde", 0.4))

    if server_msg is not None:
        key = f"client_{self.client_id}"
        cluster_models = server_msg.get("cluster_models", None)
        alpha_map = server_msg.get("client_cluster_weights", {}).get(key, None)

        if cluster_models is not None and alpha_map:
            with torch.no_grad():
                for p in self.task.model.parameters():
                    p.zero_()

                for cid, alpha in alpha_map.items():
                    cid = int(cid)
                    if cid not in cluster_models:
                        continue
                    params_k = cluster_models[cid]
                    for p, pk in zip(self.task.model.parameters(), params_k):
                        p.add_(pk.to(self.device), alpha=float(alpha))
            
            # Store target model for µ̃ penalty
            self.target_model_params = [
                p.detach().clone() for p in self.task.model.parameters()
            ]
        else:
            self.target_model_params = None

    # Override loss function to include µ̃ penalty
    if self.target_model_params is not None and mu_tilde > 0:
        original_loss_fn = self.task.loss_fn
        
        def hcfl_loss_fn(embedding, logits, labels, mask):
            base_loss = original_loss_fn(embedding, logits, labels, mask)
            penalty = 0.0
            for p_client, p_target in zip(self.task.model.parameters(), self.target_model_params):
                penalty += torch.nn.functional.mse_loss(p_client, p_target.to(self.device))
            return base_loss + mu_tilde * penalty
        
        self.task.loss_fn = hcfl_loss_fn

    self.task.train()
```

---

## 2. ρ-based Dynamic Cluster Splitting

### File: `openfgl/flcore/hcfl_plus/serverv2.py`

### Where to add: In the `__init__()` method and `execute()` method

**Step 1**: Add cluster tracking in `__init__()` (after line 38)

```python
def __init__(self, args, global_data, data_dir, message_pool, device):
    super().__init__(args, global_data, data_dir, message_pool, device)
    self.num_clusters = int(getattr(args, "num_clusters", 1))
    self.soft_temp = float(getattr(args, "hcfl_temp", 1.0))
    self.top_k = int(getattr(args, "hcfl_top_k", 0))
    self.use_soft = bool(getattr(args, "hcfl_soft", True))
    
    # ADD THESE FOR ρ-BASED SPLITTING:
    self.rho = float(getattr(args, "hcfl_rho", 0.1))  # split threshold
    self.k0 = int(getattr(args, "hcfl_k0", 1))  # initial clusters
    self.max_k = int(getattr(args, "hcfl_max_k", 10))  # max clusters
    self.clusters = {0: list(range(args.num_clients))}  # cluster_id -> [client_ids]
    self.next_cluster_id = 1
```

**Step 2**: Modify `execute()` method to use dynamic splitting (replace lines 83-102)

**Current code (lines 83-102)**:
```python
# 2) cluster clients by prototypes
k = max(1, min(self.num_clusters, n_clients))
X = np.vstack(protos)

if k == 1:
    labels = np.zeros(n_clients, dtype=int)
else:
    try:
        clustering = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(X)
    except Exception:
        sim = cosine_similarity(X)
        dist = 1.0 - sim
        clustering = AgglomerativeClustering(
            n_clusters=k, affinity="precomputed", linkage="average"
        )
        labels = clustering.fit_predict(dist)
```

**Replace with dynamic splitting**:
```python
# 2) Dynamic cluster splitting based on ρ
# Map sampled client indices to their actual IDs
sampled_client_ids = [int(key.split("_")[1]) for key in client_keys]
X = np.vstack(protos)

# Update clusters: only consider sampled clients
new_clusters = {}
for cluster_id, client_list in self.clusters.items():
    # Get sampled clients in this cluster
    cluster_sampled = [i for i, cid in enumerate(sampled_client_ids) if cid in client_list]
    if len(cluster_sampled) < 2:
        # Too small to split, keep as is
        if cluster_id not in new_clusters:
            new_clusters[cluster_id] = []
        new_clusters[cluster_id].extend([sampled_client_ids[i] for i in cluster_sampled])
        continue
    
    # Compute heterogeneity (max pairwise distance in prototype space)
    cluster_protos = X[cluster_sampled]
    if len(cluster_protos) > 1:
        # Compute pairwise distances
        from sklearn.metrics.pairwise import pairwise_distances
        dist_matrix = pairwise_distances(cluster_protos, metric='cosine')
        heterogeneity = np.max(dist_matrix)  # or np.mean for mean distance
    
        # Split if heterogeneity > ρ
        if heterogeneity > self.rho and len(cluster_sampled) >= 2 and len(new_clusters) < self.max_k:
            # Split into 2 subclusters
            sub_clustering = AgglomerativeClustering(
                n_clusters=2, metric="cosine", linkage="average"
            )
            sub_labels = sub_clustering.fit_predict(cluster_protos)
            
            # Create two new clusters
            for sub_label in [0, 1]:
                sub_indices = [cluster_sampled[i] for i in range(len(cluster_sampled)) if sub_labels[i] == sub_label]
                if len(sub_indices) > 0:
                    new_cluster_id = self.next_cluster_id
                    self.next_cluster_id += 1
                    new_clusters[new_cluster_id] = [sampled_client_ids[i] for i in sub_indices]
        else:
            # Keep cluster as is
            if cluster_id not in new_clusters:
                new_clusters[cluster_id] = []
            new_clusters[cluster_id].extend([sampled_client_ids[i] for i in cluster_sampled])
    else:
        # Single client, keep as is
        if cluster_id not in new_clusters:
            new_clusters[cluster_id] = []
        new_clusters[cluster_id].extend([sampled_client_ids[i] for i in cluster_sampled])

# Update self.clusters
self.clusters = new_clusters
k = len(self.clusters)

# Assign labels based on new clusters
labels = np.zeros(n_clients, dtype=int)
for cluster_id, client_list in self.clusters.items():
    for i, cid in enumerate(sampled_client_ids):
        if cid in client_list:
            labels[i] = cluster_id
```

---

## 3. Add Hyperparameters to Config

### File: `openfgl/config.py`

### Where to add: After line 76 (after `num_clusters`)

```python
# Add after line 76:
parser.add_argument("--hcfl_rho", type=float, default=0.1, 
                   help="HCFL+ cluster split threshold (ρ)")
parser.add_argument("--hcfl_mu_tilde", type=float, default=0.4, 
                   help="HCFL+ regularization weight (µ̃)")
parser.add_argument("--hcfl_k0", type=int, default=1, 
                   help="HCFL+ initial number of clusters")
parser.add_argument("--hcfl_max_k", type=int, default=10, 
                   help="HCFL+ maximum number of clusters")
```

---

## Summary

| Improvement | File | Method | Lines |
|------------|------|--------|-------|
| **µ̃ loss** | `clientv2.py` | `execute()` | After line 38, before `self.task.train()` |
| **ρ splitting** | `serverv2.py` | `__init__()` | After line 38 |
| **ρ splitting** | `serverv2.py` | `execute()` | Replace lines 83-102 |
| **Hyperparameters** | `config.py` | (top level) | After line 76 |




