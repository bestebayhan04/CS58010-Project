# HCFL+ Implementation Roadmap for OpenFGL

## Current Status ✅
- ✅ Basic HCFL+ integration works (clientv2/serverv2)
- ✅ Prototype-based clustering
- ✅ Soft cluster assignments
- ✅ Cluster-wise aggregation
- ✅ Runs successfully on OpenFGL datasets

## What's Missing to Make it "Real" HCFL+ (Algorithm 2)

### 1. **Encoder ϕ vs Cluster Heads Θ Split** ❌
**Current**: Entire model is treated as one parameter vector  
**Needed**: 
- Shared encoder ϕ (feature extractor) - aggregated globally
- Per-cluster heads θₖ - aggregated per cluster
- Model architecture needs explicit split

### 2. **µ̃ (mu_tilde) Regularization** ❌
**Current**: Standard cross-entropy loss only  
**Needed**: 
- Add penalty term: `µ̃ * ||θᵢ - Σₖ ω̃ᵢₖ θₖ||²`
- Couples client's head to weighted combination of cluster heads

### 3. **ρ-based Dynamic Cluster Splitting** ❌
**Current**: Static K clusters (fixed `num_clusters`)  
**Needed**:
- Track cluster heterogeneity (distance matrix)
- Split cluster when heterogeneity > ρ threshold
- Start with K₀=1, grow adaptively

### 4. **Hyperparameters in Config** ❌
**Current**: Only `num_clusters` exposed  
**Needed**: Add to `config.py`:
- `--hcfl_rho` (split threshold, default 0.1)
- `--hcfl_mu_tilde` (regularization weight, default 0.4)
- `--hcfl_k0` (initial clusters, default 1)
- `--hcfl_max_k` (max clusters, default 10)

---

## Implementation Plan

### Phase 1: Add Hyperparameters (Quick Win)
**File**: `openfgl/config.py`
```python
# Add after line 76 (num_clusters)
parser.add_argument("--hcfl_rho", type=float, default=0.1, help="HCFL+ cluster split threshold")
parser.add_argument("--hcfl_mu_tilde", type=float, default=0.4, help="HCFL+ regularization weight")
parser.add_argument("--hcfl_k0", type=int, default=1, help="HCFL+ initial number of clusters")
parser.add_argument("--hcfl_max_k", type=int, default=10, help="HCFL+ maximum number of clusters")
```

### Phase 2: Model Architecture Split (Medium)
**Files**: 
- Create `openfgl/flcore/hcfl_plus/models.py` with encoder+head wrapper
- Modify `clientv2.py` to use split model

**Approach**:
1. Wrap existing GNN models to expose:
   - `encoder`: feature extractor (all layers except last)
   - `head`: final classifier layer
2. Server maintains:
   - One global encoder ϕ
   - K cluster heads {θ₁, ..., θₖ}

### Phase 3: Add µ̃ Loss (Medium)
**File**: `openfgl/flcore/hcfl_plus/clientv2.py`

**In `execute()` method**:
1. Override `self.task.loss_fn` to include µ̃ penalty
2. Compute target head: `θ_target = Σₖ ω̃ᵢₖ θₖ`
3. Add penalty: `loss = base_loss + µ̃ * MSE(θ_client, θ_target)`

### Phase 4: Dynamic Splitting (Hard)
**File**: `openfgl/flcore/hcfl_plus/serverv2.py`

**In `execute()` method**:
1. Track cluster history: `self.clusters = {0: [all_clients]}` initially
2. For each cluster:
   - Compute pairwise prototype distances
   - Compute heterogeneity H_c (max or mean distance)
   - If H_c > ρ and cluster size > 2: split into 2 subclusters
3. Rebuild cluster models after splitting

### Phase 5: Evaluation & Comparison (Critical)
**File**: `run_hcfl_with_fedavg_table7_data.py` (already exists)

**What to run**:
1. **Baseline methods** (already in OpenFGL):
   - FedAvg, FedDC, FedProto, FedTGP, FedSage+, FedGTA, Fed-PUB, FGSSL, FedGL, AdaFGL, FGGP, FedTAD

2. **HCFL+** (your implementation):
   - Run on same datasets: Cora, CiteSeer, PubMed, Photo, Computers, Chameleon, Actor
   - Use same client counts: 5, 10, 20, 30
   - Report `best_test_accuracy` (same as baselines)

3. **Generate comparison table**:
   - Use existing `tabulate_2.py` or create new script
   - Format: Dataset × Algorithm matrix
   - Include mean ± std over multiple runs

---

## Quick Start: Make it "Real" Step-by-Step

### Option A: Minimal Enhancement (Fastest)
**Goal**: Add µ̃ loss + ρ splitting to existing code

1. **Add hyperparameters** (5 min)
   - Edit `config.py` as in Phase 1

2. **Add µ̃ loss** (30 min)
   - Modify `clientv2.execute()` to compute penalty
   - Wrap `self.task.train()` to inject custom loss

3. **Add ρ splitting** (1 hour)
   - Modify `serverv2.execute()` to track clusters
   - Add split logic when heterogeneity > ρ

4. **Test & Compare** (2 hours)
   - Run on 1-2 datasets
   - Compare with FedAvg baseline

### Option B: Full Implementation (Most Accurate)
**Goal**: Complete Algorithm 2 implementation

1. **Model split** (2 hours)
   - Create encoder/head wrapper
   - Update client/server to handle split

2. **All Phase 1-4 steps** (4-6 hours total)

3. **Comprehensive evaluation** (1 day)
   - Run all datasets × all algorithms
   - Generate full comparison table

---

## Evaluation Checklist

To prove "extend OpenFGL to support HCFL+":

- [ ] HCFL+ runs on all OpenFGL-supported datasets
- [ ] Results table shows HCFL+ vs all baseline methods
- [ ] Code is integrated into OpenFGL (via `basic_utils.py`)
- [ ] Hyperparameters are configurable
- [ ] Documentation explains how to use HCFL+

---

## Files to Modify/Create

### Modify:
1. `openfgl/config.py` - Add HCFL+ hyperparameters
2. `openfgl/flcore/hcfl_plus/clientv2.py` - Add µ̃ loss
3. `openfgl/flcore/hcfl_plus/serverv2.py` - Add ρ splitting

### Create (optional):
4. `openfgl/flcore/hcfl_plus/models.py` - Encoder/head wrapper
5. `openfgl/flcore/hcfl_plus/README.md` - Usage documentation

---

## Next Steps

**Immediate** (to make it "real"):
1. Add µ̃ loss to client (most impactful)
2. Add ρ splitting to server (matches paper)
3. Run comparison experiment

**Later** (for full Algorithm 2):
4. Model architecture split
5. Comprehensive evaluation




