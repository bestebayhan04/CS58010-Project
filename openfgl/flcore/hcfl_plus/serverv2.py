import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from openfgl.flcore.base import BaseServer


def flatten_proto(proto):
    if isinstance(proto, torch.Tensor):
        return proto.detach().view(-1).cpu().numpy()
    return np.array(proto).reshape(-1)


def safe_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


class HCFLPlusServer(BaseServer):
    """
    HCFL(+):
      - Cluster clients by prototypes (Agglomerative)
      - Build cluster models by weighted avg (num_samples)
      - For each client compute soft weights over clusters via cosine+softmax
      - Send:
          cluster_models: {cluster_id: [tensor params]}
          client_cluster_weights: {"client_i": {cluster_id: alpha, ...}}
          client_cluster_map (hard labels, debug)
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)
        self.num_clusters = int(getattr(args, "num_clusters", 1))
        self.soft_temp = float(getattr(args, "hcfl_temp", 1.0))   # softmax temperature
        self.top_k = int(getattr(args, "hcfl_top_k", 0))          # 0 => use all clusters
        self.use_soft = bool(getattr(args, "hcfl_soft", True))    # True => HCFL+, False => hard HCFL
        
        """
        # ρ-based dynamic cluster splitting parameters
        self.rho = float(getattr(args, "hcfl_rho", 0.1))  # split threshold
        self.k0 = int(getattr(args, "hcfl_k0", 1))  # initial clusters
        self.max_k = int(getattr(args, "hcfl_max_k", 10))  # max clusters
        # Initialize with all clients in one cluster
        self.clusters = {0: list(range(args.num_clients))}  # cluster_id -> [client_ids]
        self.next_cluster_id = 1
        """

    def execute(self):
        print("Server started!!")
        sampled = self.message_pool.get("sampled_clients", [])
        if not sampled:
            return

        client_keys = []
        protos = []
        sizes = []
        weights = []

        # 1) collect client messages
        for cid in sampled:
            key = f"client_{cid}"
            cm = self.message_pool.get(key, None)
            if cm is None:
                continue

            proto = cm.get("prototype", None)
            if proto is None:
                # fallback: prototype from provided weights (shouldn't happen if client sends proto)
                w = cm["weight"]
                flat = []
                for p in w:
                    flat.append(p.detach().cpu().view(-1))
                if flat:
                    proto_vec = torch.cat(flat).numpy()
                else:
                    proto_vec = np.zeros((1,), dtype=np.float32)
                proto_vec = safe_l2_normalize(proto_vec)
            else:
                proto_vec = flatten_proto(proto)
                proto_vec = safe_l2_normalize(proto_vec.astype(np.float32))

            client_keys.append(key)
            protos.append(proto_vec)
            sizes.append(int(cm["num_samples"]))
            weights.append(cm["weight"])

        n_clients = len(client_keys)
        if n_clients == 0:
            return

        X = np.vstack(protos)

        # 2) fixed-K clustering
        k = max(1, min(self.num_clusters, n_clients))
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

        # 3) cluster models: weighted avg by num_samples
        cluster_sum = {}      # c -> list[tensor sum]
        cluster_count = {}    # c -> total samples

        for i in range(n_clients):
            c = int(labels[i])
            csize = float(sizes[i])
            w = weights[i]  # list[tensor] (cpu)

            if c not in cluster_sum:
                cluster_sum[c] = [p.detach().cpu() * csize for p in w]
                cluster_count[c] = csize
            else:
                for j, p in enumerate(w):
                    cluster_sum[c][j] += p.detach().cpu() * csize
                cluster_count[c] += csize

        cluster_models = {}
        for c in range(k):
            if cluster_count.get(c, 0.0) == 0.0:
                # fallback to current server model parameters
                cluster_models[c] = [p.detach().clone().to(self.device) for p in self.task.model.parameters()]
            else:
                total = cluster_count[c]
                cluster_models[c] = [(p / total).to(self.device) for p in cluster_sum[c]]

        # 4) cluster centroids (prototype space)
        cluster_centroids = []
        for c in range(k):
            idxs = [i for i in range(n_clients) if int(labels[i]) == c]
            if not idxs:
                centroid = np.zeros_like(protos[0], dtype=np.float32)
            else:
                centroid = np.mean(np.vstack([protos[i] for i in idxs]), axis=0).astype(np.float32)
                centroid = safe_l2_normalize(centroid)
            cluster_centroids.append(centroid)
        C = np.vstack(cluster_centroids)  # (k, d)

        # 5) client -> cluster soft weights
        client_cluster_weights = {}
        client_cluster_map = {client_keys[i]: int(labels[i]) for i in range(n_clients)}  # debug

        for i in range(n_clients):
            key = client_keys[i]

            if not self.use_soft:
                lab = int(labels[i])
                client_cluster_weights[key] = {lab: 1.0}
                continue

            sims = cosine_similarity(X[i:i+1], C).reshape(-1)  # (k,)
            sims = sims / max(self.soft_temp, 1e-8)

            exps = np.exp(sims - np.max(sims))
            probs = exps / (np.sum(exps) + 1e-12)

            if self.top_k and self.top_k < k:
                top_idx = np.argsort(-probs)[: self.top_k]
                mask = np.zeros_like(probs)
                mask[top_idx] = probs[top_idx]
                s = np.sum(mask)
                probs = mask / s if s > 0 else probs

            client_cluster_weights[key] = {int(c): float(probs[c]) for c in range(k) if probs[c] > 0.0}


        """
        sampled_client_ids = [int(key.split("_")[1]) for key in client_keys]

        # 2) ρ-based dynamic cluster splitting
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
                dist_matrix = pairwise_distances(cluster_protos, metric='cosine')
                heterogeneity = np.max(dist_matrix)  # max pairwise distance
                
                # Split if heterogeneity > ρ and we haven't reached max_k
                if heterogeneity > self.rho and len(cluster_sampled) >= 2 and len(new_clusters) < self.max_k - 1:
                    # Split into 2 subclusters
                    try:
                        sub_clustering = AgglomerativeClustering(
                            n_clusters=2, metric="cosine", linkage="average"
                        )
                        sub_labels = sub_clustering.fit_predict(cluster_protos)
                    except Exception:
                        # Fallback to precomputed distance
                        sim = cosine_similarity(cluster_protos)
                        dist = 1.0 - sim
                        sub_clustering = AgglomerativeClustering(
                            n_clusters=2, affinity="precomputed", linkage="average"
                        )
                        sub_labels = sub_clustering.fit_predict(dist)
                    
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
        
        # Handle clients not in any existing cluster (shouldn't happen, but safety check)
        all_assigned = set()
        for cluster_list in new_clusters.values():
            all_assigned.update(cluster_list)
        
        unassigned = [cid for cid in sampled_client_ids if cid not in all_assigned]
        if unassigned:
            # Assign unassigned clients to nearest cluster or create new one
            if len(new_clusters) == 0:
                # No clusters exist, create one
                new_clusters[0] = unassigned
            else:
                # Assign to first cluster (or could use nearest centroid)
                first_cluster_id = list(new_clusters.keys())[0]
                new_clusters[first_cluster_id].extend(unassigned)
        
        # Update self.clusters
        self.clusters = new_clusters
        k = len(self.clusters)
        
        # Assign labels based on new clusters
        labels = np.zeros(n_clients, dtype=int)
        for cluster_id, client_list in self.clusters.items():
            for i, cid in enumerate(sampled_client_ids):
                if cid in client_list:
                    labels[i] = cluster_id

        client_cluster_map = {client_keys[i]: int(labels[i]) for i in range(n_clients)}

        # 3) build hard cluster models (weighted avg by num_samples)
        cluster_sum = {}      # cid -> list[tensor]
        cluster_count = {}    # cid -> total_samples

        for i in range(n_clients):
            c = int(labels[i])
            csize = sizes[i]
            w = weights[i]

            if c not in cluster_sum:
                cluster_sum[c] = [p.detach().cpu() * float(csize) for p in w]
                cluster_count[c] = float(csize)
            else:
                for j, p in enumerate(w):
                    cluster_sum[c][j] += p.detach().cpu() * float(csize)
                cluster_count[c] += float(csize)

        cluster_models = {}
        for c in range(k):
            if cluster_count.get(c, 0.0) == 0.0:
                # fallback: server global
                cluster_models[c] = [p.detach().clone().to(self.device) for p in self.task.model.parameters()]
            else:
                total = cluster_count[c]
                cluster_models[c] = [(p / total).to(self.device) for p in cluster_sum[c]]

        # 4) build cluster centroids in prototype space (mean of client protos in cluster)
        cluster_centroids = []
        for c in range(k):
            idxs = [i for i in range(n_clients) if int(labels[i]) == c]
            if not idxs:
                centroid = np.zeros_like(protos[0], dtype=np.float32)
            else:
                centroid = np.mean(np.vstack([protos[i] for i in idxs]), axis=0).astype(np.float32)
                centroid = safe_l2_normalize(centroid)
            cluster_centroids.append(centroid)
        C = np.vstack(cluster_centroids)  # (k, d)

        # 5) compute client -> soft weights over clusters (HCFL+)
        client_cluster_weights = {}

        for i in range(n_clients):
            key = client_keys[i]
            if not self.use_soft:
                # hard HCFL
                lab = int(labels[i])
                client_cluster_weights[key] = {lab: 1.0}
                continue

            # cosine sim between client proto and each centroid
            sims = cosine_similarity(X[i:i+1], C).reshape(-1)  # (k,)
            sims = sims / max(self.soft_temp, 1e-8)

            # softmax
            exps = np.exp(sims - np.max(sims))
            probs = exps / (np.sum(exps) + 1e-12)

            # optionally keep only top-k clusters
            if self.top_k and self.top_k < k:
                top_idx = np.argsort(-probs)[: self.top_k]
                mask = np.zeros_like(probs)
                mask[top_idx] = probs[top_idx]
                s = np.sum(mask)
                if s > 0:
                    probs = mask / s

            client_cluster_weights[key] = {int(c): float(probs[c]) for c in range(k) if probs[c] > 0.0}
"""
        # 6) publish to message_pool
        # NOTE: keep cluster_models on device tensors; client copies to its device anyway
        self.message_pool["server"] = {
            "cluster_models": cluster_models,                 # {cid: [tensor params]}
            "client_cluster_weights": client_cluster_weights, # {"client_i": {cid: alpha}}
            "client_cluster_map": client_cluster_map,         # debug hard labels
            "k": k
        }

    def send_message(self):
        # execute() already writes to message_pool["server"]
        pass
