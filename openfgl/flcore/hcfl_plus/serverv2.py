import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
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
