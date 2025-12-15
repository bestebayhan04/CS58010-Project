import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from openfgl.flcore.base import BaseServer


def flatten_prototype(proto):
    # Ensure prototype is 1D numpy array
    if isinstance(proto, torch.Tensor):
        return proto.view(-1).cpu().numpy()
    return np.array(proto).reshape(-1)


class HCFLServer(BaseServer):
    """
    Simple HCFL Server that:
      - collects per-client prototypes and weights
      - computes similarities and clusters clients
      - aggregates weights inside each cluster (weighted average by num_samples)
      - sends weights_per_cluster and client->cluster mapping back to clients
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(HCFLServer, self).__init__(args, global_data, data_dir, message_pool, device)
        # config options expected in args:
        # args.num_clusters (int) : number of clusters to form (default fallback below)
        # args.cluster_threshold (float) : optional threshold (not implemented in this simple version)
        # If you want automatic cluster number selection, set num_clusters to None and we'll fallback to 1 (global)
        self.num_clusters = getattr(args, "num_clusters", None)

    def execute(self):
        """
        1) Read sampled clients from message_pool["sampled_clients"]
        2) Build prototype matrix from client messages
        3) Run clustering (Agglomerative)
        4) Aggregate weights per cluster (weighted average by num_samples)
        5) Store results in message_pool["server"]
        """
        

        sampled = self.message_pool.get("sampled_clients", [])
        if len(sampled) == 0:
            # nothing to do
            return

        protos = []
        client_keys = []
        sizes = []
        weights = []

        # gather client messages
        for client_id in sampled:
            key = f"client_{client_id}"
            cm = self.message_pool.get(key, None)
            if cm is None:
                continue
            client_keys.append(key)
            proto = cm.get("prototype", None)
            if proto is None:
                # fall back: create proto from model param vector mean (client provided weight exists)
                w = cm["weight"]
                # flatten weights to a single vector and normalize
                flat = []
                for p in w:
                    flat.append(p.detach().cpu().view(-1))
                if len(flat) == 0:
                    proto_vec = torch.zeros(1).numpy()
                else:
                    proto_vec = torch.cat(flat).numpy()
                    norm = np.linalg.norm(proto_vec)
                    if norm > 0:
                        proto_vec = proto_vec / norm
                protos.append(proto_vec)
            else:
                protos.append(flatten_prototype(proto))

            sizes.append(cm["num_samples"])
            weights.append(cm["weight"])

        n_clients = len(client_keys)
        if n_clients == 0:
            return

        # If num_clusters not set or > n_clients, set to n_clients (each client separate) or 1
        if self.num_clusters is None:
            k = 1
        else:
            k = min(max(1, int(self.num_clusters)), n_clients)

        # form feature matrix
        X = np.vstack([p for p in protos])

        # if k == 1, all clients assigned to single cluster
        if k == 1:
            labels = np.zeros(n_clients, dtype=int)
        else:
            # Agglomerative clustering with cosine affinity -> sklearn supports 'precomputed' affinity; convert to 1 - cosine
            # but AgglomerativeClustering deprecated affinity param in 1.2, still we use 'cosine' via metric param if available.
            try:
                clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
                labels = clustering.fit_predict(X)
            except Exception:
                # fallback: compute cosine similarity matrix and use 'precomputed' with AgglomerativeClustering
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity(X)
                # distance = 1 - sim
                clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
                labels = clustering.fit_predict(1.0 - sim)

        # Build mapping: client_key -> cluster_idx
        client_cluster_map = {client_keys[i]: int(labels[i]) for i in range(n_clients)}
        print("Clusters:", client_cluster_map)
        # Aggregate weights per cluster (weighted average by num_samples)
        clusters = {}
        cluster_sizes = {}
        for i, lab in enumerate(labels):
            lab = int(lab)
            if lab not in clusters:
                clusters[lab] = None  # will hold aggregated params (list of tensors)
                cluster_sizes[lab] = 0

            csize = sizes[i]
            cluster_sizes[lab] += csize
            client_weight = weights[i]

            # accumulate
            if clusters[lab] is None:
                # initialize with weighted params
                clusters[lab] = [ (p.detach().cpu() * (csize)) for p in client_weight ]
            else:
                # add weighted params
                for j, p in enumerate(client_weight):
                    clusters[lab][j] = clusters[lab][j] + (p.detach().cpu() * (csize))

        # finalize averaging
        weights_per_cluster = []
        for lab in range(k):
            if cluster_sizes.get(lab, 0) == 0:
                # if empty cluster, fall back to server model parameters (global)
                weights_per_cluster.append(list(self.task.model.parameters()))
            else:
                total = cluster_sizes[lab]
                avg_params = []
                accum = clusters[lab]
                for p in accum:
                    avg_params.append((p / float(total)).to(self.device))
                # note: avg_params are raw tensors on device; convert into Parameter-like objects by copying into model params when sending
                weights_per_cluster.append(avg_params)

        # put result in message_pool for clients to fetch
        self.message_pool["server"] = {
            "weights_per_cluster": weights_per_cluster,
            "client_cluster_map": client_cluster_map,
            # also include a convenience global averaged model (over all sampled clients)
            "weight": list(self.task.model.parameters())
        }

    def send_message(self):
        """
        Nothing special besides execute() writing to message_pool["server"].
        Kept for interface consistency.
        """
        # the execute() already writes the server message. Keep this method for API compatibility.
        pass
