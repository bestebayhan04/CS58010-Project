# openfgl/flcore/hcfl_plus/server_improved.py
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from openfgl.flcore.base import BaseServer


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_norm(x, eps=1e-12):
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n

def _sym_ratio_from_col(col, eps=1e-12):
    """Symmetric ratio max(col_i/col_j, col_j/col_i) with eps-stability."""
    col = np.asarray(col, dtype=np.float32).reshape(-1)
    col = np.clip(col, eps, None)
    n = col.shape[0]
    R = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(col[i] / col[j])
            rsym = max(r, 1.0 / r)
            R[i, j] = rsym
            R[j, i] = rsym
    return R


def _tier4_distance_eq9(mode, base_max_dc_dlf, dc_mat, w_col, eps=1e-12):
    """
    Eq. (9) practical Tier-4 distance:
      - mode == "ASCP": max{ d_c, d_lf} * ratio 
      - mode == "CSCP": d_c * ratio
    where ratio approximated by ω̃ column for cluster k (and symmetrized).
    """
    R = _sym_ratio_from_col(w_col, eps=eps)

    mode = str(mode).upper()
    if mode == "CSCP":
        return dc_mat * R
    # default ASCP
    return base_max_dc_dlf * R


class HCFLPlusServer(BaseServer):
    """
    HCFL+ server (OpenFGL-integrated):
      - Global encoder aggregation (ϕ) via FedAvg
      - Cluster-head aggregation (θ_k)
      - Tier-4 distance uses:
          d_lf (mean embedding) + d_c (class prototypes) AND
          confidence ratio approximated by ω̃ (client-wise weights) (paper suggestion)
      - Membership: argmax ω̃ (Algorithm-2 style)
      - Adaptive split/remove uses Tier-4 distances
      - Broadcasts w_tilde_init hints when k changes (split/remove)
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super().__init__(args, global_data, data_dir, message_pool, device)
        self.num_clusters = int(getattr(args, "num_clusters", 1))
        self.dist_metric = getattr(args, "hcfl_dist_metric", "cosine")  # cosine or euclidean
        self.use_adaptive_k = bool(getattr(args, "hcfl_adaptive_k", False))
        self.rho = float(getattr(args, "hcfl_rho", 0.1))               # splitting threshold
        self.remove_patience = int(getattr(args, "hcfl_remove_patience", 0))
        self._cluster_inactive = {}  # cluster_id -> rounds w/o being top-1
        self.max_k = int(getattr(args, "hcfl_max_k", self.num_clusters))
        self.use_remove = bool(getattr(args, "hcfl_remove", False))
        self.hcfl_tier4_mode = str(getattr(args, "hcfl_tier4_mode", "ASCP")).upper()


        # ✅ cache maps / init hints to broadcast
        self._client_cluster_map = None
        self._w_tilde_init = None

        # keep latest model to broadcast in send_message (recommended)
        self._enc_global = None
        self._cluster_heads = None
        self._k = int(self.num_clusters)

        self._k_ops = None
        self._old_k_for_ops = None


    def execute(self):
        sampled = self.message_pool.get("sampled_clients", [])
        if not sampled:
            return

        # 1) collect messages
        keys = []
        sizes = []
        encs = []
        heads = []
        mean_zs = []
        class_protos = []
        w_tildes = []   # collect ω̃ from clients

        for cid in sampled:
            key = f"client_{cid}"
            msg = self.message_pool.get(key, None)
            if msg is None:
                continue

            keys.append(key)
            sizes.append(int(msg.get("num_samples", 0)))
            encs.append(msg["encoder_params"])
            heads.append(msg["cluster_heads"])

            mean_zs.append(_safe_norm(_to_np(msg.get("mean_z"))))
            class_protos.append({int(k): _safe_norm(_to_np(v)) for k, v in msg.get("class_protos", {}).items()})

            w = msg.get("w_tilde", None)
            if w is None:
                # fallback uniform if client didn't send it
                w = np.ones(self._k, dtype=np.float32) / max(self._k, 1)
            else:
                w = _to_np(w).astype(np.float32).reshape(-1)
                w = w / (w.sum() + 1e-12)
            w_tildes.append(w)

        n = len(keys)
        if n == 0:
            return

        # normalize sizes for FedAvg
        sizes_np = np.array(sizes, dtype=np.float32)
        s = float(sizes_np.sum())
        if s <= 0:
            sizes_np = np.ones_like(sizes_np) / max(len(sizes_np), 1)
        else:
            sizes_np = sizes_np / (s + 1e-12)

        # 2) build Tier-4 base distance matrix: max(d_c, d_lf)
        Z = np.vstack(mean_zs)  # [n, d]
        if self.dist_metric == "cosine":
            d_lf = 1.0 - cosine_similarity(Z)
        else:
            d_lf = pairwise_distances(Z, metric="euclidean")

        d_c = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                shared = set(class_protos[i].keys()) & set(class_protos[j].keys())
                if not shared:
                    val = 1.0
                else:
                    vals = []
                    for c in shared:
                        pi = class_protos[i][c]
                        pj = class_protos[j][c]
                        if self.dist_metric == "cosine":
                            vals.append(1.0 - float(np.dot(pi, pj)))
                        else:
                            vals.append(float(np.linalg.norm(pi - pj)))
                    val = float(np.max(vals)) if vals else 1.0
                d_c[i, j] = val
                d_c[j, i] = val

        base_max = np.maximum(d_lf, d_c)  # this is max{d_c, d_lf} used in ASCP
        eps = 1e-12


        # ----------------------------
        # 3) membership + adaptive split/remove
        # ----------------------------
        # start from previously active k (persist adaptive-k across rounds)
        k0 = int(max(1, min(getattr(self, "_k", self.num_clusters), n)))
        self._old_k_for_ops = k0
        k_ops = []   # <- record operations in order
        self._k = k0


        # Build W_full with shape [n, k0] (pad/truncate safely)
        W_full = []
        for w in w_tildes:
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            if w.shape[0] < k0:
                pad = np.full((k0 - w.shape[0],), 1e-6, dtype=np.float32)
                w = np.concatenate([w, pad], axis=0)

            else:
                w = w[:k0]
            w = w / (w.sum() + 1e-12)
            W_full.append(w)
        W_full = np.vstack(W_full)  # [n, k0]
        w_cols = [W_full[:, c].copy() for c in range(W_full.shape[1])]

        # Algorithm-2 membership: argmax ω̃
        labels = np.argmax(W_full, axis=1).astype(int)

        k = k0

        # ---- Adaptive-K splitting (ρ-based) ----
        if self.use_adaptive_k:
            max_k = int(max(1, min(self.max_k, n)))

            changed = True
            while changed and k < max_k:
                changed = False

                split_candidates = []
                for c in range(k):
                    idxs = np.where(labels == c)[0]
                    if len(idxs) <= 2:
                        continue
                    Dk_full = _tier4_distance_eq9(
                        mode=self.hcfl_tier4_mode,
                        base_max_dc_dlf=base_max,
                        dc_mat=d_c,
                        w_col=w_cols[c],
                        eps=eps,
                    )
                    Dc = Dk_full[np.ix_(idxs, idxs)]
                    vals = Dc[np.triu_indices_from(Dc, k=1)]
                    if len(vals) == 0:
                        continue
                    spread = float(np.max(vals) - np.mean(vals))
                    if spread >= self.rho:
                        split_candidates.append((spread, c))

                if not split_candidates:
                    break

                split_candidates.sort(reverse=True)
                _, split_c = split_candidates[0]
                idxs = np.where(labels == split_c)[0]

                Dk_full = _tier4_distance_eq9(
                    mode=self.hcfl_tier4_mode,
                    base_max_dc_dlf=base_max,
                    dc_mat=d_c,
                    w_col=w_cols[split_c],
                    eps=eps,
                )
                Dc = Dk_full[np.ix_(idxs, idxs)]
                try:
                    sub = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average")
                except TypeError:
                    sub = AgglomerativeClustering(n_clusters=2, affinity="precomputed", linkage="average")
                sub_labels = sub.fit_predict(Dc)

                new_cluster_id = k
                k_ops.append({"op": "split", "src": int(split_c), "dst": int(new_cluster_id)})

                # add a ω̃ column for the new cluster (same as parent for ratio computations)
                w_cols.append(w_cols[int(split_c)].copy())

                for local_pos, global_idx in enumerate(idxs):
                    if sub_labels[local_pos] == 1:
                        labels[global_idx] = new_cluster_id

                k += 1
                changed = True

        # ---- Cluster removal (patience, based on argmax ω̃ labels) ----
        if self.use_remove and self.remove_patience > 0:
            present = set(labels.tolist())
            for cid in range(k):
                if cid not in present:
                    self._cluster_inactive[cid] = self._cluster_inactive.get(cid, 0) + 1
                else:
                    self._cluster_inactive[cid] = 0

            remove_ids = [cid for cid, cnt in self._cluster_inactive.items() if cnt >= self.remove_patience]

            if remove_ids:
                keep_ids = [cid for cid in range(k) if cid not in remove_ids]
                if keep_ids:
                    for rid in remove_ids:
                        idxs = np.where(labels == rid)[0]
                        if len(idxs) == 0:
                            continue
                        for i in idxs:
                            best = None
                            best_val = None
                            for kc in keep_ids:
                                members = np.where(labels == kc)[0]
                                if len(members) == 0:
                                    continue
                                Dk_full = _tier4_distance_eq9(
                                    mode=self.hcfl_tier4_mode,
                                    base_max_dc_dlf=base_max,
                                    dc_mat=d_c,
                                    w_col=w_cols[kc],
                                    eps=eps,
                                )
                                val = float(np.mean(Dk_full[i, members]))

                                if best_val is None or val < best_val:
                                    best_val = val
                                    best = kc
                            if best is not None:
                                labels[i] = best

                unique = sorted(set(int(x) for x in labels.tolist()))
                remap = {old: new for new, old in enumerate(unique)}
                k_ops.append({
                    "op": "remove",
                    "removed": [int(r) for r in remove_ids],
                    "remap": {int(old): int(new) for old, new in remap.items()}
                })

                labels = np.array([remap[int(x)] for x in labels], dtype=int)
                k = len(unique)
                self._cluster_inactive = {i: 0 for i in range(k)}

                # ✅ remap w_cols to new cluster indexing
                new_w_cols = [None] * k
                for old_id, new_id in remap.items():
                    old_id = int(old_id)
                    new_id = int(new_id)
                    if 0 <= old_id < len(w_cols):
                        new_w_cols[new_id] = w_cols[old_id]

                # fill any None safely
                for idx in range(k):
                    if new_w_cols[idx] is None:
                        new_w_cols[idx] = np.full((n,), 1.0 / max(k, 1), dtype=np.float32)

                w_cols = new_w_cols


        # ----------------------------
        # Finalize k + build ω̃ init hints + cache maps
        # ----------------------------
        self._k = int(k)
        self._k_ops = k_ops if (self._k != self._old_k_for_ops) else None

        # Build w_tilde_init from w_cols (already updated for split/remove/remap)
        # w_cols: list length = self._k, each entry shape = [n]
        w_tilde_init = {}
        for i, key in enumerate(keys):
            vec = np.array([float(w_cols[c][i]) for c in range(self._k)], dtype=np.float32)
            s = float(vec.sum())
            if s <= 1e-12:
                vec[:] = 1.0 / max(self._k, 1)
            else:
                vec /= (s + 1e-12)
            w_tilde_init[key] = vec


        self._client_cluster_map = {keys[i]: int(labels[i]) for i in range(n)}
        self._w_tilde_init = w_tilde_init

        # 4) aggregate encoder (ϕ): FedAvg across clients
        enc_global = []
        for p_i in range(len(encs[0])):
            acc = None
            for i in range(n):
                w = sizes_np[i]
                t = encs[i][p_i].detach().cpu().float()
                acc = t * w if acc is None else acc + t * w
            enc_global.append(acc.to(self.device))

        # 5) aggregate K heads (θ_k) based on cluster assignments
        cluster_heads = {}
        for c in range(self._k):
            idxs = [i for i in range(n) if int(labels[i]) == c]
            if not idxs:
                continue

            # pick a template for sizing (any existing head params)
            any_head = heads[idxs[0]].get(0) or list(heads[idxs[0]].values())[0]
            num_params = len(any_head)

            agg = [None] * num_params
            tot = 0.0

            for i in idxs:
                hparams = heads[i].get(int(c), None)
                if hparams is None:
                    hparams = heads[i].get(str(c), None)
                if hparams is None:
                    continue  # skip this client

                w = float(sizes[i])
                tot += w
                for pi in range(num_params):
                    t = hparams[pi].detach().cpu().float() * w
                    agg[pi] = t if agg[pi] is None else agg[pi] + t

            if tot <= 0.0 or any(a is None for a in agg):
                # nobody contributed -> initialize safely from template
                cluster_heads[str(c)] = [p.detach().cpu().float().to(self.device) for p in any_head]
            else:
                for pi in range(num_params):
                    agg[pi] = (agg[pi] / tot).to(self.device)
                cluster_heads[str(c)] = agg


        # cache to broadcast in send_message
        self._enc_global = enc_global
        self._cluster_heads = cluster_heads

        # 6) publish message to clients
        self.message_pool["server"] = {
            "encoder_params": self._enc_global,
            "cluster_heads": self._cluster_heads,
            "client_cluster_map": self._client_cluster_map,
            "k": int(self._k),
            "round": int(self.message_pool.get("round", 0)),
            "w_tilde_init": self._w_tilde_init,
            "old_k": int(self._old_k_for_ops) if self._old_k_for_ops is not None else None,
            "k_ops": self._k_ops,

        }

    def send_message(self):
        """
        Trainer calls this BEFORE clients execute each round.
        So we must broadcast the latest cached server params.
        """
        # If first round and nothing cached yet, fall back to server.task.model params.
        if self._enc_global is None or self._cluster_heads is None:
            enc_layers = list(self.task.model.layers[:-1])
            enc_params = []
            for layer in enc_layers:
                enc_params.extend(list(layer.parameters()))
            self._enc_global = [p.detach().cpu().clone().to(self.device) for p in enc_params]

            last = self.task.model.layers[-1]
            k = int(getattr(self.args, "num_clusters", 1))
            self._cluster_heads = {
                str(i): [p.detach().cpu().clone().to(self.device) for p in last.parameters()]
                for i in range(k)
            }
            self._k = k

        self.message_pool["server"] = {
            "encoder_params": self._enc_global,
            "cluster_heads": self._cluster_heads,
            "client_cluster_map": self._client_cluster_map,
            "k": int(self._k),
            "round": int(self.message_pool.get("round", 0)),
            "old_k": int(self._old_k_for_ops) if self._old_k_for_ops is not None else None,
            "k_ops": self._k_ops,
            "w_tilde_init": self._w_tilde_init,  # optional
        }


