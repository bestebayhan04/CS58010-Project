import torch
import torch.nn.functional as F
from openfgl.flcore.base import BaseClient
from .model import HCFLPlusSplitModel

def _apply_k_ops_to_omega_matrix(omega: torch.Tensor, old_k: int, k_ops, device, eps=1e-12):
    """
    Applies Algorithms 4 & 5 to sample-wise ω matrix.
    omega: [N, old_k] (or current K while applying sequentially)
    Returns: transformed omega with new K and row-normalized.
    """
    if omega is None:
        return None

    omega = omega.to(device).float()
    if omega.dim() != 2:
        return None

    # fix col mismatch to old_k safely
    if omega.size(1) != old_k:
        if omega.size(1) < old_k:
            pad = torch.full((omega.size(0), old_k - omega.size(1)), 1e-6, device=device)
            omega = torch.cat([omega, pad], dim=1)
        else:
            omega = omega[:, :old_k]
        omega = omega / omega.sum(dim=1, keepdim=True).clamp_min(eps)

    if not k_ops:
        return omega / omega.sum(dim=1, keepdim=True).clamp_min(eps)

    for op in k_ops:
        if op.get("op") == "split":
            src = int(op["src"])
            dst = int(op["dst"])
            cur_k = omega.size(1)

            new_k = max(cur_k + 1, dst + 1)
            new_omega = torch.zeros((omega.size(0), new_k), device=device)
            new_omega[:, :cur_k] = omega

            wsrc = new_omega[:, src].clone()
            new_omega[:, src] = 0.5 * wsrc
            new_omega[:, dst] = 0.5 * wsrc

            omega = new_omega

        elif op.get("op") == "remove":
            remap = op.get("remap", {})
            if not remap:
                omega = omega / omega.sum(dim=1, keepdim=True).clamp_min(eps)
                continue

            cur = omega
            new_k = max(remap.values()) + 1
            new_omega = torch.zeros((cur.size(0), new_k), device=device)

            for old_id, new_id in remap.items():
                old_id = int(old_id)
                new_id = int(new_id)
                if 0 <= old_id < cur.size(1):
                    new_omega[:, new_id] = cur[:, old_id]

            row_sum = new_omega.sum(dim=1, keepdim=True)
            # if any row becomes empty, replace with uniform
            empty = row_sum.squeeze(1) <= eps
            if empty.any():
                new_omega[empty] = 1.0 / max(new_k, 1)
                row_sum = new_omega.sum(dim=1, keepdim=True)

            omega = new_omega / row_sum.clamp_min(eps)

    return omega / omega.sum(dim=1, keepdim=True).clamp_min(eps)


def _apply_k_ops_to_weight(vec: torch.Tensor, old_k: int, k_ops, device, eps=1e-12):
    """
    Applies HCFL+ Algorithms 4 & 5:
      - split: ω[src] -> ω[src]/2 and ω[dst] = ω[src]/2
      - remove: drop removed, then remap remaining, then renormalize
    vec: shape [old_k] (or current K while applying sequentially)
    """
    if vec is None:
        vec = torch.full((old_k,), 1.0 / max(old_k, 1), device=device)

    vec = vec.to(device).float().view(-1)

    # If vec length mismatches old_k, fix safely
    if vec.numel() != old_k:
        if vec.numel() < old_k:
            pad = torch.full((old_k - vec.numel(),), 1e-6, device=device)
            vec = torch.cat([vec, pad], dim=0)
        else:
            vec = vec[:old_k]
        vec = vec / (vec.sum().clamp_min(eps))

    if not k_ops:
        return vec / vec.sum().clamp_min(eps)

    for op in k_ops:
        if op.get("op") == "split":
            src = int(op["src"])
            dst = int(op["dst"])
            cur_k = vec.numel()

            new_k = max(cur_k + 1, dst + 1)
            new_vec = torch.zeros((new_k,), device=device)
            new_vec[:cur_k] = vec

            wsrc = new_vec[src].clone()
            new_vec[src] = 0.5 * wsrc
            new_vec[dst] = 0.5 * wsrc

            vec = new_vec


        elif op.get("op") == "remove":
            remap = op.get("remap", {})  # old_id -> new_id
            if not remap:
                # if no remap provided, just renormalize current
                vec = vec / vec.sum().clamp_min(eps)
                continue

            cur = vec
            new_k = max(remap.values()) + 1
            new_vec = torch.zeros((new_k,), device=device)

            for old_id, new_id in remap.items():
                old_id = int(old_id)
                new_id = int(new_id)
                if 0 <= old_id < cur.numel():
                    new_vec[new_id] = cur[old_id]

            s = new_vec.sum()
            if float(s.item()) <= eps:
                new_vec = torch.full((new_k,), 1.0 / max(new_k, 1), device=device)
            else:
                new_vec = new_vec / s

            vec = new_vec

    return vec / vec.sum().clamp_min(eps)


def _safe_softmax(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True).values
    return torch.softmax(x, dim=dim)


class HCFLPlusClient(BaseClient):
    """
    HCFL+ client (OpenFGL-integrated):
      - Receives encoder + cluster heads from server
      - Local training with SCWU (sample-wise head weights)
      - Optional mu_tilde KL regularization tying sample-wise weights to client-wise weights
      - Sends back encoder + heads + Tier-4 stats (mean_z + class prototypes)
    """

    def _ensure_split_model(self, k: int):
        if not isinstance(self.task.model, HCFLPlusSplitModel):
            self.task.model = HCFLPlusSplitModel(self.task.model, num_heads=k).to(self.device)
        else:
            self.task.model.resize_heads(k)

    """def _get_annealed_temp(self, round_idx: int):
        temp0 = float(getattr(self.args, "hcfl_scwu_temp", 1.0))
        if not bool(getattr(self.args, "hcfl_temp_anneal", False)):
            return temp0

        rate = float(getattr(self.args, "hcfl_temp_anneal_rate", 0.98))
        temp_min = float(getattr(self.args, "hcfl_temp_min", 0.1))
        t = temp0 * (rate ** max(int(round_idx), 0))
        return max(temp_min, t)"""

    def execute(self):
        # -----------------------------------
        # 0) read server msg early (k + round)
        # -----------------------------------
        server_msg = self.message_pool.get("server", None)

        k = int(getattr(self.args, "num_clusters", 1))
        # trainer already writes message_pool["round"]
        round_idx = int(self.message_pool.get("round", 0))

        if server_msg is not None:
            k = int(server_msg.get("k", k))

        self._ensure_split_model(k)

        # ----------------------------------------------------
        # Apply ω̃ init hint from server (split/remove handling)
        # ----------------------------------------------------
        prev_k = getattr(self, "_prev_k", None)

        if server_msg is not None and (prev_k is None or prev_k != k):
            # old_k can be None in round 0 -> fall back to current k
            old_k = server_msg.get("old_k", None)
            if old_k is None:
                old_k = prev_k if prev_k is not None else k
            old_k = int(old_k)

            k_ops = server_msg.get("k_ops", None)


            # ---- ω̃ (client-wise) transform (already correct) ----
            omega_tilde_prev = getattr(self, "_omega_tilde_prev", None)
            if omega_tilde_prev is None:
                omega_tilde_prev = torch.full((int(old_k),), 1.0 / max(int(old_k), 1), device=self.device)

            omega_tilde_new = _apply_k_ops_to_weight(
                omega_tilde_prev,
                old_k=int(old_k),
                k_ops=k_ops,
                device=self.device
            )

            self._omega_tilde_prev = omega_tilde_new.detach()
            self._omega_tilde_last = omega_tilde_new.detach().cpu()

            # ---- ✅ NEW: ω (sample-wise) transform (Algorithms 4 & 5) ----
            omega_prev_round = getattr(self, "_omega_prev_round", None)
            if omega_prev_round is not None:
                omega_prev_round = omega_prev_round.to(self.device).float()
                omega_prev_round = _apply_k_ops_to_omega_matrix(
                    omega_prev_round,
                    old_k=int(old_k),
                    k_ops=k_ops,
                    device=self.device
                )
                # store back (keep on device for now; we'll validate shape later when L exists)
                self._omega_prev_round = omega_prev_round.detach()


        self._prev_k = k
   
        # -----------------
        # Hyperparameters
        # -----------------

        # HCFL+ weight update knobs
        mu = float(getattr(self.args, "hcfl_mu", 0.0))                 # µ in paper (coupling strength)
        mu_tilde = float(getattr(self.args, "hcfl_mu_tilde", 0.0))     # optional additional blending
        top_k = int(getattr(self.args, "hcfl_top_k", 0))               # OPTIONAL: your extra (can keep 0)

        # -----------------------------
        # 1) sync from server (params)
        # -----------------------------
        if server_msg is not None:
            enc = server_msg.get("encoder_params", None)
            heads = server_msg.get("cluster_heads", None)  # dict: {k: [params]}

            # encoder: only layers[:-1]
            if enc is not None:
                with torch.no_grad():
                    enc_layers = list(self.task.model.base_model.layers[:-1])
                    enc_params = []
                    for layer in enc_layers:
                        enc_params.extend(list(layer.parameters()))
                    for p, pk in zip(enc_params, enc):
                        p.copy_(pk.to(self.device))

            # heads
            if heads is not None:
                with torch.no_grad():
                    for k_str, head_params in heads.items():
                        hid = int(k_str)
                        if hid < 0 or hid >= len(self.task.model.heads):
                            continue
                        for p, pk in zip(self.task.model.heads[hid].parameters(), head_params):
                            p.copy_(pk.to(self.device))

            # OPTIONAL but recommended: set which head to use for eval by default
            # (requires server to include client_cluster_map)
            cmap = server_msg.get("client_cluster_map", None)
            if isinstance(cmap, dict):
                my_key = f"client_{self.client_id}"
                if my_key in cmap:
                    self.task.model.default_head_idx = int(cmap[my_key])

        # ------------------------------------
        # 2) local training (OpenFGL Task API)
        # ------------------------------------
        optimizer = getattr(self.task, "optim", None) or getattr(self.task, "optimizer", None)
        if optimizer is None:
            raise RuntimeError("HCFLPlusClient needs task.optim or task.optimizer (OpenFGL optimizer).")

        processed_data = getattr(self.task, "processed_data", None)
        if processed_data is None:
            data = getattr(self.task, "data", None)
            if data is None:
                raise RuntimeError("HCFLPlusClient needs task.processed_data or task.data.")
        else:
            data = processed_data.get("data", None) if isinstance(processed_data, dict) else None
            if data is None:
                data = getattr(processed_data, "data", None)
            if data is None:
                raise RuntimeError("HCFLPlusClient: processed_data is present but missing 'data'.")

        data = data.to(self.device)

        y = getattr(data, "y", None)
        if y is None:
            raise RuntimeError("HCFLPlusClient: missing labels (data.y).")
        y = y.to(self.device)

        # train mask
        mask = None
        if isinstance(processed_data, dict) and "train_mask" in processed_data:
            mask = processed_data["train_mask"]
        elif hasattr(data, "train_mask"):
            mask = data.train_mask
        if mask is not None:
            mask = mask.to(self.device).bool()

        num_epochs = int(getattr(self.args, "local_epochs", getattr(self.args, "num_epochs", 1)))

        self.task.model.train()
        eps = 1e-12

        omega = None
        for _ in range(num_epochs):
            optimizer.zero_grad()

            z = self.task.model.forward_embed(data)              # [N, D]
            logits_list = self.task.model.forward_heads(z, data) # K * [N, C]

            # -----------------------------------------
            # Paper-faithful: use log-likelihoods/log-probs
            # logp_{j,k} = log P_{theta_k}(y_j | x_j)  (up to constant)
            # -----------------------------------------
            per_head_logp = []
            for logits in logits_list:
                ce = F.cross_entropy(logits, y, reduction="none")      # [N]
                if mask is not None:
                    ce = ce[mask]                                     # [N_used]
                logp = -ce                                            # [N_used]  (log-likelihood-like)
                per_head_logp.append(logp)

            # logP: [N_used, K]
            logP = torch.stack(per_head_logp, dim=1)

            if logP.shape[0] == 0:
                continue

            # For ω/ω̃ update formulas we need positive "L_k".
            # Use stabilized likelihoods: L = exp(logP - max_logP) to avoid underflow.
            logP_max = logP.max(dim=1, keepdim=True).values
            L = torch.exp(logP - logP_max)                            # [N_used, K], positive & stable



            # OPTIONAL: top-k head sparsification on weights (your extra)
            # We'll apply it later on omega_new. Keep top_k=0 to disable.

            # ============================================================
            # ✅ HCFL+ ω / ω̃ update + objective (drop-in replacement)
            # ============================================================

            N_eff = float(L.shape[0])

            # initialize ω̃ (client-wise) (persist across rounds if present)
            omega_tilde = getattr(self, "_omega_tilde_prev", None)
            if omega_tilde is None or omega_tilde.numel() != L.shape[1]:
                omega_tilde = torch.full((L.shape[1],), 1.0 / L.shape[1], device=self.device)
            else:
                omega_tilde = omega_tilde.to(self.device)

            # initialize or reuse ω (sample-wise) from previous local epoch
            # initialize or reuse ω (sample-wise)
            if omega is None or omega.shape != L.shape:
                # ✅ prefer transformed previous-round ω if it matches current N_used and K
                omega_prev_round = getattr(self, "_omega_prev_round", None)
                if omega_prev_round is not None:
                    omega_prev_round = omega_prev_round.to(self.device).float()
                    if omega_prev_round.shape == L.shape:
                        omega = omega_prev_round
                    else:
                        # shape mismatch -> fall back to ω̃
                        omega = omega_tilde.unsqueeze(0).repeat(L.shape[0], 1)
                else:
                    omega = omega_tilde.unsqueeze(0).repeat(L.shape[0], 1)
            else:
                omega = omega.to(self.device)



            # denom_j = Σ_k ω_{j,k} * L_{j,k}
            denom = (omega * L).sum(dim=1, keepdim=True).clamp_min(eps)

            # γ = ω*L / denom
            gamma = (omega * L) / denom

            # γ̃ = ω̃*L / denom
            gamma_tilde = (omega_tilde.unsqueeze(0) * L) / denom

            # ω̃^{t+1} = mean_j γ̃_{j,k}
            omega_tilde_new = gamma_tilde.mean(dim=0)
            omega_tilde_new = omega_tilde_new / omega_tilde_new.sum().clamp_min(eps)

            # ω^{t+1} = γ/(1+µN) + (µN)/(1+µN) * ω̃
            if mu > 0:
                alpha = 1.0 / (1.0 + mu * N_eff)
                beta = 1.0 - alpha
            else:
                alpha = 1.0
                beta = 0.0

            omega_new = alpha * gamma + beta * omega_tilde_new.unsqueeze(0)

            # optional extra blending (keep 0.0 to disable)
            """if mu_tilde > 0:
                omega_new = mu_tilde * omega_new + (1.0 - mu_tilde) * omega_tilde_new.unsqueeze(0)"""

            # normalize rows
            omega_new = omega_new / omega_new.sum(dim=1, keepdim=True).clamp_min(eps)

            # OPTIONAL: apply your top-k sparsification to ω (not paper)
            if top_k and top_k < omega_new.shape[1]:
                vals, idx = torch.topk(omega_new, k=top_k, dim=1)
                W_sparse = torch.zeros_like(omega_new)
                W_sparse.scatter_(1, idx, vals)
                omega_new = W_sparse / (W_sparse.sum(dim=1, keepdim=True) + eps)

            # carry ω forward to next local epoch
            omega = omega_new.detach()

            # Paper objective: maximize mean_j log( sum_k ω_{j,k} * exp(logp_{j,k}) )
            # Implement as logsumexp(log ω + logp) for stability.
            logw = torch.log(omega_new.clamp_min(eps))                # [N_used, K]
            log_mix = torch.logsumexp(logw + logP, dim=1)             # [N_used]
            loss_main = -log_mix.mean()                               # minimize negative (maximize objective)


            # OPTIONAL improvement (if you want): KL(ω || ω̃) regularizer
            # If you want a paper-faithful baseline, set hcfl_mu_tilde = 0.0
            loss_kl = torch.tensor(0.0, device=self.device)
            if mu_tilde > 0:
                wcur = omega_new.clamp_min(eps)
                wref = omega_tilde_new.unsqueeze(0).expand_as(wcur).clamp_min(eps)
                loss_kl = (wcur * (torch.log(wcur) - torch.log(wref))).sum(dim=1).mean()

            loss = loss_main + mu_tilde * loss_kl

            loss.backward()
            optimizer.step()


            # persist ω̃ for next local epoch/round and for sending to server
            self._omega_tilde_prev = omega_tilde_new.detach()
            self._omega_tilde_last = omega_tilde_new.detach().cpu()

        try:
            self._omega_prev_round = omega.detach().cpu()
        except Exception:
            self._omega_prev_round = None

        # -----------------------------------------
        # 3) Tier-4 stats: mean_z + class prototypes
        # -----------------------------------------
        self.task.model.eval()
        with torch.no_grad():
            z = self.task.model.forward_embed(data)  # [N, D]
            if mask is not None:
                z = z[mask]
                y2 = y[mask]
            else:
                y2 = y

            if z.numel() == 0:
                mean_z = torch.zeros(1)
                class_protos = {}
            else:
                z_sum = z.sum(dim=0)
                z_cnt = z.shape[0]
                mean_z = (z_sum / max(z_cnt, 1)).detach().cpu()

                proto_sum, proto_cnt = {}, {}
                for cls in torch.unique(y2):
                    cls = int(cls.item())
                    zc = z[y2 == cls]
                    if zc.numel() == 0:
                        continue
                    proto_sum[cls] = zc.sum(dim=0)
                    proto_cnt[cls] = zc.shape[0]

                class_protos = {int(c): (proto_sum[c] / max(proto_cnt[c], 1)).detach().cpu() for c in proto_sum}

        self._mean_z = mean_z
        self._class_protos = class_protos

    def send_message(self):
        enc_layers = list(self.task.model.base_model.layers[:-1])
        enc_params = []
        for layer in enc_layers:
            enc_params.extend(list(layer.parameters()))
        enc_cpu = [p.detach().cpu().clone() for p in enc_params]

        heads_cpu = {
            i: [p.detach().cpu().clone() for p in head.parameters()]
            for i, head in enumerate(self.task.model.heads)
        }

        # ✅ FIX: real sample count (labeled train nodes)
        processed_data = getattr(self.task, "processed_data", None)
        data = None
        if isinstance(processed_data, dict):
            data = processed_data.get("data", None)
            train_mask = processed_data.get("train_mask", None)
        else:
            train_mask = None

        if data is None:
            data = getattr(self.task, "data", None)

        if data is None:
            num_samples = 1
        else:
            y = getattr(data, "y", None)
            if train_mask is None and hasattr(data, "train_mask"):
                train_mask = data.train_mask
            if train_mask is not None:
                num_samples = int(train_mask.sum().item())
            else:
                num_samples = int(y.shape[0]) if y is not None else 1

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": max(1, num_samples),   # ✅ never 0
            "encoder_params": enc_cpu,
            "cluster_heads": heads_cpu,
            "mean_z": getattr(self, "_mean_z", torch.zeros(1)),
            "class_protos": getattr(self, "_class_protos", {}),
            "w_tilde": getattr(self, "_omega_tilde_last", None),
        }

        
    """def send_message(self):
        # Trainer expects this method. We already wrote payload into message_pool in execute().
        return"""
