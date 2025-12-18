import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient


def model_to_prototype(model: nn.Module) -> torch.Tensor:
    vecs = [p.detach().view(-1) for p in model.parameters()]
    if not vecs:
        return torch.zeros(1)
    proto = torch.cat(vecs).float()
    n = torch.norm(proto)
    if n > 0:
        proto = proto / n
    return proto


class HCFLPlusClient(BaseClient):

    def execute(self):
        server_msg = self.message_pool.get("server", None)
        #mu_tilde = float(getattr(self.args, "hcfl_mu_tilde", 0.4))

        if server_msg is not None:
            key = f"client_{self.client_id}"
            cluster_models = server_msg.get("cluster_models", None)
            alpha_map = server_msg.get("client_cluster_weights", {}).get(key, None)

            if cluster_models is not None and alpha_map:
                with torch.no_grad():
                    for p in self.task.model.parameters():
                        p.zero_()

                    for cid, alpha in alpha_map.items():
                        cid = int(cid)  # IMPORTANT
                        if cid not in cluster_models:
                            continue
                        params_k = cluster_models[cid]
                        for p, pk in zip(self.task.model.parameters(), params_k):
                            p.add_(pk.to(self.device), alpha=float(alpha))
                
                # Store target model for µ̃ penalty: θ_target = Σₖ ω̃ᵢₖ θₖ
                """
                self.target_model_params = [
                    p.detach().clone() for p in self.task.model.parameters()
                ]
            else:
                self.target_model_params = None
        else:
            self.target_model_params = None

        # Override loss function to include µ̃ penalty if enabled
        if self.target_model_params is not None and mu_tilde > 0:
            original_loss_fn = self.task.loss_fn
            
            def hcfl_loss_fn(embedding, logits, labels, mask):
                # Base loss (standard cross-entropy or whatever the task uses)
                base_loss = original_loss_fn(embedding, logits, labels, mask)
                
                # µ̃ penalty: ||θ_client - θ_target||²
                penalty = 0.0
                for p_client, p_target in zip(self.task.model.parameters(), self.target_model_params):
                    penalty += torch.nn.functional.mse_loss(
                        p_client, 
                        p_target.to(self.device)
                    )
                
                return base_loss + mu_tilde * penalty
            
            self.task.loss_fn = hcfl_loss_fn
            """

        self.task.train()

    def send_message(self):
        proto = model_to_prototype(self.task.model).cpu()

        # IMPORTANT: detach+cpu+clone
        weights_cpu = [p.detach().cpu().clone() for p in self.task.model.parameters()]

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": int(self.task.num_samples),
            "weight": weights_cpu,
            "prototype": proto,
        }
