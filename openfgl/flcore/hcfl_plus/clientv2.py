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
