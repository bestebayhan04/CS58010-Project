import torch
import torch.nn as nn
from openfgl.flcore.base import BaseClient


def model_to_prototype(model: nn.Module):
    """
    Convert model parameters to a compact prototype vector (1D tensor).
    Flatten all parameter tensors and normalize (L2).
    """
    vecs = []
    for p in model.parameters():
        vecs.append(p.detach().view(-1))
    if len(vecs) == 0:
        return torch.zeros(1)
    proto = torch.cat(vecs)
    proto = proto.float()
    norm = torch.norm(proto)
    if norm > 0:
        proto = proto / norm
    return proto


class HCFLClient(BaseClient):
    """
    Client-side of a simple HCFL (clustered federated learning) implementation.
    Clients train locally, compute a prototype vector from model parameters,
    and send both the updated parameters and the prototype to the server.
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super(HCFLClient, self).__init__(args, client_id, data, data_dir, message_pool, device)
        # No extra state required here; keep the same interface as FedAvgClient.

    def execute(self):
        """
        1) Synchronize local model with server-provided cluster model (if any)
        2) Run local training (self.task.train())
        """
        

        # If server has sent cluster weights and a mapping, sync to the cluster model for this client
        server_msg = self.message_pool.get("server", None)
        if server_msg is not None:
            # server_msg is expected to contain:
            #   "weights_per_cluster": list of list(parameters)
            #   "client_cluster_map": { "client_0": cluster_idx, ... }
            weights_per_cluster = server_msg.get("weights_per_cluster", None)
            client_map = server_msg.get("client_cluster_map", None)

            if weights_per_cluster is not None and client_map is not None:
                key = f"client_{self.client_id}"
                cluster_idx = client_map.get(key, None)
                print(f"Client {self.client_id} → cluster {cluster_idx}")
                if cluster_idx is None:
                    # fallback: global broadcast (if provided as single model)
                    global_weights = server_msg.get("weight", None)
                    if global_weights is not None:
                        with torch.no_grad():
                            for (local_param, global_param) in zip(self.task.model.parameters(), global_weights):
                                local_param.data.copy_(global_param)
                else:
                    # copy cluster model into local model
                    cluster_weights = weights_per_cluster[cluster_idx]
                    with torch.no_grad():
                        for (local_param, cw) in zip(self.task.model.parameters(), cluster_weights):
                            local_param.data.copy_(cw)

        # Run local training
        self.task.train()

    def send_message(self):
        """
        Send:
          - num_samples
          - weight (list of parameters)
          - prototype vector (as CPU tensor)
        """
        proto = model_to_prototype(self.task.model).cpu()
        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": self.task.num_samples,
            "weight": list(self.task.model.parameters()),
            "prototype": proto
        }
