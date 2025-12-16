# OpenFGL/openfgl/flcore/hcfl_plus/server.py
import os
import sys
import torch
from openfgl.flcore.base import BaseServer

# Ensure the cloned HCFL repo is on sys.path so we can import aggregator.py.
HCFL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "HCFL"))
if HCFL_ROOT not in sys.path:
    sys.path.append(HCFL_ROOT)

# import aggregator module from HCFL
try:
    import aggregator as hcfl_aggregator_mod   # HCFL/aggregator.py
except Exception as e:
    raise ImportError("Failed to import HCFL's aggregator.py as module 'aggregator'. "
                      "Make sure ../../../../HCFL is correct and contains aggregator.py. "
                      f"Underlying error: {e}")

# try common aggregator class/function names
HCFL_AGG_CLASS = None
for candidate in ("HCFLAggregator", "Aggregator", "HCFLPlusAggregator", "AggregatorHCFL"):
    if hasattr(hcfl_aggregator_mod, candidate):
        HCFL_AGG_CLASS = getattr(hcfl_aggregator_mod, candidate)
        break

HCFL_AGG_FN = None
for candidate in ("aggregate", "aggregate_models", "run_aggregation"):
    if hasattr(hcfl_aggregator_mod, candidate):
        HCFL_AGG_FN = getattr(hcfl_aggregator_mod, candidate)
        break


class HCFLPlusServer(BaseServer):
    """
    Server wrapper that calls HCFL's aggregator implementation to perform clustered aggregation.
    """

    def __init__(self, args, global_data, data_dir, message_pool, device):
        super(HCFLPlusServer, self).__init__(args, global_data, data_dir, message_pool, device)

        # instantiate aggregator if a class is provided
        self.aggregator = None
        if HCFL_AGG_CLASS is not None:
            try:
                # try simple constructor signatures
                try:
                    self.aggregator = HCFL_AGG_CLASS(self.task.model, args)
                except TypeError:
                    # fallback to keyword args that are common in aggregator ctor signatures
                    self.aggregator = HCFL_AGG_CLASS(model=self.task.model, args=args)
            except Exception as e:
                raise RuntimeError("Failed to instantiate HCFL aggregator class. "
                                   f"Check constructor signature in HCFL/aggregator.py. Underlying error: {e}")
        elif HCFL_AGG_FN is None:
            raise ImportError(
                "Couldn't find an aggregator class or aggregation function in HCFL/aggregator.py. "
                "Open the file and check for exported names (Aggregator/Aggregate)."
            )

        self.device = device
        self.args = args

    def execute(self):
        """
        Collect sampled clients' messages and call HCFL aggregator (class or function).
        The wrapper supports both a class instance with an 'aggregate' method or a module-level function.
        """
        sampled = self.message_pool.get("sampled_clients", [])
        if not sampled:
            return

        client_weights = []
        client_embeddings = []
        client_sizes = []

        for cid in sampled:
            key = f"client_{cid}"
            cm = self.message_pool.get(key, None)
            if cm is None:
                continue
            client_weights.append(cm.get("weight", None))
            client_embeddings.append(cm.get("embedding", None))
            client_sizes.append(cm.get("num_samples", 0))

        if len(client_weights) == 0:
            return

        # If aggregator is a class instance that exposes aggregate(...)
        if self.aggregator is not None and hasattr(self.aggregator, "aggregate"):
            # call aggregator.aggregate(client_weights, client_embeddings, client_sizes, **kwargs)
            try:
                new_global_params = self.aggregator.aggregate(client_weights,
                                                              client_embeddings,
                                                              client_sizes)
            except TypeError:
                # try different signature location
                new_global_params = self.aggregator.aggregate(client_weights, client_sizes)
        else:
            # call module-level aggregator function
            if HCFL_AGG_FN is None:
                raise RuntimeError("No aggregator function or aggregator.aggregate method available.")
            try:
                # try aggregate(weights, embeddings, sizes, args)
                new_global_params = HCFL_AGG_FN(client_weights, client_embeddings, client_sizes, self.args)
            except TypeError:
                # try aggregate(weights, sizes)
                new_global_params = HCFL_AGG_FN(client_weights, client_sizes)

        # Load aggregated params into self.task.model
        if new_global_params is None:
            # fallback: do nothing (aggregator might update server-side model internally)
            pass
        else:
            # new_global_params expected to be list of tensors matching model.parameters()
            with torch.no_grad():
                for p, new_p in zip(self.task.model.parameters(), new_global_params):
                    p.data.copy_(new_p.to(p.device))

    def send_message(self):
        # Send server model weights to clients (OpenFGL expects "server" -> "weight")
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters())
        }
