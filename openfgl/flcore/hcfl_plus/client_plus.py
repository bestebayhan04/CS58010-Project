import os
import sys
import torch
from openfgl.flcore.base import BaseClient

# Make sure the cloned HCFL repo is importable.
# Assumes HCFL is cloned at <project_root>/HCFL alongside openfgl/.
HCFL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "HCFL"))
if HCFL_ROOT not in sys.path:
    sys.path.append(HCFL_ROOT)

from HCFL.learners.learners_ensemble import LearnersEnsemble
from HCFL.client import Client as HCFLClient


class NullLogger:
    """Fallback logger that safely ignores all logging calls."""
    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop


class HCFLPlusClient(BaseClient):
    """
    OpenFGL client wrapper around the official HCFL client.

    Responsibilities:
      - Build HCFL LearnersEnsemble and logger.
      - Wrap HCFLClient and run its local training each round.
      - Expose weights + embedding + num_samples to OpenFGL's message_pool
        in the format expected by HCFLPlusServer (server_plus.py).
    """

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)

        self.device = device
        self.args = args

        # Build learner ensemble (check input_dim/output_dim against your HCFL repo)
        self.learners_ensemble = LearnersEnsemble(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            n_learners=args.n_learners,
            device=device,
        )

        # Logger (HCFL expects a logger object; use a noop logger to avoid dependency)
        self.logger = NullLogger()

        # Use OpenFGL's DataLoaders as HCFL train/val iterators
        train_loader = self.task.train_loader
        val_loader = self.task.val_loader

        # Instantiate the official HCFL client
        self.raw_client = HCFLClient(
            client_index=client_id,
            learners_ensemble=self.learners_ensemble,
            train_iterator=train_loader,
            val_iterator=val_loader,
            logger=self.logger,
            local_steps=args.local_steps,
            tune_locally=args.tune_locally,
            meta_learning=args.meta_learning,
            alpha=args.alpha,
            beta=args.beta,
            meta_step_size=args.meta_step_size,
            global_step_size=args.global_step_size,
            personalization=args.personalization,
            stochastic=args.stochastic,
            verbose=args.verbose,
            seed=args.seed,
        )

    def execute(self):
        """
        Run HCFL local training for this round.

        Adapt the method name here to the actual HCFL client API
        (e.g. local_train(), step(), train(), etc.).
        """
        # Example – if HCFL client exposes a 'local_train' method:
        if hasattr(self.raw_client, "local_train"):
            self.raw_client.local_train()
        elif hasattr(self.raw_client, "train"):
            self.raw_client.train()
        else:
            raise AttributeError(
                "HCFLClient has no 'local_train' or 'train' method. "
                "Adapt HCFLPlusClient.execute() to the correct HCFL API."
            )

    def _extract_weights_and_embedding(self):
        """
        Extract model weights and a client embedding/prototype from the HCFL client.

        You MUST adapt this to how the official HCFL code stores its model(s)
        and client representation. The server wrapper expects:

          - weights: list of tensors (like model.parameters())
          - embedding: 1D tensor or numpy array (prototype / representation)
        """
        # Example 1: if HCFL stores a single model:
        if hasattr(self.raw_client, "model"):
            model = self.raw_client.model
            weights = [p.detach().cpu() for p in model.parameters()]
        # Example 2: if HCFL uses learners_ensemble and you want the "global" learner:
        elif hasattr(self.raw_client, "learners_ensemble"):
            # This depends heavily on HCFL implementation – adjust accordingly
            model = self.raw_client.learners_ensemble[0].model
            weights = [p.detach().cpu() for p in model.parameters()]
        else:
            raise AttributeError(
                "Cannot find model parameters inside HCFLClient. "
                "Adapt _extract_weights_and_embedding() to your HCFL code."
            )

        # Example: suppose HCFL client maintains a prototype or embedding attribute
        if hasattr(self.raw_client, "prototype"):
            embedding = self.raw_client.prototype.detach().cpu()
        elif hasattr(self.raw_client, "embedding"):
            embedding = self.raw_client.embedding.detach().cpu()
        else:
            # Fallback: derive a simple embedding from the weights
            flat = torch.cat([p.view(-1) for p in weights])
            norm = torch.norm(flat)
            if norm > 0:
                flat = flat / norm
            embedding = flat

        return weights, embedding

    def send_message(self):
        """
        Send to OpenFGL server:

          - 'weight': list[tensor]   (for HCFL aggregation)
          - 'embedding': tensor/np   (client representation for clustering)
          - 'num_samples': int
        """
        weights, embedding = self._extract_weights_and_embedding()

        self.message_pool[f"client_{self.client_id}"] = {
            "num_samples": int(self.task.num_samples),
            "weight": weights,
            "embedding": embedding,
        }