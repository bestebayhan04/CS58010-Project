from HCFL.learner.learners_ensemble import LearnersEnsemble
from HCFL.utils.logger import TensorboardLogger
from HCFL.client import Client as HCFLClient

class HCFLPlusClient(BaseClient):

    def __init__(self, args, client_id, data, data_dir, message_pool, device):
        super().__init__(args, client_id, data, data_dir, message_pool, device)

        # Build learner ensemble (HCFL requires this)
        self.learners_ensemble = LearnersEnsemble(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            n_learners=args.n_learners,
            device=device
        )

        # Build logger (or dummy logger)
        self.logger = TensorboardLogger(log_dir=f"./runs/hcfl_client_{client_id}")

        # HCFL requires REAL DataLoaders, not raw tensors
        train_loader = self.task.train_loader
        val_loader = self.task.val_loader

        # Now instantiate the real HCFL client correctly
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
