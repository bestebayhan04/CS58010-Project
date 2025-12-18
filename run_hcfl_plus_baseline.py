import copy
import random
import numpy as np
import pandas as pd
import torch
import statistics as stats

import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer


# =========================
# USER SETTINGS
# =========================
ROOT = "your_data_root"

DATASETS = [
    "Cora", "CiteSeer", "PubMed", "Computers",
    "Photo", "Chameleon", "Actor"
]

SIM_MODE = "subgraph_fl_louvain"
NUM_CLIENTS = 10
NUM_CLUSTERS = 3
N_RUNS = 3

ALGORITHM = "hcfl_plus"   # must match OpenFGL name
MODEL = ["gcn"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_CSV = "hcfl_plus_results_with_added_features.csv"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================
# MAIN
# =========================
def main():
    base_args = config.args
    base_args.root = ROOT
    base_args.simulation_mode = SIM_MODE
    base_args.metrics = ["accuracy"]
    base_args.device = DEVICE

    results = []

    for dataset in DATASETS:
        accs = []

        for run_id in range(N_RUNS):

            set_seed(run_id)

            args = copy.deepcopy(base_args)

            # core experiment setup
            args.dataset = [dataset]
            args.num_clients = int(NUM_CLIENTS)
            args.fl_algorithm = ALGORITHM
            args.model = MODEL
            args.num_clusters = int(NUM_CLUSTERS)

            # ✅ HCFL+ baseline knobs (explicit, per-run)
            args.hcfl_soft = True
            #args.hcfl_scwu_temp = 1.0
            args.hcfl_top_k = 0
            args.local_epochs = 1
            args.hcfl_dist_metric = "cosine"

            # improvements toggles
            #args.hcfl_temp_anneal = True
            #args.hcfl_temp_anneal_rate = 0.98
            #args.hcfl_temp_min = 0.2

            args.hcfl_adaptive_k = True
            args.hcfl_rho = 0.10
            args.hcfl_max_k = 10

            args.hcfl_mu = 1.0          # µ in the paper (strength of ω -> ω̃ coupling)
            args.hcfl_mu_tilde = 0.4    # µ̃ in experiments for soft clustering variants (optional)


            args.hcfl_remove = True
            args.hcfl_remove_patience = 10

            # ✅ baseline safety (turn on ONLY if your paper baseline includes it)
            args.hcfl_mu_tilde = 0.0

            # reproducibility
            args.seed = run_id
            args.run = run_id

            print("=" * 80)
            print(f"Dataset={dataset} | Algo={ALGORITHM} | Clients={NUM_CLIENTS} | Run={run_id}")
            print("=" * 80)

            acc = None
            try:
                trainer = FGLTrainer(args)
                trainer.train()
                acc = trainer.evaluation_result.get("best_test_accuracy", None)
                
            except Exception as e:
                import traceback
                print("❌ Error repr:", repr(e))
                traceback.print_exc()
                acc = None  # Ensure acc is set even on exception

            if acc is not None:
                accs.append(acc)

        # keep only successful runs
        valid = [a for a in accs if isinstance(a, (int, float))]

        if len(valid) == 0:
            mean_acc = None
            std_acc = None
        elif len(valid) == 1:
            mean_acc = float(valid[0])
            std_acc = 0.0
        else:
            mean_acc = float(stats.mean(valid))
            std_acc = float(stats.stdev(valid))   # sample std across runs

        # ✅ IMPORTANT: column names match your table builder script
        results.append({
            "Dataset": dataset,
            "Algorithm": "HCFL+",
            "NumClients": int(NUM_CLIENTS),
            "best_acc_mean": mean_acc,
            "best_acc_std": std_acc,
            # optional extra columns (won't hurt)
            "Simulation": SIM_MODE,
            "NumClusters": int(NUM_CLUSTERS),
            "RunsRequested": int(N_RUNS),
            "RunsParsed": int(len(valid)),
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    print("\n✅ Finished")
    print(f"📁 Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
