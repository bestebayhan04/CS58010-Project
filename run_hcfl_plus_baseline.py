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

OUT_CSV = "hcfl_results_with4.csv"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""def print_client_clusters(trainer, num_clients: int):
    # Prefer server internal cache (your server sets this)
    cmap = None
    try:
        cmap = getattr(trainer.server, "_client_cluster_map", None)
    except Exception:
        cmap = None

    # Fallback: message_pool (server writes it every round)
    if not isinstance(cmap, dict):
        try:
            smsg = trainer.message_pool.get("server", {})
            cmap = smsg.get("client_cluster_map", None)
        except Exception:
            cmap = None

    if not isinstance(cmap, dict) or len(cmap) == 0:
        print("⚠️ No client_cluster_map found (server didn't publish it).")
        return

    # Print in order client_0..client_{n-1}
    print("\n📌 Client → Cluster assignments (hard):")
    for cid in range(int(num_clients)):
        key = f"client_{cid}"
        if key in cmap:
            print(f"  {key}  ->  cluster {int(cmap[key])}")
        else:
            print(f"  {key}  ->  (missing)")
    print()"""

def print_client_clusters_soft(trainer, num_clients: int, top_m: int = 3, thr: float = 0.15):
    """
    Prints soft cluster membership per client:
      - Hard assignment (argmax)
      - Top-M clusters
      - All clusters with prob >= thr
    """

    # ---- fetch server msg (best effort) ----
    smsg = {}
    try:
        smsg = trainer.message_pool.get("server", {}) or {}
    except Exception:
        smsg = {}

    # hard map (argmax)
    cmap = None
    try:
        cmap = getattr(trainer.server, "_client_cluster_map", None)
    except Exception:
        cmap = None
    if not isinstance(cmap, dict):
        cmap = smsg.get("client_cluster_map", None)

    # soft probs (what you want)
    probs = smsg.get("client_cluster_probs", None)

    # fallback: sometimes stored as w_tilde_init
    if not isinstance(probs, dict):
        probs = smsg.get("w_tilde_init", None)

    print("\n📌 Client → Cluster memberships")

    # If no soft info, fall back to hard only
    if not isinstance(probs, dict) or len(probs) == 0:
        print("⚠️ No soft cluster probs found. Printing hard assignments only.\n")
        if isinstance(cmap, dict):
            for cid in range(int(num_clients)):
                key = f"client_{cid}"
                print(f"  {key}  ->  cluster {int(cmap.get(key, -1))}")
        print()
        return

    # pretty print per client
    for cid in range(int(num_clients)):
        key = f"client_{cid}"
        p = probs.get(key, None)

        # If key missing, still print something
        if p is None:
            hard = int(cmap.get(key, -1)) if isinstance(cmap, dict) else -1
            print(f"  {key}: hard={hard} | (no soft probs)")
            continue

        # p can be list / np array / torch tensor
        try:
            p_list = [float(x) for x in list(p)]
        except Exception:
            hard = int(cmap.get(key, -1)) if isinstance(cmap, dict) else -1
            print(f"  {key}: hard={hard} | (soft probs unreadable)")
            continue

        K = len(p_list)
        # normalize just in case
        s = sum(p_list)
        if s > 1e-12:
            p_list = [x / s for x in p_list]

        # hard assignment from soft
        hard_soft = int(max(range(K), key=lambda i: p_list[i])) if K > 0 else -1
        hard_map = int(cmap.get(key, hard_soft)) if isinstance(cmap, dict) else hard_soft

        # top-M
        order = sorted(range(K), key=lambda i: p_list[i], reverse=True)
        top = order[:max(1, min(int(top_m), K))]
        top_str = ", ".join([f"{i}:{p_list[i]:.3f}" for i in top])

        # thresholded membership
        above = [i for i in order if p_list[i] >= float(thr)]
        if len(above) == 0:
            above = top  # if nothing passes thr, show top instead
        above_str = ", ".join([f"{i}:{p_list[i]:.3f}" for i in above])

        print(f"  {key}: hard={hard_map} | top{len(top)} [{top_str}] | >= {thr:.2f} [{above_str}]")

    print()

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
    USE_IMPROVED = True   # False -> clientv2/serverv2 baseline
                       # True  -> client_improved/server_improved


    for dataset in DATASETS:
        accs = []

        for run_id in range(N_RUNS):

            set_seed(run_id)

            args = copy.deepcopy(base_args)


            # =========================
            # 1) CORE (used in BOTH v2 + improved)
            # =========================
            args.dataset = [dataset]
            args.num_clients = int(NUM_CLIENTS)
            args.fl_algorithm = ALGORITHM
            args.model = MODEL
            args.num_clusters = int(NUM_CLUSTERS)

            args.local_epochs = 1

            # reproducibility
            args.seed = run_id
            args.run = run_id


            # =========================
            # 2) HCFL+ v2 BASELINE (clientv2/serverv2)
            #    -> comment out this whole block when using improved version
            # =========================
            
            if not USE_IMPROVED:
                args.hcfl_soft = True
                args.hcfl_temp = 1.0 
                args.hcfl_top_k = 0

                # Make sure improved-only knobs are OFF (so v2 is pure baseline)
                args.hcfl_adaptive_k = False
                args.hcfl_remove = False
                args.hcfl_remove_patience = 0
                args.hcfl_mu = 0.0
                args.hcfl_mu_tilde = 0.0

                # v2 doesn't use this; keep it unset
                if hasattr(args, "hcfl_dist_metric"):
                    args.hcfl_dist_metric = None

            else:
                # =========================
                # 3) HCFL+ IMPROVED (client_improved/server_improved)
                #    -> comment out this whole block when using v2 baseline
                # =========================

                args.hcfl_dist_metric = "cosine"   # cosine/euclidean (improved server uses it)
                args.hcfl_mu =  0.1   # not 1.0                # coupling strength in ω update (improved client uses it)
                args.hcfl_mu_tilde = 0.0          # KL regularizer strength (keep 0 for paper-faithful baseline)

                # Adaptive-K + removal (improved server uses these)
                args.hcfl_adaptive_k = True
                args.hcfl_rho = 0.10
                args.hcfl_max_k = 50
                args.hcfl_remove = True
                args.hcfl_remove_patience = 10

                # # (extras, if annealing applied)
                # # args.hcfl_top_k = 0             # improved client uses hcfl_top_k too (optional)
                # # args.hcfl_temp_anneal = True
                # # args.hcfl_temp_anneal_rate = 0.98
                # # args.hcfl_temp_min = 0.2


            print("=" * 80)
            print(f"Dataset={dataset} | Algo={ALGORITHM} | Clients={NUM_CLIENTS} | Run={run_id}")
            print("=" * 80)

            acc = None
            final_k = None
            try:
                trainer = FGLTrainer(args)
                trainer.train()

                # ✅ print client cluster ids after training
                print_client_clusters_soft(trainer, NUM_CLIENTS, top_m=3, thr=0.15)


                acc = trainer.evaluation_result.get("best_test_accuracy", None)
                # actual K at the end of training (server-side)
                
                try:
                    final_k = int(trainer.server._k)  # internal state
                except Exception:
                    # fallback: server message pool
                    smsg = trainer.message_pool.get("server", {})
                    final_k = int(smsg.get("k", args.num_clusters))

                # store for this run (optional: you can keep a list per dataset)

                
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
            #"NumClusters": int(NUM_CLUSTERS),
            "InitK": int(NUM_CLUSTERS),        # what you started with
            "FinalK": int(final_k) if final_k is not None else None,  # what you ended with
            "RunsRequested": int(N_RUNS),
            "RunsParsed": int(len(valid)),
        })

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    print("\n✅ Finished")
    print(f"📁 Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
