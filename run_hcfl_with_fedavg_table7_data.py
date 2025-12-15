import copy
import io
import os
import re
import statistics as stats
from contextlib import redirect_stdout

import pandas as pd
import torch


import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer


# =========================
# USER SETTINGS
# =========================
ROOT = "your_data_root"          # <-- set this
SIM_MODE = "subgraph_fl_louvain" # keep as in your experiments
DATASETS = ["Cora", "CiteSeer", "PubMed", "Photo", "Computers", "Chameleon", "Actor"]

# algorithms you want to compare
ALGORITHMS = {
    #"FedAvg":  {"fl_algorithm": "fedavg",     "model": ["gcn"]},
    "HCFL+":   {"fl_algorithm": "hcfl_plus",  "model": ["gcn"]},
    #"HCFL++":   {"fl_algorithm": "hcfl_v2",  "model": ["gcn"]},
}

# number of clients sweep (edit as you like)
CLIENT_COUNTS = [5, 10, 20, 30]

# do at least 3 runs per point
N_RUNS = 3

# HCFL/HCFL+ setting (static cluster count example)
NUM_CLUSTERS = 3

# Which metric to report (choose ONE and use everywhere):
#   "best" = best_test_accuracy at the round with best_val_accuracy (common in papers)
#   "final" = curr_test_accuracy at last round
REPORT_MODE = "best"  # "best" or "final"

# Privacy: do NOT store per-client metrics lines
PRIVACY_MODE = True


# =========================
# PARSING HELPERS
# =========================
RE_BEST = re.compile(r"best_test_accuracy:\s*([0-9]*\.?[0-9]+)")
RE_FINAL = re.compile(r"curr_test_accuracy:\s*([0-9]*\.?[0-9]+)")

def _extract_accuracy_from_stdout(text: str, mode: str) -> float | None:
    """
    Parse accuracy from trainer stdout.
    Returns a float in [0,1] typically, or None if not found.
    """
    if mode == "best":
        m = RE_BEST.search(text)
    else:
        m = RE_FINAL.search(text)
    if not m:
        return None
    return float(m.group(1))


def _sanitize_stdout_for_privacy(text: str) -> str:
    """
    Remove per-client lines like:
      [client 0] loss_train: ...
    """
    out_lines = []
    for line in text.splitlines():
        if line.strip().startswith("[client "):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# =========================
# RUN ONE EXPERIMENT
# =========================
def run_one(args) -> tuple[float | None, str]:
    """
    Runs trainer.train() and returns (acc, stdout_text)
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        trainer = FGLTrainer(args)
        trainer.train()
    stdout_text = buf.getvalue()
    acc = _extract_accuracy_from_stdout(stdout_text, REPORT_MODE)
    if PRIVACY_MODE:
        stdout_text = _sanitize_stdout_for_privacy(stdout_text)
    return acc, stdout_text


# =========================
# MAIN
# =========================
def main():
    base_args = config.args
    base_args.root = ROOT
    base_args.simulation_mode = SIM_MODE
    base_args.metrics = ["accuracy"]
    base_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU usage (OpenFGL typically uses torch device internally):
    # Keep it simple: if CUDA exists, we let torch pick it.
    # If your config has args.device, you can uncomment:
    # base_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    results_rows = []

    os.makedirs("summary", exist_ok=True)
    os.makedirs("summary/logs", exist_ok=True)

    for dataset in DATASETS:
        for algo_name, algo_cfg in ALGORITHMS.items():
            for n_clients in CLIENT_COUNTS:
                accs = []
                for run_id in range(N_RUNS):
                    args = copy.deepcopy(base_args)

                    # core settings
                    args.dataset = [dataset]
                    args.fl_algorithm = algo_cfg["fl_algorithm"]
                    args.model = algo_cfg["model"]

                    # sweep variable
                    args.num_clients = int(n_clients)

                    # HCFL-specific
                    if "hcfl" in args.fl_algorithm.lower():
                        args.num_clusters = NUM_CLUSTERS

                    # multiple runs: vary seed AND run id
                    # (OpenFGL may use one or both depending on implementation)
                    args.seed = run_id
                    args.run = run_id

                    print("=" * 90)
                    print(f"Dataset={dataset} | Algo={algo_name} | Clients={n_clients} | Run={run_id} | Report={REPORT_MODE}")
                    print("=" * 90)

                    try:
                        acc, out = run_one(args)
                    except Exception as e:
                        acc, out = None, f"[ERROR] {e}"

                    # Save sanitized log for debugging (privacy-safe)
                    log_path = os.path.join(
                        "summary", "logs",
                        f"{dataset}__{algo_name}__C{n_clients}__run{run_id}.log".replace("+", "plus")
                    )
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(out)

                    if acc is None:
                        print("⚠️ Could not parse accuracy from stdout for this run.")
                    else:
                        print(f"✅ Parsed accuracy ({REPORT_MODE}) = {acc:.4f}")

                    accs.append(acc)

                # compute mean/std over successful runs only
                valid = [a for a in accs if a is not None]
                if len(valid) == 0:
                    mean_acc = None
                    std_acc = None
                elif len(valid) == 1:
                    mean_acc = valid[0]
                    std_acc = 0.0
                else:
                    mean_acc = stats.mean(valid)
                    std_acc = stats.pstdev(valid)  # population stddev; use stdev() if you prefer sample std

                results_rows.append({
                    "Dataset": dataset,
                    "Algorithm": algo_name,
                    "NumClients": n_clients,
                    "RunsRequested": N_RUNS,
                    "RunsParsed": len(valid),
                    f"{REPORT_MODE}_acc_mean": mean_acc,
                    f"{REPORT_MODE}_acc_std": std_acc,
                })

    df = pd.DataFrame(results_rows)
    out_csv = os.path.join("summary", f"results_{REPORT_MODE}_mean_std_by_clients.csv")
    df.to_csv(out_csv, index=False)

    # Also produce a simple pivot table for the "default" client count (if present)
    if 10 in CLIENT_COUNTS:
        df10 = df[df["NumClients"] == 10].copy()
        pivot = df10.pivot(index="Dataset", columns="Algorithm", values=f"{REPORT_MODE}_acc_mean").reset_index()
        pivot_csv = os.path.join("summary", f"table_like_clients10_{REPORT_MODE}.csv")
        pivot.to_csv(pivot_csv, index=False)

    print("\n✅ Finished.")
    print(f"📁 Wrote: {out_csv}")
    if 10 in CLIENT_COUNTS:
        print(f"📁 Wrote: summary/table_like_clients10_{REPORT_MODE}.csv")
    print("📁 Logs (privacy-sanitized): summary/logs/")


if __name__ == "__main__":
    main()
