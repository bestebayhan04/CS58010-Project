import openfgl.config as config
from openfgl.flcore.trainer import FGLTrainer
import numpy as np
import copy
import random
import traceback
import torch

NUM_RUNS = 3

fl_algorithms = ["fedavg", "feddc", "fedproto", "fedtgp",
                 "fedsage_plus", "fedgta", "fedpub", "fgssl",
                 "fedgl", "adafgl", "fggp", "fedtad"]

fl_datasets = ["Cora", "CiteSeer", "PubMed", "Computers",
               "Photo", "ogbn-products", "Amazon-ratings",
               "Chameleon", "Actor"]

#fl_datasets = ["Photo", "ogbn-products", "Amazon-ratings","Chameleon", "Actor"]

for dataset in fl_datasets:
    for algo in fl_algorithms:

        run_results = []   # ← moved here (was buggy before)

        for _ in range(NUM_RUNS):
            args = config.args
            args.simulation_mode = "subgraph_fl_louvain"
            args.num_clients = 10
            args.model = ["gcn"]
            args.metrics = ["accuracy"]
            args.root = "your_data_root"
            args.dataset = [dataset]
            args.fl_algorithm = algo

            try:
                trainer = FGLTrainer(args)
                trainer.train()
                final_accuracy = trainer.evaluation_result["best_test_accuracy"]

                run_results.append(final_accuracy)

            except RuntimeError as e:
                # CUDA OOM CHECK
                if "CUDA out of memory" in str(e):
                    run_results.append("oom")

                    # free CUDA memory
                    torch.cuda.empty_cache()

                else:
                    # Other runtime errors
                    run_results.append(f"exception: {str(e).splitlines()[0]}")

            except Exception as e:
                # Any non-runtime exception
                run_results.append(f"exception: {str(e).splitlines()[0]}")

        # After NUM_RUNS, compute metrics only on successful accuracies
        numeric_acc = [x for x in run_results if isinstance(x, (int, float))]

        if len(numeric_acc) > 0:
            mean_acc = sum(numeric_acc) / len(numeric_acc)
            var_acc = (max(numeric_acc) - min(numeric_acc)) / 2
        else:
            mean_acc = "n/a"
            var_acc = "n/a"

        with open("res.txt", "a") as f:
            f.write(f"({dataset}, {algo}): {run_results} -> {mean_acc} +- {var_acc}\n")
