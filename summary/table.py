import pandas as pd

INPUT_FILE = "hcfl_results_with4.csv"       # senin verdiğin CSV
OUTPUT_FILE = "results_best_mean_std_by_with4.tsv"

# Load CSV
df = pd.read_csv(INPUT_FILE)

# Prepare container
data = {}
all_datasets = set()

for _, row in df.iterrows():
    dataset = row["Dataset"]
    algo = row["Algorithm"]
    clients = int(row["NumClients"])
    mean = row["best_acc_mean"]
    std = row["best_acc_std"]

    all_datasets.add(dataset)

    row_key = f"{algo} ({clients})"

    cell = f"{mean:.4f} ± {std:.4f}"

    if row_key not in data:
        data[row_key] = {}
    data[row_key][dataset] = cell

# Build DataFrame
table = pd.DataFrame.from_dict(data, orient="index")

# Ensure all datasets appear as columns
table = table.reindex(columns=sorted(all_datasets))

# Fill missing values
table = table.fillna("xcp")

# Sort rows nicely: Algorithm first, then NumClients
def sort_key(x):
    algo, rest = x.rsplit(" ", 1)
    clients = int(rest.strip("()"))
    return (algo, clients)

table = table.loc[sorted(table.index, key=sort_key)]

# Save
table.to_csv(OUTPUT_FILE, sep="\t")

print(f"Saved table to {OUTPUT_FILE}")

# Pretty print
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

print(table)
