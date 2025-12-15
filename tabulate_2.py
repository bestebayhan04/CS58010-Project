import re
import pandas as pd

INPUT_FILE = "res.txt"
OUTPUT_FILE = "results_table.tsv"

line_re = re.compile(
    r"\((?P<dataset>[^,]+),\s*(?P<algo>[^)]+)\):\s*(?P<runs>\[.*?\])\s*->\s*(?P<mean>[^ ]+)\s*\+-\s*(?P<var>.+)"
)

data = {}
all_datasets = set()   # <-- Collect all dataset names here

with open(INPUT_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        m = line_re.match(line)
        if not m:
            continue

        dataset = m.group("dataset").strip()
        algo = m.group("algo").strip()
        runs_raw = m.group("runs")
        mean_raw = m.group("mean")
        var_raw = m.group("var")

        all_datasets.add(dataset)  # <-- track dataset

        # Parse runs list safely
        try:
            runs = eval(runs_raw)
        except Exception:
            runs = []

        # Decide cell content
        if all(isinstance(x, str) and x == "oom" for x in runs):
            cell = "oom"
        elif any(isinstance(x, str) and x.startswith("exception") for x in runs):
            cell = "xcp"
        else:
            try:
                mean = float(mean_raw)
                var = float(var_raw)
                cell = f"{mean:.4f} ± {var:.4f}"
            except:
                cell = "xcp"

        if algo not in data:
            data[algo] = {}
        data[algo][dataset] = cell

# Build DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Ensure ALL dataset columns appear (no skipping)
df = df.reindex(columns=sorted(all_datasets))

# Fill missing with xcp (or "" if you prefer blank)
df = df.fillna("xcp")

# Sort algorithms alphabetically
df = df.sort_index()

df.to_csv(OUTPUT_FILE, sep="\t")
print(f"Saved table to {OUTPUT_FILE}")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

print(df)
