import re
import pandas as pd

INPUT_FILE = "res.txt"
OUTPUT_FILE = "results_table.tex"

# Regex to parse lines
line_re = re.compile(
    r"\((?P<dataset>[^,]+),\s*(?P<algo>[^)]+)\):\s*(?P<runs>\[.*?\])\s*->\s*(?P<mean>[^ ]+)\s*\+-\s*(?P<var>.+)"
)

data = {}

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

        # Parse list of runs
        try:
            runs = eval(runs_raw)
        except Exception:
            runs = []

        # Decide cell value
        if all(isinstance(x, str) and x == "oom" for x in runs):
            cell = "oom"
        elif any(isinstance(x, str) and x.startswith("exception") for x in runs):
            cell = "xcp"
        else:
            try:
                mean = float(mean_raw)
                var = float(var_raw)
                cell = f"{mean * 100:.2f} ± {var * 100:.2f}"
            except:
                cell = "xcp"

        if algo not in data:
            data[algo] = {}
        data[algo][dataset] = cell

# Convert into DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Ensure all columns for all datasets appear
all_datasets = sorted({col for row in data.values() for col in row.keys()})
df = df.reindex(columns=all_datasets)

# Sort rows alphabetically
df = df.sort_index()

# Convert to LaTeX
latex_table = df.to_latex(escape=False, na_rep="")

with open(OUTPUT_FILE, "w") as f:
    f.write(latex_table)

print("Saved LaTeX table to", OUTPUT_FILE)
print(latex_table)
