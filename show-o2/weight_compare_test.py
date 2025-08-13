import re
from pathlib import Path

# Load both text files
path1 = "output_log_15B.txt"
path2 = "output_log_2000.txt"

def parse_layers(text):
    pattern = re.compile(
        r"Layer: (?P<name>.*?) \| Shape: (?P<shape>.*?) \| Min: (?P<min>-?\d+\.\d+|\d+) \| Max: (?P<max>-?\d+\.\d+|\d+) \| Mean: (?P<mean>-?\d+\.\d+|\d+) \| Std: (?P<std>-?\d+\.\d+|\d+)"
    )
    return {m["name"]: {
                "shape": m["shape"],
                "min": float(m["min"]),
                "max": float(m["max"]),
                "mean": float(m["mean"]),
                "std": float(m["std"]),
            } for m in pattern.finditer(text)}

# Read and parse
text1 = Path(path1).read_text()
text2 = Path(path2).read_text()
data1 = parse_layers(text1)
data2 = parse_layers(text2)

# Compare and format full output
lines = []
for layer in sorted(set(data1) | set(data2)):
    if layer not in data1:
        lines.append(f"Layer only in file2: {layer}\n")
        continue
    if layer not in data2:
        lines.append(f"Layer only in file1: {layer}\n")
        continue

    d1, d2 = data1[layer], data2[layer]
    differences = []
    matches = []
    for stat in ["min", "max", "mean", "std"]:
        if d1[stat] != d2[stat]:
            differences.append(f"{stat.upper()} DIFF | {d1[stat]} != {d2[stat]}")
        else:
            matches.append(f"{stat.upper()} MATCH | {d1[stat]}")

    if differences:
        lines.append(f"Layer: {layer}\n" + "\n".join(["  " + diff for diff in differences]) + "\n")
    else:
        lines.append(f"Layer: {layer} | All statistics MATCH\n" + "\n".join(["  " + match for match in matches]) + "\n")

# Save differences
output_path = "layer_differences.txt"
Path(output_path).write_text("\n".join(lines))