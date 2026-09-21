from pathlib import Path
import csv
from collections import Counter, defaultdict

dataset_path = Path(
    "/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild"
)

split_file = dataset_path / "plantwild_split.csv"

counts = defaultdict(Counter)

with open(split_file, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        counts[row["class"]][row["split"]] += 1

print("=== PlantWild Split Verification ===")

print(f"Classes: {len(counts)}")

print("\nClass proportions:")

for class_name in sorted(counts):
    train = counts[class_name]["train"]
    validation = counts[class_name]["validation"]
    test = counts[class_name]["test"]

    total = train + validation + test

    train_pct = train / total * 100
    validation_pct = validation / total * 100
    test_pct = test / total * 100

    print(
        f"{class_name}: "
        f"train={train} ({train_pct:.1f}%), "
        f"validation={validation} ({validation_pct:.1f}%), "
        f"test={test} ({test_pct:.1f}%)"
    )