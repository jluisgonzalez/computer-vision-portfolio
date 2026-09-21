from pathlib import Path

# Find the PlantWild project folder
project_path = Path(__file__).resolve().parent.parent

# Location of the split file
split_file = project_path / "trainval.txt"

# Counters
train_count = 0
val_count = 0
test_count = 0

# Read the split file
with open(split_file, "r") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        parts = line.split("=")
        split = parts[-1]

        if split == "1":
            train_count += 1
        elif split == "2":
            val_count += 1
        elif split == "0":
            test_count += 1

print("Training images:", train_count)
print("Validation images:", val_count)
print("Test images:", test_count)
print("Total images:", train_count + val_count + test_count)