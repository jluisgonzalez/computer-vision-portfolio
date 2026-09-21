from pathlib import Path
import csv
from sklearn.model_selection import train_test_split

# PlantWild dataset location
dataset_path = Path(
    "/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild"
)
images_path = dataset_path / "images"

image_extensions = {".jpg", ".jpeg", ".png"}

# Collect image paths and their class labels
records = []

for class_dir in sorted(images_path.iterdir()):
    if class_dir.is_dir():
        for file in class_dir.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                records.append((str(file), class_dir.name))

print("=== PlantWild Dataset ===")
print(f"Total images: {len(records)}")
print(f"Total classes: {len(set(record[1] for record in records))}")

# Get labels for stratification
labels = [record[1] for record in records]

# First split: 80% training, 20% temporary
train_records, temp_records = train_test_split(
    records,
    test_size=0.20,
    stratify=labels,
    random_state=42
)

# Second split: divide the remaining 20% equally
temp_labels = [record[1] for record in temp_records]

validation_records, test_records = train_test_split(
    temp_records,
    test_size=0.50,
    stratify=temp_labels,
    random_state=42
)

print("\n=== Split Results ===")
print(f"Training:    {len(train_records)} images")
print(f"Validation:  {len(validation_records)} images")
print(f"Test:        {len(test_records)} images")

# Save the split information
output_file = dataset_path / "plantwild_split.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "class", "split"])

    for image_path, class_name in train_records:
        writer.writerow([image_path, class_name, "train"])

    for image_path, class_name in validation_records:
        writer.writerow([image_path, class_name, "validation"])

    for image_path, class_name in test_records:
        writer.writerow([image_path, class_name, "test"])

print(f"\nSplit file saved to:")
print(output_file)