from pathlib import Path

dataset_path = Path("/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild")
images_path = dataset_path / "images"

print("=== PlantWild Quick Audit ===")

# Count classes
with open(dataset_path / "classes.txt", "r") as f:
    classes = [line.strip() for line in f if line.strip()]

print(f"Classes: {len(classes)}")

# Count images
image_extensions = {".jpg", ".jpeg", ".png"}

images = [
    file for file in images_path.rglob("*")
    if file.is_file() and file.suffix.lower() in image_extensions
]

print(f"Images: {len(images)}")

from pathlib import Path
from collections import Counter

dataset_path = Path("/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild")
images_path = dataset_path / "images"

print("=== PlantWild Class Audit ===")

with open(dataset_path / "classes.txt", "r") as f:
    classes = [line.strip() for line in f if line.strip()]

print(f"Classes: {len(classes)}")

image_extensions = {".jpg", ".jpeg", ".png"}

images = [
    file for file in images_path.rglob("*")
    if file.is_file() and file.suffix.lower() in image_extensions
]

print(f"Images: {len(images)}")

counts = Counter(file.parent.name for file in images)

print("\nSmallest classes:")
for class_name, count in sorted(counts.items(), key=lambda x: x[1])[:10]:
    print(f"{class_name}: {count}")

print("\nLargest classes:")
for class_name, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{class_name}: {count}")