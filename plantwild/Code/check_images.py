from pathlib import Path
from PIL import Image

# Find the PlantWild project folder
project_path = Path(__file__).resolve().parent.parent

# Location of the images
images_path = project_path / "images"

# Image extensions we want to check
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

# Find all image files
image_files = [
    file for file in images_path.rglob("*")
    if file.suffix.lower() in image_extensions
]

print("Image files found:", len(image_files))

# Check whether images can be opened
readable = 0
unreadable = 0

dimensions = {}
modes = {}

for image_file in image_files:
    try:
        with Image.open(image_file) as image:
            image.verify()

        # Open again after verify()
        with Image.open(image_file) as image:
            size = image.size
            mode = image.mode

        readable += 1

        dimensions[size] = dimensions.get(size, 0) + 1
        modes[mode] = modes.get(mode, 0) + 1

    except Exception:
        unreadable += 1

print("Readable images:", readable)
print("Unreadable images:", unreadable)

print("\nImage dimensions:")
for size, count in sorted(dimensions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(size, ":", count)

print("\nImage modes:")
for mode, count in modes.items():
    print(mode, ":", count)