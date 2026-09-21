from pathlib import Path
from PIL import Image
from torchvision import transforms

# Find the PlantWild project folder
project_path = Path(__file__).resolve().parent.parent

# Find one image from the dataset
image_path = next((project_path / "images").rglob("*.jpg"))

# Open the image
image = Image.open(image_path)

print("Original size:", image.size)

# Create the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Apply the transformation
tensor_image = transform(image)

print("Tensor shape:", tensor_image.shape)
print("Minimum pixel value:", tensor_image.min().item())
print("Maximum pixel value:", tensor_image.max().item())