from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


DATASET_PATH = Path(
    "/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild"
)

SPLIT_FILE = DATASET_PATH / "plantwild_split.csv"


class PlantWildDataset(Dataset):

    def __init__(self, records, class_to_idx, transform=None):
        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        image = Image.open(
            record["image_path"]
        ).convert("RGB")

        label = self.class_to_idx[
            record["class"]
        ]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_dataloaders():

    records = []

    with open(SPLIT_FILE, "r") as f:

        reader = csv.DictReader(f)

        for row in reader:
            records.append({
                "image_path": row["image_path"],
                "class": row["class"],
                "split": row["split"]
            })

    classes = sorted(
        set(record["class"] for record in records)
    )

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_records = [
        record for record in records
        if record["split"] == "train"
    ]

    validation_records = [
        record for record in records
        if record["split"] == "validation"
    ]

    test_records = [
        record for record in records
        if record["split"] == "test"
    ]

    train_dataset = PlantWildDataset(
        train_records,
        class_to_idx,
        transform=train_transform
    )

    validation_dataset = PlantWildDataset(
        validation_records,
        class_to_idx,
        transform=evaluation_transform
    )

    test_dataset = PlantWildDataset(
        test_records,
        class_to_idx,
        transform=evaluation_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        classes
    )


if __name__ == "__main__":

    print("=== PlantWild Pipeline ===")

    (
        train_loader,
        validation_loader,
        test_loader,
        classes
    ) = create_dataloaders()

    print(f"Number of classes: {len(classes)}")

    print("\n=== Dataset Sizes ===")

    print(
        f"Training:   "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation: "
        f"{len(validation_loader.dataset)}"
    )

    print(
        f"Test:       "
        f"{len(test_loader.dataset)}"
    )

    images, labels = next(
        iter(train_loader)
    )

    print("\n=== First Training Batch ===")

    print(
        f"Image batch shape: "
        f"{images.shape}"
    )

    print(
        f"Label batch shape: "
        f"{labels.shape}"
    )