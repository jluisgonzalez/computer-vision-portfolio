from pathlib import Path
import csv

import torch


def calculate_class_weights(split_file):
    """
    Calculate inverse-frequency class weights using
    training data only.

    Returns:
        classes: sorted list of class names
        weights_tensor: PyTorch tensor containing one weight per class
    """

    training_labels = []

    with open(split_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["split"] == "train":
                training_labels.append(row["class"])

    classes = sorted(set(training_labels))

    class_counts = {}

    for class_name in classes:
        class_counts[class_name] = training_labels.count(class_name)

    total_training_images = len(training_labels)
    number_of_classes = len(classes)

    class_weights = {}

    for class_name in classes:
        count = class_counts[class_name]

        weight = total_training_images / (
            number_of_classes * count
        )

        class_weights[class_name] = weight

    weights_tensor = torch.tensor(
        [class_weights[class_name] for class_name in classes],
        dtype=torch.float32
    )

    return classes, weights_tensor


if __name__ == "__main__":

    dataset_path = Path(
        "/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild"
    )

    split_file = dataset_path / "plantwild_split.csv"

    classes, weights_tensor = calculate_class_weights(split_file)

    print("=== Class Imbalance Strategy ===")
    print(f"Training classes: {len(classes)}")
    print(f"Weight tensor shape: {weights_tensor.shape}")

    print("\n=== Smallest Class Weights ===")

    # Show the 10 largest weights,
    # which correspond to the rarest classes.
    sorted_weights = sorted(
        zip(classes, weights_tensor.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    for class_name, weight in sorted_weights[:10]:
        print(f"{class_name}: weight {weight:.4f}")

    print("\n=== Largest Class Weights ===")

    # Show the 10 smallest weights,
    # which correspond to the most common classes.
    for class_name, weight in sorted_weights[-10:]:
        print(f"{class_name}: weight {weight:.4f}")

    print("\n=== PyTorch Weight Tensor ===")
    print(weights_tensor)