from pathlib import Path

import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from pipeline import create_dataloaders
from models import create_resnet18, create_mobilenet_v3_small


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = Path(
    "/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild"
)

RESULTS_PATH = DATASET_PATH / "Results"

RESNET_CHECKPOINT = RESULTS_PATH / "resnet18_best.pth"
MOBILENET_CHECKPOINT = RESULTS_PATH / "mobilenet_v3_small_best.pth"

RESULTS_PATH.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=== PlantWild Model Evaluation ===")
print(f"Device: {device}")


# --------------------------------------------------
# Load test data
# --------------------------------------------------

(
    train_loader,
    validation_loader,
    test_loader,
    classes
) = create_dataloaders()

print(f"Number of classes: {len(classes)}")
print(f"Test images: {len(test_loader.dataset)}")


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(model, model_name, checkpoint_path):

    print("\n" + "=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_images = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            # Save first images for prediction examples
            if len(all_images) < 12:
                remaining = 12 - len(all_images)

                for image in images[:remaining].cpu():
                    all_images.append(image)

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    print("\n=== Test Results ===")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-score:  {f1:.4f}")

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    report_path = (
        RESULTS_PATH /
        f"{model_name}_classification_report.txt"
    )

    with open(report_path, "w") as file:
        file.write(report)

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(classes)))
    )

    cm_path = (
        RESULTS_PATH /
        f"{model_name}_confusion_matrix.png"
    )

    plt.figure(figsize=(22, 20))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(
        tick_marks,
        classes,
        rotation=90,
        fontsize=5
    )

    plt.yticks(
        tick_marks,
        classes,
        fontsize=5
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        cm_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Example predictions
    # --------------------------------------------------

    # Denormalization values
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    plt.figure(
        figsize=(16, 12)
    )

    for i in range(len(all_images)):

        image = (
            all_images[i] * std
        ) + mean

        image = torch.clamp(
            image,
            0,
            1
        )

        plt.subplot(3, 4, i + 1)

        plt.imshow(
            image.permute(1, 2, 0)
        )

        true_label = classes[
            all_labels[i]
        ]

        predicted_label = classes[
            all_predictions[i]
        ]

        plt.title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}",
            fontsize=8
        )

        plt.axis("off")

    plt.tight_layout()

    prediction_path = (
        RESULTS_PATH /
        f"{model_name}_example_predictions.png"
    )

    plt.savefig(
        prediction_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Return metrics
    # --------------------------------------------------

    return {
        "Model": model_name,
        "Test Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    }


# --------------------------------------------------
# Evaluate ResNet-18
# --------------------------------------------------

resnet_model = create_resnet18()

resnet_results = evaluate_model(
    resnet_model,
    "ResNet-18",
    RESNET_CHECKPOINT
)


# --------------------------------------------------
# Evaluate MobileNetV3-Small
# --------------------------------------------------

mobilenet_model = create_mobilenet_v3_small()

mobilenet_results = evaluate_model(
    mobilenet_model,
    "MobileNetV3-Small",
    MOBILENET_CHECKPOINT
)


# --------------------------------------------------
# Save summary metrics
# --------------------------------------------------

results = [
    resnet_results,
    mobilenet_results
]

metrics_path = RESULTS_PATH / "test_metrics.csv"

with open(
    metrics_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Model",
            "Test Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


# --------------------------------------------------
# Print final comparison
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST COMPARISON")
print("=" * 60)

for result in results:

    print(
        f"\n{result['Model']}"
    )

    print(
        f"Accuracy:        "
        f"{result['Test Accuracy']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{result['Macro Precision']:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{result['Macro Recall']:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{result['Macro F1']:.4f}"
    )

print("\n=== Evaluation Complete ===")
print(f"Results saved to: {RESULTS_PATH}")