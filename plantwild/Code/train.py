from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim

from pipeline import create_dataloaders
from class_weights import calculate_class_weights
from models import (
    create_resnet18,
    create_mobilenet_v3_small
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_PATH = Path(
    "/Users/juanluis/Documents/Projects/computer-vision-portfolio/plantwild"
)

SPLIT_FILE = DATASET_PATH / "plantwild_split.csv"

RESULTS_PATH = DATASET_PATH / "Results"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Training Configuration ===")
print(f"Device: {device}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")


# --------------------------------------------------
# Data
# --------------------------------------------------

(
    train_loader,
    validation_loader,
    test_loader,
    classes
) = create_dataloaders()

print(f"Number of classes: {len(classes)}")


# --------------------------------------------------
# Class weights
# --------------------------------------------------

_, class_weights = calculate_class_weights(
    SPLIT_FILE
)

class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# --------------------------------------------------
# Training function
# --------------------------------------------------

def train_model(model, model_name):

    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_accuracy = 0.0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):

        # ------------------------------------------
        # Training
        # ------------------------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total


        # ------------------------------------------
        # Validation
        # ------------------------------------------

        model.eval()

        validation_loss_total = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                validation_loss_total += (
                    loss.item() * images.size(0)
                )

                _, predictions = torch.max(
                    outputs,
                    1
                )

                validation_total += labels.size(0)

                validation_correct += (
                    predictions == labels
                ).sum().item()

        validation_loss = (
            validation_loss_total /
            validation_total
        )

        validation_accuracy = (
            validation_correct /
            validation_total
        )


        # ------------------------------------------
        # Store results
        # ------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["validation_accuracy"].append(
            validation_accuracy
        )


        # ------------------------------------------
        # Save best model
        # ------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                validation_accuracy
            )

            model_file = (
                RESULTS_PATH /
                f"{model_name}_best.pth"
            )

            torch.save(
                model.state_dict(),
                model_file
            )


        # ------------------------------------------
        # Print epoch results
        # ------------------------------------------

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.4f}"
        )


    training_time = time.time() - start_time

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    return history, training_time


# --------------------------------------------------
# Train ResNet-18
# --------------------------------------------------

resnet_model = create_resnet18()

resnet_history, resnet_time = train_model(
    resnet_model,
    "resnet18"
)


# --------------------------------------------------
# Train MobileNetV3-Small
# --------------------------------------------------

mobilenet_model = create_mobilenet_v3_small()

mobilenet_history, mobilenet_time = train_model(
    mobilenet_model,
    "mobilenet_v3_small"
)


print("\n=== Training Complete ===")