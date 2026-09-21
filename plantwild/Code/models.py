import torch
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


NUMBER_OF_CLASSES = 89


def create_resnet18():
    """
    Create an ImageNet-pretrained ResNet-18
    adapted to the 89 PlantWild classes.
    """

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUMBER_OF_CLASSES
    )

    return model


def create_mobilenet_v3_small():
    """
    Create an ImageNet-pretrained MobileNetV3-Small
    adapted to the 89 PlantWild classes.
    """

    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT
    )

    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features,
        NUMBER_OF_CLASSES
    )

    return model


def count_parameters(model):
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


if __name__ == "__main__":

    print("=== Loading pretrained models ===")

    resnet_model = create_resnet18()
    mobilenet_model = create_mobilenet_v3_small()

    print("\n=== Model Classifiers ===")

    print(
        f"ResNet-18 final layer: "
        f"{resnet_model.fc}"
    )

    print(
        f"MobileNetV3-Small final layer: "
        f"{mobilenet_model.classifier[-1]}"
    )

    resnet_total, resnet_trainable = count_parameters(
        resnet_model
    )

    mobilenet_total, mobilenet_trainable = count_parameters(
        mobilenet_model
    )

    print("\n=== Parameter Counts ===")

    print(
        f"ResNet-18: "
        f"{resnet_total:,} total / "
        f"{resnet_trainable:,} trainable"
    )

    print(
        f"MobileNetV3-Small: "
        f"{mobilenet_total:,} total / "
        f"{mobilenet_trainable:,} trainable"
    )