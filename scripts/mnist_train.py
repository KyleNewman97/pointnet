from pathlib import Path

import torch
from torchvision.datasets import MNIST

from pointnet.data import MNIST3DDataset, RandomRotationTransform
from pointnet.model import ClassificationPointNet
from pointnet.structs import TrainingConfig

if __name__ == "__main__":
    # Define parameters
    num_points = 200
    point_dims = 3
    num_classes = 10
    device = torch.device("cuda:0")
    dtype = torch.float32
    dataset_dir = Path("data")

    # Create the training and validation datasets
    train_mnist = MNIST(dataset_dir, train=True, download=True)
    train_dataset = MNIST3DDataset(
        train_mnist,
        num_points,
        device,
        dtype,
        [RandomRotationTransform()],
    )
    valid_mnist = MNIST(dataset_dir, train=False, download=True)
    valid_dataset = MNIST3DDataset(
        valid_mnist,
        num_points,
        device,
        dtype,
    )

    # Initialise the model
    net = ClassificationPointNet(num_points, point_dims, num_classes)
    net = net.to(device=device, dtype=dtype)

    # Train the model
    config = TrainingConfig(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        root_dir=Path("runs"),
    )
    net.fit(config)
