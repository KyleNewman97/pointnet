from pathlib import Path

import kagglehub
import torch

from pointnet.data import ModelNet40Dataset
from pointnet.model import ClassificationPointNet
from pointnet.structs import Split, TrainingConfig

if __name__ == "__main__":
    # Define parameters
    num_points = 2000
    point_dims = 3
    num_classes = 40
    device = torch.device("cuda:0")
    dtype = torch.float32
    path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")
    dataset_dir = Path(path)

    # Create the training and validation datasets
    train_dataset = ModelNet40Dataset(
        dataset_dir=dataset_dir,
        split=Split.TRAIN,
        desired_points=num_points,
        device=device,
        dtype=dtype,
    )
    valid_dataset = ModelNet40Dataset(
        dataset_dir=dataset_dir,
        split=Split.TEST,
        desired_points=num_points,
        device=device,
        dtype=dtype,
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
