from pathlib import Path

import torch

from pointnet.model import ClassificationPointNet
from pointnet.structs import TrainingConfig

if __name__ == "__main__":
    # Initialise the model
    device = torch.device("cuda:0")
    dtype = torch.float32
    net = ClassificationPointNet(200, 3, 10)
    net = net.to(device=device, dtype=dtype)

    config = TrainingConfig(dataset_dir=Path("data"), root_dir=Path("runs"))
    net.fit(config)
