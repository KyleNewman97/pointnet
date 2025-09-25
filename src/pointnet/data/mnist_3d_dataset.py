import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNIST3DDataset(Dataset):
    def __init__(
        self,
        dataset: MNIST,
        num_points: int,
        device: torch.device,
        dtype: torch.dtype,
        transforms: list[nn.Module] = [],  # trunk-ignore(ruff/B006)
    ):
        self.dataset = dataset
        self.num_points = num_points
        self.transforms = transforms
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]

        # Find all white pixels
        np_image = np.asarray(image)
        points = np.argwhere(np_image > 127).astype(np.float32)

        # Ensure we have a fixed number of points
        if self.num_points < points.shape[0]:
            # If we have too many points then downsample them
            samples = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[samples, :]
        elif points.shape[0] < self.num_points:
            # If we have too few points then upsample them
            samples = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[samples, :]

        points = points.astype(np.float32)

        # Add the z dimension
        noise = np.random.normal(0, 0.05, points.shape[0])
        noise = np.expand_dims(noise, 1)
        points = np.hstack([points, noise]).astype(np.float32)

        # Convert to a tensor
        points_tensor = torch.tensor(points, device=self.device, dtype=self.dtype)
        points_tensor = points_tensor.permute(1, 0)

        # Center points
        points_tensor[0, :] -= np_image.shape[0] // 2
        points_tensor[1, :] -= np_image.shape[1] // 2

        # Apply transformations
        for transform in self.transforms:
            points_tensor = transform.forward(points_tensor)

        return points_tensor, torch.tensor(label, device=self.device)
