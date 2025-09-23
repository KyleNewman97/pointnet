import pytest
import torch
from pathlib import Path
from torchvision.datasets import MNIST

from pointnet.data import MNIST3DDataset


class TestMNIST3DDataset:
    @pytest.fixture
    def base_dataset(self) -> MNIST:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        return MNIST(root=data_dir, train=True, download=True)

    def test_init(self, base_dataset: MNIST):
        """
        Test we can initialise an MNIST3DDataset object.
        """
        dataset = MNIST3DDataset(base_dataset, 200, torch.device("cpu"), torch.float32)
        assert isinstance(dataset, MNIST3DDataset)

    def test_len(self, base_dataset: MNIST):
        """
        Test we can get the correct size of the dataset.
        """
        dataset = MNIST3DDataset(base_dataset, 200, torch.device("cpu"), torch.float32)
        assert len(dataset) == len(base_dataset)

    def test_getitem(self, base_dataset: MNIST):
        """
        Test we can get a sample from the dataset.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        num_points = 200
        dataset = MNIST3DDataset(base_dataset, num_points, device, dtype)
        points, label = dataset[0]

        assert isinstance(points, torch.Tensor)
        assert points.shape == (num_points, 3)
        assert isinstance(label, int)
