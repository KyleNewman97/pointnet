import torch
from torch import Tensor

from pointnet.data.random_rotation_transform import RandomRotationTransform


class TestRandomRotationTransform:
    def test_init(self):
        """
        Test we can initialise a random rotation transform.
        """
        transform = RandomRotationTransform()
        assert isinstance(transform, RandomRotationTransform)

    def test_forward(self):
        """
        Test we can randomly rotate some points.
        """
        device = torch.device("cpu")
        dtype = torch.float32
        transform = RandomRotationTransform()

        points = torch.randn((3, 20), device=device, dtype=dtype)
        results = transform.forward(points)

        assert isinstance(results, Tensor)
        assert results.shape == points.shape
