import torch
from torch import Tensor

from pointnet.model.base_point_net import BasePointNet


class TestBasePointNet:
    def test_init(self):
        """
        Test that we can initialise a BasePointNet network.
        """
        net = BasePointNet(3)
        assert isinstance(net, BasePointNet)

    def test_forward(self):
        """
        Test that we can perform inference.
        """
        # Create dummy input
        device = torch.device("cpu")
        dtype = torch.float32
        x = torch.rand((8, 3, 200), device=device, dtype=dtype)

        # Try to run inference
        net = BasePointNet(3).to(device=device, dtype=dtype)
        output = net.forward(x)

        # Check the output
        assert isinstance(output, Tensor)
        assert output.device == device
        assert output.dtype == dtype
        assert output.shape == (x.shape[0], 1024)
