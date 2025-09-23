import torch
from torch import Tensor
from pointnet.model.transform_net import TransformNet


class TestTransformNet:
    def test_init(self):
        """
        Test that we can initialise a TransformNet network.
        """
        net = TransformNet(3, 3)
        assert isinstance(net, TransformNet)

    def test_forward(self):
        """
        Test that we can perform inference.
        """
        # Create dummy input
        device = torch.device("cpu")
        dtype = torch.float32
        x = torch.rand((8, 3, 200), device=device, dtype=dtype)

        # Try to run inference
        net = TransformNet(3, 3).to(device=device, dtype=dtype)
        output = net.forward(x)

        # Check the output
        assert isinstance(output, Tensor)
        assert output.device == device
        assert output.dtype == dtype
        assert output.shape == (x.shape[0], net.output_dim, net.output_dim)
