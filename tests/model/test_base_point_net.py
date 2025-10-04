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
        global_feats, trans_feats = net.forward(x)

        # Check the output
        assert isinstance(global_feats, Tensor)
        assert global_feats.device == device
        assert global_feats.dtype == dtype
        assert global_feats.shape == (x.shape[0], 1024)

        assert isinstance(trans_feats, Tensor)
        assert trans_feats.device == device
        assert trans_feats.dtype == dtype
        assert trans_feats.shape == (x.shape[0], 64, x.shape[2])
