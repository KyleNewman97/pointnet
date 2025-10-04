from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor

from pointnet.model import SegmentationPointNet


class TestSegmentationPointNet:
    def test_init(self):
        """
        Test that we can initialise a SegmentationPointNet network.
        """
        net = SegmentationPointNet(200, 3, 10)
        assert isinstance(net, SegmentationPointNet)

    def test_device(self):
        """
        Test the device property correctly reports the device.
        """
        device = torch.device("cpu")
        net = SegmentationPointNet(200, 3, 10).to(device=device)
        assert net.device == device

    def test_dtype(self):
        """
        Test the dtype property correctly reports the dtype.
        """
        dtype = torch.float32
        net = SegmentationPointNet(200, 3, 10).to(dtype=dtype)
        assert net.dtype == dtype

    def test_save_load(self):
        """
        Test we can correctly save and load the model.
        """
        device = torch.device("cpu")
        dtype = torch.float32

        model = SegmentationPointNet(200, 3, 10).to(device=device, dtype=dtype)
        with TemporaryDirectory() as temp_dir:
            # Save the model
            file = Path(temp_dir) / "model.pt"
            model.save(file)

            # Try to load the model
            loaded_model = SegmentationPointNet.load(file)

        assert isinstance(loaded_model, SegmentationPointNet)

    def test_forward(self):
        """
        Test that we can perform inference.
        """
        # Create dummy input
        device = torch.device("cpu")
        dtype = torch.float32
        num_points = 200
        x = torch.rand((8, 3, num_points), device=device, dtype=dtype)

        # Try to run inference
        num_classes = 10
        net = SegmentationPointNet(num_points, 3, num_classes).to(
            device=device, dtype=dtype
        )
        output = net.forward(x)

        # Check the output
        assert isinstance(output, Tensor)
        assert output.device == device
        assert output.dtype == dtype
        assert output.shape == (x.shape[0], num_classes, num_points)
