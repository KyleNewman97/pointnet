import torch
from torch import Tensor, nn
from torch.nn.functional import relu

from pointnet.model.transform_net import TransformNet


class BasePointNet(nn.Module):
    def __init__(self, point_dimensions: int):
        nn.Module.__init__(self)

        self.input_transform = TransformNet(point_dimensions, 3)

        self.conv1 = nn.Conv1d(point_dimensions, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.feature_transform = TransformNet(64, 64)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Tensor data of shape `(batch_size, channels, sequence_length)`.
        """
        num_points = x.shape[2]

        # Apply a 3x3 transform to the input
        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)

        # Pass through first MLP
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))

        # Apply a transform to features
        feature_transform = self.feature_transform(x)
        x = torch.bmm(feature_transform, x)

        # Pass through second MLP
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))
        x = relu(self.bn5(self.conv5(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.squeeze(2)

        return x
