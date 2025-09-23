import torch
from torch import nn, Tensor
from torch.nn.functional import relu


class TransformNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        nn.Module.__init__(self)

        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_dim * output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Tensor data of shape `(batch_size, channels, sequence_length)`.
        """
        num_points = x.shape[2]

        # Apply MLP
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))

        # Max pool across points - removing coupling on the number of points
        x = nn.MaxPool1d(num_points)(x)
        x = x.squeeze(2)

        # Apply fully connected layers
        x = self.fc1(x)
        x = relu(self.bn4(x))
        x = relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Adjust identity matrix with output
        identity_matrix = torch.eye(self.output_dim, device=x.device, dtype=x.dtype)
        x = identity_matrix + x.view(-1, self.output_dim, self.output_dim)

        return x
