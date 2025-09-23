import torch
from torch import Tensor, nn, optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

from pointnet.data import MNIST3DDataset
from pointnet.model.base_point_net import BasePointNet
from pointnet.structs import TrainingConfig
from pointnet.utils import MetaLogger


class ClassificationPointNet(nn.Module, MetaLogger):
    def __init__(
        self,
        num_points: int,
        point_dimensions: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        nn.Module.__init__(self)
        MetaLogger.__init__(self)

        self.num_points = num_points

        self.backbone = BasePointNet(point_dimensions)
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Tensor data of shape `(batch_size, channels, sequence_length)`.

        Returns
        -------
        logits:
            The classification logits, with shape `(batch_size, num_classes)`.
        """
        features = self.backbone.forward(x)
        return self.head(features)

    def _create_data_loaders(
        self, config: TrainingConfig
    ) -> tuple[DataLoader, DataLoader]:
        """
        Creates training and validation dataset loaders.

        Parameters
        ----------
        config:
            The training configuration being used by the model.


        Returns
        -------
        train_loader:
            Training dataset loader.

        valid_loader:
            Validation dataset loader.
        """
        # Create a training loader
        train_mnist = MNIST(config.dataset_dir, train=True, download=True)
        train_dataset = MNIST3DDataset(
            train_mnist, self.num_points, self.device, self.dtype
        )
        train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)

        # Create a validation loader
        valid_mnist = MNIST(config.dataset_dir, train=False, download=True)
        valid_dataset = MNIST3DDataset(
            valid_mnist, self.num_points, self.device, self.dtype
        )
        valid_loader = DataLoader(valid_dataset, config.batch_size, shuffle=True)

        return train_loader, valid_loader

    def fit(self, config: TrainingConfig):
        # Create data loaders
        train_loader, valid_loader = self._create_data_loaders(config)

        # Create the optimiser and learning rate scheduler
        optimiser = optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )
        scheduler = optim.lr_scheduler.StepLR(optimiser, 20, 0.5)

        # Run over the training dataset
        for epoch in range(config.epochs):
            self.logger.info(f"Epoch: {epoch}")

            # Run the training epoch
            clouds: Tensor
            labels: Tensor
            self.train()
            tqdm_iterator = tqdm(train_loader, ncols=88)
            tqdm_iterator.set_description_str("Train")
            for clouds, labels in tqdm_iterator:
                # Zero the gradients - this is required on each mini-batch
                optimiser.zero_grad()

                # Make predictions for this batch and calculate the loss
                logits = self.forward(clouds)
                loss = cross_entropy(logits, labels)

                # Backprop
                loss.backward()
                optimiser.step()

                tqdm_iterator.set_postfix_str(f"Loss={loss.item():.4}")

            # Update the learning rate scheduler
            scheduler.step()

            tqdm_iterator.close()


if __name__ == "__main__":
    from pathlib import Path

    # Initialise the model
    device = torch.device("cuda:0")
    dtype = torch.float32
    net = ClassificationPointNet(200, 3, 10)
    net = net.to(device=device, dtype=dtype)

    config = TrainingConfig(dataset_dir=Path("data"))
    net.fit(config)
