from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from tqdm import tqdm

from pointnet.model.base_point_net import BasePointNet
from pointnet.structs import TrainingConfig
from pointnet.utils import MetaLogger


class ClassificationPointNet(nn.Module, MetaLogger):
    """
    A multi-class classification model that operates on point cloud data. This model is
    based on [PointNet](https://arxiv.org/pdf/1612.00593).
    """

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
        self.point_dimensions = point_dimensions
        self.num_classes = num_classes
        self.dropout = dropout

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

    def save(self, file: Path):
        """
        Saves the model to the specified location.

        Parameters
        ----------
        file:
            File path to save the model to. By convention this file should end in `.pt`.
        """
        torch.save(
            {
                "model": self.state_dict(),
                "num_points": self.num_points,
                "point_dimensions": self.point_dimensions,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
            },
            file,
        )

    @classmethod
    def load(cls, file: Path) -> "ClassificationPointNet":
        """
        Load the model from the specified location.

        Parameters
        ----------
        file:
            File path to save the model to. By convention this file should end in `.pt`.

        Returns
        -------
        model:
            The loaded in model.
        """
        # Load the previous state
        model_state = torch.load(file, weights_only=True)

        # Initialise the model
        model = cls(
            model_state["num_points"],
            model_state["point_dimensions"],
            model_state["num_classes"],
            model_state["dropout"],
        )
        model.load_state_dict(model_state["model"])
        model.eval()

        return model

    def forward(self, points: Tensor) -> Tensor:
        """
        Parameters
        ----------
        points:
            Point cloud data stored in a tensor of shape `(batch_size, channels,
            sequence_length)`.

        Returns
        -------
        logits:
            The classification logits, with shape `(batch_size, num_classes)`.
        """
        features = self.backbone.forward(points)
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
        train_loader = DataLoader(config.train_dataset, config.batch_size, shuffle=True)

        # Create a validation loader
        valid_loader = DataLoader(config.valid_dataset, config.batch_size, shuffle=True)

        return train_loader, valid_loader

    def fit(self, config: TrainingConfig):
        """
        Trains the model.

        Parameters
        ----------
        config:
            A configuration object that contains parameters describing how the model
            should be trained.
        """
        self.logger.info(f"Training model: {config.run_name}")

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
        min_valid_loss = np.inf
        for epoch in range(config.epochs):
            print()
            self.logger.info(f"Epoch: {epoch}")

            # Run the training epoch
            train_loss = 0
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
                train_loss += loss.item()

            # Update the learning rate scheduler
            scheduler.step()
            tqdm_iterator.close()

            # Run over the validation dataset
            self.eval()
            valid_loss = 0
            correct, incorrect = 0, 0
            with torch.no_grad():
                tqdm_iterator = tqdm(valid_loader, ncols=88)
                tqdm_iterator.set_description_str("Valid")
                for clouds, labels in tqdm_iterator:
                    logits = self.forward(clouds)

                    # Compute loss
                    loss = cross_entropy(logits, labels)
                    valid_loss += loss.item()

                    # Calulate correct and incorrect classifications
                    predictions = logits.argmax(dim=1)
                    matches = predictions == labels
                    correct += matches.sum()
                    incorrect += len(matches) - matches.sum()

                tqdm_iterator.close()

                # Display metrics
                batch_train_loss = train_loss / len(train_loader)
                batch_valid_loss = valid_loss / len(valid_loader)
                self.logger.info(f"Train loss: {batch_train_loss:.4f}")
                self.logger.info(f"Valid loss: {batch_valid_loss:.4f}")
                self.logger.info(f"Valid accuracy: {correct / (correct + incorrect)}")

                # Save the model if it is the best
                if batch_valid_loss < min_valid_loss:
                    file = config.run_dir / "best.pt"
                    self.save(file)
