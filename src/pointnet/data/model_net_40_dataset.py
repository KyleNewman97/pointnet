from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import trimesh
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset

from pointnet.structs import Split
from pointnet.utils import MetaLogger


class Sample(BaseModel):
    object_id: str
    class_name: str
    split: Split
    object_path: Path


class ModelNet40Dataset(Dataset, MetaLogger):
    METADATA_CSV_NAME = "metadata_modelnet40.csv"

    def __init__(
        self,
        dataset_dir: Path,
        split: Split,
        desired_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.desired_points = desired_points
        self.device = device
        self.dtype = dtype

        # Read the dataset
        metadata_csv = self.dataset_dir / self.METADATA_CSV_NAME
        samples, class_name_to_id = ModelNet40Dataset.read_dataset_csv(metadata_csv)
        self.samples = samples[split]
        self.class_name_to_id = class_name_to_id

    @staticmethod
    def read_dataset_csv(
        file: Path,
    ) -> tuple[dict[Split, list[Sample]], dict[str, int]]:
        """
        Read the dataset CSV for the ModelNet40 dataset.

        Parameters
        ----------
        file:
            Path to the dataset's metadata CSV file.

        Returns
        -------
        samples:
            A dictionary mapping dataset split to the list of samples.

        class_name_to_id:
            A mapping from class name to class ID.
        """
        # Read in the CSV content
        with open(file, "r") as fp:
            lines = fp.read().split("\n")[1:]

        # Parse the individual samples
        samples: dict[Split, list[Sample]] = defaultdict(list)
        class_names = set()
        for line in lines:
            elements = line.split(",")
            if len(elements) != 4:
                continue

            class_name = elements[1]
            split = Split(elements[2])
            samples[split].append(
                Sample(
                    object_id=elements[0],
                    class_name=class_name,
                    split=split,
                    object_path=file.parent / "ModelNet40" / elements[3],
                )
            )
            class_names.add(class_name)

        # Create a mapping from ID to class name
        class_names = sorted(class_names)
        class_name_to_id = {c: i for i, c in enumerate(class_names)}

        return samples, class_name_to_id

    @staticmethod
    def read_off(
        file: Path, desired_points: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """
        Read in an OFF file's vertices.

        Parameters
        ----------
        file:
            An OFF file to read int.

        desired_points:
            The desired number of points an object has to contain. If it contains more
            points than this then it is downsampled to contained `desired_points`
            points. If it contains fewer, then points will be randomly duplicated until
            there are `desired_points` points.

        Returns
        -------
        points:
            The points read in from the OFF file. This should have a shape of:
                `(3, desired_points)`
        """

        mesh = trimesh.load_mesh(file)
        indices = np.random.randint(0, mesh.vertices.shape[0], (desired_points,))
        points = mesh.vertices[indices, :]

        points = torch.tensor(points, device=device, dtype=dtype)
        return points.permute((1, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        points = self.read_off(
            sample.object_path, self.desired_points, self.device, self.dtype
        )
        class_id = self.class_name_to_id[sample.class_name]

        return points, torch.tensor(class_id, device=self.device)
