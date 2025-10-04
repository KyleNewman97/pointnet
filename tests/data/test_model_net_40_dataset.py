import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator
from uuid import uuid4

import numpy as np
import pytest
import torch
from torch import Tensor

from pointnet.data import ModelNet40Dataset
from pointnet.structs import Split


class TestModelNet40Dataset:
    @pytest.fixture
    def dataset_dir(self) -> Generator[Path, None, None]:
        # Create the dataset directory
        temp_dir = TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        yield temp_path

        temp_dir.cleanup()

    def _create_dataset(self, dataset_dir: Path, num_samples: int) -> Path:
        """
        Populate the dataset. Creating the dataset CSV and the corresponding samples.
        """
        csv_file = dataset_dir / ModelNet40Dataset.METADATA_CSV_NAME

        classes = ["airplane", "boat", "car"]
        splits = list(Split)
        with open(csv_file, "w") as fp:
            fp.write("object_id,class,split,object_path\n")

            for _ in range(num_samples):
                guid = f"{uuid4()}"
                class_name = random.choice(classes)
                split = random.choice(splits)
                object_id = f"{class_name}_{guid}"
                file = f"{class_name}/{split.value}/{object_id}.off"

                fp.write(f"{object_id},{class_name},{split.value},{file}\n")

        return csv_file

    def _create_off(self, directory: Path) -> tuple[Path, int]:
        """
        Create an OFF file in the specified directory. The stem of the file will be a
        random uuid.

        Parameters
        ----------
        directory:
            The directory to create the OFF file in.

        Returns
        -------
        file:
            The path to the created OFF file.

        num_points:
            The number of points saved to the file.
        """
        file = directory / f"{uuid4()}.off"
        with open(file, "w") as fp:
            # Write header
            fp.write("OFF\n")

            # Write counts
            num_points = np.random.randint(0, 1000)
            num_faces = np.random.randint(0, 1000)
            num_edges = np.random.randint(0, 1000)
            fp.write(f"{num_points} {num_faces} {num_edges}\n")

            # Write verticies
            for _ in range(num_points):
                points = [np.random.random() for _ in range(3)]
                fp.write(f"{points[0]} {points[1]} {points[2]}\n")

        return file, num_points

    def test_read_dataset_csv(self, dataset_dir: Path):
        """
        Test that we can read a ModelNet40 metadata CSV.
        """
        desired_num_samples = 200
        csv_file = self._create_dataset(dataset_dir, desired_num_samples)
        samples, class_name_to_id = ModelNet40Dataset.read_dataset_csv(csv_file)

        # Check returned samples
        assert isinstance(samples, dict)
        num_samples = 0
        for split_samples in samples.values():
            num_samples += len(split_samples)
        assert num_samples == desired_num_samples

        # Check the class name mapping
        assert isinstance(class_name_to_id, dict)

    def test_read_off(self, dataset_dir: Path):
        """
        Test that we can correctly read an OFF file.
        """
        device = torch.device("cpu")
        dtype = torch.float32
        off_file, num_points = self._create_off(dataset_dir)
        points = ModelNet40Dataset.read_off(off_file, num_points, device, dtype)

        assert isinstance(points, Tensor)
        assert points.shape == (3, num_points)
