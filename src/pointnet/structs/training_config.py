from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """
    The training configuration to use when training a model.
    """

    dataset_dir: Path
    root_dir: Path
    run_name: str = Field(default_factory=lambda: f"{uuid4()}")
    epochs: int = Field(default=100)
    batch_size: int = Field(default=32)

    # Optimiser parameters
    learning_rate: float = Field(default=0.001)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)

    @property
    def run_dir(self) -> Path:
        folder = self.root_dir / self.run_name
        folder.mkdir(exist_ok=True)

        return folder
