from pathlib import Path

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """
    The training configuration to use when training a model.
    """

    dataset_dir: Path
    epochs: int = Field(default=100)
    batch_size: int = Field(default=32)

    # Optimiser parameters
    learning_rate: float = Field(default=0.001)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
