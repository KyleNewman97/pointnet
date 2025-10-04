from enum import Enum


class Split(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
