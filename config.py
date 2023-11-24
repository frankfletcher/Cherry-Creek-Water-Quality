from dataclasses import dataclass
from pathlib import Path

# import cupy as cp
import numpy as np


# create a dataclass to hold ml configuration parameters. It should be called Config
@dataclass
class Config:
    data_path = Path("./data")
    train_fname = "train.csv"
    test_fname = None

    random_state = 44725  # dynamic for creating mulitple models
    split_random_state = 96862  # keep constant for reproducibility

    def __init__(self, **kwargs):
        super().__init__()
        self.set_attrs(**kwargs)

        self.rng = np.random.default_rng(self.random_state)

    def set_attrs(self, **kwargs):
        """Sets attributes for the Config class.

        Args:
            **kwargs: The attributes to set.

        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_attr(self, attr, value):
        """Sets an attribute for the Config class.

        Args:
            attr: The attribute to set.
            value: The value to set the attribute to.

        """
        setattr(self, attr, value)
