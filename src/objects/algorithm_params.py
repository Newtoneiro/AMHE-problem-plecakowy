import numpy as np
from dataclasses import dataclass


@dataclass
class AlgorithmParams:
    """Class for parametrizing algotithms."""
    dataset: np.array
    population_size: int
    num_best: int
    learning_rate: float
    epochs: int
    backpack_capacity: float

    @property
    def genome_length(self):
        return self.dataset.shape[0]
