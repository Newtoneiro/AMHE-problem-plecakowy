import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


class DistributionKind(ABC):
    """Base class for Distribution"""

    @abstractmethod
    def generate_data(self, no_samples: int) -> tuple[np.ndarray]:
        """Generate dataset with weights and values"""
        pass


@dataclass
class NormalDistribution(DistributionKind):
    """Class for Normal/correlated data distribution"""
    mean_weight: float
    variance_weight: float
    mean_value: float
    variance_value: float

    def generate_data(self, no_samples: int) -> tuple[np.ndarray]:
        weights = np.random.normal(
            loc=self.mean_weight,
            scale=np.sqrt(self.variance_weight),
            size=no_samples
        )
        values = np.random.normal(
            loc=self.mean_value,
            scale=np.sqrt(self.variance_value),
            size=no_samples
        )

        return weights, values


@dataclass
class UniformDistribution(DistributionKind):
    """Class for Uniform/unrelated data distribution"""
    min_weight: float = 0.01
    max_weight: float = 1_000_000
    min_value: float = 0.01
    max_value: float = 1_000_000

    def generate_data(self, no_samples: int) -> tuple[np.ndarray]:
        weights = np.random.uniform(
            low=self.min_weight,
            high=self.max_weight,
            size=no_samples
        )
        values = np.random.uniform(
            low=self.min_value,
            high=self.max_value,
            size=no_samples
        )

        return weights, values
