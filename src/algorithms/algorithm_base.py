import numpy as np
from abc import ABC, abstractmethod
from src.objects.algorithm_params import AlgorithmParams

PENALTY_FACTOR_BASE = 2.0


class AlgorithmBase(ABC):
    """Base implementation for concrete algorithms"""

    def __init__(self, params: AlgorithmParams):
        self.params = params

        self._calculate_penalty_factor()

    def __getattr__(self, name):
        """Simplifier for accessing parameters values"""
        if hasattr(self.params, name):
            return getattr(self.params, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def _calculate_penalty_factor(self):
        """Initialize dynamic penalty factor based on the weights and values"""
        self.penalty_factor = (
            PENALTY_FACTOR_BASE
            * (np.sum(self.dataset[:, 1]) / np.sum(self.dataset[:, 0]))
            * self.backpack_capacity
        )

    def evaluate_fitness(self, population):
        """Calculate the value of each individual in population"""
        total_weights = population @ self.dataset[:, 0]
        total_values = population @ self.dataset[:, 1]

        penalties = (
            np.maximum(total_weights - self.backpack_capacity, 0) * self.penalty_factor
        )
        fitness = total_values - penalties

        return fitness

    @abstractmethod
    def generate_population(self) -> np.array:
        pass

    @abstractmethod
    def run(self):
        pass
