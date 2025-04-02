import numpy as np
from src.objects.algorithm_params import AlgorithmParams
from src.algorithms.algorithm_base import AlgorithmBase

BASE_PROBABILITY = 0.5


class PBIL(AlgorithmBase):
    """Contains implementation of the PBIL algorithm."""

    def __init__(
        self,
        params: AlgorithmParams,
        dataset: np.array,
        backpack_capacity: float = None,
    ):
        super().__init__(params, dataset, backpack_capacity)

        self._init_probability_vector()

    def _init_probability_vector(self):
        """Initializes the base probability vector"""
        self.best_fitness = -np.inf
        self.probability_vector = np.full(self.dataset.shape[0], BASE_PROBABILITY)

    def generate_population(self):
        """Initialize the population based on probability vector"""
        return (
            np.random.rand(self.population_size, self.dataset.shape[0])
            < self.probability_vector
        ).astype(int)

    def update_probability_vector(self, best_individuals):
        """Update the probability vector based on best individuals"""
        mean_best = np.mean(best_individuals, axis=0)
        self.probability_vector = (
            1 - self.learning_rate
        ) * self.probability_vector + self.learning_rate * mean_best

    def run(self):
        """Runs the PBIL Algorithm"""
        for epoch_nr in range(self.epochs):
            population = self.generate_population()
            fitness_values = self.evaluate_fitness(population)

            self.best_fitness = max(self.best_fitness, np.max(fitness_values))

            best_indices = np.argsort(fitness_values)[-self.num_best :]
            best_individuals = population[best_indices]
            self.update_probability_vector(best_individuals)

        return self.best_fitness
