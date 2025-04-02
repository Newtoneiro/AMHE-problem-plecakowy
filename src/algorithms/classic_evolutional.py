import numpy as np
from src.objects.algorithm_params import AlgorithmParams
from src.algorithms.algorithm_base import AlgorithmBase


class ClassicEvolutional(AlgorithmBase):
    """Contains implementation of the Classic Evolutional algorithm."""

    def __init__(
        self,
        params: AlgorithmParams,
        dataset: np.array,
        backpack_capacity: float = None,
    ):
        super().__init__(params, dataset, backpack_capacity)

        self._init_population()

    def _init_population(self):
        """Generate the initial population"""
        self.best_fitness = -np.inf
        self.population = np.round(
            np.random.rand(self.population_size, self.dataset.shape[0])
        ).astype(int)

    def generate_population(self):
        """Generate the population in classic evolution algorithm way"""
        fitness_values = self.evaluate_fitness(self.population)

        self.best_fitness = max(self.best_fitness, np.max(fitness_values))

        best_indices = np.argsort(fitness_values)[-self.num_best :]
        best_individuals = self.population[best_indices]

        offspring = []
        for _ in range(self.population_size - len(best_individuals)):
            parent1, parent2 = best_individuals[
                np.random.choice(len(best_individuals), 2, replace=False)
            ]
            child1, child2 = self.crossover(parent1, parent2)
            offspring.extend([child1, child2])

        new_population = np.vstack([best_individuals, offspring])

        new_population = self.mutate(new_population)

        return new_population

    @staticmethod
    def crossover(parent1: np.array, parent2: np.array):
        """Generate offspring of parent1 and parent2 using middle point intersection"""
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have same length")

        midpoint = len(parent1) // 2
        offspring1 = np.concatenate((parent1[:midpoint], parent2[midpoint:]))
        offspring2 = np.concatenate((parent2[:midpoint], parent1[midpoint:]))
        return offspring1, offspring2

    @staticmethod
    def mutate(individuals: np.array):
        """Mutate an array of individuals using bit-flipping on random position"""
        # Create a copy to avoid modifying originals
        mutated = individuals.copy()

        if not len(individuals):
            return mutated

        n_individuals, genome_length = individuals.shape
        flip_indices = np.random.randint(0, genome_length, size=n_individuals)

        rows = np.arange(n_individuals)
        mutated[rows, flip_indices] = 1 - mutated[rows, flip_indices]

        return mutated

    def run(self):
        """Runs the PBIL Algorithm"""
        for epoch_nr in range(self.epochs):
            self.population = self.generate_population()

        return self.best_fitness
