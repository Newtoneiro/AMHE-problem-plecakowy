from dataclasses import dataclass


BASE_LEARNING_RATE = 0.01


@dataclass
class AlgorithmParams:
    """Class for parametrizing algotithms."""

    population_size: int
    num_best: int
    epochs: int
    learning_rate: float = BASE_LEARNING_RATE

    def generate_file_name(self):
        return f"{self.population_size}_{self.num_best}_{str(self.learning_rate).split("0.")[1]}_{self.epochs}"
