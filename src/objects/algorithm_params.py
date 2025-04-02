from dataclasses import dataclass


@dataclass
class AlgorithmParams:
    """Class for parametrizing algotithms."""

    population_size: int
    num_best: int
    learning_rate: float
    epochs: int

    def generate_file_name(self):
        return f"{self.population_size}_{self.num_best}_{str(self.learning_rate).split("0.")[1]}_{self.epochs}"
