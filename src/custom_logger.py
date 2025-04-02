import os
import csv
import numpy as np
from src.objects.algorithm_params import AlgorithmParams


LOGDIR = os.path.join(os.path.abspath(__file__), "..", "..", "logs")
FIELDNAMES = [
    "algorithm_name",
    "dataset_size",
    "data_corelated",
    "avg_execution_time",
    "avg_memory_usage",
    "worst_solution_fitness",
    "best_solution_fitness",
    "avg_solution_fitness",
    "variance_solution_fitness",
]
PRECISION = 6


class Logger:
    def init_logger_file(self, params: AlgorithmParams):
        """Initializes and empty logger file"""
        file_name = params.generate_file_name()
        self.output_file = os.path.join(LOGDIR, f"{file_name}.csv")

        if not os.path.exists(self.output_file):
            with open(self.output_file, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
                writer.writeheader()
        else:
            raise FileExistsError(f"Cannot override log file {file_name}.")

    def log_step(
        self,
        alg_name: str,
        dataset: np.array,
        data_correlated: bool,
        times: np.array,
        memory: np.array,
        fitnesses: np.array,
    ):
        """Log single execution step"""
        with open(self.output_file, "a+", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            writer.writerow(
                {
                    "algorithm_name": alg_name,
                    "dataset_size": dataset.shape[0],
                    "data_corelated": data_correlated,
                    "avg_execution_time": np.round(np.mean(times), PRECISION),
                    "avg_memory_usage": np.round(np.mean(memory), PRECISION),
                    "worst_solution_fitness": np.round(np.min(fitnesses), PRECISION),
                    "best_solution_fitness": np.round(np.max(fitnesses), PRECISION),
                    "avg_solution_fitness": np.round(np.mean(fitnesses), PRECISION),
                    "variance_solution_fitness": np.round(np.var(fitnesses), PRECISION),
                }
            )
