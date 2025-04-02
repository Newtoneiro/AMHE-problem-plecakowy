import os
import csv
import numpy as np


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
    def __init__(self, output_file="algorithm_comparison.csv"):
        self.output_file = output_file

        self._init_logger_file()

    def _init_logger_file(self):
        """Initializes and empty logger file"""
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
                writer.writeheader()

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
