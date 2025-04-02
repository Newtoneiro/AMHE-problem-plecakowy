import time
import numpy as np
import tracemalloc
from tqdm import tqdm
from src.objects.algorithm_params import AlgorithmParams
from src.objects.comparer_params import ComparerParams
from src.objects.distributions import NormalDistribution, UniformDistribution
from src.algorithms.pbil import PBIL
from src.algorithms.classic_evolutional import ClassicEvolutional
from src.objects.generator_params import GeneratorParams
from src.data_generator import DataGenerator
from src.custom_logger import Logger

MAX_LEARNING_RATE = 0.01
DATASET_SIZE_LIMIT = 100
DATASET_SIZES_STEP = {
    1: 1,
    10: 2,
    50: 5,
}


class AlgorithmsComparer:
    """Logic for comparing the algorithms"""

    def __init__(self, params: ComparerParams, logger: Logger):
        self.params = params
        self.logger = logger

    def __getattr__(self, name):
        """Simplifier for accessing parameters values"""
        if hasattr(self.params, name):
            return getattr(self.params, name)

    def _run_single_comparison(
        self,
        algorithm_params: AlgorithmParams,
        dataset: np.array,
        data_correlated: bool,
    ):
        for Algorithm in [PBIL, ClassicEvolutional]:
            times = np.zeros(self.number_reruns, dtype=np.float64)
            memory = np.zeros(self.number_reruns, dtype=np.float64)
            solution_fitnesses = np.zeros(self.number_reruns, dtype=np.float64)
            for rerun_idx in range(self.number_reruns):
                algorithm = Algorithm(algorithm_params, dataset)

                tracemalloc.start()
                start_time = time.perf_counter()

                best_fitness = algorithm.run()

                elapsed = time.perf_counter() - start_time
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                times[rerun_idx] = elapsed
                memory[rerun_idx] = peak / (1024 * 1024)  # Convert to MB
                solution_fitnesses[rerun_idx] = best_fitness

            self.logger.log_step(
                Algorithm.__name__,
                dataset,
                data_correlated,
                times,
                memory,
                solution_fitnesses,
            )

    def run_comparison(self, params: AlgorithmParams):
        self.logger.init_logger_file(params)

        dataset_size = 1
        step = 1
        with tqdm(
            total=DATASET_SIZE_LIMIT, desc="Comparing Algorithms"
        ) as pbar_dataset:
            while dataset_size != DATASET_SIZE_LIMIT:
                for data_related, distribution in zip(
                    [True, False],
                    [
                        NormalDistribution(
                            dataset_size / 2,
                            dataset_size / 4,
                            dataset_size / 2,
                            dataset_size / 4,
                        ),
                        UniformDistribution(1, dataset_size, 1, dataset_size),
                    ],
                ):
                    dataset = DataGenerator.generate_dataset(
                        GeneratorParams(
                            dataset_size,
                            distribution,
                        )
                    )

                    self._run_single_comparison(params, dataset, data_related)

                dataset_size += step
                if dataset_size in DATASET_SIZES_STEP:
                    step = DATASET_SIZES_STEP[
                        dataset_size
                    ]  # 1 for small, 2 for medium and 3 for large

                pbar_dataset.update(step)
