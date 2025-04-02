from src.algorithms_comparer import AlgorithmsComparer
from src.objects.comparer_params import ComparerParams
from src.objects.algorithm_params import AlgorithmParams
from src.custom_logger import Logger


if __name__ == "__main__":
    ac = AlgorithmsComparer(ComparerParams(30), Logger())
    params = AlgorithmParams(
        population_size=300,
        num_best=50,
        learning_rate=0.01,
        epochs=300,
    )
    ac.run_comparison(params)
