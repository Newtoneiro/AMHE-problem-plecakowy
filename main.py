from src.algorithms_comparer import AlgorithmsComparer
from src.objects.comparer_params import ComparerParams
from src.objects.algorithm_params import AlgorithmParams
from src.custom_logger import Logger


if __name__ == "__main__":
    ac = AlgorithmsComparer(ComparerParams(30), Logger())
    params = AlgorithmParams(
        population_size=150,
        num_best=20,
        learning_rate=0.01,
        epochs=200,
    )
    ac.run_comparison(params)
