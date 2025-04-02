from src.algorithms_comparer import AlgorithmsComparer
from src.objects.comparer_params import ComparerParams
from src.custom_logger import Logger


if __name__ == "__main__":
    ac = AlgorithmsComparer(ComparerParams(30), Logger())
    ac.run_comparison()
