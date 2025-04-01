from src.algorithms.pbil import PBIL
from src.algorithms.classic_evolutional import ClassicEvolutional
from src.data_generator import DataGenerator
from src.objects.algorithm_params import AlgorithmParams
from src.objects.generator_params import GeneratorParams
from src.objects.distributions import UniformDistribution
from pprint import pprint as print


if __name__ == "__main__":
    dataset = DataGenerator.generate_dataset(GeneratorParams(10, UniformDistribution(1, 10, 1, 10)))
    a = PBIL(AlgorithmParams(dataset, 20, 5, 0.01, 300, 10))
    pv = a.run()
    print(dataset[:, 1])
    print(dataset[:, 0])
    print("*" * 20)
    print(pv)

    b = ClassicEvolutional(AlgorithmParams(dataset, 20, 5, 0.01, 300, 10))
    best = b.run()
    print(best)
