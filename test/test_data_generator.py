import numpy as np
from src.objects.generator_params import GeneratorParams
from src.objects.distributions import NormalDistribution, UniformDistribution
from src.data_generator import DataGenerator


class TestDataGenerator:
    def test_generator_with_normal_dist(self):
        dist = NormalDistribution(10, 2, 5, 1)
        params = GeneratorParams(100, dist)
        data = DataGenerator.generate_dataset(params)

        assert data.shape == (100, 2)
        assert np.all(data >= 0)

    def test_generator_with_uniform_dist(self):
        dist = UniformDistribution(1, 10, 1, 5)
        params = GeneratorParams(50, dist)
        data = DataGenerator.generate_dataset(params)

        assert data.shape == (50, 2)
        assert np.all((data >= 1) & (data <= 10))

    def test_generator_zero_samples(self):
        dist = UniformDistribution()
        params = GeneratorParams(0, dist)
        data = DataGenerator.generate_dataset(params)

        assert data.shape == (0, 2)
