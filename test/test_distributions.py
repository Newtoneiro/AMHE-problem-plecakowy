import pytest
import numpy as np
from src.objects.distributions import NormalDistribution, UniformDistribution


class TestNormalDistribution:
    def test_generate_data_output(self):
        dist = NormalDistribution(10, 2, 5, 1)
        weights, values = dist.generate_data(100)
        assert isinstance(weights, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert len(weights) == 100
        assert len(values) == 100

    def test_generate_data_positive_values(self):
        dist = NormalDistribution(10, 2, 5, 1)
        weights, values = dist.generate_data(1000)
        assert np.all(weights >= 0)
        assert np.all(values >= 0)

    def test_statistical_properties(self):
        mean_w, var_w = 10, 4
        mean_v, var_v = 5, 1
        dist = NormalDistribution(mean_w, var_w, mean_v, var_v)
        weights, values = dist.generate_data(100000)

        assert pytest.approx(np.mean(weights), abs=0.1) == mean_w
        assert pytest.approx(np.var(weights), abs=0.1) == var_w
        assert pytest.approx(np.mean(values), abs=0.1) == mean_v
        assert pytest.approx(np.var(values), abs=0.1) == var_v


class TestUniformDistribution:
    def test_generate_data_output(self):
        dist = UniformDistribution()
        weights, values = dist.generate_data(50)
        assert isinstance(weights, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert len(weights) == 50
        assert len(values) == 50

    def test_generate_data_range(self):
        min_w, max_w = 5, 15
        min_v, max_v = 1, 10
        dist = UniformDistribution(min_w, max_w, min_v, max_v)
        weights, values = dist.generate_data(1000)

        assert np.all(weights >= min_w)
        assert np.all(weights <= max_w)
        assert np.all(values >= min_v)
        assert np.all(values <= max_v)

    def test_statistical_properties(self):
        min_w, max_w = 5, 15
        min_v, max_v = 1, 10
        dist = UniformDistribution(min_w, max_w, min_v, max_v)
        weights, values = dist.generate_data(10000)

        for q in [0.25, 0.5, 0.75]:
            expected = min_w + q * (max_w - min_w)
            assert pytest.approx(np.quantile(weights, q), rel=0.05) == expected

        for q in [0.25, 0.5, 0.75]:
            expected = min_v + q * (max_v - min_v)
            assert pytest.approx(np.quantile(values, q), rel=0.05) == expected

    def test_independence(self):
        dist = UniformDistribution()
        weights, values = dist.generate_data(10000)
        corr = np.corrcoef(weights, values)[0, 1]
        assert pytest.approx(corr, abs=0.05) == 0
