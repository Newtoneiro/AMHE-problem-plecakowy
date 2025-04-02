import pytest
import numpy as np
from src.objects.algorithm_params import AlgorithmParams
from src.algorithms.pbil import PBIL, BASE_PROBABILITY


@pytest.fixture
def sample_dataset():
    return np.array([[1, 2], [3, 4], [5, 6]])


@pytest.fixture
def algorithm_params():
    return AlgorithmParams(
        population_size=100,
        num_best=10,
        learning_rate=0.1,
        epochs=50,
    )


def test_pbil_init(algorithm_params, sample_dataset):
    pbil = PBIL(algorithm_params, sample_dataset)
    assert hasattr(pbil, "probability_vector")
    assert pbil.probability_vector.shape == (sample_dataset.shape[0],)
    assert np.all(pbil.probability_vector == BASE_PROBABILITY)


def test_pbil_getattr_existing_param(algorithm_params, sample_dataset):
    pbil = PBIL(algorithm_params, sample_dataset)
    assert pbil.population_size == algorithm_params.population_size
    assert pbil.learning_rate == algorithm_params.learning_rate
    assert np.array_equal(pbil.dataset, sample_dataset)


def test_pbil_getattr_existing_attr(algorithm_params, sample_dataset):
    pbil = PBIL(algorithm_params, sample_dataset)
    pbil.custom_attr = 42
    assert pbil.custom_attr == 42


def test_pbil_getattr_missing(algorithm_params, sample_dataset):
    pbil = PBIL(algorithm_params, sample_dataset)
    with pytest.raises(AttributeError):
        _ = pbil.non_existent_attr


def test_init_probability_vector(sample_dataset):
    params = AlgorithmParams(
        population_size=50,
        num_best=5,
        learning_rate=0.2,
        epochs=100,
    )
    pbil = PBIL(params, sample_dataset)
    assert len(pbil.probability_vector) == len(sample_dataset)
    assert np.all(pbil.probability_vector == BASE_PROBABILITY)


def test_calculate_penalty_factor():
    dataset = np.array([[5, 10], [3, 6], [2, 8]])
    params = AlgorithmParams(
        population_size=50,
        num_best=5,
        learning_rate=0.2,
        epochs=100,
    )
    pbil = PBIL(params, dataset, backpack_capacity=10)
    pbil._calculate_penalty_factor()

    assert pytest.approx(pbil.penalty_factor) == 48


def test_generate_population(sample_dataset):
    params = AlgorithmParams(
        population_size=50,
        num_best=5,
        learning_rate=0.2,
        epochs=100,
    )
    pbil = PBIL(params, sample_dataset)
    population = pbil.generate_population()
    assert population.shape == (50, 3)


def test_generate_population_probability_distribution(algorithm_params, sample_dataset):
    pbil = PBIL(algorithm_params, sample_dataset)
    # Set specific probabilities for testing
    test_probabilities = np.array([0.0, 0.25, 0.75])
    pbil.probability_vector = test_probabilities
    population = pbil.generate_population()

    # Test each gene's distribution
    for i, prob in enumerate(test_probabilities):
        gene_values = population[:, i]
        actual_prob = np.mean(gene_values)

        assert abs(actual_prob - prob) < 0.1  # 10% tolerance


def test_evaluate_fitness():
    dataset = np.array([[5, 10], [3, 8], [6, 5]])
    params = AlgorithmParams(
        population_size=10,
        num_best=5,
        learning_rate=0.1,
        epochs=50,
    )
    pbil = PBIL(params, dataset, backpack_capacity=10)

    # valid solution
    population = np.array([[1, 1, 0]])
    fitness = pbil.evaluate_fitness(population)
    assert fitness[0] == 18

    # another valid solution
    population = np.array([[0, 1, 1]])
    fitness = pbil.evaluate_fitness(population)
    assert fitness[0] == 13

    # invalid solution
    population = np.array([[1, 0, 1]])
    fitness = pbil.evaluate_fitness(population)
    assert np.round(fitness[0]) == -18.0


def test_update_probability_vector_basic():
    dataset = np.array([[1, 2], [3, 4]])
    params = AlgorithmParams(
        population_size=10,
        num_best=3,
        learning_rate=0.5,
        epochs=50,
    )
    pbil = PBIL(params, dataset)
    pbil.probability_vector = np.array([0.3, 0.7])

    best_individuals = np.array([[1, 0], [1, 0], [0, 1]])

    pbil.update_probability_vector(best_individuals)

    assert pytest.approx(pbil.probability_vector[0], abs=0.01) == 0.48
    assert pytest.approx(pbil.probability_vector[1], abs=0.01) == 0.515
