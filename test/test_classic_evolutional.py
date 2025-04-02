import pytest
import numpy as np
from src.objects.algorithm_params import AlgorithmParams
from src.algorithms.classic_evolutional import ClassicEvolutional


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


def test_classic_evolutional_init(algorithm_params, sample_dataset):
    ce = ClassicEvolutional(algorithm_params, sample_dataset)
    assert hasattr(ce, "population")
    assert ce.population.shape == (
        algorithm_params.population_size,
        sample_dataset.shape[0],
    )


def test_crossover_even_length_returns_two_offspring():
    parent1 = np.array([1, 1, 0, 0])
    parent2 = np.array([0, 0, 1, 1])
    offspring1, offspring2 = ClassicEvolutional.crossover(parent1, parent2)

    assert np.array_equal(offspring1, [1, 1, 1, 1])
    assert np.array_equal(offspring2, [0, 0, 0, 0])


def test_crossover_odd_length():
    parent1 = np.array([1, 0, 1, 0, 1])
    parent2 = np.array([0, 1, 0, 1, 0])
    offspring1, offspring2 = ClassicEvolutional.crossover(parent1, parent2)

    assert np.array_equal(offspring1, [1, 0, 0, 1, 0])
    assert np.array_equal(offspring2, [0, 1, 1, 0, 1])


def test_crossover_different_lengths_raises_error():
    parent1 = np.array([1, 0, 1])
    parent2 = np.array([0, 1])
    with pytest.raises(ValueError, match="Parents must have same length"):
        ClassicEvolutional.crossover(parent1, parent2)


def test_crossover_single_gene():
    parent1 = np.array([1])
    parent2 = np.array([0])
    offspring1, offspring2 = ClassicEvolutional.crossover(parent1, parent2)

    assert np.array_equal(offspring1, [0])  # Takes entire parent2
    assert np.array_equal(offspring2, [1])  # Takes entire parent1


def test_mutate_no_mutation_when_rate_zero():
    originals = np.array([[1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 0, 0]])
    mutateds = ClassicEvolutional.mutate(originals)

    diffs = originals != mutateds
    assert np.all(diffs.sum(axis=1) == 1)  # One True per row


def test_mutate_handles_empty_array():
    original = np.array([], dtype=int)
    mutated = ClassicEvolutional.mutate(original)
    assert len(mutated) == 0
