import numpy as np
from src.objects.generator_params import GeneratorParams


class DataGenerator:
    """
    Class for handling data generation.
    """

    def generate_dataset(params: GeneratorParams) -> np.ndarray:
        """
        Generates a dataset of items with weights and values.

        Args:
            params: GeneratorParams determining the size of dataset plus the distribution

        Returns:
            np.ndarray: Array of shape (no_samples, 2) where first column is weight
                       and second is value
        """
        weights, values = params.data_distribution.generate_data(params.no_samples)
        return np.column_stack((np.abs(weights), np.abs(values)))
