from dataclasses import dataclass
from src.objects.distributions import DistributionKind


@dataclass
class GeneratorParams:
    """Class for parametrizing data generation."""

    no_samples: int
    data_distribution: DistributionKind
