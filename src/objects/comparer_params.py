from dataclasses import dataclass, field

DATASET_SIZE_LIMIT = 100
DATASET_SIZES_STEP = {
    1: 1,
    10: 2,
    50: 5,
}


@dataclass
class ComparerParams:
    """Class for parametrizing comparer."""

    number_reruns: int
    dataset_size_limit: int = DATASET_SIZE_LIMIT
    dataset_sizes_step: dict = field(default_factory=lambda: DATASET_SIZES_STEP)
